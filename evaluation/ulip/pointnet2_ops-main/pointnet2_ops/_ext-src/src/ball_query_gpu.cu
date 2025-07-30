#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "cuda_utils.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
template<typename scalar_t>
__global__ void query_ball_point_kernel(
    int b,
    int n,
    int m,
    float radius,
    int nsample,
    const scalar_t *__restrict__ new_xyz,
    const scalar_t *__restrict__ xyz,
    int *__restrict__ idx) {
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= m) return;

  new_xyz += bs_idx * m * 3 + pt_idx * 3;
  xyz += bs_idx * n * 3;
  idx += bs_idx * m * nsample + pt_idx * nsample;

  float radius2 = radius * radius;
  scalar_t new_x = new_xyz[0];
  scalar_t new_y = new_xyz[1];
  scalar_t new_z = new_xyz[2];

  int cnt = 0;
  for (int k = 0; k < n; ++k) {
      scalar_t x = xyz[k * 3 + 0];
      scalar_t y = xyz[k * 3 + 1];
      scalar_t z = xyz[k * 3 + 2];
      scalar_t d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
      if (d2 < radius2){
          if (cnt == 0){
              for (int l = 0; l < nsample; ++l) {
                  idx[l] = k;
              }
          }
          idx[cnt] = k;
          ++cnt;
          if (cnt >= nsample) break;
      }
  }
}

at::Tensor query_ball_point_kernel_wrapper(
    int b,
    int n,
    int m,
    float radius,
    int nsample,
    const at::Tensor new_xyz,
    const at::Tensor xyz) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor idx = torch::zeros(
      {new_xyz.size(0), new_xyz.size(1), nsample},
      at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    xyz.scalar_type(), "query_ball_cuda", ([&] {
      query_ball_point_kernel<scalar_t><<<b, opt_n_threads(m), 0, stream>>>(
        b,
        n,
        m,
        radius,
        nsample,
        new_xyz.data_ptr<scalar_t>(),
        xyz.data_ptr<scalar_t>(),
        idx.data_ptr<int>());
  }));
  CUDA_CHECK_ERRORS();
  return idx;
}
