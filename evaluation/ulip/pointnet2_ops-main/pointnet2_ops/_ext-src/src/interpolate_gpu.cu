#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <torch/extension.h>

#include "cuda_utils.h"

// input: unknown(b, n, 3) known(b, m, 3)
// output: dist2(b, n, 3), idx(b, n, 3)
template<typename scalar_t>
__global__ void three_nn_kernel(
    int b,
    int n,
    int m,
    const scalar_t *__restrict__ unknown,
    const scalar_t *__restrict__ known,
    scalar_t *__restrict__ dist2,
    int *__restrict__ idx) {
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= n) return;

  unknown += bs_idx * n * 3 + pt_idx * 3;
  known += bs_idx * m * 3;
  dist2 += bs_idx * n * 3 + pt_idx * 3;
  idx += bs_idx * n * 3 + pt_idx * 3;

  scalar_t ux = unknown[0];
  scalar_t uy = unknown[1];
  scalar_t uz = unknown[2];

  double best1 = 1e40, best2 = 1e40, best3 = 1e40;
  int besti1 = 0, besti2 = 0, besti3 = 0;
  for (int k = 0; k < m; ++k) {
      scalar_t x = known[k * 3 + 0];
      scalar_t y = known[k * 3 + 1];
      scalar_t z = known[k * 3 + 2];
      scalar_t d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
      if (d < best1) {
          best3 = best2; besti3 = besti2;
          best2 = best1; besti2 = besti1;
          best1 = d; besti1 = k;
      }
      else if (d < best2) {
          best3 = best2; besti3 = besti2;
          best2 = d; besti2 = k;
      }
      else if (d < best3) {
          best3 = d; besti3 = k;
      }
  }
  dist2[0] = best1; dist2[1] = best2; dist2[2] = best3;
  idx[0] = besti1; idx[1] = besti2; idx[2] = besti3;
}

std::vector<at::Tensor> three_nn_kernel_wrapper(
    int b,
    int n,
    int m,
    const at::Tensor unknown,
    const at::Tensor known) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor idx = torch::zeros(
    {unknown.size(0), unknown.size(1), 3},
    at::device(unknown.device()).dtype(at::ScalarType::Int));
  at::Tensor dist2 =
      torch::zeros({unknown.size(0), unknown.size(1), 3},
      at::device(unknown.device()).dtype(unknown.scalar_type()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    unknown.scalar_type(), "three_nn_kernel_cuda", ([&] {
      three_nn_kernel<scalar_t><<<b, opt_n_threads(n), 0, stream>>>(
          b,
          n,
          m,
          unknown.data_ptr<scalar_t>(),
          known.data_ptr<scalar_t>(),
          dist2.data_ptr<scalar_t>(),
          idx.data_ptr<int>());
  }));

  CUDA_CHECK_ERRORS();
  return {dist2, idx};
}

// input: points(b, c, m), idx(b, n, 3), weight(b, n, 3)
// output: out(b, c, n)
template<typename scalar_t>
__global__ void three_interpolate_kernel(
    int b,
    int c,
    int m,
    int n,
    const scalar_t *__restrict__ points,
    const int *__restrict__ idx,
    const scalar_t *__restrict__ weight,
    scalar_t *__restrict__ out) {
  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

  weight += bs_idx * n * 3 + pt_idx * 3;
  points += bs_idx * c * m + c_idx * m;
  idx += bs_idx * n * 3 + pt_idx * 3;
  out += bs_idx * c * n + c_idx * n;

  out[pt_idx] = weight[0] * points[idx[0]] + weight[1] * points[idx[1]] + weight[2] * points[idx[2]];
}

at::Tensor three_interpolate_kernel_wrapper(
    int b,
    int c,
    int m,
    int n,
    const at::Tensor points,
    const at::Tensor idx,
    const at::Tensor weight) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor out =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(points.scalar_type()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points.scalar_type(), "three_interpolate_cuda", ([&] {
        three_interpolate_kernel<scalar_t><<<b, opt_block_config(n, c), 0, stream>>>(
          b,
          c,
          m,
          n,
          points.data_ptr<scalar_t>(),
          idx.data_ptr<int>(),
          weight.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>());
        }));

  CUDA_CHECK_ERRORS();
  return out;
}

// input: grad_out(b, c, n), idx(b, n, 3), weight(b, n, 3)
// output: grad_points(b, c, m)

template<typename scalar_t>
__global__ void three_interpolate_grad_kernel(
    int b,
    int c,
    int n,
    int m,
    const scalar_t *__restrict__ grad_out,
    const int *__restrict__ idx,
    const scalar_t *__restrict__ weight,
    scalar_t *__restrict__ grad_points) {
  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

  grad_out += bs_idx * c * n + c_idx * n + pt_idx;
  weight += bs_idx * n * 3 + pt_idx * 3;
  grad_points += bs_idx * c * m + c_idx * m;
  idx += bs_idx * n * 3 + pt_idx * 3;


  atomicAdd(grad_points + idx[0], grad_out[0] * weight[0]);
  atomicAdd(grad_points + idx[1], grad_out[0] * weight[1]);
  atomicAdd(grad_points + idx[2], grad_out[0] * weight[2]);
}

at::Tensor three_interpolate_grad_kernel_wrapper(
    int b,
    int c,
    int n,
    int m,
    const at::Tensor grad_out,
    const at::Tensor idx,
    const at::Tensor weight) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor grad_points = torch::zeros(
      {grad_out.size(0), grad_out.size(1), m},
      at::device(grad_out.device()).dtype(grad_out.scalar_type()));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.scalar_type(), "three_interpolate_grad_cuda", ([&] {
        three_interpolate_grad_kernel<<<b, opt_block_config(n, c), 0, stream>>>(
            b,
            c,
            n,
            m,
            grad_out.data_ptr<scalar_t>(),
            idx.data_ptr<int>(),
            weight.data_ptr<scalar_t>(),
            grad_points.data_ptr<scalar_t>());
        }));
  CUDA_CHECK_ERRORS();
  return grad_points;
}
