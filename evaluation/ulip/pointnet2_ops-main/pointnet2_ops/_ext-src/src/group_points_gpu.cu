#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "cuda_utils.h"

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
template<typename scalar_t>
__global__ void group_points_kernel(
    int b,
    int c,
    int n,
    int npoints,
    int nsample,
    const scalar_t *__restrict__ points,
    const int *__restrict__ idx,
    scalar_t *__restrict__ out) {
  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int pt_idx = index / nsample;
  if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

  int sample_idx = index % nsample;

  idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx;
  int in_idx = bs_idx * c * n + c_idx * n + idx[0];
  int out_idx = bs_idx * c * npoints * nsample + c_idx * npoints * nsample + pt_idx * nsample + sample_idx;

  out[out_idx] = points[in_idx];
}

at::Tensor group_points_kernel_wrapper(
    int b,
    int c,
    int n,
    int npoints,
    int nsample,
    const at::Tensor points,
    const at::Tensor idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor out = torch::zeros(
    {points.size(0), points.size(1), idx.size(1), idx.size(2)},
    at::device(points.device()).dtype(points.scalar_type()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    points.scalar_type(), "group_points_cuda", ([&] {
      group_points_kernel<scalar_t><<<b, opt_block_config(npoints, c), 0, stream>>>(
          b,
          c,
          n,
          npoints,
          nsample,
          points.data_ptr<scalar_t>(),
          idx.data_ptr<int>(),
          out.data_ptr<scalar_t>());
  }));
  CUDA_CHECK_ERRORS();
  return out;
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
template<typename scalar_t>
__global__ void group_points_grad_kernel(
    int b,
    int c,
    int n,
    int npoints,
    int nsample,
    const scalar_t *__restrict__ grad_out,
    const int *__restrict__ idx,
    scalar_t *__restrict__ grad_points) {
  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int pt_idx = index / nsample;
  if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

  int sample_idx = index % nsample;
  grad_out += bs_idx * c * npoints * nsample + c_idx * npoints * nsample + pt_idx * nsample + sample_idx;
  idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx;

  atomicAdd(grad_points + bs_idx * c * n + c_idx * n + idx[0] , grad_out[0]);
}

at::Tensor group_points_grad_kernel_wrapper(
    int b,
    int c,
    int n,
    int npoints,
    int nsample,
    const at::Tensor grad_out,
    const at::Tensor idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor grad_points = torch::zeros(
      {grad_out.size(0), grad_out.size(1), n},
      at::device(grad_out.device()).dtype(grad_out.scalar_type()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad_out.scalar_type(), "group_points_grad_cuda", ([&] {
      group_points_grad_kernel<scalar_t><<<b, opt_block_config(npoints, c), 0, stream>>>(
        b,
        c,
        n,
        npoints,
        nsample,
        grad_out.data_ptr<scalar_t>(),
        idx.data_ptr<int>(),
        grad_points.data_ptr<scalar_t>());
  }));
  CUDA_CHECK_ERRORS();
  return grad_points;
}
