#include "sampling.h"
#include "utils.h"

at::Tensor gather_points_kernel_wrapper(
    int b,
    int c,
    int n,
    int npoints,
    const at::Tensor points,
    const at::Tensor idx);

at::Tensor gather_points_grad_kernel_wrapper(
    int b,
    int c,
    int n,
    int npoints,
    const at::Tensor grad_out,
    const at::Tensor idx);

at::Tensor furthest_point_sampling_kernel_wrapper(
    int b,
    int n,
    int m,
    const at::Tensor points);

at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
  CHECK_INPUT(points);
  CHECK_INPUT(idx);
  CHECK_IS_INT(idx);
  // TODO check types for points?


  return gather_points_kernel_wrapper(
      points.size(0),
      points.size(1),
      points.size(2),
      idx.size(1),
      points,
      idx);
}

at::Tensor gather_points_grad(
    at::Tensor grad_out,
    at::Tensor idx,
    const int n) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(idx);
  CHECK_IS_INT(idx);
  // TODO Check scalar type for grad_out?

  return gather_points_grad_kernel_wrapper(
      grad_out.size(0),
      grad_out.size(1),
      n,
      idx.size(1),
      grad_out,
      idx);
}

at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_INPUT(points);
  return furthest_point_sampling_kernel_wrapper(
      points.size(0),
      points.size(1),
      nsamples,
      points);
}
