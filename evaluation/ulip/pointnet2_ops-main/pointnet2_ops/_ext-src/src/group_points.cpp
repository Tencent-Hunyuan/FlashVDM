#include "group_points.h"
#include "utils.h"

at::Tensor group_points_kernel_wrapper(
    int b,
    int c,
    int n,
    int npoints,
    int nsample,
    const at::Tensor points,
    const at::Tensor idx);

at::Tensor group_points_grad_kernel_wrapper(
    int b,
    int c,
    int n,
    int npoints,
    int nsample,
    const at::Tensor grad_out,
    const at::Tensor idx);

at::Tensor group_points(at::Tensor points, at::Tensor idx) {
  CHECK_INPUT(points);
  CHECK_INPUT(idx);
  CHECK_IS_INT(idx);

  return group_points_kernel_wrapper(
    points.size(0),
    points.size(1),
    points.size(2),
    idx.size(1),
    idx.size(2),
    points,
    idx);
}

at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(idx);
  CHECK_IS_INT(idx);
  // Maybe should also check if grad_out is double/float/half?

  return group_points_grad_kernel_wrapper(
      grad_out.size(0),
      grad_out.size(1),
      n,
      idx.size(1),
      idx.size(2),
      grad_out,
      idx);
}
