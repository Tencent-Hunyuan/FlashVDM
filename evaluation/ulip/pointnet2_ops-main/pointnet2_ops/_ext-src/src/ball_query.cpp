#include "ball_query.h"
#include "utils.h"

at::Tensor query_ball_point_kernel_wrapper(
    int b,
    int n,
    int m,
    float radius,
    int nsample,
    const at::Tensor new_xyz,
    const at::Tensor xyz);

at::Tensor ball_query(
    at::Tensor new_xyz,
    at::Tensor xyz,
    const float radius,
    const int nsample) {
  CHECK_INPUT(new_xyz);
  CHECK_INPUT(xyz);

  return query_ball_point_kernel_wrapper(
    xyz.size(0),
    xyz.size(1),
    new_xyz.size(1),
    radius,
    nsample,
    new_xyz,
    xyz);
}
