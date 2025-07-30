#include "interpolate.h"
#include "utils.h"

std::vector<at::Tensor> three_nn_kernel_wrapper(
    int b,
    int n,
    int m,
    const at::Tensor unknown,
    const at::Tensor known);

at::Tensor three_interpolate_kernel_wrapper(
    int b,
    int c,
    int m,
    int n,
    const at::Tensor points,
    const at::Tensor idx,
    const at::Tensor weight);

at::Tensor three_interpolate_grad_kernel_wrapper(
    int b,
    int c,
    int n,
    int m,
    const at::Tensor grad_out,
    const at::Tensor idx,
    const at::Tensor weight);

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knowns) {
  CHECK_INPUT(unknowns);
  CHECK_INPUT(knowns);
  // TODO maybe include a check that its floating/scalar type

  return three_nn_kernel_wrapper(
      unknowns.size(0),
      unknowns.size(1),
      knowns.size(1),
      unknowns,
      knowns);
}

at::Tensor three_interpolate(
    at::Tensor points,
    at::Tensor idx,
    at::Tensor weight) {
  CHECK_INPUT(points);
  CHECK_INPUT(idx);
  CHECK_INPUT(weight);
  CHECK_IS_INT(idx);
  // TODO maybe check types for point and weight?

  return three_interpolate_kernel_wrapper(
    points.size(0),
    points.size(1),
    points.size(2),
    idx.size(1),
    points,
    idx,
    weight);
}
at::Tensor three_interpolate_grad(
    at::Tensor grad_out,
    at::Tensor idx,
    at::Tensor weight,
    const int m) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(idx);
  CHECK_INPUT(weight);
  CHECK_IS_INT(idx);
  // TODO maybe check type for weight and grad_out?

  return three_interpolate_grad_kernel_wrapper(
    grad_out.size(0),
    grad_out.size(1),
    grad_out.size(2),
    m,
    grad_out,
    idx,
    weight);
}
