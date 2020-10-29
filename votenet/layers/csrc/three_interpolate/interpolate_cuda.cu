#include "interpolate.h"
#include "utils.h"

namespace votenet {

void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
                             const float *known, float *dist2, int *idx);
void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const float *points, const int *idx,
                                      const float *weight, float *out);
void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
                                           const float *grad_out,
                                           const int *idx, const float *weight,
                                           float *grad_points);

std::vector<at::Tensor> three_nn_cuda(
    const at::Tensor& unknowns,
    const at::Tensor& knows) {
  CHECK_IS_FLOAT(unknowns);
  CHECK_IS_FLOAT(knows);

  at::Tensor idx =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Int));
  at::Tensor dist2 =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Float));

  three_nn_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
                          unknowns.data<float>(), knows.data<float>(),
                          dist2.data<float>(), idx.data<int>());

  return {dist2, idx};
}

at::Tensor three_interpolate_forward_cuda(
    const at::Tensor& points,
    const at::Tensor& idx,
    const at::Tensor& weight) {
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  three_interpolate_kernel_wrapper(
      points.size(0), points.size(1), points.size(2), idx.size(1),
      points.data<float>(), idx.data<int>(), weight.data<float>(),
      output.data<float>());

  return output;
}

at::Tensor three_interpolate_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& idx,
    const at::Tensor& weight,
    const int m) {
  CHECK_IS_FLOAT(grad);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  at::Tensor output =
      torch::zeros({grad.size(0), grad.size(1), m},
                   at::device(grad.device()).dtype(at::ScalarType::Float));

  three_interpolate_grad_kernel_wrapper(
      grad.size(0), grad.size(1), grad.size(2), m,
      grad.data<float>(), idx.data<int>(), weight.data<float>(),
      output.data<float>());

  return output;
}

} // namespace votenet
