#include "group_points.h"
#include "utils.h"

namespace votenet {

void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out);

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points);

at::Tensor group_points_forward_cuda(
    const at::Tensor& points, const at::Tensor& idx) {
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  group_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                              idx.size(1), idx.size(2), points.data<float>(),
                              idx.data<int>(), output.data<float>());

  return output;
}

at::Tensor group_points_backward_cuda(
    const at::Tensor& grad, const at::Tensor& idx, const int n) {
  CHECK_IS_FLOAT(grad);
  CHECK_IS_INT(idx);

  at::Tensor output =
      torch::zeros({grad.size(0), grad.size(1), n},
                   at::device(grad.device()).dtype(at::ScalarType::Float));

  group_points_grad_kernel_wrapper(
      grad.size(0), grad.size(1), n, idx.size(1), idx.size(2),
      grad.data<float>(), idx.data<int>(), output.data<float>());

  return output;
}

} // namespace votenet
