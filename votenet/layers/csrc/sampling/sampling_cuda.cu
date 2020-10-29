// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "sampling.h"
#include "utils.h"

namespace votenet {

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out);
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points);

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);

at::Tensor gather_points_forward_cuda(
    const at::Tensor& points,
    const at::Tensor& idx) {
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                               idx.size(1), points.data<float>(),
                               idx.data<int>(), output.data<float>());

  return output;
}

at::Tensor gather_points_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& idx,
    const int n) {
  CHECK_IS_FLOAT(grad);
  CHECK_IS_INT(idx);

  at::Tensor output =
      torch::zeros({grad.size(0), grad.size(1), n},
                   at::device(grad.device()).dtype(at::ScalarType::Float));

  gather_points_grad_kernel_wrapper(grad.size(0), grad.size(1), n,
                                    idx.size(1), grad.data<float>(),
                                    idx.data<int>(), output.data<float>());

  return output;
}

at::Tensor furthest_point_sampling_cuda(
    const at::Tensor& points,
    const int num_samples) {
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), num_samples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  furthest_point_sampling_kernel_wrapper(
      points.size(0), points.size(1), num_samples, points.data<float>(),
      tmp.data<float>(), output.data<int>());

  return output;
}

} // namespace votenet
