// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "ball_query.h"
#include "utils.h"

namespace votenet {

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx);

at::Tensor ball_query_cuda(
    const at::Tensor& new_xyz,
    const at::Tensor& xyz,
    const float radius,
    const int num_samples) {
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), num_samples},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                  radius, num_samples, new_xyz.data<float>(),
                                  xyz.data<float>(), idx.data<int>());

  return idx;
}

} // namespace votenet
