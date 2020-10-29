#pragma once
#include <torch/types.h>

namespace votenet {

#if defined(WITH_CUDA) || defined(WITH_HIP)
at::Tensor gather_points_forward_cuda(
    const at::Tensor& points,
    const at::Tensor& idx);

at::Tensor gather_points_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& idx,
    const int n);

at::Tensor furthest_point_sampling_cuda(
    const at::Tensor& points,
    const int num_samples);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
inline at::Tensor gather_points_forward(
    const at::Tensor& points,
    const at::Tensor& idx) {
  assert(points.device().is_cuda() == idx.device().is_cuda());
  if (points.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return gather_points_forward_cuda(points.contiguous(), idx.contiguous());
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CPU is not implemented");
  }
}

inline at::Tensor gather_points_backward(
    const at::Tensor& grad,
    const at::Tensor& idx,
    const int n) {
  assert(grad.device().is_cuda() == idx.device().is_cuda());
  if (grad.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return gather_points_backward_cuda(grad.contiguous(), idx.contiguous(), n);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CPU is not implemented");
  }
}

inline at::Tensor furthest_point_sampling(
    const at::Tensor& points,
    const int num_samples) {
  if (points.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return furthest_point_sampling_cuda(points.contiguous(), num_samples);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CPU is not implemented");
  }
}

} // namespace votenet
