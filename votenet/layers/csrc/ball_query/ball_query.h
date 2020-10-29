#pragma once
#include <torch/types.h>

namespace votenet {

#if defined(WITH_CUDA) || defined(WITH_HIP)
at::Tensor ball_query_cuda(
    const at::Tensor& new_xyz,
    const at::Tensor& xyz,
    const float radius,
    const int num_samples);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
inline at::Tensor ball_query(
    const at::Tensor& new_xyz,
    const at::Tensor& xyz,
    const float radius,
    const int num_samples) {
  assert(new_xyz.device().is_cuda() == xyz.device().is_cuda());
  if (new_xyz.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return ball_query_cuda(new_xyz.contiguous(), xyz.contiguous(), radius, num_samples);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CPU is not implemented");
  }
}

} // namespace votenet
