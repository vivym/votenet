#pragma once
#include <torch/types.h>

namespace votenet {

#if defined(WITH_CUDA) || defined(WITH_HIP)
std::vector<at::Tensor> three_nn_cuda(
    const at::Tensor& unknowns,
    const at::Tensor& knows);

at::Tensor three_interpolate_forward_cuda(
    const at::Tensor& points,
    const at::Tensor& idx,
    const at::Tensor& weight);

at::Tensor three_interpolate_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& idx,
    const at::Tensor& weight,
    const int m);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
inline std::vector<at::Tensor> three_nn(
    const at::Tensor& unknowns,
    const at::Tensor& knows) {
  assert(unknowns.device().is_cuda() == knows.device().is_cuda());
  if (unknowns.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return three_nn_cuda(unknowns.contiguous(), knows.contiguous());
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CPU is not implemented");
  }
}

inline at::Tensor three_interpolate_forward(
    const at::Tensor& points,
    const at::Tensor& idx,
    const at::Tensor& weight) {
  assert(points.device().is_cuda() == idx.device().is_cuda());
  assert(idx.device().is_cuda() == weight.device().is_cuda());
  if (points.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return three_interpolate_forward_cuda(points.contiguous(), idx.contiguous(), weight.contiguous());
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CPU is not implemented");
  }
}

inline at::Tensor three_interpolate_backward(
    const at::Tensor& grad,
    const at::Tensor& idx,
    const at::Tensor& weight,
    const int m) {
  assert(grad.device().is_cuda() == idx.device().is_cuda());
  assert(idx.device().is_cuda() == weight.device().is_cuda());
  if (grad.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return three_interpolate_backward_cuda(
        grad.contiguous(), idx.contiguous(), weight.contiguous(), m);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CPU is not implemented");
  }
}

} // namespace votenet
