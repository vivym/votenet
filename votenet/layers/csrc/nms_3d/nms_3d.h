#pragma once
#include <torch/types.h>

namespace votenet {

at::Tensor nms_3d_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);

#if defined(WITH_CUDA) || defined(WITH_HIP)
at::Tensor nms_3d_cuda(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
inline at::Tensor nms_3d(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold) {
  assert(dets.device().is_cuda() == scores.device().is_cuda());
  if (dets.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return nms_3d_cuda(
        dets.contiguous(), scores.contiguous(), iou_threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return nms_3d_cpu(dets.contiguous(), scores.contiguous(), iou_threshold);
}

} // namespace votenet
