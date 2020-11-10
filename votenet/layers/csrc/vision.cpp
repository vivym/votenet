#include <torch/extension.h>
#include "ball_query/ball_query.h"
#include "box_iou_rotated/box_iou_rotated.h"
#include "group_points/group_points.h"
#include "nms_3d/nms_3d.h"
#include "nms_rotated/nms_rotated.h"
#include "sampling/sampling.h"
#include "three_interpolate/interpolate.h"

namespace votenet {

#if defined(WITH_CUDA) || defined(WITH_HIP)
extern int get_cudart_version();
#endif

std::string get_cuda_version() {
#if defined(WITH_CUDA) || defined(WITH_HIP)
  std::ostringstream oss;

#if defined(WITH_CUDA)
  oss << "CUDA ";
#else
  oss << "HIP ";
#endif

  // copied from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp#L231
  auto printCudaStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };
  printCudaStyleVersion(get_cudart_version());
  return oss.str();
#else // neither CUDA nor HIP
  return std::string("not available");
#endif
}

// similar to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
std::string get_compiler_version() {
  std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__

#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif

  { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
#endif
#endif

#if defined(__clang_major__)
  {
    ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
       << __clang_patchlevel__;
  }
#endif

#if defined(_MSC_VER)
  { ss << "MSVC " << _MSC_FULL_VER; }
#endif
  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
  m.def("get_cuda_version", &get_cuda_version, "get_cuda_version");

  m.def("ball_query", &ball_query, "ball query");

  m.def("box_iou_rotated", &box_iou_rotated, "IoU for rotated boxes");

  m.def("group_points_forward", &group_points_forward, "group_points_forward");
  m.def("group_points_backward", &group_points_backward, "group_points_backward");

  m.def("nms_3d", &nms_3d, "NMS for 3d boxes");

  m.def("nms_rotated", &nms_rotated, "NMS for rotated boxes");

  m.def("gather_points_forward", &gather_points_forward, "gather_points_forward");
  m.def("gather_points_backward", &gather_points_backward, "gather_points_backward");
  m.def("furthest_point_sampling", &furthest_point_sampling, "furthest_point_sampling");

  m.def("three_nn", &three_nn, "three_nn");
  m.def("three_interpolate_forward", &three_interpolate_forward, "three_interpolate_forward");
  m.def("three_interpolate_backward", &three_interpolate_backward, "three_interpolate_backward");
}

} // namespace votenet
