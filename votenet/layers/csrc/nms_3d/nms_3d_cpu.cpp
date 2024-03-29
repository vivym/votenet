#include "nms_3d.h"

namespace votenet {

template <typename scalar_t>
at::Tensor nms_3d_cpu_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold) {
  TORCH_CHECK(!dets.is_cuda(), "dets must be a CPU tensor");
  TORCH_CHECK(!scores.is_cuda(), "scores must be a CPU tensor");
  TORCH_CHECK(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong));

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto z1_t = dets.select(1, 2).contiguous();
  auto x2_t = dets.select(1, 3).contiguous();
  auto y2_t = dets.select(1, 4).contiguous();
  auto z2_t = dets.select(1, 5).contiguous();

  at::Tensor volumes_t = (x2_t - x1_t) * (y2_t - y1_t) * (z2_t - z1_t);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));
  at::Tensor keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto keep = keep_t.data_ptr<int64_t>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<scalar_t>();
  auto y1 = y1_t.data_ptr<scalar_t>();
  auto z1 = z1_t.data_ptr<scalar_t>();
  auto x2 = x2_t.data_ptr<scalar_t>();
  auto y2 = y2_t.data_ptr<scalar_t>();
  auto z2 = z2_t.data_ptr<scalar_t>();
  auto volumes = volumes_t.data_ptr<scalar_t>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    keep[num_to_keep++] = i;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto iz1 = z1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iz2 = z2[i];
    auto ivolume = volumes[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto zz1 = std::max(iz1, z1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);
      auto zz2 = std::min(iz2, z2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
      auto d = std::max(static_cast<scalar_t>(0), yy2 - yy1);
      auto h = std::max(static_cast<scalar_t>(0), zz2 - zz1);
      auto inter = w * d * h;
      auto ovr = inter / (ivolume + volumes[j] - inter);
      if (ovr > iou_threshold)
        suppressed[j] = 1;
    }
  }
  return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

at::Tensor nms_3d_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold) {
  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 6,
      "boxes should have 6 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));

  auto result = at::empty({0}, dets.options());

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms", [&] {
    result = nms_3d_cpu_kernel<scalar_t>(dets, scores, iou_threshold);
  });
  return result;
}

} // namespace votenet
