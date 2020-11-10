#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <iostream>
#include <vector>

namespace votenet {

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template <typename T>
__device__ inline bool devIoU(T const* const a, T const* const b, const float threshold) {
  T left = max(a[0], b[0]), right = min(a[3], b[3]);
  T back = max(a[1], b[1]), front = min(a[4], b[4]);
  T top = max(a[2], b[2]), bottom = min(a[5], b[5]);
  T width = max(right - left, (T)0), depth = max(front - back, (T)0), height = max(bottom - top, (T)0);
  T interV = width * depth * height;
  T Va = (a[3] - a[0]) * (a[4] - a[1]) * (a[5] - a[2]);
  T Vb = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]);
  return (interV / (Va + Vb - interV)) > threshold;
}

template <typename T>
__global__ void nms_3d_kernel(
    int n_boxes,
    float iou_threshold,
    const T* dev_boxes,
    unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  if (row_start > col_start) return;

  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ T block_boxes[threadsPerBlock * 6];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 6 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 0];
    block_boxes[threadIdx.x * 6 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 1];
    block_boxes[threadIdx.x * 6 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 2];
    block_boxes[threadIdx.x * 6 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 3];
    block_boxes[threadIdx.x * 6 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 4];
    block_boxes[threadIdx.x * 6 + 5] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 5];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * 6;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU<T>(cur_box, block_boxes + i * 6, iou_threshold)) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = ceil_div(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

at::Tensor nms_3d_cuda(const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold) {
  TORCH_CHECK(dets.is_cuda(), "dets must be a CUDA tensor");
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");

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
      scores.size(0))

#if defined(WITH_CUDA) || defined(WITH_HIP)
  at::cuda::CUDAGuard device_guard(dets.device());
#else
  TORCH_CHECK(false, "Not compiled with GPU support");
#endif

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto dets_sorted = dets.index_select(0, order_t).contiguous();

  int dets_num = dets.size(0);

  const int col_blocks = ceil_div(dets_num, threadsPerBlock);

  at::Tensor mask =
      at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      dets_sorted.scalar_type(), "nms_kernel_cuda", [&] {
        nms_3d_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            dets_sorted.data_ptr<scalar_t>(),
            (unsigned long long*)mask.data_ptr<int64_t>());
      });

  at::Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long* mask_host = (unsigned long long*)mask_cpu.data_ptr<int64_t>();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep =
      at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data_ptr<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < dets_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return order_t.index(
      {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
           .to(order_t.device(), keep.scalar_type())});
}

} // namespace votenet
