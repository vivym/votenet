#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace votenet {

#define CHECK_CUDA(x)                                          \
  do {                                                         \
    AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                         \
  do {                                                              \
    AT_ASSERTM(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                              \
  do {                                               \
    AT_ASSERTM(x.scalar_type() == at::ScalarType::Int, \
               #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                              \
  do {                                                 \
    AT_ASSERTM(x.scalar_type() == at::ScalarType::Float, \
             #x " must be a float tensor");            \
  } while (0)

} // namespace votenet
