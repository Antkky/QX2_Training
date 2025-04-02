#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>
#include <ATen/ops/_softmax_meta.h>

namespace at {
namespace native {
struct TORCH_API structured_softmax_cpu_out : public at::meta::structured__softmax {
void impl(const at::Tensor & self, int64_t dim, bool half_to_float, const at::Tensor & out);
};
struct TORCH_API structured_softmax_cuda_out : public at::meta::structured__softmax {
void impl(const at::Tensor & self, int64_t dim, bool half_to_float, const at::Tensor & out);
};
TORCH_API at::Tensor softmax_nested(const at::Tensor & self, int64_t dim, bool half_to_float);
TORCH_API at::Tensor mkldnn_softmax(const at::Tensor & self, int64_t dim, bool half_to_float);
} // namespace native
} // namespace at
