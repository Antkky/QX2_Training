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
#include <ATen/ops/scatter_add_meta.h>

namespace at {
namespace native {
struct TORCH_API structured_scatter_add : public at::meta::structured_scatter_add {
void impl(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, const at::Tensor & out);
};
TORCH_API at::Tensor scatter_add(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src);
} // namespace native
} // namespace at
