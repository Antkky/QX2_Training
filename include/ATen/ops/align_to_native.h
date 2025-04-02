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


namespace at {
namespace native {
TORCH_API at::Tensor align_to(const at::Tensor & self, at::DimnameList names);
TORCH_API at::Tensor align_to(const at::Tensor & self, at::DimnameList order, int64_t ellipsis_idx);
} // namespace native
} // namespace at
