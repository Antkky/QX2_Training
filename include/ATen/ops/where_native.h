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
TORCH_API at::Tensor where(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other);
TORCH_API at::Tensor & where_self_out(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other, at::Tensor & out);
TORCH_API at::Tensor NestedTensor_where(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other);
TORCH_API at::Tensor & NestedTensor_where_out(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other, at::Tensor & out);
TORCH_API at::Tensor where(const at::Tensor & condition, const at::Scalar & self, const at::Tensor & other);
TORCH_API at::Tensor where(const at::Tensor & condition, const at::Tensor & self, const at::Scalar & other);
TORCH_API at::Tensor where(const at::Tensor & condition, const at::Scalar & self, const at::Scalar & other);
TORCH_API ::std::vector<at::Tensor> where(const at::Tensor & condition);
} // namespace native
} // namespace at
