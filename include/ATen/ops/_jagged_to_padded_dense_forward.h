#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>



#include <ATen/ops/_jagged_to_padded_dense_forward_ops.h>

namespace at {


// aten::_jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, SymInt[] max_lengths, float padding_value=0.0) -> Tensor
inline at::Tensor _jagged_to_padded_dense_forward(const at::Tensor & values, at::TensorList offsets, at::IntArrayRef max_lengths, double padding_value=0.0) {
    return at::_ops::_jagged_to_padded_dense_forward::call(values, offsets, c10::fromIntArrayRefSlow(max_lengths), padding_value);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor _jagged_to_padded_dense_forward(const at::Tensor & values, at::TensorList offsets, at::IntArrayRef max_lengths, double padding_value=0.0) {
    return at::_ops::_jagged_to_padded_dense_forward::call(values, offsets, c10::fromIntArrayRefSlow(max_lengths), padding_value);
  }
}

// aten::_jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, SymInt[] max_lengths, float padding_value=0.0) -> Tensor
inline at::Tensor _jagged_to_padded_dense_forward_symint(const at::Tensor & values, at::TensorList offsets, c10::SymIntArrayRef max_lengths, double padding_value=0.0) {
    return at::_ops::_jagged_to_padded_dense_forward::call(values, offsets, max_lengths, padding_value);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor _jagged_to_padded_dense_forward(const at::Tensor & values, at::TensorList offsets, c10::SymIntArrayRef max_lengths, double padding_value=0.0) {
    return at::_ops::_jagged_to_padded_dense_forward::call(values, offsets, max_lengths, padding_value);
  }
}

}
