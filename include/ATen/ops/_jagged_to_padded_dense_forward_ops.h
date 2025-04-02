#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API _jagged_to_padded_dense_forward {
  using schema = at::Tensor (const at::Tensor &, at::TensorList, c10::SymIntArrayRef, double);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::_jagged_to_padded_dense_forward";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "_jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, SymInt[] max_lengths, float padding_value=0.0) -> Tensor";
  static at::Tensor call(const at::Tensor & values, at::TensorList offsets, c10::SymIntArrayRef max_lengths, double padding_value);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & values, at::TensorList offsets, c10::SymIntArrayRef max_lengths, double padding_value);
};

}} // namespace at::_ops
