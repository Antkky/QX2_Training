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


struct TORCH_API _weight_int4pack_mm {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, int64_t, const at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::_weight_int4pack_mm";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "_weight_int4pack_mm(Tensor self, Tensor mat2, int qGroupSize, Tensor qScaleAndZeros) -> Tensor";
  static at::Tensor call(const at::Tensor & self, const at::Tensor & mat2, int64_t qGroupSize, const at::Tensor & qScaleAndZeros);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2, int64_t qGroupSize, const at::Tensor & qScaleAndZeros);
};

}} // namespace at::_ops
