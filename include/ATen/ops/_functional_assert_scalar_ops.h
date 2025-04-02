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


struct TORCH_API _functional_assert_scalar {
  using schema = at::Tensor (const at::Scalar &, c10::string_view, const at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::_functional_assert_scalar";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "_functional_assert_scalar(Scalar self, str assert_msg, Tensor dep_token) -> Tensor";
  static at::Tensor call(const at::Scalar & self, c10::string_view assert_msg, const at::Tensor & dep_token);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, c10::string_view assert_msg, const at::Tensor & dep_token);
};

}} // namespace at::_ops
