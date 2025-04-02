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


struct TORCH_API linalg_vector_norm {
  using schema = at::Tensor (const at::Tensor &, const at::Scalar &, at::OptionalIntArrayRef, bool, ::std::optional<at::ScalarType>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::linalg_vector_norm";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor";
  static at::Tensor call(const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype);
};

struct TORCH_API linalg_vector_norm_out {
  using schema = at::Tensor & (const at::Tensor &, const at::Scalar &, at::OptionalIntArrayRef, bool, ::std::optional<at::ScalarType>, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::linalg_vector_norm";
  static constexpr const char* overload_name = "out";
  static constexpr const char* schema_str = "linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out);
};

}} // namespace at::_ops
