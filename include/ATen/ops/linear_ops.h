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


struct TORCH_API linear {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::linear";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor";
  static at::Tensor call(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias);
};

struct TORCH_API linear_out {
  using schema = at::Tensor & (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::linear";
  static constexpr const char* overload_name = "out";
  static constexpr const char* schema_str = "linear.out(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & out);
};

}} // namespace at::_ops
