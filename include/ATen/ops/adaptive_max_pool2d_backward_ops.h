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


struct TORCH_API adaptive_max_pool2d_backward_grad_input {
  using schema = at::Tensor & (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::adaptive_max_pool2d_backward";
  static constexpr const char* overload_name = "grad_input";
  static constexpr const char* schema_str = "adaptive_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input);
};

struct TORCH_API adaptive_max_pool2d_backward {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::adaptive_max_pool2d_backward";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor";
  static at::Tensor call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices);
};

}} // namespace at::_ops
