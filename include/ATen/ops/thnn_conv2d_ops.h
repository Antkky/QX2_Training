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


struct TORCH_API thnn_conv2d_out {
  using schema = at::Tensor & (const at::Tensor &, const at::Tensor &, c10::SymIntArrayRef, const ::std::optional<at::Tensor> &, c10::SymIntArrayRef, c10::SymIntArrayRef, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::thnn_conv2d";
  static constexpr const char* overload_name = "out";
  static constexpr const char* schema_str = "thnn_conv2d.out(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, *, Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & out);
};

struct TORCH_API thnn_conv2d {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, c10::SymIntArrayRef, const ::std::optional<at::Tensor> &, c10::SymIntArrayRef, c10::SymIntArrayRef);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::thnn_conv2d";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "thnn_conv2d(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0) -> Tensor";
  static at::Tensor call(const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding);
};

}} // namespace at::_ops
