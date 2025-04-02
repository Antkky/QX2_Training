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


struct TORCH_API conv3d {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, c10::SymIntArrayRef, c10::SymIntArrayRef, c10::SymIntArrayRef, c10::SymInt);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::conv3d";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "conv3d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0, SymInt[3] dilation=1, SymInt groups=1) -> Tensor";
  static at::Tensor call(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups);
};

struct TORCH_API conv3d_padding {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, c10::SymIntArrayRef, c10::string_view, c10::SymIntArrayRef, c10::SymInt);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::conv3d";
  static constexpr const char* overload_name = "padding";
  static constexpr const char* schema_str = "conv3d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=1, str padding=\"valid\", SymInt[3] dilation=1, SymInt groups=1) -> Tensor";
  static at::Tensor call(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::string_view padding, c10::SymIntArrayRef dilation, c10::SymInt groups);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::string_view padding, c10::SymIntArrayRef dilation, c10::SymInt groups);
};

}} // namespace at::_ops
