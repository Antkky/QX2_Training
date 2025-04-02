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


struct TORCH_API _scaled_dot_product_attention_math {
  using schema = ::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const ::std::optional<at::Tensor> &, double, bool, const ::std::optional<at::Tensor> &, ::std::optional<double>, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::_scaled_dot_product_attention_math";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "_scaled_dot_product_attention_math(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, Tensor? dropout_mask=None, *, float? scale=None, bool enable_gqa=False) -> (Tensor, Tensor)";
  static ::std::tuple<at::Tensor,at::Tensor> call(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, const ::std::optional<at::Tensor> & dropout_mask, ::std::optional<double> scale, bool enable_gqa);
  static ::std::tuple<at::Tensor,at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, const ::std::optional<at::Tensor> & dropout_mask, ::std::optional<double> scale, bool enable_gqa);
};

}} // namespace at::_ops
