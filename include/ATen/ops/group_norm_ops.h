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


struct TORCH_API group_norm {
  using schema = at::Tensor (const at::Tensor &, int64_t, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, double, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::group_norm";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor";
  static at::Tensor call(const at::Tensor & input, int64_t num_groups, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps, bool cudnn_enabled);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, int64_t num_groups, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps, bool cudnn_enabled);
};

}} // namespace at::_ops
