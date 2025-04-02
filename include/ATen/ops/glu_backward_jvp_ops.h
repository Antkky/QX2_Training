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


struct TORCH_API glu_backward_jvp {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::glu_backward_jvp";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "glu_backward_jvp(Tensor grad_x, Tensor grad_glu, Tensor x, Tensor dgrad_glu, Tensor dx, int dim) -> Tensor";
  static at::Tensor call(const at::Tensor & grad_x, const at::Tensor & grad_glu, const at::Tensor & x, const at::Tensor & dgrad_glu, const at::Tensor & dx, int64_t dim);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_x, const at::Tensor & grad_glu, const at::Tensor & x, const at::Tensor & dgrad_glu, const at::Tensor & dx, int64_t dim);
};

struct TORCH_API glu_backward_jvp_out {
  using schema = at::Tensor & (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::glu_backward_jvp";
  static constexpr const char* overload_name = "out";
  static constexpr const char* schema_str = "glu_backward_jvp.out(Tensor grad_x, Tensor grad_glu, Tensor x, Tensor dgrad_glu, Tensor dx, int dim, *, Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & grad_x, const at::Tensor & grad_glu, const at::Tensor & x, const at::Tensor & dgrad_glu, const at::Tensor & dx, int64_t dim, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_x, const at::Tensor & grad_glu, const at::Tensor & x, const at::Tensor & dgrad_glu, const at::Tensor & dx, int64_t dim, at::Tensor & out);
};

}} // namespace at::_ops
