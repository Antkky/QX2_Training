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


struct TORCH_API diff {
  using schema = at::Tensor (const at::Tensor &, int64_t, int64_t, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::diff";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor";
  static at::Tensor call(const at::Tensor & self, int64_t n, int64_t dim, const ::std::optional<at::Tensor> & prepend, const ::std::optional<at::Tensor> & append);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n, int64_t dim, const ::std::optional<at::Tensor> & prepend, const ::std::optional<at::Tensor> & append);
};

struct TORCH_API diff_out {
  using schema = at::Tensor & (const at::Tensor &, int64_t, int64_t, const ::std::optional<at::Tensor> &, const ::std::optional<at::Tensor> &, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::diff";
  static constexpr const char* overload_name = "out";
  static constexpr const char* schema_str = "diff.out(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None, *, Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & self, int64_t n, int64_t dim, const ::std::optional<at::Tensor> & prepend, const ::std::optional<at::Tensor> & append, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n, int64_t dim, const ::std::optional<at::Tensor> & prepend, const ::std::optional<at::Tensor> & append, at::Tensor & out);
};

}} // namespace at::_ops
