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


struct TORCH_API _sparse_softmax_int {
  using schema = at::Tensor (const at::Tensor &, int64_t, ::std::optional<at::ScalarType>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::_sparse_softmax";
  static constexpr const char* overload_name = "int";
  static constexpr const char* schema_str = "_sparse_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor";
  static at::Tensor call(const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype);
};

struct TORCH_API _sparse_softmax_Dimname {
  using schema = at::Tensor (const at::Tensor &, at::Dimname, ::std::optional<at::ScalarType>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::_sparse_softmax";
  static constexpr const char* overload_name = "Dimname";
  static constexpr const char* schema_str = "_sparse_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor";
  static at::Tensor call(const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype);
};

struct TORCH_API _sparse_softmax {
  using schema = at::Tensor (const at::Tensor &, int64_t, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::_sparse_softmax";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "_sparse_softmax(Tensor self, int dim, bool half_to_float) -> Tensor";
  static at::Tensor call(const at::Tensor & self, int64_t dim, bool half_to_float);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float);
};

struct TORCH_API _sparse_softmax_out {
  using schema = at::Tensor & (const at::Tensor &, int64_t, bool, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::_sparse_softmax";
  static constexpr const char* overload_name = "out";
  static constexpr const char* schema_str = "_sparse_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out);
};

}} // namespace at::_ops
