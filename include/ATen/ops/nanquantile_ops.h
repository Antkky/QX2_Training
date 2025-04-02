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


struct TORCH_API nanquantile {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, ::std::optional<int64_t>, bool, c10::string_view);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::nanquantile";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor";
  static at::Tensor call(const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation);
};

struct TORCH_API nanquantile_out {
  using schema = at::Tensor & (const at::Tensor &, const at::Tensor &, ::std::optional<int64_t>, bool, c10::string_view, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::nanquantile";
  static constexpr const char* overload_name = "out";
  static constexpr const char* schema_str = "nanquantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear', Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out);
};

struct TORCH_API nanquantile_scalar {
  using schema = at::Tensor (const at::Tensor &, double, ::std::optional<int64_t>, bool, c10::string_view);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::nanquantile";
  static constexpr const char* overload_name = "scalar";
  static constexpr const char* schema_str = "nanquantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor";
  static at::Tensor call(const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation);
};

struct TORCH_API nanquantile_scalar_out {
  using schema = at::Tensor & (const at::Tensor &, double, ::std::optional<int64_t>, bool, c10::string_view, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::nanquantile";
  static constexpr const char* overload_name = "scalar_out";
  static constexpr const char* schema_str = "nanquantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation='linear', Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out);
};

}} // namespace at::_ops
