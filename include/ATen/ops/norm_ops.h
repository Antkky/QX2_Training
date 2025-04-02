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


struct TORCH_API norm_ScalarOpt_dtype {
  using schema = at::Tensor (const at::Tensor &, const ::std::optional<at::Scalar> &, at::ScalarType);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::norm";
  static constexpr const char* overload_name = "ScalarOpt_dtype";
  static constexpr const char* schema_str = "norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor";
  static at::Tensor call(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::ScalarType dtype);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::ScalarType dtype);
};

struct TORCH_API norm_Scalar {
  using schema = at::Tensor (const at::Tensor &, const at::Scalar &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::norm";
  static constexpr const char* overload_name = "Scalar";
  static constexpr const char* schema_str = "norm.Scalar(Tensor self, Scalar p=2) -> Tensor";
  static at::Tensor call(const at::Tensor & self, const at::Scalar & p);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & p);
};

struct TORCH_API norm_ScalarOpt_dim_dtype {
  using schema = at::Tensor (const at::Tensor &, const ::std::optional<at::Scalar> &, at::IntArrayRef, bool, at::ScalarType);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::norm";
  static constexpr const char* overload_name = "ScalarOpt_dim_dtype";
  static constexpr const char* schema_str = "norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor";
  static at::Tensor call(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype);
};

struct TORCH_API norm_ScalarOpt_dim {
  using schema = at::Tensor (const at::Tensor &, const ::std::optional<at::Scalar> &, at::IntArrayRef, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::norm";
  static constexpr const char* overload_name = "ScalarOpt_dim";
  static constexpr const char* schema_str = "norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor";
  static at::Tensor call(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim);
};

struct TORCH_API norm_dtype_out {
  using schema = at::Tensor & (const at::Tensor &, const ::std::optional<at::Scalar> &, at::IntArrayRef, bool, at::ScalarType, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::norm";
  static constexpr const char* overload_name = "dtype_out";
  static constexpr const char* schema_str = "norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor & out);
};

struct TORCH_API norm_out {
  using schema = at::Tensor & (const at::Tensor &, const ::std::optional<at::Scalar> &, at::IntArrayRef, bool, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::norm";
  static constexpr const char* overload_name = "out";
  static constexpr const char* schema_str = "norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::Tensor & out);
};

struct TORCH_API norm_names_ScalarOpt_dim_dtype {
  using schema = at::Tensor (const at::Tensor &, const ::std::optional<at::Scalar> &, at::DimnameList, bool, at::ScalarType);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::norm";
  static constexpr const char* overload_name = "names_ScalarOpt_dim_dtype";
  static constexpr const char* schema_str = "norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor";
  static at::Tensor call(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype);
};

struct TORCH_API norm_names_ScalarOpt_dim {
  using schema = at::Tensor (const at::Tensor &, const ::std::optional<at::Scalar> &, at::DimnameList, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::norm";
  static constexpr const char* overload_name = "names_ScalarOpt_dim";
  static constexpr const char* schema_str = "norm.names_ScalarOpt_dim(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False) -> Tensor";
  static at::Tensor call(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim);
};

struct TORCH_API norm_names_dtype_out {
  using schema = at::Tensor & (const at::Tensor &, const ::std::optional<at::Scalar> &, at::DimnameList, bool, at::ScalarType, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::norm";
  static constexpr const char* overload_name = "names_dtype_out";
  static constexpr const char* schema_str = "norm.names_dtype_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype, at::Tensor & out);
};

struct TORCH_API norm_names_out {
  using schema = at::Tensor & (const at::Tensor &, const ::std::optional<at::Scalar> &, at::DimnameList, bool, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::norm";
  static constexpr const char* overload_name = "names_out";
  static constexpr const char* schema_str = "norm.names_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::Tensor & out);
};

struct TORCH_API norm_ScalarOpt_dtype_out {
  using schema = at::Tensor & (const at::Tensor &, const ::std::optional<at::Scalar> &, at::ScalarType, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::norm";
  static constexpr const char* overload_name = "ScalarOpt_dtype_out";
  static constexpr const char* schema_str = "norm.ScalarOpt_dtype_out(Tensor self, Scalar? p, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::ScalarType dtype, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::ScalarType dtype, at::Tensor & out);
};

struct TORCH_API norm_Scalar_out {
  using schema = at::Tensor & (const at::Tensor &, const at::Scalar &, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::norm";
  static constexpr const char* overload_name = "Scalar_out";
  static constexpr const char* schema_str = "norm.Scalar_out(Tensor self, Scalar p=2, *, Tensor(a!) out) -> Tensor(a!)";
  static at::Tensor & call(const at::Tensor & self, const at::Scalar & p, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & p, at::Tensor & out);
};

}} // namespace at::_ops
