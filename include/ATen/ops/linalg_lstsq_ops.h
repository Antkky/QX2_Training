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


struct TORCH_API linalg_lstsq {
  using schema = ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, ::std::optional<double>, ::std::optional<c10::string_view>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::linalg_lstsq";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "linalg_lstsq(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)";
  static ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> call(const at::Tensor & self, const at::Tensor & b, ::std::optional<double> rcond, ::std::optional<c10::string_view> driver);
  static ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & b, ::std::optional<double> rcond, ::std::optional<c10::string_view> driver);
};

struct TORCH_API linalg_lstsq_out {
  using schema = ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> (const at::Tensor &, const at::Tensor &, ::std::optional<double>, ::std::optional<c10::string_view>, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::linalg_lstsq";
  static constexpr const char* overload_name = "out";
  static constexpr const char* schema_str = "linalg_lstsq.out(Tensor self, Tensor b, float? rcond=None, *, str? driver=None, Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values) -> (Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values)";
  static ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> call(const at::Tensor & self, const at::Tensor & b, ::std::optional<double> rcond, ::std::optional<c10::string_view> driver, at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values);
  static ::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & b, ::std::optional<double> rcond, ::std::optional<c10::string_view> driver, at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values);
};

}} // namespace at::_ops
