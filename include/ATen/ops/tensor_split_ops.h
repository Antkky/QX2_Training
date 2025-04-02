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


struct TORCH_API tensor_split_sections {
  using schema = ::std::vector<at::Tensor> (const at::Tensor &, c10::SymInt, int64_t);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::tensor_split";
  static constexpr const char* overload_name = "sections";
  static constexpr const char* schema_str = "tensor_split.sections(Tensor(a -> *) self, SymInt sections, int dim=0) -> Tensor(a)[]";
  static ::std::vector<at::Tensor> call(const at::Tensor & self, c10::SymInt sections, int64_t dim);
  static ::std::vector<at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::SymInt sections, int64_t dim);
};

struct TORCH_API tensor_split_indices {
  using schema = ::std::vector<at::Tensor> (const at::Tensor &, c10::SymIntArrayRef, int64_t);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::tensor_split";
  static constexpr const char* overload_name = "indices";
  static constexpr const char* schema_str = "tensor_split.indices(Tensor(a -> *) self, SymInt[] indices, int dim=0) -> Tensor(a)[]";
  static ::std::vector<at::Tensor> call(const at::Tensor & self, c10::SymIntArrayRef indices, int64_t dim);
  static ::std::vector<at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::SymIntArrayRef indices, int64_t dim);
};

struct TORCH_API tensor_split_tensor_indices_or_sections {
  using schema = ::std::vector<at::Tensor> (const at::Tensor &, const at::Tensor &, int64_t);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::tensor_split";
  static constexpr const char* overload_name = "tensor_indices_or_sections";
  static constexpr const char* schema_str = "tensor_split.tensor_indices_or_sections(Tensor(a -> *) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]";
  static ::std::vector<at::Tensor> call(const at::Tensor & self, const at::Tensor & tensor_indices_or_sections, int64_t dim);
  static ::std::vector<at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor_indices_or_sections, int64_t dim);
};

}} // namespace at::_ops
