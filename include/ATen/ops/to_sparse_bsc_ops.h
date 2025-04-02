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


struct TORCH_API to_sparse_bsc {
  using schema = at::Tensor (const at::Tensor &, at::IntArrayRef, ::std::optional<int64_t>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::to_sparse_bsc";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "to_sparse_bsc(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor";
  static at::Tensor call(const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim);
};

}} // namespace at::_ops
