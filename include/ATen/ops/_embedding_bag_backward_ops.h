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


struct TORCH_API _embedding_bag_backward {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, c10::SymInt, bool, int64_t, bool, const ::std::optional<at::Tensor> &, int64_t);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::_embedding_bag_backward";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, SymInt num_weights, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor";
  static at::Tensor call(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, c10::SymInt num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, int64_t padding_idx);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, c10::SymInt num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, int64_t padding_idx);
};

}} // namespace at::_ops
