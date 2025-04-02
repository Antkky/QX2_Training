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


struct TORCH_API fbgemm_pack_quantized_matrix {
  using schema = at::Tensor (const at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::fbgemm_pack_quantized_matrix";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "fbgemm_pack_quantized_matrix(Tensor input) -> Tensor";
  static at::Tensor call(const at::Tensor & input);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input);
};

struct TORCH_API fbgemm_pack_quantized_matrix_KN {
  using schema = at::Tensor (const at::Tensor &, int64_t, int64_t);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::fbgemm_pack_quantized_matrix";
  static constexpr const char* overload_name = "KN";
  static constexpr const char* schema_str = "fbgemm_pack_quantized_matrix.KN(Tensor input, int K, int N) -> Tensor";
  static at::Tensor call(const at::Tensor & input, int64_t K, int64_t N);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, int64_t K, int64_t N);
};

}} // namespace at::_ops
