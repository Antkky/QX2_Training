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


struct TORCH_API fft_fftshift {
  using schema = at::Tensor (const at::Tensor &, at::OptionalIntArrayRef);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::fft_fftshift";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "fft_fftshift(Tensor self, int[1]? dim=None) -> Tensor";
  static at::Tensor call(const at::Tensor & self, at::OptionalIntArrayRef dim);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::OptionalIntArrayRef dim);
};

}} // namespace at::_ops
