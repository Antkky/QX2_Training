#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API at::Tensor set(const at::Tensor & self, at::Storage source);
TORCH_API at::Tensor & set_source_Storage_out(const at::Tensor & self, at::Storage source, at::Tensor & out);
TORCH_API at::Tensor & set_(at::Tensor & self, at::Storage source);
TORCH_API at::Tensor set_symint(const at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride={});
TORCH_API at::Tensor & set_source_Storage_storage_offset_out_symint(const at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::Tensor & out);
TORCH_API at::Tensor & set_storage_cpu_(at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride={});
TORCH_API at::Tensor & set_storage_cuda_(at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride={});
TORCH_API at::Tensor & set_storage_meta__symint(at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride={});
TORCH_API at::Tensor & set_storage_quantized_(at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride={});
TORCH_API at::Tensor & set__symint(at::Tensor & self, const at::Tensor & source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride={});
TORCH_API at::Tensor set(const at::Tensor & self, const at::Tensor & source);
TORCH_API at::Tensor & set_source_Tensor_out(const at::Tensor & self, const at::Tensor & source, at::Tensor & out);
TORCH_API at::Tensor & set_tensor_(at::Tensor & self, const at::Tensor & source);
TORCH_API at::Tensor set(const at::Tensor & self);
TORCH_API at::Tensor & set_out(const at::Tensor & self, at::Tensor & out);
TORCH_API at::Tensor & set_cpu_(at::Tensor & self);
TORCH_API at::Tensor & set_cuda_(at::Tensor & self);
TORCH_API at::Tensor & set_meta_(at::Tensor & self);
} // namespace native
} // namespace at
