#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <optional>



#include <ATen/ops/tensor_split_ops.h>

namespace at {


// aten::tensor_split.sections(Tensor(a -> *) self, SymInt sections, int dim=0) -> Tensor(a)[]
inline ::std::vector<at::Tensor> tensor_split(const at::Tensor & self, int64_t sections, int64_t dim=0) {
    return at::_ops::tensor_split_sections::call(self, sections, dim);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  ::std::vector<at::Tensor> tensor_split(const at::Tensor & self, int64_t sections, int64_t dim=0) {
    return at::_ops::tensor_split_sections::call(self, sections, dim);
  }
}

// aten::tensor_split.sections(Tensor(a -> *) self, SymInt sections, int dim=0) -> Tensor(a)[]
inline ::std::vector<at::Tensor> tensor_split_symint(const at::Tensor & self, c10::SymInt sections, int64_t dim=0) {
    return at::_ops::tensor_split_sections::call(self, sections, dim);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  ::std::vector<at::Tensor> tensor_split(const at::Tensor & self, c10::SymInt sections, int64_t dim=0) {
    return at::_ops::tensor_split_sections::call(self, sections, dim);
  }
}

// aten::tensor_split.indices(Tensor(a -> *) self, SymInt[] indices, int dim=0) -> Tensor(a)[]
inline ::std::vector<at::Tensor> tensor_split(const at::Tensor & self, at::IntArrayRef indices, int64_t dim=0) {
    return at::_ops::tensor_split_indices::call(self, c10::fromIntArrayRefSlow(indices), dim);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  ::std::vector<at::Tensor> tensor_split(const at::Tensor & self, at::IntArrayRef indices, int64_t dim=0) {
    return at::_ops::tensor_split_indices::call(self, c10::fromIntArrayRefSlow(indices), dim);
  }
}

// aten::tensor_split.indices(Tensor(a -> *) self, SymInt[] indices, int dim=0) -> Tensor(a)[]
inline ::std::vector<at::Tensor> tensor_split_symint(const at::Tensor & self, c10::SymIntArrayRef indices, int64_t dim=0) {
    return at::_ops::tensor_split_indices::call(self, indices, dim);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  ::std::vector<at::Tensor> tensor_split(const at::Tensor & self, c10::SymIntArrayRef indices, int64_t dim=0) {
    return at::_ops::tensor_split_indices::call(self, indices, dim);
  }
}

// aten::tensor_split.tensor_indices_or_sections(Tensor(a -> *) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]
inline ::std::vector<at::Tensor> tensor_split(const at::Tensor & self, const at::Tensor & tensor_indices_or_sections, int64_t dim=0) {
    return at::_ops::tensor_split_tensor_indices_or_sections::call(self, tensor_indices_or_sections, dim);
}

}
