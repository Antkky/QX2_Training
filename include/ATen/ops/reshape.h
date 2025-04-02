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



#include <ATen/ops/reshape_ops.h>

namespace at {


// aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)
inline at::Tensor reshape(const at::Tensor & self, at::IntArrayRef shape) {
    return at::_ops::reshape::call(self, c10::fromIntArrayRefSlow(shape));
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor reshape(const at::Tensor & self, at::IntArrayRef shape) {
    return at::_ops::reshape::call(self, c10::fromIntArrayRefSlow(shape));
  }
}

// aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)
inline at::Tensor reshape_symint(const at::Tensor & self, c10::SymIntArrayRef shape) {
    return at::_ops::reshape::call(self, shape);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor reshape(const at::Tensor & self, c10::SymIntArrayRef shape) {
    return at::_ops::reshape::call(self, shape);
  }
}

}
