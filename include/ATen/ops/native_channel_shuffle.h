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



#include <ATen/ops/native_channel_shuffle_ops.h>

namespace at {


// aten::native_channel_shuffle(Tensor self, SymInt groups) -> Tensor
inline at::Tensor native_channel_shuffle(const at::Tensor & self, int64_t groups) {
    return at::_ops::native_channel_shuffle::call(self, groups);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor native_channel_shuffle(const at::Tensor & self, int64_t groups) {
    return at::_ops::native_channel_shuffle::call(self, groups);
  }
}

// aten::native_channel_shuffle(Tensor self, SymInt groups) -> Tensor
inline at::Tensor native_channel_shuffle_symint(const at::Tensor & self, c10::SymInt groups) {
    return at::_ops::native_channel_shuffle::call(self, groups);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor native_channel_shuffle(const at::Tensor & self, c10::SymInt groups) {
    return at::_ops::native_channel_shuffle::call(self, groups);
  }
}

}
