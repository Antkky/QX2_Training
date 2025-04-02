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



#include <ATen/ops/sum_to_size_ops.h>

namespace at {


namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor sum_to_size(const at::Tensor & self, at::IntArrayRef size) {
    return at::_ops::sum_to_size::call(self, c10::fromIntArrayRefSlow(size));
  }
}

namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor sum_to_size(const at::Tensor & self, c10::SymIntArrayRef size) {
    return at::_ops::sum_to_size::call(self, size);
  }
}

}
