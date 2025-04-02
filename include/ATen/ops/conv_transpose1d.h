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



#include <ATen/ops/conv_transpose1d_ops.h>

namespace at {


// aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, SymInt[1] padding=0, SymInt[1] output_padding=0, SymInt groups=1, SymInt[1] dilation=1) -> Tensor
inline at::Tensor conv_transpose1d(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias={}, at::IntArrayRef stride=1, at::IntArrayRef padding=0, at::IntArrayRef output_padding=0, int64_t groups=1, at::IntArrayRef dilation=1) {
    return at::_ops::conv_transpose1d::call(input, weight, bias, c10::fromIntArrayRefSlow(stride), c10::fromIntArrayRefSlow(padding), c10::fromIntArrayRefSlow(output_padding), groups, c10::fromIntArrayRefSlow(dilation));
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor conv_transpose1d(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias={}, at::IntArrayRef stride=1, at::IntArrayRef padding=0, at::IntArrayRef output_padding=0, int64_t groups=1, at::IntArrayRef dilation=1) {
    return at::_ops::conv_transpose1d::call(input, weight, bias, c10::fromIntArrayRefSlow(stride), c10::fromIntArrayRefSlow(padding), c10::fromIntArrayRefSlow(output_padding), groups, c10::fromIntArrayRefSlow(dilation));
  }
}

// aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, SymInt[1] padding=0, SymInt[1] output_padding=0, SymInt groups=1, SymInt[1] dilation=1) -> Tensor
inline at::Tensor conv_transpose1d_symint(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias={}, c10::SymIntArrayRef stride=c10::SymInt(1), c10::SymIntArrayRef padding=c10::SymInt(0), c10::SymIntArrayRef output_padding=c10::SymInt(0), c10::SymInt groups=1, c10::SymIntArrayRef dilation=c10::SymInt(1)) {
    return at::_ops::conv_transpose1d::call(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor conv_transpose1d(const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias={}, c10::SymIntArrayRef stride=c10::SymInt(1), c10::SymIntArrayRef padding=c10::SymInt(0), c10::SymIntArrayRef output_padding=c10::SymInt(0), c10::SymInt groups=1, c10::SymIntArrayRef dilation=c10::SymInt(1)) {
    return at::_ops::conv_transpose1d::call(input, weight, bias, stride, padding, output_padding, groups, dilation);
  }
}

}
