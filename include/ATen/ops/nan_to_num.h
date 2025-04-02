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



#include <ATen/ops/nan_to_num_ops.h>

namespace at {


// aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
inline at::Tensor nan_to_num(const at::Tensor & self, ::std::optional<double> nan=::std::nullopt, ::std::optional<double> posinf=::std::nullopt, ::std::optional<double> neginf=::std::nullopt) {
    return at::_ops::nan_to_num::call(self, nan, posinf, neginf);
}

// aten::nan_to_num_(Tensor(a!) self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor(a!)
inline at::Tensor & nan_to_num_(at::Tensor & self, ::std::optional<double> nan=::std::nullopt, ::std::optional<double> posinf=::std::nullopt, ::std::optional<double> neginf=::std::nullopt) {
    return at::_ops::nan_to_num_::call(self, nan, posinf, neginf);
}

// aten::nan_to_num.out(Tensor self, float? nan=None, float? posinf=None, float? neginf=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & nan_to_num_out(at::Tensor & out, const at::Tensor & self, ::std::optional<double> nan=::std::nullopt, ::std::optional<double> posinf=::std::nullopt, ::std::optional<double> neginf=::std::nullopt) {
    return at::_ops::nan_to_num_out::call(self, nan, posinf, neginf, out);
}
// aten::nan_to_num.out(Tensor self, float? nan=None, float? posinf=None, float? neginf=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & nan_to_num_outf(const at::Tensor & self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf, at::Tensor & out) {
    return at::_ops::nan_to_num_out::call(self, nan, posinf, neginf, out);
}

}
