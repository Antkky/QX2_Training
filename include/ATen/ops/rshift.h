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



#include <ATen/ops/rshift_ops.h>

namespace at {


// aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor
inline at::Tensor __rshift__(const at::Tensor & self, const at::Scalar & other) {
    return at::_ops::__rshift___Scalar::call(self, other);
}

// aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor
inline at::Tensor __rshift__(const at::Tensor & self, const at::Tensor & other) {
    return at::_ops::__rshift___Tensor::call(self, other);
}

// aten::__rshift__.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & __rshift___out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other) {
    return at::_ops::__rshift___Scalar_out::call(self, other, out);
}
// aten::__rshift__.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & __rshift___outf(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    return at::_ops::__rshift___Scalar_out::call(self, other, out);
}

// aten::__rshift__.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & __rshift___out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
    return at::_ops::__rshift___Tensor_out::call(self, other, out);
}
// aten::__rshift__.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & __rshift___outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    return at::_ops::__rshift___Tensor_out::call(self, other, out);
}

}
