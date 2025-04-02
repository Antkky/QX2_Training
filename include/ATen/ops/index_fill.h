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



#include <ATen/ops/index_fill_ops.h>

namespace at {


// aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
inline at::Tensor index_fill(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
    return at::_ops::index_fill_int_Scalar::call(self, dim, index, value);
}

// aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor
inline at::Tensor index_fill(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
    return at::_ops::index_fill_int_Tensor::call(self, dim, index, value);
}

// aten::index_fill.Dimname_Scalar(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
inline at::Tensor index_fill(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
    return at::_ops::index_fill_Dimname_Scalar::call(self, dim, index, value);
}

// aten::index_fill.Dimname_Tensor(Tensor self, Dimname dim, Tensor index, Tensor value) -> Tensor
inline at::Tensor index_fill(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
    return at::_ops::index_fill_Dimname_Tensor::call(self, dim, index, value);
}

// aten::index_fill.int_Scalar_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & index_fill_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
    return at::_ops::index_fill_int_Scalar_out::call(self, dim, index, value, out);
}
// aten::index_fill.int_Scalar_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & index_fill_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out) {
    return at::_ops::index_fill_int_Scalar_out::call(self, dim, index, value, out);
}

// aten::index_fill.int_Tensor_out(Tensor self, int dim, Tensor index, Tensor value, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & index_fill_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
    return at::_ops::index_fill_int_Tensor_out::call(self, dim, index, value, out);
}
// aten::index_fill.int_Tensor_out(Tensor self, int dim, Tensor index, Tensor value, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & index_fill_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value, at::Tensor & out) {
    return at::_ops::index_fill_int_Tensor_out::call(self, dim, index, value, out);
}

}
