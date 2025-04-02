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



#include <ATen/ops/_nested_view_from_jagged_ops.h>

namespace at {


// aten::_nested_view_from_jagged(Tensor(a) self, Tensor offsets, Tensor dummy, Tensor? lengths=None, int ragged_idx=1, Tensor? min_seqlen=None, Tensor? max_seqlen=None) -> Tensor(a)
inline at::Tensor _nested_view_from_jagged(const at::Tensor & self, const at::Tensor & offsets, const at::Tensor & dummy, const ::std::optional<at::Tensor> & lengths={}, int64_t ragged_idx=1, const ::std::optional<at::Tensor> & min_seqlen={}, const ::std::optional<at::Tensor> & max_seqlen={}) {
    return at::_ops::_nested_view_from_jagged::call(self, offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen);
}

}
