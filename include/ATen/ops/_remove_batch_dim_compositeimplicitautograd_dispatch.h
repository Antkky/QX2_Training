#pragma once
// @generated by torchgen/gen.py from DispatchKeyFunction.h

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {

namespace compositeimplicitautograd {

TORCH_API at::Tensor _remove_batch_dim(const at::Tensor & self, int64_t level, int64_t batch_size, int64_t out_dim);
TORCH_API at::Tensor _remove_batch_dim_symint(const at::Tensor & self, int64_t level, c10::SymInt batch_size, int64_t out_dim);

} // namespace compositeimplicitautograd
} // namespace at
