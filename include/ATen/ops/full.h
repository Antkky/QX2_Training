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



#include <ATen/ops/full_ops.h>

namespace at {


// aten::full.names(int[] size, Scalar fill_value, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor full(at::IntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::DimnameList> names, at::TensorOptions options={}) {
    return at::_ops::full_names::call(size, fill_value, names, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
// aten::full.names(int[] size, Scalar fill_value, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor full(at::IntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::full_names::call(size, fill_value, names, dtype, layout, device, pin_memory);
}

// aten::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor full(at::IntArrayRef size, const at::Scalar & fill_value, at::TensorOptions options={}) {
    return at::_ops::full::call(c10::fromIntArrayRefSlow(size), fill_value, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor full(at::IntArrayRef size, const at::Scalar & fill_value, at::TensorOptions options={}) {
    return at::_ops::full::call(c10::fromIntArrayRefSlow(size), fill_value, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor full(at::IntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::full::call(c10::fromIntArrayRefSlow(size), fill_value, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor full(at::IntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::full::call(c10::fromIntArrayRefSlow(size), fill_value, dtype, layout, device, pin_memory);
  }
}

// aten::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor full_symint(c10::SymIntArrayRef size, const at::Scalar & fill_value, at::TensorOptions options={}) {
    return at::_ops::full::call(size, fill_value, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor full(c10::SymIntArrayRef size, const at::Scalar & fill_value, at::TensorOptions options={}) {
    return at::_ops::full::call(size, fill_value, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor full_symint(c10::SymIntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::full::call(size, fill_value, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor full(c10::SymIntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::full::call(size, fill_value, dtype, layout, device, pin_memory);
  }
}

// aten::full.out(SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & full_out(at::Tensor & out, at::IntArrayRef size, const at::Scalar & fill_value) {
    return at::_ops::full_out::call(c10::fromIntArrayRefSlow(size), fill_value, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor & full_out(at::Tensor & out, at::IntArrayRef size, const at::Scalar & fill_value) {
    return at::_ops::full_out::call(c10::fromIntArrayRefSlow(size), fill_value, out);
  }
}

// aten::full.out(SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & full_outf(at::IntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
    return at::_ops::full_out::call(c10::fromIntArrayRefSlow(size), fill_value, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor & full_outf(at::IntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
    return at::_ops::full_out::call(c10::fromIntArrayRefSlow(size), fill_value, out);
  }
}

// aten::full.out(SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & full_symint_out(at::Tensor & out, c10::SymIntArrayRef size, const at::Scalar & fill_value) {
    return at::_ops::full_out::call(size, fill_value, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor & full_out(at::Tensor & out, c10::SymIntArrayRef size, const at::Scalar & fill_value) {
    return at::_ops::full_out::call(size, fill_value, out);
  }
}

// aten::full.out(SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & full_symint_outf(c10::SymIntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
    return at::_ops::full_out::call(size, fill_value, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor & full_outf(c10::SymIntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
    return at::_ops::full_out::call(size, fill_value, out);
  }
}

// aten::full.names_out(int[] size, Scalar fill_value, *, Dimname[]? names, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & full_out(at::Tensor & out, at::IntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::DimnameList> names) {
    return at::_ops::full_names_out::call(size, fill_value, names, out);
}
// aten::full.names_out(int[] size, Scalar fill_value, *, Dimname[]? names, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & full_outf(at::IntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::DimnameList> names, at::Tensor & out) {
    return at::_ops::full_names_out::call(size, fill_value, names, out);
}

}
