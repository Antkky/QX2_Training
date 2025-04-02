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



#include <ATen/ops/rand_ops.h>

namespace at {


// aten::rand.names(SymInt[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand(at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options={}) {
    return at::_ops::rand_names::call(c10::fromIntArrayRefSlow(size), names, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor rand(at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options={}) {
    return at::_ops::rand_names::call(c10::fromIntArrayRefSlow(size), names, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::rand.names(SymInt[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand(at::IntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand_names::call(c10::fromIntArrayRefSlow(size), names, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor rand(at::IntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand_names::call(c10::fromIntArrayRefSlow(size), names, dtype, layout, device, pin_memory);
  }
}

// aten::rand.names(SymInt[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand_symint(c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options={}) {
    return at::_ops::rand_names::call(size, names, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor rand(c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options={}) {
    return at::_ops::rand_names::call(size, names, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::rand.names(SymInt[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand_symint(c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand_names::call(size, names, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor rand(c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand_names::call(size, names, dtype, layout, device, pin_memory);
  }
}

// aten::rand.generator_with_names(SymInt[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand(at::IntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::TensorOptions options={}) {
    return at::_ops::rand_generator_with_names::call(c10::fromIntArrayRefSlow(size), generator, names, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor rand(at::IntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::TensorOptions options={}) {
    return at::_ops::rand_generator_with_names::call(c10::fromIntArrayRefSlow(size), generator, names, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::rand.generator_with_names(SymInt[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand(at::IntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand_generator_with_names::call(c10::fromIntArrayRefSlow(size), generator, names, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor rand(at::IntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand_generator_with_names::call(c10::fromIntArrayRefSlow(size), generator, names, dtype, layout, device, pin_memory);
  }
}

// aten::rand.generator_with_names(SymInt[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand_symint(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::TensorOptions options={}) {
    return at::_ops::rand_generator_with_names::call(size, generator, names, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor rand(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::TensorOptions options={}) {
    return at::_ops::rand_generator_with_names::call(size, generator, names, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::rand.generator_with_names(SymInt[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand_symint(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand_generator_with_names::call(size, generator, names, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor rand(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand_generator_with_names::call(size, generator, names, dtype, layout, device, pin_memory);
  }
}

// aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand(at::IntArrayRef size, at::TensorOptions options={}) {
    return at::_ops::rand::call(c10::fromIntArrayRefSlow(size), c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor rand(at::IntArrayRef size, at::TensorOptions options={}) {
    return at::_ops::rand::call(c10::fromIntArrayRefSlow(size), c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand::call(c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor rand(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand::call(c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory);
  }
}

// aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand_symint(c10::SymIntArrayRef size, at::TensorOptions options={}) {
    return at::_ops::rand::call(size, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor rand(c10::SymIntArrayRef size, at::TensorOptions options={}) {
    return at::_ops::rand::call(size, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand_symint(c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand::call(size, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor rand(c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand::call(size, dtype, layout, device, pin_memory);
  }
}

// aten::rand.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand(at::IntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options={}) {
    return at::_ops::rand_generator::call(c10::fromIntArrayRefSlow(size), generator, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor rand(at::IntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options={}) {
    return at::_ops::rand_generator::call(c10::fromIntArrayRefSlow(size), generator, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::rand.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand(at::IntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand_generator::call(c10::fromIntArrayRefSlow(size), generator, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor rand(at::IntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand_generator::call(c10::fromIntArrayRefSlow(size), generator, dtype, layout, device, pin_memory);
  }
}

// aten::rand.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand_symint(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options={}) {
    return at::_ops::rand_generator::call(size, generator, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor rand(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options={}) {
    return at::_ops::rand_generator::call(size, generator, c10::optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::rand.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor rand_symint(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand_generator::call(size, generator, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor rand(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
    return at::_ops::rand_generator::call(size, generator, dtype, layout, device, pin_memory);
  }
}

// aten::rand.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_out(at::Tensor & out, at::IntArrayRef size) {
    return at::_ops::rand_out::call(c10::fromIntArrayRefSlow(size), out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor & rand_out(at::Tensor & out, at::IntArrayRef size) {
    return at::_ops::rand_out::call(c10::fromIntArrayRefSlow(size), out);
  }
}

// aten::rand.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_outf(at::IntArrayRef size, at::Tensor & out) {
    return at::_ops::rand_out::call(c10::fromIntArrayRefSlow(size), out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor & rand_outf(at::IntArrayRef size, at::Tensor & out) {
    return at::_ops::rand_out::call(c10::fromIntArrayRefSlow(size), out);
  }
}

// aten::rand.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_symint_out(at::Tensor & out, c10::SymIntArrayRef size) {
    return at::_ops::rand_out::call(size, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor & rand_out(at::Tensor & out, c10::SymIntArrayRef size) {
    return at::_ops::rand_out::call(size, out);
  }
}

// aten::rand.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_symint_outf(c10::SymIntArrayRef size, at::Tensor & out) {
    return at::_ops::rand_out::call(size, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor & rand_outf(c10::SymIntArrayRef size, at::Tensor & out) {
    return at::_ops::rand_out::call(size, out);
  }
}

// aten::rand.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_out(at::Tensor & out, at::IntArrayRef size, ::std::optional<at::Generator> generator) {
    return at::_ops::rand_generator_out::call(c10::fromIntArrayRefSlow(size), generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor & rand_out(at::Tensor & out, at::IntArrayRef size, ::std::optional<at::Generator> generator) {
    return at::_ops::rand_generator_out::call(c10::fromIntArrayRefSlow(size), generator, out);
  }
}

// aten::rand.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_outf(at::IntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::rand_generator_out::call(c10::fromIntArrayRefSlow(size), generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor & rand_outf(at::IntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::rand_generator_out::call(c10::fromIntArrayRefSlow(size), generator, out);
  }
}

// aten::rand.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_symint_out(at::Tensor & out, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator) {
    return at::_ops::rand_generator_out::call(size, generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor & rand_out(at::Tensor & out, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator) {
    return at::_ops::rand_generator_out::call(size, generator, out);
  }
}

// aten::rand.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_symint_outf(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::rand_generator_out::call(size, generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor & rand_outf(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::rand_generator_out::call(size, generator, out);
  }
}

// aten::rand.names_out(SymInt[] size, *, Dimname[]? names, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_out(at::Tensor & out, at::IntArrayRef size, ::std::optional<at::DimnameList> names) {
    return at::_ops::rand_names_out::call(c10::fromIntArrayRefSlow(size), names, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor & rand_out(at::Tensor & out, at::IntArrayRef size, ::std::optional<at::DimnameList> names) {
    return at::_ops::rand_names_out::call(c10::fromIntArrayRefSlow(size), names, out);
  }
}

// aten::rand.names_out(SymInt[] size, *, Dimname[]? names, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_outf(at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor & out) {
    return at::_ops::rand_names_out::call(c10::fromIntArrayRefSlow(size), names, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor & rand_outf(at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor & out) {
    return at::_ops::rand_names_out::call(c10::fromIntArrayRefSlow(size), names, out);
  }
}

// aten::rand.names_out(SymInt[] size, *, Dimname[]? names, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_symint_out(at::Tensor & out, c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names) {
    return at::_ops::rand_names_out::call(size, names, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor & rand_out(at::Tensor & out, c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names) {
    return at::_ops::rand_names_out::call(size, names, out);
  }
}

// aten::rand.names_out(SymInt[] size, *, Dimname[]? names, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_symint_outf(c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor & out) {
    return at::_ops::rand_names_out::call(size, names, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor & rand_outf(c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor & out) {
    return at::_ops::rand_names_out::call(size, names, out);
  }
}

// aten::rand.generator_with_names_out(SymInt[] size, *, Generator? generator, Dimname[]? names, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_out(at::Tensor & out, at::IntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names) {
    return at::_ops::rand_generator_with_names_out::call(c10::fromIntArrayRefSlow(size), generator, names, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor & rand_out(at::Tensor & out, at::IntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names) {
    return at::_ops::rand_generator_with_names_out::call(c10::fromIntArrayRefSlow(size), generator, names, out);
  }
}

// aten::rand.generator_with_names_out(SymInt[] size, *, Generator? generator, Dimname[]? names, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_outf(at::IntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::Tensor & out) {
    return at::_ops::rand_generator_with_names_out::call(c10::fromIntArrayRefSlow(size), generator, names, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
  at::Tensor & rand_outf(at::IntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::Tensor & out) {
    return at::_ops::rand_generator_with_names_out::call(c10::fromIntArrayRefSlow(size), generator, names, out);
  }
}

// aten::rand.generator_with_names_out(SymInt[] size, *, Generator? generator, Dimname[]? names, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_symint_out(at::Tensor & out, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names) {
    return at::_ops::rand_generator_with_names_out::call(size, generator, names, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor & rand_out(at::Tensor & out, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names) {
    return at::_ops::rand_generator_with_names_out::call(size, generator, names, out);
  }
}

// aten::rand.generator_with_names_out(SymInt[] size, *, Generator? generator, Dimname[]? names, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & rand_symint_outf(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::Tensor & out) {
    return at::_ops::rand_generator_with_names_out::call(size, generator, names, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same_v<T, c10::SymInt>>>
  at::Tensor & rand_outf(c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::Tensor & out) {
    return at::_ops::rand_generator_with_names_out::call(size, generator, names, out);
  }
}

}
