// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_IR_UTILCONSTRAINTS_H_
#define IREE_COMPILER_DIALECT_UTIL_IR_UTILCONSTRAINTS_H_

#include <optional>

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::iree_compiler::detail {

//===----------------------------------------------------------------------===//
// Constraint Predicate Helpers
//===----------------------------------------------------------------------===//
//
// These helpers are used in TableGen constraint predicates. They live in
// ::mlir::iree_compiler::detail so that unqualified `detail::` lookup works
// from any dialect's constraint impl namespace (e.g., IREE::Loom::impl)
// via C++ parent namespace lookup.
//
// Usage in TableGen verifyPredicate:
//   let verifyPredicate = [{
//     auto value = detail::castIntegerAttrOr($myAttr, 0);
//     return detail::isPowerOfTwo(value);
//   }];
//
// Available helpers:
//   detail::tryCastIntegerAttr(attr)        - Returns std::optional<int64_t>
//   detail::castIntegerAttr(attr)           - Returns int64_t (asserts)
//   detail::castIntegerAttrOr(attr, def)    - Returns int64_t with default
//   detail::tryCastStringAttr(attr)         - Returns std::optional<StringRef>
//   detail::castStringAttr(attr)            - Returns StringRef (asserts)
//   detail::castStringAttrOr(attr, def)     - Returns StringRef with default
//   detail::isPowerOfTwo(value)             - Checks if int64_t is power of 2
//   detail::isIntegerAttrPowerOfTwo(attr)   - Combined cast + power of 2 check
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Attribute Casting Helpers
//===----------------------------------------------------------------------===//

inline std::optional<int64_t> tryCastIntegerAttr(Attribute attr) {
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
    return intAttr.getInt();
  return std::nullopt;
}

inline int64_t castIntegerAttr(Attribute attr) {
  return cast<IntegerAttr>(attr).getInt();
}

inline int64_t castIntegerAttrOr(Attribute attr, int64_t defaultValue) {
  if (auto value = tryCastIntegerAttr(attr))
    return *value;
  return defaultValue;
}

inline std::optional<StringRef> tryCastStringAttr(Attribute attr) {
  if (auto strAttr = dyn_cast_or_null<StringAttr>(attr))
    return strAttr.getValue();
  return std::nullopt;
}

inline StringRef castStringAttr(Attribute attr) {
  return cast<StringAttr>(attr).getValue();
}

inline StringRef castStringAttrOr(Attribute attr, StringRef defaultValue) {
  if (auto value = tryCastStringAttr(attr))
    return *value;
  return defaultValue;
}

//===----------------------------------------------------------------------===//
// Numeric Property Helpers
//===----------------------------------------------------------------------===//

inline bool isPowerOfTwo(int64_t value) {
  return value > 0 && llvm::isPowerOf2_64(static_cast<uint64_t>(value));
}

inline bool isIntegerAttrPowerOfTwo(Attribute attr) {
  if (auto v = tryCastIntegerAttr(attr))
    return isPowerOfTwo(*v);
  return false;
}

} // namespace mlir::iree_compiler::detail

// clang-format off: must be included after all LLVM/MLIR headers
#include "iree/compiler/Dialect/Util/IR/UtilConstraints.h.inc"
// clang-format on

#endif // IREE_COMPILER_DIALECT_UTIL_IR_UTILCONSTRAINTS_H_
