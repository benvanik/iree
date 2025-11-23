// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilErrors.h"

namespace mlir::iree_compiler::IREE::Util {

// Placeholder for implementation details.
// The error emission system is currently header-only.
// Use the ergonomic API from iree/compiler/Utils/Diagnostics.h:
//   return emitErrorCode<ERR_UTIL_VERIFY_0001>(op, ...);
//
// This file exists for potential future enhancements such as:
// - Configuration loading
// - Statistics collection
// - Error logging to files
// - Integration with IREE's telemetry system

} // namespace mlir::iree_compiler::IREE::Util
