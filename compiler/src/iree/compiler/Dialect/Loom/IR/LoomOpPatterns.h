// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_LOOMOPPATTERNS_H_
#define IREE_COMPILER_DIALECT_LOOM_IR_LOOMOPPATTERNS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Loom {

//===----------------------------------------------------------------------===//
// CopySubrangeOpInterface Patterns
//===----------------------------------------------------------------------===//

/// Populates interface-based canonicalization patterns for ops implementing
/// CopySubrangeOpInterface. These patterns handle common fold cases:
///
/// - Slice source poison → result poison (ERR_LOOM_FOLD_0002)
/// - Update target poison → result poison (ERR_LOOM_FOLD_0003)
/// - Out-of-bounds access → result poison (ERR_LOOM_FOLD_0004)
/// - Negative offset → result poison (ERR_LOOM_FOLD_0005)
///
/// Call this from LoomDialect::getCanonicalizationPatterns() to enable
/// shared patterns for all slice/update ops.
void populateCopySubrangeOpInterfacePatterns(RewritePatternSet &patterns);

}  // namespace mlir::iree_compiler::IREE::Loom

#endif  // IREE_COMPILER_DIALECT_LOOM_IR_LOOMOPPATTERNS_H_
