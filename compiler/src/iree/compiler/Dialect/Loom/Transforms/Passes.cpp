// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/Transforms/Passes.h"

#include "iree/compiler/Dialect/Loom/IR/LoomDialect.h"

// Generated pass registration will be included here when passes are added:
//
// #define GEN_PASS_DEF
// #include "iree/compiler/Dialect/Loom/Transforms/Passes.h.inc"

namespace mlir::iree_compiler::IREE::Loom {

void registerPasses() {
  // Pass registration will be added here when passes are created, for example:
  //
  // ::mlir::registerPass([]() { return createLoomTileFusionPass(); });
}

}  // namespace mlir::iree_compiler::IREE::Loom
