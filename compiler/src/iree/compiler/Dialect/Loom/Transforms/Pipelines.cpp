// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Loom/Transforms/Pipelines.h"

#include "iree/compiler/Dialect/Loom/IR/LoomDialect.h"
#include "iree/compiler/Dialect/Loom/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::IREE::Loom {

void registerLoomPipelines() {
  // Pipeline registration will be added here when pipelines are created,
  // for example:
  //
  // ::mlir::PassPipelineRegistration<>(
  //     "loom-tile-pipeline",
  //     "Run the Loom tile optimization pipeline",
  //     buildLoomTilePipeline);
}

}  // namespace mlir::iree_compiler::IREE::Loom
