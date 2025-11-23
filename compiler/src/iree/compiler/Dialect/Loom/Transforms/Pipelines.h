// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_TRANSFORMS_PIPELINES_H_
#define IREE_COMPILER_DIALECT_LOOM_TRANSFORMS_PIPELINES_H_

#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::Loom {

// Registration function for all Loom pass pipelines.
void registerLoomPipelines();

// Future pipeline builder functions will be declared here, for example:
//
// void buildLoomTilePipeline(::mlir::OpPassManager &pm);
// void buildLoomOptimizationPipeline(::mlir::OpPassManager &pm);

}  // namespace mlir::iree_compiler::IREE::Loom

#endif  // IREE_COMPILER_DIALECT_LOOM_TRANSFORMS_PIPELINES_H_
