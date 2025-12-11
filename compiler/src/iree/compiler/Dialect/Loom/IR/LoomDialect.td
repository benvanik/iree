// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_LOOMDIALECT
#define IREE_COMPILER_DIALECT_LOOM_IR_LOOMDIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// `loom` dialect
//===----------------------------------------------------------------------===//

def Loom_Dialect : Dialect {
  let name = "loom";
  let cppNamespace = "::mlir::iree_compiler::IREE::Loom";

  let summary = "IREE Loom dialect for pipelined kernel execution";
  let description = [{
    The Loom dialect represents pipelined, multi-stage kernel execution with
    explicit data movement, synchronization, and tile-level operations.

    Key concepts:
    - **Tiles**: Shaped, encoded sub-regions for computation (!loom.tile)
    - **Tensors**: Global, potentially device-resident buffers (!loom.tensor)
    - **Groups**: Synchronization domains for ordered execution (!loom.group)
    - **Encodings**: Physical layout specifications (#loom.encoding.*)

    The dialect supports dynamic shapes with SSA value tracking for symbolic
    reasoning and efficient lowering to HAL operations.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;
  let hasCanonicalizer = 1;
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::iree_compiler::IREE::Util::UtilDialect",
    "::mlir::ub::UBDialect",
  ];

  let extraClassDeclaration = [{
  public:
    // Public helper for external op registration. Wraps protected addOperations.
    template <typename... Args>
    void registerOperations() {
      addOperations<Args...>();
    }

  private:
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_LOOMDIALECT
