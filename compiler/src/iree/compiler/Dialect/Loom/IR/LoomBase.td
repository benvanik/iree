// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Base definitions for the Loom dialect: common imports, type/attribute
// constraints, and op base class.

#ifndef IREE_COMPILER_DIALECT_LOOM_BASE
#define IREE_COMPILER_DIALECT_LOOM_BASE

include "iree/compiler/Dialect/Loom/IR/LoomDialect.td"
include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Base dialect op/type/attr classes
//===----------------------------------------------------------------------===//

class Loom_Op<string fullMnemonic, list<Trait> traits = []> :
    Op<
      Loom_Dialect,
      !subst(Loom_Dialect.name # ".", "", fullMnemonic),
      traits
    > {
  let hasCustomAssemblyFormat = 1;

  // Optional documentation metadata for rich doc generation.
  Util_OpDocMetadata docMetadata = ?;

  // Errors that this op can emit from its verifier.
  list<Util_ErrorBase> errors = [];
}

class Loom_PureOp<string fullMnemonic, list<Trait> traits = []> :
    Loom_Op<fullMnemonic, !listconcat(traits, [Pure])>;

class Loom_TypeDef<string fullMnemonic, string name, list<Trait> traits = []> :
    TypeDef<Loom_Dialect, name, traits> {
  let mnemonic = !subst(Loom_Dialect.name # ".", "", fullMnemonic);

  // Optional documentation metadata for rich doc generation.
  Util_TypeDocMetadata docMetadata = ?;
}

class Loom_AttrDef<string fullMnemonic, string name, list<Trait> traits = []> :
    AttrDef<Loom_Dialect, name, traits> {
  let mnemonic = !subst(Loom_Dialect.name # ".", "", fullMnemonic);

  // Optional documentation metadata for rich doc generation.
  Util_AttrDocMetadata docMetadata = ?;
}

//===----------------------------------------------------------------------===//
// Enum attributes
//===----------------------------------------------------------------------===//

class Loom_I32EnumAttr<string name, string summary, list<I32EnumAttrCase> cases> :
    I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Loom";
  let genSpecializedAttr = 1;
}

//===----------------------------------------------------------------------===//
// Interface base classes
//===----------------------------------------------------------------------===//

class Loom_OpInterface<string name, list<Interface> baseInterfaces = []> :
    OpInterface<name, baseInterfaces> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Loom";
  Util_InterfaceDocMetadata docMetadata = ?;

  // Errors that interface verifiers can emit.
  list<Util_ErrorBase> errors = [];
}

class Loom_TypeInterface<string name, list<Interface> baseInterfaces = []> :
    TypeInterface<name, baseInterfaces> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Loom";
  Util_InterfaceDocMetadata docMetadata = ?;
}

class Loom_AttrInterface<string name, list<Interface> baseInterfaces = []> :
    AttrInterface<name, baseInterfaces> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Loom";
  Util_InterfaceDocMetadata docMetadata = ?;
}

#endif // IREE_COMPILER_DIALECT_LOOM_BASE
