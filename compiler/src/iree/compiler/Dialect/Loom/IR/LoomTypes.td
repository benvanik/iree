// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_LOOMTYPES_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_LOOMTYPES_TD

include "iree/compiler/Dialect/Loom/IR/LoomBase.td"
include "iree/compiler/Dialect/Loom/IR/LoomInterfaces.td"

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

def Loom_Tile : DialectType<Loom_Dialect,
    CPred<"llvm::isa<::mlir::iree_compiler::IREE::Loom::TileType>($_self)">,
    "!loom.tile", "::mlir::iree_compiler::IREE::Loom::TileType">;

def Loom_Tensor : DialectType<Loom_Dialect,
    CPred<"llvm::isa<::mlir::iree_compiler::IREE::Loom::TensorType>($_self)">,
    "!loom.tensor", "::mlir::iree_compiler::IREE::Loom::TensorType">;

def Loom_Group : DialectType<Loom_Dialect,
    CPred<"llvm::isa<::mlir::iree_compiler::IREE::Loom::GroupType>($_self)">,
    "!loom.group", "::mlir::iree_compiler::IREE::Loom::GroupType">;

//===----------------------------------------------------------------------===//
// !loom.tensor<...>
//===----------------------------------------------------------------------===//

def Loom_TensorType : Loom_TypeDef<"loom.tensor", "Tensor", [
  ShapedTypeInterface
]> {
  let summary = "Shaped global tensor";
  let description = [{
    Represents a global, potentially device-resident tensor with shape and
    element type. May carry optional encoding attributes for physical layout
    hints.

    Relationship to !loom.tile:
    - Tensors are global buffers; tiles are localized views
    - Tiles are created from tensors via loom.tile.view
    - Tensors may have dynamic dimensions

    Examples:
      !loom.tensor<16x16xf32>
      !loom.tensor<?x128xf16>
      !loom.tensor<8x8xf32, #loom.encoding.row>
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    OptionalParameter<"EncodingAttrInterface">:$encoding
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType,
      CArg<"EncodingAttrInterface", "{}">:$encoding
    )>
  ];

  let assemblyFormat = "`<` custom<ShapeAndType>($shape, $elementType) (`,` $encoding^)? `>`";

  let extraClassDeclaration = [{
    // ShapedTypeInterface methods
    bool hasRank() const { return true; }
    int64_t getRank() const { return static_cast<int64_t>(getShape().size()); }
    ArrayRef<int64_t> getShapeDims() const { return getShape(); }
    bool hasStaticShape() const {
      return llvm::none_of(getShape(), ShapedType::isDynamic);
    }
    int64_t getNumDynamicDims() const {
      return llvm::count_if(getShape(), ShapedType::isDynamic);
    }

    // Dimension access helpers
    int64_t getDimSize(unsigned idx) const {
      assert(idx < getRank() && "dimension index out of bounds");
      return getShape()[idx];
    }

    // Returns the index into the list of dynamic dimensions for a given
    // dimension. Asserts if the dimension is not dynamic.
    unsigned getDynamicDimIndex(unsigned idx) const {
      assert(ShapedType::isDynamic(getDimSize(idx)) && "expected dynamic dim");
      unsigned dynamicIdx = 0;
      for (unsigned i = 0; i < idx; ++i) {
        if (ShapedType::isDynamic(getShape()[i])) {
          ++dynamicIdx;
        }
      }
      return dynamicIdx;
    }

    TensorType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                         Type elementType) const {
      return TensorType::get(getContext(),
                             shape.value_or(getShape()),
                             elementType,
                             getEncoding());
    }

    // Encoding access
    bool hasEncoding() const { return getEncoding() != nullptr; }
  }];
}

//===----------------------------------------------------------------------===//
// !loom.tile<...>
//===----------------------------------------------------------------------===//

def Loom_TileType : Loom_TypeDef<"loom.tile", "Tile", [
  ShapedTypeInterface
]> {
  let summary = "Shaped tile with optional encoding";
  let description = [{
    Represents a tile (sub-region of a tensor) with explicit shape and optional encoding.
    The encoding determines physical layout and access pattern when specified.

    Key differences from !loom.tensor:
    - Tiles represent localized computation regions (tensors are global buffers)
    - Tiles are typically smaller, shaped for specific compute patterns
    - Tiles are created from tensors via loom.tile.view

    The encoding attribute is optional but recommended for performance-critical tiles.
    When omitted, the implementation chooses a default layout.

    Examples:
      !loom.tile<16x16xf32>                                    (default encoding)
      !loom.tile<16x16xf32, #loom.encoding.row_major>          (explicit encoding)
      !loom.tile<?x128xf16, #loom.encoding.mma<m=16, n=8, k=16>>
      !loom.tile<8x8xbf16, #loom.encoding.row_major<alignment=16>>
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    OptionalParameter<"EncodingAttrInterface">:$encoding
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType,
      CArg<"EncodingAttrInterface", "{}">:$encoding
    )>
  ];

  let assemblyFormat = "`<` custom<ShapeAndType>($shape, $elementType) (`,` $encoding^)? `>`";

  let extraClassDeclaration = [{
    // ShapedTypeInterface methods
    bool hasRank() const { return true; }
    int64_t getRank() const { return static_cast<int64_t>(getShape().size()); }
    ArrayRef<int64_t> getShapeDims() const { return getShape(); }
    bool hasStaticShape() const {
      return llvm::none_of(getShape(), ShapedType::isDynamic);
    }
    int64_t getNumDynamicDims() const {
      return llvm::count_if(getShape(), ShapedType::isDynamic);
    }

    // Dimension access helpers
    int64_t getDimSize(unsigned idx) const {
      assert(idx < getRank() && "dimension index out of bounds");
      return getShape()[idx];
    }

    // Returns the index into the list of dynamic dimensions for a given
    // dimension. Asserts if the dimension is not dynamic.
    unsigned getDynamicDimIndex(unsigned idx) const {
      assert(ShapedType::isDynamic(getDimSize(idx)) && "expected dynamic dim");
      unsigned dynamicIdx = 0;
      for (unsigned i = 0; i < idx; ++i) {
        if (ShapedType::isDynamic(getShape()[i])) {
          ++dynamicIdx;
        }
      }
      return dynamicIdx;
    }

    TileType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                       Type elementType) const {
      return TileType::get(getContext(),
                           shape.value_or(getShape()),
                           elementType,
                           getEncoding());
    }

    // Encoding access
    bool hasEncoding() const { return getEncoding() != nullptr; }
  }];
}

//===----------------------------------------------------------------------===//
// !loom.group
//===----------------------------------------------------------------------===//

def Loom_GroupType : Loom_TypeDef<"loom.group", "Group"> {
  let summary = "Synchronization group";
  let description = [{
    Represents a synchronization domain (workgroup, subgroup, etc.) for
    ordered execution and memory visibility.

    Groups are used with operations that have explicit ordering requirements,
    such as:
    - loom.tile.fill (optional group for async fills)
    - loom.group.await (synchronization barrier)
    - loom.group.commit (publish results to group)

    Examples:
      !loom.group<workgroup>
      !loom.group<subgroup>
  }];

  let parameters = (ins
    StringRefParameter<>:$scope
  );

  let assemblyFormat = "`<` $scope `>`";
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_LOOMTYPES_TD
