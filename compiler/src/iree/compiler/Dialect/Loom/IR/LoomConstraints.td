// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reusable constraint helpers for Loom ops (shape, type, region, layout).

#ifndef IREE_COMPILER_DIALECT_LOOM_IR_LOOMCONSTRAINTS_TD
#define IREE_COMPILER_DIALECT_LOOM_IR_LOOMCONSTRAINTS_TD

include "mlir/IR/OpBase.td"
include "iree/compiler/Dialect/Loom/IR/LoomErrors.td"
include "iree/compiler/Dialect/Util/IR/UtilConstraints.td"

//===----------------------------------------------------------------------===//
// Loom Constraint Wrapper
//===----------------------------------------------------------------------===//

// Loom constraint wrapper - adds IREE::Loom namespace prefix for OpTrait lookup.
// Example: def Loom_RankAtLeast<...> : Loom_Constraint<"RankAtLeast_result_operand"> { ... }
// The trait will be generated in ::mlir::OpTrait::IREE::Loom::<name>.
class Loom_Constraint<string name> : Util_ConstraintBase<"IREE::Loom::" # name>;

//===----------------------------------------------------------------------===//
// Property Extractors
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Structured Constraint Property Extractors
//===----------------------------------------------------------------------===//
//
// These extractors work with TileType for structured error reporting.
// They follow the Util_PropertyExtractor pattern for code generation.
//
// Property extractors use $field syntax to reference operation fields.
// iree-tblgen uses ODS introspection to generate correct accessor calls.
//
// Example usage in constraints:
//   class Loom_MyConstraint<string operandName> {
//     let verifyPredicate = [{
//       unsigned rank = }] # Loom_TileExtractor<operandName>.extractRank # [{;
//       return rank > 0;
//     }];
//   }
//
// For field "operand_dims" in a Loom op, this generates:
//   unsigned rank = cast<IREE::Loom::TileType>(operand_dims.getType()).getRank();
//
// See UtilErrorBase.td for complete $field syntax documentation.
//
//===----------------------------------------------------------------------===//

class Loom_TileExtractor<string name> : Util_PropertyExtractor<name> {
  // Store raw field name for iree-tblgen processing.
  string fieldName = name;

  let extractRank = [{
    cast<IREE::Loom::TileType>($}] # fieldName # [{.getType()).getRank()
  }];

  let extractType = [{
    $}] # fieldName # [{.getType()
  }];

  let extractElementType = [{
    cast<IREE::Loom::TileType>($}] # fieldName # [{.getType()).getElementType()
  }];

  let extractShape = [{
    cast<IREE::Loom::TileType>($}] # fieldName # [{.getType()).getShapeDims()
  }];

  // Extract encoding attribute (Loom-specific).
  code extractEncoding = [{
    cast<IREE::Loom::TileType>($}] # fieldName # [{.getType()).getEncoding()
  }];
}

class Loom_TensorExtractor<string name> : Util_PropertyExtractor<name> {
  // Store raw field name for iree-tblgen processing.
  string fieldName = name;

  let extractRank = [{
    cast<IREE::Loom::TensorType>($}] # fieldName # [{.getType()).getRank()
  }];

  let extractType = [{
    $}] # fieldName # [{.getType()
  }];

  let extractElementType = [{
    cast<IREE::Loom::TensorType>($}] # fieldName # [{.getType()).getElementType()
  }];

  let extractShape = [{
    cast<IREE::Loom::TensorType>($}] # fieldName # [{.getType()).getShapeDims()
  }];

  // Extract encoding attribute (Loom-specific).
  code extractEncoding = [{
    cast<IREE::Loom::TensorType>($}] # fieldName # [{.getType()).getEncoding()
  }];
}

//===----------------------------------------------------------------------===//
// Single-Value Property Helpers
//===----------------------------------------------------------------------===//
// Convenience wrappers around Loom_TileExtractor for common properties.
// These follow the same pattern as Util_RankOf, Util_ShapeOf, etc.

// Extract encoding attribute of a TileType operand/result.
// Usage: Loom_TileEncodingOf<"source">.value
class Loom_TileEncodingOf<string operandName> {
  string value = Loom_TileExtractor<operandName>.extractEncoding;
}

// Extract rank of a TileType operand/result.
// Usage: Loom_TileRankOf<"source">.value
class Loom_TileRankOf<string operandName> {
  string value = Loom_TileExtractor<operandName>.extractRank;
}

// Extract shape of a TileType operand/result.
// Usage: Loom_TileShapeOf<"source">.value
class Loom_TileShapeOf<string operandName> {
  string value = Loom_TileExtractor<operandName>.extractShape;
}

// Extract rank of a TensorType operand/result.
// Usage: Loom_TensorRankOf<"source">.value
class Loom_TensorRankOf<string operandName> {
  string value = Loom_TensorExtractor<operandName>.extractRank;
}

//===----------------------------------------------------------------------===//
// Variadic Property Helpers
//===----------------------------------------------------------------------===//
// Build array initializer lists from multiple operands.
// These follow the same pattern as Util_ShapesOfVariadic, etc.

// Build ArrayRef<Attribute> initializer list with TileType encodings.
// Usage: Loom_TileEncodingsOfVariadic<["source", "result"]>.value
// Generates: { enc1, enc2 } which constructs ArrayRef<Attribute> directly.
class Loom_TileEncodingsOfVariadic<list<string> operandNames> {
  string value = Util_ArrayInit<
    !foreach(name, operandNames, Loom_TileEncodingOf<name>.value)
  >.value;
}

//===----------------------------------------------------------------------===//
// Element Type Constraints
//===----------------------------------------------------------------------===//

// Verifies that all specified operands/results have the same element type.
// Works with any ShapedType (TileType, TensorType, or mix).
class Loom_SameElementType<list<string> operands> :
    Loom_Constraint<"SameElementType_" # !interleave(operands, "_")> {
  let error = ERR_LOOM_TYPE_0002;

  let verifyPredicate = [{
    Type firstElementType = }] # Util_ElementTypeOf<!head(operands)>.value # [{;
    }] # !foldl("", !tail(operands), acc, operand, acc # [{
    if (firstElementType != }] # Util_ElementTypeOf<operand>.value # [{) return false;
    }]) # [{
    return true;
  }];

  let errorArgs = (binds
    (arg "operandNames", "ArrayRef<StringRef>",
      Util_NamesOfVariadic<operands>.value),
    (arg "elementTypes", "ArrayRef<Type>",
      Util_ElementTypesOfVariadic<operands>.value)
  );
}

//===----------------------------------------------------------------------===//
// Type Property Constraints
//===----------------------------------------------------------------------===//

// Verifies that all specified operands/results have the same TileType encoding.
// Encodings are compared with null-safety (nullptr encodings match each other).
// Example: Loom_SameEncoding<["source", "result"]>
class Loom_SameEncoding<list<string> operands> :
    Loom_Constraint<"SameEncoding_" # !interleave(operands, "_")> {
  let error = ERR_LOOM_ENCODING_0001;

  let verifyPredicate = [{
    IREE::Loom::EncodingAttrInterface firstEnc = }] # Loom_TileEncodingOf<!head(operands)>.value # [{;
    }] # !foldl("", !tail(operands), acc, operand, acc # [{
    if (firstEnc != }] # Loom_TileEncodingOf<operand>.value # [{) return false;
    }]) # [{
    return true;
  }];

  let errorArgs = (binds
    (arg "operandNames", "ArrayRef<StringRef>",
      Util_NamesOfVariadic<operands>.value),
    (arg "encodings", "ArrayRef<Attribute>",
      Loom_TileEncodingsOfVariadic<operands>.value)
  );
}

// Verifies that an optional operand (if present) is a !loom.group type.
// Example: Loom_OptionalGroupType<"group">
class Loom_OptionalGroupType<string operandName> :
    Loom_Constraint<"OptionalGroupType_" # !subst(".", "_", operandName)> {
  let error = ERR_LOOM_TYPE_0001;

  let verifyPredicate = [{
    return !$}] # operandName # [{ ||
           isa<IREE::Loom::GroupType>($}] # operandName # [{.getType());
  }];

  let errorArgs = (binds
    (arg "operandName", "StringRef", [{"}] # operandName # [{"}]),
    (arg "actualType", "Type", [{$}] # operandName # [{.getType()}])
  );
}

//===----------------------------------------------------------------------===//
// Variadic Operand Constraints
//===----------------------------------------------------------------------===//

// Verifies all TileType operands in a variadic range have the same shape.
// Uses Loom TileType extraction for shape comparison.
// Example: Loom_AllTileShapesMatch<"inputs">
class Loom_AllTileShapesMatch<string operandRange> :
    Loom_Constraint<"AllTileShapesMatch_" # !subst(".", "_", operandRange)> {
  let error = ERR_LOOM_ELEMENTWISE_0002;

  let verifyPredicate = [{
    if ($}] # operandRange # [{.empty()) return true;
    auto firstShape = cast<IREE::Loom::TileType>(
        $}] # operandRange # [{[0].getType()).getShapeDims();
    for (auto operand : $}] # operandRange # [{.drop_front()) {
      auto shape = cast<IREE::Loom::TileType>(operand.getType()).getShapeDims();
      if (firstShape != shape) return false;
    }
    return true;
  }];

  let errorArgs = (binds
    (arg "firstType", "Type", [{$}] # operandRange # [{[0].getType()}]),
    (arg "otherType", "Type", [{[&]() -> Type {
      auto first = cast<IREE::Loom::TileType>($}] # operandRange # [{[0].getType()).getShapeDims();
      for (auto op : $}] # operandRange # [{.drop_front()) {
        auto shape = cast<IREE::Loom::TileType>(op.getType()).getShapeDims();
        if (first != shape) return op.getType();
      }
      return $}] # operandRange # [{[0].getType();
    }()}])
  );
}

//===----------------------------------------------------------------------===//
// Shape Relationship Constraints
//===----------------------------------------------------------------------===//

// Verifies that result TileType shape matches first input TileType shape.
// Example: Loom_TileResultShapeMatchesInputs<"result", "inputs">
class Loom_TileResultShapeMatchesInputs<string result, string inputs> :
    Loom_Constraint<"TileResultShapeMatchesInputs_" # !subst(".", "_", result) #
                    "_" # !subst(".", "_", inputs)> {
  let error = ERR_LOOM_ELEMENTWISE_0003;

  let verifyPredicate = [{
    if ($}] # inputs # [{.empty()) return true;
    auto inputShape = cast<IREE::Loom::TileType>(
        $}] # inputs # [{[0].getType()).getShapeDims();
    auto resultShape = cast<IREE::Loom::TileType>(
        $}] # result # [{.getType()).getShapeDims();
    return inputShape == resultShape;
  }];

  let errorArgs = (binds
    (arg "resultType", "Type", [{$}] # result # [{.getType()}]),
    (arg "inputType", "Type", [{$}] # inputs # [{[0].getType()}])
  );
}

//===----------------------------------------------------------------------===//
// Region Block Argument Constraints
//===----------------------------------------------------------------------===//

// Verifies region block has exactly one argument per input tile.
// Example: Loom_TileBlockArgCount<"body", "inputs">
class Loom_TileBlockArgCount<string region, string inputs> :
    Loom_Constraint<"TileBlockArgCount_" # !subst(".", "_", region) #
                    "_" # !subst(".", "_", inputs)> {
  let error = ERR_LOOM_ELEMENTWISE_0004;

  let verifyPredicate = [{
    return $}] # region # [{.front().getNumArguments() ==
           $}] # inputs # [{.size();
  }];

  let errorArgs = (binds
    (arg "actualCount", "unsigned", [{
      static_cast<unsigned>($}] # region # [{.front().getNumArguments())
    }]),
    (arg "expectedCount", "unsigned", [{
      static_cast<unsigned>($}] # inputs # [{.size())
    }])
  );
}

// Verifies each block argument has the element type of corresponding input tile.
// Example: Loom_TileBlockArgsMatchElements<"body", "inputs">
class Loom_TileBlockArgsMatchElements<string region, string inputs> :
    Loom_Constraint<"TileBlockArgsMatchElements_" # !subst(".", "_", region) #
                    "_" # !subst(".", "_", inputs)> {
  let error = ERR_LOOM_ELEMENTWISE_0005;

  let locals = (binds
    (local "mismatchIdx", "unsigned", [{
      [&]() -> unsigned {
        unsigned idx = 0;
        for (auto [input, arg] : llvm::zip($}] # inputs # [{,
                                           $}] # region # [{.front().getArguments())) {
          auto tileType = cast<IREE::Loom::TileType>(input.getType());
          if (arg.getType() != tileType.getElementType()) return idx;
          ++idx;
        }
        return 0;
      }()
    }])
  );

  let verifyPredicate = [{
    for (auto [input, arg] : llvm::zip($}] # inputs # [{,
                                       $}] # region # [{.front().getArguments())) {
      auto tileType = cast<IREE::Loom::TileType>(input.getType());
      if (arg.getType() != tileType.getElementType()) return false;
    }
    return true;
  }];

  let errorArgs = (binds
    (arg "argIndex", "unsigned", "$mismatchIdx"),
    (arg "argType", "Type", [{
      $}] # region # [{.front().getArgument(mismatchIdx).getType()
    }]),
    (arg "elementType", "Type", [{
      cast<IREE::Loom::TileType>($}] # inputs # [{[mismatchIdx].getType()).getElementType()
    }])
  );
}

//===----------------------------------------------------------------------===//
// Region Terminator Constraints
//===----------------------------------------------------------------------===//

// Verifies terminator has exactly one operand.
// Example: Loom_TileYieldSingleValue<"body">
class Loom_TileYieldSingleValue<string region> :
    Loom_Constraint<"TileYieldSingleValue_" # !subst(".", "_", region)> {
  let error = ERR_LOOM_ELEMENTWISE_0006;

  let verifyPredicate = [{
    Operation *term = $}] # region # [{.front().getTerminator();
    return term && term->getNumOperands() == 1;
  }];

  let errorArgs = (binds
    (arg "actualCount", "unsigned", [{
      static_cast<unsigned>($}] # region # [{.front().getTerminator()->getNumOperands())
    }])
  );
}

// Verifies yielded value type matches result tile element type.
// Example: Loom_TileYieldTypeMatchesResult<"body", "result">
class Loom_TileYieldTypeMatchesResult<string region, string result> :
    Loom_Constraint<"TileYieldTypeMatchesResult_" # !subst(".", "_", region) #
                    "_" # !subst(".", "_", result)> {
  let error = ERR_LOOM_ELEMENTWISE_0007;

  let verifyPredicate = [{
    Operation *term = $}] # region # [{.front().getTerminator();
    if (!term || term->getNumOperands() != 1) return true; // Covered by other constraints
    Type yieldType = term->getOperand(0).getType();
    Type resultElem = cast<IREE::Loom::TileType>(
        $}] # result # [{.getType()).getElementType();
    return yieldType == resultElem;
  }];

  let errorArgs = (binds
    (arg "yieldType", "Type", [{
      $}] # region # [{.front().getTerminator()->getOperand(0).getType()
    }]),
    (arg "resultElementType", "Type", [{
      cast<IREE::Loom::TileType>($}] # result # [{.getType()).getElementType()
    }])
  );
}

//===----------------------------------------------------------------------===//
// Dimension Index Constraints
//===----------------------------------------------------------------------===//

// Verifies that a dimension index (if constant) is within bounds for the source
// tile type. When the index is an SSA value (not constant), passes unconditionally.
// Example: Loom_TileDimIndexInBounds<"source", "index">
class Loom_TileDimIndexInBounds<string sourceOp, string indexOp> :
    Loom_Constraint<"TileDimIndexInBounds_" # !subst(".", "_", sourceOp) #
                    "_" # !subst(".", "_", indexOp)> {
  let error = ERR_LOOM_DIM_0001;

  let locals = (binds
    (local "constIndex", "std::optional<int64_t>", [{
      ::mlir::getConstantIntValue($}] # indexOp # [{)
    }]),
    (local "rank", "int64_t", Loom_TileRankOf<sourceOp>.value)
  );

  let verifyPredicate = [{
    // If index is not a constant, we can't check bounds at compile time - pass.
    if (!constIndex.has_value()) {
      return true;
    }
    // Check bounds: 0 <= index < rank
    int64_t idx = constIndex.value();
    return idx >= 0 && idx < rank;
  }];

  let errorArgs = (binds
    (arg "dimIndex", "int64_t", [{
      ::mlir::getConstantIntValue($}] # indexOp # [{).value_or(-1)
    }]),
    (arg "rank", "int64_t", Loom_TileRankOf<sourceOp>.value)
  );
}

// Verifies that a dimension index (if constant) is within bounds for the source
// tensor type. When the index is an SSA value (not constant), passes unconditionally.
// Example: Loom_TensorDimIndexInBounds<"source", "index">
class Loom_TensorDimIndexInBounds<string sourceOp, string indexOp> :
    Loom_Constraint<"TensorDimIndexInBounds_" # !subst(".", "_", sourceOp) #
                    "_" # !subst(".", "_", indexOp)> {
  let error = ERR_LOOM_DIM_0001;

  let locals = (binds
    (local "constIndex", "std::optional<int64_t>", [{
      ::mlir::getConstantIntValue($}] # indexOp # [{)
    }]),
    (local "rank", "int64_t", Loom_TensorRankOf<sourceOp>.value)
  );

  let verifyPredicate = [{
    // If index is not a constant, we can't check bounds at compile time - pass.
    if (!constIndex.has_value()) {
      return true;
    }
    // Check bounds: 0 <= index < rank
    int64_t idx = constIndex.value();
    return idx >= 0 && idx < rank;
  }];

  let errorArgs = (binds
    (arg "dimIndex", "int64_t", [{
      ::mlir::getConstantIntValue($}] # indexOp # [{).value_or(-1)
    }]),
    (arg "rank", "int64_t", Loom_TensorRankOf<sourceOp>.value)
  );
}

//===----------------------------------------------------------------------===//
// Generic Subrange Constraints
//===----------------------------------------------------------------------===//
// Generic constraints for subrange operations (slice, update, copy).
// These use the new ERR_LOOM_SUBRANGE_* errors with operand name templating,
// making them reusable across all subrange ops.

// Verifies offset count matches tile rank.
// Example: Loom_OffsetCountMatchesTileRank<"source", "static_source_offsets">
class Loom_OffsetCountMatchesTileRank<string tile, string offsets> :
    Loom_Constraint<"OffsetCountMatchesTileRank_" # !subst(".", "_", tile) #
                    "_" # !subst(".", "_", offsets)> {
  let error = ERR_LOOM_SUBRANGE_0001;

  let verifyPredicate = [{
    return static_cast<int64_t>($}] # offsets # [{.size()) ==
           }] # Loom_TileRankOf<tile>.value # [{;
  }];

  let errorArgs = (binds
    (arg "operandName", "StringRef", [{"}] # tile # [{"}]),
    (arg "offsetCount", "unsigned", [{
      static_cast<unsigned>($}] # offsets # [{.size())
    }]),
    (arg "rank", "int64_t", Loom_TileRankOf<tile>.value)
  );
}

// Verifies size count matches tile rank.
// Example: Loom_SizeCountMatchesTileRank<"source", "static_sizes">
class Loom_SizeCountMatchesTileRank<string tile, string sizes> :
    Loom_Constraint<"SizeCountMatchesTileRank_" # !subst(".", "_", tile) #
                    "_" # !subst(".", "_", sizes)> {
  let error = ERR_LOOM_SUBRANGE_0003;

  let verifyPredicate = [{
    return static_cast<int64_t>($}] # sizes # [{.size()) ==
           }] # Loom_TileRankOf<tile>.value # [{;
  }];

  let errorArgs = (binds
    (arg "operandName", "StringRef", [{"}] # tile # [{"}]),
    (arg "sizeCount", "unsigned", [{
      static_cast<unsigned>($}] # sizes # [{.size())
    }]),
    (arg "rank", "int64_t", Loom_TileRankOf<tile>.value)
  );
}

// Verifies two tile operands have the same rank.
// Example: Loom_TileRanksMatch<"source", "target">
class Loom_TileRanksMatch<string tile1, string tile2> :
    Loom_Constraint<"TileRanksMatch_" # !subst(".", "_", tile1) #
                    "_" # !subst(".", "_", tile2)> {
  let error = ERR_LOOM_SUBRANGE_0002;

  let verifyPredicate = [{
    return }] # Loom_TileRankOf<tile1>.value # [{ ==
           }] # Loom_TileRankOf<tile2>.value # [{;
  }];

  let errorArgs = (binds
    (arg "operand1Name", "StringRef", [{"}] # tile1 # [{"}]),
    (arg "rank1", "int64_t", Loom_TileRankOf<tile1>.value),
    (arg "operand2Name", "StringRef", [{"}] # tile2 # [{"}]),
    (arg "rank2", "int64_t", Loom_TileRankOf<tile2>.value)
  );
}

//===----------------------------------------------------------------------===//
// Slice Operation Constraints
//===----------------------------------------------------------------------===//
// For loom.tensor.slice (tensor->tile) and loom.tile.slice (tile->tile).
// NOTE: These use the older ERR_LOOM_SLICE_* errors. For new ops, prefer
// the generic Loom_OffsetCountMatchesTileRank and Loom_TileRanksMatch.

// Verifies offset count matches source tensor rank.
// Example: Loom_TensorSliceOffsetCountMatchesRank<"source", "static_offsets">
class Loom_TensorSliceOffsetCountMatchesRank<string source, string offsets> :
    Loom_Constraint<"TensorSliceOffsetCountMatchesRank_" # !subst(".", "_", source) #
                    "_" # !subst(".", "_", offsets)> {
  let error = ERR_LOOM_SLICE_0001;

  let verifyPredicate = [{
    return static_cast<int64_t>($}] # offsets # [{.size()) ==
           }] # Loom_TensorRankOf<source>.value # [{;
  }];

  let errorArgs = (binds
    (arg "offsetCount", "unsigned", [{
      static_cast<unsigned>($}] # offsets # [{.size())
    }]),
    (arg "sourceRank", "int64_t", Loom_TensorRankOf<source>.value)
  );
}

// Verifies result tile rank matches source tensor rank.
// Example: Loom_TensorSliceRanksMatch<"source", "result">
class Loom_TensorSliceRanksMatch<string source, string result> :
    Loom_Constraint<"TensorSliceRanksMatch_" # !subst(".", "_", source) #
                    "_" # !subst(".", "_", result)> {
  let error = ERR_LOOM_SLICE_0002;

  let verifyPredicate = [{
    return }] # Loom_TileRankOf<result>.value # [{ ==
           }] # Loom_TensorRankOf<source>.value # [{;
  }];

  let errorArgs = (binds
    (arg "resultRank", "int64_t", Loom_TileRankOf<result>.value),
    (arg "sourceRank", "int64_t", Loom_TensorRankOf<source>.value)
  );
}

// Verifies offset count matches source tile rank (for tile->tile slice).
// Example: Loom_TileSliceOffsetCountMatchesRank<"source", "static_offsets">
class Loom_TileSliceOffsetCountMatchesRank<string source, string offsets> :
    Loom_Constraint<"TileSliceOffsetCountMatchesRank_" # !subst(".", "_", source) #
                    "_" # !subst(".", "_", offsets)> {
  let error = ERR_LOOM_SLICE_0001;

  let verifyPredicate = [{
    return static_cast<int64_t>($}] # offsets # [{.size()) ==
           }] # Loom_TileRankOf<source>.value # [{;
  }];

  let errorArgs = (binds
    (arg "offsetCount", "unsigned", [{
      static_cast<unsigned>($}] # offsets # [{.size())
    }]),
    (arg "sourceRank", "int64_t", Loom_TileRankOf<source>.value)
  );
}

// Verifies result tile rank matches source tile rank (for tile->tile slice).
// Example: Loom_TileSliceRanksMatch<"source", "result">
class Loom_TileSliceRanksMatch<string source, string result> :
    Loom_Constraint<"TileSliceRanksMatch_" # !subst(".", "_", source) #
                    "_" # !subst(".", "_", result)> {
  let error = ERR_LOOM_SLICE_0002;

  let verifyPredicate = [{
    return }] # Loom_TileRankOf<result>.value # [{ ==
           }] # Loom_TileRankOf<source>.value # [{;
  }];

  let errorArgs = (binds
    (arg "resultRank", "int64_t", Loom_TileRankOf<result>.value),
    (arg "sourceRank", "int64_t", Loom_TileRankOf<source>.value)
  );
}

//===----------------------------------------------------------------------===//
// Update Operation Constraints
//===----------------------------------------------------------------------===//
// For loom.tensor.update (tile->tensor) and loom.tile.update (tile->tile).

// Verifies offset count matches target tensor rank.
// Example: Loom_TensorUpdateOffsetCountMatchesRank<"target", "static_offsets">
class Loom_TensorUpdateOffsetCountMatchesRank<string target, string offsets> :
    Loom_Constraint<"TensorUpdateOffsetCountMatchesRank_" # !subst(".", "_", target) #
                    "_" # !subst(".", "_", offsets)> {
  let error = ERR_LOOM_UPDATE_0001;

  let verifyPredicate = [{
    return static_cast<int64_t>($}] # offsets # [{.size()) ==
           }] # Loom_TensorRankOf<target>.value # [{;
  }];

  let errorArgs = (binds
    (arg "offsetCount", "unsigned", [{
      static_cast<unsigned>($}] # offsets # [{.size())
    }]),
    (arg "targetRank", "int64_t", Loom_TensorRankOf<target>.value)
  );
}

// Verifies update tile rank matches target tensor rank.
// Example: Loom_TensorUpdateRanksMatch<"update", "target">
class Loom_TensorUpdateRanksMatch<string update, string target> :
    Loom_Constraint<"TensorUpdateRanksMatch_" # !subst(".", "_", update) #
                    "_" # !subst(".", "_", target)> {
  let error = ERR_LOOM_UPDATE_0002;

  let verifyPredicate = [{
    return }] # Loom_TileRankOf<update>.value # [{ ==
           }] # Loom_TensorRankOf<target>.value # [{;
  }];

  let errorArgs = (binds
    (arg "updateRank", "int64_t", Loom_TileRankOf<update>.value),
    (arg "targetRank", "int64_t", Loom_TensorRankOf<target>.value)
  );
}

// Verifies offset count matches target tile rank (for tile->tile update).
// Example: Loom_TileUpdateOffsetCountMatchesRank<"target", "static_offsets">
class Loom_TileUpdateOffsetCountMatchesRank<string target, string offsets> :
    Loom_Constraint<"TileUpdateOffsetCountMatchesRank_" # !subst(".", "_", target) #
                    "_" # !subst(".", "_", offsets)> {
  let error = ERR_LOOM_UPDATE_0001;

  let verifyPredicate = [{
    return static_cast<int64_t>($}] # offsets # [{.size()) ==
           }] # Loom_TileRankOf<target>.value # [{;
  }];

  let errorArgs = (binds
    (arg "offsetCount", "unsigned", [{
      static_cast<unsigned>($}] # offsets # [{.size())
    }]),
    (arg "targetRank", "int64_t", Loom_TileRankOf<target>.value)
  );
}

// Verifies update tile rank matches target tile rank (for tile->tile update).
// Example: Loom_TileUpdateRanksMatch<"update", "target">
class Loom_TileUpdateRanksMatch<string update, string target> :
    Loom_Constraint<"TileUpdateRanksMatch_" # !subst(".", "_", update) #
                    "_" # !subst(".", "_", target)> {
  let error = ERR_LOOM_UPDATE_0002;

  let verifyPredicate = [{
    return }] # Loom_TileRankOf<update>.value # [{ ==
           }] # Loom_TileRankOf<target>.value # [{;
  }];

  let errorArgs = (binds
    (arg "updateRank", "int64_t", Loom_TileRankOf<update>.value),
    (arg "targetRank", "int64_t", Loom_TileRankOf<target>.value)
  );
}

#endif // IREE_COMPILER_DIALECT_LOOM_IR_LOOMCONSTRAINTS_TD
