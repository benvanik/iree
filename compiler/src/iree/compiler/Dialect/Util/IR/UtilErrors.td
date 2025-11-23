// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_IR_UTIL_ERRORS_TD
#define IREE_COMPILER_DIALECT_UTIL_IR_UTIL_ERRORS_TD

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilErrorBase.td"

//===----------------------------------------------------------------------===//
// UTIL Domain Errors (Generic Infrastructure)
//===----------------------------------------------------------------------===//
//
// Error IDs use underscores for grep-ability: ERR_UTIL_VERIFY_0001
// This ensures `grep ERR_UTIL_VERIFY_0001` finds all occurrences across:
// - TableGen definitions (this file)
// - Constraint definitions (UtilConstraints.td)
// - Test files (*.mlir with expected-error)
// - Documentation (ERRORS.md)
// - Runtime logs and JSON output
//
//===----------------------------------------------------------------------===//

// Convenience wrapper for Util dialect errors.
// Follows the pattern of Util_Op wrapping Op<Util_Dialect, ...>.
// The dialect prefix ("UTIL") is automatically provided.
// Example: Util_Error<"VERIFY", "0001"> → ERR_UTIL_VERIFY_0001
class Util_Error<string domainName, string codeNum>
  : Util_ErrorBase<Util_Dialect, "UTIL", domainName, codeNum>;

//===----------------------------------------------------------------------===//
// Verification Errors (VERIFY domain)
//===----------------------------------------------------------------------===//

def ERR_UTIL_VERIFY_0001 : Util_Error<"VERIFY", "0001"> {
  let summary = [{Higher-rank operand must have rank >= lower-rank operand.}];
  let description = [{
    Triggered when an operation requires one operand to have rank greater than
    or equal to another operand's rank, but the constraint is violated.
  }];
  let message = [{rank of '{higherName}' must be >= rank of '{lowerName}'}];
  let schema = (ins
    StringRefParameter<>:$higherName,
    StringRefParameter<>:$lowerName,
    "int64_t":$higherRank,
    "int64_t":$lowerRank,
    "Type":$higherType,
    "Type":$lowerType
  );
  let fixHint = [{'{higherName}' (rank {higherRank}) must have rank >= '{lowerName}' (rank {lowerRank})}];
  let example = [{
    // Correct: result has >= operand rank
    %result = op %operand : tensor<4x8xf32> -> tensor<4x8x16xf32>
  }];
}

def ERR_UTIL_VERIFY_0002 : Util_Error<"VERIFY", "0002"> {
  let summary = [{Element types must match across operands.}];
  let description = [{
    Triggered when an operation requires all operands to have the same element
    type, but they differ (e.g., f32 vs f16, or i32 vs i64).
  }];
  let message = [{element types must match across '{operandNames}'}];
  let schema = (ins
    ArrayRefParameter<"StringRef">:$operandNames,
    ArrayRefParameter<"Type">:$elementTypes
  );
  let fixHint = [{Cast operands '{operandNames}' to a common element type (got {elementTypes})}];
  let example = [{
    // Insert cast before operation
    %casted = util.cast %operand : tensor<4xf32> to tensor<4xf16>
    %result = op %casted, %other : tensor<4xf16>, tensor<4xf16>
  }];
}

def ERR_UTIL_VERIFY_0003 : Util_Error<"VERIFY", "0003"> {
  let summary = [{Shapes must match across operands.}];
  let description = [{
    Triggered when an operation requires all operands to have identical shapes
    (same rank and same dimensions), but they differ.
  }];
  let message = [{shapes must match for '{operandNames}'}];
  let schema = (ins
    ArrayRefParameter<"StringRef">:$operandNames,
    ArrayRefParameter<"ArrayRef<int64_t>">:$shapes
  );
  let fixHint = [{Reshape operands '{operandNames}' to matching shapes}];
  let example = [{
    // Use broadcast or reshape as needed
    %reshaped = util.reshape %operand : tensor<4xf32> to tensor<2x2xf32>
  }];
}

def ERR_UTIL_VERIFY_0004 : Util_Error<"VERIFY", "0004"> {
  let message = [{
    rank of '{operandName}' must be exactly {expectedRank}
  }];
  let schema = (ins
    StringRefParameter<>:$operandName,
    "int64_t":$actualRank,
    "int64_t":$expectedRank,
    "Type":$operandType
  );
  let fixHint = [{Reshape '{operandName}' to rank {expectedRank} (got {actualRank})}];
  let example = [{
    // For 2D operations
    %matrix = op %input : tensor<4x8xf32>
  }];
}

def ERR_UTIL_VERIFY_0005 : Util_Error<"VERIFY", "0005"> {
  let message = [{operand '{operandName}' must not alias with '{otherName}'}];
  let schema = (ins
    StringRefParameter<>:$operandName,
    StringRefParameter<>:$otherName
  );
  let fixHint = [{Use different SSA values for '{operandName}' and '{otherName}'}];
  let example = [{
    // Use different values
    %result = op %a, %b : tensor<4xf32>, tensor<4xf32>
  }];
}

//===----------------------------------------------------------------------===//
// Folding/Canonicalization Errors (FOLD domain)
//===----------------------------------------------------------------------===//

def ERR_UTIL_FOLD_0001 : Util_Error<"FOLD", "0001"> {
  let message = [{cannot fold operation with non-constant operands}];
  let schema = (ins
    ArrayRefParameter<"StringRef">:$operandNames
  );
  let fixHint = [{Operands '{operandNames}' must all be constants for folding}];
  let example = [{
    // All operands must be constants
    %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  }];
}

def ERR_UTIL_FOLD_0002 : Util_Error<"FOLD", "0002"> {
  let message = [{folding would produce invalid result}];
  let schema = (ins
    StringRefParameter<>:$reason
  );
  let fixHint = [{Cannot fold: '{reason}'}];
}

//===----------------------------------------------------------------------===//
// Attribute Errors (ATTR domain)
//===----------------------------------------------------------------------===//

def ERR_UTIL_ATTR_0001 : Util_Error<"ATTR", "0001"> {
  let message = [{attribute '{attrName}' must be non-negative}];
  let schema = (ins
    StringRefParameter<>:$attrName,
    "int64_t":$value
  );
  let fixHint = [{Set '{attrName}' to >= 0 (got {value})}];
  let example = [{
    // Use non-negative values
    some_attr = 0 : i64  // OK
    some_attr = 42 : i64 // OK
  }];
}

def ERR_UTIL_ATTR_0002 : Util_Error<"ATTR", "0002"> {
  let message = [{
    attribute '{attrName}' value {value} exceeds maximum {maxValue}
  }];
  let schema = (ins
    StringRefParameter<>:$attrName,
    "int64_t":$value,
    "int64_t":$maxValue
  );
  let fixHint = [{Set '{attrName}' to at most {maxValue} (got {value})}];
  let example = [{
    // Keep within bounds
    alignment = 16 : i64  // If max is 64
  }];
}

//===----------------------------------------------------------------------===//
// Shape Errors (SHAPE domain)
//===----------------------------------------------------------------------===//

def ERR_UTIL_SHAPE_0001 : Util_Error<"SHAPE", "0001"> {
  let message = [{rank of '{higherName}' must be >= rank of '{lowerName}'}];
  let schema = (ins
    StringRefParameter<>:$higherName,
    "int64_t":$higherRank,
    StringRefParameter<>:$lowerName,
    "int64_t":$lowerRank,
    "Type":$higherType,
    "Type":$lowerType
  );
  let fixHint = [{'{higherName}' (rank {higherRank}) must have rank >= '{lowerName}' (rank {lowerRank})}];
  let example = [{
    // Correct: result has >= operand rank
    %result = op %operand : tensor<4x8xf32> -> tensor<4x8x16xf32>
  }];
}

def ERR_UTIL_SHAPE_0006 : Util_Error<"SHAPE", "0006"> {
  let message = [{
    rank of '{operandName}' must be exactly {expectedRank}
  }];
  let schema = (ins
    StringRefParameter<>:$operandName,
    "int64_t":$actualRank,
    "int64_t":$expectedRank,
    "Type":$operandType
  );
  let fixHint = [{Reshape '{operandName}' to rank {expectedRank} (got {actualRank})}];
  let example = [{
    // For 2D operations
    %matrix = op %input : tensor<4x8xf32>
  }];
}

def ERR_UTIL_SHAPE_0002 : Util_Error<"SHAPE", "0002"> {
  let message = [{element types must match across operands}];
  let schema = (ins
    ArrayRefParameter<"StringRef">:$names,
    ArrayRefParameter<"Type">:$types
  );
  let fixHint = [{Cast operands '{names}' to a common element type (got {types})}];
  let example = [{
    // Insert cast before operation
    %casted = util.cast %operand : tensor<4xf32> to tensor<4xf16>
    %result = op %casted, %other : tensor<4xf16>, tensor<4xf16>
  }];
}

def ERR_UTIL_SHAPE_0003 : Util_Error<"SHAPE", "0003"> {
  let message = [{ranks must match across operands}];
  let schema = (ins
    ArrayRefParameter<"StringRef">:$names,
    ArrayRefParameter<"int64_t">:$ranks
  );
  let fixHint = [{Reshape operands '{names}' to matching ranks}];
  let example = [{
    // Use reshape to match ranks
    %reshaped = util.reshape %operand : tensor<8xf32> to tensor<1x8xf32>
    %result = op %reshaped, %other : tensor<1x8xf32>, tensor<1x8xf32>
  }];
}

def ERR_UTIL_SHAPE_0004 : Util_Error<"SHAPE", "0004"> {
  let message = [{shapes must match across operands}];
  let schema = (ins
    ArrayRefParameter<"StringRef">:$names,
    ArrayRefParameter<"ArrayRef<int64_t>">:$shapes
  );
  let fixHint = [{Reshape operands '{names}' to matching shapes}];
  let example = [{
    // Use broadcast or reshape as needed
    %reshaped = util.reshape %operand : tensor<4xf32> to tensor<2x2xf32>
  }];
}

def ERR_UTIL_SHAPE_0005 : Util_Error<"SHAPE", "0005"> {
  let message = [{shapes must be broadcast-compatible}];
  let schema = (ins
    ArrayRefParameter<"StringRef">:$names,
    ArrayRefParameter<"ArrayRef<int64_t>">:$shapes
  );
  let fixHint = [{Operands '{names}' have incompatible shapes for broadcasting}];
  let example = [{
    // Compatible: [4, 1] broadcasts with [1, 8] -> [4, 8]
    %a = ... : tensor<4x1xf32>
    %b = ... : tensor<1x8xf32>
    %result = op %a, %b : tensor<4x1xf32>, tensor<1x8xf32> -> tensor<4x8xf32>
  }];
}

// NOTE: Due to a TableGen quirk, 'let schema' with Util_RangeParameter must
// come FIRST among let statements; otherwise preceding statements are ignored.
def ERR_UTIL_SHAPE_0007 : Util_Error<"SHAPE", "0007"> {
  let schema = (ins
    Util_RangeParameter<"std::string">:$names,
    Util_RangeParameter<"int64_t">:$ranks
  );
  let summary = [{Variadic operand ranks must all match.}];
  let description = [{
    Triggered when a variadic operand list requires all operands to have the
    same tensor rank, but they differ. The error lists each operand name with
    its rank for easy identification of mismatches.
  }];
  let message = [{
    variadic operands have mismatched ranks: {names|ranks}
  }];
  let fixHint = [{Reshape operands to have matching ranks}];
  let example = [{
    // All variadic operands must have the same rank
    %result = op %a, %b, %c : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  }];
}

def ERR_UTIL_SHAPE_0008 : Util_Error<"SHAPE", "0008"> {
  let schema = (ins
    Util_RangeParameter<"std::string">:$names,
    Util_RangeParameter<"ArrayRef<int64_t>">:$shapes
  );
  let summary = [{Variadic operand shapes must all match.}];
  let description = [{
    Triggered when a variadic operand list requires all operands to have
    identical shapes (same rank and dimensions), but they differ. The error
    lists each operand name with its shape for easy identification.
  }];
  let message = [{
    variadic operands have mismatched shapes
  }];
  let fixHint = [{Reshape operands to identical shapes}];
  let example = [{
    // All variadic operands must have identical shapes
    %result = op %a, %b : tensor<4x8xf32>, tensor<4x8xf32>
  }];
}

def ERR_UTIL_SHAPE_0009 : Util_Error<"SHAPE", "0009"> {
  let schema = (ins
    Util_RangeParameter<"std::string">:$names,
    Util_RangeParameter<"Type">:$types
  );
  let summary = [{Variadic operand element types must all match.}];
  let description = [{
    Triggered when a variadic operand list requires all operands to have the
    same element type (e.g., all f32 or all i64), but they differ. The error
    lists each operand name with its element type for easy identification.
  }];
  let message = [{
    variadic operands have mismatched element types: {names|types}
  }];
  let fixHint = [{Cast operands to a common element type}];
  let example = [{
    // All variadic operands must have the same element type
    %result = op %a, %b : tensor<4xf32>, tensor<8xf32>  // OK: both f32
    // Bad: tensor<4xf32>, tensor<4xf16>  // Mismatched: f32 vs f16
  }];
}

def ERR_UTIL_SHAPE_0010 : Util_Error<"SHAPE", "0010"> {
  let summary = [{Dynamic dimension count mismatch.}];
  let description = [{
    Triggered when the number of dynamic dimension values provided does not
    match the number of dynamic dimensions (?) in the shaped type.
  }];
  let message = [{
    {kind} '{type}' has {typeNumDynDims} dynamic dimensions but {providedCount} were provided
  }];
  let schema = (ins
    StringRefParameter<>:$kind,
    "Type":$type,
    "unsigned":$typeNumDynDims,
    "unsigned":$providedCount
  );
  let fixHint = [{Provide exactly {typeNumDynDims} dynamic dimension values for the '{kind}'}];
}

//===----------------------------------------------------------------------===//
// Attribute Errors (ATTR domain - continue sequence)
//===----------------------------------------------------------------------===//

def ERR_UTIL_ATTR_0003 : Util_Error<"ATTR", "0003"> {
  let message = [{attribute '{attrName}' must be a valid permutation array}];
  let schema = (ins
    StringRefParameter<>:$attrName,
    "ArrayAttr":$value,
    "int64_t":$expectedLength
  );
  let fixHint = [{'{attrName}' must contain each value [0, {expectedLength}) exactly once}];
}

def ERR_UTIL_ATTR_0004 : Util_Error<"ATTR", "0004"> {
  let message = [{attribute '{attrName}' must have exactly {expectedLength} elements}];
  let schema = (ins
    StringRefParameter<>:$attrName,
    "int64_t":$actualLength,
    "int64_t":$expectedLength
  );
  let fixHint = [{'{attrName}' needs {expectedLength} elements (got {actualLength})}];
}

def ERR_UTIL_ATTR_0005 : Util_Error<"ATTR", "0005"> {
  let message = [{dynamic operand count must match -1 markers in '{staticAttrName}'}];
  let schema = (ins
    StringRefParameter<>:$staticAttrName,
    StringRefParameter<>:$dynamicOperandName,
    "int64_t":$markerCount,
    "int64_t":$operandCount
  );
  let fixHint = [{'{staticAttrName}' has {markerCount} dynamic markers (-1) but '{dynamicOperandName}' has {operandCount} values}];
}

def ERR_UTIL_ATTR_0006 : Util_Error<"ATTR", "0006"> {
  let message = [{attribute '{attrName}' must be positive (> 0)}];
  let schema = (ins
    StringRefParameter<>:$attrName,
    "int64_t":$value
  );
  let fixHint = [{Set '{attrName}' to > 0 (got {value})}];
  let example = [{\n    // Use positive values\n    some_attr = 1 : i64  // OK\n    some_attr = 42 : i64 // OK\n  }];
}

//===----------------------------------------------------------------------===//
// Verification Errors (VERIFY domain - continue sequence)
//===----------------------------------------------------------------------===//

def ERR_UTIL_VERIFY_0006 : Util_Error<"VERIFY", "0006"> {
  let message = [{operand range '{operandRange}' must be tied to result range '{resultRange}'}];
  let schema = (ins
    StringRefParameter<>:$operandRange,
    StringRefParameter<>:$resultRange
  );
  let fixHint = [{Tie operand range '{operandRange}' to result range '{resultRange}' for in-place semantics}];
}

def ERR_UTIL_VERIFY_0007 : Util_Error<"VERIFY", "0007"> {
  let message = [{iter_args types must match result types}];
  let schema = (ins
    "TypeRange":$iterArgTypes,
    "TypeRange":$resultTypes
  );
  let fixHint = [{Ensure iter_args types match result types}];
}

def ERR_UTIL_VERIFY_0008 : Util_Error<"VERIFY", "0008"> {
  let message = [{memref '{operandName}' must have identity layout}];
  let schema = (ins
    StringRefParameter<>:$operandName,
    "Type":$operandType
  );
  let fixHint = [{Convert '{operandName}' ({operandType}) to use identity layout}];
}

def ERR_UTIL_VERIFY_0009 : Util_Error<"VERIFY", "0009"> {
  let summary = [{Variadic operand range must not be empty.}];
  let description = [{
    Triggered when a variadic operand list requires at least one operand
    but none were provided.
  }];
  let message = [{variadic operand '{operandName}' must not be empty}];
  let schema = (ins
    StringRefParameter<>:$operandName
  );
  let fixHint = [{Provide at least one operand for '{operandName}'}];
}

//===----------------------------------------------------------------------===//
// Region Errors (REGION domain)
//===----------------------------------------------------------------------===//

def ERR_UTIL_REGION_0001 : Util_Error<"REGION", "0001"> {
  let message = [{region '{regionName}' must contain exactly one block}];
  let schema = (ins
    StringRefParameter<>:$regionName,
    "int64_t":$actualBlockCount
  );
  let fixHint = [{Region '{regionName}' has '{actualBlockCount}' blocks, expected 1}];
}

def ERR_UTIL_REGION_0002 : Util_Error<"REGION", "0002"> {
  let message = [{block must have exactly {expectedCount} arguments}];
  let schema = (ins
    "int64_t":$actualCount,
    "int64_t":$expectedCount
  );
  let fixHint = [{Block has '{actualCount}' arguments, expected '{expectedCount}'}];
}

def ERR_UTIL_REGION_0003 : Util_Error<"REGION", "0003"> {
  let message = [{block argument types do not match expected types}];
  let schema = (ins
    ArrayRefParameter<"Type">:$actualTypes,
    ArrayRefParameter<"Type">:$expectedTypes
  );
  let fixHint = [{Block args have types '{actualTypes}', expected '{expectedTypes}'}];
}

#endif // IREE_COMPILER_DIALECT_UTIL_IR_UTIL_ERRORS_TD
