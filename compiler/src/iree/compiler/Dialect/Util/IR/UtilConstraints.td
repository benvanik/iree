// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_IR_UTIL_CONSTRAINTS_TD
#define IREE_COMPILER_DIALECT_UTIL_IR_UTIL_CONSTRAINTS_TD

include "iree/compiler/Dialect/Util/IR/UtilErrorBase.td"
include "iree/compiler/Dialect/Util/IR/UtilErrors.td"

//===----------------------------------------------------------------------===//
// IREE Constraint System
//===----------------------------------------------------------------------===//
//
// Design Principles:
// - Type Safety: Catch errors at TableGen time via ODS field introspection
// - Composability: Constraints combine naturally without code duplication
// - Rich Errors: Structured error messages with context (operand names, types)
// - Migration: Coexists with hasVerifier=1 during incremental adoption
//
// Operations can use both constraints and hasVerifier=1.
// Generated verify() calls constraint checks first and then the manual verifier
// so that we can (hopefully) emit the higher-quality errors first and use the
// verifier as a fallback for what we don't have covered.
//
//===----------------------------------------------------------------------===//
// AUTHORING GUIDE: Creating New Constraints
//===----------------------------------------------------------------------===//
//
// Quick Start - Use Existing Constraints When Possible:
// ----------------------------------
// Check if an existing constraint covers your use case before creating new:
//
// Shape/Rank Constraints:
//   Util_RankAtLeast<"result", "operand">     - result rank >= operand rank
//   Util_RankEquals<"operand", 2>             - operand must be 2D
//   Util_SameRank<["lhs", "rhs", "result"]>   - all have same rank
//   Util_SameShape<["lhs", "rhs", "result"]>  - all have same shape
//   Util_ShapeCompatible<["input", "output"]> - broadcast-compatible shapes
//
// Type Constraints:
//   Util_SameElementType<["input", "output"]> - element types match
//
// Attribute Constraints:
//   Util_NonNegativeI64Attr<"offset">         - attr >= 0
//   Util_PositiveI64Attr<"lanes">             - attr > 0
//   Util_MaxValueI64Attr<"alignment", 64>     - attr <= max
//   Util_PermutationArray<"perm", 4>          - valid permutation [0,4)
//   Util_StaticArrayLength<"dims", 4>         - array has N elements
//   Util_DynamicCountMatchesMarkers<"static", "dynamic"> - -1 markers match
//
// Variadic Constraints:
//   Util_AllRanksMatch<"inputs">              - variadic operands same rank
//   Util_AllShapesMatch<"inputs">             - variadic operands same shape
//   Util_AllElementTypesMatch<"inputs">       - variadic operands same type
//
// Region Constraints:
//   Util_SingleBlockRegion<"body">            - region has exactly 1 block
//   Util_BlockArgCount<"block", 3>            - block has N arguments
//
// Other:
//   Util_NoAlias<"input", "output">           - different SSA values
//   Util_IdentityLayout<"memref">             - memref has identity layout
//
// Creating a New Constraint:
// ----------------------------------
// Template:
//   class MyDialect_MyConstraint<string operand> :
//       MyDialect_Constraint<"MyConstraint_" # !subst(".", "_", operand)> {
//     let error = ERR_MYDIALECT_DOMAIN_NNNN;  // Reference existing error
//
//     let locals = (binds
//       (local "value", "int64_t", [{extract code using $}] # operand # [{}])
//     );
//
//     let errorArgs = (binds
//       (arg "operandName", "StringRef", operand),
//       (arg "actualValue", "int64_t", "$value")  // Reference local
//     );
//
//     let verifyPredicate = [{
//       return value > 0;  // Use locals, return bool
//     }];
//   }
//
// Key Concepts:
// ----------------------------------
// 1. $field syntax: Access op fields (operands, results, attributes)
//    - In verifyPredicate: $operand → concreteOp.getOperand()
//    - String concatenation: [{$}] # operandName # [{.getType()}]
//
// 2. locals: Extract values before the predicate runs
//    - (local "name", "C++ type", [{extraction code}])
//    - Available in verifyPredicate and can be referenced in errorArgs
//
// 3. errorArgs: Values passed to error on failure
//    - (arg "name", "type", value) where value can be:
//      - Literal string: "myOperand"
//      - Local reference: "$localName"
//      - Inline code: [{some C++ expression}]
//
// 4. Trait naming: Include operand names in trait class name
//    - "MyConstraint_" # !subst(".", "_", operand)
//    - Dots replaced with underscores for valid C++ identifiers
//
// Property Extractors (for accessing shaped type properties):
// ----------------------------------
//   Util_RankOf<"operand">.value        - rank as int64_t
//   Util_TypeOf<"operand">.value        - full Type
//   Util_ElementTypeOf<"operand">.value - element Type
//   Util_ShapeOf<"operand">.value       - ArrayRef<int64_t> shape
//
// Variadic Helpers (for error arguments):
// ----------------------------------
//   Util_NamesOfVariadic<["a", "b"]>.value         - { "a", "b" }
//   Util_RanksOfVariadic<["a", "b"]>.value         - { rank_a, rank_b }
//   Util_ShapesOfVariadic<["a", "b"]>.value        - { shape_a, shape_b }
//   Util_ElementTypesOfVariadic<["a", "b"]>.value  - { type_a, type_b }
//
// Common Patterns:
// ----------------------------------
// Iterate over list in verifyPredicate using !foldl:
//   let verifyPredicate = [{
//     }] # !foldl("", values, acc, val, acc # [{
//     if (firstRank != }] # Util_RankOf<val>.value # [{) return false;
//     }]) # [{
//     return true;
//   }];
//
// Handle optional attributes:
//   let verifyPredicate = [{
//     if (auto attr = $}] # attrName # [{) {
//       // Attribute present, validate it
//       return validate(attr);
//     }
//     return true;  // Missing optional attr is OK
//   }];
//
// Use helper functions from detail namespace:
//   detail::castIntegerAttr(attr)        - Cast attr to int64_t
//   detail::castIntegerAttrOr(attr, def) - With default value
//   detail::tryCastIntegerAttr(attr)     - Returns std::optional
//   detail::isPowerOfTwo(value)          - Check power of 2
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Generic Property Extractors
//===----------------------------------------------------------------------===//
//
// These extractors work with standard MLIR tensor types.
// Dialect-specific types (like loom.tile) should define their own extractors.
//
//===----------------------------------------------------------------------===//

class Util_GenericExtractor<string name> : Util_PropertyExtractor<name> {
  // Store raw field name for iree-tblgen processing.
  string fieldName = name;

  let extractRank = [{
    ::llvm::cast<::mlir::RankedTensorType>($}] # fieldName # [{.getType()).getRank()
  }];

  let extractType = [{
    $}] # fieldName # [{.getType()
  }];

  let extractElementType = [{
    ::llvm::cast<::mlir::RankedTensorType>($}] # fieldName # [{.getType()).getElementType()
  }];

  let extractShape = [{
    ::llvm::cast<::mlir::RankedTensorType>($}] # fieldName # [{.getType()).getShape()
  }];
}

//===----------------------------------------------------------------------===//
// Shorthand Extractors
//===----------------------------------------------------------------------===//

// Extract rank of a shaped operand/result.
// Usage: Util_RankOf<"operand">
class Util_RankOf<string operandName> {
  string value = Util_ShapedTypeExtractor<operandName>.extractRank;
}

// Extract type of an operand/result.
// Usage: Util_TypeOf<"operand">
class Util_TypeOf<string operandName> {
  string value = Util_ShapedTypeExtractor<operandName>.extractType;
}

// Extract element type of a shaped operand/result.
// Usage: Util_ElementTypeOf<"operand">
class Util_ElementTypeOf<string operandName> {
  string value = Util_ShapedTypeExtractor<operandName>.extractElementType;
}

// Extract shape of a shaped operand/result.
// Usage: Util_ShapeOf<"operand">
class Util_ShapeOf<string operandName> {
  string value = Util_ShapedTypeExtractor<operandName>.extractShape;
}

//===----------------------------------------------------------------------===//
// Array Initializer Helper
//===----------------------------------------------------------------------===//

// Build a C++ array initializer list from a list of value strings.
// Usage: Util_ArrayInit<["val1", "val2"]>.value -> "{ val1, val2 }"
// This eliminates boilerplate in variadic property helpers.
class Util_ArrayInit<list<string> values> {
  string value = "{ " # !interleave(values, ", ") # " }";
}

//===----------------------------------------------------------------------===//
// Variadic Property Helpers
//===----------------------------------------------------------------------===//

// Build ArrayRef<StringRef> initializer list with field names.
// Usage: Util_NamesOfVariadic<["lhs", "rhs"]>.value
// Generates: { "lhs", "rhs" } which constructs ArrayRef<StringRef> directly.
class Util_NamesOfVariadic<list<string> operandNames> {
  string value = Util_ArrayInit<
    !foreach(name, operandNames, "\"" # name # "\"")
  >.value;
}

// Build ArrayRef<int64_t> initializer list with ranks.
// Usage: Util_RanksOfVariadic<["lhs", "rhs"]>.value
// Generates: { rank1, rank2 } which constructs ArrayRef<int64_t> directly.
class Util_RanksOfVariadic<list<string> operandNames> {
  string value = Util_ArrayInit<
    !foreach(name, operandNames, Util_RankOf<name>.value)
  >.value;
}

// Build ArrayRef<ArrayRef<int64_t>> initializer list with shapes.
// Usage: Util_ShapesOfVariadic<["lhs", "rhs"]>.value
// Generates: { shape1, shape2 } which constructs ArrayRef<ArrayRef<int64_t>> directly.
class Util_ShapesOfVariadic<list<string> operandNames> {
  string value = Util_ArrayInit<
    !foreach(name, operandNames, Util_ShapeOf<name>.value)
  >.value;
}

// Build ArrayRef<Type> initializer list with element types.
// Usage: Util_ElementTypesOfVariadic<["lhs", "rhs"]>.value
// Generates: { type1, type2 } which constructs ArrayRef<Type> directly.
class Util_ElementTypesOfVariadic<list<string> operandNames> {
  string value = Util_ArrayInit<
    !foreach(name, operandNames, Util_ElementTypeOf<name>.value)
  >.value;
}

//===----------------------------------------------------------------------===//
// Range Mapping Helpers for Variadic Constraints
//===----------------------------------------------------------------------===//
//
// These helpers produce lazy iterators (function_ref-based) for use with
// Util_RangeParameter in error schemas. They avoid materializing temporary
// containers when iterating over ValueRange properties.
//
//===----------------------------------------------------------------------===//

// Streams names as "<prefix>[idx]" for each element of a ValueRange.
class Util_EnumerateNames<code range, string prefix> {
  code value = [{
    [=](::llvm::function_ref<void(std::string)> sink) {
      size_t idx = 0;
      for (auto v : }] # range # [{) {
        (void)v;
        sink((llvm::Twine("}] # prefix # [{[") + llvm::Twine(idx++) + "]").str());
      }
    }
  }];
}

// Streams ranks for a ValueRange of shaped values.
class Util_MapRanks<code range> {
  code value = [{
    [=](::llvm::function_ref<void(int64_t)> sink) {
      for (auto v : }] # range # [{)
        sink(cast<ShapedType>(v.getType()).getRank());
    }
  }];
}

// Streams shapes (ArrayRef<int64_t>) for a ValueRange of shaped values.
class Util_MapShapes<code range> {
  code value = [{
    [=](::llvm::function_ref<void(::llvm::ArrayRef<int64_t>)> sink) {
      for (auto v : }] # range # [{)
        sink(cast<ShapedType>(v.getType()).getShape());
    }
  }];
}

// Streams element types for a ValueRange of shaped values.
class Util_MapElementTypes<code range> {
  code value = [{
    [=](::llvm::function_ref<void(Type)> sink) {
      for (auto v : }] # range # [{)
        sink(cast<ShapedType>(v.getType()).getElementType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// Shape and Rank Constraints
//===----------------------------------------------------------------------===//

// Verifies that the rank of 'higher' >= rank of 'lower'.
// Works with any ShapedType (tensors, memrefs, tile types).
// Example: Util_RankAtLeast<"result", "operand">
class Util_RankAtLeast<string higher, string lower> :
    Util_Constraint<"RankAtLeast_" # !subst(".", "_", higher) # "_" #
                        !subst(".", "_", lower)> {
  let error = ERR_UTIL_SHAPE_0001;

  let locals = (binds
    (local "higherRank", "int64_t", Util_RankOf<higher>.value),
    (local "lowerRank", "int64_t", Util_RankOf<lower>.value)
  );

  let errorArgs = (binds
    (arg "higherName", "StringRef", higher),
    (arg "higherRank", "int64_t", "$higherRank"),
    (arg "lowerName", "StringRef", lower),
    (arg "lowerRank", "int64_t", "$lowerRank"),
    (arg "higherType", "Type", Util_TypeOf<higher>.value),
    (arg "lowerType", "Type", Util_TypeOf<lower>.value)
  );

  let verifyPredicate = [{
    return higherRank >= lowerRank;
  }];
}

// Verifies that an operand/result has exactly the specified rank.
// Works with any ShapedType (tensors, memrefs, tile types).
// Example: Util_RankEquals<"operand", 2> for matrices
class Util_RankEquals<string operand, int rank> :
    Util_Constraint<"RankEquals_" # !subst(".", "_", operand) # "_" # !cast<string>(rank)> {
  let error = ERR_UTIL_SHAPE_0006;

  let locals = (binds
    (local "actualRank", "int64_t", Util_RankOf<operand>.value)
  );

  let errorArgs = (binds
    (arg "operandName", "StringRef", operand),
    (arg "actualRank", "int64_t", "$actualRank"),
    (arg "expectedRank", "int64_t", rank),
    (arg "operandType", "Type", Util_TypeOf<operand>.value)
  );

  let verifyPredicate = [{
    return actualRank == }] # !cast<string>(rank) # [{;
  }];
}

// Verifies that ranks match across multiple operands/results.
// Works with any ShapedType (tensors, memrefs, tile types).
// Example: Util_SameRank<["lhs", "rhs", "result"]>
class Util_SameRank<list<string> values> :
    Util_Constraint<"SameRank_" # !interleave(values, "_")> {
  let error = ERR_UTIL_SHAPE_0003;

  let locals = (binds
    (local "firstRank", "int64_t", Util_RankOf<!head(values)>.value)
  );

  let errorArgs = (binds
    (arg "names", "ArrayRef<StringRef>", Util_NamesOfVariadic<values>.value),
    (arg "ranks", "ArrayRef<int64_t>", Util_RanksOfVariadic<values>.value)
  );

  let verifyPredicate = [{
    }] # !foldl("", !tail(values), acc, val, acc # [{
    if (firstRank != }] # Util_RankOf<val>.value # [{)
      return false;
    }]) # [{
    return true;
  }];
}

// Verifies that all shapes match across multiple operands/results.
// Works with any ShapedType (tensors, memrefs, tile types).
// Example: Util_SameShape<["lhs", "rhs", "result"]>
class Util_SameShape<list<string> values> :
    Util_Constraint<"SameShape_" # !interleave(values, "_")> {
  let error = ERR_UTIL_SHAPE_0004;

  let locals = (binds
    (local "firstShape", "ArrayRef<int64_t>", Util_ShapeOf<!head(values)>.value)
  );

  let errorArgs = (binds
    (arg "names", "ArrayRef<StringRef>", Util_NamesOfVariadic<values>.value),
    (arg "shapes", "ArrayRef<ArrayRef<int64_t>>", Util_ShapesOfVariadic<values>.value)
  );

  let verifyPredicate = [{
    }] # !foldl("", !tail(values), acc, val, acc # [{
    auto }] # val # [{Shape = }] # Util_ShapeOf<val>.value # [{;
    if (firstShape != }] # val # [{Shape)
      return false;
    }]) # [{
    return true;
  }];
}

// Verifies that shapes are broadcast-compatible.
// Broadcasting: dimensions must either match or one must be 1.
// Operands are aligned from trailing dimension.
// Works with any ShapedType (tensors, memrefs, tile types).
// Example: Util_ShapeCompatible<["input", "result"]>
class Util_ShapeCompatible<list<string> values> :
    Util_Constraint<"ShapeCompatible_" # !interleave(values, "_")> {
  let error = ERR_UTIL_SHAPE_0005;

  let errorArgs = (binds
    (arg "names", "ArrayRef<StringRef>", Util_NamesOfVariadic<values>.value),
    (arg "shapes", "ArrayRef<ArrayRef<int64_t>>", Util_ShapesOfVariadic<values>.value)
  );

  let verifyPredicate = [{
    SmallVector<ArrayRef<int64_t>> shapes;
    }] # !foldl("", values, acc, val, acc # [{
    shapes.push_back(}] # Util_ShapeOf<val>.value # [{);
    }]) # [{

    // Check broadcast compatibility for all pairs.
    for (size_t i = 0; i < shapes.size(); ++i) {
      for (size_t j = i + 1; j < shapes.size(); ++j) {
        auto shape1 = shapes[i];
        auto shape2 = shapes[j];
        int64_t rank1 = shape1.size();
        int64_t rank2 = shape2.size();
        int64_t maxRank = std::max(rank1, rank2);
        // Check dimensions from trailing edge.
        // Dimensions must match or one must be 1 (or dynamic -1).
        for (int64_t k = 0; k < maxRank; ++k) {
          int64_t dim1 = (k < rank1) ? shape1[rank1 - 1 - k] : 1;
          int64_t dim2 = (k < rank2) ? shape2[rank2 - 1 - k] : 1;
          if (dim1 != dim2 && dim1 != 1 && dim2 != 1 &&
              dim1 != ShapedType::kDynamic && dim2 != ShapedType::kDynamic) {
            return false;
          }
        }
      }
    }
    return true;
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

// Verifies that element types match across multiple operands/results.
// Works with any ShapedType (tensors, memrefs, tile types).
// Example: Util_SameElementType<["operand", "result"]>
class Util_SameElementType<list<string> values> :
    Util_Constraint<"SameElementType_" # !interleave(values, "_")> {
  let error = ERR_UTIL_SHAPE_0002;

  let locals = (binds
    (local "firstType", "Type", Util_ElementTypeOf<!head(values)>.value)
  );

  let errorArgs = (binds
    (arg "names", "ArrayRef<StringRef>", Util_NamesOfVariadic<values>.value),
    (arg "types", "ArrayRef<Type>", Util_ElementTypesOfVariadic<values>.value)
  );

  let verifyPredicate = [{
    }] # !foldl("", !tail(values), acc, val, acc # [{
    if (firstType != }] # Util_ElementTypeOf<val>.value # [{)
      return false;
    }]) # [{
    return true;
  }];
}

//===----------------------------------------------------------------------===//
// SSA Value Constraints
//===----------------------------------------------------------------------===//

// Verifies that two operands do not alias (are different SSA values).
// Example: Util_NoAlias<"input", "output">
class Util_NoAlias<string first, string second> :
    Util_Constraint<"NoAlias_" # !subst(".", "_", first) # "_" #
                  !subst(".", "_", second)> {
  let error = ERR_UTIL_VERIFY_0005;

  let errorArgs = (binds
    (arg "operandName", "StringRef", first),
    (arg "otherName", "StringRef", second)
  );

  let verifyPredicate = [{
    return $}] # first # [{ != $}] # second # [{;
  }];
}

//===----------------------------------------------------------------------===//
// Attribute Constraints
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Attribute Constraint Base Classes (Parameterized by Storage Type)
//===----------------------------------------------------------------------===//

// Base: Non-negative integer attribute constraint.
// Parameterized by storage type for extensibility (int64_t, uint8_t, etc.).
// Example: Util_AttrNonNegative<"offset", "int64_t">
class Util_AttrNonNegative<string attrName, string storageType> :
    Util_Constraint<"AttrNonNegative_" # !subst(".", "_", attrName)> {
  let error = ERR_UTIL_ATTR_0001;

  let locals = (binds
    (local "value", storageType, [{
      static_cast<}] # storageType # [{>(
          detail::castIntegerAttrOr($}] # attrName # [{, 0))
    }])
  );

  let errorArgs = (binds
    (arg "attrName", "StringRef", attrName),
    (arg "value", "int64_t", [{static_cast<int64_t>(}] # "$value" # [{)}])
  );

  let verifyPredicate = [{
    if (auto attrValue = $}] # attrName # [{) {
      if (auto intValue = detail::tryCastIntegerAttr(attrValue))
        return *intValue >= 0;
      return false;
    }
    return true;  // Optional attribute, missing is OK.
  }];
}

// Base: Maximum value integer attribute constraint.
// Parameterized by storage type for extensibility (int64_t, uint8_t, etc.).
// Example: Util_AttrMaxValue<"alignment", 64, "int64_t">
class Util_AttrMaxValue<string attrName, int maxValue, string storageType> :
    Util_Constraint<"AttrMaxValue_" # !subst(".", "_", attrName) # "_" #
                  !cast<string>(maxValue)> {
  let error = ERR_UTIL_ATTR_0002;

  let locals = (binds
    (local "value", storageType, [{
      static_cast<}] # storageType # [{>(
          detail::castIntegerAttrOr($}] # attrName # [{, 0))
    }])
  );

  let errorArgs = (binds
    (arg "attrName", "StringRef", attrName),
    (arg "value", "int64_t", [{static_cast<int64_t>(}] # "$value" # [{)}]),
    (arg "maxValue", "int64_t", maxValue)
  );

  let verifyPredicate = [{
    if (auto attrValue = $}] # attrName # [{) {
      if (auto intValue = detail::tryCastIntegerAttr(attrValue))
        return *intValue <= }] # !cast<string>(maxValue) # [{;
      return false;
    }
    return true;  // Optional attribute, missing is OK.
  }];
}

//===----------------------------------------------------------------------===//
// Type-Specific Attribute Constraint Wrappers
//===----------------------------------------------------------------------===//
//
// Convenience wrappers for common integer types.
// Follow constraint-first naming: Util_[Constraint][Type]Attr
//
//===----------------------------------------------------------------------===//

// I64 (int64_t) attribute constraints - most common case.
class Util_NonNegativeI64Attr<string attrName> :
    Util_AttrNonNegative<attrName, "int64_t">;

class Util_MaxValueI64Attr<string attrName, int maxValue> :
    Util_AttrMaxValue<attrName, maxValue, "int64_t">;

// Index attribute constraints (Index type is int64_t in MLIR).
class Util_NonNegativeIndexAttr<string attrName> :
    Util_AttrNonNegative<attrName, "int64_t">;

class Util_MaxValueIndexAttr<string attrName, int maxValue> :
    Util_AttrMaxValue<attrName, maxValue, "int64_t">;

//===----------------------------------------------------------------------===//
// Variadic Operand Constraints
//===----------------------------------------------------------------------===//

// Verifies that a variadic operand range is not empty.
// Example: Util_VariadicNotEmpty<"inputs">
class Util_VariadicNotEmpty<string operandRange> :
    Util_Constraint<"VariadicNotEmpty_" # !subst(".", "_", operandRange)> {
  let error = ERR_UTIL_VERIFY_0009;

  let verifyPredicate = [{
    return !$}] # operandRange # [{.empty();
  }];

  let errorArgs = (binds
    (arg "operandName", "StringRef", [{"}] # operandRange # [{"}])
  );
}

// Verifies that all operands in a variadic range have the same rank.
// Example: Util_AllRanksMatch<"inputs"> where inputs is Variadic<AnyTensor>
class Util_AllRanksMatch<string operandRange> :
    Util_Constraint<"AllRanksMatch_" # !subst(".", "_", operandRange)> {
  let error = ERR_UTIL_SHAPE_0007;

  let verifyPredicate = [{
    int64_t firstRank = -1;
    for (auto operand : $}] # operandRange # [{) {
      int64_t rank = cast<ShapedType>(operand.getType()).getRank();
      if (firstRank < 0) firstRank = rank;
      else if (rank != firstRank) return false;
    }
    return true;
  }];

  let errorArgs = (binds
    (arg "names", Util_RangeParameter<"std::string">,
      Util_EnumerateNames<[{$}] # operandRange # [{}], "}] # operandRange # [{">.value),
    (arg "ranks", Util_RangeParameter<"int64_t">,
      Util_MapRanks<[{$}] # operandRange # [{}]>.value)
  );
}

// Verifies that all operands in a variadic range have the same shape.
// Example: Util_AllShapesMatch<"inputs"> where inputs is Variadic<AnyTensor>
class Util_AllShapesMatch<string operandRange> :
    Util_Constraint<"AllShapesMatch_" # !subst(".", "_", operandRange)> {
  let error = ERR_UTIL_SHAPE_0008;

  let verifyPredicate = [{
    bool first = true;
    SmallVector<int64_t> firstShape;
    for (auto operand : $}] # operandRange # [{) {
      auto shape = cast<ShapedType>(operand.getType()).getShape();
      if (first) {
        firstShape.assign(shape.begin(), shape.end());
        first = false;
        continue;
      }
      if (!std::equal(shape.begin(), shape.end(), firstShape.begin(),
                      firstShape.end(), [](int64_t a, int64_t b) {
                        return ShapedType::isDynamic(a) ||
                               ShapedType::isDynamic(b) || a == b;
                      }))
        return false;
    }
    return true;
  }];

  let errorArgs = (binds
    (arg "names", Util_RangeParameter<"std::string">,
      Util_EnumerateNames<[{$}] # operandRange # [{}], "}] # operandRange # [{">.value),
    (arg "shapes", Util_RangeParameter<"ArrayRef<int64_t>">,
      Util_MapShapes<[{$}] # operandRange # [{}]>.value)
  );
}

// Verifies that all operands in a variadic range have the same element type.
// Example: Util_AllElementTypesMatch<"inputs"> where inputs is Variadic<AnyTensor>
class Util_AllElementTypesMatch<string operandRange> :
    Util_Constraint<"AllElementTypesMatch_" # !subst(".", "_", operandRange)> {
  let error = ERR_UTIL_SHAPE_0009;

  let verifyPredicate = [{
    Type firstElemType;
    bool first = true;
    for (auto operand : $}] # operandRange # [{) {
      Type elemType = cast<ShapedType>(operand.getType()).getElementType();
      if (first) {
        firstElemType = elemType;
        first = false;
      } else if (elemType != firstElemType) {
        return false;
      }
    }
    return true;
  }];

  let errorArgs = (binds
    (arg "names", Util_RangeParameter<"std::string">,
      Util_EnumerateNames<[{$}] # operandRange # [{}], "}] # operandRange # [{">.value),
    (arg "types", Util_RangeParameter<"Type">,
      Util_MapElementTypes<[{$}] # operandRange # [{}]>.value)
  );
}

//===----------------------------------------------------------------------===//
// Array Attribute Constraints
//===----------------------------------------------------------------------===//

// Verifies that an array attribute is a valid permutation.
// A valid permutation contains each value in [0, length) exactly once.
// Example: Util_PermutationArray<"permutation", 4> for [0,2,1,3]
class Util_PermutationArray<string attrName, int expectedLength> :
    Util_Constraint<"PermutationArray_" # !subst(".", "_", attrName)> {
  let error = ERR_UTIL_ATTR_0003;

  let locals = (binds
    (local "arrayAttr", "ArrayAttr", [{$}] # attrName # [{}])
  );

  let errorArgs = (binds
    (arg "attrName", "StringRef", attrName),
    (arg "value", "ArrayAttr", "$arrayAttr"),
    (arg "expectedLength", "int64_t", expectedLength)
  );

  let verifyPredicate = [{
    if (!arrayAttr || arrayAttr.size() != }] # !cast<string>(expectedLength) # [{) return false;
    llvm::SmallDenseSet<int64_t> seen;
    for (auto elem : arrayAttr) {
      auto intAttr = dyn_cast<IntegerAttr>(elem);
      if (!intAttr) return false;
      int64_t val = intAttr.getInt();
      if (val < 0 || val >= }] # !cast<string>(expectedLength) # [{ || seen.contains(val)) {
        return false;
      }
      seen.insert(val);
    }
    return true;
  }];
}

//===----------------------------------------------------------------------===//
// Operand/Result Relationship Constraints
//===----------------------------------------------------------------------===//

// Verifies that an operand range is tied to a result range for in-place updates.
// Example: Util_TiedOperandRange<"inputs", "outputs">
class Util_TiedOperandRange<string operandRange, string resultRange> :
    Util_Constraint<"TiedOperandRange_" # !subst(".", "_", operandRange)> {
  let error = ERR_UTIL_VERIFY_0006;

  let errorArgs = (binds
    (arg "operandRange", "StringRef", operandRange),
    (arg "resultRange", "StringRef", resultRange)
  );

  let verifyPredicate = [{
    return $}] # operandRange # [{.size() == $}] # resultRange # [{.size();
  }];
}

// Verifies that loop iter_args match loop results (standard loop invariant).
// Example: Util_IterArgsMatchResults<"iter_args", "results">
class Util_IterArgsMatchResults<string iterArgsName, string resultsName> :
    Util_Constraint<"IterArgsMatchResults_" # !subst(".", "_", iterArgsName)> {
  let error = ERR_UTIL_VERIFY_0007;

  let verifyPredicate = [{
    return TypeRange($}] # iterArgsName # [{) == TypeRange($}] # resultsName # [{);
  }];

  let errorArgs = (binds
    (arg "iterArgTypes", "TypeRange", [{TypeRange($}] # iterArgsName # [{)}]),
    (arg "resultTypes", "TypeRange", [{TypeRange($}] # resultsName # [{)}])
  );
}

//===----------------------------------------------------------------------===//
// Region and Block Constraints
//===----------------------------------------------------------------------===//

// Verifies that a region contains exactly one block.
// Example: Util_SingleBlockRegion<"body">
class Util_SingleBlockRegion<string regionName> :
    Util_Constraint<"SingleBlockRegion_" # !subst(".", "_", regionName)> {
  let error = ERR_UTIL_REGION_0001;

  let locals = (binds
    (local "blockCount", "int64_t", [{
      static_cast<int64_t>($}] # regionName # [{.getBlocks().size())
    }])
  );

  let errorArgs = (binds
    (arg "regionName", "StringRef", regionName),
    (arg "actualBlockCount", "int64_t", "$blockCount")
  );

  let verifyPredicate = [{
    return blockCount == 1;
  }];
}

// Verifies that a block has exactly the expected number of arguments.
// Example: Util_BlockArgCount<"body_block", 3>
class Util_BlockArgCount<string blockName, int expectedCount> :
    Util_Constraint<"BlockArgCount_" # !subst(".", "_", blockName)> {
  let error = ERR_UTIL_REGION_0002;

  let locals = (binds
    (local "actualCount", "int64_t", [{
      static_cast<int64_t>($}] # blockName # [{.getNumArguments())
    }])
  );

  let errorArgs = (binds
    (arg "actualCount", "int64_t", "$actualCount"),
    (arg "expectedCount", "int64_t", expectedCount)
  );

  let verifyPredicate = [{
    return actualCount == }] # !cast<string>(expectedCount) # [{;
  }];
}

// Verifies that block argument types match expected types.
// Example: Util_BlockArgTypesMatch<"body_block", "input_types">
class Util_BlockArgTypesMatch<string blockName, string expectedTypesSource> :
    Util_Constraint<"BlockArgTypesMatch_" # !subst(".", "_", blockName)> {
  let error = ERR_UTIL_REGION_0003;

  let verifyPredicate = [{
    auto block = $}] # blockName # [{;
    auto expectedTypes = $}] # expectedTypesSource # [{;
    SmallVector<Type> actualTypes;
    SmallVector<Type> expTypes;
    for (auto arg : block.getArguments()) actualTypes.push_back(arg.getType());
    for (auto type : expectedTypes) expTypes.push_back(type);
    return actualTypes == expTypes;
  }];

  let errorArgs = (binds
    (arg "actualTypes", "ArrayRef<Type>", [{actualTypes}]),
    (arg "expectedTypes", "ArrayRef<Type>", [{expTypes}])
  );
}

//===----------------------------------------------------------------------===//
// Array Attribute Constraints
//===----------------------------------------------------------------------===//

// Verifies that an array attribute has a specific length.
// Example: Util_StaticArrayLength<"static_offsets", 4>
class Util_StaticArrayLength<string attrName, int expectedLength> :
    Util_Constraint<"StaticArrayLength_" # !subst(".", "_", attrName)> {
  let error = ERR_UTIL_ATTR_0004;

  let locals = (binds
    (local "actualLength", "int64_t", [{
      static_cast<int64_t>($}] # attrName # [{.size())
    }])
  );

  let verifyPredicate = [{
    return actualLength == }] # !cast<string>(expectedLength) # [{;
  }];

  let errorArgs = (binds
    (arg "attrName", "StringRef", [{"}] # attrName # [{"}]),
    (arg "actualLength", "int64_t", "$actualLength"),
    (arg "expectedLength", "int64_t", expectedLength)
  );
}

// Verifies that the number of dynamic operands matches count of -1 markers in static array.
// Example: Util_DynamicCountMatchesMarkers<"static_offsets", "dynamic_offsets">
class Util_DynamicCountMatchesMarkers<string staticAttrName, string dynamicOperandName> :
    Util_Constraint<"DynamicCountMatchesMarkers_" # !subst(".", "_", staticAttrName)> {
  let error = ERR_UTIL_ATTR_0005;

  let locals = (binds
    (local "markerCount", "int64_t", [{
      [&]() {
        int64_t count = 0;
        for (auto elem : $}] # staticAttrName # [{) {
          if (cast<IntegerAttr>(elem).getInt() == -1) ++count;
        }
        return count;
      }()
    }]),
    (local "operandCount", "int64_t", [{
      static_cast<int64_t>($}] # dynamicOperandName # [{.size())
    }])
  );

  let verifyPredicate = [{
    return markerCount == operandCount;
  }];

  let errorArgs = (binds
    (arg "staticAttrName", "StringRef", [{"}] # staticAttrName # [{"}]),
    (arg "dynamicOperandName", "StringRef", [{"}] # dynamicOperandName # [{"}]),
    (arg "markerCount", "int64_t", "$markerCount"),
    (arg "operandCount", "int64_t", "$operandCount")
  );
}

// Verifies that an integer attribute is positive (> 0).
// Example: Util_PositiveI64Attr<"lanes">
class Util_PositiveI64Attr<string attrName> :
    Util_Constraint<"PositiveI64Attr_" # !subst(".", "_", attrName)> {
  let error = ERR_UTIL_ATTR_0006;

  let locals = (binds
    (local "value", "int64_t", [{
      $}] # attrName # [{
    }])
  );

  let verifyPredicate = [{
    return value > 0;
  }];

  let errorArgs = (binds
    (arg "attrName", "StringRef", [{"}] # attrName # [{"}]),
    (arg "value", "int64_t", "$value")
  );
}

//===----------------------------------------------------------------------===//
// Memory Layout Constraints
//===----------------------------------------------------------------------===//

// Verifies that a memref operand has identity layout.
// Example: Util_IdentityLayout<"memref_result">
class Util_IdentityLayout<string operandName> :
    Util_Constraint<"IdentityLayout_" # !subst(".", "_", operandName)> {
  let error = ERR_UTIL_VERIFY_0008;

  let verifyPredicate = [{
    return cast<MemRefType>($}] # operandName # [{.getType()).getLayout().isIdentity();
  }];

  let errorArgs = (binds
    (arg "operandName", "StringRef", [{"}] # operandName # [{"}]),
    (arg "operandType", "Type", [{$}] # operandName # [{.getType()}])
  );
}

#endif // IREE_COMPILER_DIALECT_UTIL_IR_UTIL_CONSTRAINTS_TD
