// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_IR_UTIL_ERROR_BASE_TD
#define IREE_COMPILER_DIALECT_UTIL_IR_UTIL_ERROR_BASE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Error Definition Base Classes
//===----------------------------------------------------------------------===//
//
// This file defines the base infrastructure for IREE's structured error
// handling system. All error IDs use underscores (not dashes) for consistent
// grep-ability across TableGen definitions, test files, and runtime output.
//
// Error ID Format: ERR_<DIALECT>_<DOMAIN>_<CODE>
// - DIALECT: UTIL, LOOM, STREAM, etc. (provided by wrapper class)
// - DOMAIN: Category/scope (VERIFY, FOLD, BROADCAST, TILE, etc.)
// - CODE: Sequential number within domain (0001, 0002, etc.)
//
// Example: ERR_UTIL_VERIFY_0001, ERR_LOOM_BROADCAST_0001
//
// Domain-based scoping allows:
// - Distributed error definitions in multiple files
// - Per-domain sequential numbering (no global coordination needed)
// - Dialect-specific domains (Loom: BROADCAST, SHAPE; Util: VERIFY, FOLD)
//
// Op-Specific Error Emission:
// Ops can declare which errors they emit via `let errors = [...]` field.
// iree-tblgen generates tag-dispatched emitErrorCode overloads that
// automatically extract fields from concrete op types, eliminating manual
// argument construction boilerplate.
//
// Example:
//   def TileBroadcastOp : Loom_Op<...> {
//     let errors = [ERR_LOOM_BROADCAST_0001];
//     // Auto-detected from constraints: ERR_LOOM_ENCODING_0001
//   }
//
// Usage in verifier:
//   // Automatically extracts operand/result encodings
//   return emitErrorCode<ERR_LOOM_ENCODING_0001>(*this);
//
//===----------------------------------------------------------------------===//
// AUTHORING GUIDE: Creating New Error Definitions
//===----------------------------------------------------------------------===//
//
// Quick Start Template:
//   def ERR_<DIALECT>_<DOMAIN>_<NNNN> : <Dialect>_Error<"DOMAIN", "NNNN"> {
//     let summary = [{One-line description.}];
//     let message = [{Error message with '{argName}' placeholders.}];
//     let schema = (ins
//       StringRefParameter<>:$argName,
//       "int64_t":$numericArg
//     );
//     let fixHint = [{Actionable fix with '{argName}' (got {numericArg}).}];
//   }
//
// CRITICAL: TableGen Syntax Pitfalls
// ----------------------------------
// 1. ALWAYS use multiline [{ }] blocks - never single-line for message/fixHint:
//    GOOD: let message = [{
//            operand '{name}' has invalid rank
//          }];
//    BAD:  let message = [{operand '{name}' has invalid rank}];
//
// 2. ALWAYS quote placeholders to avoid TableGen parse errors:
//    GOOD: [{Set '{attrName}' to >= 0 (got '{value}')}]
//    BAD:  [{Set {attrName} to >= 0 (got {value})}]  // May silently skip!
//
// 3. The pattern `word {placeholder}` without quotes can confuse TableGen:
//    GOOD: [{Block has '{actualCount}' arguments}]
//    BAD:  [{Block has {actualCount} arguments}]  // Breaks subsequent defs!
//
// 4. Placeholders ending with }] are especially problematic (}}}] pattern):
//    GOOD: [{expected '{count}'}]  // Quote provides separator
//    GOOD: [{expected {count}\n}]  // Newline provides separator
//    BAD:  [{expected {count}}]    // }}}] confuses parser
//
// Error Schema Parameter Types:
// ----------------------------------
// Simple C++ types (quoted strings):
//   "int64_t"       → 64-bit integer
//   "unsigned"      → unsigned integer
//   "Type"          → mlir::Type
//   "TypeRange"     → mlir::TypeRange
//   "StringRef"     → llvm::StringRef
//   "ArrayAttr"     → mlir::ArrayAttr
//
// MLIR Parameter classes (no quotes):
//   StringRefParameter<>                → StringRef (with allocation)
//   ArrayRefParameter<"Type">           → ArrayRef<Type>
//   ArrayRefParameter<"StringRef">      → ArrayRef<StringRef>
//   ArrayRefParameter<"int64_t">        → ArrayRef<int64_t>
//   ArrayRefParameter<"ArrayRef<int64_t>"> → ArrayRef<ArrayRef<int64_t>>
//
// Lazy range parameters (for variadic operands):
//   Util_RangeParameter<"std::string">  → Lazy string iterator
//   Util_RangeParameter<"Type">         → Lazy type iterator
//
// Placeholder Syntax in message/fixHint:
// ----------------------------------
//   {argName}           - Single value, streamed directly
//   {names|values}      - Zipped pair of Range parameters, formats as:
//                         'name1' (value1), 'name2' (value2), ...
//
// The zipped syntax is useful for showing correlated variadic data:
//   let message = [{
//     mismatched ranks: {names|ranks}
//   }];
//   // Output: mismatched ranks: 'a' (2), 'b' (3), 'c' (2)
//
// Best Practices for Error Messages:
// ----------------------------------
// 1. Start message with lowercase (follows MLIR convention)
// 2. Include relevant operand/attribute names in messages
// 3. Use '{name}' for string args, {value} for numeric args
// 4. Keep messages under 80 characters when possible
// 5. Provide actionable fixHint with actual values when available
//
// Diagnostic Severity:
// ----------------------------------
// Use the `severity` field to control how the diagnostic is emitted:
//   - "error" (default): Verification failure, compilation stops
//   - "warning": Non-fatal issue, compilation continues
//   - "remark": Informational only (e.g., optimization decisions)
//
// Example with remark severity (for poison folds):
//   def ERR_LOOM_FOLD_0001 : Loom_Error<"FOLD", "0001"> {
//     let severity = "remark";
//     let message = [{'{opName}' folded to poison: '{reason}'}];
//     let schema = (ins StringRefParameter<>:$opName, StringRefParameter<>:$reason);
//     let fixHint = [{Check bounds constraints - this indicates undefined behavior.}];
//   }
//
// Example - Complete Error Definition:
//   def ERR_LOOM_SHAPE_0001 : Loom_Error<"SHAPE", "0001"> {
//     let summary = [{Rank mismatch between source and result.}];
//     let description = [{
//       The result rank must be >= source rank for broadcast operations.
//       This ensures the source can be broadcast to the result shape.
//     }];
//     let message = [{rank of '{resultName}' ({resultRank}) must be >= '{sourceName}' ({sourceRank})}];
//     let schema = (ins
//       StringRefParameter<>:$resultName,
//       "int64_t":$resultRank,
//       StringRefParameter<>:$sourceName,
//       "int64_t":$sourceRank
//     );
//     let fixHint = [{Ensure '{resultName}' has rank >= '{sourceRank}' or add dimensions}];
//     let example = [{
//       // Valid: <4xf32> -> <2x4xf32> (rank 1 -> rank 2)
//       %result = loom.broadcast %source : <4xf32> -> <2x4xf32>
//     }];
//     let seeAlso = ["ERR_LOOM_SHAPE_0002"];
//   }
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Constraint Binding Operators
//===----------------------------------------------------------------------===//
//
// DAG operators for defining constraint argument bindings:
// - binds: Container for binding lists
// - arg: Error argument binding (name, type, value)
// - local: Local variable binding (name, type, extractor)
//
// Usage in constraints:
//   let locals = (binds
//     (local "actualRank", "unsigned", Util_ShapedTypeExtractor<operand>.extractRank)
//   );
//   let errorArgs = (binds
//     (arg "operandName", "StringRef", operand),
//     (arg "actualRank", "unsigned", $actualRank)
//   );
//
//===----------------------------------------------------------------------===//

def binds;  // Container for binding lists
def arg;    // Error argument binding (name, type, value)
def local;  // Local variable binding (name, type, extractor)

//===----------------------------------------------------------------------===//
// Error Argument Parameter Types
//===----------------------------------------------------------------------===//
//
// Error arguments use DAG syntax matching MLIR's Op/TypeDef parameter pattern.
// This reuses MLIR's existing Parameter infrastructure from AttrTypeBase.td.
//
// Available parameter types:
//
// Simple C++ types (use quoted strings):
//   "unsigned"      → unsigned
//   "int64_t"       → int64_t
//   "Type"          → mlir::Type (pulled into mlir namespace)
//   "StringRef"     → llvm::StringRef (pulled into mlir namespace)
//
// MLIR Parameter classes (no quotes, from AttrTypeBase.td):
//   StringRefParameter<>                → llvm::StringRef (with allocation)
//   ArrayRefParameter<"Type">           → ArrayRef<mlir::Type>
//   ArrayRefParameter<"StringRef">      → ArrayRef<llvm::StringRef>
//   ArrayRefParameter<"int64_t">        → ArrayRef<int64_t>
//
// Lazy range parameter (IREE extension):
//   Util_RangeParameter<"std::string">  → function_ref<void(function_ref<void(std::string)>)>
//   Util_RangeParameter<"ArrayRef<int64_t>"> → function_ref<void(function_ref<void(ArrayRef<int64_t>)>)>
//
// Example usage:
//   def ERR_UTIL_EXAMPLE_0001 : Util_Error<"EXAMPLE", "0001"> {
//     let message = [{operand '{name}' rank ({rank}) incompatible}];
//     let schema = (ins
//       StringRefParameter<>:$name,           // String with allocation
//       "unsigned":$rank,                     // Plain C++ type
//       "Type":$operandType,                  // MLIR Type
//       ArrayRefParameter<"StringRef">:$dims  // Array of strings
//     );
//   }
//
// The message template uses {argName} placeholders that are replaced with
// the actual argument values. ArrayRef arguments are automatically formatted
// using llvm::interleaveComma() for comma-separated output.
//
//===----------------------------------------------------------------------===//

// Base class for all IREE error definitions.
// Error IDs use underscores for consistency and grep-ability.
// Parameterized by:
//   - dialect: Dialect record (for C++ namespace extraction)
//   - dialectName: Dialect name string (e.g., "UTIL", "LOOM")
//   - domainName: Error domain/category (e.g., "VERIFY", "BROADCAST")
//   - codeNum: Sequential error number within domain (e.g., "0001")
class Util_ErrorBase<Dialect dialect, string dialectName, string domainName, string codeNum> {
  // The dialect this error belongs to (for C++ namespace extraction).
  Dialect errorDialect = dialect;

  // Error domain/category (e.g., "VERIFY", "FOLD", "BROADCAST", "TILE").
  // Passed as template argument.
  // Domains are dialect-specific and provide scoped error numbering.
  string domain = domainName;

  // Numeric code within domain (e.g., "0001").
  // Passed as template argument.
  // Sequential within each domain, allowing distributed definitions.
  string errorCode = codeNum;

  // Full error identifier auto-constructed from components.
  // Format: ERR_<DIALECT>_<DOMAIN>_<CODE>
  // Example: "ERR_UTIL_VERIFY_0001", "ERR_LOOM_BROADCAST_0001"
  string errorId = "ERR_" # dialectName # "_" # domainName # "_" # codeNum;

  // Brief one-line summary of the error (emitted as C++ comment).
  string summary = "";

  // Detailed description of when this error occurs (emitted as C++ comment).
  string description = "";

  // Human-readable message template.
  // Can use {argName} placeholders for interpolation.
  code message = [{}];

  // Typed arguments expected by this error.
  dag schema = (ins);

  // Suggested fix for the error.
  code fixHint = [{}];

  // Code example showing correct usage.
  code example = [{}];

  // Related error IDs for cross-reference.
  list<string> seeAlso = [];

  // Diagnostic severity: "error", "warning", or "remark".
  // Default is "error" for backward compatibility.
  // Use "remark" for informational diagnostics (e.g., poison folds).
  // Use "warning" for non-fatal issues that may indicate problems.
  string severity = "error";
}

//===----------------------------------------------------------------------===//
// Constraint Definition Base Classes
//===----------------------------------------------------------------------===//
//
// Constraints generate C++ trait classes that verify operation invariants.
// They integrate with MLIR's ODS (Operation Definition Specification) system
// to access operation fields through generated accessor methods.
//
//===----------------------------------------------------------------------===//

// Range parameter for lazy iteration in error schemas.
// C++ type: function_ref<void(function_ref<void(elemType)>)>
// Inherits from AttrOrTypeParameter so it can be used in DAG operand lists.
class Util_RangeParameter<string elemType>
    : AttrOrTypeParameter<"::llvm::function_ref<void(::llvm::function_ref<void("
                          # elemType # ")>)>", "lazy range of " # elemType> {
  string elementType = elemType;
}

//===----------------------------------------------------------------------===//
// Field Reference Syntax ($field)
//===----------------------------------------------------------------------===//
//
// Constraints and property extractors use MLIR-style $field references to
// access operation fields (operands, results, attributes). iree-tblgen uses
// ODS introspection to generate correct accessor calls.
//
// Usage in TableGen:
//   let verifyPredicate = [{
//     return cast<TileType>($operand.getType()).getRank() > 0;
//   }];
//
// For trait headers (verifyTrait): $field → concreteOp.getField()
// For constraint impls (check functions): $field → field (parameter name)
//
// For parametric constraints using list<string>, concatenate with the field:
//   class Loom_SameEncoding<list<string> operands> {
//     let verifyPredicate = [{
//       auto firstEnc = cast<TileType>($}] # !head(operands) # [{.getType()).getEncoding();
//       ...
//     }];
//   }
//
// The $field syntax works for all field types:
//   - Operands:   $operand_name
//   - Results:    $result_name
//   - Attributes: $attr_name
//
// iree-tblgen uses op.getGetterName() from MLIR ODS for exact accessor names.
//
//===----------------------------------------------------------------------===//

// Base class for constraint definitions that generate C++ trait classes.
// Inherits from NativeOpTrait so constraints can be used directly in op trait lists.
// Traits are generated in the ::mlir::OpTrait namespace (MLIR convention).
class Util_ConstraintBase<string traitClassName> :
    NativeOpTrait<traitClassName> {
  // The error definition associated with this constraint.
  // Can reference errors from any dialect (Util, Loom, Stream, etc.).
  Util_ErrorBase error;

  // Name of the generated C++ trait class.
  // Example: "Util_RankAtLeast_result_operand"
  string traitName = traitClassName;

  //===--------------------------------------------------------------------===//
  // Verification Logic
  //===--------------------------------------------------------------------===//

  // Local variables extracted for use in the predicate.
  // These are computed before the predicate and available in its scope.
  // Format: (binds (local "name", "type", extractor), ...)
  dag locals = (binds);

  // Arguments to pass to the error on failure.
  // Can reference locals with $localName or extract new values.
  // Format: (binds (arg "name", "type", value), ...)
  dag errorArgs = (binds);

  // C++ code that returns bool for the verification predicate.
  // Variables from 'locals' are automatically in scope.
  // Use $field syntax to reference operation fields (e.g., $operand).
  code verifyPredicate = [{}];

  // Manual error emission code.
  // If not specified, error emission is auto-generated from 'errorArgs'.
  // Should return InFlightDiagnostic.
  code emitErrorCode = [{}];
}

// Util constraint wrapper - adds IREE::Util namespace prefix for OpTrait lookup.
// Example: def Util_SameShape<...> : Util_Constraint<"SameShape_a_b_result"> { ... }
// The trait will be generated in ::mlir::OpTrait::IREE::Util::<name>.
class Util_Constraint<string name> : Util_ConstraintBase<"IREE::Util::" # name>;

// Property extractor for getting values from operations.
// Used to generate code that extracts ranks, types, etc.
// Extractors use $field syntax to access operation fields.
class Util_PropertyExtractor<string name> {
  // Name of the value to extract from (e.g., "result", "operand_dims").
  string valueName = name;

  // C++ code to extract the rank.
  // Use $field syntax: $fieldName.getType()
  code extractRank = [{}];

  // C++ code to extract the type.
  // Use $field syntax: $fieldName.getType()
  code extractType = [{}];

  // C++ code to extract the element type.
  // Use $field syntax: cast<ShapedType>($fieldName.getType()).getElementType()
  code extractElementType = [{}];

  // C++ code to extract the shape.
  // Use $field syntax: cast<ShapedType>($fieldName.getType()).getShape()
  code extractShape = [{}];
}

// Property extractor for ShapedType operations (tile types, tensors, memrefs).
// Provides uniform access to shape properties across all shaped types via
// ShapedTypeInterface, enabling generic shape constraints that work with any
// type implementing the interface.
class Util_ShapedTypeExtractor<string name> : Util_PropertyExtractor<name> {
  let extractType = [{
    $}] # name # [{.getType()
  }];

  let extractRank = [{
    cast<ShapedType>($}] # name # [{.getType()).getRank()
  }];

  let extractElementType = [{
    cast<ShapedType>($}] # name # [{.getType()).getElementType()
  }];

  let extractShape = [{
    cast<ShapedType>($}] # name # [{.getType()).getShape()
  }];
}

//===----------------------------------------------------------------------===//
// Constraint Traits
//===----------------------------------------------------------------------===//

// Marker trait for constraints that are generated by iree-tblgen.
// The actual C++ trait will be generated from the constraint definition.
class Util_GeneratedConstraintTrait<string className> :
    NativeOpTrait<"IREE::Util::" # className> {
}

#endif // IREE_COMPILER_DIALECT_UTIL_IR_UTIL_ERROR_BASE_TD
