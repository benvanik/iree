// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared utilities for IREE TableGen generators.

#ifndef IREE_COMPILER_TABLEGEN_UTILS_H_
#define IREE_COMPILER_TABLEGEN_UTILS_H_

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Record.h"

namespace mlir::iree_compiler::tblgen {

//===----------------------------------------------------------------------===//
// Field Type Information
//===----------------------------------------------------------------------===//

// Represents the kind of field for constraint parameterization.
enum class FieldKind {
  Value,             // Single operand or result (::mlir::Value).
  Variadic,          // Variadic operands or results (::mlir::ValueRange).
  Attribute,         // Attribute (::mlir::Attribute).
  Region,            // Single region (::mlir::Region&).
  VariadicRegion,    // Variadic regions
                     // (::mlir::MutableArrayRef<::mlir::Region>).
  Successor,         // Single successor (::mlir::Block*).
  VariadicSuccessor, // Variadic successors (::mlir::SuccessorRange).
  Property,          // Property (interface type from ODS).
  Unknown            // Could not determine field type.
};

// Information about an operation field for constraint generation.
struct FieldInfo {
  std::string name;         // Snake_case field name from ODS.
  std::string accessorName; // CamelCase accessor name (e.g., "OperandDims").
  FieldKind kind;           // Kind of field (Value, Variadic, etc.).
  std::string cppType;      // C++ type for constraint function parameter.
  bool isVariadic;          // True if variadic operand/result/region.
};

//===----------------------------------------------------------------------===//
// Binding Structures
//===----------------------------------------------------------------------===//

// Represents a local variable binding in a constraint.
// Format: (local "name", "type", extractor)
struct LocalBinding {
  std::string name;      // Variable name (e.g., "actualRank").
  std::string type;      // C++ type (e.g., "unsigned").
  std::string extractor; // Code to extract value.
};

// Represents an error argument binding in a constraint.
// Format: (arg "name", "type", value)
struct ErrorArgBinding {
  std::string name;  // Argument name (e.g., "operandName").
  std::string type;  // C++ type (e.g., "StringRef").
  std::string value; // Code or literal value.
  bool isLocalRef;   // True if value is $localName reference.
};

//===----------------------------------------------------------------------===//
// String Processing
//===----------------------------------------------------------------------===//

// Strips consistent leading whitespace from description text.
// Detects the pattern `= [{\n  ...` and removes that leading whitespace
// from all lines. Preserves relative indentation for ASCII art, etc.
// Returns original text if parsing fails at any point.
std::string cleanDescription(llvm::StringRef desc);

// Cleans assembly format by replacing newlines with spaces.
// Assembly format strings in TableGen files may have newlines for readability,
// but they're ignored by the parser so we normalize to single line.
std::string cleanAssemblyFormat(llvm::StringRef format);

// Resolves relative path to absolute path using the defining record's location.
// This is best-effort resolution - returns original path if resolution fails.
// Paths in metadata are relative to the repository root (e.g.,
// "test/foo.mlir"), so we walk up from the .td file location to find the
// repository root. Note: Assumes IREE-style repo structure with "compiler/src/"
// directory.
std::string resolveRelativePath(llvm::StringRef relativePath,
                                const llvm::Record *definingRecord);

//===----------------------------------------------------------------------===//
// Source Location
//===----------------------------------------------------------------------===//

// Emits source location information for a record as JSON.
// Gracefully degrades: if location can't be determined, emits nothing.
// This is acceptable as source location is supplementary information.
void emitSourceLocation(const llvm::Record *def,
                        const llvm::RecordKeeper &records,
                        llvm::json::OStream &J);

// Emits a #line directive to make compiler errors point to the TableGen source.
// This helps authors debug when their verifyPredicate code has syntax errors.
// Returns true if a directive was emitted.
bool emitLineDirective(llvm::raw_ostream &os, llvm::SMLoc loc);

//===----------------------------------------------------------------------===//
// Field Processing
//===----------------------------------------------------------------------===//

// Builds a map from field names to their type information by scanning all
// operation definitions in the TableGen records. This enables type-correct
// constraint function generation.
llvm::StringMap<FieldInfo> buildFieldTypeMap(const llvm::RecordKeeper &records);

// Extracts field names from $field references in constraint code.
// Returns a list of unique field names referenced in the constraint.
// Uses MLIR-style $field syntax (e.g., $operand, $result).
// Example: code with "$lhs" and "$rhs" returns ["lhs", "rhs"].
std::vector<std::string> extractFieldNames(llvm::StringRef code);

// Processes $field references in code strings.
// For tag dispatch (useConcreteOp=true): $field -> concreteOp.getField()
// For check functions (useConcreteOp=false): $field -> field (parameter name)
std::string
processFieldReferences(llvm::StringRef code,
                       const llvm::StringMap<FieldInfo> &fieldTypeMap,
                       bool useConcreteOp);

//===----------------------------------------------------------------------===//
// Binding Parsing
//===----------------------------------------------------------------------===//

// Parses a single binding from a DAG arg.
// Format: (local "name", "type", extractor) or (arg "name", "type", value)
// Returns tuple of (name, type, value/extractor).
std::tuple<std::string, std::string, std::string>
parseBindingArgs(const llvm::DagInit *bindingDag);

// Parses locals bindings from a constraint.
// Format: let locals = (binds (local "name", "type", extractor), ...)
std::vector<LocalBinding> parseLocals(const llvm::Record *constraint);

// Parses errorArgs bindings from a constraint.
// Format: let errorArgs = (binds (arg "name", "type", value), ...)
std::vector<ErrorArgBinding> parseErrorArgs(const llvm::Record *constraint);

// Validates that errorArgs bindings match the error schema.
// Returns true if valid, false otherwise (with error messages printed).
bool validateErrorArgsAgainstSchema(
    const llvm::Record *constraint,
    const std::vector<ErrorArgBinding> &errorArgs,
    const std::vector<LocalBinding> &locals, const llvm::Record *errorRecord);

//===----------------------------------------------------------------------===//
// Namespace Handling
//===----------------------------------------------------------------------===//

// Extracts trait namespace suffix from a dialect's cppNamespace.
// This ensures the tool works for any dialect, not just IREE.
// Examples:
//   "::mlir::iree_compiler::IREE::Stream" -> "IREE::Stream"
//   "::mlir::spirv" -> "spirv"
std::string getTraitNamespaceSuffix(llvm::StringRef cppNamespace);

// Extracts dialect name from a fully qualified interface name.
// Tries multiple common namespace patterns:
// 1. ::mlir::iree_compiler::IREE::DialectName::ClassName (IREE pattern)
// 2. ::mlir::DialectName::ClassName (simple MLIR pattern)
// 3. Generic: second-to-last namespace component before ClassName
// Returns empty optional if extraction fails.
std::optional<std::string> extractDialectFromNamespace(llvm::StringRef fqn);

} // namespace mlir::iree_compiler::tblgen

#endif // IREE_COMPILER_TABLEGEN_UTILS_H_
