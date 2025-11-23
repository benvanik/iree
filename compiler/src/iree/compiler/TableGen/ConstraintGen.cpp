// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <map>
#include <set>

#include "Utils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "mlir/TableGen/GenInfo.h"

using namespace llvm;

namespace mlir::iree_compiler::tblgen {

// Command-line option to specify which dialect's constraints to generate.
// This filters constraints by their error's errorDialect field.
static cl::opt<std::string>
    targetDialect("dialect",
                  cl::desc("Target dialect name (e.g., 'Loom', 'Util')"),
                  cl::cat(cl::getGeneralCategory()));

//===----------------------------------------------------------------------===//
// Constraint Traits Header Generation (All Logic Inlined)
//===----------------------------------------------------------------------===//

// NOTE ON NAMESPACE RESOLUTION:
// Traits are generated in mlir::OpTrait::IREE::* to follow MLIR conventions.
// However, dialect types live in mlir::iree_compiler::IREE::*. This creates
// a namespace conflict: inside mlir::OpTrait::IREE::Loom, a qualified reference
// to IREE::Loom::TileType finds mlir::OpTrait::IREE::Loom (wrong) instead of
// mlir::iree_compiler::IREE::Loom (correct). This happens because C++ qualified
// lookup starts from the innermost enclosing namespace.
//
// We solve this with a namespace alias at the beginning of each trait
// namespace:
//   namespace IREE = ::mlir::iree_compiler::IREE;
// This makes the name "IREE" in the trait namespace explicitly refer to the
// dialect namespace, shadowing the OpTrait parent namespace. The alias is
// scoped to the trait namespace, not the global namespace.
//
// IMPORTANT: .td files must use IREE::Dialect:: prefixed types (e.g.,
// IREE::Loom::TileType, not just TileType). Unqualified types won't resolve.
//
// NOTE: "using namespace ::mlir::iree_compiler;" does NOT work here because
// using-directives only affect *unqualified* lookup, not *qualified* lookup.
// A qualified reference like "IREE::Loom::*" starts lookup from the innermost
// namespace and finds mlir::OpTrait::IREE before checking the using-directive.
//
// Alternative approaches considered:
// - Put traits in mlir::iree_compiler::* (breaks MLIR convention)
// - Require fully qualified ::mlir::iree_compiler::... (too verbose)
// - String-replace IREE:: with ::mlir::iree_compiler::IREE:: (fragile)
//
// If MLIR ever allows traits outside mlir::OpTrait::*, we should reconsider.

bool emitConstraintTraits(const RecordKeeper &records, raw_ostream &os) {
  // Find all Util_ConstraintBase records.
  const Record *constraintClass = records.getClass("Util_ConstraintBase");
  if (!constraintClass) {
    return false;
  }

  // Get the target dialect from command line (required for header guard
  // naming).
  if (targetDialect.empty()) {
    PrintError("--dialect flag is required (e.g., --dialect=Loom)");
    return true;
  }

  // Collect ALL constraint instances (not filtered by dialect).
  // The dialect flag is only used for header guard naming.
  // Constraints from various error dialects may be used together (e.g., Loom
  // ops using both Loom and Util constraints).
  std::vector<const Record *> constraints;
  for (const auto &def : records.getDefs()) {
    if (!def.second->isSubClassOf(constraintClass))
      continue;

    // Must have an error record with errorDialect.
    const Record *errorRecord = def.second->getValueAsDef("error");
    if (!errorRecord)
      continue;
    if (!errorRecord->getValueAsDef("errorDialect"))
      continue;

    constraints.push_back(def.second.get());
  }

  if (constraints.empty()) {
    // No constraints found - emit empty header.
    return false;
  }

  // Build the field type map from all operations.
  StringMap<FieldInfo> fieldTypeMap = buildFieldTypeMap(records);

  // Use the target dialect for header guard and naming.
  std::string dialectNameUpper = StringRef(targetDialect).upper();

  // Emit header guard.
  os << "#ifndef IREE_COMPILER_DIALECT_" << dialectNameUpper
     << "_IR_GENERATED_CONSTRAINT_TRAITS_H_\n";
  os << "#define IREE_COMPILER_DIALECT_" << dialectNameUpper
     << "_IR_GENERATED_CONSTRAINT_TRAITS_H_\n\n";

  // Emit includes.
  os << "#include \"mlir/IR/Diagnostics.h\"\n";
  os << "#include \"mlir/IR/OpDefinition.h\"\n";
  os << "#include \"mlir/IR/Operation.h\"\n";
  os << "#include \"mlir/IR/Types.h\"\n";

  // Collect unique error dialect headers to include.
  std::set<std::string> errorHeaderIncludes;
  for (const Record *constraint : constraints) {
    const Record *errorRecord = constraint->getValueAsDef("error");
    if (!errorRecord)
      continue;
    const Record *errorDialect = errorRecord->getValueAsDef("errorDialect");
    std::string dialName = errorDialect->getValueAsString("name").str();
    if (!dialName.empty())
      dialName[0] = std::toupper(dialName[0]);
    std::string header =
        "iree/compiler/Dialect/" + dialName + "/IR/" + dialName + "Errors.h";
    errorHeaderIncludes.insert(header);
  }
  for (const auto &header : errorHeaderIncludes) {
    os << "#include \"" << header << "\"\n";
  }
  os << "\n";

  // Group constraints by their trait namespace.
  std::map<std::string, std::vector<const Record *>> constraintsByNamespace;
  for (const Record *constraint : constraints) {
    StringRef traitName = constraint->getValueAsString("traitName");
    if (traitName.empty())
      continue;

    // Extract namespace from trait name (e.g., "IREE::Loom::Foo" ->
    // "IREE::Loom").
    size_t lastColons = traitName.rfind("::");
    std::string ns = "IREE";
    if (lastColons != StringRef::npos) {
      ns = traitName.substr(0, lastColons).str();
    }
    constraintsByNamespace[ns].push_back(constraint);
  }

  // Emit trait classes grouped by namespace.
  for (const auto &[ns, nsConstraints] : constraintsByNamespace) {
    os << "namespace mlir::OpTrait::" << ns << " {\n\n";

    // Emit namespace alias to resolve IREE::* types correctly.
    // See NOTE ON NAMESPACE RESOLUTION above.
    os << "namespace IREE = ::mlir::iree_compiler::IREE;\n\n";

    for (const Record *constraint : nsConstraints) {
      StringRef traitName = constraint->getValueAsString("traitName");
      StringRef simpleClassName = traitName;
      size_t lastColons = traitName.rfind("::");
      if (lastColons != StringRef::npos) {
        simpleClassName = traitName.substr(lastColons + 2);
      }

      const Record *errorRecord = constraint->getValueAsDef("error");
      if (!errorRecord)
        continue;

      StringRef errorId = errorRecord->getValueAsString("errorId");
      const Record *errorDialect = errorRecord->getValueAsDef("errorDialect");
      std::string errorDialectNs = getTraitNamespaceSuffix(
          errorDialect->getValueAsString("cppNamespace"));

      // Parse constraint definition.
      std::vector<LocalBinding> locals = parseLocals(constraint);
      std::vector<ErrorArgBinding> errorArgs = parseErrorArgs(constraint);
      std::string verifyPredicate =
          constraint->getValueAsString("verifyPredicate").str();

      // Extract all field names referenced.
      std::vector<std::string> fieldNames;
      {
        auto predicateFields = extractFieldNames(verifyPredicate);
        for (const auto &field : predicateFields) {
          if (std::find(fieldNames.begin(), fieldNames.end(), field) ==
              fieldNames.end()) {
            fieldNames.push_back(field);
          }
        }
        for (const auto &local : locals) {
          auto fields = extractFieldNames(local.extractor);
          for (const auto &field : fields) {
            if (std::find(fieldNames.begin(), fieldNames.end(), field) ==
                fieldNames.end()) {
              fieldNames.push_back(field);
            }
          }
        }
        for (const auto &arg : errorArgs) {
          if (!arg.isLocalRef) {
            auto fields = extractFieldNames(arg.value);
            for (const auto &field : fields) {
              if (std::find(fieldNames.begin(), fieldNames.end(), field) ==
                  fieldNames.end()) {
                fieldNames.push_back(field);
              }
            }
          }
        }
      }

      // Emit trait class header.
      os << "//===----------------------------------------------------------"
            "------------===//\n";
      os << "// " << simpleClassName << " (Error: " << errorId << ")\n";
      os << "//===----------------------------------------------------------"
            "------------===//\n\n";

      os << "template <typename ConcreteType>\n";
      os << "class " << simpleClassName
         << " : public ::mlir::OpTrait::TraitBase<ConcreteType, "
         << simpleClassName << "> {\n";
      os << "public:\n";
      os << "  static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) "
            "{\n";

      // Extract field values.
      if (!fieldNames.empty()) {
        os << "    auto concreteOp = ::llvm::cast<ConcreteType>(op);\n";
        for (const auto &fieldName : fieldNames) {
          std::string accessorName;
          auto it = fieldTypeMap.find(fieldName);
          if (it != fieldTypeMap.end()) {
            accessorName = it->second.accessorName;
          } else {
            accessorName = "get" + convertToCamelFromSnakeCase(fieldName, true);
          }
          os << "    decltype(auto) " << fieldName << " = concreteOp."
             << accessorName << "();\n";
        }
        os << "\n";
      }

      // Process field references.
      std::string processedPredicate =
          processFieldReferences(verifyPredicate, fieldTypeMap, false);

      // Wrap verification in a lambda for multi-statement predicates.
      os << "    return [&]() -> ::mlir::LogicalResult {\n";

      // Emit locals.
      if (!locals.empty()) {
        for (const auto &local : locals) {
          std::string processedExtractor =
              processFieldReferences(local.extractor, fieldTypeMap, false);
          os << "      " << local.type << " " << local.name << " = "
             << processedExtractor << ";\n";
        }
      }

      // Emit predicate check.
      // Inner lambda handles multi-statement predicates.
      os << "      if (![&]() -> bool {\n";

      // Emit #line directive.
      if (const RecordVal *predicateVal =
              constraint->getValue("verifyPredicate")) {
        os << "        ";
        emitLineDirective(os, predicateVal->getLoc());
      }

      os << "        " << processedPredicate << "\n";
      os << "      }()) {\n";

      // Emit error.
      if (!errorArgs.empty()) {
        os << "        return " << errorDialectNs << "::Errors::" << errorId
           << "::emit(op, {";

        for (size_t i = 0; i < errorArgs.size(); ++i) {
          if (i > 0)
            os << ", ";

          const auto &arg = errorArgs[i];
          if (arg.isLocalRef) {
            std::string localName = arg.value.substr(1);
            os << localName;
          } else {
            std::string processedValue =
                processFieldReferences(arg.value, fieldTypeMap, false);

            // Quote simple identifiers of StringRef type.
            bool needsQuoting = false;
            if (arg.type == "StringRef" && !processedValue.empty()) {
              needsQuoting = true;
              for (char c : processedValue) {
                if (!std::isalnum(c) && c != '_') {
                  needsQuoting = false;
                  break;
                }
              }
            }

            if (needsQuoting) {
              os << "\"" << processedValue << "\"";
            } else {
              os << processedValue;
            }
          }
        }

        os << "});\n";
      } else {
        os << "        return ::mlir::emitError(op->getLoc()) << \"" << errorId
           << "\";\n";
      }

      os << "      }\n";
      os << "      return ::mlir::success();\n";
      os << "    }();\n";
      os << "  }\n";
      os << "};\n\n";
    }

    os << "} // namespace mlir::OpTrait::" << ns << "\n\n";
  }

  os << "#endif // IREE_COMPILER_DIALECT_" << dialectNameUpper
     << "_IR_GENERATED_CONSTRAINT_TRAITS_H_\n";

  return false;
}

} // namespace mlir::iree_compiler::tblgen

// Register the constraint trait generator.
static mlir::GenRegistration genConstraintTraits(
    "gen-iree-constraint-traits",
    "Generate C++ trait classes from constraint definitions",
    mlir::iree_compiler::tblgen::emitConstraintTraits);
