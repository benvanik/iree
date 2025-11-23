// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cctype>
#include <map>
#include <set>
#include <vector>

#include "Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

using namespace llvm;

// Command line option for warning about unused errors.
static cl::opt<bool> warnUnusedErrors(
    "warn-unused-errors",
    cl::desc("Warn about error definitions not referenced by constraints"),
    cl::init(false));

namespace mlir::iree_compiler::tblgen {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

enum class ArgKind { Normal, ArrayRef, Range };

struct ArgInfo {
  std::string name;
  std::string cppType;
  ArgKind kind;
  std::string rangeElementType; // only for Range
};

// Helper to extract argument metadata from a DAG.
static std::vector<ArgInfo>
extractErrorArgsFromDag(const DagInit *errorArgsDag) {
  std::vector<ArgInfo> args;

  if (!errorArgsDag) {
    return args;
  }

  if (errorArgsDag->getNumArgs() == 0) {
    return args;
  }

  for (unsigned i = 0; i < errorArgsDag->getNumArgs(); ++i) {
    StringRef argName = errorArgsDag->getArgNameStr(i);
    const Init *argInit = errorArgsDag->getArg(i);

    std::string cppType;
    ArgKind kind = ArgKind::Normal;
    std::string elementType;
    if (const auto *SI = dyn_cast<StringInit>(argInit)) {
      // Plain quoted string: "unsigned", "int64_t", "Type", "StringRef".
      cppType = SI->getValue().str();

      // Normalize common types without full namespace qualification.
      // We're in mlir namespace and MLIR's support header pulls StringRef in.
      if (cppType == "Type") {
        cppType = "Type"; // Already correct, in mlir namespace.
      } else if (cppType == "StringRef") {
        cppType = "StringRef"; // Already correct, pulled into mlir namespace.
      }
    } else if (const auto *DI = dyn_cast<DefInit>(argInit)) {
      // Parameter class: StringRefParameter, ArrayRefParameter, etc.
      const Record *paramDef = DI->getDef();
      if (paramDef->getValue("cppType")) {
        cppType = paramDef->getValueAsString("cppType").str();
      } else {
        // Fallback if no cppType field (shouldn't happen with proper Parameter
        // classes).
        cppType = "UNKNOWN_TYPE";
      }

      if (paramDef->isSubClassOf("Util_RangeParameter")) {
        kind = ArgKind::Range;
        if (paramDef->getValue("elementType"))
          elementType = paramDef->getValueAsString("elementType").str();
      } else if (cppType.find("ArrayRef<") != std::string::npos) {
        kind = ArgKind::ArrayRef;
      }
    } else {
      // Unexpected Init type.
      cppType = "UNKNOWN_TYPE";
    }

    args.push_back({argName.str(), cppType, kind, elementType});
  }

  return args;
}

// Extract all placeholder names from a template string.
// Handles {name} and {name1|name2} (zipped) syntax.
// Returns a set of placeholder names found.
static std::set<std::string> extractPlaceholders(StringRef templateStr) {
  std::set<std::string> placeholders;
  size_t pos = 0;
  while (pos < templateStr.size()) {
    size_t start = templateStr.find('{', pos);
    if (start == StringRef::npos)
      break;
    size_t end = templateStr.find('}', start);
    if (end == StringRef::npos)
      break;

    StringRef placeholder = templateStr.slice(start + 1, end);
    // Handle zipped syntax {name1|name2}.
    if (placeholder.contains('|')) {
      auto [first, second] = placeholder.split('|');
      if (!first.empty())
        placeholders.insert(first.str());
      if (!second.empty())
        placeholders.insert(second.str());
    } else if (!placeholder.empty()) {
      placeholders.insert(placeholder.str());
    }
    pos = end + 1;
  }
  return placeholders;
}

// Validate that all placeholders in a template string exist in the schema.
// Emits a fatal error if invalid - this catches TableGen parsing issues where
// code blocks ending with {placeholder}}] get corrupted.
static void validatePlaceholders(const Record *error, StringRef fieldName,
                                 StringRef templateStr,
                                 const std::vector<ArgInfo> &argInfos) {
  std::set<std::string> schemaArgs;
  for (const auto &arg : argInfos)
    schemaArgs.insert(arg.name);

  std::set<std::string> placeholders = extractPlaceholders(templateStr);

  for (const std::string &placeholder : placeholders) {
    if (schemaArgs.find(placeholder) == schemaArgs.end()) {
      // This usually indicates TableGen silently corrupted the record due to
      // a code block parsing issue (e.g., {placeholder}}] at end of block).
      PrintFatalError(error->getLoc(),
                      "Error '" + error->getValueAsString("errorId") +
                          "': placeholder '{" + placeholder + "}' in " +
                          fieldName + " not found in schema. " +
                          "This may indicate TableGen mis-parsed the code "
                          "block - check for {placeholder}}] patterns at the "
                          "end of [{...}] blocks (wrap in quotes: "
                          "'{placeholder}').");
    }
  }
}

// Helper to emit code for a single Range placeholder.
static void emitRangePlaceholder(raw_ostream &os, const std::string &indent,
                                 StringRef placeholderName,
                                 const ArgInfo &info) {
  bool elementIsArrayRef =
      info.rangeElementType.find("ArrayRef<int64_t>") != std::string::npos;
  os << indent << "{ bool first = true; args." << placeholderName
     << "([&](auto v) {\n";
  os << indent << "  if (!first) stream << \", \";\n";
  os << indent << "  first = false;\n";
  if (elementIsArrayRef) {
    os << indent << "  ::llvm::interleaveComma(v, stream);\n";
  } else {
    os << indent << "  stream << v;\n";
  }
  os << indent << "}); }\n";
}

// Helper to emit code for zipped Range placeholders: {name1:name2}
// Produces output like: 'a' (2), 'b' (3)
// The first range is treated as names (quoted), second as values.
static void emitZippedRangePlaceholder(
    raw_ostream &os, const std::string &indent, StringRef firstName,
    StringRef secondName, const ArgInfo &firstInfo, const ArgInfo &secondInfo) {
  bool secondIsArrayRef = secondInfo.rangeElementType.find(
                              "ArrayRef<int64_t>") != std::string::npos;

  // Collect both ranges into vectors, then zip-iterate.
  os << indent << "{\n";
  os << indent << "  ::llvm::SmallVector<" << firstInfo.rangeElementType << "> "
     << firstName << "Vec;\n";
  os << indent << "  ::llvm::SmallVector<" << secondInfo.rangeElementType
     << "> " << secondName << "Vec;\n";
  os << indent << "  args." << firstName << "([&](auto v) { " << firstName
     << "Vec.push_back(v); });\n";
  os << indent << "  args." << secondName << "([&](auto v) { " << secondName
     << "Vec.push_back(v); });\n";
  os << indent << "  for (size_t i = 0; i < " << firstName
     << "Vec.size() && i < " << secondName << "Vec.size(); ++i) {\n";
  os << indent << "    if (i > 0) stream << \", \";\n";
  os << indent << "    stream << \"'\" << " << firstName
     << "Vec[i] << \"' (\";\n";
  if (secondIsArrayRef) {
    os << indent << "    ::llvm::interleaveComma(" << secondName
       << "Vec[i], stream);\n";
  } else {
    os << indent << "    stream << " << secondName << "Vec[i];\n";
  }
  os << indent << "    stream << \")\";\n";
  os << indent << "  }\n";
  os << indent << "}\n";
}

// Helper to generate code for named placeholder substitution.
// Parses template with {name} placeholders and generates C++ code that builds
// the result string using llvm::raw_string_ostream.
//
// Supports two placeholder syntaxes:
//   {name}        - Single placeholder, streams the value directly
//   {name1|name2} - Zipped pair of Range parameters, formats as:
//                   'name1_val' (name2_val), 'name1_val' (name2_val), ...
//
// The zipped syntax uses pipe (|) instead of colon (:) to avoid TableGen
// parsing issues with [{ }] code blocks. With colon, {a:b}}] would be
// mis-parsed as the code block ending at the first }.
//
// The zipped syntax is useful for showing correlated data like operand names
// with their ranks/shapes/types.
static void emitTemplateSubstitution(raw_ostream &os, StringRef templateStr,
                                     const std::string &indent,
                                     const std::vector<ArgInfo> &argInfos) {
  StringMap<size_t> argIndexMap;
  for (size_t i = 0; i < argInfos.size(); ++i)
    argIndexMap[argInfos[i].name] = i;

  os << indent << "std::string result;\n";
  os << indent << "::llvm::raw_string_ostream stream(result);\n";

  size_t pos = 0;
  while (pos < templateStr.size()) {
    size_t start = templateStr.find('{', pos);

    // Emit literal segment before the placeholder.
    if (start == StringRef::npos) {
      // No more placeholders, emit rest of string.
      if (pos < templateStr.size()) {
        StringRef literal = templateStr.substr(pos);
        os << indent << "stream << R\"(" << literal << ")\";\n";
      }
      break;
    }

    // Emit literal before placeholder.
    if (start > pos) {
      StringRef literal = templateStr.substr(pos, start - pos);
      os << indent << "stream << R\"(" << literal << ")\";\n";
    }

    // Find end of placeholder.
    size_t end = templateStr.find('}', start);
    if (end == StringRef::npos) {
      // Malformed template, just emit rest as literal.
      StringRef literal = templateStr.substr(start);
      os << indent << "stream << R\"(" << literal << ")\";\n";
      break;
    }

    // Extract placeholder content (may be "name" or "name1:name2").
    StringRef placeholderContent =
        templateStr.substr(start + 1, end - start - 1);

    // Check for zipped pair syntax: {name1|name2}
    // Uses pipe instead of colon to avoid TableGen parsing issues with [{ }]
    // code blocks (e.g., {a:b}}] would be mis-parsed as ending at first }).
    size_t pipePos = placeholderContent.find('|');
    if (pipePos != StringRef::npos) {
      StringRef firstName = placeholderContent.substr(0, pipePos);
      StringRef secondName = placeholderContent.substr(pipePos + 1);

      auto firstIt = argIndexMap.find(firstName);
      auto secondIt = argIndexMap.find(secondName);

      if (firstIt != argIndexMap.end() && secondIt != argIndexMap.end()) {
        const ArgInfo &firstInfo = argInfos[firstIt->second];
        const ArgInfo &secondInfo = argInfos[secondIt->second];

        // Both must be Range parameters for zipped iteration.
        if (firstInfo.kind == ArgKind::Range &&
            secondInfo.kind == ArgKind::Range) {
          emitZippedRangePlaceholder(os, indent, firstName, secondName,
                                     firstInfo, secondInfo);
        } else {
          // Fallback: emit as literal if types don't match.
          os << indent << "stream << R\"({" << placeholderContent << "})\";\n";
        }
      } else {
        // Unknown args, emit as literal.
        os << indent << "stream << R\"({" << placeholderContent << "})\";\n";
      }
    } else {
      // Single placeholder: {name}
      StringRef placeholderName = placeholderContent;

      auto it = argIndexMap.find(placeholderName);
      if (it != argIndexMap.end()) {
        const ArgInfo &info = argInfos[it->second];
        if (info.kind == ArgKind::ArrayRef) {
          os << indent << "::llvm::interleaveComma(args." << placeholderName
             << ", stream);\n";
        } else if (info.kind == ArgKind::Range) {
          emitRangePlaceholder(os, indent, placeholderName, info);
        } else {
          os << indent << "stream << args." << placeholderName << ";\n";
        }
      } else {
        // Unknown placeholder - emit as literal (keep the braces).
        os << indent << "stream << R\"({" << placeholderName << "})\";\n";
      }
    }

    pos = end + 1;
  }

  os << indent << "return stream.str();\n";
}

//===----------------------------------------------------------------------===//
// Error Validation
//===----------------------------------------------------------------------===//

// Validates that all constraint error references point to defined errors.
// Returns true if validation passes, false if errors found.
static bool validateErrorReferences(const RecordKeeper &records) {
  // Find all defined errors.
  const Record *errorBaseClass = records.getClass("Util_ErrorBase");
  if (!errorBaseClass) {
    return true; // No error infrastructure present.
  }

  std::set<std::string> definedErrors;
  for (const auto &def : records.getDefs()) {
    if (def.second->isSubClassOf(errorBaseClass)) {
      StringRef errorId = def.second->getValueAsString("errorId");
      definedErrors.insert(errorId.str());
    }
  }

  // Find all constraints and validate their error references.
  const Record *constraintClass = records.getClass("Util_ConstraintBase");
  if (!constraintClass) {
    return true; // No constraints present.
  }

  bool allValid = true;
  for (const auto &def : records.getDefs()) {
    if (def.second->isSubClassOf(constraintClass)) {
      const Record *constraint = def.second.get();

      // Check if constraint has an error field.
      if (!constraint->getValue("error")) {
        continue;
      }

      const Record *errorRecord = constraint->getValueAsDef("error");
      if (!errorRecord) {
        continue;
      }

      StringRef errorId = errorRecord->getValueAsString("errorId");
      if (definedErrors.count(errorId.str()) == 0) {
        PrintError(constraint->getLoc(),
                   "Constraint '" + constraint->getName() +
                       "' references undefined error '" + errorId + "'");
        allValid = false;
      }
    }
  }

  return allValid;
}

// Collects which errors are referenced by constraints.
// Returns set of referenced error IDs.
static std::set<std::string>
collectReferencedErrors(const RecordKeeper &records) {
  std::set<std::string> referencedErrors;

  const Record *constraintClass = records.getClass("Util_ConstraintBase");
  if (!constraintClass) {
    return referencedErrors;
  }

  for (const auto &def : records.getDefs()) {
    if (def.second->isSubClassOf(constraintClass)) {
      const Record *constraint = def.second.get();

      if (!constraint->getValue("error")) {
        continue;
      }

      const Record *errorRecord = constraint->getValueAsDef("error");
      if (errorRecord) {
        StringRef errorId = errorRecord->getValueAsString("errorId");
        referencedErrors.insert(errorId.str());
      }
    }
  }

  return referencedErrors;
}

//===----------------------------------------------------------------------===//
// Error Struct Generation
//===----------------------------------------------------------------------===//

bool emitErrorStructs(const RecordKeeper &records, raw_ostream &os) {
  // Find all Util_ErrorBase records.
  const Record *errorBaseClass = records.getClass("Util_ErrorBase");
  if (!errorBaseClass) {
    return false;
  }

  std::vector<const Record *> errors;
  for (const auto &def : records.getDefs()) {
    if (def.second->isSubClassOf(errorBaseClass)) {
      errors.push_back(def.second.get());
    }
  }

  if (errors.empty()) {
    return false;
  }

  // Validate error references before generation.
  if (!validateErrorReferences(records)) {
    return false; // Errors already printed by validateErrorReferences.
  }

  // Warn about unused errors (only if --warn-unused-errors is set).
  if (warnUnusedErrors) {
    std::set<std::string> referencedErrors = collectReferencedErrors(records);
    for (const Record *error : errors) {
      StringRef errorId = error->getValueAsString("errorId");
      if (referencedErrors.count(errorId.str()) == 0) {
        PrintWarning(error->getLoc(),
                     "Error '" + errorId +
                         "' is defined but not referenced by any constraint");
      }
    }
  }

  // Extract dialect from first error.
  const Record *dialectRecord = errors[0]->getValueAsDef("errorDialect");
  if (!dialectRecord) {
    return false;
  }

  StringRef dialectName = dialectRecord->getValueAsString("name");
  StringRef dialectCppNamespace =
      dialectRecord->getValueAsString("cppNamespace");

  // Convert to proper case.
  std::string dialectNameCapitalized = dialectName.str();
  if (!dialectNameCapitalized.empty()) {
    dialectNameCapitalized[0] = std::toupper(dialectNameCapitalized[0]);
  }
  std::string dialectNameUpper = dialectName.upper();

  // Emit header guard.
  os << "#ifndef IREE_COMPILER_DIALECT_" << dialectNameUpper
     << "_IR_ERRORS_H_\n";
  os << "#define IREE_COMPILER_DIALECT_" << dialectNameUpper
     << "_IR_ERRORS_H_\n\n";

  // Emit includes.
  os << "#include \"mlir/IR/Diagnostics.h\"\n";
  os << "#include \"mlir/IR/Operation.h\"\n";
  os << "#include \"llvm/ADT/StringRef.h\"\n";
  os << "#include \"llvm/Support/raw_ostream.h\"\n";
  os << "#include <string>\n\n";

  // Emit namespace (strip leading :: if present).
  StringRef ns = dialectCppNamespace;
  if (ns.starts_with("::")) {
    ns = ns.substr(2);
  }
  os << "namespace " << ns << " {\n\n";

  // Emit CRTP ErrorBase template.
  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n";
  os << "// CRTP Error Base\n";
  os << "//"
        "===-------------------------------------------------------------------"
        "---===//\n\n";

  os << "template <typename Derived>\n";
  os << "struct ErrorBase {\n";
  os << "  // CRTP pattern - derived class provides specifics.\n";
  os << "};\n\n";

  // Emit error namespace.
  os << "namespace Errors {\n\n";

  // Emit each error struct.
  for (const Record *error : errors) {
    StringRef errorId = error->getValueAsString("errorId");
    StringRef summary = error->getValueAsString("summary");
    StringRef domain = error->getValueAsString("domain");
    StringRef fixHint = error->getValueAsString("fixHint");
    StringRef example = error->getValueAsString("example");
    StringRef severity = error->getValueAsString("severity");

    // Determine emit function based on severity.
    StringRef emitFn = "emitError";
    if (severity == "warning") {
      emitFn = "emitWarning";
    } else if (severity == "remark") {
      emitFn = "emitRemark";
    }

    // Extract arguments from DAG.
    const DagInit *schemaDag = error->getValueAsDag("schema");
    std::vector<ArgInfo> argInfos = extractErrorArgsFromDag(schemaDag);

    // Validate that all placeholders in template strings match schema args.
    // This catches TableGen parsing bugs where code blocks get corrupted.
    // Note: example field is not validated since examples are static text
    // that may contain MLIR region syntax with { } delimiters.
    StringRef message = error->getValueAsString("message");
    validatePlaceholders(error, "message", message, argInfos);
    validatePlaceholders(error, "fixHint", fixHint, argInfos);

    // Emit comment with argument documentation.
    os << "// " << errorId;
    if (!summary.empty()) {
      os << ": " << summary.trim();
    }
    os << "\n";

    // Document expected arguments to help catch mismatches at usage sites.
    if (!argInfos.empty()) {
      os << "// Expects " << argInfos.size() << " argument"
         << (argInfos.size() > 1 ? "s" : "") << ":\n";
      for (const auto &info : argInfos) {
        os << "//   " << info.cppType << " " << info.name << "\n";
      }
    } else {
      os << "// Expects no arguments.\n";
    }

    // Emit struct.
    os << "struct " << errorId << " : ErrorBase<" << errorId << "> {\n";
    os << "  static constexpr const char *id = \"" << errorId << "\";\n";
    os << "  static constexpr const char *domain = \"" << domain << "\";\n\n";

    // Emit Args struct if we have arguments.
    if (!argInfos.empty()) {
      os << "  struct Args {\n";
      for (const auto &info : argInfos) {
        os << "    " << info.cppType << " " << info.name << ";\n";
      }
      os << "  };\n\n";

      // Generate formatMessage method.
      os << "  static std::string formatMessage(const Args& args) {\n";
      emitTemplateSubstitution(os, message.trim(), "    ", argInfos);
      os << "  }\n\n";

      // Generate formatFixHint method.
      os << "  static std::string formatFixHint(const Args& args) {\n";
      if (!fixHint.empty()) {
        emitTemplateSubstitution(os, fixHint.trim(), "    ", argInfos);
      } else {
        os << "    return \"\";\n";
      }
      os << "  }\n\n";

      // Generate formatExample method.
      os << "  static std::string formatExample(const Args& args) {\n";
      if (!example.empty()) {
        emitTemplateSubstitution(os, example.trim(), "    ", argInfos);
      } else {
        os << "    return \"\";\n";
      }
      os << "  }\n\n";

      // Generate emit() method for Operation*.
      os << "  static ::mlir::InFlightDiagnostic emit(::mlir::Operation *op,\n";
      os << "                                          const Args& args) {\n";
      os << "    std::string message = formatMessage(args);\n";
      os << "    auto diag = ::mlir::" << emitFn << "(op->getLoc())\n";
      os << "      << id << \": \" << message;\n";
      os << "    \n";
      os << "    std::string fixHint = formatFixHint(args);\n";
      os << "    if (!fixHint.empty()) {\n";
      os << "      diag.attachNote() << \"Fix: \" << fixHint;\n";
      os << "    }\n";
      os << "    \n";
      os << "    std::string example = formatExample(args);\n";
      os << "    if (!example.empty()) {\n";
      os << "      diag.attachNote() << \"Example: \" << example;\n";
      os << "    }\n";
      os << "    \n";
      os << "    return diag;\n";
      os << "  }\n\n";

      // Generate emit() method for Location.
      os << "  static ::mlir::InFlightDiagnostic emit(::mlir::Location loc,\n";
      os << "                                          const Args& args) {\n";
      os << "    std::string message = formatMessage(args);\n";
      os << "    auto diag = ::mlir::" << emitFn << "(loc)\n";
      os << "      << id << \": \" << message;\n";
      os << "    \n";
      os << "    std::string fixHint = formatFixHint(args);\n";
      os << "    if (!fixHint.empty()) {\n";
      os << "      diag.attachNote() << \"Fix: \" << fixHint;\n";
      os << "    }\n";
      os << "    \n";
      os << "    std::string example = formatExample(args);\n";
      os << "    if (!example.empty()) {\n";
      os << "      diag.attachNote() << \"Example: \" << example;\n";
      os << "    }\n";
      os << "    \n";
      os << "    return diag;\n";
      os << "  }\n";
    } else {
      // No-args errors: generate emit() methods without Args parameter.

      // Generate emit() method for Operation*.
      os << "  static ::mlir::InFlightDiagnostic emit(::mlir::Operation *op) "
            "{\n";
      os << "    auto diag = ::mlir::" << emitFn << "(op->getLoc())\n";
      os << "      << id << \": \" << R\"(" << message.trim() << ")\";\n";
      if (!fixHint.empty()) {
        os << "    diag.attachNote() << \"Fix: \" << R\"(" << fixHint.trim()
           << ")\";\n";
      }
      if (!example.empty()) {
        os << "    diag.attachNote() << \"Example: \" << R\"(" << example.trim()
           << ")\";\n";
      }
      os << "    return diag;\n";
      os << "  }\n\n";

      // Generate emit() method for Location.
      os << "  static ::mlir::InFlightDiagnostic emit(::mlir::Location loc) "
            "{\n";
      os << "    auto diag = ::mlir::" << emitFn << "(loc)\n";
      os << "      << id << \": \" << R\"(" << message.trim() << ")\";\n";
      if (!fixHint.empty()) {
        os << "    diag.attachNote() << \"Fix: \" << R\"(" << fixHint.trim()
           << ")\";\n";
      }
      if (!example.empty()) {
        os << "    diag.attachNote() << \"Example: \" << R\"(" << example.trim()
           << ")\";\n";
      }
      os << "    return diag;\n";
      os << "  }\n";
    }

    os << "};\n\n";
  }

  os << "} // namespace Errors\n\n";

  // Close namespace.
  os << "} // namespace " << ns << "\n\n";

  // Close header guard.
  os << "#endif // IREE_COMPILER_DIALECT_" << dialectNameUpper
     << "_IR_ERRORS_H_\n";

  return false;
}

//===----------------------------------------------------------------------===//
// Op-Error Pair Collection for Tag Dispatch
//===----------------------------------------------------------------------===//

/// Represents an (operation, error, constraint) tuple for tag dispatch
/// generation.
struct OpErrorPair {
  const Record *op;    // Operation definition.
  const Record *error; // Error definition.
  const Record
      *constraint;         // Constraint that detected this error (may be null).
  std::string opClassName; // C++ class name (e.g., "TileBroadcastOp").
  std::string opNamespace; // C++ namespace (e.g.,
                           // "::mlir::iree_compiler::IREE::Loom").
  std::string errorId;     // Error ID (e.g., "ERR_LOOM_ENCODING_0001").
  std::string errorNamespace; // Error's dialect namespace (e.g.,
                              // "IREE::Util::Errors").
};

/// Collects all (op, error, constraint) tuples from constraint usage.
/// Only constraint-detected errors are collected, as they have well-defined
/// extraction logic in the constraint's errorArgs field.
/// Manually-declared errors in op's `errors` field are NOT included because
/// they typically require custom computed arguments from verifier logic.
static std::vector<OpErrorPair>
collectOpErrorPairs(const RecordKeeper &records) {
  std::vector<OpErrorPair> pairs;
  std::set<std::pair<std::string, std::string>>
      seen; // Dedup (op, error) pairs.

  // Find all Op records.
  const Record *opClass = records.getClass("Op");
  if (!opClass) {
    return pairs;
  }

  // Find the constraint base class.
  const Record *constraintBaseClass = records.getClass("Util_ConstraintBase");
  if (!constraintBaseClass) {
    return pairs;
  }

  for (const auto &def : records.getDefs()) {
    const Record *opRecord = def.second.get();
    if (!opRecord->isSubClassOf(opClass)) {
      continue;
    }

    // Use mlir::tblgen::Operator to get proper C++ class name and namespace.
    mlir::tblgen::Operator op(opRecord);
    std::string opClassName = op.getCppClassName().str();
    std::string opNamespace = op.getCppNamespace().str();

    // Analyze the op's traits list to find constraint traits.
    const ListInit *traits = opRecord->getValueAsListInit("traits");
    for (const Init *traitInit : traits->getElements()) {
      const DefInit *traitDef = dyn_cast<DefInit>(traitInit);
      if (!traitDef) {
        continue;
      }

      const Record *trait = traitDef->getDef();

      // Check if this trait is a constraint (subclass of Util_ConstraintBase).
      if (!trait->isSubClassOf(constraintBaseClass)) {
        continue;
      }

      // Get the error from the constraint.
      if (trait->isValueUnset("error")) {
        continue;
      }

      const Record *error = trait->getValueAsDef("error");
      std::string errorId = error->getValueAsString("errorId").str();

      // Get the error's dialect namespace to use fully-qualified error types.
      const Record *errorDialect = error->getValueAsDef("errorDialect");
      std::string errorDialectNs =
          errorDialect->getValueAsString("cppNamespace").str();
      // Build the error namespace suffix (e.g., "IREE::Util::Errors").
      std::string errorNamespace =
          getTraitNamespaceSuffix(errorDialectNs) + "::Errors";

      // Deduplicate - same error may be used by multiple constraints on same
      // op. Include namespace in key to distinguish ops with same name in
      // different namespaces.
      auto key = std::make_pair(opNamespace + "::" + opClassName, errorId);
      if (seen.count(key)) {
        continue;
      }
      seen.insert(key);

      pairs.push_back({opRecord, error, trait, opClassName, opNamespace,
                       errorId, errorNamespace});
    }
  }

  return pairs;
}

//===----------------------------------------------------------------------===//
// Tag Dispatch Generation
//===----------------------------------------------------------------------===//

bool emitOpErrorTagDispatch(const RecordKeeper &records, raw_ostream &os) {
  // Collect all (op, error, constraint) tuples.
  std::vector<OpErrorPair> pairs = collectOpErrorPairs(records);

  if (pairs.empty()) {
    return false; // No op-error pairs to generate.
  }

  // Build field type map for $field reference processing.
  StringMap<FieldInfo> fieldTypeMap = buildFieldTypeMap(records);

  // Emit header comment.
  os << "// Automatically generated tag dispatch implementations for "
        "op-specific error emission.\n";
  os << "// DO NOT EDIT - regenerate with iree-tblgen.\n\n";

  // Emit includes.
  os << "#include \"iree/compiler/Utils/Diagnostics.h\"\n";
  os << "#include \"mlir/IR/Diagnostics.h\"\n";
  os << "#include \"mlir/IR/Operation.h\"\n\n";

  // Group pairs by op namespace for consistent namespace context.
  // This allows local type names (e.g., TileType) to work within the dialect,
  // matching the behavior of constraint trait implementations.
  std::map<std::string, std::vector<const OpErrorPair *>> pairsByNamespace;
  for (const OpErrorPair &pair : pairs) {
    std::string nsSuffix = getTraitNamespaceSuffix(pair.opNamespace);
    pairsByNamespace[nsSuffix].push_back(&pair);
  }

  // Generate emitImpl overloads grouped by op namespace.
  // ADL finds these via the op type's associated namespace.
  for (const auto &[opNs, nsPairs] : pairsByNamespace) {
    os << "namespace mlir::iree_compiler::" << opNs << " {\n\n";

    for (const OpErrorPair *pairPtr : nsPairs) {
      const OpErrorPair &pair = *pairPtr;
      const Record *constraint = pair.constraint;
      if (!constraint) {
        continue; // Skip pairs without constraints.
      }

      // Validate errorArgs against the schema to catch typos/omissions early.
      // Reuses the same helper used by constraint impl generation.
      {
        std::vector<LocalBinding> locals = parseLocals(constraint);
        std::vector<ErrorArgBinding> errorArgs = parseErrorArgs(constraint);
        if (!validateErrorArgsAgainstSchema(constraint, errorArgs, locals,
                                            pair.error)) {
          continue; // Error already reported by validate; skip emission.
        }
      }

      // Parse locals and errorArgs from the constraint.
      std::vector<LocalBinding> locals = parseLocals(constraint);
      std::vector<ErrorArgBinding> errorArgs = parseErrorArgs(constraint);

      // Use unqualified op type since we're in the op's namespace.
      os << "// " << pair.opClassName << " x " << pair.errorId << "\n";
      os << "::mlir::InFlightDiagnostic emitImpl(\n";
      os << "    ::mlir::iree_compiler::ErrorTag<" << pair.errorNamespace
         << "::" << pair.errorId << ">,\n";
      os << "    " << pair.opClassName << " concreteOp) {\n";

      // Build set of referenced local names to filter unused locals.
      std::set<std::string> referencedLocals;
      for (const auto &arg : errorArgs) {
        if (arg.isLocalRef) {
          referencedLocals.insert(arg.value.substr(1)); // Strip $ prefix.
        }
      }

      // Emit local variable declarations first (these may be referenced by
      // errorArgs via $localName syntax). Only emit locals that are referenced.
      bool hasLocals = false;
      for (const auto &local : locals) {
        if (!referencedLocals.count(local.name)) {
          continue; // Skip unreferenced locals.
        }
        if (!hasLocals) {
          os << "  // Local variables computed for error args.\n";
          hasLocals = true;
        }
        std::string processedExtractor =
            processFieldReferences(local.extractor, fieldTypeMap,
                                   /*useConcreteOp=*/true);
        os << "  " << local.type << " " << local.name << " = "
           << processedExtractor << ";\n";
      }
      if (hasLocals) {
        os << "\n";
      }

      // Generate error arg variable declarations.
      // Skip declarations for local refs - we'll use the local variable
      // directly. Process $$ACCESSOR() markers and handle StringRef quoting.
      for (const ErrorArgBinding &binding : errorArgs) {
        if (binding.isLocalRef) {
          // Local reference - will use the local variable directly in Args.
          continue;
        }

        // Direct value or expression.
        std::string processedValue =
            processFieldReferences(binding.value, fieldTypeMap,
                                   /*useConcreteOp=*/true);

        // Trim leading/trailing whitespace from processed value.
        size_t start = processedValue.find_first_not_of(" \t\n\r");
        size_t end = processedValue.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
          processedValue = processedValue.substr(start, end - start + 1);
        } else {
          // All whitespace or empty - reset to empty.
          processedValue.clear();
        }

        // Validate non-empty value - empty produces broken C++ like "Type x =
        // ;".
        if (processedValue.empty()) {
          PrintFatalError(constraint->getLoc(),
                          "Empty value for error arg '" + binding.name +
                              "' in constraint '" + constraint->getName() +
                              "'");
        }

        // Quote simple identifiers for StringRef types. This handles template
        // parameter substitution like higher="result" -> StringRef higherName =
        // "result". Expressions containing operators, dots, or parentheses are
        // NOT quoted (the isalnum check excludes them). Local variable
        // references ($localName) are handled earlier via isLocalRef. If a
        // constraint author wants to pass a variable, they should define it as
        // a local first.
        bool needsQuoting = false;
        if (binding.type == "StringRef" && !processedValue.empty()) {
          // Check if it's a simple identifier: only alphanumeric and
          // underscore. Expressions with . ( ) :: etc. will fail this check
          // and not be quoted.
          needsQuoting = true;
          for (char c : processedValue) {
            if (!std::isalnum(c) && c != '_') {
              needsQuoting = false;
              break;
            }
          }
        }

        // Handle ArrayRef types with brace-initialized values.
        // ArrayRef<T> cannot be directly brace-initialized because the backing
        // array is a temporary. We need to declare a storage array first.
        StringRef typeRef(binding.type);
        bool isArrayRef = typeRef.starts_with("ArrayRef<") &&
                          !processedValue.empty() && processedValue[0] == '{';

        if (isArrayRef) {
          // Extract element type from ArrayRef<ElementType>.
          StringRef elementType =
              typeRef.drop_front(9).drop_back(1); // Remove "ArrayRef<" and ">".
          os << "  " << elementType << " " << binding.name
             << "Storage[] = " << processedValue << ";\n";
          os << "  " << binding.type << " " << binding.name << "("
             << binding.name << "Storage);\n";
        } else {
          os << "  " << binding.type << " " << binding.name << " = ";
          if (needsQuoting) {
            os << "\"" << processedValue << "\"";
          } else {
            os << processedValue;
          }
          os << ";\n";
        }
      }

      // Emit the error using the extracted values.
      os << "  return " << pair.errorNamespace << "::" << pair.errorId
         << "::emit(concreteOp.getOperation(),\n";
      os << "    " << pair.errorNamespace << "::" << pair.errorId
         << "::Args{\n";
      for (size_t i = 0; i < errorArgs.size(); ++i) {
        os << "      ." << errorArgs[i].name << " = ";
        if (errorArgs[i].isLocalRef) {
          // Use the local variable directly (strip $ prefix).
          os << errorArgs[i].value.substr(1);
        } else {
          os << errorArgs[i].name;
        }
        if (i + 1 < errorArgs.size()) {
          os << ",";
        }
        os << "\n";
      }
      os << "    });\n";
      os << "}\n\n";
    }

    os << "} // namespace mlir::iree_compiler::" << opNs << "\n\n";
  }

  return false;
}

} // namespace mlir::iree_compiler::tblgen

// Register the error generators.
static mlir::GenRegistration
    genErrorStructs("gen-iree-error-structs",
                    "Generate C++ error structs from error definitions",
                    mlir::iree_compiler::tblgen::emitErrorStructs);

static mlir::GenRegistration genOpErrorTagDispatch(
    "gen-iree-op-error-tag-dispatch",
    "Generate tag dispatch implementations for op-specific error emission",
    mlir::iree_compiler::tblgen::emitOpErrorTagDispatch);
