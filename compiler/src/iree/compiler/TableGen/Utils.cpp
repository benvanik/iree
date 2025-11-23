// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Utils.h"

#include <set>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/TableGen/Error.h"
#include "mlir/TableGen/Operator.h"

using namespace llvm;

namespace mlir::iree_compiler::tblgen {

//===----------------------------------------------------------------------===//
// String Processing
//===----------------------------------------------------------------------===//

std::string cleanDescription(StringRef desc) {
  if (desc.empty()) {
    return "";
  }

  SmallVector<StringRef> lines;
  desc.split(lines, '\n');

  // Find minimum indentation (excluding empty lines).
  // Use std::optional to avoid npos sentinel value bugs.
  std::optional<size_t> minIndent;
  for (StringRef line : lines) {
    size_t firstNonWS = line.find_first_not_of(" \t");
    if (firstNonWS != StringRef::npos) {
      // Non-empty line found.
      if (!minIndent || firstNonWS < *minIndent) {
        minIndent = firstNonWS;
      }
    }
  }

  // If no non-empty lines or indentation is zero, return original.
  if (!minIndent || *minIndent == 0) {
    return desc.str();
  }

  // Strip the common indentation from all lines.
  std::string result;
  for (size_t i = 0; i < lines.size(); ++i) {
    StringRef line = lines[i];
    if (line.find_first_not_of(" \t") == StringRef::npos) {
      // Empty/whitespace-only line - preserve it.
      if (i > 0) {
        result += '\n';
      }
    } else {
      // Non-empty line - strip common indentation.
      if (line.size() < *minIndent) {
        // Line is shorter than expected indentation - bail cleanly.
        // This shouldn't happen if logic is correct, but handle defensively.
        return desc.str();
      }
      if (i > 0) {
        result += '\n';
      }
      result += line.substr(*minIndent).str();
    }
  }

  // Trim leading/trailing blank lines.
  while (!result.empty() && result.front() == '\n') {
    result.erase(0, 1);
  }
  while (!result.empty() && result.back() == '\n') {
    result.pop_back();
  }

  return result;
}

std::string cleanAssemblyFormat(StringRef format) {
  if (format.empty()) {
    return "";
  }

  std::string result = format.str();
  // Replace all newlines with spaces.
  for (char &c : result) {
    if (c == '\n') {
      c = ' ';
    }
  }

  // Collapse multiple spaces into single space.
  size_t writePos = 0;
  bool lastWasSpace = false;
  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i] == ' ') {
      if (!lastWasSpace) {
        result[writePos++] = ' ';
        lastWasSpace = true;
      }
    } else {
      result[writePos++] = result[i];
      lastWasSpace = false;
    }
  }
  result.resize(writePos);

  // Trim leading/trailing spaces.
  while (!result.empty() && result.front() == ' ') {
    result.erase(0, 1);
  }
  while (!result.empty() && result.back() == ' ') {
    result.pop_back();
  }

  return result;
}

std::string resolveRelativePath(StringRef relativePath,
                                const Record *definingRecord) {
  if (relativePath.empty()) {
    return "";
  }

  // Get source location of the record (the .td file).
  ArrayRef<SMLoc> locs = definingRecord->getLoc();
  if (locs.empty() || !locs[0].isValid()) {
    // No valid location - can't resolve, return original path.
    return relativePath.str();
  }

  // Find buffer containing this location.
  unsigned bufferID = SrcMgr.FindBufferContainingLoc(locs[0]);
  if (bufferID == 0) {
    // Buffer not found - return original path.
    return relativePath.str();
  }

  // Get .td file path from buffer.
  const MemoryBuffer *buffer = SrcMgr.getMemoryBuffer(bufferID);
  if (!buffer) {
    // Defensive: buffer should exist if bufferID != 0, but check anyway.
    return relativePath.str();
  }
  StringRef tdFilePath = buffer->getBufferIdentifier();

  // Find repository root by walking up from .td file directory.
  // The repository root is assumed to be the directory containing
  // "compiler/src/".
  SmallString<256> currentDir(tdFilePath);
  sys::path::remove_filename(currentDir); // Start from .td file's directory.

  SmallString<256> repoRoot;
  bool foundRoot = false;

  // Walk up the directory tree looking for "compiler/src/".
  while (!currentDir.empty() && currentDir != "/") {
    SmallString<256> compilerSrcPath(currentDir);
    sys::path::append(compilerSrcPath, "compiler", "src");

    // Check if this directory contains "compiler/src/".
    bool isDir = false;
    std::error_code ec = sys::fs::is_directory(compilerSrcPath, isDir);
    if (!ec && isDir) {
      // Found the repository root.
      repoRoot = currentDir;
      foundRoot = true;
      break;
    }
    // If error checking directory, continue walking up (graceful degradation).

    // Move up one directory.
    sys::path::remove_filename(currentDir);
  }

  // Fallback: if root-finding failed, try extracting from .td file path.
  if (!foundRoot) {
    // Try to find "compiler/src/" in the .td file path itself.
    StringRef tdFileStr(tdFilePath);
    size_t compilerSrcPos = tdFileStr.find("compiler/src/");
    if (compilerSrcPos != StringRef::npos) {
      // Extract everything before "compiler/src/" as repo root.
      repoRoot = tdFileStr.substr(0, compilerSrcPos);
      foundRoot = true;
    } else {
      // Could not determine repository root - return original path unchanged.
      // This is acceptable for non-IREE projects or different layouts.
      return relativePath.str();
    }
  }

  // Resolve relative path against repository root.
  SmallString<256> absolutePath(repoRoot);
  sys::path::append(absolutePath, relativePath);

  // Normalize to remove . and ..
  sys::path::remove_dots(absolutePath, /*remove_dot_dot=*/true);

  return std::string(absolutePath.str());
}

//===----------------------------------------------------------------------===//
// Source Location
//===----------------------------------------------------------------------===//

void emitSourceLocation(const Record *def, const RecordKeeper &records,
                        json::OStream &J) {
  (void)records; // Not needed - using global SrcMgr.

  // Get the primary location (first element of location array).
  ArrayRef<SMLoc> locs = def->getLoc();
  if (locs.empty() || !locs[0].isValid()) {
    // No location available - skip emitting sourceLocation attribute.
    return;
  }

  SMLoc loc = locs[0];

  // Find which buffer contains this location.
  unsigned bufferID = SrcMgr.FindBufferContainingLoc(loc);
  if (bufferID == 0) {
    // Location not associated with a buffer - skip sourceLocation.
    return;
  }

  // Get the file path from buffer.
  const MemoryBuffer *buffer = SrcMgr.getMemoryBuffer(bufferID);
  if (!buffer) {
    // Defensive: buffer should exist if bufferID != 0, but check anyway.
    return;
  }
  StringRef filename = buffer->getBufferIdentifier();

  // Get line and column numbers.
  std::pair<unsigned, unsigned> lineCol =
      SrcMgr.getLineAndColumn(loc, bufferID);

  // Emit as JSON.
  J.attributeObject("sourceLocation", [&] {
    J.attribute("file", filename);
    J.attribute("line", lineCol.first);
    J.attribute("column", lineCol.second);
  });
}

bool emitLineDirective(raw_ostream &os, SMLoc loc) {
  if (!loc.isValid()) {
    return false;
  }

  unsigned bufferID = SrcMgr.FindBufferContainingLoc(loc);
  if (bufferID == 0) {
    return false;
  }

  const MemoryBuffer *buffer = SrcMgr.getMemoryBuffer(bufferID);
  if (!buffer) {
    return false;
  }

  StringRef filename = buffer->getBufferIdentifier();
  std::pair<unsigned, unsigned> lineCol =
      SrcMgr.getLineAndColumn(loc, bufferID);

  // Emit #line directive pointing to where the field value was defined.
  os << "#line " << lineCol.first << " \"" << filename << "\"\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Field Processing
//===----------------------------------------------------------------------===//

StringMap<FieldInfo> buildFieldTypeMap(const RecordKeeper &records) {
  StringMap<FieldInfo> fieldMap;

  // Iterate through all MLIR operations to collect field type information.
  for (const auto &defPair : records.getDefs()) {
    const Record *record = defPair.second.get();

    // Check if this is an Op definition.
    if (!record->isSubClassOf("Op")) {
      continue;
    }

    // Use mlir::tblgen::Operator to access ODS metadata.
    mlir::tblgen::Operator op(record);

    // Scan operands.
    for (const auto &operand : op.getOperands()) {
      FieldInfo info;
      info.name = operand.name.str();
      info.isVariadic = operand.isVariadic();
      // Determine kind and C++ type.
      if (operand.isVariadic()) {
        info.kind = FieldKind::Variadic;
        info.cppType = "::mlir::ValueRange";
      } else {
        info.kind = FieldKind::Value;
        info.cppType = "::mlir::Value";
      }
      // Use ODS to get the accessor name - same function MLIR uses.
      info.accessorName = op.getGetterName(operand.name);
      fieldMap[operand.name] = info;
    }

    // Scan results.
    for (const auto &result : op.getResults()) {
      FieldInfo info;
      info.name = result.name.str();
      info.isVariadic = result.isVariadic();
      // Determine kind and C++ type.
      if (result.isVariadic()) {
        info.kind = FieldKind::Variadic;
        info.cppType = "::mlir::ValueRange";
      } else {
        info.kind = FieldKind::Value;
        info.cppType = "::mlir::Value";
      }
      // Use ODS to get the accessor name - same function MLIR uses.
      info.accessorName = op.getGetterName(result.name);
      fieldMap[result.name] = info;
    }

    // Scan attributes.
    for (const auto &attr : op.getAttributes()) {
      FieldInfo info;
      info.name = attr.name.str();
      info.kind = FieldKind::Attribute;
      // Use the specific attribute type (e.g., ArrayAttr, IntegerAttr)
      // from ODS instead of generic ::mlir::Attribute.
      info.cppType = attr.attr.getReturnType().str();
      if (info.cppType.empty()) {
        info.cppType = "::mlir::Attribute"; // Fallback if not specified.
      }
      info.isVariadic = false;
      // Use ODS to get the accessor name - same function MLIR uses.
      info.accessorName = op.getGetterName(attr.name);
      fieldMap[attr.name] = info;
    }

    // Scan regions.
    for (const auto &region : op.getRegions()) {
      FieldInfo info;
      info.name = region.name.str();
      info.isVariadic = region.isVariadic();
      if (region.isVariadic()) {
        info.kind = FieldKind::VariadicRegion;
        info.cppType = "::mlir::MutableArrayRef<::mlir::Region>";
      } else {
        info.kind = FieldKind::Region;
        info.cppType = "::mlir::Region&";
      }
      // Use ODS to get the accessor name - same function MLIR uses.
      info.accessorName = op.getGetterName(region.name);
      fieldMap[region.name] = info;
    }

    // Scan successors.
    for (const auto &successor : op.getSuccessors()) {
      FieldInfo info;
      info.name = successor.name.str();
      info.isVariadic = successor.isVariadic();
      if (successor.isVariadic()) {
        info.kind = FieldKind::VariadicSuccessor;
        info.cppType = "::mlir::SuccessorRange";
      } else {
        info.kind = FieldKind::Successor;
        info.cppType = "::mlir::Block*";
      }
      // Use ODS to get the accessor name - same function MLIR uses.
      info.accessorName = op.getGetterName(successor.name);
      fieldMap[successor.name] = info;
    }

    // Scan properties.
    for (const auto &prop : op.getProperties()) {
      FieldInfo info;
      info.name = prop.name.str();
      info.kind = FieldKind::Property;
      // Use interface type for properties, falling back to storage type.
      info.cppType = prop.prop.getInterfaceType().str();
      if (info.cppType.empty()) {
        info.cppType = prop.prop.getStorageType().str();
      }
      info.isVariadic = false;
      // Use ODS to get the accessor name - same function MLIR uses.
      info.accessorName = op.getGetterName(prop.name);
      fieldMap[prop.name] = info;
    }
  }

  return fieldMap;
}

std::vector<std::string> extractFieldNames(StringRef code) {
  std::vector<std::string> fieldNames;

  size_t pos = 0;
  while (pos < code.size()) {
    // Find next '$'.
    pos = code.find('$', pos);
    if (pos == StringRef::npos) {
      break;
    }

    // Skip escaped $$ or standalone $.
    if (pos + 1 >= code.size()) {
      break;
    }

    // Check for $$ (skip, not a field reference).
    if (code[pos + 1] == '$') {
      pos += 2;
      continue;
    }

    // Extract field name: $[a-zA-Z_][a-zA-Z0-9_]*
    size_t startPos = pos + 1;
    if (!std::isalpha(code[startPos]) && code[startPos] != '_') {
      pos = startPos;
      continue;
    }

    size_t endPos = startPos + 1;
    while (endPos < code.size() &&
           (std::isalnum(code[endPos]) || code[endPos] == '_')) {
      ++endPos;
    }

    StringRef fieldName = code.substr(startPos, endPos - startPos);

    // Check if we've already seen this field name.
    if (std::find(fieldNames.begin(), fieldNames.end(), fieldName.str()) ==
        fieldNames.end()) {
      fieldNames.push_back(fieldName.str());
    }

    pos = endPos;
  }

  return fieldNames;
}

std::string processFieldReferences(StringRef code,
                                   const StringMap<FieldInfo> &fieldTypeMap,
                                   bool useConcreteOp) {
  std::string result = code.str();

  size_t pos = 0;
  while (pos < result.size()) {
    // Find next '$'.
    pos = result.find('$', pos);
    if (pos == std::string::npos) {
      break;
    }

    // Skip escaped $$.
    if (pos + 1 < result.size() && result[pos + 1] == '$') {
      pos += 2;
      continue;
    }

    // Extract field name: $[a-zA-Z_][a-zA-Z0-9_]*
    size_t startPos = pos + 1;
    if (startPos >= result.size() ||
        (!std::isalpha(result[startPos]) && result[startPos] != '_')) {
      ++pos;
      continue;
    }

    size_t endPos = startPos + 1;
    while (endPos < result.size() &&
           (std::isalnum(result[endPos]) || result[endPos] == '_')) {
      ++endPos;
    }

    std::string fieldName = result.substr(startPos, endPos - startPos);

    // Look up field info in the map.
    std::string replacement;
    auto it = fieldTypeMap.find(fieldName);
    if (it != fieldTypeMap.end()) {
      if (useConcreteOp) {
        // Tag dispatch: $field -> concreteOp.getField()
        replacement = "concreteOp." + it->second.accessorName + "()";
      } else {
        // Check function: $field -> field (parameter name)
        replacement = fieldName;
      }
    } else {
      // Field not in map - fall back to camelCase conversion.
      if (useConcreteOp) {
        std::string accessorName =
            "get" + convertToCamelFromSnakeCase(fieldName,
                                                /*capitalizeFirst=*/true);
        replacement = "concreteOp." + accessorName + "()";
      } else {
        replacement = fieldName;
      }
    }

    result.replace(pos, endPos - pos, replacement);
    pos += replacement.size();
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Binding Parsing
//===----------------------------------------------------------------------===//

std::tuple<std::string, std::string, std::string>
parseBindingArgs(const DagInit *bindingDag) {
  if (!bindingDag || bindingDag->getNumArgs() < 3) {
    return {"", "", ""};
  }

  // Extract name (first arg).
  std::string name;
  if (const auto *nameInit = dyn_cast<StringInit>(bindingDag->getArg(0))) {
    name = nameInit->getValue().str();
  }

  // Extract type (second arg).
  std::string type;
  if (const auto *typeInit = dyn_cast<StringInit>(bindingDag->getArg(1))) {
    type = typeInit->getValue().str();
  }

  // Extract value/extractor (third arg).
  // Handle StringInit, VarInit (for $localRef), and other Init types.
  std::string value;
  const Init *valueInit = bindingDag->getArg(2);
  if (const auto *strInit = dyn_cast<StringInit>(valueInit)) {
    value = strInit->getValue().str();
  } else if (const auto *varInit = dyn_cast<VarInit>(valueInit)) {
    // VarInit represents $variable syntax in TableGen.
    // Preserve the $ prefix to mark it as a local variable reference.
    value = "$" + varInit->getName().str();
  } else {
    // For other Init types (like integer literals, DefInit, etc.),
    // use print() as a fallback.
    if (valueInit) {
      raw_string_ostream OS(value);
      valueInit->print(OS);
    }
  }

  return {name, type, value};
}

std::vector<LocalBinding> parseLocals(const Record *constraint) {
  std::vector<LocalBinding> locals;

  // Get the locals DAG.
  const DagInit *localsDag = constraint->getValueAsDag("locals");
  if (!localsDag || localsDag->getNumArgs() == 0) {
    return locals;
  }

  // Iterate over each binding in the DAG.
  std::set<std::string> seenNames;
  for (unsigned i = 0; i < localsDag->getNumArgs(); ++i) {
    const Init *argInit = localsDag->getArg(i);

    // Each arg should be a DAG: (local "name", "type", extractor)
    if (const auto *bindingDag = dyn_cast<DagInit>(argInit)) {
      auto [name, type, extractor] = parseBindingArgs(bindingDag);
      if (!name.empty() && !type.empty()) {
        // Check for empty extractor.
        if (extractor.empty()) {
          PrintError(constraint->getLoc(),
                     "Local binding '" + name + "' has an empty extractor");
          continue;
        }
        // Check for duplicate names.
        if (seenNames.count(name)) {
          PrintError(constraint->getLoc(),
                     "Duplicate local variable name '" + name + "'");
          continue;
        }
        seenNames.insert(name);
        locals.push_back({name, type, extractor});
      }
    }
  }

  return locals;
}

std::vector<ErrorArgBinding> parseErrorArgs(const Record *constraint) {
  std::vector<ErrorArgBinding> errorArgs;

  // Get the errorArgs DAG.
  const DagInit *errorArgsDag = constraint->getValueAsDag("errorArgs");
  if (!errorArgsDag || errorArgsDag->getNumArgs() == 0) {
    return errorArgs;
  }

  // Iterate over each binding in the DAG.
  std::set<std::string> seenNames;
  for (unsigned i = 0; i < errorArgsDag->getNumArgs(); ++i) {
    const Init *argInit = errorArgsDag->getArg(i);

    // Each arg should be a DAG: (arg "name", "type", value)
    if (const auto *bindingDag = dyn_cast<DagInit>(argInit)) {
      auto [name, type, value] = parseBindingArgs(bindingDag);
      if (!name.empty() && !type.empty()) {
        // Check for duplicate names.
        if (seenNames.count(name)) {
          PrintError(constraint->getLoc(),
                     "Duplicate error argument name '" + name + "'");
          continue;
        }
        seenNames.insert(name);

        // Check if value is a simple $localRef (e.g., "$actualRank").
        // Field expressions like "$field.getType()" are NOT local refs.
        bool isLocalRef = false;
        if (!value.empty() && value[0] == '$' && value.size() > 1) {
          // Check if the rest is a simple identifier (no dots, parens, etc.)
          isLocalRef = true;
          for (size_t j = 1; j < value.size(); ++j) {
            if (!std::isalnum(value[j]) && value[j] != '_') {
              isLocalRef = false;
              break;
            }
          }
        }
        errorArgs.push_back({name, type, value, isLocalRef});
      }
    }
  }

  return errorArgs;
}

bool validateErrorArgsAgainstSchema(
    const Record *constraint, const std::vector<ErrorArgBinding> &errorArgs,
    const std::vector<LocalBinding> &locals, const Record *errorRecord) {
  // Build set of local variable names for quick lookup.
  std::set<std::string> localNames;
  for (const auto &local : locals) {
    localNames.insert(local.name);
  }

  // Extract schema from error.
  const DagInit *schemaDag = errorRecord->getValueAsDag("schema");
  if (!schemaDag) {
    // No schema defined - nothing to validate against.
    return true;
  }

  // Build map of expected argument names and types from schema.
  std::map<std::string, std::string> schemaTypes;
  for (unsigned i = 0; i < schemaDag->getNumArgs(); ++i) {
    StringRef argName = schemaDag->getArgNameStr(i);
    const Init *argInit = schemaDag->getArg(i);

    std::string cppType;
    if (const auto *SI = dyn_cast<StringInit>(argInit)) {
      cppType = SI->getValue().str();
    } else if (const auto *DI = dyn_cast<DefInit>(argInit)) {
      const Record *paramDef = DI->getDef();
      if (paramDef->getValue("cppType")) {
        cppType = paramDef->getValueAsString("cppType").str();
      }
    }

    if (!cppType.empty()) {
      schemaTypes[argName.str()] = cppType;
    }
  }

  // Validate each errorArg binding against schema.
  bool allValid = true;
  for (const auto &binding : errorArgs) {
    auto it = schemaTypes.find(binding.name);
    if (it == schemaTypes.end()) {
      PrintError(constraint->getLoc(),
                 "Constraint '" + constraint->getName() +
                     "' errorArgs binding '" + binding.name +
                     "' does not match any argument in error schema");
      allValid = false;
      continue;
    }

    // Type checking disabled to avoid false positives.
    // String-based type comparison is too fragile for complex types
    // (templates, const qualifiers, namespace variants).
    // The C++ compiler will catch actual type mismatches.

    // Validate $localRef references.
    if (binding.isLocalRef) {
      std::string localName = binding.value.substr(1); // Strip '$'.
      if (localNames.find(localName) == localNames.end()) {
        PrintError(constraint->getLoc(),
                   "Constraint '" + constraint->getName() +
                       "' errorArgs binding '" + binding.name +
                       "' references undefined local variable '" + localName +
                       "'");
        allValid = false;
      }
    }
  }

  // Check for missing required arguments.
  for (const auto &[schemaName, schemaType] : schemaTypes) {
    bool found = false;
    for (const auto &binding : errorArgs) {
      if (binding.name == schemaName) {
        found = true;
        break;
      }
    }
    if (!found) {
      PrintError(constraint->getLoc(), "Constraint '" + constraint->getName() +
                                           "' errorArgs missing argument '" +
                                           schemaName +
                                           "' required by error schema");
      allValid = false;
    }
  }

  return allValid;
}

//===----------------------------------------------------------------------===//
// Namespace Handling
//===----------------------------------------------------------------------===//

std::string getTraitNamespaceSuffix(StringRef cppNamespace) {
  // Strip leading "::mlir::" if present.
  if (cppNamespace.starts_with("::mlir::")) {
    cppNamespace = cppNamespace.drop_front(8);
  }

  // Look for "IREE::" - if found, use that suffix (preserves hierarchy).
  size_t ireePos = cppNamespace.find("IREE::");
  if (ireePos != StringRef::npos) {
    return cppNamespace.substr(ireePos).str();
  }

  // Otherwise take last component or use whole thing.
  size_t lastColons = cppNamespace.rfind("::");
  if (lastColons != StringRef::npos) {
    return cppNamespace.substr(lastColons + 2).str();
  }

  return cppNamespace.str();
}

std::optional<std::string> extractDialectFromNamespace(StringRef fqn) {
  // Try IREE-specific pattern: ::mlir::iree_compiler::IREE::DialectName::
  StringRef ireePrefix = "::mlir::iree_compiler::IREE::";
  if (fqn.starts_with(ireePrefix)) {
    StringRef afterPrefix = fqn.substr(ireePrefix.size());
    size_t nextColon = afterPrefix.find("::");
    if (nextColon != StringRef::npos) {
      return afterPrefix.substr(0, nextColon).lower();
    }
  }

  // Try simple MLIR pattern: ::mlir::DialectName::ClassName
  StringRef mlirPrefix = "::mlir::";
  if (fqn.starts_with(mlirPrefix)) {
    StringRef afterMlir = fqn.substr(mlirPrefix.size());
    size_t firstColon = afterMlir.find("::");
    if (firstColon != StringRef::npos) {
      StringRef candidate = afterMlir.substr(0, firstColon);
      // Make sure there's another :: after this (i.e., ClassName exists).
      StringRef afterCandidate = afterMlir.substr(firstColon + 2);
      if (!afterCandidate.empty() &&
          afterCandidate.find("::") == StringRef::npos) {
        // Pattern matches: ::mlir::DialectName::ClassName
        return candidate.lower();
      }
    }
  }

  // Generic fallback: extract second-to-last namespace component.
  // Find last :: to get ClassName, then find previous :: for DialectName.
  size_t lastColons = fqn.rfind("::");
  if (lastColons != StringRef::npos && lastColons > 0) {
    StringRef beforeLast = fqn.substr(0, lastColons);
    size_t secondLastColons = beforeLast.rfind("::");
    if (secondLastColons != StringRef::npos) {
      StringRef dialectName = beforeLast.substr(secondLastColons + 2);
      if (!dialectName.empty()) {
        return dialectName.lower();
      }
    }
  }

  // Could not extract dialect name - return empty.
  return std::nullopt;
}

} // namespace mlir::iree_compiler::tblgen
