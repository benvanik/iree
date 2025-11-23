// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE TableGen tool.
// Generator implementations are in compiler/src/iree/compiler/TableGen/.
// VM generators are in compiler/src/iree/compiler/Dialect/VM/Tools/.

#include "llvm/TableGen/Record.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"

// Debug generator that prints all records.
static mlir::GenRegistration genPrintRecords(
    "print-records", "Print all records to stdout",
    [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
      os << records;
      return false;
    });

int main(int argc, char **argv) { return mlir::MlirTblgenMain(argc, argv); }
