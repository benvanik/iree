# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "dim_op.mlir"
    "dim_op_fold.mlir"
    "dim_op_invalid.mlir"
    "slice_op.mlir"
    "slice_op_fold.mlir"
    "slice_op_invalid.mlir"
    "update_op.mlir"
    "update_op_fold.mlir"
    "update_op_invalid.mlir"
  TOOLS
    FileCheck
    iree-opt
)
