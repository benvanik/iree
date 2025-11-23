# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TableGen generator implementations for IREE.
# These source files are compiled directly into iree-tblgen (in tools/)
# rather than as a separate library, because TableGen tools must be
# statically linked against internal LLVM libraries.
# See tools/CMakeLists.txt for the iree-tblgen target definition.

iree_add_all_subdirs()
