// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/sequence_emulation.h"

// TODO: implement sequence_emulation.

iree_status_t iree_async_sequence_emulation_begin(
    iree_async_sequence_emulator_t* emulator,
    iree_async_sequence_operation_t* sequence) {
  (void)emulator;
  (void)sequence;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "sequence_emulation not yet implemented");
}

void iree_async_sequence_emulation_step_completed(
    iree_async_sequence_emulator_t* emulator,
    iree_async_sequence_operation_t* sequence, iree_status_t step_status) {
  (void)emulator;
  (void)sequence;
  iree_status_ignore(step_status);
}
