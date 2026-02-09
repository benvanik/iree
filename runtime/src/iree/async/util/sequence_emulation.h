// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_UTIL_SEQUENCE_EMULATION_H_
#define IREE_ASYNC_UTIL_SEQUENCE_EMULATION_H_

#include "iree/async/operation.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Sequence emulation
//===----------------------------------------------------------------------===//

// Callback used by the sequence emulator to submit the next step operation.
// The backend provides this during initialization so the emulator can submit
// individual step operations through the backend's normal submission path.
//
// |proactor| is the proactor to submit through.
// |operation| is the next step to submit (already prepared by the emulator).
//
// Returns OK if submission succeeded, or an error that will abort the sequence.
typedef iree_status_t (*iree_async_sequence_submit_fn_t)(
    iree_async_proactor_t* proactor, iree_async_operation_t* operation);

// Drives sequence operations step-by-step for backends that lack kernel-level
// operation chaining (linked SQEs).
//
// ## Why this exists
//
// io_uring can chain multiple operations as linked SQEs: the kernel executes
// them in order with zero user-space round-trips between steps. This is the
// optimal path for pre-planned pipelines like "wait → recv → signal."
//
// Other backends (kqueue, IOCP, threaded emulation) have no kernel-level
// chaining. They must execute sequence steps one at a time, returning to user
// space between each step. This utility provides that stepping logic so that
// each backend doesn't need to reimplement it.
//
// ## How it works
//
//   Backend receives a SEQUENCE operation via submit()
//     → Backend calls iree_async_sequence_emulation_begin()
//       → Emulator submits step 0 via the submit callback
//         → Backend's normal operation processing handles step 0
//           → Step 0 completes
//             → Backend calls iree_async_sequence_emulation_step_completed()
//               → Emulator calls step_fn (if set) for inter-step logic
//               → Emulator submits step 1 via the submit callback
//                 → ... continues until all steps complete ...
//                   → Emulator fires the sequence's base callback with OK
//
//   If any step fails:
//     → Emulator fires the sequence's base callback with that error
//     → Remaining steps are never submitted
//
//   If step_fn returns an error:
//     → Same as step failure — sequence aborts with that error
//
// ## Step operation callbacks
//
// The emulator overwrites each step operation's completion_fn and user_data
// before submitting it, pointing them at internal trampolines. The original
// step callbacks (if any) are NOT invoked — the sequence's base callback is
// the only external-facing callback.
//
// This is by design: sequence steps are internal pipeline stages, not
// independently observable operations. The sequence as a whole has exactly
// one completion callback, fired exactly once.
//
// ## Backend integration
//
// Backends that don't support linked SQEs use this in two places:
//
//   1. In submit(): detect SEQUENCE type, call _begin() instead of
//      submitting directly.
//
//   2. In the completion path: the emulator's internal trampolines call
//      _step_completed() automatically — the backend doesn't need to know
//      it's inside a sequence. The trampolines are regular completion
//      callbacks that the backend dispatches like any other.
//
// ## Thread safety
//
// Not thread-safe. All calls must happen from the same thread (the proactor
// thread). This is naturally satisfied because begin() is called from
// submit() and step_completed() is called from poll()-dispatched callbacks.
//
// ## Memory
//
// The emulator stores no per-sequence state beyond what's already in the
// iree_async_sequence_operation_t struct. It uses the sequence's current_step
// field to track progress and overwrites step callbacks in-place. No
// additional allocation is needed.
typedef struct iree_async_sequence_emulator_t {
  // The proactor that owns this emulator (for submitting subsequent steps).
  iree_async_proactor_t* proactor;

  // Backend-provided function for submitting individual step operations.
  iree_async_sequence_submit_fn_t submit_fn;
} iree_async_sequence_emulator_t;

// Initializes a sequence emulator.
// |proactor| is the owning proactor (used as context for submit calls).
// |submit_fn| is the backend's submission function for individual operations.
//
// The emulator struct is typically embedded in the backend's proactor
// implementation struct (zero additional allocation).
static inline void iree_async_sequence_emulator_initialize(
    iree_async_sequence_emulator_t* emulator, iree_async_proactor_t* proactor,
    iree_async_sequence_submit_fn_t submit_fn) {
  emulator->proactor = proactor;
  emulator->submit_fn = submit_fn;
}

// Begins execution of a sequence operation.
// Submits step 0 and sets up internal trampolines for step advancement.
// The sequence's current_step is reset to 0.
//
// The caller (backend submit path) must not touch the sequence or its steps
// after this call — the emulator owns them until the base callback fires.
//
// Returns OK if step 0 was submitted successfully, or the submission error
// if it failed (in which case the sequence's base callback fires with that
// error before this function returns).
iree_status_t iree_async_sequence_emulation_begin(
    iree_async_sequence_emulator_t* emulator,
    iree_async_sequence_operation_t* sequence);

// Called by the emulator's internal trampolines when a step completes.
// Advances the sequence to the next step, calling step_fn if present.
//
// Backend code does NOT call this directly — it is invoked through the
// completion callback trampolines that _begin() installs on each step.
// This function is exposed in the header for testing purposes only.
void iree_async_sequence_emulation_step_completed(
    iree_async_sequence_emulator_t* emulator,
    iree_async_sequence_operation_t* sequence, iree_status_t step_status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_SEQUENCE_EMULATION_H_
