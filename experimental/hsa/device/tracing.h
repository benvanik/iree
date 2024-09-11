// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_DEVICE_TRACING_H_
#define IREE_HAL_DRIVERS_HSA_DEVICE_TRACING_H_

#include "experimental/hsa/device/support/opencl.h"
#include "experimental/hsa/device/support/ringbuffer.h"

//===----------------------------------------------------------------------===//
// iree_amdgpu_tracing_context_t
//===----------------------------------------------------------------------===//

// DO NOT SUBMIT
//   uint32_t type: begin_internal | end_internal
//   uint32_t srcloc or id
//   uint64_t timestamp

// ZONE_BEGIN_INTERNAL
//   uint8_t type;  BEGIN_INTERNAL
//   uint8_t reserved;
//   uint16_t queue_id;
//   uint64_t src_loc;
//   uint64_t timestamp;

// ZONE_END_INTERNAL
//   uint8_t type;  END_INTERNAL
//   uint8_t reserved;
//   uint16_t queue_id;
//   uint64_t reserved;
//   uint64_t timestamp;

// iree_amdgpu_zone_id_t zone_id = iree_amdgpu_trace_zone_begin(ctx, loc);
// ...
// iree_amdgpu_trace_zone_end(ctx, zone_id);

// table of functions?
// QUEUE_ISSUE_FOO -> "queue_issue_foo"

// choose fixed size
// use aql agent queue?
// then wasting 64b
// dedicated tracing queue?

// dispatch timings?
// may overlap

//===----------------------------------------------------------------------===//
// Device-side API
//===----------------------------------------------------------------------===//

#if defined(IREE_CL_TARGET_DEVICE)

// Initializes the signal pool with the given set of HSA signals.
// The signals must remain valid for the lifetime of the pool.
void iree_amdgpu_signal_pool_initialize(
    IREE_CL_GLOBAL iree_amdgpu_signal_pool_t* IREE_CL_RESTRICT signal_pool,
    uint32_t signal_count,
    IREE_CL_GLOBAL iree_hsa_signal_t* IREE_CL_RESTRICT signals);

// Acquires a binary signal with an initial value as specified.
// If the pool is exhausted the returned signal will have a 0 value handle and
// callers should check with `iree_hsa_signal_is_null(signal)`.
iree_hsa_signal_t iree_amdgpu_signal_pool_acquire(
    IREE_CL_GLOBAL iree_amdgpu_signal_pool_t* IREE_CL_RESTRICT signal_pool,
    int64_t initial_value);

// Returns a signal to the pool.
// Only signals acquired from the pool may be released.
void iree_amdgpu_signal_pool_release(
    IREE_CL_GLOBAL iree_amdgpu_signal_pool_t* IREE_CL_RESTRICT signal_pool,
    iree_hsa_signal_t signal);

#endif  // IREE_CL_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_HSA_DEVICE_TRACING_H_
