// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_DEVICE_HOST_H_
#define IREE_HAL_DRIVERS_HSA_DEVICE_HOST_H_

#include "experimental/hsa/device/allocator.h"
#include "experimental/hsa/device/support/opencl.h"
#include "experimental/hsa/device/support/queue.h"
#include "experimental/hsa/device/support/signal.h"

//===----------------------------------------------------------------------===//
// iree_amdgpu_host_t
//===----------------------------------------------------------------------===//

enum iree_amdgpu_host_call_e {
  // Host will route to iree_hal_hsa_pool_grow.
  //
  // Signature:
  //   arg0: iree_hal_hsa_pool_t* pool
  //   arg1: block?
  //   arg2: uint64_t allocation_size
  //   arg3: uint32_t allocation_offset (offset into block)
  //         uint32_t min_alignment
  //   return_address: iree_amdgpu_allocation_handle_t* handle
  //   completion_signal: signaled when the pool has grown
  IREE_AMDGPU_HOST_CALL_POOL_GROW = 0u,

  // Host will route to iree_hal_hsa_pool_trim.
  //
  // Signature:
  //   arg0: iree_hal_hsa_pool_t* pool
  //   arg1: block?
  //   arg2:
  //   arg3:
  //   return_address:
  //   completion_signal: signaled when the pool has been trimmed
  IREE_AMDGPU_HOST_CALL_POOL_TRIM,

  // Host will call iree_hal_resource_release on each non-NULL resource pointer.
  //
  // Signature:
  //   arg0: iree_hal_resource_t* resource0
  //   arg1: iree_hal_resource_t* resource1
  //   arg2: iree_hal_resource_t* resource2
  //   arg3: iree_hal_resource_t* resource3
  //   return_address: unused
  IREE_AMDGPU_HOST_CALL_POST_RELEASE,

  // Host will mark the device as lost and start returning failures.
  // The provided code and arguments will be included in the failure messages.
  //
  // Signature:
  //   arg0: uint64_t reserved 0
  //   arg1: uint64_t code
  //   arg2: uint64_t error-specific arg0
  //   arg3: uint64_t error-specific arg1
  //   return_address: unused
  IREE_AMDGPU_HOST_CALL_POST_ERROR,

  // Host will notify any registered listeners of the semaphore signal.
  //
  // Signature:
  //   arg0: iree_amdgpu_semaphore_t* semaphore
  //   arg1: uint64_t payload
  //   arg2: unused
  //   arg3: unused
  //   return_address: unused
  IREE_AMDGPU_HOST_CALL_POST_SIGNAL,

  // DO NOT SUBMIT
  // leaf vs nested
  // leaf is type, srcloc, timestamp start, timestamp end
  // nested is begin: type, id, srcloc, timestamp, end: id, timestamp
  IREE_AMDGPU_HOST_CALL_POST_TRACE_ZONE_EVENT,
};

// Represents the host runtime that is managing a single queue.
// The host can be used to perform
typedef struct iree_amdgpu_host_t {
  // DO NOT SUBMIT
  // queue/agent
  int reserved;
} iree_amdgpu_host_t;

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

#if defined(IREE_CL_TARGET_DEVICE)

// Enqueues a host agent packet.
// The completion signal is optional and may be 0.
void iree_amdgpu_host_enqueue(
    IREE_CL_GLOBAL iree_amdgpu_host_t* IREE_CL_RESTRICT host, uint16_t type,
    uint64_t return_address, uint64_t arg0, uint64_t arg1, uint64_t arg2,
    uint64_t arg3, iree_hsa_signal_t completion_signal);

// Posts a multi-resource release request to the host.
// The host will call iree_hal_resource_release on each non-NULL resource
// pointer provided.
void iree_amdgpu_host_post_release(
    IREE_CL_GLOBAL iree_amdgpu_host_t* IREE_CL_RESTRICT host,
    uint64_t resource0, uint64_t resource1, uint64_t resource2,
    uint64_t resource3);

// Posts an error code to the host.
// The provided arguments are appended to the error message emitted.
// After posting an error it may not be possible to continue execution and the
// device is considered "lost".
void iree_amdgpu_host_post_error(
    IREE_CL_GLOBAL iree_amdgpu_host_t* IREE_CL_RESTRICT host, uint64_t code,
    uint64_t arg0, uint64_t arg1);

// Posts a semaphore signal notification to the host.
// The order is not guaranteed and by the time the host processes the message
// the semaphore may have already advanced past the specified payload value.
void iree_amdgpu_host_post_signal(
    IREE_CL_GLOBAL iree_amdgpu_host_t* IREE_CL_RESTRICT host,
    IREE_CL_GLOBAL iree_amdgpu_semaphore_t* IREE_CL_RESTRICT semaphore,
    uint64_t payload);

#endif  // IREE_CL_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_HSA_DEVICE_HOST_H_
