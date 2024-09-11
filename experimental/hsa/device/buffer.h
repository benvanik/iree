// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_DEVICE_BUFFER_H_
#define IREE_HAL_DRIVERS_HSA_DEVICE_BUFFER_H_

#include "experimental/hsa/device/kernel.h"
#include "experimental/hsa/device/support/opencl.h"

//===----------------------------------------------------------------------===//
// iree_amdgpu_allocation_handle_t
//===----------------------------------------------------------------------===//

// DO NOT SUBMIT
// host side allocates (or pools) these and iree_hal_buffer_t refs them
// host free of hal buffer would enqueue device dealloca
// can have size
// union struct for pool storage (bucket base, etc)
typedef struct iree_amdgpu_allocation_handle_t {
  void* ptr;
  // pool it was allocated from?
  // block it was allocated from?
} iree_amdgpu_allocation_handle_t;

//===----------------------------------------------------------------------===//
// iree_amdgpu_buffer_ref_t
//===----------------------------------------------------------------------===//

// Identifies the type of a buffer reference and how it should be resolved.
typedef uint8_t iree_amdgpu_buffer_type_t;
enum iree_amdgpu_buffer_type_e {
  // Reference is to an absolute device pointer that can be directly accessed.
  IREE_AMDGPU_BUFFER_TYPE_PTR = 0u,
  // Reference is to a queue-ordered allocation handle that is only valid at
  // the time the buffer is committed. The handle will be valid for the lifetime
  // of the logical buffer and any resources referencing it but the pointer must
  // only be resolved between a corresponding alloca/dealloca.
  IREE_AMDGPU_BUFFER_TYPE_HANDLE,
  // Reference is to a slot in the binding table provided during execution.
  // Only one indirection is allowed (table slots cannot reference other slots
  // - yet).
  IREE_AMDGPU_BUFFER_TYPE_SLOT,
};

// The ordinal of a slot in the binding table.
typedef uint32_t iree_amdgpu_buffer_ordinal_t;

// Describes a subrange of a buffer that can be bound to a binding slot.
typedef IREE_CL_ALIGNAS(32) struct iree_amdgpu_buffer_ref_t {
  // Buffer being bound to the slot.
  iree_amdgpu_buffer_type_t type;
  uint8_t reserved[7];  // for padding
  // Offset, in bytes, into the buffer that the binding starts at.
  // This will be added to the offset specified on each usage of the slot.
  uint64_t offset;
  // Length, in bytes, of the buffer that is available to the executable.
  uint64_t length;
  union {
    // IREE_AMDGPU_BUFFER_TYPE_PTR: raw device pointer.
    IREE_CL_GLOBAL void* ptr;
    // IREE_AMDGPU_BUFFER_TYPE_HANDLE: queue-ordered allocation handle.
    IREE_CL_GLOBAL iree_amdgpu_allocation_handle_t* handle;
    // IREE_AMDGPU_BUFFER_TYPE_SLOT: binding table slot.
    iree_amdgpu_buffer_ordinal_t slot;
  } value;
} iree_amdgpu_buffer_ref_t;
IREE_CL_STATIC_ASSERT(sizeof(iree_amdgpu_buffer_ref_t) == 32,
                      "binding table entries should be 8 byte aligned");

// DO NOT SUBMIT
// test code size with linkage - we don't want a function call per resolve but
// also don't want to duplicate this 100 times

// Resolves a buffer reference to an absolute device pointer.
// Expects that the binding table is provided if needed and has sufficient
// capacity for any slot that may be referenced. All queue-ordered allocations
// that may be provided via allocation handles must be committed prior to
// attempting to resolve them and must remain committed until all commands using
// the returned device pointer have completed.
static inline IREE_CL_GLOBAL void* iree_amdgpu_buffer_ref_resolve(
    iree_amdgpu_buffer_ref_t buffer_ref,
    IREE_CL_ALIGNAS(64)
        const iree_amdgpu_buffer_ref_t* IREE_CL_RESTRICT binding_table) {
  if (buffer_ref.type == IREE_AMDGPU_BUFFER_TYPE_SLOT) {
    const iree_amdgpu_buffer_ref_t binding =
        binding_table[buffer_ref.value.slot];
    buffer_ref.type = binding.type;
    buffer_ref.offset += binding.offset;
    buffer_ref.length = buffer_ref.length == UINT64_MAX
                            ? binding.length - buffer_ref.offset
                            : buffer_ref.length;
    buffer_ref.value = binding.value;
  }
  if (buffer_ref.type == IREE_AMDGPU_BUFFER_TYPE_HANDLE) {
    buffer_ref.value.ptr = buffer_ref.value.handle->ptr;
  }
  if (buffer_ref.value.ptr == NULL) {
    return NULL;
  }
  return (IREE_CL_GLOBAL const uint8_t*)buffer_ref.value.ptr +
         buffer_ref.offset;
}

//===----------------------------------------------------------------------===//
// Blit Kernels
//===----------------------------------------------------------------------===//

#define IREE_AMDGPU_BUFFER_FILL_KERNARG_SIZE (3 * sizeof(void*))
#define IREE_AMDGPU_BUFFER_COPY_KERNARG_SIZE (3 * sizeof(void*))

#if defined(IREE_CL_TARGET_DEVICE__)

// Enqueues a fill dispatch packet in the target queue.
// The packet will be acquired at the current write_index and the queue doorbell
// will be signaled.
void iree_amdgpu_buffer_fill_enqueue(
    IREE_CL_GLOBAL void* target_ptr, const uint64_t length,
    const uint64_t pattern, const uint8_t pattern_length,
    IREE_CL_GLOBAL const iree_amdgpu_kernels_t* kernels,
    IREE_CL_GLOBAL uint64_t* IREE_CL_RESTRICT kernarg_ptr,
    IREE_CL_GLOBAL iree_amdgpu_queue_t* IREE_CL_RESTRICT queue);

// Emplaces a fill dispatch packet in the target queue at the given index.
// The queue doorbell will not be signaled.
void iree_amdgpu_buffer_fill_emplace(
    IREE_CL_GLOBAL void* target_ptr, const uint64_t length,
    const uint64_t pattern, const uint8_t pattern_length,
    IREE_CL_GLOBAL const iree_amdgpu_kernels_t* kernels,
    IREE_CL_GLOBAL uint64_t* IREE_CL_RESTRICT kernarg_ptr,
    IREE_CL_GLOBAL iree_amdgpu_queue_t* IREE_CL_RESTRICT queue,
    const uint32_t queue_index);

// Enqueues a copy dispatch packet in the target queue.
// The packet will be acquired at the current write_index and the queue doorbell
// will be signaled.
void iree_amdgpu_buffer_copy_enqueue(
    IREE_CL_GLOBAL const void* source_ptr, IREE_CL_GLOBAL void* target_ptr,
    const uint64_t length, IREE_CL_GLOBAL const iree_amdgpu_kernels_t* kernels,
    IREE_CL_GLOBAL uint64_t* IREE_CL_RESTRICT kernarg_ptr,
    IREE_CL_GLOBAL iree_amdgpu_queue_t* IREE_CL_RESTRICT queue);

// Emplaces a copy dispatch packet in the target queue at the given index.
// The queue doorbell will not be signaled.
void iree_amdgpu_buffer_copy_emplace(
    IREE_CL_GLOBAL const void* source_ptr, IREE_CL_GLOBAL void* target_ptr,
    const uint64_t length, IREE_CL_GLOBAL const iree_amdgpu_kernels_t* kernels,
    IREE_CL_GLOBAL uint64_t* IREE_CL_RESTRICT kernarg_ptr,
    IREE_CL_GLOBAL iree_amdgpu_queue_t* IREE_CL_RESTRICT queue,
    const uint32_t queue_index);

#endif  // IREE_CL_TARGET_DEVICE__

#endif  // IREE_HAL_DRIVERS_HSA_DEVICE_BUFFER_H_
