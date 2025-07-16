// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_QUEUE_ENTRY_FLAG_NONE_H_
#define IREE_HAL_DRIVERS_AMDGPU_QUEUE_ENTRY_FLAG_NONE_H_

#include "iree/hal/drivers/amdgpu/device/command_buffer.h"
#include "iree/hal/drivers/amdgpu/device/semaphore.h"
#include "iree/hal/drivers/amdgpu/device/support/common.h"

//===----------------------------------------------------------------------===//
// Queue Entries
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_queue_entry_header_t
    iree_hal_amdgpu_queue_entry_header_t;

// Queue entry type indicating the type and size of the arguments.
typedef uint8_t iree_hal_amdgpu_queue_entry_type_t;
enum iree_hal_amdgpu_queue_entry_type_e {
  IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_ALLOCA = 0,
  IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_DEALLOCA,
  IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_FILL,
  IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_UPDATE,
  IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_COPY,
  IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_READ,
  IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_WRITE,
  IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_EXECUTE,
  IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_BARRIER,
  IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_HOST_CALL,
};

// Flags indicating how queue entries are to be processed.
typedef uint16_t iree_hal_amdgpu_queue_entry_flags_t;
enum iree_hal_amdgpu_queue_entry_flag_bits_e {
  IREE_HAL_AMDGPU_QUEUE_ENTRY_FLAG_NONE = 0u,
};

// Header for all scheduler queue entries.
// Each entry contains a set of fixed information used when initially enqueuing
// it in the incoming scheduler mailbox and dynamic information maintained by
// the scheduler as the entry moves through its state machine.
typedef struct iree_hal_amdgpu_queue_entry_header_t {
  // Type of the queue entry used to issue the operation.
  iree_hal_amdgpu_queue_entry_type_t type;

  // Index into the active set the entry has been assigned while active.
  // This may change over the lifetime of the entry if it is made active
  // multiple times (such as after yielding).
  uint8_t active_bit_index;

  // Flags controlling queue entry behavior.
  iree_hal_amdgpu_queue_entry_flags_t flags;

  // Monotonically increasing value with lower values indicating entries that
  // were enqueued first. This is used to ensure FIFO execution ordering when
  // inserting into the run list. Assigned by the scheduler when accepting the
  // entry as there may be multiple producers and the epoch should be
  // scheduler-specific.
  uint32_t epoch;

  // Maximum number of bytes of the execution kernarg ringbuffer are required.
  // The entry will stall before issuing until capacity is available.
  // Must be aligned to IREE_HAL_AMDGPU_DEVICE_KERNARG_ALIGNMENT.
  uint32_t max_kernarg_capacity;

  // Whether the queue entry was allocated from the small (0) or large (1) block
  // pool.
  // TODO(benvanik): find another way to derive this bit. If we had size buckets
  // instead of just small/large we'd need more bits for that. The alternative
  // is storing a 64-bit pointer back to the pool and that feels excessive.
  uint32_t allocation_pool;

  // Host-side iree_hal_amdgpu_block_token_t from the queue entry allocation.
  // TODO(benvanik): move the allocator to be shared with the device-side
  // library so we can free entries from the device.
  uint64_t allocation_token;

  // Allocated absolute kernarg ringbuffer offset of max_kernarg_capacity bytes.
  // May be suballocated by the entry. Use
  // iree_hal_amdgpu_device_kernarg_ringbuffer_resolve to get the pointer.
  // UINT64_MAX indicates no kernargs are used (as would
  // max_kernarg_capacity=0).
  uint64_t kernarg_offset;

  // Semaphores that must be signaled before the queue entry is issued.
  // Semaphores will be removed from the list as they complete.
  // The semaphores must remain valid for the lifetime of the queue entry.
  iree_hal_amdgpu_device_semaphore_list_t* wait_list;

  // Semaphores to be signaled when the queue entry completes.
  // Semaphores that can be signaled on the device will be removed from the list
  // while any host-only semaphores (ones external to the HAL implementation or
  // that need a host callback) will remain for the host-side entry retirement
  // to handle.
  // The semaphores must remain valid for the lifetime of the queue entry.
  iree_hal_amdgpu_device_semaphore_list_t* signal_list;

  // Host-side iree_hal_resource_set_t tracking all resources (including wait
  // and signal semaphores) used by this entry. All will be kept live until the
  // entry is retired on the host.
  uint64_t resource_set;

  // Intrusive pointer used when the entry is in a linked list (wait list, run
  // list, etc).
  iree_hal_amdgpu_queue_entry_header_t* list_next;
} iree_hal_amdgpu_queue_entry_header_t;
static_assert(sizeof(iree_hal_amdgpu_queue_entry_header_t) <= 64,
              "queue entries should be kept as small as possible; avoid adding "
              "to the fixed header struct that increases the size of all "
              "entries unless it is something used by all entry types");

#define IREE_HAL_AMDGPU_QUEUE_RETIRE_ENTRY_KERNARG_SIZE (2 * sizeof(uint64_t))

typedef struct iree_hal_amdgpu_queue_alloca_entry_t {
  // Type: IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_ALLOCA
  iree_hal_amdgpu_queue_entry_header_t header;
  iree_hal_amdgpu_device_allocation_pool_id_t pool_id;
  uint32_t min_alignment;
  uint64_t allocation_size;
  iree_hal_amdgpu_device_allocation_handle_t* handle;
} iree_hal_amdgpu_queue_alloca_entry_t;
#define IREE_HAL_AMDGPU_QUEUE_ALLOCA_MAX_KERNARG_CAPACITY \
  (IREE_HAL_AMDGPU_QUEUE_RETIRE_ENTRY_KERNARG_SIZE)

typedef struct iree_hal_amdgpu_queue_dealloca_entry_t {
  // Type: IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_DEALLOCA
  iree_hal_amdgpu_queue_entry_header_t header;
  iree_hal_amdgpu_device_allocation_handle_t* handle;
} iree_hal_amdgpu_queue_dealloca_entry_t;
#define IREE_HAL_AMDGPU_QUEUE_DEALLOCA_MAX_KERNARG_CAPACITY \
  (IREE_HAL_AMDGPU_QUEUE_RETIRE_ENTRY_KERNARG_SIZE)

typedef struct iree_hal_amdgpu_queue_fill_entry_t {
  // Type: IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_FILL
  iree_hal_amdgpu_queue_entry_header_t header;
  iree_hal_amdgpu_device_buffer_ref_t target_ref;
  uint64_t pattern;
  uint8_t pattern_length;
} iree_hal_amdgpu_queue_fill_entry_t;
#define IREE_HAL_AMDGPU_QUEUE_FILL_MAX_KERNARG_CAPACITY \
  (IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE +    \
   IREE_HAL_AMDGPU_QUEUE_RETIRE_ENTRY_KERNARG_SIZE)

typedef struct iree_hal_amdgpu_queue_update_entry_t {
  // Type: IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_UPDATE
  iree_hal_amdgpu_queue_entry_header_t header;
  const IREE_AMDGPU_DEVICE_PTR void* source_ptr;
  iree_hal_amdgpu_device_buffer_ref_t target_ref;
} iree_hal_amdgpu_queue_update_entry_t;
#define IREE_HAL_AMDGPU_QUEUE_UPDATE_MAX_KERNARG_CAPACITY \
  (IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE +      \
   IREE_HAL_AMDGPU_QUEUE_RETIRE_ENTRY_KERNARG_SIZE)

typedef struct iree_hal_amdgpu_queue_copy_entry_t {
  // Type: IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_COPY
  iree_hal_amdgpu_queue_entry_header_t header;
  iree_hal_amdgpu_device_buffer_ref_t source_ref;
  iree_hal_amdgpu_device_buffer_ref_t target_ref;
} iree_hal_amdgpu_queue_copy_entry_t;
#define IREE_HAL_AMDGPU_QUEUE_COPY_MAX_KERNARG_CAPACITY \
  (IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE +    \
   IREE_HAL_AMDGPU_QUEUE_RETIRE_ENTRY_KERNARG_SIZE)

typedef struct iree_hal_amdgpu_queue_read_entry_t {
  // Type: IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_READ
  iree_hal_amdgpu_queue_entry_header_t header;
  uint64_t queue_affinity;
  uint64_t source_file;  // iree_hal_file_t*
  uint64_t source_offset;
  iree_hal_amdgpu_device_buffer_ref_t target_ref;
} iree_hal_amdgpu_queue_read_entry_t;
#define IREE_HAL_AMDGPU_QUEUE_READ_KERNARG_SIZE (2 * sizeof(uint64_t))
#define IREE_HAL_AMDGPU_QUEUE_READ_MAX_KERNARG_CAPACITY \
  (IREE_HAL_AMDGPU_QUEUE_READ_KERNARG_SIZE +            \
   IREE_HAL_AMDGPU_QUEUE_RETIRE_ENTRY_KERNARG_SIZE)

typedef struct iree_hal_amdgpu_queue_write_entry_t {
  // Type: IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_WRITE
  iree_hal_amdgpu_queue_entry_header_t header;
  uint64_t queue_affinity;
  iree_hal_amdgpu_device_buffer_ref_t source_ref;
  uint64_t target_file;  // iree_hal_file_t*
  uint64_t target_offset;
} iree_hal_amdgpu_queue_write_entry_t;
#define IREE_HAL_AMDGPU_QUEUE_WRITE_KERNARG_SIZE (2 * sizeof(uint64_t))
#define IREE_HAL_AMDGPU_QUEUE_WRITE_MAX_KERNARG_CAPACITY \
  (IREE_HAL_AMDGPU_QUEUE_WRITE_KERNARG_SIZE +            \
   IREE_HAL_AMDGPU_QUEUE_RETIRE_ENTRY_KERNARG_SIZE)

typedef struct iree_hal_amdgpu_queue_execute_entry_t {
  // Type: IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_EXECUTE
  iree_hal_amdgpu_queue_entry_header_t header;
  // Execution control flags.
  iree_hal_amdgpu_device_execution_flags_t flags;
  // Command buffer being executed.
  const iree_hal_amdgpu_device_command_buffer_t* command_buffer;
  // State used during command buffer execution. Mutated in-place.
  // Enqueuers only need to populate bindings.
  //
  // TODO(benvanik): move this to ensure device-local memory instead of sharing
  // with the queue storage. We'd need to have the bindings here and then
  // replicate them. It would allow us to remove the allocation_handle
  // dereferences from the command issue path by doing it once at entry issue.
  iree_hal_amdgpu_device_execution_state_t state;
} iree_hal_amdgpu_queue_execute_entry_t;
#define IREE_HAL_AMDGPU_QUEUE_EXECUTE_MAX_KERNARG_CAPACITY( \
    command_buffer_max_kernarg_capacity)                    \
  (IREE_HAL_AMDGPU_DEVICE_EXECUTION_CONTROL_KERNARG_SIZE +  \
   (command_buffer_max_kernarg_capacity))

typedef struct iree_hal_amdgpu_queue_barrier_entry_t {
  // Type: IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_BARRIER
  iree_hal_amdgpu_queue_entry_header_t header;
} iree_hal_amdgpu_queue_barrier_entry_t;
#define IREE_HAL_AMDGPU_QUEUE_BARRIER_MAX_KERNARG_CAPACITY 0

typedef struct iree_hal_amdgpu_queue_host_call_entry_t {
  // Type: IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_HOST_CALL
  iree_hal_amdgpu_queue_entry_header_t header;
  // Flags defining call behavior.
  iree_hal_amdgpu_host_call_flags_t flags;
} iree_hal_amdgpu_queue_host_call_entry_t;
#define IREE_HAL_AMDGPU_QUEUE_HOST_CALL_MAX_KERNARG_CAPACITY \
  (2 * sizeof(uint64_t))

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// TODO(benvanik): implement device queue entry logic.

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_QUEUE_ENTRY_FLAG_NONE_H_
