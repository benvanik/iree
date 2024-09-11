// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_DEVICE_SCHEDULER_H_
#define IREE_HAL_DRIVERS_HSA_DEVICE_SCHEDULER_H_

#include "experimental/hsa/device/allocator.h"
#include "experimental/hsa/device/command_buffer.h"
#include "experimental/hsa/device/support/opencl.h"
#include "experimental/hsa/device/support/queue.h"
#include "experimental/hsa/device/support/signal.h"

//===----------------------------------------------------------------------===//
// DO NOT SUBMIT
//===----------------------------------------------------------------------===//

// Queue entry type indicating the type and size of the arguments.
typedef uint8_t iree_amdgpu_queue_entry_type_t;
enum iree_amdgpu_queue_entry_type_e {
  IREE_AMDGPU_QUEUE_ENTRY_TYPE_INITIALIZE = 0,
  IREE_AMDGPU_QUEUE_ENTRY_TYPE_DEINITIALIZE,
  IREE_AMDGPU_QUEUE_ENTRY_TYPE_ALLOCA,
  IREE_AMDGPU_QUEUE_ENTRY_TYPE_DEALLOCA,
  IREE_AMDGPU_QUEUE_ENTRY_TYPE_FILL,
  IREE_AMDGPU_QUEUE_ENTRY_TYPE_COPY,
  IREE_AMDGPU_QUEUE_ENTRY_TYPE_EXECUTE,
  IREE_AMDGPU_QUEUE_ENTRY_TYPE_BARRIER,
};

// Flags indicating how queue entries are to be processed.
typedef uint16_t iree_amdgpu_queue_entry_flags_t;
enum iree_amdgpu_queue_entry_flag_bits_e {
  IREE_AMDGPU_DEVICE_QUEUE_ENTRY_FLAG_NONE = 0u,
};

typedef struct IREE_CL_ALIGNAS(64) iree_amdgpu_queue_entry_header_t {
  iree_amdgpu_queue_entry_type_t type;

  uint8_t reserved0;

  iree_amdgpu_queue_entry_flags_t flags;

  uint8_t reserved[4];

  // Optional HAL semaphore reference.
  // If populated then the HAL semaphore will be signaled to the given payload.
  // The HAL semaphore may
  IREE_CL_GLOBAL iree_amdgpu_semaphore_t* signal_semaphore;
  uint64_t signal_payload;
} iree_amdgpu_queue_entry_header_t;

typedef struct iree_amdgpu_queue_initialize_args_t {
  iree_amdgpu_queue_entry_header_t header;

  // Total number of available signals. Must be a power-of-two.
  uint32_t signal_count;
  // Allocated signals used for the signal pool.
  // Storage and signals must remain valid for the lifetime of the scheduler.
  IREE_CL_GLOBAL iree_hsa_signal_t* signals;
} iree_amdgpu_queue_initialize_args_t;

typedef struct iree_amdgpu_queue_deinitialize_args_t {
  iree_amdgpu_queue_entry_header_t header;
  // DO NOT SUBMIT
} iree_amdgpu_queue_deinitialize_args_t;

typedef struct iree_amdgpu_queue_alloca_args_t {
  iree_amdgpu_queue_entry_header_t header;
  uint32_t pool;
  uint32_t min_alignment;
  uint64_t allocation_size;
  IREE_CL_GLOBAL iree_amdgpu_allocation_handle_t* handle;
} iree_amdgpu_queue_alloca_args_t;

typedef struct iree_amdgpu_queue_dealloca_args_t {
  iree_amdgpu_queue_entry_header_t header;
  IREE_CL_GLOBAL iree_amdgpu_allocation_handle_t* handle;
} iree_amdgpu_queue_dealloca_args_t;

typedef struct iree_amdgpu_queue_fill_args_t {
  iree_amdgpu_queue_entry_header_t header;
  iree_amdgpu_buffer_ref_t target_ref;
  uint64_t pattern;
  uint8_t pattern_length;
} iree_amdgpu_queue_fill_args_t;

typedef struct iree_amdgpu_queue_copy_args_t {
  iree_amdgpu_queue_entry_header_t header;
  iree_amdgpu_buffer_ref_t source_ref;
  iree_amdgpu_buffer_ref_t target_ref;
} iree_amdgpu_queue_copy_args_t;

typedef struct iree_amdgpu_queue_execute_args_t {
  iree_amdgpu_queue_entry_header_t header;
  IREE_CL_GLOBAL iree_amdgpu_execution_state_t* state;
  // binding table stored in state
} iree_amdgpu_queue_execute_args_t;

typedef struct iree_amdgpu_queue_barrier_args_t {
  iree_amdgpu_queue_entry_header_t header;
} iree_amdgpu_queue_barrier_args_t;

//===----------------------------------------------------------------------===//
// DO NOT SUBMIT
//===----------------------------------------------------------------------===//

// device->host
//
// resource_set reclaim
// arg0: resource set
// arg1:
// arg2:
// arg3:
// return:
// no completion signal? (unidirectional)
//
// error
// arg0: status
// arg1: fail arg0
// arg2: fail arg1
// arg3: fail arg2
// return: HAL semaphore?
//
// or just notify timepoint reached?
// hal semaphore ptr
// bit to indicate whether needed? device->device semaphores
// IREE_HAL_SEMAPHORE_FLAG_<<DEVICE_LOCAL>> (same agent)
// only fire if not device local

// semaphores <-> signals
// semaphore unpack to signal
// may want to feed back completion?
//   signal immediately, allow device->device
//   post to host to allow it to clear anything required on the semaphore
//
// want waiters/callbacks on semaphore?
// ordered linked list of waiters
// could add waiter for reclaiming resources/etc
// timepoint?
//   prev/next (for insertion)
//   payload
//   user_data
//   callback
//
// signal pool
// iree_amdgpu_signal_pool_t
//   capacity
//   signals
//   free_list (atomic ops to acquire/release)
// iree_amdgpu_signal_pool_acquire
// iree_amdgpu_signal_pool_release
//
// could have one in device memory just for device-side signals?
// device: used in command buffers and such
// host: device->host signals
//
//
// parking lot
// if work enqueued on HAL semaphore that does not have a pending signal to the
// work value then need to park?
// to start ignore
// could use timepoints on waiters - add timepoint to each to decremental signal
//
// completion_signal is always a decrement
// barrier ops wait for == 0
// so need timepoint == HSA signal for CP
// can still use signal with payload to map to semaphore?
// or always have scheduler insert
// hsa_amd_barrier_value_packet_t for timeline wait
// can't do AND/OR
// if multiple semaphores use multiple packets?
//
// want HSA_AMD_SIGNAL_AMD_GPU_ONLY
// uses DefaultSignal which is just memory-based
// if not set then need an InterruptSignal, which goes through KFD events
// needed for non-local events?
// or cpu?
// https://github.com/ROCm/ROCR-Runtime/blob/ec4bb54b01c26b44eff96468884cc3cd040a27f7/runtime/hsa-runtime/core/runtime/hsa_ext_amd.cpp#L511
//
//
// may still need soft queue for our own queue
// if yielding then need to keep the pipeline full
//
// soft queue is:
//   hsa soft queue - signal + ringbuffer
//   parking lot?
// host/device puts requests in queue
// scheduler kick - could do conditionally if not pending
// may always want to kick to avoid races
// otherwise may need atomics in both directions (chatty)
// hw queue runs scheduler and then other things the scheduler fills with
// if device yields then kick must happen
//   scheduler: dequeue execute cb w/ 2 chunks
//     enqueue cb chunk 0 (could run inline to avoid latency)
//       dispatches[]
//       tail cb chunk 1
//     enqueue cb chunk 1
//       dispatches[]
//       tail scheduler kick
//   scheduler

// split into device-side and host-side? host encompasses all, but keep cache
// lines separate? don't want ringbuffer atomics doing PCI transactions

typedef struct iree_amdgpu_queue_scheduler_t {
  // Host agent used to perform services at the request of the device runtime.
  // May be shared with multiple schedulers.
  IREE_CL_GLOBAL iree_amdgpu_host_t* host;

  // Device-side allocator.
  // May be shared with multiple schedulers but always represents device-local
  // memory.
  IREE_CL_GLOBAL iree_amdgpu_allocator_t* allocator;

  // Queue used for launching the top-level scheduler after execution completes.
  IREE_CL_GLOBAL iree_amdgpu_queue_t* scheduler_queue;

  // Queue used for command buffer execution.
  // This may differ from the top-level scheduling queue.
  //
  // TODO(benvanik): allow multiple queues? We could allow multiple command
  // buffers to issue/execute concurrently so long as their dependencies are
  // respected.
  IREE_CL_GLOBAL iree_amdgpu_queue_t* execution_queue;

  // Pool of HSA signals that can be used by device code.
  // The pool will be used by the scheduler as well as various subsystems to
  // get signals as they are opaque objects that must have been allocated on the
  // host. Note that when the pool is exhausted the scheduler will abort.
  IREE_CL_GLOBAL iree_amdgpu_signal_pool_t* signal_pool;

  // ringbuffer for args
  // incoming queue
  // current packet (copied from queue)?
  // cached kernarg buffer (and other info) for rescheduling

  // Handles to opaque kernel objects used to dispatch builtin kernels.
  iree_amdgpu_kernels_t kernels;
} iree_amdgpu_queue_scheduler_t;

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

#define IREE_AMDGPU_QUEUE_SCHEDULER_KERNARG_SIZE (3 * sizeof(void*))

// Indicates why the scheduler has been enqueued.
typedef uint8_t iree_amdgpu_queue_scheduling_reason_t;
enum iree_amdgpu_queue_scheduling_reason_e {
  // Scheduler is enqueued as new work is available for execution.
  // Note that by the time the scheduler runs all of the work may have been
  // processed.
  IREE_AMDGPU_QUEUE_SCHEDULING_REASON_WORK_AVAILABLE = 0u,
  // Scheduler is enqueued after a command buffer has completed execution.
  // The reason_arg passed to the kernel is the iree_amdgpu_execution_state_t
  // of the command buffer that is returning.
  IREE_AMDGPU_QUEUE_SCHEDULING_REASON_COMMAND_BUFFER_RETURN,
};

#if defined(IREE_CL_TARGET_DEVICE)

// Enqueues a scheduler tick on the scheduling queue.
void iree_amdgpu_queue_scheduler_enqueue(
    IREE_CL_GLOBAL iree_amdgpu_queue_scheduler_t* IREE_CL_RESTRICT scheduler,
    iree_amdgpu_queue_scheduling_reason_t reason, uint64_t reason_arg);

#endif  // IREE_CL_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_HSA_DEVICE_SCHEDULER_H_
