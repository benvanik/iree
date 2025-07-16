// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue.h"

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/threading.h"
#include "iree/hal/drivers/amdgpu/buffer_pool.h"
#include "iree/hal/drivers/amdgpu/device/queue_entry.h"
#include "iree/hal/drivers/amdgpu/system.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_host_queue_t
//===----------------------------------------------------------------------===//

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable;

typedef struct iree_hal_amdgpu_host_queue_t {
  iree_hal_amdgpu_virtual_queue_t base;

  // Optional callback issued when an asynchronous queue error occurs.
  iree_hal_amdgpu_error_callback_t error_callback;

  // Service used for host operations.
  // This queue is host-based but we still use the host service as there may be
  // other devices that have device-side queues and we want to be consistent.
  iree_hal_amdgpu_host_service_t* host_service;

  // OS handle to the worker thread.
  iree_thread_t* thread;

  // internal queue/ring

  // hsa queue

  // tracking of last signaled semaphores (somehow)

} iree_hal_amdgpu_host_queue_t;

static int iree_hal_amdgpu_host_queue_main(void* entry_arg);

static iree_hal_amdgpu_host_queue_t* iree_hal_amdgpu_host_queue_cast(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue) {
  IREE_ASSERT_ARGUMENT(virtual_queue);
  IREE_ASSERT_EQ(virtual_queue->vtable, &iree_hal_amdgpu_host_queue_vtable);
  return (iree_hal_amdgpu_host_queue_t*)virtual_queue;
}

iree_host_size_t iree_hal_amdgpu_host_queue_calculate_size(
    const iree_hal_amdgpu_queue_options_t* options) {
  IREE_ASSERT_EQ(options->placement, IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST);
  // TODO(benvanik): factor in dynamic sizes (execution queue count, etc).
  return sizeof(iree_hal_amdgpu_host_queue_t);
}

iree_status_t iree_hal_amdgpu_host_queue_initialize(
    iree_hal_amdgpu_system_t* system, iree_hal_amdgpu_queue_options_t options,
    hsa_agent_t device_agent, iree_host_size_t device_ordinal,
    hsa_agent_t host_agent, iree_hal_amdgpu_host_service_t* host_service,
    iree_arena_block_pool_t* host_block_pool,
    iree_hal_amdgpu_block_allocators_t block_allocators,
    iree_hal_amdgpu_buffer_pool_t* buffer_pool,
    iree_hal_amdgpu_error_callback_t error_callback,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_virtual_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(system);
  IREE_ASSERT_EQ(options.placement, IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST);
  IREE_ASSERT_ARGUMENT(host_service);
  IREE_ASSERT_ARGUMENT(host_block_pool);
  IREE_ASSERT_ARGUMENT(buffer_pool);
  IREE_ASSERT_ARGUMENT(out_queue);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, device_ordinal);

  const iree_hal_amdgpu_libhsa_t* libhsa = &system->libhsa;

  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)out_queue;
  queue->base.vtable = &iree_hal_amdgpu_host_queue_vtable;
  queue->base.device_ordinal = device_ordinal;
  queue->base.host_block_pool = host_block_pool;
  queue->base.block_allocators = block_allocators;
  queue->base.buffer_pool = buffer_pool;
  queue->error_callback = error_callback;
  queue->host_service = host_service;

  // NUMA node.
  uint32_t host_agent_node = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), host_agent,
                                  HSA_AGENT_INFO_NODE, &host_agent_node));

  // Pin the thread to the NUMA node specified.
  // We don't care which core but do want it to be one of those associated with
  // the devices this worker is servicing.
  iree_thread_affinity_t thread_affinity = {0};
  iree_thread_affinity_set_group_any(host_agent_node, &thread_affinity);

  // TODO(benvanik): implement the host queue.
  iree_status_t status = iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "host-side queuing not yet implemented");

  // Create the worker thread for handling device library requests.
  // The worker may start immediately and use the queue/doorbell.
  if (iree_status_is_ok(status)) {
    char thread_name[32];
    snprintf(thread_name, IREE_ARRAYSIZE(thread_name),
             "iree-amdgpu-queue-%" PRIhsz "-%" PRIhsz, device_ordinal,
             device_ordinal);
    const iree_thread_create_params_t thread_params = {
        .name = iree_make_cstring_view(thread_name),
        .stack_size = 0,  // default
        .create_suspended = false,
        .priority_class = IREE_THREAD_PRIORITY_CLASS_HIGH,
        .initial_affinity = thread_affinity,
    };
    status = iree_thread_create(iree_hal_amdgpu_host_queue_main, queue,
                                thread_params, host_allocator, &queue->thread);
  }

  if (!iree_status_is_ok(status)) {
    out_queue->vtable->deinitialize(out_queue);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_host_queue_deinitialize(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT request thread shutdown
  (void)queue;

  // Join thread after it has shut down.
  if (queue->thread) {
    iree_thread_join(queue->thread);
    iree_thread_release(queue->thread);
    queue->thread = NULL;
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_host_queue_trim(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue) {
  iree_hal_amdgpu_host_queue_t* queue =
      iree_hal_amdgpu_host_queue_cast(virtual_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  (void)queue;

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Queue Entry Management
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_host_queue_commit_entry(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue,
    iree_hal_amdgpu_queue_entry_header_t* entry) {
  IREE_TRACE_ZONE_BEGIN(z0);

// DO NOT SUBMIT
#if 0

  // Acquire a mailbox slot (spin if full).
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_mailbox_t* mailbox =
      queue->scheduler_ptrs.mailbox;
  const uint64_t entry_index = iree_atomic_fetch_add(&mailbox->write_index, 1,
                                                     iree_memory_order_release);
  while ((entry_index -
          iree_atomic_load(&mailbox->read_index, iree_memory_order_acquire)) >=
         IREE_ARRAYSIZE(mailbox->entries)) {
    iree_thread_yield();
  }
  const uint64_t entry_mask = IREE_ARRAYSIZE(mailbox->entries) - 1;
  IREE_AMDGPU_DEVICE_PTR iree_atomic_uint64_t* entry_ptr =
      (iree_atomic_uint64_t*)&mailbox->entries[entry_index & entry_mask];

  // Spin until the slot is available for use - the scheduler should be draining
  // it ASAP and changing it to INVALID when it no longer needs it.
  uint64_t invalid_entry = IREE_HAL_AMDGPU_DEVICE_MAILBOX_ENTRY_INVALID;
  while (!iree_atomic_compare_exchange_strong(
      entry_ptr, &invalid_entry, (uint64_t)entry, iree_memory_order_acq_rel,
      iree_memory_order_relaxed)) {
    iree_thread_yield();
  }

  // Kick off a scheduler run if one is not already pending.
  iree_hal_amdgpu_queue_request_tick(
      queue, IREE_HAL_AMDGPU_HOST_QUEUE_TICK_ACTION_INCOMING);

#endif

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_host_queue_request_retire(
    iree_hal_amdgpu_virtual_queue_t* virtual_queue,
    iree_hal_amdgpu_queue_entry_header_t* entry) {
  // DO NOT SUBMIT
#if 0
  // Mark the queue entry as retired. The device-side scheduler may immediately
  // reclaim it if it is already running a tick.
  //
  // NOTE: this touches device memory and may be very slow.
  iree_amdgpu_scoped_atomic_fetch_or(
      &queue->scheduler_ptrs.active_set->retire_bits,
      1ul << entry->active_bit_index, iree_amdgpu_memory_order_release,
      iree_amdgpu_memory_scope_system);

  // Request a tick if one is not already pending. Ticks will always scan for
  // retired entries.
  iree_hal_amdgpu_queue_request_tick(
      queue, IREE_HAL_AMDGPU_HOST_QUEUE_TICK_ACTION_RETIRE);
#endif
}

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable = {
        .deinitialize = iree_hal_amdgpu_host_queue_deinitialize,
        .trim = iree_hal_amdgpu_host_queue_trim,
        .commit_entry = iree_hal_amdgpu_host_queue_commit_entry,
        .request_retire = iree_hal_amdgpu_host_queue_request_retire,
};

//===----------------------------------------------------------------------===//
// Queue Operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_host_queue_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_alloca_entry_t* entry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_virtual_queue_retire_entry(&queue->base,
                                                        &entry->header);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_dealloca_entry_t* entry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_virtual_queue_retire_entry(&queue->base,
                                                        &entry->header);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_fill(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_fill_entry_t* entry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_virtual_queue_retire_entry(&queue->base,
                                                        &entry->header);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_update(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_update_entry_t* entry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_virtual_queue_retire_entry(&queue->base,
                                                        &entry->header);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_copy(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_copy_entry_t* entry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_virtual_queue_retire_entry(&queue->base,
                                                        &entry->header);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_read(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_read_entry_t* entry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_virtual_queue_retire_entry(&queue->base,
                                                        &entry->header);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_write(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_write_entry_t* entry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_virtual_queue_retire_entry(&queue->base,
                                                        &entry->header);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_execute(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_execute_entry_t* entry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_virtual_queue_retire_entry(&queue->base,
                                                        &entry->header);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_barrier(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_barrier_entry_t* entry) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED);

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_virtual_queue_retire_entry(&queue->base,
                                                        &entry->header);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Issues ready |entry| on the queue.
// Upon return the entry may not have completed execution if it is able to run
// asynchronously on the execution queues.
static iree_status_t iree_hal_amdgpu_host_queue_issue(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_queue_entry_header_t* entry) {
  switch (entry->type) {
    case IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_ALLOCA:
      return iree_hal_amdgpu_host_queue_alloca(
          queue, (iree_hal_amdgpu_queue_alloca_entry_t*)entry);
    case IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_DEALLOCA:
      return iree_hal_amdgpu_host_queue_dealloca(
          queue, (iree_hal_amdgpu_queue_dealloca_entry_t*)entry);
    case IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_FILL:
      return iree_hal_amdgpu_host_queue_fill(
          queue, (iree_hal_amdgpu_queue_fill_entry_t*)entry);
    case IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_UPDATE:
      return iree_hal_amdgpu_host_queue_update(
          queue, (iree_hal_amdgpu_queue_update_entry_t*)entry);
    case IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_COPY:
      return iree_hal_amdgpu_host_queue_copy(
          queue, (iree_hal_amdgpu_queue_copy_entry_t*)entry);
    case IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_READ:
      return iree_hal_amdgpu_host_queue_read(
          queue, (iree_hal_amdgpu_queue_read_entry_t*)entry);
    case IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_WRITE:
      return iree_hal_amdgpu_host_queue_write(
          queue, (iree_hal_amdgpu_queue_write_entry_t*)entry);
    case IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_EXECUTE:
      return iree_hal_amdgpu_host_queue_execute(
          queue, (iree_hal_amdgpu_queue_execute_entry_t*)entry);
    case IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_BARRIER:
      return iree_hal_amdgpu_host_queue_barrier(
          queue, (iree_hal_amdgpu_queue_barrier_entry_t*)entry);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid queue entry type %d", (int)entry->type);
  }
}

//===----------------------------------------------------------------------===//
// Queue Worker Thread
//===----------------------------------------------------------------------===//

static int iree_hal_amdgpu_host_queue_main(void* entry_arg) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)entry_arg;

  // DO NOT SUBMIT
  (void)queue;

  while (true) {
    iree_hal_amdgpu_queue_entry_header_t* entry = NULL;

    iree_status_t status = iree_ok_status();

    // check waits

    // DO NOT SUBMIT
    status = iree_hal_amdgpu_host_queue_issue(queue, entry);

    if (!iree_status_is_ok(status)) {
      // DO NOT SUBMIT error handler
    }
  }

  return 0;
}
