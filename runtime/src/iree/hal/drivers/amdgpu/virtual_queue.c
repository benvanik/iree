// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/virtual_queue.h"

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/buffer_pool.h"
#include "iree/hal/drivers/amdgpu/command_buffer.h"
#include "iree/hal/drivers/amdgpu/device/queue_entry.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_queue_options_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_queue_infer_placement(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t cpu_agent,
    hsa_agent_t gpu_agent, iree_hal_amdgpu_queue_placement_t* out_placement) {
  // TODO(benvanik): implement conditions:
  // * PCIe Atomics
  // * !PCIe Atomics && APU
  // * !PCIe Atomics && gfx90a && xGMI
  *out_placement = IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST;
  return iree_ok_status();
}

void iree_hal_amdgpu_queue_options_initialize(
    iree_hal_amdgpu_queue_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
  out_options->placement = IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST;
  out_options->flags = IREE_HAL_AMDGPU_QUEUE_FLAG_NONE;
  out_options->mode = IREE_HAL_AMDGPU_QUEUE_SCHEDULING_MODE_DEFAULT;
  out_options->control_queue_capacity =
      IREE_HAL_AMDGPU_DEFAULT_CONTROL_QUEUE_CAPACITY;
  out_options->execution_queue_count =
      IREE_HAL_AMDGPU_DEFAULT_EXECUTION_QUEUE_COUNT;
  out_options->execution_queue_capacity =
      IREE_HAL_AMDGPU_DEFAULT_EXECUTION_QUEUE_CAPACITY;
  out_options->kernarg_ringbuffer_capacity =
      IREE_HAL_AMDGPU_DEFAULT_KERNARG_RINGBUFFER_CAPACITY;
  out_options->trace_buffer_capacity =
      IREE_HAL_AMDGPU_DEFAULT_TRACE_BUFFER_CAPACITY;
}

// Verifies that the given |queue_capacity| is between the agent min/max
// requirements and a power-of-two.
static iree_status_t iree_hal_amdgpu_verify_hsa_queue_size(
    iree_string_view_t queue_name, iree_host_size_t queue_size,
    uint32_t queue_min_size, uint32_t queue_max_size) {
  // Queues must meet the min/max size requirements.
  if (queue_size < queue_min_size || queue_size > queue_max_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s queue capacity on this agent must be between "
        "HSA_AGENT_INFO_QUEUE_MIN_SIZE=%u and HSA_AGENT_INFO_QUEUE_MAX_SIZE=%u "
        "(provided %" PRIhsz ")",
        (int)queue_name.size, queue_name.data, queue_min_size, queue_max_size,
        queue_size);
  }

  // All queues must be a power-of-two due to ringbuffer masking.
  if (!iree_host_size_is_power_of_two(queue_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s queue capacity must be a power of two (provided %" PRIhsz ")",
        (int)queue_name.size, queue_name.data, queue_size);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_queue_options_verify(
    const iree_hal_amdgpu_queue_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t cpu_agent,
    hsa_agent_t gpu_agent) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(libhsa);

  // If the queue is placed on the device it must support PCIe atomics or be
  // connected via xGMI.
  if (options->placement == IREE_HAL_AMDGPU_QUEUE_PLACEMENT_DEVICE) {
    iree_hal_amdgpu_queue_placement_t possible_placement =
        IREE_HAL_AMDGPU_QUEUE_PLACEMENT_ANY;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_infer_placement(
        libhsa, cpu_agent, gpu_agent, &possible_placement));
    if (possible_placement != options->placement) {
      return iree_make_status(
          IREE_STATUS_INCOMPATIBLE,
          "device-side queue placement requested but the device does not meet "
          "the minimum requirements (PCIe atomics, xGMI, or APU)");
    }
  }

  // Query agent min/max queue size.
  uint32_t queue_min_size = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), gpu_agent,
                                               HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                                               &queue_min_size));
  uint32_t queue_max_size = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), gpu_agent,
                                               HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                                               &queue_max_size));

  // Verify HSA queues.
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_verify_hsa_queue_size(
      IREE_SV("control"), options->control_queue_capacity, queue_min_size,
      queue_max_size));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_verify_hsa_queue_size(
      IREE_SV("execution"), options->execution_queue_capacity, queue_min_size,
      queue_max_size));

  // Verify kernarg ringbuffer capacity (our ringbuffer so no HSA min/max
  // required).
  if (!iree_device_size_is_power_of_two(options->kernarg_ringbuffer_capacity)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernarg ringbuffer capacity must be a power of two (provided %" PRIdsz
        ")",
        options->kernarg_ringbuffer_capacity);
  }

  // Verify trace buffer capacity (our ringbuffer so no HSA min/max required).
  if (options->trace_buffer_capacity &&
      !iree_device_size_is_power_of_two(options->trace_buffer_capacity)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "trace buffer capacity must be a power of two (provided %" PRIdsz ")",
        options->trace_buffer_capacity);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// HAL API Utilities
//===----------------------------------------------------------------------===//

// Resolves a HAL buffer to a device-side buffer reference.
// Verifies (roughly) that it's usable but not that it's accessible to any
// particular agent.
static iree_status_t iree_hal_amdgpu_resolve_buffer_ref(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_amdgpu_device_buffer_ref_t* out_ref) {
  iree_hal_amdgpu_device_buffer_type_t type = 0;
  uint64_t bits = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer(buffer, &type, &bits));
  out_ref->type = type;
  out_ref->offset = offset;
  out_ref->length = length;
  out_ref->value.bits = bits;
  return iree_ok_status();
}

// Resolves a HAL buffer binding to a device-side buffer reference.
// Verifies (roughly) that it's usable but not that it's accessible to any
// particular agent.
static iree_status_t iree_hal_amdgpu_resolve_binding(
    iree_hal_buffer_binding_t binding,
    iree_hal_amdgpu_device_buffer_ref_t* out_device_ref) {
  iree_hal_amdgpu_device_buffer_type_t type = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer(
      binding.buffer, &type, &out_device_ref->value.bits));
  out_device_ref->type = type;
  out_device_ref->offset = binding.offset;
  out_device_ref->length =
      binding.length != IREE_HAL_WHOLE_BUFFER
          ? binding.length
          : iree_hal_buffer_byte_length(binding.buffer) - binding.offset;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Queue Entry Management
//===----------------------------------------------------------------------===//

// Reserves a new queue entry of the given |type|.
// At least |base_size| bytes will be allocated for the queue entry in addition
// to any internal allocations like the semaphore lists which will be tacked
// onto the end.
//
// If |out_resource_set| is provided a resource set will be acquired from the
// block pool and returned to the caller. The resource set will be freed when
// the queue entry retires.
static iree_status_t iree_hal_amdgpu_virtual_queue_reserve_entry(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_queue_entry_type_t type, iree_host_size_t base_size,
    iree_host_size_t max_kernarg_capacity,
    IREE_AMDGPU_DEVICE_PTR void** out_entry,
    iree_hal_resource_set_t** out_resource_set) {
  *out_entry = NULL;
  if (out_resource_set) *out_resource_set = NULL;

  // There's a lot going on here. We need at least two allocations: the queue
  // entry in device-visible memory and the resource set tracking lifetime of
  // all referenced resources in host memory. The hope is that we have a 100%
  // hit rate in the pools in the steady state and this boils down to mostly the
  // pointer math to ensure we stick to only two allocations.

  // Allocate queue entry on the device.
  const iree_host_size_t wait_list_size =
      iree_hal_amdgpu_device_semaphore_list_size(wait_semaphore_list.count);
  const iree_host_size_t signal_list_size =
      iree_hal_amdgpu_device_semaphore_list_size(signal_semaphore_list.count);
  const iree_host_size_t total_size =
      base_size + wait_list_size + signal_list_size;
  iree_hal_amdgpu_block_allocator_t* block_allocator =
      total_size >= queue->block_allocators.small.page_size
          ? &queue->block_allocators.large
          : &queue->block_allocators.small;
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_queue_entry_header_t* entry = NULL;
  iree_hal_amdgpu_block_token_t entry_token = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_block_allocator_allocate(
      block_allocator, total_size, (void**)&entry, &entry_token));
  entry->type = type;
  entry->flags = IREE_HAL_AMDGPU_QUEUE_ENTRY_FLAG_NONE;
  entry->allocation_token = entry_token.bits;
  entry->allocation_pool =
      block_allocator == &queue->block_allocators.large ? 1 : 0;

  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_semaphore_list_t* wait_list =
      (iree_hal_amdgpu_device_semaphore_list_t*)((uint8_t*)entry + base_size);
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_semaphore_list_t* signal_list =
      (iree_hal_amdgpu_device_semaphore_list_t*)((uint8_t*)entry + base_size +
                                                 wait_list_size);

  // Used to reserve kernarg space on device. Not allocated until issued.
  entry->max_kernarg_capacity = max_kernarg_capacity;

  entry->epoch = 0;             // managed by device scheduler on accept
  entry->active_bit_index = 0;  // managed by device scheduler on issue
  entry->kernarg_offset = 0;    // managed by device scheduler on issue
  entry->list_next = NULL;      // managed by device scheduler

  // Allocate a resource set used to track all of the resources associated with
  // the entry.
  iree_hal_resource_set_t* resource_set = NULL;
  iree_status_t status =
      iree_hal_resource_set_allocate(queue->host_block_pool, &resource_set);
  if (iree_status_is_ok(status) && wait_semaphore_list.count > 0) {
    status =
        iree_hal_resource_set_insert(resource_set, wait_semaphore_list.count,
                                     &wait_semaphore_list.semaphores[0]);
  }
  if (iree_status_is_ok(status) && signal_semaphore_list.count > 0) {
    status =
        iree_hal_resource_set_insert(resource_set, signal_semaphore_list.count,
                                     &signal_semaphore_list.semaphores[0]);
  }

  // Translate the semaphores referenced in the wait/signal lists to device-side
  // semaphore handles. This may fail if any semaphore is incompatible with the
  // device.
  entry->wait_list = wait_list;
  wait_list->count = (uint16_t)wait_semaphore_list.count;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < wait_semaphore_list.count; ++i) {
    status = iree_hal_amdgpu_resolve_semaphore(
        wait_semaphore_list.semaphores[i], &wait_list->entries[i].ref);
    wait_list->entries[i].payload = wait_semaphore_list.payload_values[i];
  }
  entry->signal_list = signal_list;
  signal_list->count = (uint16_t)signal_semaphore_list.count;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < signal_semaphore_list.count; ++i) {
    status = iree_hal_amdgpu_resolve_semaphore(
        signal_semaphore_list.semaphores[i], &signal_list->entries[i].ref);
    signal_list->entries[i].payload = signal_semaphore_list.payload_values[i];
  }

  if (iree_status_is_ok(status)) {
    entry->resource_set = (uint64_t)resource_set;
    *out_entry = entry;
    if (out_resource_set) *out_resource_set = resource_set;
  } else {
    if (resource_set) {
      iree_hal_resource_set_free(resource_set);
    }
    iree_hal_amdgpu_block_allocator_free(block_allocator, entry, entry_token);
  }
  return status;
}

// NOTE: this accesses device memory to perform the free and will be slow.
// Consider this useful for error handling cleanup only.
static void iree_hal_amdgpu_virtual_queue_free_entry(
    iree_hal_amdgpu_virtual_queue_t* queue,
    iree_hal_amdgpu_queue_entry_header_t* entry) {
  if (!entry) return;

  // Free all resources retained by the entry (including the semaphore list).
  iree_hal_resource_set_free((iree_hal_resource_set_t*)entry->resource_set);

  // Free the allocation holding the queue entry back to the block pool.
  iree_hal_amdgpu_block_token_t entry_token = {entry->allocation_token};
  iree_hal_amdgpu_block_allocator_free(entry->allocation_pool
                                           ? &queue->block_allocators.large
                                           : &queue->block_allocators.small,
                                       entry, entry_token);
}

iree_status_t iree_hal_amdgpu_virtual_queue_retire_entry(
    iree_hal_amdgpu_virtual_queue_t* queue,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_queue_entry_header_t* entry) {
  return iree_hal_amdgpu_virtual_queue_retire_entry_explicit(
      queue, entry, entry->signal_list->count > 0, entry->allocation_pool,
      (iree_hal_amdgpu_block_token_t){entry->allocation_token},
      (iree_hal_resource_set_t*)entry->resource_set);
}

iree_status_t iree_hal_amdgpu_virtual_queue_retire_entry_explicit(
    iree_hal_amdgpu_virtual_queue_t* queue,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_queue_entry_header_t* entry,
    bool has_signals, uint32_t allocation_pool,
    iree_hal_amdgpu_block_token_t allocation_token,
    iree_hal_resource_set_t* resource_set) {
  // Signal any semaphores that were not able to be signaled by the device.
  // NOTE: this is going to read device memory and should be avoided unless
  // has_signals indicates we have signals to process.
  if (has_signals) {
    const iree_host_size_t signal_count = entry->signal_list->count;
    for (iree_host_size_t i = 0; i < signal_count; ++i) {
      // TODO(benvanik): external/callback semaphores.
      iree_hal_amdgpu_device_semaphore_ref_t semaphore_ref =
          entry->signal_list->entries[i].ref;
      const uint64_t payload = entry->signal_list->entries[i].payload;
      IREE_RETURN_IF_ERROR(
          iree_hal_semaphore_signal(semaphore_ref.host_handle, payload),
          "signaling host-only semaphore");
    }
  }

  // Free all resources retained by the entry (including the semaphore list).
  iree_hal_resource_set_free(resource_set);

  // Free the allocation holding the queue entry back to the block pool.
  iree_hal_amdgpu_block_allocator_free(allocation_pool
                                           ? &queue->block_allocators.large
                                           : &queue->block_allocators.small,
                                       entry, allocation_token);

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Queue Operations
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_virtual_queue_alloca(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // TODO(benvanik): use params.queue_affinity to restrict access? By default
  // the device allocation handle pool is accessible to all devices in the
  // system but this can be inefficient if the handle is only ever used on a
  // single device (where we can place it in a device-specific pool).

  // TODO(benvanik): pool IDs.
  // DO NOT SUBMIT pool_id mapping
  iree_hal_amdgpu_device_allocation_pool_id_t pool_id = {
      .device_pool = NULL,
      .host_pool = 0,
  };

  // Allocate placeholder HAL buffer handle. This has no backing storage beyond
  // the allocation handle in the device-visible memory pool.
  iree_hal_buffer_t* buffer = NULL;
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_allocation_handle_t* handle =
      NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_buffer_pool_acquire(queue->buffer_pool, params,
                                          allocation_size, &buffer, &handle),
      "acquiring allocation handle");

  // NOTE: if entry reserve/commit fails we need to clean up the allocation
  // handle.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_queue_alloca_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  iree_status_t status = iree_status_annotate(
      iree_hal_amdgpu_virtual_queue_reserve_entry(
          queue, wait_semaphore_list, signal_semaphore_list,
          IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_ALLOCA, sizeof(*entry),
          IREE_HAL_AMDGPU_QUEUE_ALLOCA_MAX_KERNARG_CAPACITY, (void**)&entry,
          &resource_set),
      IREE_SV("reserving alloca queue entry"));
  if (iree_status_is_ok(status)) {
    entry->pool_id = pool_id;
    entry->min_alignment = params.min_alignment;
    entry->allocation_size = allocation_size;
    entry->handle = handle;

    // Insert newly allocated buffer into resource set to keep it live for the
    // lifetime of the queue operation. Users usually don't allocate and then
    // immediately drop the last reference to something but they will in error
    // handling cases.
    status = iree_hal_resource_set_insert(resource_set, 1, &buffer);
  }

  if (iree_status_is_ok(status)) {
    queue->vtable->commit_entry(queue, &entry->header);
    *out_buffer = buffer;
  } else {
    iree_hal_amdgpu_virtual_queue_free_entry(queue, &entry->header);
    if (buffer) iree_hal_buffer_release(buffer);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_queue_dealloca(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  // Must be a transient buffer. This will fail for other buffer types.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_allocation_handle_t* handle =
      NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_resolve_transient_buffer(buffer, &handle));

  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_queue_dealloca_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_queue_reserve_entry(
          queue, wait_semaphore_list, signal_semaphore_list,
          IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_DEALLOCA, sizeof(*entry),
          IREE_HAL_AMDGPU_QUEUE_DEALLOCA_MAX_KERNARG_CAPACITY, (void**)&entry,
          &resource_set),
      "reserving dealloca queue entry");
  entry->handle = handle;

  // Insert the deallocated buffer handle into resource set to keep it live for
  // the lifetime of the queue operation. We have to keep it live as until the
  // deallocation in the queue timeline the handle may be in use.
  iree_status_t status = iree_hal_resource_set_insert(resource_set, 1, &buffer);

  if (iree_status_is_ok(status)) {
    queue->vtable->commit_entry(queue, &entry->header);
  } else {
    iree_hal_amdgpu_virtual_queue_free_entry(queue, &entry->header);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_queue_fill(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_amdgpu_device_buffer_ref_t target_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer_ref(
                           target_buffer, target_offset, length, &target_ref),
                       "resolving `target_ref`");

  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_queue_fill_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_queue_reserve_entry(
          queue, wait_semaphore_list, signal_semaphore_list,
          IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_FILL, sizeof(*entry),
          IREE_HAL_AMDGPU_QUEUE_FILL_MAX_KERNARG_CAPACITY, (void**)&entry,
          &resource_set),
      "reserving fill queue entry");
  entry->target_ref = target_ref;
  entry->pattern = pattern_bits;
  entry->pattern_length = pattern_length;

  // Insert the target buffer into resource set to keep it live for the lifetime
  // of the queue operation.
  iree_status_t status =
      iree_hal_resource_set_insert(resource_set, 1, &target_buffer);

  if (iree_status_is_ok(status)) {
    queue->vtable->commit_entry(queue, &entry->header);
  } else {
    iree_hal_amdgpu_virtual_queue_free_entry(queue, &entry->header);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_queue_update(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_amdgpu_device_buffer_ref_t target_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer_ref(
                           target_buffer, target_offset, length, &target_ref),
                       "resolving `target_ref`");

  // NOTE: we allocate extra storage in the queue entry for the update contents.
  // This limits the size of the update data to the large block pool size minus
  // the queue entry overhead. If we wanted to fix the max update size as the
  // block size we'd have to allocate it separately.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_queue_update_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_queue_reserve_entry(
          queue, wait_semaphore_list, signal_semaphore_list,
          IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_UPDATE, sizeof(*entry) + length,
          IREE_HAL_AMDGPU_QUEUE_UPDATE_MAX_KERNARG_CAPACITY, (void**)&entry,
          &resource_set),
      "reserving update queue entry");
  entry->source_ptr = (uint8_t*)entry + sizeof(*entry);
  entry->target_ref = target_ref;

  // Insert the target buffer into resource set to keep it live for the lifetime
  // of the queue operation. The source buffer is copied into the queue entry.
  iree_status_t status =
      iree_hal_resource_set_insert(resource_set, 1, &target_buffer);

  // Copy source contents into the queue entry.
  if (iree_status_is_ok(status)) {
    iree_memcpy_stream_dst((uint8_t*)entry + sizeof(*entry),
                           (const uint8_t*)source_buffer + source_offset,
                           length);
  }

  if (iree_status_is_ok(status)) {
    queue->vtable->commit_entry(queue, &entry->header);
  } else {
    iree_hal_amdgpu_virtual_queue_free_entry(queue, &entry->header);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_queue_copy(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_amdgpu_device_buffer_ref_t source_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer_ref(
                           source_buffer, source_offset, length, &source_ref),
                       "resolving `source_ref`");
  iree_hal_amdgpu_device_buffer_ref_t target_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer_ref(
                           target_buffer, target_offset, length, &target_ref),
                       "resolving `target_ref`");

  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_queue_copy_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_queue_reserve_entry(
          queue, wait_semaphore_list, signal_semaphore_list,
          IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_COPY, sizeof(*entry),
          IREE_HAL_AMDGPU_QUEUE_COPY_MAX_KERNARG_CAPACITY, (void**)&entry,
          &resource_set),
      "reserving copy queue entry");
  entry->source_ref = source_ref;
  entry->target_ref = target_ref;

  // Insert buffers into the resource set to keep them live for the lifetime of
  // the queue operation.
  const void* resources[] = {
      source_buffer,
      target_buffer,
  };
  iree_status_t status = iree_hal_resource_set_insert(
      resource_set, IREE_ARRAYSIZE(resources), resources);

  if (iree_status_is_ok(status)) {
    queue->vtable->commit_entry(queue, &entry->header);
  } else {
    iree_hal_amdgpu_virtual_queue_free_entry(queue, &entry->header);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_queue_read(
    iree_hal_amdgpu_virtual_queue_t* queue,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_amdgpu_device_buffer_ref_t target_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer_ref(
                           target_buffer, target_offset, length, &target_ref),
                       "resolving `target_ref`");

  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_queue_read_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_queue_reserve_entry(
          queue, wait_semaphore_list, signal_semaphore_list,
          IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_READ, sizeof(*entry),
          IREE_HAL_AMDGPU_QUEUE_READ_MAX_KERNARG_CAPACITY, (void**)&entry,
          &resource_set),
      "reserving read queue entry");
  entry->queue_affinity = queue_affinity;
  entry->source_file = (uint64_t)source_file;
  entry->source_offset = source_offset;
  entry->target_ref = target_ref;

  // Insert the source file and target buffer into resource set to keep them
  // live for the lifetime of the queue operation.
  const void* resources[] = {
      source_file,
      target_buffer,
  };
  iree_status_t status = iree_hal_resource_set_insert(
      resource_set, IREE_ARRAYSIZE(resources), resources);

  if (iree_status_is_ok(status)) {
    queue->vtable->commit_entry(queue, &entry->header);
  } else {
    iree_hal_amdgpu_virtual_queue_free_entry(queue, &entry->header);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_queue_write(
    iree_hal_amdgpu_virtual_queue_t* queue,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_amdgpu_device_buffer_ref_t source_ref = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_resolve_buffer_ref(
                           source_buffer, source_offset, length, &source_ref),
                       "resolving `source_ref`");

  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_queue_write_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_queue_reserve_entry(
          queue, wait_semaphore_list, signal_semaphore_list,
          IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_WRITE, sizeof(*entry),
          IREE_HAL_AMDGPU_QUEUE_WRITE_MAX_KERNARG_CAPACITY, (void**)&entry,
          &resource_set),
      "reserving write queue entry");
  entry->queue_affinity = queue_affinity;
  entry->source_ref = source_ref;
  entry->target_file = (uint64_t)target_file;
  entry->target_offset = target_offset;

  // Insert the source buffer and target file into resource set to keep them
  // live for the lifetime of the queue operation.
  const void* resources[] = {
      source_buffer,
      target_file,
  };
  iree_status_t status = iree_hal_resource_set_insert(
      resource_set, IREE_ARRAYSIZE(resources), resources);

  if (iree_status_is_ok(status)) {
    queue->vtable->commit_entry(queue, &entry->header);
  } else {
    iree_hal_amdgpu_virtual_queue_free_entry(queue, &entry->header);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_queue_barrier(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_execute_flags_t flags) {
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_queue_barrier_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_queue_reserve_entry(
          queue, wait_semaphore_list, signal_semaphore_list,
          IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_BARRIER, sizeof(*entry),
          IREE_HAL_AMDGPU_QUEUE_BARRIER_MAX_KERNARG_CAPACITY, (void**)&entry,
          /*resource_set=*/NULL),
      "reserving barrier queue entry");
  queue->vtable->commit_entry(queue, &entry->header);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_virtual_queue_execute(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  // Query the device-side resource requirements and per-device copy of the
  // command buffer information. All other information is handled device-side
  // during the issue of the execute operation.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_command_buffer_t*
      device_command_buffer = NULL;
  iree_host_size_t command_buffer_max_kernarg_capacity = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_command_buffer_query_execution_state(
          command_buffer, queue->device_ordinal, &device_command_buffer,
          &command_buffer_max_kernarg_capacity),
      "querying execution state for device %" PRIhsz, queue->device_ordinal);

  // Kernarg requirements are for the device-side control dispatches as well as
  // execution of any block of commands in the command buffer.
  const iree_host_size_t max_kernarg_capacity =
      IREE_HAL_AMDGPU_QUEUE_EXECUTE_MAX_KERNARG_CAPACITY(
          command_buffer_max_kernarg_capacity);

  // Reserve an entry in the queue for populating.
  // The device-side scheduler will not begin processing it until after it has
  // been committed below.
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_queue_execute_entry_t* entry = NULL;
  iree_hal_resource_set_t* resource_set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_queue_reserve_entry(
          queue, wait_semaphore_list, signal_semaphore_list,
          IREE_HAL_AMDGPU_QUEUE_ENTRY_TYPE_EXECUTE,
          sizeof(*entry) +
              binding_table.count * sizeof(iree_hal_amdgpu_device_buffer_ref_t),
          max_kernarg_capacity, (void**)&entry, &resource_set),
      "reserving execute queue entry");

  // NOTE: we only need to populate the flags and command buffer/binding table.
  // Other fields are setup when the operation is issued on device.

  // DO NOT SUBMIT queue entry flags

  iree_hal_amdgpu_device_execution_flags_t device_flags =
      IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_NONE;

  // TODO(benvanik): if there are few commands then set ISSUE_SERIALLY
  // (IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_ISSUE_SERIALLY) to reduce latency.
  // Serial issuing should really be per-block and we may want to turn this into
  // a block-level option.

  device_flags |= IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_SERIALIZE;
  device_flags |= IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_UNCACHED;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  device_flags |= IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_CONTROL;
  device_flags |= IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_DISPATCH;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  entry->flags = device_flags;

  // Execution will begin at the entry block (block[0]).
  entry->command_buffer = device_command_buffer;

  // Resolve all provided binding table entries to their device handles or
  // pointers. Note that this may fail if any binding is invalid and we need to
  // clean up the allocated queue entry (we do this so that we can resolve
  // in-place and not need an extra allocation).
  //
  // TODO(benvanik): store in the entry instead of the state and allow the
  // device to resolve allocation_handles.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
    status = iree_hal_amdgpu_resolve_binding(binding_table.bindings[i],
                                             &entry->state.bindings[i]);
    if (!iree_status_is_ok(status)) break;
  }

  // Insert all bindings into the queue entry resource set.
  if (iree_status_is_ok(status) && binding_table.count > 0) {
    status = iree_hal_resource_set_insert_strided(
        resource_set, binding_table.count, &binding_table.bindings[0].buffer,
        offsetof(iree_hal_buffer_binding_t, buffer),
        sizeof(iree_hal_buffer_binding_t));
  }

  if (iree_status_is_ok(status)) {
    queue->vtable->commit_entry(queue, &entry->header);
  } else {
    iree_hal_amdgpu_virtual_queue_free_entry(queue, &entry->header);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_queue_flush(
    iree_hal_amdgpu_virtual_queue_t* queue) {
  return iree_ok_status();
}
