// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/scheduler.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_kernarg_ringbuffer_t
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_kernarg_reclaim_list_enqueue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_kernarg_reclaim_list_t*
        IREE_OCL_RESTRICT list,
    uint64_t min, uint64_t max) {
  const uint64_t entry_id = iree_hal_amdgpu_device_atomic_fetch_add_explicit(
      &list->write_index, 1u, iree_hal_amdgpu_device_memory_order_relaxed,
      iree_hal_amdgpu_device_memory_scope_all_svm_devices);
  while (entry_id - iree_hal_amdgpu_device_atomic_load_explicit(
                        &list->read_index,
                        iree_hal_amdgpu_device_memory_order_acquire,
                        iree_hal_amdgpu_device_memory_scope_all_svm_devices) >=
         iree_hal_amdgpu_device_kernarg_reclaim_list_capacity(list)) {
    iree_hal_amdgpu_device_yield();  // spinning
  }
  const uint64_t slot_index =
      entry_id &
      (iree_hal_amdgpu_device_kernarg_reclaim_list_capacity(list) - 1);
  list->slots[slot_index].min = min;
  list->slots[slot_index].max = max;
}

// Returns true if the |ringbuffer| has capacity for |required_size|.
// |required_size| must be aligned to 8.
static bool iree_hal_amdgpu_device_kernarg_ringbuffer_has_capacity(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_kernarg_ringbuffer_t*
        IREE_OCL_RESTRICT ringbuffer,
    uint64_t required_size) {
  const bool reclaim_list_full =
      iree_hal_amdgpu_device_atomic_load_explicit(
          &ringbuffer->reclaim_list.write_index,
          iree_hal_amdgpu_device_memory_order_relaxed,
          iree_hal_amdgpu_device_memory_scope_all_svm_devices) -
          iree_hal_amdgpu_device_atomic_load_explicit(
              &ringbuffer->reclaim_list.read_index,
              iree_hal_amdgpu_device_memory_order_acquire,
              iree_hal_amdgpu_device_memory_scope_all_svm_devices) >=
      iree_hal_amdgpu_device_kernarg_reclaim_list_capacity(
          &ringbuffer->reclaim_list);
  return !reclaim_list_full &&
         (ringbuffer->write_offset + required_size - ringbuffer->read_offset <=
          ringbuffer->capacity);
}

static inline IREE_OCL_ATTRIBUTE_ALWAYS_INLINE IREE_OCL_GLOBAL uint8_t*
iree_hal_amdgpu_device_kernarg_ringbuffer_resolve(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_kernarg_ringbuffer_t*
        IREE_OCL_RESTRICT ringbuffer,
    uint64_t offset) {
  return ringbuffer->base_ptr + (offset & (ringbuffer->capacity - 1));
}

static uint64_t iree_hal_amdgpu_device_kernarg_ringbuffer_acquire(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_kernarg_ringbuffer_t*
        IREE_OCL_RESTRICT ringbuffer,
    uint64_t size, IREE_OCL_GLOBAL uint8_t* IREE_OCL_PRIVATE* out_ptr) {
  const uint64_t offset = ringbuffer->write_offset;
  ringbuffer->write_offset += size;
  *out_ptr =
      iree_hal_amdgpu_device_kernarg_ringbuffer_resolve(ringbuffer, offset);
  return offset;
}

static void iree_hal_amdgpu_device_kernarg_ringbuffer_release(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_kernarg_ringbuffer_t*
        IREE_OCL_RESTRICT ringbuffer,
    uint64_t offset, uint64_t size) {
  if (offset == ringbuffer->read_offset) {
    // Releasing from the ringbuffer read offset; just advance the offset.
    // This is the path we hope for.
    ringbuffer->read_offset += size;
  } else {
    // Out-of-order release; add to the reclaim list so that it gets processed
    // in the future.
    iree_hal_amdgpu_device_kernarg_reclaim_list_enqueue(
        &ringbuffer->reclaim_list, offset, offset + size);
  }
}

static void iree_hal_amdgpu_device_kernarg_ringbuffer_reclaim(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_kernarg_ringbuffer_t*
        IREE_OCL_RESTRICT ringbuffer) {
  // Capture the valid range of reclaim requests at the start and work only
  // within that. Nothing can change the read index but us.
  const uint64_t base_read_index = iree_hal_amdgpu_device_atomic_load_explicit(
      &ringbuffer->reclaim_list.read_index,
      iree_hal_amdgpu_device_memory_order_relaxed,
      iree_hal_amdgpu_device_memory_scope_device);
  const uint64_t write_index = iree_hal_amdgpu_device_atomic_load_explicit(
      &ringbuffer->reclaim_list.write_index,
      iree_hal_amdgpu_device_memory_order_acquire,
      iree_hal_amdgpu_device_memory_scope_all_svm_devices);
  if (base_read_index == write_index) {
    return;  // fast-path for empty reclaim rings
  }

  // Terribly loop through releasing ranges until we find a discontinuity. This
  // is bad. A better data structure would not require this.
  uint64_t new_read_index = base_read_index;
  bool did_change = false;
  do {
    for (uint64_t i = new_read_index; i < write_index; ++i) {
      const uint64_t slot_id =
          i & (iree_hal_amdgpu_device_kernarg_reclaim_list_capacity(
                   &ringbuffer->reclaim_list) -
               1);
      if (ringbuffer->reclaim_list.slots[slot_id].min ==
          ringbuffer->read_offset) {
        // Found an entry that releases at the head of the list.
        // If it's at the base read index we can advance the read index to drop
        // it from the ring. If it's not then we can swap it with the one that
        // was at the base, bump the index, and then continue on. We don't break
        // from the loop until we reach the write index so that we have fewer
        // overall loops (we get all we can every pass) - this may be the wrong
        // tradeoff in pathological cases where the reclaim ranges are in exact
        // reverse order.
        ringbuffer->read_offset = ringbuffer->reclaim_list.slots[slot_id].max;
        if (i == new_read_index) {
          // At the head of the read range; just skip the entry.
          ++new_read_index;
        } else {
          // Out-of-order; swap the entry with the head of the read range and
          // then skip past it. We've already checked the head entry and can
          // ignore it this pass as we continue the loop.
          const uint64_t head_slot_id =
              new_read_index &
              (iree_hal_amdgpu_device_kernarg_reclaim_list_capacity(
                   &ringbuffer->reclaim_list) -
               1);
          ringbuffer->reclaim_list.slots[slot_id] =
              ringbuffer->reclaim_list.slots[head_slot_id];
          ++new_read_index;
        }
        did_change = true;
      }
    }
  } while (did_change);

  // If we consumed any entries bump the read index to the last consumed.
  if (new_read_index != base_read_index) {
    iree_hal_amdgpu_device_atomic_store_explicit(
        &ringbuffer->reclaim_list.read_index, new_read_index,
        iree_hal_amdgpu_device_memory_order_release,
        iree_hal_amdgpu_device_memory_scope_all_svm_devices);
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_queue_list_t
//===----------------------------------------------------------------------===//

// Appends the given |entry| to the end of the |list|. Exclusively using this
// will make the list be treated like a queue with respect to the list
// manipulations but will not order entries with respect to when they were
// originally submitted.
static inline void iree_hal_amdgpu_device_queue_list_append(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_list_t* IREE_OCL_RESTRICT list,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
        IREE_OCL_RESTRICT entry) {
  entry->list_next = NULL;
  if (list->head == NULL) {
    list->head = entry;
    list->tail = entry;
  } else {
    list->tail->list_next = entry;
    list->tail = entry;
  }
}

// Inserts the given |entry| in the |list| immediately before the first entry
// with a larger epoch. Exclusively using this will make the list be treated
// like a FIFO.
static inline void iree_hal_amdgpu_device_queue_list_insert(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_list_t* IREE_OCL_RESTRICT list,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
        IREE_OCL_RESTRICT entry) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* list_cursor =
      list->head;
  if (list->head == NULL) {
    // First entry in the list.
    list->head = entry;
    list->tail = entry;
  } else {
    // Find the insertion point and splice in. Insert immediately prior to the
    // next epoch greater than the requested (or the tail).
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* list_prev =
        NULL;
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* list_cursor =
        list->head;
    while (list_cursor != NULL) {
      if (list_cursor == list->tail) {
        list_cursor->list_next = entry;
        list->tail = entry;
        break;
      } else if (list_cursor->epoch > entry->epoch) {
        if (list_prev != NULL) {
          if (list_prev->list_next != NULL) {
            entry->list_next = list_prev->list_next;
          }
          list_prev->list_next = entry;
        }
        break;
      }
      list_prev = list_cursor;
      list_cursor = list_cursor->list_next;
    }
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_queue_active_set_t
//===----------------------------------------------------------------------===//

// Returns true if the active set is empty.
static inline IREE_OCL_ATTRIBUTE_ALWAYS_INLINE bool
iree_hal_amdgpu_device_queue_active_set_is_empty(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_queue_active_set_t*
        IREE_OCL_RESTRICT active_set) {
  return active_set->active_bits == 0;
}

// Returns true if the active set is full.
static inline IREE_OCL_ATTRIBUTE_ALWAYS_INLINE bool
iree_hal_amdgpu_device_queue_active_set_is_full(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_queue_active_set_t*
        IREE_OCL_RESTRICT active_set) {
  return active_set->active_bits == UINT64_MAX;
}

// Inserts an entry into the active set.
// The queue entry bit index will be set to the assigned entry.
static inline void iree_hal_amdgpu_device_queue_active_set_insert(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_active_set_t* IREE_OCL_RESTRICT
        active_set,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
        IREE_OCL_RESTRICT entry) {
  // Find the index of the first unset bit indicating an unused entry.
  // We'll reserve it for the entry and add it to the set.
  const uint8_t bit_index = IREE_OCL_LASTBIT_U64(~active_set->active_bits);
  entry->active_bit_index = bit_index;
  active_set->entries[bit_index] = entry;
}

// Erases an entry from the active set.
static inline void iree_hal_amdgpu_device_queue_active_set_erase(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_active_set_t* IREE_OCL_RESTRICT
        active_set,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
        IREE_OCL_RESTRICT entry) {
  active_set->active_bits &= ~(1ul << entry->active_bit_index);
  active_set->entries[entry->active_bit_index] = NULL;
  entry->active_bit_index = 0xFFu;
}

static inline IREE_OCL_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_device_queue_active_set_reschedule(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_active_set_t* IREE_OCL_RESTRICT
        active_set,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
        IREE_OCL_RESTRICT entry) {
  // NOTE: we could scope to the current device if we are use the caller matches
  // the scheduler location.
  iree_hal_amdgpu_device_atomic_fetch_or_explicit(
      &active_set->reschedule_bits, 1ul << entry->active_bit_index,
      iree_hal_amdgpu_device_memory_order_release,
      iree_hal_amdgpu_device_memory_scope_all_svm_devices);
}

static inline IREE_OCL_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_device_queue_active_set_retire(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_active_set_t* IREE_OCL_RESTRICT
        active_set,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
        IREE_OCL_RESTRICT entry) {
  // NOTE: we could scope to the current device if we are use the caller matches
  // the scheduler location.
  iree_hal_amdgpu_device_atomic_fetch_or_explicit(
      &active_set->retire_bits, 1ul << entry->active_bit_index,
      iree_hal_amdgpu_device_memory_order_release,
      iree_hal_amdgpu_device_memory_scope_all_svm_devices);
}

//===----------------------------------------------------------------------===//
// Queue Retirement
//===----------------------------------------------------------------------===//

// Retires the |entry| by releasing any resources it retains and signaling all
// waiters. The queue entry memory may be invalidated by the time this call
// returns. This is only meant to run on the control queue of the scheduler that
// currently owns the queue entry.
static void iree_hal_amdgpu_device_queue_retire(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_queue_entry_header_t*
        IREE_OCL_RESTRICT entry) {
  // Release kernarg resources if any were used.
  if (entry->max_kernarg_capacity > 0) {
    iree_hal_amdgpu_device_kernarg_ringbuffer_release(
        &scheduler->kernarg_ringbuffer, entry->kernarg_offset,
        entry->max_kernarg_capacity);
  }

  // Signal all waiters by enqueuing them in the wake set.
  // Wakes of target queues will be accumulated across all retires and issued
  // at once at the end of the tick.
  for (uint32_t i = 0; i < entry->signal_list->count; ++i) {
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_semaphore_t* semaphore =
        entry->signal_list->entries[i].semaphore;
    const uint64_t payload = entry->signal_list->entries[i].payload;
    iree_hal_amdgpu_device_semaphore_signal(semaphore, payload,
                                            &scheduler->wake_set);
  }

  // Post a retire notification to the host. The host may reclaim queue entry
  // resources immediately or do so on a delay.
  //
  // TODO(benvanik): a way to route to a device-side allocator. Today only the
  // host creates the queue entries and is responsible for cleaning them up.
  // We'd need to figure out ownership/ref counting rules.
  iree_hal_amdgpu_device_host_post_retire(&scheduler->host, (uint64_t)scheduler,
                                          (uint64_t)entry);
}

#define IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE \
  (2 * sizeof(uint64_t))

// Retires the provided scheduler queue entry from an execution queue.
// This is just a trampoline to allow us to sequence the retirement in an
// execution queue after the operations performed by the entry.
__kernel IREE_OCL_ATTRIBUTE_SINGLE_WORK_ITEM void
iree_hal_amdgpu_device_queue_retire_entry(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    uint64_t scheduler_queue_entry) {
  return iree_hal_amdgpu_device_queue_scheduler_retire_from_execution_queue(
      scheduler, scheduler_queue_entry);
}

// Enqueues a `iree_hal_amdgpu_device_queue_retire_entry` kernel on the
// given queue. |kernarg_ptr| must point to an 8-byte aligned kernarg region
// with at least IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE bytes of
// space.
static void iree_hal_amdgpu_device_queue_retire_entry_enqueue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL iree_hsa_queue_t* IREE_OCL_RESTRICT queue,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
        IREE_OCL_RESTRICT entry,
    IREE_OCL_GLOBAL uint64_t* IREE_OCL_RESTRICT kernarg_ptr) {
  // Store kernargs.
  kernarg_ptr[0] = (uint64_t)scheduler;
  kernarg_ptr[1] = (uint64_t)entry;

  // Reserve the next packet in the target queue.
  const uint64_t packet_id = iree_hsa_queue_add_write_index(
      queue, 1u, iree_hal_amdgpu_device_memory_order_relaxed);
  while (packet_id - iree_hsa_queue_load_read_index(
                         queue, iree_hal_amdgpu_device_memory_order_acquire) >=
         queue->size) {
    iree_hal_amdgpu_device_yield();  // spinning
  }

  // Construct the control packet.
  // Note that the header is not written until the end so that the
  // hardware command processor stalls until we're done writing.
  const uint64_t queue_mask = queue->size - 1;  // power of two
  IREE_OCL_GLOBAL iree_hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT
      retire_packet = queue->base_address + (packet_id & queue_mask) * 64;
  const iree_hal_amdgpu_device_kernel_args_t retire_args =
      scheduler->kernels->retire_entry;
  retire_packet->setup = retire_args.setup;
  retire_packet->workgroup_size[0] = retire_args.workgroup_size[0];
  retire_packet->workgroup_size[1] = retire_args.workgroup_size[1];
  retire_packet->workgroup_size[2] = retire_args.workgroup_size[2];
  retire_packet->reserved0 = 0;
  retire_packet->grid_size[0] = 1;
  retire_packet->grid_size[1] = 1;
  retire_packet->grid_size[2] = 1;
  retire_packet->private_segment_size = retire_args.private_segment_size;
  retire_packet->group_segment_size = retire_args.group_segment_size;
  retire_packet->kernel_object = retire_args.kernel_object;
  retire_packet->kernarg_address = kernarg_ptr;
  retire_packet->reserved2 = 0;

  // Mark the update packet as ready to execute. The hardware command
  // processor may begin executing it immediately after performing the
  // atomic swap.
  uint16_t retire_header = IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH
                           << IREE_HSA_PACKET_HEADER_TYPE;

  // Barrier bit set to ensure all prior commands complete.
  retire_header |= 1 << IREE_HSA_PACKET_HEADER_BARRIER;

  // TODO(benvanik): ensure we need these scopes? We may need release agent
  // (as we touch agent resources when enqueuing the scheduler?) but we may not
  // need to acquire anything. For now we conservatively acquire/release both.
  retire_header |= IREE_HSA_FENCE_SCOPE_AGENT
                   << IREE_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  retire_header |= IREE_HSA_FENCE_SCOPE_AGENT
                   << IREE_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;

  const uint32_t retire_header_setup =
      retire_header | (uint32_t)(retire_packet->setup << 16);
  iree_hal_amdgpu_device_atomic_store_explicit(
      (IREE_OCL_GLOBAL iree_hal_amdgpu_device_atomic_uint32_t*)retire_packet,
      retire_header_setup, iree_hal_amdgpu_device_memory_order_release,
      iree_hal_amdgpu_device_memory_scope_all_svm_devices);

  // Signal the queue doorbell indicating the packet has been updated.
  iree_hsa_signal_store(queue->doorbell_signal, packet_id,
                        iree_hal_amdgpu_device_memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// Queue Operations
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_queue_issue_initialize(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_initialize_entry_t*
        IREE_OCL_RESTRICT entry) {
  IREE_OCL_TRACE_BUFFER_DEFINE(scheduler->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);

  // Initialize the signal pool with the provided HSA signals.
  iree_hal_amdgpu_device_signal_pool_initialize(
      scheduler->signal_pool, entry->signal_count, entry->signals);

  // Initialize the wake set that will be used during ticks to accumulate wake
  // targets. Each set tracks the 'self' target so that a tick can chain with
  // itself on the local queue.
  iree_hal_amdgpu_wake_target_t self_target = {
      .scheduler = scheduler,
  };
  iree_hal_amdgpu_device_wake_set_initialize(self_target, &scheduler->wake_set);
  iree_hal_amdgpu_device_wake_pool_initialize(self_target,
                                              &scheduler->wake_pool);

  // Mark the queue entry as retired.
  iree_hal_amdgpu_device_queue_active_set_retire(&scheduler->active_set,
                                                 &entry->header);

  IREE_OCL_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_device_queue_issue_deinitialize(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_deinitialize_entry_t*
        IREE_OCL_RESTRICT entry) {
  IREE_OCL_TRACE_BUFFER_DEFINE(scheduler->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): tear down? spin until all active entries retire?

  // Mark the queue entry as retired.
  // The host may immediately reclaim the queue memory as soon as it is woken.
  iree_hal_amdgpu_device_queue_active_set_retire(&scheduler->active_set,
                                                 &entry->header);

  IREE_OCL_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_device_queue_issue_alloca(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_alloca_entry_t*
        IREE_OCL_RESTRICT entry) {
  IREE_OCL_TRACE_BUFFER_DEFINE(scheduler->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);

  // Begin alloca operation. This may require asynchronous requests to the host
  // and not complete immediately. If asynchronous the allocator will issue the
  // retire after the allocation handle has been updated.
  if (iree_hal_amdgpu_device_allocator_alloca(
          scheduler->allocator, (uint64_t)&entry->header, entry->pool,
          entry->min_alignment, entry->allocation_size, entry->handle)) {
    // Operation completed synchronously - mark the queue entry as retired.
    iree_hal_amdgpu_device_queue_active_set_retire(&scheduler->active_set,
                                                   &entry->header);
  }

  IREE_OCL_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_device_queue_issue_dealloca(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_dealloca_entry_t*
        IREE_OCL_RESTRICT entry) {
  IREE_OCL_TRACE_BUFFER_DEFINE(scheduler->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);

  // Begin dealloca operation. This may require asynchronous requests to the
  // host and not complete immediately. If asynchronous the allocator will issue
  // the retire after the allocation handle has been updated.
  if (iree_hal_amdgpu_device_allocator_dealloca(scheduler->allocator,
                                                (uint64_t)&entry->header,
                                                entry->pool, entry->handle)) {
    // Operation completed synchronously - mark the queue entry as retired.
    iree_hal_amdgpu_device_queue_active_set_retire(&scheduler->active_set,
                                                   &entry->header);
  }

  IREE_OCL_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_device_queue_issue_fill(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_fill_entry_t* IREE_OCL_RESTRICT
        entry) {
  IREE_OCL_TRACE_BUFFER_DEFINE(scheduler->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);

  // Translate target buffer to a device pointer.
  IREE_OCL_GLOBAL void* IREE_OCL_RESTRICT target_ptr =
      iree_hal_amdgpu_device_buffer_ref_resolve(entry->target_ref, NULL);

  // Acquire kernarg storage from the ringbuffer.
  // The issue should have been blocked until there was capacity available so
  // this will always succeed.
  IREE_OCL_GLOBAL uint8_t* kernarg_ptr = NULL;
  entry->header.kernarg_offset =
      iree_hal_amdgpu_device_kernarg_ringbuffer_acquire(
          &scheduler->kernarg_ringbuffer,
          IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE +
              IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE,
          &kernarg_ptr);

  // Enqueue the fill blit or DMA operation to the execution queue.
  iree_hal_amdgpu_device_buffer_fill_enqueue(
      &scheduler->transfer_state, target_ptr,
      iree_hal_amdgpu_device_buffer_ref_length(entry->target_ref),
      entry->pattern, entry->pattern_length,
      (IREE_OCL_GLOBAL uint64_t*)kernarg_ptr);

  // Enqueue the retire dispatch that will mark the entry as completed and
  // launch the scheduler.
  iree_hal_amdgpu_device_queue_retire_entry_enqueue(
      scheduler, scheduler->transfer_state.queue, &entry->header,
      (IREE_OCL_GLOBAL uint64_t*)(kernarg_ptr +
                                  IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE));

  IREE_OCL_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_device_queue_issue_copy(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_copy_entry_t* IREE_OCL_RESTRICT
        entry) {
  IREE_OCL_TRACE_BUFFER_DEFINE(scheduler->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);

  // Translate source/target buffers to device pointers.
  IREE_OCL_GLOBAL void* IREE_OCL_RESTRICT source_ptr =
      iree_hal_amdgpu_device_buffer_ref_resolve(entry->source_ref, NULL);
  IREE_OCL_GLOBAL void* IREE_OCL_RESTRICT target_ptr =
      iree_hal_amdgpu_device_buffer_ref_resolve(entry->target_ref, NULL);

  // Acquire kernarg storage from the ringbuffer.
  // The issue should have been blocked until there was capacity available.
  IREE_OCL_GLOBAL uint8_t* kernarg_ptr = NULL;
  entry->header.kernarg_offset =
      iree_hal_amdgpu_device_kernarg_ringbuffer_acquire(
          &scheduler->kernarg_ringbuffer,
          IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE +
              IREE_HAL_AMDGPU_DEVICE_QUEUE_RETIRE_ENTRY_KERNARG_SIZE,
          &kernarg_ptr);

  // Enqueue the copy blit or DMA operation to the execution queue.
  iree_hal_amdgpu_device_buffer_copy_enqueue(
      &scheduler->transfer_state, source_ptr, target_ptr,
      iree_hal_amdgpu_device_buffer_ref_length(entry->target_ref),
      (IREE_OCL_GLOBAL uint64_t*)kernarg_ptr);

  // Enqueue the retire dispatch that will mark the entry as completed and
  // launch the scheduler.
  iree_hal_amdgpu_device_queue_retire_entry_enqueue(
      scheduler, scheduler->transfer_state.queue, &entry->header,
      (IREE_OCL_GLOBAL uint64_t*)(kernarg_ptr +
                                  IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE));

  IREE_OCL_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_device_queue_issue_execute(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_execute_entry_t*
        IREE_OCL_RESTRICT entry) {
  IREE_OCL_TRACE_BUFFER_DEFINE(scheduler->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);

  // Setup the block for entry. Most of this state should have been specified
  // when the execution request was enqueued but ensuring it's valid here is
  // cheap.
  entry->state.block = entry->state.command_buffer->blocks[0];
  entry->state.scheduler = scheduler;
  entry->state.scheduler_queue_entry = (uint64_t)entry;
  entry->state.transfer_state = scheduler->transfer_state;

  // Allocate kernarg storage. The queue entry is only issued when there is
  // capacity available so we know this will not fail.
  // We split the pointers between kernels running on the control queue vs those
  // running on the execution queue in case we ever decide to put those on
  // different devices. Today they're just portions of the same base allocation.
  IREE_OCL_GLOBAL uint8_t* kernarg_ptr = NULL;
  entry->header.kernarg_offset =
      iree_hal_amdgpu_device_kernarg_ringbuffer_acquire(
          &scheduler->kernarg_ringbuffer, entry->header.max_kernarg_capacity,
          &kernarg_ptr);
  entry->state.control_kernarg_storage = kernarg_ptr;
  entry->state.execution_kernarg_storage =
      kernarg_ptr + IREE_HAL_AMDGPU_DEVICE_EXECUTION_CONTROL_KERNARG_SIZE;

  // Enqueue the entry block on the control queue.
  // This will run and issue all of the entry block commands on the execution
  // queue. Once the command buffer has completed it will retire itself and tick
  // the scheduler - we don't need to do anything special here.
  iree_hal_amdgpu_device_command_buffer_enqueue_next_block(&entry->state);

  IREE_OCL_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_device_queue_issue_barrier(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_barrier_entry_t*
        IREE_OCL_RESTRICT entry) {
  IREE_OCL_TRACE_BUFFER_DEFINE(scheduler->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);

  // Any waits have been resolved by the time this is issued so we can retire
  // inline. The scheduler will reclaim the entry after it is done issuing.
  iree_hal_amdgpu_device_queue_active_set_retire(&scheduler->active_set,
                                                 &entry->header);

  IREE_OCL_TRACE_ZONE_END(z0);
}

// Issues the given queue entry.
// Assumes that the entry is allowed to begin executing immediately.
static void iree_hal_amdgpu_device_queue_issue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
        IREE_OCL_RESTRICT entry) {
  IREE_OCL_TRACE_BUFFER_DEFINE(scheduler->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);
  switch (entry->type) {
    default:
      break;
    case IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_INITIALIZE:
      iree_hal_amdgpu_device_queue_issue_initialize(
          scheduler,
          (IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_initialize_entry_t*)
              entry);
      break;
    case IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_DEINITIALIZE:
      iree_hal_amdgpu_device_queue_issue_deinitialize(
          scheduler,
          (IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_deinitialize_entry_t*)
              entry);
      break;
    case IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_ALLOCA:
      iree_hal_amdgpu_device_queue_issue_alloca(
          scheduler,
          (IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_alloca_entry_t*)entry);
      break;
    case IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_DEALLOCA:
      iree_hal_amdgpu_device_queue_issue_dealloca(
          scheduler,
          (IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_dealloca_entry_t*)
              entry);
      break;
    case IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_FILL:
      iree_hal_amdgpu_device_queue_issue_fill(
          scheduler,
          (IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_fill_entry_t*)entry);
      break;
    case IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_COPY:
      iree_hal_amdgpu_device_queue_issue_copy(
          scheduler,
          (IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_copy_entry_t*)entry);
      break;
    case IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_EXECUTE:
      iree_hal_amdgpu_device_queue_issue_execute(
          scheduler,
          (IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_execute_entry_t*)entry);
      break;
    case IREE_HAL_AMDGPU_DEVICE_QUEUE_ENTRY_TYPE_BARRIER:
      iree_hal_amdgpu_device_queue_issue_barrier(
          scheduler,
          (IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_barrier_entry_t*)entry);
      break;
  }
  IREE_OCL_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_queue_scheduler_t
//===----------------------------------------------------------------------===//

// when wait satisfied:
//   clear semaphore required bit
//   move to incoming queue if more bits set
//   move to run list of no bits set (all waits satisfied)

//
// iree_hal_amdgpu_device_queue_entry_header_t
//   contains storage for wait list?
//   wait list is linked, contains all waiting entries
//   only one registered wake at a time per entry
//   wait list is run down, move to run list

// Accepts all incoming queue operations from the HSA softqueue.
// Operations are immediately moved into the scheduler run list if they have
// no dependencies and otherwise are put in the scheduler wait list to be
// evaluated during the tick.
// Returns true if any operations were added to the wait list.
static bool iree_hal_amdgpu_device_queue_scheduler_accept_incoming(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler) {
  // drain softqueue
  // move to waitlist or runlist
  // if no wait semas then
  // iree_hal_amdgpu_device_queue_list_insert(ready_list, entry);
  // set epoch++
  return false;
}

// Checks each waiting queue entry for whether it is able to be run.
// Maintains the per-semaphore wake lists and does other bookkeeping as-needed.
// Upon return the scheduler run list may have new entries in it.
static void iree_hal_amdgpu_device_queue_scheduler_check_wait_list(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler) {
  IREE_OCL_TRACE_BUFFER_DEFINE(scheduler->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);

  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* list_prev = NULL;
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* list_cursor =
      scheduler->wait_list.head;
  while (list_cursor != NULL) {
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* list_next =
        list_cursor->list_next;

    IREE_OCL_GLOBAL iree_hal_amdgpu_device_semaphore_list_t* semaphore_list =
        list_cursor->wait_list;
    do {
      // Get the semaphore wake list for the head wait. Note that waits are
      // unordered so this is just "the next wait to check" and not "the first
      // wait that must be satisfied".
      IREE_OCL_GLOBAL iree_hal_amdgpu_device_semaphore_t* semaphore =
          semaphore_list->entries[0].semaphore;

      // Reserve (or find) the wake list entry in the scheduler pool.
      // We may already be registered to wait on the semaphore in which case
      // we'll no-op this check or modify the minimum required value if this new
      // wait happens to be less than the old one. If not already waiting the
      // entry we get back will be initialized for use.
      IREE_OCL_GLOBAL iree_hal_amdgpu_wake_list_entry_t* wake_list_entry =
          iree_hal_amdgpu_device_wake_pool_reserve(&scheduler->wake_pool,
                                                   semaphore);

      // Break on the first wait that isn't satisfied - we only need one to
      // track as the barrier is a wait-all and so long as any single wait is
      // not satisfied we can't progress.
      //
      // This operation takes the lock on the target semaphore wake list and if
      // it returns true it means that this scheduler will be woken when the
      // requested value is reached. If it returns false we know the value is
      // already satisfied and can treat the wait as resolved.
      const bool is_waiting = iree_hal_amdgpu_device_semaphore_update_wait(
          semaphore, wake_list_entry, semaphore_list->entries[0].payload);
      if (is_waiting) {
        // Already waiting or now waiting - either way, we're blocked until the
        // wake resolves. Stop processing the waits for this entry and move on
        // to the next.
        list_prev = list_cursor;
        break;
      } else {
        // Not waiting - release the reserved wake list entry.
        iree_hal_amdgpu_device_wake_pool_release(&scheduler->wake_pool,
                                                 wake_list_entry);
      }

      // Remove the semaphore from the wait list by swapping in the last
      // element. If this was the last wait in the list then the operation is
      // ready to run and can be moved to the run list.
      if (semaphore_list->count == 1) {
        // Last wait - move to run list.
        // Note we leave the list_prev pointer at the prior entry so that the
        // next wait list entry loop will be able to remove itself if needed.
        if (list_prev != NULL) {
          list_prev->list_next = list_next;
        } else {
          scheduler->wait_list.head = list_next;
        }
        if (list_next == NULL) {
          scheduler->wait_list.tail = NULL;
        }
        list_cursor->list_next = NULL;
        iree_hal_amdgpu_device_queue_list_insert(&scheduler->ready_list,
                                                 list_cursor);
        break;
      } else {
        // Remaining waits - swap in one (order doesn't matter) and retry.
        semaphore_list->entries[0] =
            semaphore_list->entries[semaphore_list->count - 1];
        --semaphore_list->count;
        continue;
      }
    } while (semaphore_list->count > 0);

    // NOTE: list_prev set above based on whether the entry is moved to the run
    // list or not.
    list_cursor = list_next;
  }

  IREE_OCL_TRACE_ZONE_END(z0);
}

// Reschedules all entries that have requested it at the time of the call.
// Other entries may request rescheduling asynchronously before the call
// returns. Rescheduled entries will be moved to the ready list and await the
// next tick.
static void iree_hal_amdgpu_device_queue_scheduler_reschedule_entries(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler) {
  // Acquire set of reschedule entries.
  uint64_t reschedule_bits = iree_hal_amdgpu_device_atomic_exchange_explicit(
      &scheduler->active_set.reschedule_bits, 0u,
      iree_hal_amdgpu_device_memory_order_acquire,
      iree_hal_amdgpu_device_memory_scope_all_svm_devices);
  if (!reschedule_bits) return;

  // For each rescheduling entry remove it from the active set and move it to
  // the ready list.
  uint32_t bit_index = 0;
  do {
    // Find the bit index of the entry to reschedule next.
    uint32_t bit_offset = IREE_OCL_LASTBIT_U64(reschedule_bits);
    uint32_t reschedule_index = bit_index + bit_offset;
    bit_index += bit_offset + 1;
    reschedule_bits = reschedule_bits >> (bit_offset + 1);

    // Remove the reschedule entry from the active set.
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
        reschedule_entry = scheduler->active_set.entries[reschedule_index];
    iree_hal_amdgpu_device_queue_active_set_erase(&scheduler->active_set,
                                                  reschedule_entry);

    // Move to the ready list; inserted in epoch order.
    iree_hal_amdgpu_device_queue_list_insert(&scheduler->ready_list,
                                             reschedule_entry);
  } while (reschedule_bits != 0);
}

// Retires all entries that have requested it at the time of the call.
// Other entries may complete asynchronously before the call returns.
static void iree_hal_amdgpu_device_queue_scheduler_retire_entries(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler) {
  // Acquire set of retire entries.
  uint64_t retire_bits = iree_hal_amdgpu_device_atomic_exchange_explicit(
      &scheduler->active_set.retire_bits, 0u,
      iree_hal_amdgpu_device_memory_order_acquire,
      iree_hal_amdgpu_device_memory_scope_all_svm_devices);
  if (!retire_bits) return;

  // For each retiring entry remove it from the active set and retire it.
  uint32_t bit_index = 0;
  do {
    // Find the bit index of the entry to retire next.
    uint32_t bit_offset = IREE_OCL_LASTBIT_U64(retire_bits);
    uint32_t retire_index = bit_index + bit_offset;
    bit_index += bit_offset + 1;
    retire_bits = retire_bits >> (bit_offset + 1);

    // Remove the retire entry from the active set.
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* retire_entry =
        scheduler->active_set.entries[retire_index];
    iree_hal_amdgpu_device_queue_active_set_erase(&scheduler->active_set,
                                                  retire_entry);

    // Retire the entry.
    iree_hal_amdgpu_device_queue_retire(scheduler, retire_entry);
  } while (retire_bits != 0);
}

#if 0
__kernel IREE_OCL_ATTRIBUTE_SINGLE_WORK_ITEM void
iree_hal_amdgpu_device_queue_scheduler_tick(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler) {


  // who owns kernargs?
  // ringbuffer? or always as part of queue operation?
  // can't mix kernarg region with non-kernarg region
  // command buffer return could provide as part of its storage
  // could be per-execution-queue (simultaneous command buffer count)
  // if only one scheduler tick can be pending at a time could be reused
  // need fancy atomics?
  //   if (atomic inc scheduler_request_pending == 0) {
  //     none was pending
  //     update kernargs
  //     enqueue
  //   }
  //   on tick: atomic dec scheduler_request_pending
  //
  // then reason needs to be an atomic bitmask? request pending could be
  // atomic OR the reason for scheduling
  // is reason needed?
  //
  // scheduler could poke execution state of all running
  // then could use static kernargs: each execution uses the same scheduler ptr
  // command buffer return could just be a bit indicating that it should be
  // checked in the next schedule run ("an execution completed")
  //
  // chaining signals? completion signal assigned by scheduler
  // command buffer return signals
  // do we need a barrier command at RETURN?
  // barrier command could use completion signal
  //
  // NEED AQL PACKET THEN!
}
#endif

__kernel IREE_OCL_ATTRIBUTE_SINGLE_WORK_ITEM void
iree_hal_amdgpu_device_queue_scheduler_tick(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler) {
  // Acquire the set of required actions.
  // If no actions are pending we are allowed to exit early. If multiple agents
  // (the host, other devices, etc) requested a tick all of them will have ORed
  // their required actions into the set. Until we acquire the set and exchange
  // it with 0 (IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_NONE) any other agents
  // _may_ no-op their tick requests so that we don't end up with one tick
  // dispatch per request even though we process everything in a single tick.
  const iree_hal_amdgpu_device_queue_tick_action_set_t action_set =
      iree_hal_amdgpu_device_atomic_exchange_explicit(
          &scheduler->tick_action_set,
          IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_NONE,
          iree_hal_amdgpu_device_memory_order_acquire,
          iree_hal_amdgpu_device_memory_scope_all_svm_devices);
  if (!action_set) {
    // No actions pending - a prior tick must have taken care of them.
    return;
  }

  IREE_OCL_TRACE_BUFFER_DEFINE(scheduler->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);
  IREE_OCL_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, action_set);

  // Accept all incoming queue operations from the HSA softqueue.
  // This may immediately place operations in the run list if they have no
  // dependencies or are known to have been satisfied. If any entries are added
  // to the wait list then we'll do a full verification below.
  bool check_wait_list =
      iree_hal_amdgpu_device_queue_scheduler_accept_incoming(scheduler);

  // TODO(benvanik): quickly scan the wait set to see if any semaphore we're
  // waiting on has changed in a way that may wake one of our wait list entries.
  // For now this just forces a full check each tick. We could do something
  // like:
  //   if (wake_pool.slots[i].wake_entry.last_value >
  //       wake_pool.slots[i].wake_entry.minimum_value) {
  //     check_wait_list = true;
  //     break;
  //   }
  // We'd need a fast way of knowing which slots are valid, though.
  check_wait_list = true;

  // Refresh the wait list by checking the leading wait of each entry.
  // If the leading wait has been satisfied then we can move on to the next wait
  // and if all waits are satisfied the entry is moved to the run list.
  if (check_wait_list) {
    iree_hal_amdgpu_device_queue_scheduler_check_wait_list(scheduler);
  }

  // If any entries are asking to be rescheduled move them to the ready list.
  // Entries will be inserted into the ready list ordered by epoch such that we
  // will attempt to schedule older entries first.
  iree_hal_amdgpu_device_queue_scheduler_reschedule_entries(scheduler);

  // Attempt to reclaim kernarg ringbuffer space prior to scheduling. Our
  // retires from the previous tick or retires from other devices may be in the
  // reclaim list and we want to ensure we have the most kernarg capacity
  // available.
  iree_hal_amdgpu_device_kernarg_ringbuffer_reclaim(
      &scheduler->kernarg_ringbuffer);

  // Drain the ready list and issue all pending queue operations we can.
  // Note that if resources are exhausted we will break early before issuing
  // all operations; the next tick (usually scheduled by a retiring entry) will
  // try scheduling again.
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* ready_entry =
      scheduler->ready_list.head;
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
      prev_ready_entry = NULL;
  while (ready_entry != NULL) {
    // If the scheduler is running in exclusive mode then only allow there to be
    // one active entry at a time. If we bail here with entries remaining in the
    // ready list they'll be retried at the next tick.
    if ((scheduler->mode &
         IREE_HAL_AMDGPU_DEVICE_QUEUE_SCHEDULING_MODE_EXCLUSIVE) &&
        !iree_hal_amdgpu_device_queue_active_set_is_empty(
            &scheduler->active_set)) {
      break;  // active_set not empty in exclusive mode
    }

    // If the active set is full then skip readying any more this tick.
    // Once an entry is active it must complete so we know there will eventually
    // be space.
    if (iree_hal_amdgpu_device_queue_active_set_is_full(
            &scheduler->active_set)) {
      break;  // active_set full, wait until capacity available
    }

    // Skip entries that are waiting on kernarg space.
    const bool kernargs_capacity_available =
        iree_hal_amdgpu_device_kernarg_ringbuffer_has_capacity(
            &scheduler->kernarg_ringbuffer, ready_entry->max_kernarg_capacity);
    if (!kernargs_capacity_available) {
      if (scheduler->mode &
          IREE_HAL_AMDGPU_DEVICE_QUEUE_SCHEDULING_MODE_WORK_CONSERVING) {
        // Work-preserving mode: keep processing the ready list to see if there
        // are any that will fit within the available capacity.
        prev_ready_entry = ready_entry;
        ready_entry = prev_ready_entry->list_next;
        continue;  // kernarg ringbuffer capacity exhausted, try another entry
      } else {
        // Non-work-conserving mode; if capacity is not available then bail now
        // instead of processing other entries.
        // This blocks the head of the queue at the next ready entry.
        break;  // kernarg ringbuffers full, wait until available capacity
      }
    }

    // Unlink the entry from the ready list.
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*
        next_ready_entry = ready_entry->list_next;
    if (prev_ready_entry) {
      prev_ready_entry->list_next = next_ready_entry;
    } else {
      scheduler->ready_list.head = next_ready_entry;
    }
    if (!next_ready_entry) {
      scheduler->ready_list.tail = prev_ready_entry;
    }
    ready_entry->list_next = NULL;

    // Assign the queue entry an active set slot.
    iree_hal_amdgpu_device_queue_active_set_insert(&scheduler->active_set,
                                                   ready_entry);

    // Issue the ready-to-run queue entry. Provide the wake set but note that
    // the operation may be asynchronous and not wake anything yet.
    iree_hal_amdgpu_device_queue_issue(scheduler, ready_entry);

    // Continue walking the ready list.
    ready_entry = next_ready_entry;
  }
  scheduler->ready_list.head = scheduler->ready_list.tail = NULL;

  // If any entries have retired and need to be cleaned up then do so prior to
  // returning from the tick. This may produce new work that requires us to wake
  // again.
  iree_hal_amdgpu_device_queue_scheduler_retire_entries(scheduler);

  // Notifies all targets that may now be able to progress due to retired
  // entries. If self_wake is true it means that we ourselves have new work and
  // should restart processing after the run list is empty.
  const bool self_wake =
      iree_hal_amdgpu_device_wake_set_flush(&scheduler->wake_set);

  // To give the hardware queue some time to breathe we re-enqueue ourselves.
  // This may increase latency but makes debugging easier and ensures we don't
  // end up in an infinite loop within the tick.
  if (self_wake) {
    iree_hal_amdgpu_device_queue_scheduler_enqueue_from_control_queue(
        scheduler, IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_CONTINUE);
  }

  IREE_OCL_TRACE_ZONE_END(z0);

#if IREE_HAL_AMDGPU_TRACING_FEATURES
  // Flush the trace buffer, if needed.
  // This will contain any trace events emitted during this tick as well as any
  // imported from command buffers. The host may be notified with an interrupt.
  // NOTE: we do this after exiting the trace zone above so that the entire tick
  // zone will be visible to the host.
  if (iree_hal_amdgpu_device_trace_commit_range(scheduler->trace_buffer)) {
    // TODO(benvanik): insert a barrier on the queue to wait until the flush
    // has completed? This will add latency but ensure the next queue tick will
    // have capacity.
    iree_hsa_signal_t completion_signal = iree_hsa_signal_null();
    iree_hal_amdgpu_device_host_post_trace_flush(&scheduler->host,
                                                 completion_signal);
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURES
}

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_queue_scheduler_enqueue_packet(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler) {
  // Reserve the next packet in the target queue.
  const uint64_t packet_id = iree_hsa_queue_add_write_index(
      scheduler->control_queue, 1u,
      iree_hal_amdgpu_device_memory_order_relaxed);
  while (packet_id - iree_hsa_queue_load_read_index(
                         scheduler->control_queue,
                         iree_hal_amdgpu_device_memory_order_acquire) >=
         scheduler->control_queue->size) {
    iree_hal_amdgpu_device_yield();  // spinning
  }

  // Construct the control packet.
  // Note that the header is not written until the end so that the
  // hardware command processor stalls until we're done writing.
  const uint64_t queue_mask =
      scheduler->control_queue->size - 1;  // power of two
  IREE_OCL_GLOBAL iree_hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT
      tick_packet = scheduler->control_queue->base_address +
                    (packet_id & queue_mask) * 64;
  const iree_hal_amdgpu_device_kernel_args_t tick_args =
      scheduler->kernels->scheduler_tick;
  tick_packet->setup = tick_args.setup;
  tick_packet->workgroup_size[0] = tick_args.workgroup_size[0];
  tick_packet->workgroup_size[1] = tick_args.workgroup_size[1];
  tick_packet->workgroup_size[2] = tick_args.workgroup_size[2];
  tick_packet->reserved0 = 0;
  tick_packet->grid_size[0] = 1;
  tick_packet->grid_size[1] = 1;
  tick_packet->grid_size[2] = 1;
  tick_packet->private_segment_size = tick_args.private_segment_size;
  tick_packet->group_segment_size = tick_args.group_segment_size;
  tick_packet->kernel_object = tick_args.kernel_object;
  tick_packet->kernarg_address = scheduler->control_tick_kernarg;
  tick_packet->reserved2 = 0;

  // Mark the update packet as ready to execute. The hardware command
  // processor may begin executing it immediately after performing the
  // atomic swap.
  uint16_t tick_header = IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH
                         << IREE_HSA_PACKET_HEADER_TYPE;

  // Barrier bit set to ensure only one scheduler tick runs at a time.
  tick_header |= 1 << IREE_HSA_PACKET_HEADER_BARRIER;

  // TODO(benvanik): scope to agent if enqueuing to self.
  tick_header |= IREE_HSA_FENCE_SCOPE_SYSTEM
                 << IREE_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  tick_header |= IREE_HSA_FENCE_SCOPE_SYSTEM
                 << IREE_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;

  const uint32_t tick_header_setup =
      tick_header | (uint32_t)(tick_packet->setup << 16);
  iree_hal_amdgpu_device_atomic_store_explicit(
      (IREE_OCL_GLOBAL iree_hal_amdgpu_device_atomic_uint32_t*)tick_packet,
      tick_header_setup, iree_hal_amdgpu_device_memory_order_release,
      iree_hal_amdgpu_device_memory_scope_all_svm_devices);

  // Signal the queue doorbell indicating the packet has been updated.
  iree_hsa_signal_store(scheduler->control_queue->doorbell_signal, packet_id,
                        iree_hal_amdgpu_device_memory_order_relaxed);
}

void iree_hal_amdgpu_device_queue_scheduler_enqueue_from_control_queue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    iree_hal_amdgpu_device_queue_tick_action_set_t action_set) {
  IREE_OCL_TRACE_BUFFER_DEFINE(scheduler->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);
  IREE_OCL_TRACE_ZONE_APPEND_VALUE_I64(z0, action_set);

  // OR in the requester action set to the pending scheduler set.
  // This must happen after all dependent resources have been updated (such as
  // the retire ringbuffer) as the tick may run immediately after this point.
  if (iree_hal_amdgpu_device_atomic_fetch_or_explicit(
          &scheduler->tick_action_set, action_set,
          iree_hal_amdgpu_device_memory_order_release,
          iree_hal_amdgpu_device_memory_scope_all_svm_devices)) {
    // If any action was pending it means a tick was enqueued. No-op this one.
    // The next tick that runs will pick up any of the work that was requested
    // by the caller.
    IREE_OCL_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "nop");
    IREE_OCL_TRACE_ZONE_END(z0);
    return;
  }
  IREE_OCL_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "enqueue");

  iree_hal_amdgpu_device_queue_scheduler_enqueue_packet(scheduler);

  IREE_OCL_TRACE_ZONE_END(z0);
}

void iree_hal_amdgpu_device_queue_scheduler_enqueue_from_execution_queue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    iree_hal_amdgpu_device_queue_tick_action_set_t action_set) {
  // OR in the requester action set to the pending scheduler set.
  // This must happen after all dependent resources have been updated (such as
  // the retire ringbuffer) as the tick may run immediately after this point.
  if (iree_hal_amdgpu_device_atomic_fetch_or_explicit(
          &scheduler->tick_action_set, action_set,
          iree_hal_amdgpu_device_memory_order_release,
          iree_hal_amdgpu_device_memory_scope_all_svm_devices)) {
    // If any action was pending it means a tick was enqueued. No-op this one.
    // The next tick that runs will pick up any of the work that was requested
    // by the caller.
    return;
  }

  // Enqueue the scheduler packet on its control queue.
  iree_hal_amdgpu_device_queue_scheduler_enqueue_packet(scheduler);
}

void iree_hal_amdgpu_device_queue_scheduler_reschedule_from_execution_queue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    uint64_t scheduler_queue_entry) {
  // Mark the queue entry as needing to be rescheduled in the active set. The
  // next tick that processes the reschedule list will move the entry to the
  // ready list.
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* entry =
      (IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*)
          scheduler_queue_entry;
  iree_hal_amdgpu_device_queue_active_set_reschedule(&scheduler->active_set,
                                                     entry);

  // Possibly wake the scheduler on the control queue.
  iree_hal_amdgpu_device_queue_scheduler_enqueue_from_execution_queue(
      scheduler, IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_RESCHEDULE);
}

void iree_hal_amdgpu_device_queue_scheduler_retire_from_execution_queue(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_scheduler_t* IREE_OCL_RESTRICT
        scheduler,
    uint64_t scheduler_queue_entry) {
  // Mark the queue entry as retired in the active set. The next tick that
  // processes the retire list will retire the entry.
  // The entry may be immediately cleaned up after this call.
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t* entry =
      (IREE_OCL_GLOBAL iree_hal_amdgpu_device_queue_entry_header_t*)
          scheduler_queue_entry;
  iree_hal_amdgpu_device_queue_active_set_retire(&scheduler->active_set, entry);

  // Possibly wake the scheduler on the control queue.
  iree_hal_amdgpu_device_queue_scheduler_enqueue_from_execution_queue(
      scheduler, IREE_HAL_AMDGPU_DEVICE_QUEUE_TICK_ACTION_RETIRE);
}
