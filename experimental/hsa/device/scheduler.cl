// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hsa/device/command_processor.h"

//===----------------------------------------------------------------------===//
// Queue Operations
//===----------------------------------------------------------------------===//

static IREE_CL_ATTRIBUTE_ALWAYS_INLINE void iree_amdgpu_queue_issue_initialize(
    IREE_CL_GLOBAL iree_amdgpu_queue_scheduler_t* IREE_CL_RESTRICT scheduler,
    IREE_CL_GLOBAL const iree_amdgpu_device_queue_initialize_args_t*
        IREE_CL_RESTRICT args) {
  // Initialize the signal pool with the provided HSA signals.
  iree_amdgpu_signal_pool_initialize(scheduler->signal_pool, args->signal_count,
                                     args->signals);
}

static IREE_CL_ATTRIBUTE_ALWAYS_INLINE void
iree_amdgpu_queue_issue_deinitialize(
    IREE_CL_GLOBAL iree_amdgpu_queue_scheduler_t* IREE_CL_RESTRICT scheduler,
    IREE_CL_GLOBAL const iree_amdgpu_device_queue_deinitialize_args_t*
        IREE_CL_RESTRICT args) {
  //
}

static IREE_CL_ATTRIBUTE_ALWAYS_INLINE void iree_amdgpu_queue_issue_alloca(
    IREE_CL_GLOBAL iree_amdgpu_queue_scheduler_t* IREE_CL_RESTRICT scheduler,
    IREE_CL_GLOBAL const iree_amdgpu_device_queue_alloca_args_t*
        IREE_CL_RESTRICT args) {
  // check satisfied
  // lookup pool
  // switch on pool type
  // call pool handler method
  //   if host needed then pass queue
  //   set suspend state?
}

static IREE_CL_ATTRIBUTE_ALWAYS_INLINE void iree_amdgpu_queue_issue_dealloca(
    IREE_CL_GLOBAL iree_amdgpu_queue_scheduler_t* IREE_CL_RESTRICT scheduler,
    IREE_CL_GLOBAL const iree_amdgpu_device_queue_dealloca_args_t*
        IREE_CL_RESTRICT args) {
  //
}

static IREE_CL_ATTRIBUTE_ALWAYS_INLINE void iree_amdgpu_queue_issue_fill(
    IREE_CL_GLOBAL iree_amdgpu_queue_scheduler_t* IREE_CL_RESTRICT scheduler,
    IREE_CL_GLOBAL const iree_amdgpu_device_queue_fill_args_t* IREE_CL_RESTRICT
        args) {
  // check satisfied
  // enqueue blit kernel (today)
  // barrier + enqueue signal (if needed)
}

static IREE_CL_ATTRIBUTE_ALWAYS_INLINE void iree_amdgpu_queue_issue_copy(
    IREE_CL_GLOBAL iree_amdgpu_queue_scheduler_t* IREE_CL_RESTRICT scheduler,
    IREE_CL_GLOBAL const iree_amdgpu_device_queue_copy_args_t* IREE_CL_RESTRICT
        args) {
  // check satisfied
  // enqueue blit kernel (today)
  // barrier + enqueue signal (if needed)
}

static IREE_CL_ATTRIBUTE_ALWAYS_INLINE void iree_amdgpu_queue_issue_execute(
    IREE_CL_GLOBAL iree_amdgpu_queue_scheduler_t* IREE_CL_RESTRICT scheduler,
    IREE_CL_GLOBAL const iree_amdgpu_device_queue_execute_args_t*
        IREE_CL_RESTRICT args) {
  //
  // enqueue command buffer launch kernel
  //   could do first chunk inline?
  // barrier + enqueue signal (if needed)
}

static IREE_CL_ATTRIBUTE_ALWAYS_INLINE void iree_amdgpu_queue_issue_barrier(
    IREE_CL_GLOBAL iree_amdgpu_queue_scheduler_t* IREE_CL_RESTRICT scheduler,
    IREE_CL_GLOBAL const iree_amdgpu_device_queue_barrier_args_t*
        IREE_CL_RESTRICT args) {
  //
  // barrier + enqueue signal (if needed)
}

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

__kernel IREE_CL_ATTRIBUTE_SINGLE_WORK_ITEM void
iree_amdgpu_queue_scheduler_tick(
    IREE_CL_GLOBAL iree_amdgpu_queue_scheduler_t* IREE_CL_RESTRICT scheduler,
    iree_amdgpu_queue_scheduling_reason_t reason, uint64_t reason_arg) {
  // DO NOT SUBMIT
  //
  // if reason is COMMAND_BUFFER_RETURN:
  //   check reason_arg against execution state and cleanup
  //
  // dequeue work from queue and try to run it?
  // enqueue as much as possible?
  // need to chain signals

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

void iree_amdgpu_queue_scheduler_enqueue(
    IREE_CL_GLOBAL iree_amdgpu_queue_scheduler_t* IREE_CL_RESTRICT scheduler,
    iree_amdgpu_queue_scheduling_reason_t reason, uint64_t reason_arg) {
  // Pass the reason.
  IREE_CL_GLOBAL void* control_kernargs[3] = {
      scheduler,
      reason,
      reason_arg,
  };
  // DO NOT SUBMIT implicit opencl args
  IREE_CL_GLOBAL void* control_kernarg_ptr = state->control_kernarg_storage;
  memcpy(kernarg_ptr, control_kernargs, sizeof(control_kernargs));

  // Construct the control packet.
  // Note that the header is not written until the end so that the
  // hardware command processor stalls until we're done writing.
  const iree_amdgpu_kernel_args_t tick_args = scheduler->kernels.scheduler_tick;
  // DO NOT SUBMIT queue_index
  IREE_CL_GLOBAL hsa_kernel_dispatch_packet_t* tick_packet = NULL;
  tick_packet->setup = control_args.setup;
  tick_packet->workgroup_size_x = control_args.workgroup_size_x;
  tick_packet->workgroup_size_y = control_args.workgroup_size_y;
  tick_packet->workgroup_size_z = control_args.workgroup_size_z;
  tick_packet->reserved0 = 0;
  tick_packet->grid_size_x = 1;
  tick_packet->grid_size_y = 1;
  tick_packet->grid_size_z = 1;
  tick_packet->private_segment_size = control_args.private_segment_size;
  tick_packet->group_segment_size = control_args.kernel_args.group_segment_size;
  tick_packet->kernel_object = control_args.kernel_object;
  tick_packet->kernarg_address = kernarg_ptr;
  tick_packet->reserved2 = 0;

  // DO NOT SUBMIT
  // tick_packet->completion_signal = ;

  // NOTE: we implicitly assume IREE_AMDGPU_CMD_FLAG_QUEUE_AWAIT_BARRIER and may
  // always want to do that - though _technically_ we should be processing all
  // commands submitted in multiple command buffers as if they were submitted in
  // a single one the granularity is such that the 0.001% of potential
  // concurrency is not worth the risk. If we did want to allow command buffers
  // to execute concurrently we'd need to probably re-evaluate
  // cross-command-buffer event handles such that a signal in one could be used
  // in another and that's currently out of scope.

  // Mark the update packet as ready to execute. The hardware command processor
  // may begin executing it immediately after performing the atomic swap.
  //
  // DO NOT SUBMIT
  // tick_packet->header
  // barrier bit
  // fence bits type INVALID -> KERNEL_DISPATCH
}
