// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hsa/device/host.h"

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

void iree_amdgpu_host_enqueue(
    IREE_CL_GLOBAL iree_amdgpu_host_t* IREE_CL_RESTRICT host, uint16_t type,
    uint64_t return_address, uint64_t arg0, uint64_t arg1, uint64_t arg2,
    uint64_t arg3, iree_hsa_signal_t completion_signal) {
  // Reserve a packet write index and wait for it to become available in cases
  // where the queue is exhausted.
  uint64_t packet_id = iree_hsa_queue_add_write_index(
      host->queue, 1u, iree_amdgpu_memory_order_relaxed);
  while (packet_id - iree_hsa_queue_load_read_index(
                         host->queue, iree_amdgpu_memory_order_acquire) >=
         queue->size) {
    iree_amdgpu_yield();  // spinning
  }
  const uint64_t queue_mask = queue->size - 1;  // power of two
  IREE_CL_GLOBAL hsa_agent_dispatch_packet_t* agent_packet =
      queue->base_address + (packet_id & queue_mask) * 64;

  // Populate all of the packet besides the header.
  agent_packet->reserved0 = 0;
  agent_packet->return_address = (void*)return_address;
  agent_packet->arg[0] = arg0;
  agent_packet->arg[1] = arg1;
  agent_packet->arg[2] = arg2;
  agent_packet->arg[3] = arg3;
  agent_packet->reserved2 = 0;
  agent_packet->completion_signal = completion_signal;

  // Populate the header and release the packet to the queue.
  // Note that we need to release to all devices so that the host can see it.
  uint16_t header = HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE;
  header |= 0 << HSA_PACKET_HEADER_BARRIER;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
  uint32_t header_type = header | (type << 16);
  iree_amdgpu_atomic_store_explicit((IREE_CL_GLOBAL uint32_t*)agent_packet,
                                    header_type,
                                    iree_amdgpu_memory_order_release,
                                    iree_amdgpu_memory_scope_all_svm_devices);

  // Signal the queue doorbell.
  iree_hsa_signal_store(host->queue->doorbell_signal, packet_id,
                        iree_amdgpu_memory_order_relaxed);
}

void iree_amdgpu_host_post_release(
    IREE_CL_GLOBAL iree_amdgpu_host_t* IREE_CL_RESTRICT host,
    uint64_t resource0, uint64_t resource1, uint64_t resource2,
    uint64_t resource3) {
  iree_amdgpu_host_enqueue(host, IREE_AMDGPU_HOST_CALL_POST_RELEASE,
                           /*return_address=*/0, resource0, resource1,
                           resource2, resource3, iree_hsa_signal_null());
}

void iree_amdgpu_host_post_error(
    IREE_CL_GLOBAL iree_amdgpu_host_t* IREE_CL_RESTRICT host, uint64_t code,
    uint64_t arg0, uint64_t arg1) {
  iree_amdgpu_host_enqueue(
      host, IREE_AMDGPU_HOST_CALL_POST_ERROR, /*return_address=*/0,
      /*reserved=*/0, code, arg0, arg1, iree_hsa_signal_null());
}

void iree_amdgpu_host_post_signal(
    IREE_CL_GLOBAL iree_amdgpu_host_t* IREE_CL_RESTRICT host,
    IREE_CL_GLOBAL iree_amdgpu_semaphore_t* IREE_CL_RESTRICT semaphore,
    uint64_t payload) {
  iree_amdgpu_host_enqueue(host, IREE_AMDGPU_HOST_CALL_POST_SIGNAL,
                           /*return_address=*/0, (uint64_t)semaphore, payload,
                           /*unused=*/0,
                           /*unused=0*/, iree_hsa_signal_null());
}
