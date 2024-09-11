// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hsa/device/buffer.h"

//===----------------------------------------------------------------------===//
// iree_amdgpu_buffer_fill_*
//===----------------------------------------------------------------------===//

__kernel void iree_amdgpu_buffer_fill_x1(
    IREE_CL_GLOBAL void* IREE_CL_RESTRICT target_ptr, const uint64_t length,
    const uint8_t pattern) {
  // DO NOT SUBMIT
  // runtime/src/iree/hal/drivers/metal/builtin/fill_buffer_generic.metal
  // fill_buffer_1byte
}

__kernel void iree_amdgpu_buffer_fill_x2(
    IREE_CL_GLOBAL void* IREE_CL_RESTRICT target_ptr, const uint64_t length,
    const uint16_t pattern) {
  // DO NOT SUBMIT
  // runtime/src/iree/hal/drivers/metal/builtin/fill_buffer_generic.metal
  // fill_buffer_2byte
}

__kernel void iree_amdgpu_buffer_fill_x4(
    IREE_CL_GLOBAL void* IREE_CL_RESTRICT target_ptr, const uint64_t length,
    const uint32_t pattern) {
  // DO NOT SUBMIT
  // runtime/src/iree/hal/drivers/metal/builtin/fill_buffer_generic.metal
  // fill_buffer_4byte
}

__kernel void iree_amdgpu_buffer_fill_x8(
    IREE_CL_GLOBAL void* IREE_CL_RESTRICT target_ptr, const uint64_t length,
    const uint64_t pattern) {
  // DO NOT SUBMIT
  // runtime/src/iree/hal/drivers/metal/builtin/fill_buffer_generic.metal
  // fill_buffer_8byte
}

void iree_amdgpu_buffer_fill_enqueue(
    IREE_CL_GLOBAL void* target_ptr, const uint64_t length,
    const uint64_t pattern, const uint8_t pattern_length,
    IREE_CL_GLOBAL const iree_amdgpu_kernels_t* kernels,
    IREE_CL_GLOBAL uint64_t* IREE_CL_RESTRICT kernarg_ptr,
    IREE_CL_GLOBAL iree_amdgpu_queue_t* IREE_CL_RESTRICT queue) {
  // Reserve the next packet in the queue.
  // DO NOT SUBMIT
  // reserve write_index
  const uint32_t queue_index = 0;

  // Emplace the dispatch packet into the queue.
  // Note that the dispatch may begin executing immediately.
  iree_amdgpu_buffer_fill_emplace(target_ptr, length, pattern, pattern_length,
                                  kernels, kernarg_ptr, queue, queue_index);

  // Signal the queue doorbell indicating the packet has been updated.
  // DO NOT SUBMIT
  // knock doorbell queue_index
}

void iree_amdgpu_buffer_fill_emplace(
    IREE_CL_GLOBAL void* target_ptr, const uint64_t length,
    const uint64_t pattern, const uint8_t pattern_length,
    IREE_CL_GLOBAL const iree_amdgpu_kernels_t* kernels,
    IREE_CL_GLOBAL uint64_t* IREE_CL_RESTRICT kernarg_ptr,
    IREE_CL_GLOBAL iree_amdgpu_queue_t* IREE_CL_RESTRICT queue,
    const uint32_t queue_index) {
  // Update kernargs (same for all kernels).
  kernarg_ptr[0] = target_ptr;
  kernarg_ptr[1] = length;
  kernarg_ptr[2] = pattern;
  // DO NOT SUBMIT implicit opencl args

  // Select the kernel for the fill operation.
  const iree_amdgpu_kernel_args_t kernel_args;
  uint64_t block_size = 0;
  switch (pattern_length) {
    case 1:
      kernel_args = kernels->blit.fill_x1;
      block_size = 1;
      break;
    case 2:
      kernel_args = kernels->blit.fill_x2;
      block_size = 1;
      break;
    case 4:
      kernel_args = kernels->blit.fill_x4;
      block_size = 1;
      break;
    case 8:
      kernel_args = kernels->blit.fill_x8;
      block_size = 1;
      break;
  }

  // Populate the packet.
  // DO NOT SUBMIT queue_index
  IREE_CL_GLOBAL hsa_kernel_dispatch_packet_t* dispatch_packet = NULL;
  dispatch_packet->setup = kernel_args.setup;
  dispatch_packet->workgroup_size_x = kernel_args.workgroup_size_x;
  dispatch_packet->workgroup_size_y = kernel_args.workgroup_size_y;
  dispatch_packet->workgroup_size_z = kernel_args.workgroup_size_z;
  dispatch_packet->reserved0 = 0;
  dispatch_packet->grid_size_x = 0;  // DO NOT SUBMIT block count?
  dispatch_packet->grid_size_y = 1;
  dispatch_packet->grid_size_z = 1;
  dispatch_packet->private_segment_size = kernel_args.private_segment_size;
  dispatch_packet->group_segment_size = kernel_args.group_segment_size;
  dispatch_packet->kernel_object = kernel_args.kernel_object;
  dispatch_packet->kernarg_address = kernarg_ptr;
  dispatch_packet->reserved2 = 0;

  // DO NOT SUBMIT
  // dispatch_packet->completion_signal = ;

  // Mark the packet as ready for execution by swapping the header.
  // At this point the hardware command processor may begin executing
  // immediately.

  // DO NOT SUBMIT
  // barrier bit?
}

//===----------------------------------------------------------------------===//
// iree_amdgpu_buffer_copy_*
//===----------------------------------------------------------------------===//

__kernel void iree_amdgpu_buffer_copy_x1(
    IREE_CL_GLOBAL const uint8_t* IREE_CL_RESTRICT source_ptr,
    IREE_CL_GLOBAL uint8_t* IREE_CL_RESTRICT target_ptr,
    const uint64_t length) {
  // DO NOT SUBMIT
  // runtime/src/iree/hal/drivers/metal/builtin/copy_buffer_generic.metal
  // copy_buffer_1byte
}

__kernel void iree_amdgpu_buffer_copy_x2(
    IREE_CL_GLOBAL const uint16_t* IREE_CL_RESTRICT source_ptr,
    IREE_CL_GLOBAL uint16_t* IREE_CL_RESTRICT target_ptr,
    const uint64_t length) {
  // DO NOT SUBMIT
}

__kernel void iree_amdgpu_buffer_copy_x4(
    IREE_CL_GLOBAL const uint16_t* IREE_CL_RESTRICT source_ptr,
    IREE_CL_GLOBAL uint16_t* IREE_CL_RESTRICT target_ptr,
    const uint64_t length) {
  // DO NOT SUBMIT
}

__kernel void iree_amdgpu_buffer_copy_x8(
    IREE_CL_GLOBAL const uint16_t* IREE_CL_RESTRICT source_ptr,
    IREE_CL_GLOBAL uint16_t* IREE_CL_RESTRICT target_ptr,
    const uint64_t length) {
  // DO NOT SUBMIT
}

// TODO(benvanik): experiment with best widths for bulk transfers.
__kernel void iree_amdgpu_buffer_copy_x64(
    IREE_CL_GLOBAL const uint16_t* IREE_CL_RESTRICT source_ptr,
    IREE_CL_GLOBAL uint16_t* IREE_CL_RESTRICT target_ptr,
    const uint64_t length) {
  // DO NOT SUBMIT
}

void iree_amdgpu_buffer_copy_enqueue(
    IREE_CL_GLOBAL const void* source_ptr, IREE_CL_GLOBAL void* target_ptr,
    const uint64_t length, IREE_CL_GLOBAL const iree_amdgpu_kernels_t* kernels,
    IREE_CL_GLOBAL uint64_t* IREE_CL_RESTRICT kernarg_ptr,
    IREE_CL_GLOBAL iree_amdgpu_queue_t* IREE_CL_RESTRICT queue) {
  // Reserve the next packet in the queue.
  // DO NOT SUBMIT
  // reserve write_index
  const uint32_t queue_index = 0;

  // Emplace the dispatch packet into the queue.
  // Note that the dispatch may begin executing immediately.
  iree_amdgpu_buffer_copy_emplace(source_ptr, target_ptr, length, kernels,
                                  kernarg_ptr, queue, queue_index);

  // Signal the queue doorbell indicating the packet has been updated.
  // DO NOT SUBMIT
  // knock doorbell queue_index
}

// TODO(benvanik): experiment with enqueuing SDMA somehow (may need to take a
// DMA queue as well as the dispatch queue).
void iree_amdgpu_buffer_copy_emplace(
    IREE_CL_GLOBAL const void* source_ptr, IREE_CL_GLOBAL void* target_ptr,
    const uint64_t length, IREE_CL_GLOBAL const iree_amdgpu_kernels_t* kernels,
    IREE_CL_GLOBAL uint64_t* IREE_CL_RESTRICT kernarg_ptr,
    IREE_CL_GLOBAL iree_amdgpu_queue_t* IREE_CL_RESTRICT queue,
    const uint32_t queue_index) {
  // Update kernargs (same for all kernels).
  kernarg_ptr[0] = source_ptr;
  kernarg_ptr[1] = target_ptr;
  kernarg_ptr[2] = length;
  // DO NOT SUBMIT implicit opencl args

  // Select the kernel for the copy operation.
  // TODO(benvanik): switch kernel based on source/target/length alignment.
  const iree_amdgpu_kernel_args_t kernel_args = kernels->blit.copy_x1;
  const uint64_t block_size = 128;

  // Populate the packet.
  // DO NOT SUBMIT queue_index
  IREE_CL_GLOBAL hsa_kernel_dispatch_packet_t* dispatch_packet = NULL;
  dispatch_packet->setup = kernel_args.setup;
  dispatch_packet->workgroup_size_x = kernel_args.workgroup_size_x;
  dispatch_packet->workgroup_size_y = kernel_args.workgroup_size_y;
  dispatch_packet->workgroup_size_z = kernel_args.workgroup_size_z;
  dispatch_packet->reserved0 = 0;
  dispatch_packet->grid_size_x = 0;  // DO NOT SUBMIT block count?
  dispatch_packet->grid_size_y = 1;
  dispatch_packet->grid_size_z = 1;
  dispatch_packet->private_segment_size = kernel_args.private_segment_size;
  dispatch_packet->group_segment_size = kernel_args.group_segment_size;
  dispatch_packet->kernel_object = kernel_args.kernel_object;
  dispatch_packet->kernarg_address = kernarg_ptr;
  dispatch_packet->reserved2 = 0;

  // DO NOT SUBMIT
  // dispatch_packet->completion_signal = ;

  // Mark the packet as ready for execution by swapping the header.
  // At this point the hardware command processor may begin executing
  // immediately.

  // DO NOT SUBMIT
  // barrier bit?
}
