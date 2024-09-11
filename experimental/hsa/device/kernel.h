// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_DEVICE_KERNEL_H_
#define IREE_HAL_DRIVERS_HSA_DEVICE_KERNEL_H_

#include "experimental/hsa/device/support/opencl.h"

//===----------------------------------------------------------------------===//
// iree_amdgpu_kernel_args_t
//===----------------------------------------------------------------------===//

// Kernel arguments used for fixed-size kernels.
// This must match what the kernel was compiled to support.
typedef struct iree_amdgpu_kernel_args_t {
  // Opaque handle to the kernel object to execute.
  uint64_t kernel_object;
  // hsa_kernel_dispatch_packet_setup_t (grid dimension count).
  uint16_t setup;
  // X dimension of work-group, in work-items. Must be greater than 0.
  uint16_t workgroup_size_x;
  // Y dimension of work-group, in work-items. Must be greater than 0.
  // If the grid has 1 dimension, the only valid value is 1.
  uint16_t workgroup_size_y;
  // Z dimension of work-group, in work-items. Must be greater than 0.
  // If the grid has 1 or 2 dimensions, the only valid value is 1.
  uint16_t workgroup_size_z;
  // Size in bytes of private memory allocation request (per work-item).
  uint32_t private_segment_size;
  // Size in bytes of group memory allocation request (per work-group). Must
  // not be less than the sum of the group memory used by the kernel (and the
  // functions it calls directly or indirectly) and the dynamically allocated
  // group segment variables.
  uint32_t group_segment_size;
} iree_amdgpu_kernel_args_t;

//===----------------------------------------------------------------------===//
// iree_amdgpu_kernels_t
//===----------------------------------------------------------------------===//

// Opaque handles used to launch builtin kernels.
// Stored on the command buffer as they are constant for the lifetime of the
// program and we may have command buffers opt into different DMA modes.
typedef struct iree_amdgpu_kernels_t {
  // `iree_amdgpu_queue_scheduler_tick` kernel.
  iree_amdgpu_kernel_args_t scheduler_tick;
  // `iree_amdgpu_command_buffer_issue_block` kernel.
  iree_amdgpu_kernel_args_t issue_block;
  // `iree_amdgpu_command_buffer_workgroup_count_update` kernel.
  iree_amdgpu_kernel_args_t workgroup_count_update;
  // Kernels used to implement DMA-like operations.
  struct {
    iree_amdgpu_kernel_args_t fill_x1;   // iree_amdgpu_buffer_fill_x1
    iree_amdgpu_kernel_args_t fill_x2;   // iree_amdgpu_buffer_fill_x2
    iree_amdgpu_kernel_args_t fill_x4;   // iree_amdgpu_buffer_fill_x4
    iree_amdgpu_kernel_args_t fill_x8;   // iree_amdgpu_buffer_fill_x8
    iree_amdgpu_kernel_args_t copy_x1;   // iree_amdgpu_buffer_copy_x1
    iree_amdgpu_kernel_args_t copy_x2;   // iree_amdgpu_buffer_copy_x2
    iree_amdgpu_kernel_args_t copy_x4;   // iree_amdgpu_buffer_copy_x4
    iree_amdgpu_kernel_args_t copy_x8;   // iree_amdgpu_buffer_copy_x8
    iree_amdgpu_kernel_args_t copy_x64;  // iree_amdgpu_buffer_copy_x64
  } blit;
} iree_amdgpu_kernels_t;

#endif  // IREE_HAL_DRIVERS_HSA_DEVICE_KERNEL_H_
