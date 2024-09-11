// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// NOTE: builtins are defined in the LLVM AMDGPU device library that is linked
// into the device runtime. We need to redefine them as externs here as they are
// not defined in any accessible headers.
//
// Sources:
// https://github.com/ROCm/rocMLIR/blob/develop/external/llvm-project/amd/device-libs/README.md

#ifndef IREE_HAL_DRIVERS_HSA_DEVICE_SUPPORT_OPENCL_H_
#define IREE_HAL_DRIVERS_HSA_DEVICE_SUPPORT_OPENCL_H_

//===----------------------------------------------------------------------===//
// Compiler Configuration
//===----------------------------------------------------------------------===//

#if defined(__OPENCL_C_VERSION__)
#define IREE_CL_TARGET_DEVICE 1
#else
#define IREE_CL_TARGET_HOST 1
#endif  // __OPENCL_C_VERSION__

//===----------------------------------------------------------------------===//
// OpenCL Attributes
//===----------------------------------------------------------------------===//

#if defined(IREE_CL_TARGET_DEVICE)

#define IREE_CL_RESTRICT __restrict__
#define IREE_CL_ALIGNAS(x) __attribute__((aligned(x)))
#define IREE_CL_GLOBAL __global

#define IREE_CL_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define IREE_CL_ATTRIBUTE_CONST __attribute__((const))
#define IREE_CL_ATTRIBUTE_SINGLE_WORK_ITEM __attribute__((work_group_size_hint(1, 1, 1))
#define IREE_CL_ATTRIBUTE_PACKED __attribute__((__packed__))

#define IREE_CL_LIKELY(x) (__builtin_expect(!!(x), 1))
#define IREE_CL_UNLIKELY(x) (__builtin_expect(!!(x), 0))

#define IREE_CL_STATIC_ASSERT(x, y) IREE_CL_STATIC_ASSERT__(x, __COUNTER__)
#define IREE_CL_STATIC_ASSERT__(x, y) IREE_CL_STATIC_ASSERT___(x, y)
#define IREE_CL_STATIC_ASSERT___(x, y) \
  typedef char __assert_##y[(x) ? 1 : -1] __attribute__((__unused__))

#else

#define IREE_CL_RESTRICT IREE_RESTRICT
#define IREE_CL_ALIGNAS(x) iree_alignas(x)
#define IREE_CL_GLOBAL

#define IREE_CL_ATTRIBUTE_ALWAYS_INLINE IREE_ATTRIBUTE_ALWAYS_INLINE
#define IREE_CL_ATTRIBUTE_CONST
#define IREE_CL_ATTRIBUTE_SINGLE_WORK_ITEM
#define IREE_CL_ATTRIBUTE_PACKED IREE_ATTRIBUTE_PACKED

#define IREE_CL_LIKELY(x) IREE_LIKELY(x)
#define IREE_CL_UNLIKELY(x) IREE_UNLIKELY(x)

#define IREE_CL_STATIC_ASSERT(x, y) static_assert(x, y)

#endif  // IREE_CL_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// Alignment
//===----------------------------------------------------------------------===//

#define IREE_CL_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define IREE_CL_MAX(a, b) (((a) > (b)) ? (a) : (b))

#define IREE_CL_CEIL_DIV(lhs, rhs) (((lhs) + (rhs) - 1) / (rhs))

//===----------------------------------------------------------------------===//
// OpenCL Atomics
//===----------------------------------------------------------------------===//

#define iree_amdgpu_destructive_interference_size 64
#define iree_amdgpu_constructive_interference_size 64

#if defined(IREE_CL_TARGET_DEVICE)

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef uint32_t iree_amdgpu_memory_order_t;
#define iree_amdgpu_memory_order_relaxed __ATOMIC_RELAXED
#define iree_amdgpu_memory_order_acquire __ATOMIC_ACQUIRE
#define iree_amdgpu_memory_order_release __ATOMIC_RELEASE
#define iree_amdgpu_memory_order_acq_rel __ATOMIC_ACQ_REL
#define iree_amdgpu_memory_order_seq_cst __ATOMIC_SEQ_CST

#define iree_amdgpu_memory_scope_work_item memory_scope_work_item
#define iree_amdgpu_memory_scope_work_group memory_scope_work_group
#define iree_amdgpu_memory_scope_device memory_scope_device
#define iree_amdgpu_memory_scope_all_svm_devices memory_scope_all_svm_devices
#define iree_amdgpu_memory_scope_sub_group memory_scope_sub_group

#define IREE_AMDGPU_ATOMIC_INIT(object, value) atomic_init((object), (value))

typedef atomic_int iree_amdgpu_atomic_int32_t;
typedef atomic_long iree_amdgpu_atomic_int64_t;
typedef atomic_uint iree_amdgpu_atomic_uint32_t;
typedef atomic_ulong iree_amdgpu_atomic_uint64_t;

#define iree_amdgpu_atomic_load_explicit(object, memory_order, memory_scope) \
  __opencl_atomic_load((object), (memory_order), (memory_scope))
#define iree_amdgpu_atomic_store_explicit(object, desired, memory_order, \
                                          memory_scope)                  \
  __opencl_atomic_store((object), (desired), (memory_order), (memory_scope))

#define iree_amdgpu_atomic_compare_exchange_weak_explicit(               \
    object, expected, desired, memory_order_success, memory_order_fail,  \
    memory_scope)                                                        \
  __opencl_atomic_compare_exchange_weak((object), (expected), (desired), \
                                        (memory_order_success),          \
                                        (memory_order_fail), (memory_scope))
#define iree_amdgpu_atomic_compare_exchange_strong_explicit(               \
    object, expected, desired, memory_order_success, memory_order_fail,    \
    memory_scope)                                                          \
  __opencl_atomic_compare_exchange_strong((object), (expected), (desired), \
                                          (memory_order_success),          \
                                          (memory_order_fail), (memory_scope))

#else

#define IREE_AMDGPU_ATOMIC_INIT(object, value) \
  *(object) = IREE_ATOMIC_VAR_INIT(value)

typedef iree_atomic_int32_t iree_amdgpu_atomic_int32_t;
typedef iree_atomic_int64_t iree_amdgpu_atomic_int64_t;
typedef iree_atomic_uint32_t iree_amdgpu_atomic_uint32_t;
typedef iree_atomic_uint64_t iree_amdgpu_atomic_uint64_t;

#endif  // IREE_CL_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// OpenCL Dispatch ABI
//===----------------------------------------------------------------------===//

#if defined(IREE_CL_TARGET_DEVICE)

extern IREE_CL_ATTRIBUTE_CONST size_t __ockl_get_global_id(unsigned dim);
extern IREE_CL_ATTRIBUTE_CONST size_t __ockl_get_local_id(unsigned dim);
extern IREE_CL_ATTRIBUTE_CONST size_t __ockl_get_group_id(unsigned dim);
extern IREE_CL_ATTRIBUTE_CONST size_t __ockl_get_local_size(unsigned dim);
extern IREE_CL_ATTRIBUTE_CONST size_t __ockl_get_num_groups(unsigned dim);

#define iree_amdgpu_global_id_x() __ockl_get_global_id(0)
#define iree_amdgpu_global_id_y() __ockl_get_global_id(1)
#define iree_amdgpu_global_id_z() __ockl_get_global_id(2)

#define iree_amdgpu_group_id_x() __ockl_get_group_id(0)
#define iree_amdgpu_group_id_y() __ockl_get_group_id(1)
#define iree_amdgpu_group_id_z() __ockl_get_group_id(2)

#define iree_amdgpu_group_count_x() __ockl_get_num_groups(0)
#define iree_amdgpu_group_count_y() __ockl_get_num_groups(1)
#define iree_amdgpu_group_count_z() __ockl_get_num_groups(2)

#define iree_amdgpu_local_id_x() __ockl_get_local_id(0)
#define iree_amdgpu_local_id_y() __ockl_get_local_id(1)
#define iree_amdgpu_local_id_z() __ockl_get_local_id(2)

#define iree_amdgpu_workgroup_size_x() __ockl_get_local_size(0)
#define iree_amdgpu_workgroup_size_y() __ockl_get_local_size(1)
#define iree_amdgpu_workgroup_size_z() __ockl_get_local_size(2)

extern IREE_CL_ATTRIBUTE_CONST __constant void* iree_amdgcn_dispatch_ptr(
    void) __asm("llvm.amdgcn.dispatch.ptr");
extern IREE_CL_ATTRIBUTE_CONST __constant void* iree_amdgcn_implicitarg_ptr(
    void) __asm("llvm.amdgcn.implicitarg.ptr");

#endif  // IREE_CL_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// Timing
//===----------------------------------------------------------------------===//

#if defined(IREE_CL_TARGET_DEVICE)

// Tick in the agent domain.
// This can be converted to the system domain for correlation across agents and
// the host with hsa_amd_profiling_convert_tick_to_system_domain.
typedef uint64_t iree_amdgpu_tick_t;

extern iree_amdgpu_tick_t __builtin_readsteadycounter(void);
extern void __builtin_amdgcn_s_sleep(int);

// Returns a tick in the agent domain.
// This can be converted to the system domain for correlation across agents and
// the host with hsa_amd_profiling_convert_tick_to_system_domain. The value is
// the same as that placed into signal start_ts/end_ts by the command processor.
#define iree_amdgpu_timestamp __builtin_readsteadycounter

// Sleeps the current thread for some "short" amount of time.
// This maps to the S_SLEEP instruction that varies on different architectures
// in how long it can delay execution. The behavior cannot be mapped to wall
// time as it suspends for 64*arg + 1-64 clocks but archs have different limits,
// clock speed can vary over the course of execution, etc. This is mostly only
// useful as a "yield for a few instructions to stop hammering a memory
// location" primitive.
static inline IREE_CL_ATTRIBUTE_ALWAYS_INLINE void iree_amdgpu_yield(void) {
  __builtin_amdgcn_s_sleep(1);
}

#endif  // IREE_CL_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_HSA_DEVICE_SUPPORT_OPENCL_H_
