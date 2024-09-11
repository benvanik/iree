// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
// Types
//===----------------------------------------------------------------------===//

#if defined(IREE_CL_TARGET_DEVICE)

typedef char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long int64_t;
typedef unsigned long uint64_t;

#else

#include <stdint.h>

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
// Sleep
//===----------------------------------------------------------------------===//

#if defined(IREE_CL_TARGET_DEVICE)

extern void __builtin_amdgcn_s_sleep(int);

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

// const __constant int __oclc_ABI_version = 500;

const __constant bool __oclc_unsafe_math_opt = false;
const __constant bool __oclc_daz_opt = false;
const __constant bool __oclc_correctly_rounded_sqrt32 = true;
const __constant bool __oclc_finite_only_opt = false;
const __constant bool __oclc_wavefrontsize64 =
    __AMDGCN_WAVEFRONT_SIZE__ == 64 ? 1 : 0;

#if defined(__gfx700__)
const __constant unsigned __oclc_ISA_version = 7000;
#elif defined(__gfx701__)
const __constant unsigned __oclc_ISA_version = 7001;
#elif defined(__gfx702__)
const __constant unsigned __oclc_ISA_version = 7002;
#elif defined(__gfx703__)
const __constant unsigned __oclc_ISA_version = 7003;
#elif defined(__gfx704__)
const __constant unsigned __oclc_ISA_version = 7004;
#elif defined(__gfx705__)
const __constant unsigned __oclc_ISA_version = 7005;
#elif defined(__gfx801__)
const __constant unsigned __oclc_ISA_version = 8001;
#elif defined(__gfx802__)
const __constant unsigned __oclc_ISA_version = 8002;
#elif defined(__gfx803__)
const __constant unsigned __oclc_ISA_version = 8003;
#elif defined(__gfx805__)
const __constant unsigned __oclc_ISA_version = 8005;
#elif defined(__gfx810__)
const __constant unsigned __oclc_ISA_version = 8100;
#elif defined(__gfx900__)
const __constant unsigned __oclc_ISA_version = 9000;
#elif defined(__gfx902__)
const __constant unsigned __oclc_ISA_version = 9002;
#elif defined(__gfx904__)
const __constant unsigned __oclc_ISA_version = 9004;
#elif defined(__gfx906__)
const __constant unsigned __oclc_ISA_version = 9006;
#elif defined(__gfx908__)
const __constant unsigned __oclc_ISA_version = 9008;
#elif defined(__gfx909__)
const __constant unsigned __oclc_ISA_version = 9009;
#elif defined(__gfx90a__)
const __constant unsigned __oclc_ISA_version = 9010;
#elif defined(__gfx90c__)
const __constant unsigned __oclc_ISA_version = 9012;
#elif defined(__gfx940__)
const __constant unsigned __oclc_ISA_version = 9400;
#elif defined(__gfx941__)
const __constant unsigned __oclc_ISA_version = 9401;
#elif defined(__gfx942__)
const __constant unsigned __oclc_ISA_version = 9402;
#elif defined(__gfx1010__)
const __constant unsigned __oclc_ISA_version = 10100;
#elif defined(__gfx1011__)
const __constant unsigned __oclc_ISA_version = 10101;
#elif defined(__gfx1012__)
const __constant unsigned __oclc_ISA_version = 10102;
#elif defined(__gfx1013__)
const __constant unsigned __oclc_ISA_version = 10103;
#elif defined(__gfx1030__)
const __constant unsigned __oclc_ISA_version = 10300;
#elif defined(__gfx1031__)
const __constant unsigned __oclc_ISA_version = 10301;
#elif defined(__gfx1032__)
const __constant unsigned __oclc_ISA_version = 10302;
#elif defined(__gfx1033__)
const __constant unsigned __oclc_ISA_version = 10303;
#elif defined(__gfx1034__)
const __constant unsigned __oclc_ISA_version = 10304;
#elif defined(__gfx1035__)
const __constant unsigned __oclc_ISA_version = 10305;
#elif defined(__gfx1036__)
const __constant unsigned __oclc_ISA_version = 10306;
#elif defined(__gfx1100__)
const __constant unsigned __oclc_ISA_version = 11000;
#elif defined(__gfx1101__)
const __constant unsigned __oclc_ISA_version = 11001;
#elif defined(__gfx1102__)
const __constant unsigned __oclc_ISA_version = 11002;
#elif defined(__gfx1103__)
const __constant unsigned __oclc_ISA_version = 11003;
#elif defined(__gfx1150__)
const __constant unsigned __oclc_ISA_version = 11500;
#elif defined(__gfx1151__)
const __constant unsigned __oclc_ISA_version = 11501;
#elif defined(__gfx1200__)
const __constant unsigned __oclc_ISA_version = 12000;
#elif defined(__gfx1201__)
const __constant unsigned __oclc_ISA_version = 12001;
#else
#error "Unknown AMDGPU architecture"
#endif

__kernel void add_one(uint32_t n,
                      IREE_CL_GLOBAL uint32_t* IREE_CL_RESTRICT buffer) {
  const size_t idx = iree_amdgpu_global_id_x();
  if (idx < n) {
    buffer[idx] += 1;
  }
}

__kernel void add_one_with_timestamp(
    uint32_t n, IREE_CL_GLOBAL uint32_t* IREE_CL_RESTRICT buffer) {
  const size_t idx = iree_amdgpu_global_id_x();
  if (idx < n) {
    buffer[idx] += 1;
  }
  if (idx == 0) {
    // correlates with signal start_ts/end_ts
    // can be converted with hsa_amd_profiling_convert_tick_to_system_domain
    uint64_t t = __builtin_readsteadycounter();
    buffer[0] = (uint32_t)(t >> 32);
    buffer[1] = (uint32_t)t;
  }
}

__kernel void mul_x(uint32_t x, uint32_t n,
                    IREE_CL_GLOBAL uint32_t* IREE_CL_RESTRICT buffer) {
  const size_t idx = iree_amdgpu_global_id_x();
  if (idx < n) {
    buffer[idx] *= x;
  }
}

typedef struct iree_hsa_signal_s {
  // Opaque handle. The value 0 is reserved.
  uint64_t handle;
} iree_hsa_signal_t;

#define iree_hsa_signal_null() \
  (iree_hsa_signal_t) { 0 }
#define iree_hsa_signal_is_null(signal) ((signal).handle == 0)

//===----------------------------------------------------------------------===//
// Device Library Externs
//===----------------------------------------------------------------------===//

#define iree_hsa_signal_load __ockl_hsa_signal_load
#define iree_hsa_signal_add __ockl_hsa_signal_add
#define iree_hsa_signal_and __ockl_hsa_signal_and
#define iree_hsa_signal_or __ockl_hsa_signal_or
#define iree_hsa_signal_xor __ockl_hsa_signal_xor
#define iree_hsa_signal_exchange __ockl_hsa_signal_exchange
#define iree_hsa_signal_subtract __ockl_hsa_signal_subtract
#define iree_hsa_signal_cas __ockl_hsa_signal_cas
#define iree_hsa_signal_store __ockl_hsa_signal_store

extern int64_t __ockl_hsa_signal_load(const iree_hsa_signal_t signal,
                                      iree_amdgpu_memory_order_t memory_order);
extern void __ockl_hsa_signal_add(iree_hsa_signal_t signal, int64_t value,
                                  iree_amdgpu_memory_order_t memory_order);
extern void __ockl_hsa_signal_and(iree_hsa_signal_t signal, int64_t value,
                                  iree_amdgpu_memory_order_t memory_order);
extern void __ockl_hsa_signal_or(iree_hsa_signal_t signal, int64_t value,
                                 iree_amdgpu_memory_order_t memory_order);
extern void __ockl_hsa_signal_xor(iree_hsa_signal_t signal, int64_t value,
                                  iree_amdgpu_memory_order_t memory_order);
extern int64_t __ockl_hsa_signal_exchange(
    iree_hsa_signal_t signal, int64_t value,
    iree_amdgpu_memory_order_t memory_order);
extern void __ockl_hsa_signal_subtract(iree_hsa_signal_t signal, int64_t value,
                                       iree_amdgpu_memory_order_t memory_order);
extern int64_t __ockl_hsa_signal_cas(iree_hsa_signal_t signal, int64_t expected,
                                     int64_t value,
                                     iree_amdgpu_memory_order_t memory_order);
extern void __ockl_hsa_signal_store(iree_hsa_signal_t signal, int64_t value,
                                    iree_amdgpu_memory_order_t memory_order);

//===----------------------------------------------------------------------===//
// HSA/AMDGPU AQL Queue
//===----------------------------------------------------------------------===//

typedef enum {
  // Queue supports multiple producers.
  IREE_HSA_QUEUE_TYPE_MULTI = 0,
  // Queue only supports a single producer.
  IREE_HSA_QUEUE_TYPE_SINGLE = 1
} iree_hsa_queue_type_t;

typedef struct iree_hsa_queue_s {
  // Queue type.
  iree_hsa_queue_type_t type;

  // Queue features mask. This is a bit-field of hsa_queue_feature_t
  // values. Applications should ignore any unknown set bits.
  uint32_t features;

  IREE_CL_GLOBAL void* base_address;

  // Signal object used by the application to indicate the ID of a packet that
  // is ready to be processed. The HSA runtime manages the doorbell signal. If
  // the application tries to replace or destroy this signal, the behavior is
  // undefined.
  //
  // If type is HSA_QUEUE_TYPE_SINGLE the doorbell signal value must be
  // updated in a monotonically increasing fashion. If type is
  // HSA_QUEUE_TYPE_MULTI the doorbell signal value can be updated with any
  // value.
  iree_hsa_signal_t doorbell_signal;

  // Maximum number of packets the queue can hold. Must be a power of 2.
  uint32_t size;
  // Reserved. Must be 0.
  uint32_t reserved1;
  // Queue identifier, which is unique over the lifetime of the application.
  uint64_t id;
} iree_hsa_queue_t;

#define iree_hsa_queue_load_read_index __ockl_hsa_queue_load_read_index
#define iree_hsa_queue_load_write_index __ockl_hsa_queue_load_write_index
#define iree_hsa_queue_add_write_index __ockl_hsa_queue_add_write_index
#define iree_hsa_queue_cas_write_index __ockl_hsa_queue_cas_write_index
#define iree_hsa_queue_store_write_index __ockl_hsa_queue_store_write_index

extern uint64_t __ockl_hsa_queue_load_read_index(
    const IREE_CL_GLOBAL iree_hsa_queue_t* queue,
    iree_amdgpu_memory_order_t mem_order);
extern uint64_t __ockl_hsa_queue_load_write_index(
    const IREE_CL_GLOBAL iree_hsa_queue_t* queue,
    iree_amdgpu_memory_order_t mem_order);
extern uint64_t __ockl_hsa_queue_add_write_index(
    IREE_CL_GLOBAL iree_hsa_queue_t* queue, uint64_t value,
    iree_amdgpu_memory_order_t mem_order);
extern uint64_t __ockl_hsa_queue_cas_write_index(
    IREE_CL_GLOBAL iree_hsa_queue_t* queue, uint64_t expected, uint64_t value,
    iree_amdgpu_memory_order_t mem_order);
extern void __ockl_hsa_queue_store_write_index(
    IREE_CL_GLOBAL iree_hsa_queue_t* queue, uint64_t value,
    iree_amdgpu_memory_order_t mem_order);

typedef enum {
  HSA_PACKET_TYPE_VENDOR_SPECIFIC = 0,
  HSA_PACKET_TYPE_INVALID = 1,
  HSA_PACKET_TYPE_KERNEL_DISPATCH = 2,
  HSA_PACKET_TYPE_BARRIER_AND = 3,
  HSA_PACKET_TYPE_AGENT_DISPATCH = 4,
  HSA_PACKET_TYPE_BARRIER_OR = 5
} hsa_packet_type_t;

typedef enum {
  HSA_PACKET_HEADER_TYPE = 0,
  HSA_PACKET_HEADER_BARRIER = 8,
  HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = 9,
  HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = 11,
} hsa_packet_header_t;

typedef enum {
  HSA_FENCE_SCOPE_NONE = 0,
  HSA_FENCE_SCOPE_AGENT = 1,
  HSA_FENCE_SCOPE_SYSTEM = 2
} hsa_fence_scope_t;

typedef struct hsa_agent_dispatch_packet_s {
  uint16_t header;
  uint16_t type;
  uint32_t reserved0;
  void* return_address;
  uint64_t arg[4];
  uint64_t reserved2;
  iree_hsa_signal_t completion_signal;
} iree_hsa_agent_dispatch_packet_t;

void iree_amdgpu_host_enqueue(
    IREE_CL_GLOBAL iree_hsa_queue_t* IREE_CL_RESTRICT queue, uint16_t type,
    uint64_t return_address, uint64_t arg0, uint64_t arg1, uint64_t arg2,
    uint64_t arg3, iree_hsa_signal_t completion_signal) {
  // Reserve a packet write index and wait for it to become available in cases
  // where the queue is exhausted.
  uint64_t packet_id = iree_hsa_queue_add_write_index(
      queue, 1u, iree_amdgpu_memory_order_relaxed);
  while (packet_id - iree_hsa_queue_load_read_index(
                         queue, iree_amdgpu_memory_order_acquire) >=
         queue->size) {
    iree_amdgpu_yield();  // spinning
  }
  const uint64_t queue_mask = queue->size - 1;  // power of two
  IREE_CL_GLOBAL iree_hsa_agent_dispatch_packet_t* agent_packet =
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
  iree_amdgpu_atomic_store_explicit(
      (IREE_CL_GLOBAL iree_amdgpu_atomic_uint64_t*)agent_packet, header_type,
      iree_amdgpu_memory_order_release,
      iree_amdgpu_memory_scope_all_svm_devices);

  // Signal the queue doorbell.
  iree_hsa_signal_store(queue->doorbell_signal, packet_id,
                        iree_amdgpu_memory_order_relaxed);
}

__kernel void issue_host_call(
    IREE_CL_GLOBAL iree_hsa_queue_t* IREE_CL_RESTRICT queue,
    iree_hsa_signal_t completion_signal, uint32_t arg) {
  iree_amdgpu_host_enqueue(queue, 123, 0x100, arg, 0x201, 0x202, 0x203,
                           iree_hsa_signal_null());
  iree_amdgpu_host_enqueue(queue, 456, 0x100, arg, 0x201, 0x202, 0x203,
                           iree_hsa_signal_null());
  iree_amdgpu_host_enqueue(queue, 789, 0x100, arg, 0x201, 0x202, 0x203,
                           completion_signal);
}

static inline size_t iree_host_align(size_t value, size_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

// AMD Signal Kind Enumeration Values.
enum iree_amd_signal_kind_t {
  IREE_AMD_SIGNAL_KIND_INVALID = 0,
  IREE_AMD_SIGNAL_KIND_USER = 1,
  IREE_AMD_SIGNAL_KIND_DOORBELL = -1,
  IREE_AMD_SIGNAL_KIND_LEGACY_DOORBELL = -2
};
typedef int64_t iree_amd_signal_kind64_t;

typedef struct IREE_CL_ALIGNAS(64) iree_amd_signal_s {
  iree_amd_signal_kind64_t kind;
  union {
    volatile int64_t value;
    IREE_CL_GLOBAL volatile uint32_t* legacy_hardware_doorbell_ptr;
    IREE_CL_GLOBAL volatile uint64_t* hardware_doorbell_ptr;
  };
  uint64_t event_mailbox_ptr;
  uint32_t event_id;
  uint32_t reserved1;
  uint64_t start_ts;
  uint64_t end_ts;
  union {
    IREE_CL_GLOBAL /*iree_amd_queue_t*/ void* queue_ptr;
    uint64_t reserved2;
  };
  uint32_t reserved3[2];
} iree_amd_signal_t;

typedef enum {
  HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS = 0
} hsa_kernel_dispatch_packet_setup_t;
typedef struct hsa_kernel_dispatch_packet_s {
  uint16_t header;
  uint16_t setup;
  uint16_t workgroup_size_x;
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;
  uint16_t reserved0;
  uint32_t grid_size_x;
  uint32_t grid_size_y;
  uint32_t grid_size_z;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;
  IREE_CL_GLOBAL void* kernarg_address;
  uint64_t reserved2;
  iree_hsa_signal_t completion_signal;
} hsa_kernel_dispatch_packet_t;

typedef struct implicit_kernargs_t {
  uint32_t block_count[3];    // + 0/4/8
  uint16_t group_size[3];     // + 12/14/16
  uint16_t remainder[3];      // + 18/20/22
  uint64_t reserved0;         // + 24 hidden_tool_correlation_id
  uint64_t reserved1;         // + 32
  uint64_t global_offset[3];  // + 40/48/56
  uint16_t grid_dims;         // + 64
} implicit_kernargs_t;
__kernel void issue_dispatch(
    IREE_CL_GLOBAL iree_hsa_queue_t* IREE_CL_RESTRICT queue,
    uint64_t mul_x_object, uint32_t mul_x_private_size,
    uint32_t mul_x_group_size, iree_hsa_signal_t completion_signal,
    IREE_CL_GLOBAL void* buffer,
    IREE_CL_GLOBAL void* IREE_CL_RESTRICT kernarg_storage,
    uint32_t element_count, uint32_t mul_by) {
  typedef struct mul_x_args_t {
    uint32_t x;
    uint32_t n;
    IREE_CL_GLOBAL void* buffer;
  } mul_x_args_t;

  IREE_CL_GLOBAL mul_x_args_t* mul_x_kernargs =
      (IREE_CL_GLOBAL mul_x_args_t*)kernarg_storage;
  mul_x_kernargs->x = mul_by;
  mul_x_kernargs->n = element_count;
  mul_x_kernargs->buffer = buffer;

  uint32_t grid_size[3] = {element_count, 1, 1};
  uint16_t workgroup_size[3] = {32, 1, 1};

  IREE_CL_GLOBAL implicit_kernargs_t* implicit_kernargs =
      (IREE_CL_GLOBAL implicit_kernargs_t*)((uint8_t*)kernarg_storage +
                                            iree_host_align(
                                                sizeof(mul_x_args_t), 8));
  implicit_kernargs->block_count[0] = grid_size[0] / workgroup_size[0];
  implicit_kernargs->block_count[1] = grid_size[1] / workgroup_size[1];
  implicit_kernargs->block_count[2] = grid_size[2] / workgroup_size[2];
  implicit_kernargs->group_size[0] = workgroup_size[0];
  implicit_kernargs->group_size[1] = workgroup_size[1];
  implicit_kernargs->group_size[2] = workgroup_size[2];
  implicit_kernargs->remainder[0] =
      (uint16_t)(grid_size[0] % workgroup_size[0]);
  implicit_kernargs->remainder[1] =
      (uint16_t)(grid_size[1] % workgroup_size[1]);
  implicit_kernargs->remainder[2] =
      (uint16_t)(grid_size[2] % workgroup_size[2]);
  implicit_kernargs->reserved0 = 0;
  implicit_kernargs->reserved1 = 0;
  implicit_kernargs->global_offset[0] = 0;  // newOffset[0];
  implicit_kernargs->global_offset[1] = 0;  // newOffset[1];
  implicit_kernargs->global_offset[2] = 0;  // newOffset[2];
  implicit_kernargs->grid_dims = 3;

  // DO NOT SUBMIT should do nontemporal kernarg update?

  hsa_kernel_dispatch_packet_t packet;
  packet.header = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
  packet.setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  packet.workgroup_size_x = workgroup_size[0];
  packet.workgroup_size_y = workgroup_size[1];
  packet.workgroup_size_z = workgroup_size[2];
  packet.reserved0 = 0;
  packet.grid_size_x = grid_size[0];
  packet.grid_size_y = grid_size[1];
  packet.grid_size_z = grid_size[2];
  packet.private_segment_size = mul_x_private_size;
  packet.group_segment_size = mul_x_group_size;
  packet.kernel_object = mul_x_object;
  packet.kernarg_address = kernarg_storage;
  packet.reserved2 = 0;
  packet.completion_signal = completion_signal;

  uint16_t packet_header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  uint32_t packet_header_setup = packet_header | (packet.setup << 16);

  uint64_t packet_id = iree_hsa_queue_add_write_index(
      queue, 1, iree_amdgpu_memory_order_release);
  while ((packet_id - iree_hsa_queue_load_read_index(
                          queue, iree_amdgpu_memory_order_acquire)) >=
         queue->size) {
    iree_amdgpu_yield();
  }

  IREE_CL_GLOBAL hsa_kernel_dispatch_packet_t* packet_ptr =
      (IREE_CL_GLOBAL hsa_kernel_dispatch_packet_t*)((IREE_CL_GLOBAL uint8_t*)
                                                         queue->base_address +
                                                     (packet_id &
                                                      (queue->size - 1)) *
                                                         64);
  // memcpy(packet_ptr, &packet, sizeof(packet));
  *packet_ptr = packet;
  iree_amdgpu_atomic_store_explicit((volatile atomic_uint*)packet_ptr,
                                    packet_header_setup,
                                    iree_amdgpu_memory_order_release,
                                    iree_amdgpu_memory_scope_all_svm_devices);

  // value ignored in MULTI cases
  iree_hsa_signal_store(queue->doorbell_signal, packet_id,
                        iree_amdgpu_memory_order_relaxed);
}
