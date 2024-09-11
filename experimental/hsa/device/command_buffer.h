// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_DEVICE_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_HSA_DEVICE_COMMAND_BUFFER_H_

#include "experimental/hsa/device/allocator.h"
#include "experimental/hsa/device/kernel.h"
#include "experimental/hsa/device/support/opencl.h"
#include "experimental/hsa/device/support/queue.h"
#include "experimental/hsa/device/support/signal.h"

typedef struct iree_amdgpu_queue_scheduler_t iree_amdgpu_queue_scheduler_t;

//===----------------------------------------------------------------------===//
// Device-side Command Buffer
//===----------------------------------------------------------------------===//
//
// Command buffers are represented by a host-side wrapper that implements the
// IREE HAL API and a device-side data structure holding the recorded contents.
// All information required to execute a command buffer lives on the device and
// a command buffer can be submitted from the device without host involvement.
// Command buffer data structures are immutable once constructed and can be
// executed concurrently and repeatedly based on the same recording because
// mutable execution state is stored separately.
//
// The recorded command buffer is partitioned into one or more command blocks.
// Each block represents a yieldable point in the execution where the command
// buffer scheduler is allowed to suspend processing. Segmenting allows for
// basic control flow to be implemented within a command buffer by skipping,
// branching, or looping over blocks and also enables execution when hardware
// queues may not have capacity for the entire command buffer. Conceptually
// command buffers are like coroutines/fibers in that any number may be
// simultaneously executing on the same hardware resources.
//
// +-------------------------------+    +-------------------------------------+
// | iree_hal_hsa_command_buffer_t +----> iree_amdgpu_command_buffer_t        |
// +-------------------------------+    +----------------+--------------------+
//                                                       |
//      +------------+------------+------------+---------+--+------------+
//      |            |            |            |            |            |
// +----v----+  +----v----+  +----v----+  +----v----+  +----v----+  +----v----+
// |  block  |..|  block  |..|  block  |..|  block  |..|  block  |..|  block  |
// +----+----+  +---------+  +---------+  +---------+  +---------+  +---------+
//      |
//      |    +------------------------------+
//      +----> command entries              | fixed length struct array
//      |    +------------------------------+
//      +----> embedded command data        | variable length packed buffer
//           +------------------------------+
//
// Each block contains one or more commands encoded in fixed length entries.
// Commands can be indexed by ordinal within the block such that command
// processing can be parallelized even though commands may require different
// amounts of additional data. An extra buffer is used to embed the additional
// data in read-only memory such as update buffers, dispatch constants, and
// dispatch binding references. Execution-invariant information is stored in the
// command and any execution-dependent information is stored as either
// deltas/relative values or bits that can be used to derive the information
// when it is issued.
//
// Execution starts with the first block and progresses through blocks based
// on control commands. Blocks are translated to AQL packets in parallel via
// the command scheduler kernel. Each command may translate to one or more AQL
// packets and space is reserved for the maximum potential AQL packets that are
// required when the block is launched. Execution uses a state structure that
// resides on device and is valid for the full duration of the command buffer
// execution. Every concurrently executing instance of a command buffer has its
// own state referencing its own kernel arguments buffer.
//
// Processing behavior:
//   1. Initialize iree_amdgpu_execution_state_t:
//     a. Allocate execution state from the queue ringbuffer
//     b. Assign target hardware AQL queue to receive packets
//     c. Reserve kernel arguments buffer with max size used by any block
//     d. Copy binding table into the state (if present)
//     e. Assign the first command buffer block as the entry block
//   2. Enqueue iree_amdgpu_command_buffer_issue_block:
//     a. Reserve queue space for all command AQL packets
//     b. Enqueue command processor kernel for the next block with barrier bit
//   3. Command processor, parallelized over each command in a block:
//     a. Assign/copy kernel arguments to scratch buffer (if needed)
//     b. Construct AQL packet(s) for the command
//     c. Change from type INVALID to the real type
//   4. Repeat 2 and 3 until all blocks completed
//   5. Enqueue top-level queue scheduler upon completion
//   6. Deinitialize execution state (release resources)
//
//=----------------------------------------------------------------------------=
//
// Command buffers are recorded with a forward progress guarantee ensuring that
// once issued they will complete even if no other work can be executed on the
// same queue. Events used within the command buffer have a signal-before-wait
// requirement when used on the same queue.
//
// Dispatches have their kernel arguments packed while their packets are
// constructed and enqueued. Some arguments are fixed (constants, directly
// referenced buffers) and copied directly from the command data buffer while
// others may be substituted with per-invocation state (indirectly referenced
// buffers from a binding table).
//
// Though most AQL packets are written once during their initial enqueuing some
// commands such as indirect dispatches require updating the packets after they
// have been placed in the target queue. Indirect dispatch parameters may either
// be declared static and captured at the start of command buffer processing
// or dynamic until immediately prior to when the particular dispatch is
// executed. Static parameters are preferred as the command scheduler can
// enqueue the dispatch packet by dereferencing the workgroups buffer while
// constructing the AQL packet. Dynamic parameters require dispatching a special
// fixup kernel immediately prior to the actual dispatch that does the
// indirection and updates the following packet in the queue. The AQL queue
// processing model is exploited by having the actual dispatch packet encoded as
// INVALID and thus halting the hardware command processor and the fixup
// dispatch is what switches it to a valid KERNEL_DISPATCH type.
//
//=----------------------------------------------------------------------------=
//
// AQL agents launch packets in order but may complete them in any order.
// The two mechanisms of controlling the launch timeline are the barrier bit and
// barrier packets. When set on a packet the barrier bit indicates that all
// prior work on the queue must complete before the packet can be launched and
// matches our HAL execution barrier. Barrier packets can be used to set up
// dependencies via HSA signals roughly matching our HAL events.
//
// When a command buffer is recorded we use the execution barrier commands to
// set the barrier bit on recorded packets and in many cases end up with no
// additional barrier packets:
//  +------------+
//  | barrier    |      (no aql packet needed)
//  +------------+
//  | dispatch A |  --> dispatch w/ barrier = true (await all prior)
//  +------------+
//  | barrier    |      (no aql packet needed)
//  +------------+
//  | dispatch B |  --> dispatch w/ barrier = true (await dispatch A)
//  +------------+
//
// In cases of concurrency a nop packet is needed to allow multiple dispatches
// to launch without blocking. The complication is that at the time we get the
// execution barrier command we don't know how many commands will follow before
// the next barrier. To support single-pass recording we do some tricks with
// moving packets in order to insert barrier packets as required:
//  +------------+
//  | barrier    |  --> nop w/ barrier = true (await all prior)
//  +------------+
//  | dispatch A |  --> dispatch w/ barrier = false
//  +------------+
//  | dispatch B |  --> dispatch w/ barrier = false
//  +------------+
//
// Fence acquire/release behavior is supported on the nop barrier packets
// allowing for commands on either side to potentially avoid setting the
// behavior themselves. For example in serialized cases without the barrier
// packets the dispatches would need to acquire/release:
//  +------------+
//  | dispatch A |  --> acquire (as needed), release AGENT
//  +------------+
//  | dispatch B |  --> acquire AGENT, release (as needed)
//  +------------+
// While the barrier packet can allow this to be avoided:
//  +------------+
//  | barrier    |  --> acquire (as needed), release AGENT
//  +------------+
//  | dispatch A |  --> acquire/release NONE
//  +------------+
//  | dispatch B |  --> acquire/release NONE
//  +------------+
//  | barrier    |  --> release (as needed)
//  +------------+
// The recording logic is more complex than desired but by figuring it out at
// record-time the command buffer logic running here on device is kept much
// more straightforward.
//
// TODO(benvanik): define how events map to packets.

//===----------------------------------------------------------------------===//
// iree_amdgpu_cmd_t
//===----------------------------------------------------------------------===//

// Defines the recorded command type.
// Note that commands may expand to zero or more AQL packets in the target
// execution queue as they may be routed to other queues or require multiple
// packets to complete.
typedef uint8_t iree_amdgpu_cmd_type_t;
enum iree_amdgpu_cmd_type_e {
  // iree_amdgpu_cmd_barrier_t
  IREE_AMDGPU_CMD_BARRIER = 0u,
  // iree_amdgpu_cmd_signal_event_t
  IREE_AMDGPU_CMD_SIGNAL_EVENT,
  // iree_amdgpu_cmd_reset_event_t
  IREE_AMDGPU_CMD_RESET_EVENT,
  // iree_amdgpu_cmd_wait_events_t
  IREE_AMDGPU_CMD_WAIT_EVENTS,
  // iree_amdgpu_cmd_fill_buffer_t
  IREE_AMDGPU_CMD_FILL_BUFFER,
  // iree_amdgpu_cmd_copy_buffer_t
  IREE_AMDGPU_CMD_COPY_BUFFER,
  // iree_amdgpu_cmd_dispatch_t
  IREE_AMDGPU_CMD_DISPATCH,
  // iree_amdgpu_cmd_dispatch_t
  IREE_AMDGPU_CMD_DISPATCH_INDIRECT_DYNAMIC,
  // iree_amdgpu_cmd_branch_t
  IREE_AMDGPU_CMD_BRANCH,
  // iree_amdgpu_cmd_return_t
  IREE_AMDGPU_CMD_RETURN,
};

// Flags controlling command processing behavior.
typedef uint8_t iree_amdgpu_cmd_flags_t;
enum iree_amdgpu_cmd_flag_bits_t {
  IREE_AMDGPU_CMD_FLAG_NONE = 0u,
  // Sets the barrier bit in the first AQL packet of the command in order to
  // force a wait on all prior packets to complete before processing the command
  // packets. This is much lighter weight than barriers and signals for the
  // common case of straight-line execution.
  IREE_AMDGPU_CMD_FLAG_QUEUE_AWAIT_BARRIER = 1u << 0,
  // DO NOT SUBMIT
  //
  // (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE)
  // invalidates I, K and L1
  //
  // (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE)
  // invalidates L1, L2 and flushes L2
  //
  // Includes a HSA_FENCE_SCOPE_AGENT HSA_FENCE_SCOPE_SYSTEM
  // HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
  IREE_AMDGPU_CMD_FLAG_FENCE_ACQUIRE_SYSTEM = 1u << 1,
  // HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
  IREE_AMDGPU_CMD_FLAG_FENCE_RELEASE_SYSTEM = 1u << 2,
};

// Commands are fixed-size to allow for indexing into an array of commands.
// Additional variable-length data is stored out-of-band of the command struct.
#define IREE_AMDGPU_CMD_SIZE 64

// Header at the start of every command used to control command processing.
typedef struct IREE_CL_ALIGNAS(64) iree_amdgpu_cmd_header_t {
  // Command type indicating the parent structure.
  iree_amdgpu_cmd_type_t type;
  // Flags controlling command processing behavior.
  iree_amdgpu_cmd_flags_t flags;
  // Offset into the queue where AQL packets for the command should be placed.
  // If more than one packet is required they are stored contiguously from the
  // base offset.
  uint16_t packet_offset;
} iree_amdgpu_cmd_header_t;

// Performs a full queue barrier causing subsequent commands to block until all
// prior commands have completed. This is effectively a no-op packet that just
// has the IREE_AMDGPU_CMD_FLAG_QUEUE_AWAIT_BARRIER bit set.
//
// Recorded by:
//  iree_hal_command_buffer_execution_barrier (sometimes)
typedef struct iree_amdgpu_cmd_barrier_t {
  iree_amdgpu_cmd_header_t header;
} iree_amdgpu_cmd_barrier_t;

// TODO(benvanik): rework events so that they can be reused. We really should
// have an events table-like thing or something that allows capture at time of
// issue (if we even want to allow events to be used across command buffers).
// Today events are similar to Vulkan ones which don't support concurrent issue
// and that limits us here.
//
// Storing an ordinal to the event table would let us bulk allocate them as part
// of the execution state. Recording would need to track the unique set of
// events used in order to determine the capacity. We could make it be declared
// similar to the binding table capacity and swap to recording with ordinals but
// that makes it more difficult for users to compose. Recording could also only
// support events created from the command buffer during recording
// (iree_hal_command_buffer_acquire_event, etc) and that could also be used to
// verify lifetime and invalid cross-command-buffer usage. The event handle
// could just be an integer all the way into the compiler.
//
// For now the event-based code below uses an opaque value that we can
// substitute with whatever we come up with.
typedef uint32_t iree_amdgpu_event_ordinal_t;

// Signals event after prior commands complete.
// The AQL signal will be decremented from a value of 1 to 0 to allow AQL
// dependencies to be satisfied directly.
//
// Recorded by:
//  iree_hal_command_buffer_signal_event
typedef struct iree_amdgpu_cmd_signal_event_t {
  iree_amdgpu_cmd_header_t header;
  iree_amdgpu_event_ordinal_t event;
} iree_amdgpu_cmd_signal_event_t;
#define IREE_AMDGPU_CMD_SIGNAL_EVENT_AQL_PACKET_COUNT 1

// Resets event to unsignaled after prior commands complete.
// The AQL signal will be set to a value of 1.
//
// Recorded by:
//  iree_hal_command_buffer_reset_event
typedef struct iree_amdgpu_cmd_reset_event_t {
  iree_amdgpu_cmd_header_t header;
  iree_amdgpu_event_ordinal_t event;
} iree_amdgpu_cmd_reset_event_t;
#define IREE_AMDGPU_CMD_RESET_EVENT_AQL_PACKET_COUNT 1

// Number of events that can be stored inline in a iree_amdgpu_cmd_wait_events_t
// command. This is the same as the AQL barrier-and packet and allows us to
// avoid additional storage/indirections in the common case of waits one or two
// events.
#define IREE_AMDGPU_CMD_WAIT_EVENT_INLINE_CAPACITY 5

// Waits for the given events to be signaled before proceeding.
// All events much reach a value of 0. May be decomposed into multiple barrier
// packets if the event count exceeds the capacity of the barrier-and packet.
//
// Recorded by:
//  iree_hal_command_buffer_wait_events
typedef struct iree_amdgpu_cmd_wait_events_t {
  iree_amdgpu_cmd_header_t header;
  // Number of events being waited upon.
  uint32_t event_count;
  union {
    // Inlined events if event_count is less than
    // IREE_AMDGPU_CMD_WAIT_EVENT_INLINE_CAPACITY.
    iree_amdgpu_event_ordinal_t
        events[IREE_AMDGPU_CMD_WAIT_EVENT_INLINE_CAPACITY];
    // Externally stored events if event_count is greater than
    // IREE_AMDGPU_CMD_WAIT_EVENT_INLINE_CAPACITY.
    iree_amdgpu_event_ordinal_t* events_ptr;
  };
} iree_amdgpu_cmd_wait_events_t;
#define IREE_AMDGPU_CMD_WAIT_EVENTS_PER_AQL_PACKET 5
#define IREE_AMDGPU_CMD_WAIT_EVENTS_AQL_PACKET_COUNT(event_count) \
  IREE_CL_CEIL_DIV((event_count), IREE_AMDGPU_CMD_WAIT_EVENTS_PER_AQL_PACKET)

// Fills a buffer with a repeating pattern.
// Performed via a blit kernel.
//
// Recorded by:
//  iree_hal_command_buffer_fill_buffer
typedef struct iree_amdgpu_cmd_fill_buffer_t {
  iree_amdgpu_cmd_header_t header;
  // Target buffer to fill.
  iree_amdgpu_buffer_ref_t target_ref;
  // 1 to 8 pattern bytes, little endian.
  uint64_t pattern;
  // Length in bytes of the pattern.
  uint8_t pattern_length;
  // Block-relative kernel arguments address.
  uint32_t kernarg_offset;
} iree_amdgpu_cmd_fill_buffer_t;
#define IREE_AMDGPU_CMD_FILL_BUFFER_AQL_PACKET_COUNT 1

// Copies between buffers.
// Performed via a blit kernel. May be implementable with SDMA but it is
// currently unverified.
//
// Recorded by:
//  iree_hal_command_buffer_update_buffer
//  iree_hal_command_buffer_copy_buffer
typedef struct iree_amdgpu_cmd_copy_buffer_t {
  iree_amdgpu_cmd_header_t header;
  // Copy source.
  iree_amdgpu_buffer_ref_t source_ref;
  // Copy target.
  iree_amdgpu_buffer_ref_t target_ref;
  // Block-relative kernel arguments address.
  uint32_t kernarg_offset;
} iree_amdgpu_cmd_copy_buffer_t;
#define IREE_AMDGPU_CMD_COPY_BUFFER_AQL_PACKET_COUNT 1

// AQL dispatch parameters as recorded.
// Some parameters may be overwritten as the packet is enqueued or during
// execution (such as for indirect dispatches).
typedef struct iree_amdgpu_cmd_dispatch_packet_t {
  // Kernel arguments used to dispatch the kernel.
  iree_amdgpu_kernel_args_t kernel_args;
  // Dispatch setup parameters. Used to configure kernel dispatch parameters
  // such as the number of dimensions in the grid. The parameters are
  // described by hsa_kernel_dispatch_packet_setup_t.
  uint16_t setup;
  // X dimension of grid, in work-items. Must be greater than 0.
  // Must not be smaller than workgroup_size_x.
  uint32_t grid_size_x;
  // Y dimension of grid, in work-items. Must be greater than 0.
  // If the grid has 1 dimension, the only valid value is 1.
  // Must not be smaller than workgroup_size_y.
  uint32_t grid_size_y;
  // Z dimension of grid, in work-items. Must be greater than 0.
  // If the grid has 1 or 2 dimensions, the only valid value is 1.
  // Must not be smaller than workgroup_size_z.
  uint32_t grid_size_z;
} iree_amdgpu_cmd_dispatch_packet_t;
static_assert(sizeof(iree_amdgpu_cmd_dispatch_packet_t) == 40,
              "dispatch packet template is inlined into cmd structs and must "
              "be kept small");

// Bitfield specifying flags controlling a dispatch operation.
typedef uint16_t iree_amdgpu_dispatch_flags_t;
enum iree_amdgpu_dispatch_flag_bits_t {
  IREE_AMDGPU_DISPATCH_FLAG_NONE = 0,
  // Dispatch uses an indirect workgroup count that is constant and available
  // prior to command buffer execution. The command processor will read the
  // workgroup count and embed it directly in the AQL kernel dispatch packet.
  IREE_AMDGPU_DISPATCH_FLAG_INDIRECT_STATIC = 1u << 0,
  // Dispatch uses an indirect workgroup count that is dynamic and may change up
  // to the exact moment the dispatch is executed. The command processor will
  // enqueue a kernel that performs the indirection and updates the kernel
  // dispatch packet with the value before allowing the hardware queue to
  // continue.
  IREE_AMDGPU_DISPATCH_FLAG_INDIRECT_DYNAMIC = 1u << 1,
};

#define IREE_AMDGPU_WORKGROUP_COUNT_UPDATE_KERNARG_SIZE (3 * sizeof(void*))

// Dispatches (directly or indirectly) a kernel.
// All information required to build the AQL packet is stored within the command
// such that it can be enqueued without additional indirection.
//
// Recorded by:
//  iree_hal_command_buffer_dispatch
//  iree_hal_command_buffer_dispatch_indirect
typedef struct iree_amdgpu_cmd_dispatch_t {
  iree_amdgpu_cmd_header_t header;
  // Dispatch control flags.
  iree_amdgpu_dispatch_flags_t flags;
  // Total number of 4-byte constants used by the dispatch.
  uint16_t constant_count;
  // Total number of bindings used by the dispatch.
  uint16_t binding_count;
  // Block-relative kernel arguments address.
  // This will be added to the per-execution base kernel arguments address
  // during packet production.
  // If the IREE_AMDGPU_DISPATCH_FLAG_INDIRECT_DYNAMIC bit is set then this will
  // include an additional IREE_AMDGPU_WORKGROUP_COUNT_UPDATE_KERNARG_SIZE
  // prefix that is used for dispatching the
  // `iree_amdgpu_command_buffer_workgroup_count_update` builtin kernel.
  uint32_t kernarg_offset;
  // AQL packet template. Copied (and possibly modified) as part of enqueuing.
  iree_amdgpu_cmd_dispatch_packet_t packet;
  // Optional buffer containing the workgroup count.
  // Processing is controlled by the IREE_AMDGPU_DISPATCH_FLAG_INDIRECT_* flags.
  iree_amdgpu_buffer_ref_t workgroups_ref;
  // Optional pointer to the dispatch constants passed to the kernel.
  IREE_CL_GLOBAL const void* constants;
  // Pointer to references describing how binding pointers are passed to the
  // kernel. References may include direct device pointers, allocation handles,
  // or slots in the binding table included as part of the execution request.
  IREE_CL_GLOBAL const iree_amdgpu_buffer_ref_t* bindings;
} iree_amdgpu_cmd_dispatch_t;
#define IREE_AMDGPU_CMD_DISPATCH_DIRECT_AQL_PACKET_COUNT 1
#define IREE_AMDGPU_CMD_DISPATCH_INDIRECT_STATIC_AQL_PACKET_COUNT 1
#define IREE_AMDGPU_CMD_DISPATCH_INDIRECT_DYNAMIC_AQL_PACKET_COUNT 2
#define IREE_AMDGPU_CMD_DISPATCH_AQL_PACKET_COUNT(dispatch_flags)            \
  (((dispatch_flags) & IREE_AMDGPU_DISPATCH_FLAG_INDIRECT_STATIC) != 0)      \
      ? IREE_AMDGPU_CMD_DISPATCH_INDIRECT_STATIC_AQL_PACKET_COUNT            \
      : ((((dispatch_flags) & IREE_AMDGPU_DISPATCH_FLAG_INDIRECT_DYNAMIC) != \
          0)                                                                 \
             ? IREE_AMDGPU_CMD_DISPATCH_INDIRECT_DYNAMIC_AQL_PACKET_COUNT    \
             : IREE_AMDGPU_CMD_DISPATCH_DIRECT_AQL_PACKET_COUNT)

// TODO(benvanik): better specify control flow; maybe conditional support.
// The current implementation is a placeholder for more sophisticated control
// flow both within a command buffer (branching) and across command buffers
// (calls). Calls will require nesting execution state and we may need to
// preallocate that (a primary command buffer keeping track of the max nesting
// depth).

// Unconditionally branches from the current block to a new block within the
// same command buffer.
typedef struct iree_amdgpu_cmd_branch_t {
  iree_amdgpu_cmd_header_t header;
  // Block ordinal within the parent command buffer where execution will
  // continue. The block pointer can be retrieved from the command buffer
  // blocks list.
  uint32_t target_block;
} iree_amdgpu_cmd_branch_t;
#define IREE_AMDGPU_CMD_BRANCH_AQL_PACKET_COUNT 1

// Returns from processing a command buffer by launching the scheduler.
//
// TODO(benvanik): differentiate return to scheduler from return to caller
// command buffer. Today this always assumes the scheduler is going to be the
// target.
typedef struct iree_amdgpu_cmd_return_t {
  iree_amdgpu_cmd_header_t header;
} iree_amdgpu_cmd_return_t;
#define IREE_AMDGPU_CMD_RETURN_AQL_PACKET_COUNT 1

IREE_CL_STATIC_ASSERT(sizeof(iree_amdgpu_cmd_signal_event_t) <=
                          IREE_AMDGPU_CMD_SIZE,
                      "commands must fit within the fixed command size");
IREE_CL_STATIC_ASSERT(sizeof(iree_amdgpu_cmd_reset_event_t) <=
                          IREE_AMDGPU_CMD_SIZE,
                      "commands must fit within the fixed command size");
IREE_CL_STATIC_ASSERT(sizeof(iree_amdgpu_cmd_wait_events_t) <=
                          IREE_AMDGPU_CMD_SIZE,
                      "commands must fit within the fixed command size");
IREE_CL_STATIC_ASSERT(sizeof(iree_amdgpu_cmd_fill_buffer_t) <=
                          IREE_AMDGPU_CMD_SIZE,
                      "commands must fit within the fixed command size");
IREE_CL_STATIC_ASSERT(sizeof(iree_amdgpu_cmd_copy_buffer_t) <=
                          IREE_AMDGPU_CMD_SIZE,
                      "commands must fit within the fixed command size");
IREE_CL_STATIC_ASSERT(sizeof(iree_amdgpu_cmd_dispatch_t) <=
                          IREE_AMDGPU_CMD_SIZE,
                      "commands must fit within the fixed command size");
IREE_CL_STATIC_ASSERT(sizeof(iree_amdgpu_cmd_branch_t) <= IREE_AMDGPU_CMD_SIZE,
                      "commands must fit within the fixed command size");
IREE_CL_STATIC_ASSERT(sizeof(iree_amdgpu_cmd_return_t) <= IREE_AMDGPU_CMD_SIZE,
                      "commands must fit within the fixed command size");

// A command describing an operation that may translate to zero or more AQL
// packets.
typedef struct IREE_CL_ALIGNAS(64) iree_amdgpu_cmd_t {
  union {
    iree_amdgpu_cmd_header_t header;
    iree_amdgpu_cmd_signal_event_t signal_event;
    iree_amdgpu_cmd_reset_event_t reset_event;
    iree_amdgpu_cmd_wait_events_t wait_events;
    iree_amdgpu_cmd_fill_buffer_t fill_buffer;
    iree_amdgpu_cmd_copy_buffer_t copy_buffer;
    iree_amdgpu_cmd_dispatch_t dispatch;
    iree_amdgpu_cmd_branch_t branch;
    iree_amdgpu_cmd_return_t return;
  };
} iree_amdgpu_cmd_t;

IREE_CL_STATIC_ASSERT(sizeof(iree_amdgpu_cmd_t) <= IREE_AMDGPU_CMD_SIZE,
                      "commands must fit within the fixed command size");

//===----------------------------------------------------------------------===//
// iree_amdgpu_command_buffer_t
//===----------------------------------------------------------------------===//

// A block of commands within a command buffer.
// Each block represents one or more commands that should be issued to target
// AQL queues as part of a single parallelized issue in a single contiguous
// span.
//
// Blocks are immutable once recorded and a block may be executed multiple
// times concurrently or serially with pipelining. Blocks are replicated per
// device such that any embedded device-local pointers are always valid for any
// queue the block is issued on. Any pointers that reference per-execution
// state (such as kernel argument buffers) are encoded as relative offsets to be
// added to whatever base pointer is reserved for the execution.
//
// Blocks are allocated as flat slabs with padding added to ensure alignment:
// +--------------+-----+------------+-----+------------------+
// | block header | pad | commands[] | pad | embedded_data... |
// +--------------+-----+------------+-----+------------------+
//
// Blocks are stored in a read-only memory region.
typedef struct IREE_CL_ALIGNAS(64) iree_amdgpu_command_block_t {
  // Maximum number of AQL packets that the block will enqueue during a single
  // execution. Fewer packets may be used but they will still be populated with
  // valid no-op AQL packets to ensure processing by the hardware.
  uint32_t max_packet_count;
  // Total number of commands in the block.
  uint32_t command_count;
  // Aligned storage for fixed-length command structures.
  IREE_CL_GLOBAL const iree_amdgpu_cmd_t* commands;
  // Aligned storage for embedded data used by commands (update buffers,
  // constants, etc).
  IREE_CL_GLOBAL const void* embedded_data;
} iree_amdgpu_command_block_t;

// A program consisting of one or more blocks of commands and control flow
// between them. Command buffers are immutable once recorded and retained in
// device local memory. A command buffer may be enqueued multiple times
// concurrently or in sequence as any state needed is stored separately in
// iree_amdgpu_execution_state_t.
//
// Execution of a command buffer starts at block[0] and continues based on
// control flow commands at the tail of each block. Blocks may direct execution
// within the same command buffer or transfer control to other command buffers
// by nesting. Upon completion a return command at the tail of a block will
// return back to the caller.
//
// Command buffers are stored in a read-only memory region.
typedef struct IREE_CL_ALIGNAS(64) iree_amdgpu_command_buffer_t {
  // Minimum required kernel argument buffer capacity to execute all blocks.
  // Only one block executes at a time and the storage will be reused.
  uint32_t max_kernarg_capacity;

  // Total number of blocks in the command buffer.
  uint32_t block_count;
  // A list of all blocks with block[0] being the entry point.
  // Commands reference blocks by ordinal in this list.
  IREE_CL_GLOBAL iree_amdgpu_command_block_t* blocks[];  // tail array
} iree_amdgpu_command_buffer_t;

//===----------------------------------------------------------------------===//
// iree_amdgpu_execution_state_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): include implicit arg count?
#define IREE_AMDGPU_EXECUTION_ISSUE_BLOCK_KERNARG_SIZE (2 * sizeof(void*))
#define IREE_AMDGPU_EXECUTION_CONTROL_KERNARG_SIZE \
  IREE_CL_MAX(8 * sizeof(void*), IREE_AMDGPU_EXECUTION_ISSUE_BLOCK_KERNARG_SIZE)

// Transient state used during the execution of a command buffer.
// Command buffers are executed like coroutines by having the command processor
// issue a sequence of commands before tail-enqueuing further processing or a
// return back to the top-level scheduler.
//
// Execution state is stored in mutable global memory so that the scheduler can
// manipulate it.
typedef struct IREE_CL_ALIGNAS(64) iree_amdgpu_execution_state_t {
  // Command buffer being executed.
  IREE_CL_GLOBAL const iree_amdgpu_command_buffer_t* command_buffer;

  // TODO(benvanik): signal pool reference where events were allocated from.
  // Upon completion the events will be freed.
  // TODO(benvanik): signal list mapping signal ordinals as assigned in commands
  // to dynamically acquired signals.

  // DO NOT SUBMIT
  // completion signal (1->0 when complete)
  // chained on to final packet
  // barrier on scheduler with dep on completion signal

  // Scheduler that is managing the execution state lifetime.
  // When the command buffer completes it will be scheduled to handle cleanup
  // and resuming queue processing.
  IREE_CL_GLOBAL iree_amdgpu_queue_scheduler_t* scheduler;

  // Handles to opaque kernel objects used to dispatch builtin kernels.
  IREE_CL_GLOBAL const iree_amdgpu_kernels_t* kernels;

  // Storage with space for control kernel arguments. Reused by issue_block and
  // return operations as only one is allowed to be pending at a time. Must be
  // at least IREE_AMDGPU_EXECUTION_CONTROL_KERNARG_SIZE bytes.
  IREE_CL_GLOBAL uint8_t* control_kernarg_storage;

  // Reserved storage for kernel arguments of at least the size specified by the
  // command buffer required_kernarg_capacity. Only one block can be executed
  // at a time and storage is reused. Note that storage is uninitialized and
  // must be fully specified by the command processor.
  IREE_CL_GLOBAL uint8_t* execution_kernarg_storage;

  // Queue used for command buffer execution.
  // This may differ from the top-level scheduling queue.
  IREE_CL_GLOBAL iree_amdgpu_queue_t* execution_queue;

  // TODO(benvanik): stack for remembering resume blocks when returning from
  // nested command buffers. For now we don't have calls so it's not needed.
  // uint32_t block_stack[...];

  // Binding table used to resolve indirect binding references.
  // Contains enough elements to satisfy all slots referenced by
  // iree_amdgpu_buffer_ref_t in the command buffer.
  // Note that bindings here will not reference slots (though maybe we could
  // support that in the future for silly aliasing tricks).
  IREE_CL_ALIGNAS(64) iree_amdgpu_buffer_ref_t bindings[];  // tail array
} iree_amdgpu_execution_state_t;

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

#if defined(IREE_CL_TARGET_DEVICE)

// Launches a command buffer with the given initialized execution state.
// The command buffer will begin execution at the entry block and continue
// (possibly rescheduling itself) until a return command is reached.
//
// The parent scheduler should not progress until the completion signal
// indicates that the command buffer has fully completed execution.
// Forward progress is only guaranteed so long as the hardware queue is not
// blocked (such as by waiting on the completion signal). Upon completion the
// command buffer will enqueue the scheduler so that it can clean up the
// execution state and resume processing the queue.
void iree_amdgpu_command_buffer_enqueue(
    IREE_CL_GLOBAL iree_amdgpu_execution_state_t* IREE_CL_RESTRICT state);

#endif  // IREE_CL_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_HSA_DEVICE_COMMAND_BUFFER_H_
