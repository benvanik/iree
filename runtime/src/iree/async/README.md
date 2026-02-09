# iree/async/ — Proactor-Based Async I/O System

## What This Is

A unified async I/O layer that lets the kernel and hardware do the work. Network
sends, file reads, GPU synchronization, and timer scheduling all flow through
one submission/completion interface — and on modern kernels, most of it happens
without returning to user space at all.

**The problem**: ML inference across multiple machines requires moving large
tensors between GPUs, across networks, and sometimes through storage — with
latency measured in microseconds. Traditional approaches (thread-per-connection,
epoll + read/write, staging through host memory) add copies, context switches,
and CPU round-trips at every boundary.

**The approach**: Push work into the kernel and hardware, then get out of the way.

- **Kernel-sequenced pipelines**: A "wait for GPU → receive from network →
  signal GPU" pipeline can execute as linked io_uring SQEs — three operations,
  one syscall, zero user-space round-trips between steps.
- **Hardware offloading**: Registered dmabuf memory lets NICs read directly from
  GPU VRAM (GPUDirect RDMA) and NVMe controllers write directly to device memory
  (GPU Direct Storage). The CPU never touches the data.
- **NUMA-aware scaling**: Each proactor thread is pinned to a CPU complex with
  buffer pools allocated on the local NUMA node. A 4-socket server with 8 GPUs
  runs 4 proactor threads, each handling 2 GPUs and their attached NICs — no
  cross-socket memory traffic on the data path.
- **Device fence bridging**: GPU completion signals (sync_file fds) feed directly
  into the proactor's event loop, which triggers network sends without waking the
  host CPU. The reverse path (network receive → GPU fence) is equally toll-free.
- **Unified event model**: Files, sockets, timers, semaphores, and device fences
  are all just operations submitted to the same proactor. One callback model, one
  thread, one mental model — from a single laptop GPU to a rack of 8×H100s.

The same code runs everywhere: io_uring on Linux for production throughput,
kqueue on macOS for development, a threaded fallback for testing and embedded
targets. Capabilities are discovered at runtime and the caller falls back
gracefully.

---

## Overview

`iree/async/` is the foundational async I/O layer for IREE 2.0. It provides a
completion-based (proactor pattern) abstraction over platform-specific async
mechanisms: io_uring on Linux, kqueue on macOS, IOCP on Windows, and a portable
threaded fallback for testing and embedded contexts.

Everything above this layer — HAL drivers, networking, task executors, the VM
runtime — builds on these primitives. The async system itself depends only on
`iree/base/`.

**Design principles:**

- **Completion-based**: Operations are submitted and callbacks fire when complete.
  No readiness polling. Natural alignment with io_uring's submission/completion
  queue model.
- **Caller-driven**: The proactor makes progress only when `poll()` is called.
  No hidden threads, no surprises about which thread callbacks run on. A utility
  wrapper (`util/proactor_thread.h`) provides optional dedicated-thread operation.
- **Zero-copy capable**: Registered memory regions, scatter-gather I/O, fixed
  file descriptors. The abstraction preserves every optimization the kernel offers.
- **Vtable-dispatched**: Proactors and semaphores are polymorphic. Custom
  implementations for testing, embedding, or bridging other systems.
- **Rock-solid error handling**: Every operation carries status. Errors propagate
  with rich annotations. No silent failures.
- **Pervasive tracing**: Tracy zones on all significant paths. Fiber tracking for
  async context. Latency measurement built in.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Applications                                     │
├─────────────┬─────────────┬─────────────┬─────────────┬──────────────────┤
│  iree/hal/  │  iree/net/  │ iree/task/  │  iree/vm/   │      ...         │
│  (drivers)  │  (network)  │ (executors) │  (runtime)  │                  │
├─────────────┴─────────────┴─────────────┴─────────────┴──────────────────┤
│                          iree/async/                                     │
│                                                                          │
│  Proactor          Semaphore       Frontier        Operations            │
│  (vtable)          (vtable)        Tracker         (subtypes)            │
│                                                                          │
│  Socket  File  Event  Region/Span   Op Pool  Buffer Pool  Thread Wrapper │
│  (ref)   (ref) (ref)  (ref/value)   (util)   (util)       (utility)      │
├──────────────────────────────────────────────────────────────────────────┤
│                          iree/base/                                      │
│  (allocators, status, atomics, threading, tracing)                       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Dependency Rules

- `iree/async/` depends ONLY on `iree/base/`
- `iree/net/` depends on `iree/async/` (never on `iree/hal/`)
- `iree/hal/` depends on `iree/async/` (implements `iree_async_semaphore_vtable_t`)
- No circular dependencies — async is the foundation

### How Layers Connect

HAL drivers bridge to the async world by implementing `iree_async_semaphore_t`.
When GPU work completes, the HAL signals an async semaphore. When network I/O
needs to gate on GPU work, it waits on the same semaphore via the proactor.

```
GPU work completes
    → HAL driver signals iree_async_semaphore_t
        → Proactor sees semaphore reached target value
            → Net layer's wait_semaphore operation completes
                → Net layer submits send with GPU output buffer
```

---

## Getting Started

A minimal example showing the submit/poll/callback lifecycle:

```c
#include "iree/async/api.h"
#include "iree/async/util/proactor_thread.h"

// Callback fires from poll() context when the timer expires.
static void on_timer(void* user_data, iree_async_operation_t* op,
                     iree_status_t status,
                     iree_async_completion_flags_t flags) {
  bool* fired = (bool*)user_data;
  *fired = iree_status_is_ok(status);
}

int main() {
  iree_allocator_t host_allocator = iree_allocator_system();

  // 1. Create a platform-optimal proactor.
  iree_async_proactor_t* proactor = NULL;
  IREE_CHECK_OK(iree_async_proactor_create_platform(
      /*options=*/NULL, host_allocator, &proactor));

  // 2. Set up a timer operation (fires in 100ms).
  bool timer_fired = false;
  iree_async_timer_operation_t timer = {0};
  timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer.base.completion_fn = on_timer;
  timer.base.user_data = &timer_fired;
  timer.deadline_ns = iree_time_now() + iree_make_duration_ms(100);

  // 3. Submit and poll until callback fires.
  IREE_CHECK_OK(iree_async_proactor_submit_one(proactor, &timer.base));
  while (!timer_fired) {
    iree_host_size_t count = 0;
    IREE_CHECK_OK(iree_async_proactor_poll(
        proactor, iree_make_timeout_ms(200), &count));
  }

  // 4. Cleanup.
  iree_async_proactor_release(proactor);
  return 0;
}
```

**Key pattern**: Fill operation struct → submit → poll → callback fires → done.
All callbacks run from `poll()`, on the thread that calls `poll()`. No surprise
scheduling.

---

## Core Concepts

Each concept has a dedicated header file. This section explains how the pieces
fit together; see the referenced headers for authoritative type definitions and
function signatures.

### Proactor (`proactor.h`)

The central abstraction. Manages async operation submission and completion
dispatch. Vtable-dispatched for backend polymorphism.

The proactor's event loop is the three-method cycle:

- **`submit(operations)`**: Hand operations to the proactor for async execution.
  Maps to a single `io_uring_submit()` or `kevent()` call internally.
- **`poll(timeout)`**: Block until completions arrive (or timeout). Invoke
  callbacks for all completed operations. Returns the count of callbacks fired.
- **`wake()`**: Thread-safe. Interrupts a blocked `poll()` from another thread.
  Idempotent — safe to call multiple times.

Additional methods: `cancel()` for in-flight operations, `query_capabilities()`
for feature detection, resource creation (sockets, files, events), and buffer
registration (see Zero-Copy I/O below).

### Semaphores (`semaphore.h`)

Cross-layer synchronization primitive with timeline semantics (monotonically
increasing uint64 values). This is the bridge between GPU work and I/O:

- HAL drivers implement the semaphore vtable (signal when GPU work completes)
- The async layer waits on semaphores before submitting dependent I/O
- Software semaphores exist for pure-CPU coordination

Key vtable methods: `query()`, `signal(value, frontier)`, `fail(status)`,
`acquire_timepoint()`, `cancel_timepoint()`, `query_frontier()`,
`export_primitive()`.

The `signal` method takes an optional `iree_async_frontier_t*` for causal
context propagation. Pass NULL for local-only signals. Pass non-NULL when the
signal's ordering must propagate to remote machines (see Frontiers below).

### Operations (`operation.h`, `operations/*.h`)

All operations inherit from `iree_async_operation_t`. Caller-owned storage
(intrusive — no proactor allocation on submit). The proactor invokes the
callback when the operation completes.

Operation subtypes live in `operations/`:
- `operations/scheduling.h` — nop, timer, event_wait, sequence
- `operations/semaphore.h` — semaphore wait, semaphore signal
- `operations/net.h` — accept, connect, recv, recv_pool, send, close
- `operations/file.h` — open, read, write, close

Each subtype extends the base with type-specific parameters (inputs) and
results (outputs filled by the proactor on completion).

### Resources (`socket.h`, `file.h`, `event.h`)

Proactor-managed handles wrapping platform primitives. All are ref-counted.

- **Sockets**: TCP, UDP, Unix stream/dgram. Created with immutable options
  (REUSE_ADDR, NO_DELAY, etc.), then configured with bind/listen (synchronous,
  one-shot). Async operations: accept, connect, recv, send, close.
- **Files**: Positioned I/O (pread/pwrite semantics — no shared file position).
  Async operations: open, read, write, close.
- **Events**: Lightweight signaling primitive for cross-thread wakeup. `set()`
  from any thread, wait via `event_wait` operations submitted to the proactor.
  Events use platform-native mechanisms (eventfd on Linux, pipes on macOS/BSD,
  Win32 events on Windows) for efficient kernel-level signaling.

### Notifications (`notification.h`)

Level-triggered signaling for waking worker threads from I/O completions.

Unlike events (edge-triggered, one signal per wait), notifications use epoch
counting: multiple signals coalesce, and waiters observe any signal that
occurred after their wait was submitted. This makes them ideal for worker
thread pools where the poll thread needs to fan out to multiple consumers.

**Key operations**:
- `iree_async_notification_signal(notification, wake_count)` — Thread-safe,
  async-signal-safe. Call from any context including completion callbacks.
- `iree_async_notification_wait(notification, timeout)` — Blocking wait for
  worker threads outside the proactor's poll loop.
- `NOTIFICATION_WAIT` / `NOTIFICATION_SIGNAL` operations — Async variants
  that integrate with the proactor's event loop and support LINK chains.

**Why this matters**: The proactor's poll thread is single-threaded by design
(predictable callback ordering, no hidden threads). When I/O completions need
to wake N worker threads, the poll thread signals a notification, and workers
blocked on `iree_async_notification_wait()` wake up. This keeps the proactor
simple while enabling efficient fan-out:

```
I/O completes → poll() callback → notification_signal(N)
                                         ↓
Worker 1: notification_wait() ──────► wakes, processes
Worker 2: notification_wait() ──────► wakes, processes
Worker N: notification_wait() ──────► wakes, processes
```

**Platform mapping**: Futex on Linux 6.7+ (optimal), eventfd on older Linux,
platform-specific elsewhere. The abstraction handles fallbacks automatically.

### Memory (`region.h`, `span.h`, `types.h`)

Registered memory for zero-copy I/O.

- **Region**: Ref-counted registered memory block with backend-specific handles
  (RDMA MR keys, io_uring buffer IDs, dmabuf descriptors). Created by the
  proactor during `register_buffer()` / `register_dmabuf()`.
- **Span**: Value-type subrange of a region `{region, offset, length}`. Used in
  all I/O operations. Non-owning (like `iree_string_view_t`), but the proactor
  retains the span's region during in-flight operations.
- **Buffer registration state** (`types.h`): Header-only types embeddable in
  HAL buffers. Tracks which proactors a buffer is registered with.

### Frontiers (`frontier.h`, `frontier_tracker.h`)

Vector clocks for causal ordering across machines.

A frontier is a set of `(axis, epoch)` pairs. Each axis identifies a causal
source (a GPU queue, a collective, a host thread). Each epoch is a monotonic
timeline value. A frontier says "I depend on all of these axes having reached
at least these epochs."

The frontier tracker maps axes to semaphores and dispatches waiters when
frontiers are satisfied. See `frontier.h` for the axis encoding scheme
(session | machine | domain | ordinal) and the comparison/merge operations.

**When to use frontiers**: Frontiers are for remote coordination. Local
operations (same machine) use semaphores directly. When a signal needs to
carry ordering guarantees across the network (e.g., "GPU on machine A finished
epoch 42 on queue 3"), the frontier captures that context and the wire format
encodes it compactly for the receiver.

### Affinity (`affinity.h`)

NUMA-aware locality domain. Groups CPU cores, memory controllers, and PCIe
devices. Used at pool and proactor creation time to ensure NUMA-local allocation.

### Buffer Pool (`buffer_pool.h`)

Pre-registered slab of fixed-size buffers with O(1) acquire/release. Used for
pool-based multishot receives where the kernel (io_uring) or NIC (RDMA) selects
the receive buffer.

---

## Memory Ownership

### Operation Ownership

The ownership rule: **caller owns the operation before submit and after the final
callback. The proactor owns it in between.**

```
Single-shot:
  1. Caller allocates/acquires operation
  2. Caller fills parameters
  3. Caller submits to proactor          → proactor owns
  4. Poll invokes callback               → caller owns again (can reuse/release)

Multishot (ACCEPT, RECV with MULTISHOT flag):
  1-3. Same as single-shot
  4. Poll invokes callback with IREE_ASYNC_COMPLETION_FLAG_MORE  → proactor still owns
  5. ... more callbacks with MORE flag ...
  6. Final callback without MORE flag    → caller owns again

Cancellation:
  1. Caller calls cancel(operation)
  2. Proactor eventually invokes callback with IREE_STATUS_CANCELLED, no MORE flag
  3. Caller owns operation again
```

### Multishot Termination

Multishot operations persist until one of these conditions:

1. **Resource close**: Closing the underlying socket/file generates a final
   callback (error status, no MORE flag).
2. **Error**: Network disconnect, peer close, etc. generates a final callback.
3. **Explicit cancellation**: `cancel()` generates a final callback with
   `IREE_STATUS_CANCELLED`.

**Critical**: `release()` does NOT terminate multishot operations. The operation
holds a reference to the resource, keeping it alive. This is intentional —
release is for ownership transfer, not cleanup.

**Correct cleanup pattern**:
```c
// 1. Close the socket (async operation).
iree_async_socket_close_operation_t close_op = {0};
close_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE;
close_op.socket = listener;
close_op.base.completion_fn = on_close;
iree_async_proactor_submit_one(proactor, &close_op.base);

// 2. Wait for:
//    - The close completion (close_op callback fires)
//    - The multishot final callback (no MORE flag)
// Both will fire; order depends on kernel behavior.

// 3. Release the socket AFTER both callbacks complete.
// The multishot's final callback indicates the operation is done.
// The close callback indicates the socket is closed.
iree_async_socket_release(listener);
```

**Incorrect pattern** (causes use-after-free):
```c
// BAD: Releasing without closing doesn't terminate multishot.
iree_async_socket_release(listener);  // Multishot still holds a reference!
// Test ends, stack-allocated operation struct is destroyed.
// Multishot callback fires with dangling pointer → crash.
```

This pattern applies to all backends: io_uring, kqueue, IOCP, and the threaded
fallback. Each backend must ensure that closing a resource terminates all
in-flight multishot operations on that resource.

### Span Region Lifetime

Spans are non-owning, but operations that embed spans may outlive the caller's
scope. The proactor guarantees region safety:

- **At submit time**: the proactor retains each span's region (one atomic
  increment per span).
- **After the final callback**: the proactor releases each span's region. For
  multishot operations, the release happens only on the final invocation
  (without `IREE_ASYNC_COMPLETION_FLAG_MORE`).
- **NULL regions**: no retain/release. The caller manages raw memory lifetime.

This means callers can safely unregister buffers before all operations complete —
the proactor's retained reference keeps the region alive.

### Buffer Registration Ownership

Buffer registration follows a clear allocation protocol (see `proactor.h` and
`types.h` for details):

- The **proactor allocates** the registration entry during `register_buffer()` /
  `register_dmabuf()`.
- The entry is linked into the caller's `iree_async_buffer_registration_state_t`.
- Entry cleanup happens via `cleanup_fn` (set by the proactor) — either
  explicitly via `unregister_buffer()` or automatically when the buffer is
  destroyed (via `iree_async_buffer_registration_state_cleanup()`).
- The **proactor must outlive** all registrations. Destroy registered buffers
  before releasing the proactor.

### Resource Lifetime Rules

- **Proactor** outlives all resources created from it (sockets, files, events,
  pools, registrations).
- **Sockets/files/events** must not be destroyed while operations referencing
  them are in flight. Cancel or wait for completion first.
- **Buffer pools** must have all leases returned before `_free()`.
- **Semaphores** are retained by their timepoints. Safe to release the caller's
  reference while timepoints are pending — the semaphore stays alive.

---

## Thread Safety Model

| Type | Thread Safety |
|------|---------------|
| Proactor `submit()` | Thread-safe (batched internally) |
| Proactor `poll()` | Single-threaded (one poller per proactor) |
| Proactor `wake()` | Thread-safe, idempotent |
| Semaphore `signal()` | Thread-safe |
| Semaphore `query()` | Thread-safe (atomic load) |
| Semaphore `acquire_timepoint()` | Thread-safe |
| Event `set()` | Thread-safe |
| Buffer pool acquire/release | **NOT** thread-safe (proactor-thread only) |
| Frontier tracker `advance()` | Thread-safe (multiple axes concurrently) |
| Frontier tracker `wait()` | Thread-safe |
| Registration state | **NOT** thread-safe (serialize setup, then read-only) |

**The golden rule**: Callbacks fire from `poll()`, on the thread that calls
`poll()`. If you use `iree_async_proactor_thread_t`, that's the proactor's
dedicated thread. You control this — there are no hidden threads.

---

## Connection Lifecycle

### Server

```c
// 1. Create a socket with options (immutable, applied to kernel handle).
iree_async_socket_t* listen_socket = NULL;
IREE_RETURN_IF_ERROR(iree_async_socket_create(
    proactor, IREE_ASYNC_SOCKET_TYPE_TCP,
    IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR |
    IREE_ASYNC_SOCKET_OPTION_REUSE_PORT |
    IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
    &listen_socket));

// 2. Bind and listen (synchronous, one-shot configuration).
iree_async_address_t address;
iree_async_address_from_ipv4("0.0.0.0", 8080, &address);
IREE_RETURN_IF_ERROR(iree_async_socket_bind(listen_socket, &address));
IREE_RETURN_IF_ERROR(iree_async_socket_listen(listen_socket, /*backlog=*/128));

// 3. Submit multishot accept (one SQE → many connections).
iree_async_socket_accept_operation_t accept_op = {0};
accept_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT;
accept_op.base.flags = IREE_ASYNC_OPERATION_FLAG_MULTISHOT;
accept_op.base.completion_fn = on_accept;
accept_op.base.user_data = server_context;
accept_op.listen_socket = listen_socket;
IREE_RETURN_IF_ERROR(
    iree_async_proactor_submit_one(proactor, &accept_op.base));

// 4. In the accept callback:
static void on_accept(void* user_data, iree_async_operation_t* op,
                      iree_status_t status,
                      iree_async_completion_flags_t flags) {
  iree_async_socket_accept_operation_t* accept =
      (iree_async_socket_accept_operation_t*)op;
  if (!iree_status_is_ok(status)) {
    // Accept failed (listen socket closed, or proactor shutting down).
    return;
  }
  // accept->accepted_socket is a new ref. Start recv on it.
  start_connection(accept->accepted_socket, &accept->peer_address);
  // If MORE flag is set, proactor keeps delivering more connections.
}

// 5. Poll loop (or use proactor_thread for automatic polling).
while (running) {
  iree_host_size_t count = 0;
  iree_async_proactor_poll(proactor, iree_make_timeout_ms(100), &count);
}
```

### Client

```c
// 1. Create socket.
iree_async_socket_t* socket = NULL;
IREE_RETURN_IF_ERROR(iree_async_socket_create(
    proactor, IREE_ASYNC_SOCKET_TYPE_TCP,
    IREE_ASYNC_SOCKET_OPTION_NO_DELAY, &socket));

// 2. Submit async connect.
iree_async_socket_connect_operation_t connect_op = {0};
connect_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT;
connect_op.base.completion_fn = on_connected;
connect_op.socket = socket;
iree_async_address_from_ipv4("192.168.1.100", 8080, &connect_op.address);
IREE_RETURN_IF_ERROR(
    iree_async_proactor_submit_one(proactor, &connect_op.base));

// 3. In the connect callback, start send/recv.
static void on_connected(void* user_data, iree_async_operation_t* op,
                          iree_status_t status,
                          iree_async_completion_flags_t flags) {
  if (!iree_status_is_ok(status)) {
    // Connection refused, timeout, unreachable, etc.
    return;
  }
  iree_async_socket_connect_operation_t* connect =
      (iree_async_socket_connect_operation_t*)op;
  start_protocol(connect->socket);
}
```

---

## Error Recovery

### Per-Operation Errors

Every callback receives an `iree_status_t`. The proactor never swallows errors.
The status has a full annotation chain (e.g., "recv failed: connection reset by
peer").

### Socket Sticky Failure

Once a socket encounters an error, it enters a permanently failed state.
Subsequent operations on it complete immediately with the recorded failure.
This matches HAL semaphore behavior — first error wins, no recovery.

**What to do when a socket fails**: Close it (async close operation), release
your reference, and reconnect if appropriate. There is no "reset" or "retry"
at the socket level.

### Semaphore Failure Propagation

When a semaphore is failed (e.g., GPU device lost), all pending
`wait_semaphore` operations on that semaphore complete with the failure status.
This propagates hardware failures through the dependency graph — dependent
network operations fail rather than hanging.

### Sequence Failure

When any step in a sequence fails:
- Remaining steps are skipped (not submitted).
- The sequence's base callback fires with the failing step's error.
- The caller owns the sequence again (including all step operations).

### Proactor-Level Errors

If `poll()` itself fails (ring corruption, kernel error), it returns a status.
This is fatal — the proactor is likely unusable. The thread wrapper stores the
error and invokes the error callback. Recovery requires creating a new proactor.

### Backpressure

When a buffer pool is exhausted (`_acquire` returns
`IREE_STATUS_RESOURCE_EXHAUSTED`), this is non-blocking backpressure. The
caller should:
- Defer the operation (queue it for retry after a release).
- Apply flow control upstream (stop accepting new connections, pause recv).
- **Never** spin-wait on acquire.

For pool-based multishot receives, the proactor handles this automatically:
when the pool is empty, multishot pauses and resumes when buffers are returned.

---

## Testing Guide

### The Threaded Backend

The `proactor_threaded` backend implements the full proactor API using blocking
I/O on worker threads. It requires no kernel async support and provides a
portable, deterministic test environment.

```c
// Use threaded backend for tests instead of platform-specific.
iree_async_proactor_t* proactor = NULL;
IREE_CHECK_OK(iree_async_proactor_create_posix(
    /*options=*/NULL, host_allocator, &proactor));
```

### Determinism

The proactor model is inherently deterministic for testing:
- `poll()` is synchronous — you call it, callbacks fire, it returns.
- No background threads (unless you create a `proactor_thread`).
- You control exactly when completions are processed.

```c
// Test: submit operation, poll until callback fires.
bool completed = false;
submit_some_operation(proactor, &completed);
while (!completed) {
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(
      proactor, iree_make_timeout_ms(1000), &count));
}
// Assert results after callback has fired.
```

### Capability Gating

Tests that require specific backend features should check capabilities and skip
gracefully rather than using external excluded-test lists:

```c
iree_async_proactor_capabilities_t caps =
    iree_async_proactor_query_capabilities(proactor);
if (!(caps & IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS)) {
  GTEST_SKIP() << "Backend does not support linked operations";
}
```

### Fresh Proactor Per Test

Each test should create its own proactor to prevent state leakage between tests.
The threaded backend is cheap to create/destroy.

---

## Platform Backends

### io_uring (Linux 5.6+)

The primary backend. Maps naturally to the proactor model:

| Proactor Method | io_uring Mapping |
|-----------------|------------------|
| `submit()` | Fill SQEs, `io_uring_submit()` |
| `poll()` | `io_uring_wait_cqe_timeout()`, drain CQEs |
| `wake()` | Submit NOP with sentinel, or registered eventfd |
| `cancel()` | `IORING_OP_ASYNC_CANCEL` |
| `create_socket()` | `socket()` + apply options + register as fixed file |
| `register_buffer()` | `IORING_REGISTER_BUFFERS` or buffer ring |

**io_uring-specific optimizations:**
- Fixed files (`IORING_REGISTER_FILES`): avoid fd lookup overhead
- Registered buffers (`IORING_REGISTER_BUFFERS`): avoid page pinning per-op
- Buffer rings (`IORING_REGISTER_PBUF_RING`): kernel picks buffer for multishot recv
- Linked SQEs (`IOSQE_IO_LINK`): kernel-chained sequences
- Multishot operations: one SQE, many CQEs (accept, recv)
- Zero-copy send (`IORING_OP_SEND_ZC`): two CQEs (sent + buffer released)
- Event waits and device fence import via `IORING_OP_POLL_ADD` (eventfd, sync_file)
- dmabuf registration for GPUDirect RDMA and devmem TCP paths

### kqueue (macOS, BSD)

Reactor emulated as proactor. The backend waits for readiness via `kevent()`,
then performs non-blocking I/O to completion, and posts synthetic completion
events.

| Proactor Method | kqueue Mapping |
|-----------------|----------------|
| `submit()` | Register interest via `kevent()` changelist |
| `poll()` | `kevent()` → readiness → non-blocking I/O → invoke callbacks |
| `wake()` | `EVFILT_USER` with `NOTE_TRIGGER` |
| `cancel()` | Remove filter + mark operation cancelled |

**Limitations vs io_uring:**
- No native zero-copy send/recv (kernel copy always happens)
- No linked operations (sequences always callback-based)
- No multishot (emulated by re-arming after each event)
- No fixed file optimization

### Windows IOCP (Future)

True proactor (like io_uring). Maps cleanly:

| Proactor Method | IOCP Mapping |
|-----------------|--------------|
| `submit()` | `WSASend`/`WSARecv`/`ReadFile` with OVERLAPPED |
| `poll()` | `GetQueuedCompletionStatusEx()` |
| `wake()` | `PostQueuedCompletionStatus()` |
| `cancel()` | `CancelIoEx()` |

### Threaded Emulation (Testing/Embedded)

No kernel async support needed. Uses blocking I/O on worker threads. Each
submitted operation dispatches to a thread pool, posts completion to a queue.

Useful for:
- Unit testing proactor-dependent code without io_uring
- Embedded platforms without async kernel APIs
- Debugging (deterministic, single-threaded mode)

---

## Zero-Copy I/O and Device Fence Bridging

The proactor provides toll-free data paths between GPU memory, NIC hardware,
and NVMe storage — eliminating host-side copies and CPU synchronization
round-trips where the kernel and hardware support it.

### Buffer Registration

Buffer registration pins memory and pre-computes backend handles so that I/O
operations can reference the memory by handle rather than re-mapping on every
operation:

```
                    register_buffer() / register_dmabuf()
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│                  iree_async_region_t                     │
│                                                          │
│  base_ptr ──────► pinned host/device memory              │
│  access_flags ──► READ | WRITE | REMOTE_READ | ...       │
│  handles.rdma ──► { lkey, rkey, mr }   (RDMA verbs)      │
│  handles.iouring► { buffer_group_id }  (io_uring fixed)  │
│  handles.dmabuf─► { fd, offset }       (device memory)   │
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
              iree_async_span_t { region, offset, length }
                     used in recv/send/read/write ops
```

**Host memory** (`register_buffer`): Covers RDMA MR registration,
io_uring fixed buffers, and TCP zero-copy send. The proactor pins pages and
caches handles. Subsequent I/O operations on spans referencing this region
avoid per-operation page pinning.

**Device memory** (`register_dmabuf`): Covers dmabuf-backed GPU memory.
Enables toll-free paths between GPU VRAM and I/O hardware:

| Path | Mechanism | Use Case |
|------|-----------|----------|
| GPU → NIC | GPUDirect RDMA / dmabuf peer mapping | Send GPU output to peer |
| NIC → GPU | devmem TCP (kernel 6.10+) | Receive directly into VRAM |
| GPU → NVMe | GPU Direct Storage | Checkpoint to disk from VRAM |
| GPU → GPU | P2P dmabuf export/import | Multi-GPU data sharing |

### Access Flags

Registration includes access flags that control how the backend configures
hardware permissions:

- `IREE_ASYNC_BUFFER_ACCESS_FLAG_READ` — local read (recv, file_read)
- `IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE` — local write (send, file_write)
- `IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_READ` — remote read (RDMA READ target)
- `IREE_ASYNC_BUFFER_ACCESS_FLAG_REMOTE_WRITE` — remote write (RDMA WRITE target)

For RDMA, `REMOTE_READ` and `REMOTE_WRITE` control whether the registration
generates an rkey suitable for peer access. This is a security boundary: only
buffers explicitly marked for remote access can be targeted by peer RDMA
operations.

### Device Fence Bridging

The proactor bridges between kernel device fences (sync_file fds) and the
async semaphore system, enabling GPU↔I/O synchronization without host-side
polling:

```
GPU COMPLETION → PROACTOR (import_fence)
═══════════════════════════════════════════

  GPU command buffer finishes
    → Driver exports sync_file fd
      → iree_async_semaphore_import_fence(proactor, fd, sem, value)
        → Proactor polls the fd (io_uring POLL_ADD / kqueue EVFILT_READ)
          → fd signals (GPU work done)
            → Proactor signals semaphore to |value|
              → Downstream wait_semaphore operations complete
                → Network sends / file writes fire automatically


PROACTOR → GPU SUBMISSION (export_fence)
═══════════════════════════════════════════

  Network receive completes
    → Proactor signals semaphore
      → iree_async_semaphore_export_fence(proactor, sem, value, &fd)
        → Proactor creates sync_file fd internally
          → When semaphore reaches |value|, fd signals
            → GPU command buffer waits on this fd before executing
              → GPU processes received data with no host round-trip
```

**Why this matters for remoting:** In a multi-machine pipeline, GPU work on
machine A produces data, the network layer sends it to machine B, and machine
B's GPU consumes it. Without fence bridging, the host CPU must poll for each
transition. With fence bridging, the entire pipeline runs without host
involvement after initial setup.

### Capability Detection

Not all backends support all zero-copy features. Query before using:

```c
iree_async_proactor_capabilities_t caps =
    iree_async_proactor_query_capabilities(proactor);
if (iree_any_bit_set(caps, IREE_ASYNC_PROACTOR_CAPABILITY_REGISTERED_BUFFERS)) {
  // Can use register_buffer for fixed buffer I/O.
}
if (iree_any_bit_set(caps, IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND)) {
  // Can create sockets with IREE_ASYNC_SOCKET_OPTION_ZERO_COPY.
}
if (iree_any_bit_set(caps, IREE_ASYNC_PROACTOR_CAPABILITY_DMABUF)) {
  // Can use register_dmabuf for device memory paths.
}
if (iree_any_bit_set(caps, IREE_ASYNC_PROACTOR_CAPABILITY_DEVICE_FENCE)) {
  // Can use import_fence / export_fence.
}
if (iree_any_bit_set(caps, IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS)) {
  // Sequences without step_fn may use kernel-chained execution.
}
```

Fall back gracefully when capabilities are absent. For example, if DMABUF is
not supported, stage through a host buffer. If DEVICE_FENCE is not supported,
poll the semaphore from the proactor thread.

---

## Usage Scenarios

### GPU→Network Pipeline with Semaphores

The canonical remoting use case: GPU produces data, network sends it to a peer,
peer's GPU consumes it. The buffer is dmabuf-registered for zero-copy NIC access.

```c
// Steps: wait for GPU → send to peer → signal buffer reusable
iree_async_semaphore_wait_operation_t wait_gpu = {0};
wait_gpu.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
wait_gpu.semaphores = &gpu_done_semaphore;
wait_gpu.values = &gpu_done_value;
wait_gpu.count = 1;
wait_gpu.mode = IREE_ASYNC_WAIT_MODE_ALL;

// peer_socket was created with IREE_ASYNC_SOCKET_OPTION_ZERO_COPY.
iree_async_socket_send_operation_t send_data = {0};
send_data.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND;
send_data.socket = peer_socket;
send_data.buffers = (iree_async_span_list_t){.values = &gpu_output_span,
                                              .count = 1};
send_data.send_flags = IREE_ASYNC_SOCKET_SEND_FLAG_NONE;

iree_async_semaphore_signal_operation_t signal_reuse = {0};
signal_reuse.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
signal_reuse.semaphores = &buffer_reuse_semaphore;
signal_reuse.values = &next_reuse_value;
signal_reuse.count = 1;
signal_reuse.frontier = NULL;  // Local buffer reuse — no causal propagation.

// Chain as sequence (io_uring may use linked SQEs).
iree_async_operation_t* steps[] = {
    &wait_gpu.base, &send_data.base, &signal_reuse.base};
iree_async_sequence_operation_t pipeline = {0};
pipeline.base.type = IREE_ASYNC_OPERATION_TYPE_SEQUENCE;
pipeline.base.completion_fn = on_send_pipeline_done;
pipeline.steps = steps;
pipeline.step_count = 3;
pipeline.step_fn = NULL;  // Pre-committed → linked SQEs possible.
IREE_CHECK_OK(iree_async_proactor_submit_one(proactor, &pipeline.base));
```

### Pool-Based Multishot Receive

High-throughput receive path where the kernel selects buffers from a
pre-registered pool:

```c
// Create receive pool (16 buffers × 64KB, NUMA-local).
iree_async_buffer_pool_options_t pool_opts = {
    .buffer_size = 64 * 1024,
    .buffer_count = 16,
    .access_flags = IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
    .affinity = &affinities[0],
};
iree_async_buffer_pool_t* rx_pool = NULL;
IREE_CHECK_OK(iree_async_buffer_pool_allocate(
    proactor, &pool_opts, allocator, &rx_pool));

// Submit multishot pool-based recv.
iree_async_socket_recv_pool_operation_t recv_op = {0};
recv_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL;
recv_op.base.flags = IREE_ASYNC_OPERATION_FLAG_MULTISHOT;
recv_op.base.completion_fn = on_pool_recv;
recv_op.socket = connection;
recv_op.pool = rx_pool;
IREE_CHECK_OK(iree_async_proactor_submit_one(proactor, &recv_op.base));

// Each completion carries a lease. Process and return.
static void on_pool_recv(void* user_data, iree_async_operation_t* op,
                          iree_status_t status,
                          iree_async_completion_flags_t flags) {
  iree_async_socket_recv_pool_operation_t* recv =
      (iree_async_socket_recv_pool_operation_t*)op;
  if (iree_status_is_ok(status)) {
    iree_byte_span_t data = iree_async_span_data(recv->lease.span);
    process_message(data.data, recv->bytes_received);
    iree_async_buffer_pool_release(&recv->lease);
  }
}
```

### HAL Driver Implementing iree_async_semaphore_t

```c
typedef struct my_hal_semaphore_t {
  iree_async_semaphore_t base;  // ref_count + vtable at offset 0
  hsa_signal_t hsa_signal;
  iree_slim_mutex_t timepoint_mutex;
  iree_async_semaphore_timepoint_t* timepoints;
} my_hal_semaphore_t;

static iree_status_t my_hal_semaphore_signal(
    iree_async_semaphore_t* base, uint64_t value,
    const iree_async_frontier_t* frontier) {
  my_hal_semaphore_t* sem = (my_hal_semaphore_t*)base;
  // Monotonic update of native GPU signal.
  uint64_t current = hsa_signal_load_scacquire(sem->hsa_signal);
  while (value > current) {
    if (hsa_signal_cas_scacqrel(sem->hsa_signal, current, value)) break;
    current = hsa_signal_load_scacquire(sem->hsa_signal);
  }
  notify_timepoints(sem, value);
  // Frontier (if non-NULL) is for the remoting layer to propagate causal
  // context. HAL-only backends that don't participate in remoting ignore it.
  (void)frontier;
  return iree_ok_status();
}

static const iree_async_semaphore_vtable_t my_hal_semaphore_vtable = {
    .destroy = my_hal_semaphore_destroy,
    .query = my_hal_semaphore_query,
    .signal = my_hal_semaphore_signal,
    .query_frontier = my_hal_semaphore_query_frontier,
    .fail = my_hal_semaphore_fail,
    .acquire_timepoint = my_hal_semaphore_acquire_timepoint,
    .cancel_timepoint = my_hal_semaphore_cancel_timepoint,
    .export_primitive = my_hal_semaphore_export_primitive,
};
```

---

## Implementing a Backend

Implementing a new proactor backend requires filling the vtable in `proactor.h`.
Here is the contract each method must satisfy:

### submit()

- **Caller-owned operations**: The `operations` list pointer may be stack-local.
  Copy the pointers if needed beyond this call.
- **Span retention**: Implementations MUST call `iree_async_span_retain_region()`
  for each span embedded in the submitted operations. Release on final callback.
- **Tracing**: Fill `submit_time_ns` on each operation if `IREE_TRACE` is enabled.
- **Batch semantics**: All operations in a single submit should map to one kernel
  call where possible (one `io_uring_submit()`, one `kevent()` changelist).
- **Error reporting**: If submission fails for operation N, operations 0..N-1 may
  already be in flight. The caller is responsible for cancelling them.

### poll()

- **Callback invocation**: All callbacks fire from within this call, on the
  calling thread. No deferred or cross-thread callback dispatch.
- **Timeout semantics**: Block for at most `timeout`. Return early if completions
  are available. Zero timeout means non-blocking drain.
- **Multishot**: For multishot operations, invoke the callback with
  `IREE_ASYNC_COMPLETION_FLAG_MORE` on all but the final invocation.
- **Region release**: Release span regions on the final callback invocation
  (no MORE flag).

### cancel()

- **Asynchronous**: The callback WILL fire eventually (with CANCELLED status,
  no MORE flag). cancel() itself does not guarantee the callback has fired.
- **Idempotent**: Cancelling an already-completed operation is a no-op.

### Resource creation

- **Socket options**: `create_socket` receives immutable options, applies them
  via `setsockopt()`, and discards them. The options are not stored.
- **Fixed file registration**: Backends that support it (io_uring) should
  register new sockets/files in the fixed file table and set `fixed_file_index`.

### Buffer registration

- **Entry allocation**: The proactor allocates the entry (including any
  backend-specific trailing data) using `host_allocator`.
- **cleanup_fn**: Set this to a function that releases the region reference and
  frees the entry memory. It will be called during unregistration or buffer
  destroy.
- **Backend resources**: Create RDMA MRs, io_uring buffer table entries, etc.
  during registration. Tear them down in `unregister_buffer` or `cleanup_fn`.

---

## Build Structure

**Consumer dependency pattern:**

```python
# Most consumers: full API + platform-optimal backend.
deps = [
    "//runtime/src/iree/async",
    "//runtime/src/iree/async:platform",
]

# HAL drivers: embed registration state (header-only, no link dependency).
deps = [
    "//runtime/src/iree/async:types",
]

# Tests: use threaded backend for determinism.
deps = [
    "//runtime/src/iree/async",
    "//runtime/src/iree/async:threaded",
]

# Utilities (optional, separate targets):
#   //runtime/src/iree/async/util:operation_pool
#   //runtime/src/iree/async/util:proactor_thread
#   //runtime/src/iree/async/util:sequence_emulation
```

**Key targets:**
- `:async` — Core API (proactor, semaphore, operations, resources, frontiers)
- `:types` — Header-only embeddable types (no link dependency)
- `:platform` — Platform-optimal backend (selects io_uring/kqueue/threaded)
- `:io_uring`, `:kqueue`, `:threaded` — Specific backends
- `util:operation_pool` — Size-class freelist for operation structs
- `util:proactor_thread` — Dedicated poll thread wrapper
- `util:sequence_emulation` — Callback-based chaining for backends without
  linked SQEs
