// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Buffer and slab registration for io_uring proactor.
//
// This module handles registration of memory regions with the kernel for
// zero-copy I/O operations. It supports:
//   - Simple buffer registration (wraps memory in a region)
//   - DMA-buf registration (mmaps GPU memory for CPU access)
//   - Slab registration (indexed buffers for zero-copy send/recv)

#include <errno.h>
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include "iree/async/platform/io_uring/buffer_ring.h"
#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/platform/io_uring/proactor.h"
#include "iree/base/internal/memory.h"

//===----------------------------------------------------------------------===//
// Buffer registration types
//===----------------------------------------------------------------------===//

// Combined allocation for registration entry + region.
// This keeps them together in memory and simplifies cleanup.
typedef struct iree_async_io_uring_buffer_registration_t {
  iree_async_buffer_registration_entry_t entry;
  iree_async_region_t region;
} iree_async_io_uring_buffer_registration_t;

// Combined allocation for dmabuf registration entry + region.
// Tracks the mmap state for cleanup.
typedef struct iree_async_io_uring_dmabuf_registration_t {
  iree_async_buffer_registration_entry_t entry;
  iree_async_region_t region;
  void* mapped_ptr;  // mmap'd address (for munmap on cleanup)
  iree_host_size_t mapped_length;
  int dmabuf_fd;  // Original fd (not owned, stored for region handles)
} iree_async_io_uring_dmabuf_registration_t;

// Destroy callback for buffer registration regions.
// Called when the region's ref count reaches zero.
static void iree_async_io_uring_buffer_registration_destroy(
    iree_async_region_t* region) {
  // Region is embedded at a known offset in the combined allocation.
  // This is safe because we only set this destroy_fn on regions we create
  // with iree_async_io_uring_buffer_registration_t layout.
  iree_async_io_uring_buffer_registration_t* registration =
      (iree_async_io_uring_buffer_registration_t*)((char*)region -
                                                   offsetof(
                                                       iree_async_io_uring_buffer_registration_t,
                                                       region));
  iree_allocator_free(region->proactor->allocator, registration);
}

// Cleanup function for buffer registrations.
// Called when the registration state is cleaned up.
static void iree_async_io_uring_buffer_registration_cleanup(
    void* entry_ptr, void* proactor_ptr) {
  iree_async_io_uring_buffer_registration_t* registration =
      (iree_async_io_uring_buffer_registration_t*)entry_ptr;
  (void)proactor_ptr;
  // Release our reference. If other code retained the region, it stays alive
  // until those references are released. The destroy callback frees the
  // combined allocation when the last reference drops.
  iree_async_region_release(&registration->region);
}

// Destroy callback for dmabuf registration regions.
// Called when the region's ref count reaches zero.
static void iree_async_io_uring_dmabuf_registration_destroy(
    iree_async_region_t* region) {
  iree_async_io_uring_dmabuf_registration_t* registration =
      (iree_async_io_uring_dmabuf_registration_t*)((char*)region -
                                                   offsetof(
                                                       iree_async_io_uring_dmabuf_registration_t,
                                                       region));
  // Unmap the dmabuf memory.
  if (registration->mapped_ptr) {
    munmap(registration->mapped_ptr, registration->mapped_length);
  }
  iree_allocator_free(region->proactor->allocator, registration);
}

// Cleanup function for dmabuf registrations.
// Called when the registration state is cleaned up.
static void iree_async_io_uring_dmabuf_registration_cleanup(
    void* entry_ptr, void* proactor_ptr) {
  iree_async_io_uring_dmabuf_registration_t* registration =
      (iree_async_io_uring_dmabuf_registration_t*)entry_ptr;
  (void)proactor_ptr;
  // Release our reference. The destroy callback handles munmap and free.
  iree_async_region_release(&registration->region);
}

//===----------------------------------------------------------------------===//
// Buffer registration vtable implementations
//===----------------------------------------------------------------------===//

iree_status_t iree_async_proactor_io_uring_register_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_entry = NULL;

  // Allocate combined entry + region.
  iree_async_io_uring_buffer_registration_t* registration = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(proactor->allocator, sizeof(*registration),
                                (void**)&registration));

  // Initialize the region.
  iree_async_region_t* region = &registration->region;
  iree_atomic_ref_count_init(&region->ref_count);
  region->proactor = proactor;
  region->slab = NULL;  // Not slab-backed.
  region->destroy_fn = iree_async_io_uring_buffer_registration_destroy;
  region->type = IREE_ASYNC_REGION_TYPE_IOURING;
  region->access_flags = access_flags;
  region->base_ptr = (void*)buffer.data;
  region->length = buffer.data_length;
  region->recycle = iree_async_buffer_recycle_callback_null();
  // For io_uring, we don't use IORING_REGISTER_BUFFERS for single buffers
  // because that API requires registering all buffers at once. Instead, we
  // just wrap the memory in a region for span-based access. The actual I/O
  // uses the raw address in the span.
  region->buffer_size = 0;
  region->buffer_count = 0;                      // Not indexed (use address).
  region->handles.iouring.buffer_group_id = -1;  // Not a provided buffer ring.
  region->handles.iouring.base_buffer_index = 0;

  // Initialize the entry.
  iree_async_buffer_registration_entry_t* entry = &registration->entry;
  entry->next = NULL;
  entry->proactor = proactor;
  entry->cleanup_fn = iree_async_io_uring_buffer_registration_cleanup;
  entry->region = region;

  // Link into the caller's state.
  iree_async_buffer_registration_state_add(state, entry);

  *out_entry = entry;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_async_proactor_io_uring_register_dmabuf(
    iree_async_proactor_t* base_proactor,
    iree_async_buffer_registration_state_t* state, int dmabuf_fd,
    uint64_t offset, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_entry = NULL;

  // TODO(benvanik): Implement true devmem TCP zero-copy when available.
  //
  // devmem TCP enables GPU→NIC zero-copy without CPU-side mmap:
  //   - Kernel 6.12+: RX path (SO_DEVMEM_DONTNEED, SCM_DEVMEM_DMABUF cmsg)
  //   - Kernel 6.13+: TX path
  //   - Requires NIC with header-split support (mlx5, ice, etc.)
  //   - Requires hardware flow steering configuration via ethtool
  //   - Requires netlink binding of dmabuf to specific RX/TX queues
  //
  // Current fallback: mmap the dmabuf and use standard I/O paths.
  // This provides coherent access to GPU memory but involves CPU copies.

  // Determine mmap protection flags from access flags.
  int prot = 0;
  if (access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_READ) prot |= PROT_READ;
  if (access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE) prot |= PROT_WRITE;

  // mmap requires page-aligned offset. Align offset down to page boundary and
  // adjust length up to cover the full requested range. We track the delta so
  // we can set base_ptr to point to the user's requested data, not the page
  // boundary.
  iree_host_size_t page_size = iree_memory_query_info().normal_page_size;
  uint64_t aligned_offset = offset & ~((uint64_t)page_size - 1);
  iree_host_size_t offset_delta = (iree_host_size_t)(offset - aligned_offset);
  iree_host_size_t aligned_length =
      iree_host_align(offset_delta + length, page_size);

  // mmap the dmabuf fd to get a CPU-accessible pointer.
  void* mapped_ptr =
      mmap(NULL, aligned_length, prot, MAP_SHARED, dmabuf_fd, aligned_offset);
  if (mapped_ptr == MAP_FAILED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "mmap of dmabuf fd %d failed", dmabuf_fd);
  }

  // Allocate registration struct.
  iree_async_io_uring_dmabuf_registration_t* registration = NULL;
  iree_status_t status = iree_allocator_malloc(
      base_proactor->allocator, sizeof(*registration), (void**)&registration);
  if (!iree_status_is_ok(status)) {
    munmap(mapped_ptr, aligned_length);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  memset(registration, 0, sizeof(*registration));

  // Initialize region with DMABUF type.
  // base_ptr points to the user's requested offset within the mapped range.
  iree_async_region_t* region = &registration->region;
  iree_atomic_ref_count_init(&region->ref_count);
  region->proactor = base_proactor;
  region->slab = NULL;  // Not slab-backed.
  region->destroy_fn = iree_async_io_uring_dmabuf_registration_destroy;
  region->type = IREE_ASYNC_REGION_TYPE_DMABUF;
  region->base_ptr = (uint8_t*)mapped_ptr + offset_delta;
  region->length = length;
  region->access_flags = access_flags;
  region->recycle = iree_async_buffer_recycle_callback_null();
  region->handles.dmabuf.fd = dmabuf_fd;
  region->handles.dmabuf.offset = offset;

  // Track aligned mmap parameters for cleanup.
  registration->mapped_ptr = mapped_ptr;
  registration->mapped_length = aligned_length;
  registration->dmabuf_fd = dmabuf_fd;

  // Setup entry and cleanup.
  iree_async_buffer_registration_entry_t* entry = &registration->entry;
  entry->next = NULL;
  entry->proactor = base_proactor;
  entry->cleanup_fn = iree_async_io_uring_dmabuf_registration_cleanup;
  entry->region = region;
  iree_async_buffer_registration_state_add(state, entry);

  *out_entry = entry;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_async_proactor_io_uring_unregister_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_entry_t* entry,
    iree_async_buffer_registration_state_t* state) {
  // Remove from the registration state list.
  iree_async_buffer_registration_state_remove(state, entry);
  // Invoke the cleanup callback to free resources (munmap for dmabuf, etc).
  entry->cleanup_fn(entry, proactor);
}

//===----------------------------------------------------------------------===//
// Slab registration (indexed zero-copy)
//===----------------------------------------------------------------------===//

// Recycle callback for slab regions with provided buffer rings.
// Called from iree_async_buffer_lease_release() to return a buffer to the
// kernel's PBUF_RING for multishot recv operations.
static void iree_async_io_uring_slab_region_recycle(void* context,
                                                    uint32_t index) {
  iree_io_uring_buffer_ring_t* buffer_ring =
      (iree_io_uring_buffer_ring_t*)context;
  iree_io_uring_buffer_ring_recycle(buffer_ring, (uint16_t)index);
}

// Region storage for slab registrations. Heap-allocated, returned to caller.
// Contains the region plus tracking state for cleanup.
typedef struct iree_async_io_uring_slab_region_t {
  iree_async_region_t region;
  // Optional: provided buffer ring for recv operations.
  // Created when access_flags includes WRITE. NULL for send-only.
  iree_io_uring_buffer_ring_t* buffer_ring;
  // True if this region registered with the fixed buffer table
  // (IORING_REGISTER_BUFFERS). Used to know whether to unregister.
  bool registered_fixed_buffers;
  // Allocator used for freeing this struct.
  iree_allocator_t allocator;
} iree_async_io_uring_slab_region_t;

// Destroy callback for slab regions. Called when region ref count reaches zero.
// Unregisters from kernel, frees buffer ring, releases slab ref, frees struct.
static void iree_async_io_uring_slab_region_destroy(
    iree_async_region_t* region) {
  iree_async_io_uring_slab_region_t* slab_region =
      (iree_async_io_uring_slab_region_t*)region;
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(region->proactor);

  // Free the provided buffer ring if present (for recv).
  if (slab_region->buffer_ring) {
    iree_io_uring_buffer_ring_free(slab_region->buffer_ring);
    slab_region->buffer_ring = NULL;
  }

  // Unregister the fixed buffer table from the kernel if we registered it.
  if (slab_region->registered_fixed_buffers &&
      proactor->registered_buffer_count > 0) {
    long ret = 0;
    int saved_errno = 0;
    do {
      ret = syscall(IREE_IO_URING_SYSCALL_REGISTER, proactor->ring.ring_fd,
                    IREE_IORING_UNREGISTER_BUFFERS, NULL, 0);
      saved_errno = errno;
    } while (ret < 0 && saved_errno == EINTR);
    if (ret < 0) {
      // EBUSY means the kernel still has SQEs referencing this buffer table.
      // This is a programming error: the region was released while I/O was
      // still in-flight. We MUST NOT free the slab memory or the kernel will
      // DMA into freed/reallocated pages. Abort immediately.
      IREE_ASSERT(false,
                  "IORING_UNREGISTER_BUFFERS failed with errno %d; if EBUSY, "
                  "the region was released while I/O was in-flight - this is "
                  "a fatal programming error",
                  saved_errno);
      // In case asserts are disabled, leak rather than corrupt.
      return;
    }
    proactor->registered_buffer_count = 0;
  }

  // Release the slab reference.
  if (region->slab) {
    iree_async_slab_release(region->slab);
  }

  // Free the combined allocation.
  iree_allocator_free(slab_region->allocator, slab_region);
}

iree_status_t iree_async_proactor_io_uring_register_slab(
    iree_async_proactor_t* base_proactor, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_region_t** out_region) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(slab);
  IREE_ASSERT_ARGUMENT(out_region);
  *out_region = NULL;

  iree_host_size_t buffer_size = iree_async_slab_buffer_size(slab);
  iree_host_size_t buffer_count = iree_async_slab_buffer_count(slab);
  void* base_ptr = iree_async_slab_base_ptr(slab);

  // Check singleton constraint for fixed buffer table (READ access).
  // io_uring allows only one buffer table registration at a time.
  bool needs_fixed_buffers =
      (access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_READ) != 0;
  if (needs_fixed_buffers && proactor->registered_buffer_count > 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "fixed buffer table already registered with this proactor; io_uring "
        "allows only one buffer table per ring");
  }

  // Validate buffer count fits in the region handles.
  if (buffer_count > UINT16_MAX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_count %" PRIhsz
                            " exceeds maximum %u for indexed zero-copy",
                            buffer_count, (unsigned)UINT16_MAX);
  }

  // If slab will be used for recv (WRITE access) AND the kernel supports
  // MULTISHOT (5.19+), create a provided buffer ring for kernel-managed buffer
  // selection.
  bool create_recv_ring =
      (access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE) != 0 &&
      iree_any_bit_set(proactor->capabilities,
                       IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT);
  // PBUF_RING requires power-of-2 buffer count.
  if (create_recv_ring && (buffer_count & (buffer_count - 1)) != 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_count %" PRIhsz
                            " must be power of 2 for recv registrations",
                            buffer_count);
  }

  // Validate buffer size fits in the region handles.
  if (buffer_size > UINT32_MAX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_size %" PRIhsz
                            " exceeds maximum %u for indexed zero-copy",
                            buffer_size, (unsigned)UINT32_MAX);
  }

  // Reject zero buffer_size - would cause divide-by-zero in index derivation.
  if (buffer_size == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_size must be > 0 for slab registration");
  }

  // Allocate the slab region struct.
  iree_async_io_uring_slab_region_t* slab_region = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(base_proactor->allocator, sizeof(*slab_region),
                                (void**)&slab_region));
  memset(slab_region, 0, sizeof(*slab_region));
  slab_region->allocator = base_proactor->allocator;

  // Register with kernel's fixed buffer table for send path (READ access).
  if (needs_fixed_buffers) {
    // Build iovec array for IORING_REGISTER_BUFFERS.
    struct iovec* iovecs = NULL;
    struct iovec stack_iovecs[64];
    bool iovecs_heap_allocated = false;
    if (buffer_count <= 64) {
      iovecs = stack_iovecs;
    } else {
      iree_status_t status = iree_allocator_malloc(
          base_proactor->allocator, buffer_count * sizeof(struct iovec),
          (void**)&iovecs);
      if (!iree_status_is_ok(status)) {
        iree_allocator_free(base_proactor->allocator, slab_region);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
      iovecs_heap_allocated = true;
    }

    for (iree_host_size_t i = 0; i < buffer_count; ++i) {
      iovecs[i].iov_base = (uint8_t*)base_ptr + i * buffer_size;
      iovecs[i].iov_len = buffer_size;
    }

    // Register with kernel via IORING_REGISTER_BUFFERS.
    long ret = 0;
    do {
      ret = syscall(IREE_IO_URING_SYSCALL_REGISTER, proactor->ring.ring_fd,
                    IREE_IORING_REGISTER_BUFFERS, iovecs, buffer_count);
    } while (ret < 0 && errno == EINTR);

    if (iovecs_heap_allocated) {
      iree_allocator_free(base_proactor->allocator, iovecs);
    }

    if (ret < 0) {
      iree_allocator_free(base_proactor->allocator, slab_region);
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(iree_status_code_from_errno(errno),
                              "IORING_REGISTER_BUFFERS failed (%d)", errno);
    }

    proactor->registered_buffer_count = (uint16_t)buffer_count;
    slab_region->registered_fixed_buffers = true;
  }

  // Create provided buffer ring for recv path (WRITE access).
  iree_io_uring_buffer_ring_t* buffer_ring = NULL;
  int16_t buffer_group_id = -1;
  if (create_recv_ring) {
    iree_io_uring_buffer_ring_options_t ring_options =
        iree_io_uring_buffer_ring_options_default();
    ring_options.buffer_base = base_ptr;
    ring_options.buffer_size = buffer_size;
    ring_options.buffer_count = buffer_count;
    ring_options.group_id = proactor->next_group_id++;

    iree_status_t status = iree_io_uring_buffer_ring_allocate(
        proactor->ring.ring_fd, &ring_options, base_proactor->allocator,
        &buffer_ring);
    if (!iree_status_is_ok(status)) {
      // Unregister the buffer table on failure if we registered it.
      if (slab_region->registered_fixed_buffers) {
        syscall(IREE_IO_URING_SYSCALL_REGISTER, proactor->ring.ring_fd,
                IREE_IORING_UNREGISTER_BUFFERS, NULL, 0);
        proactor->registered_buffer_count = 0;
      }
      iree_allocator_free(base_proactor->allocator, slab_region);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    buffer_group_id = (int16_t)ring_options.group_id;
  }
  slab_region->buffer_ring = buffer_ring;

  // Initialize the region.
  iree_async_region_t* region = &slab_region->region;
  iree_atomic_ref_count_init(&region->ref_count);
  region->proactor = base_proactor;
  region->slab = slab;
  iree_async_slab_retain(slab);
  region->destroy_fn = iree_async_io_uring_slab_region_destroy;
  region->type = IREE_ASYNC_REGION_TYPE_IOURING;
  region->access_flags = access_flags;
  region->base_ptr = base_ptr;
  region->length = iree_async_slab_total_size(slab);

  // Set recycle callback for recv regions with provided buffer rings.
  if (buffer_ring) {
    region->recycle.fn = iree_async_io_uring_slab_region_recycle;
    region->recycle.user_data = buffer_ring;
  } else {
    region->recycle = iree_async_buffer_recycle_callback_null();
  }

  // Store indexed buffer info for SEND_ZC index derivation and recv.
  region->buffer_size = buffer_size;
  region->buffer_count = (uint32_t)buffer_count;
  region->handles.iouring.buffer_group_id = buffer_group_id;
  region->handles.iouring.base_buffer_index = 0;

  *out_region = region;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
