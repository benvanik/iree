// device to device dispatch enqueue

#include <errno.h>
#include <fcntl.h>  // open
#include <inttypes.h>
#include <memory.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "third_party/hsa-runtime-headers/include/hsa/hsa.h"
#include "third_party/hsa-runtime-headers/include/hsa/hsa_ext_amd.h"

static inline size_t iree_host_align(size_t value, size_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

__attribute__((always_inline)) static inline void nontemporalMemcpy(
    void* __restrict dst, const void* __restrict src, size_t size) {
  memcpy(dst, src, size);
}

// AMD Signal Kind Enumeration Values.
enum iree_amd_signal_kind_t {
  IREE_AMD_SIGNAL_KIND_INVALID = 0,
  IREE_AMD_SIGNAL_KIND_USER = 1,
  IREE_AMD_SIGNAL_KIND_DOORBELL = -1,
  IREE_AMD_SIGNAL_KIND_LEGACY_DOORBELL = -2
};
typedef int64_t iree_amd_signal_kind64_t;

typedef struct __attribute__((aligned(64))) iree_amd_signal_s {
  iree_amd_signal_kind64_t kind;
  union {
    volatile int64_t value;
    volatile uint32_t* legacy_hardware_doorbell_ptr;
    volatile uint64_t* hardware_doorbell_ptr;
  };
  uint64_t event_mailbox_ptr;
  uint32_t event_id;
  uint32_t reserved1;
  uint64_t start_ts;
  uint64_t end_ts;
  union {
    /*iree_amd_queue_t*/ void* queue_ptr;
    uint64_t reserved2;
  };
  uint32_t reserved3[2];
} iree_amd_signal_t;

typedef struct {
  uint32_t agent_count;
  hsa_agent_t all_agents[32];
  hsa_agent_t cpu_agent;
  hsa_agent_t gpu_agents[2];
} agents_t;

static hsa_status_t iterate_agent(hsa_agent_t agent, void* user_data) {
  agents_t* agents = (agents_t*)user_data;
  agents->all_agents[agents->agent_count++] = agent;
  return HSA_STATUS_SUCCESS;
}

typedef struct {
  uint32_t count;
  hsa_amd_memory_pool_t pools[32];
} memory_pools_t;

static hsa_status_t iterate_memory_pool(hsa_amd_memory_pool_t memory_pool,
                                        void* user_data) {
  memory_pools_t* memory_pools = (memory_pools_t*)user_data;
  memory_pools->pools[memory_pools->count++] = memory_pool;
  return HSA_STATUS_SUCCESS;
}

typedef struct {
  uint32_t count;
  hsa_region_t regions[32];
} regions_t;

static hsa_status_t iterate_regions(hsa_region_t region, void* user_data) {
  regions_t* regions = (regions_t*)user_data;
  regions->regions[regions->count++] = region;
  return HSA_STATUS_SUCCESS;
}

static void gpu_queue_callback(hsa_status_t status, hsa_queue_t* queue,
                               void* user_data) {
  const char* status_str = NULL;
  hsa_status_string(status, &status_str);
  fprintf(stderr, "gpu_queue_callback %s", status_str);
}

int main(int argc, char** argv) {
  hsa_status_t err;

  err = hsa_init();

  agents_t agents;
  memset(&agents, 0, sizeof(agents));
  err = hsa_iterate_agents(iterate_agent, &agents);
  uint32_t gpu_count = 0;
  for (uint32_t i = 0; i < agents.agent_count; ++i) {
    hsa_device_type_t device_type = 0;
    err = hsa_agent_get_info(agents.all_agents[i], HSA_AGENT_INFO_DEVICE,
                             &device_type);
    if (device_type == HSA_DEVICE_TYPE_GPU) {
      agents.gpu_agents[gpu_count++] = agents.all_agents[i];
      err = hsa_agent_get_info(agents.all_agents[i],
                               (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NEAREST_CPU,
                               &agents.cpu_agent);
    }
  }

  memory_pools_t gpu_memory_pools[2];
  memset(&gpu_memory_pools, 0, sizeof(gpu_memory_pools));
  err = hsa_amd_agent_iterate_memory_pools(
      agents.gpu_agents[0], iterate_memory_pool, &gpu_memory_pools[0]);
  err = hsa_amd_agent_iterate_memory_pools(
      agents.gpu_agents[1], iterate_memory_pool, &gpu_memory_pools[1]);

  memory_pools_t cpu_memory_pools;
  memset(&cpu_memory_pools, 0, sizeof(cpu_memory_pools));
  err = hsa_amd_agent_iterate_memory_pools(
      agents.cpu_agent, iterate_memory_pool, &cpu_memory_pools);
  regions_t cpu_regions;
  memset(&cpu_regions, 0, sizeof(cpu_regions));
  err = hsa_agent_iterate_regions(agents.cpu_agent, iterate_regions,
                                  &cpu_regions);
  hsa_region_t cpu_fine_region;
  for (uint32_t i = 0; i < cpu_regions.count; ++i) {
    hsa_region_segment_t segment;
    err = hsa_region_get_info(cpu_regions.regions[i], HSA_REGION_INFO_SEGMENT,
                              &segment);
    if (segment == HSA_REGION_SEGMENT_GLOBAL) {
      hsa_region_global_flag_t flags;
      err = hsa_region_get_info(cpu_regions.regions[i],
                                HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
      if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
        cpu_fine_region = cpu_regions.regions[i];
      }
    }
  }

  hsa_amd_memory_pool_t cpu_fine_pool = {0};
  hsa_amd_memory_pool_t cpu_coarse_pool = {0};
  hsa_amd_memory_pool_t kernarg_pool = {0};
  for (uint32_t i = 0; i < cpu_memory_pools.count; ++i) {
    hsa_amd_memory_pool_t pool = cpu_memory_pools.pools[i];
    hsa_region_segment_t segment;
    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                       &segment);
    if (segment == HSA_REGION_SEGMENT_GLOBAL) {
      hsa_region_global_flag_t global_flag;
      err = hsa_amd_memory_pool_get_info(
          pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag);
      if (global_flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
        kernarg_pool = pool;
      } else if (global_flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
        cpu_coarse_pool = pool;
      } else if (global_flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
        cpu_fine_pool = pool;
      }
    }
  }

  hsa_amd_memory_pool_t device_fine_pools[2] = {0, 0};
  for (uint32_t j = 0; j < 2; ++j) {
    for (uint32_t i = 0; i < gpu_memory_pools[j].count; ++i) {
      hsa_amd_memory_pool_t pool = gpu_memory_pools[j].pools[i];
      hsa_region_segment_t segment;
      err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                         &segment);
      if (segment == HSA_REGION_SEGMENT_GLOBAL) {
        hsa_region_global_flag_t global_flag;
        err = hsa_amd_memory_pool_get_info(
            pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag);
        if (global_flag &
            (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
             HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED)) {
          device_fine_pools[j] = pool;
        }
      }
    }
  }

  // assumes same params
  uint32_t gpu_queue_min_size = 0;
  err = hsa_agent_get_info(agents.gpu_agents[0], HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                           &gpu_queue_min_size);
  uint32_t gpu_queue_max_size = 0;
  err = hsa_agent_get_info(agents.gpu_agents[0], HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                           &gpu_queue_max_size);
  uint32_t gpu_queue_size = gpu_queue_max_size;
  hsa_queue_t* gpu_queues[2] = {NULL, NULL};
  err = hsa_queue_create(agents.gpu_agents[0], gpu_queue_size,
                         HSA_QUEUE_TYPE_MULTI, gpu_queue_callback,
                         /*callback_data=*/NULL,
                         /*private_segment_size=*/UINT32_MAX,
                         /*group_segment_size=*/UINT32_MAX, &gpu_queues[0]);
  err = hsa_queue_create(agents.gpu_agents[1], gpu_queue_size,
                         HSA_QUEUE_TYPE_MULTI, gpu_queue_callback,
                         /*callback_data=*/NULL,
                         /*private_segment_size=*/UINT32_MAX,
                         /*group_segment_size=*/UINT32_MAX, &gpu_queues[1]);

  //
  hsa_file_t object_file =
      open("experimental/hsa_tests/kernels_cl.elf", O_RDONLY);
  // hsa_file_t object_file =
  //     open("experimental/hsa_tests/kernels_hip.elf", O_RDONLY);
  hsa_code_object_reader_t object_reader;
  err = hsa_code_object_reader_create_from_file(object_file, &object_reader);
  hsa_executable_t executable;
  err = hsa_executable_create_alt(HSA_PROFILE_FULL,
                                  HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL,
                                  &executable);
  err = hsa_executable_load_agent_code_object(executable, agents.gpu_agents[0],
                                              object_reader, NULL, NULL);
  err = hsa_executable_load_agent_code_object(executable, agents.gpu_agents[1],
                                              object_reader, NULL, NULL);
  err = hsa_executable_freeze(executable, NULL);

  struct kernel_info_t {
    uint64_t handle;
    uint32_t private_size;
    uint32_t group_size;
    uint32_t kernarg_alignment;
    uint32_t kernarg_size;
  } kernel_info[2];
  hsa_executable_symbol_t symbol;
  err = hsa_executable_get_symbol_by_name(executable, "issue_dispatch.kd",
                                          &agents.gpu_agents[0], &symbol);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_info[0].handle);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
      &kernel_info[0].private_size);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
      &kernel_info[0].group_size);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
      &kernel_info[0].kernarg_alignment);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
      &kernel_info[0].kernarg_size);
  err = hsa_executable_get_symbol_by_name(executable, "mul_x.kd",
                                          &agents.gpu_agents[1], &symbol);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_info[1].handle);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
      &kernel_info[1].private_size);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
      &kernel_info[1].group_size);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
      &kernel_info[1].kernarg_alignment);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
      &kernel_info[1].kernarg_size);

  void* kernarg_storages[2] = {NULL, NULL};
  err = hsa_amd_memory_pool_allocate(kernarg_pool, kernel_info[0].kernarg_size,
                                     HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
                                     &kernarg_storages[0]);
  err = hsa_amd_agents_allow_access(1, &agents.gpu_agents[0], NULL,
                                    kernarg_storages[0]);
  err = hsa_amd_memory_pool_allocate(kernarg_pool, kernel_info[1].kernarg_size,
                                     HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
                                     &kernarg_storages[1]);
  // must be for both ages (write 0/read 1)
  err = hsa_amd_agents_allow_access(2, agents.gpu_agents, NULL,
                                    kernarg_storages[1]);

  // allocate a fully device->device signal
  // allocated on device[1] (as that'll be waiting)
  // shared with both devices
  // in fine-grained memory (system-wide atomics)
  //
  // MAYBE: point-to-point HSA_AMD_MEMORY_POOL_PCIE_FLAG? forces fine-grained
  // even if pool is not? kmt_alloc_flags.ui32.CoarseGrain forced 0
  iree_amd_signal_t* device1_dispatch_signal = NULL;
  err = hsa_amd_memory_pool_allocate(
      device_fine_pools[1], sizeof(*device1_dispatch_signal),
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG, (void**)&device1_dispatch_signal);
  err = hsa_amd_agents_allow_access(2, agents.gpu_agents, NULL,
                                    device1_dispatch_signal);
  iree_amd_signal_t device1_dispatch_signal_init;
  memset(&device1_dispatch_signal_init, 0,
         sizeof(device1_dispatch_signal_init));
  device1_dispatch_signal_init.kind = IREE_AMD_SIGNAL_KIND_USER;
  device1_dispatch_signal_init.value = 1;
  device1_dispatch_signal_init.queue_ptr = gpu_queues[1];
  err = hsa_memory_copy(device1_dispatch_signal, &device1_dispatch_signal_init,
                        sizeof(*device1_dispatch_signal));

  hsa_signal_t dispatch_signals[2];
  dispatch_signals[0].handle = (uint64_t)device1_dispatch_signal;
  // err = hsa_amd_signal_create(1, 0, NULL, HSA_AMD_SIGNAL_AMD_GPU_ONLY,
  // &dispatch_signals[0]);
  err = hsa_signal_create(1, 0, NULL, &dispatch_signals[1]);

  typedef struct implicit_kernargs_t {
    uint32_t block_count[3];    // + 0/4/8
    uint16_t group_size[3];     // + 12/14/16
    uint16_t remainder[3];      // + 18/20/22
    uint64_t reserved0;         // + 24 hidden_tool_correlation_id
    uint64_t reserved1;         // + 32
    uint64_t global_offset[3];  // + 40/48/56
    uint16_t grid_dims;         // + 64
  } implicit_kernargs_t;

  uint32_t element_count = 65;
  uint32_t* buffer = NULL;
  err = hsa_amd_memory_pool_allocate(
      cpu_coarse_pool, element_count * sizeof(uint32_t),
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG, (void**)&buffer);
  for (uint32_t i = 0; i < element_count; ++i) {
    buffer[i] = i;
  }
  // buffer only used on 1 (and host)
  err = hsa_amd_agents_allow_access(1, &agents.gpu_agents[1], NULL, buffer);

  // gpu0: issue_dispatch
  // gpu1: mul_x
  // enqueue gpu1 before gpu0 to ensure wait happens

  typedef struct issue_dispatch_args_t {
    hsa_queue_t* queue;
    uint64_t mul_x_object;
    uint32_t mul_x_private_size;
    uint32_t mul_x_group_size;
    hsa_signal_t completion_signal;
    void* buffer;
    void* kernarg_storage;
    uint32_t element_count;
    uint32_t mul_by;
  } issue_dispatch_args_t;

  issue_dispatch_args_t* issue_dispatch_kernargs =
      (issue_dispatch_args_t*)kernarg_storages[0];
  issue_dispatch_kernargs->queue = gpu_queues[1];
  issue_dispatch_kernargs->mul_x_object = kernel_info[1].handle;
  issue_dispatch_kernargs->mul_x_private_size = kernel_info[1].private_size;
  issue_dispatch_kernargs->mul_x_group_size = kernel_info[1].group_size;
  issue_dispatch_kernargs->completion_signal = dispatch_signals[1];
  issue_dispatch_kernargs->buffer = buffer;
  issue_dispatch_kernargs->kernarg_storage = kernarg_storages[1];
  issue_dispatch_kernargs->element_count = element_count;
  issue_dispatch_kernargs->mul_by = 4;

// DO NOT SUBMIT no implicit needed?
#if 0
  implicit_kernargs_t* implicit_kernargs =
      (implicit_kernargs_t*)((uint8_t*)kernarg_storages[0] +
                             iree_host_align(sizeof(issue_dispatch_args_t), 8));
  implicit_kernargs->block_count[0] = 1;
  implicit_kernargs->block_count[1] = 1;
  implicit_kernargs->block_count[2] = 1;
  implicit_kernargs->group_size[0] = 1;
  implicit_kernargs->group_size[1] = 1;
  implicit_kernargs->group_size[2] = 1;
  implicit_kernargs->remainder[0] = 0;
  implicit_kernargs->remainder[1] = 0;
  implicit_kernargs->remainder[2] = 0;
  implicit_kernargs->reserved0 = 0;
  implicit_kernargs->reserved1 = 0;
  implicit_kernargs->global_offset[0] = 0;
  implicit_kernargs->global_offset[1] = 0;
  implicit_kernargs->global_offset[2] = 0;
  implicit_kernargs->grid_dims = 3;
#endif

  // DO NOT SUBMIT should do nontemporal kernarg update?

  hsa_kernel_dispatch_packet_t packet0;
  packet0.header = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
  packet0.setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  packet0.workgroup_size_x = 1;
  packet0.workgroup_size_y = 1;
  packet0.workgroup_size_z = 1;
  packet0.reserved0 = 0;
  packet0.grid_size_x = 1;
  packet0.grid_size_y = 1;
  packet0.grid_size_z = 1;
  packet0.private_segment_size = kernel_info[0].private_size;
  packet0.group_segment_size = kernel_info[0].group_size;
  packet0.kernel_object = kernel_info[0].handle;
  packet0.kernarg_address = kernarg_storages[0];
  packet0.reserved2 = 0;
  packet0.completion_signal = dispatch_signals[0];

  uint16_t packet0_header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  uint32_t packet0_header_setup = packet0_header | (packet0.setup << 16);

  uint64_t packet0_id = hsa_queue_add_write_index_screlease(gpu_queues[0], 1);
  while ((packet0_id - hsa_queue_load_read_index_acquire(gpu_queues[0])) >=
         gpu_queues[0]->size) {
    sleep(0);
  }

  hsa_kernel_dispatch_packet_t* packet0_ptr =
      (hsa_kernel_dispatch_packet_t*)((uint8_t*)gpu_queues[0]->base_address +
                                      (packet0_id & (gpu_queues[0]->size - 1)) *
                                          64);
  nontemporalMemcpy(packet0_ptr, &packet0, sizeof(packet0));
  atomic_store_explicit((volatile atomic_uint*)packet0_ptr,
                        packet0_header_setup, memory_order_release);

  // value ignored in MULTI cases
  hsa_signal_store_relaxed(gpu_queues[0]->doorbell_signal, packet0_id);

  hsa_signal_value_t wait_value =
      hsa_signal_wait_scacquire(dispatch_signals[1], HSA_SIGNAL_CONDITION_EQ, 0,
                                UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  if (wait_value != 0) {
    fprintf(stderr, "wait failed\n");
  }

  for (uint32_t i = 0; i < element_count; ++i) {
    fprintf(stderr, "%u ", buffer[i]);
  }
  fprintf(stderr, "\n");

  err = hsa_amd_memory_pool_free(buffer);

  // user allocated signal
  // err = hsa_signal_destroy(dispatch_signals[0]);
  err = hsa_signal_destroy(dispatch_signals[1]);

  err = hsa_amd_memory_pool_free(kernarg_storages[0]);
  err = hsa_amd_memory_pool_free(kernarg_storages[1]);

  err = hsa_queue_destroy(gpu_queues[0]);
  err = hsa_queue_destroy(gpu_queues[1]);

  err = hsa_executable_destroy(executable);
  err = hsa_code_object_reader_destroy(object_reader);

  err = hsa_shut_down();

  return 0;
}
