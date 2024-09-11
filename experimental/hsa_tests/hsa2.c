// pc_sampling?
// doesn't work

#include <alloca.h>
#include <errno.h>
#include <fcntl.h>  // open
#include <memory.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
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

typedef struct {
  uint32_t agent_count;
  hsa_agent_t all_agents[32];
  hsa_agent_t cpu_agent;
  hsa_agent_t gpu_agent;
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

static hsa_status_t pcs_config_callback(
    const hsa_ven_amd_pcs_configuration_t* configuration, void* callback_data) {
  //
  return HSA_STATUS_SUCCESS;
}

#include "/home/nod/src/ROCR-Runtime/libhsakmt/include/hsakmt/hsakmt.h"
#include "/home/nod/src/ROCR-Runtime/libhsakmt/include/hsakmt/linux/kfd_ioctl.h"

void pcs_data_ready(void* client_callback_data, size_t data_size,
                    size_t lost_sample_count,
                    hsa_ven_amd_pcs_data_copy_callback_t data_copy_callback,
                    void* hsa_callback_data) {
  perf_sample_snapshot_v1_t* sample_buffer =
      (perf_sample_snapshot_v1_t*)client_callback_data;
  fprintf(stderr, "PCS; data size = %zu, lost samples = %zu\n", data_size,
          lost_sample_count);
  hsa_status_t err =
      data_copy_callback(hsa_callback_data, data_size, sample_buffer);
}

int main(int argc, char** argv) {
  hsa_status_t err;

  err = hsa_init();

  agents_t agents;
  memset(&agents, 0, sizeof(agents));
  err = hsa_iterate_agents(iterate_agent, &agents);
  for (uint32_t i = 0; i < agents.agent_count; ++i) {
    hsa_device_type_t device_type = 0;
    err = hsa_agent_get_info(agents.all_agents[i], HSA_AGENT_INFO_DEVICE,
                             &device_type);
    if (device_type == HSA_DEVICE_TYPE_GPU) {
      agents.gpu_agent = agents.all_agents[i];
      err = hsa_agent_get_info(agents.gpu_agent,
                               (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NEAREST_CPU,
                               &agents.cpu_agent);
      break;
    }
  }

  memory_pools_t gpu_memory_pools;
  memset(&gpu_memory_pools, 0, sizeof(gpu_memory_pools));
  err = hsa_amd_agent_iterate_memory_pools(
      agents.gpu_agent, iterate_memory_pool, &gpu_memory_pools);
  regions_t gpu_regions;
  memset(&gpu_regions, 0, sizeof(gpu_regions));
  err = hsa_agent_iterate_regions(agents.gpu_agent, iterate_regions,
                                  &gpu_regions);

  memory_pools_t cpu_memory_pools;
  memset(&cpu_memory_pools, 0, sizeof(cpu_memory_pools));
  err = hsa_amd_agent_iterate_memory_pools(
      agents.cpu_agent, iterate_memory_pool, &cpu_memory_pools);
  regions_t cpu_regions;
  memset(&cpu_regions, 0, sizeof(cpu_regions));
  err = hsa_agent_iterate_regions(agents.cpu_agent, iterate_regions,
                                  &cpu_regions);

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

  uint32_t gpu_queue_min_size = 0;
  err = hsa_agent_get_info(agents.gpu_agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                           &gpu_queue_min_size);
  uint32_t gpu_queue_max_size = 0;
  err = hsa_agent_get_info(agents.gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                           &gpu_queue_max_size);
  uint32_t gpu_queue_size = gpu_queue_max_size;
  hsa_queue_t* gpu_queue = NULL;
  err = hsa_queue_create(agents.gpu_agent, gpu_queue_size, HSA_QUEUE_TYPE_MULTI,
                         gpu_queue_callback,
                         /*callback_data=*/NULL,
                         /*private_segment_size=*/UINT32_MAX,
                         /*group_segment_size=*/UINT32_MAX, &gpu_queue);

  uint32_t gpu_node_id = 0;
  err = hsa_agent_get_info(agents.gpu_agent, HSA_AGENT_INFO_NODE, &gpu_node_id);

  int kmtfd = open("/dev/kfd", O_RDWR);
  struct kfd_ioctl_pc_sample_args args;
  memset(&args, 0, sizeof(args));
  args.op = KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES;
  // cat /sys/devices/virtual/kfd/kfd/topology/nodes/1/gpu_id
  args.gpu_id = 56588;
  args.sample_info_ptr = 0;  //(uint64_t)sample_info;
  args.num_sample_info = 0;  // sample_info_sz;
  args.flags = 0;
  int ioctl_ret = 0;
  do {
    ioctl_ret = ioctl(kmtfd, AMDKFD_IOC_PC_SAMPLE, &args);
  } while (ioctl_ret == -1 && (errno == EINTR || errno == EAGAIN));
  fprintf(stderr, "ioctl %d\n", ioctl_ret);
  // uint32_t size = 0;
  // HSAKMT_STATUS ret =
  //     hsaKmtPcSamplingQueryCapabilities(gpu_node_id, NULL, 0, &size);
  // if (ret != HSAKMT_STATUS_SUCCESS || size == 0) {
  //   fprintf(stderr, "KMT FAIL\n");
  // }

  // HsaPcSamplingInfo* sample_info_list =
  //     alloca(size * sizeof(HsaPcSamplingInfo));
  // ret = hsaKmtPcSamplingQueryCapabilities(gpu_node_id, sample_info_list,
  // size,
  //                                         &size);

  // iterate configs not found?
  // HSA_EXTENSION_AMD_PC_SAMPLING extension bit
  // err = hsa_ven_amd_pcs_iterate_configuration(agents.gpu_agent,
  // pcs_config_callback, NULL);
  perf_sample_snapshot_v1_t sample_buffer[16 * 1024];
  hsa_ven_amd_pcs_t pc_sampling;
  err = hsa_ven_amd_pcs_create(
      agents.gpu_agent, HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1,
      HSA_VEN_AMD_PCS_INTERVAL_UNITS_MICRO_SECONDS, 100, 0,
      sizeof(sample_buffer), pcs_data_ready, sample_buffer, &pc_sampling);

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
  err = hsa_executable_load_agent_code_object(executable, agents.gpu_agent,
                                              object_reader, NULL, NULL);
  err = hsa_executable_freeze(executable, NULL);

  struct kernel_info_t {
    uint64_t handle;
    uint32_t private_size;
    uint32_t group_size;
    uint32_t kernarg_alignment;
    uint32_t kernarg_size;
  } kernel_info;
  hsa_executable_symbol_t symbol;
  err = hsa_executable_get_symbol_by_name(executable, "add_one.kd",
                                          &agents.gpu_agent, &symbol);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_info.handle);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
      &kernel_info.private_size);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
      &kernel_info.group_size);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
      &kernel_info.kernarg_alignment);
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
      &kernel_info.kernarg_size);

  err = hsa_ven_amd_pcs_start(pc_sampling);

  void* kernarg_storage = NULL;
  err = hsa_amd_memory_pool_allocate(kernarg_pool, kernel_info.kernarg_size,
                                     HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
                                     &kernarg_storage);
  err =
      hsa_amd_agents_allow_access(1, &agents.gpu_agent, NULL, kernarg_storage);

  hsa_signal_t dispatch_signal;
  err = hsa_signal_create(1, 0, NULL, &dispatch_signal);

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
  err = hsa_amd_agents_allow_access(1, &agents.gpu_agent, NULL, buffer);

  uint32_t grid_size[3] = {element_count, 1, 1};
  uint16_t workgroup_size[3] = {32, 1, 1};

  typedef struct add_one_args_t {
    uint32_t n;
    void* buffer;
  } add_one_args_t;

  add_one_args_t* explicit_kernargs = (add_one_args_t*)kernarg_storage;
  explicit_kernargs->n = element_count;
  explicit_kernargs->buffer = buffer;

  implicit_kernargs_t* implicit_kernargs =
      (implicit_kernargs_t*)((uint8_t*)kernarg_storage +
                             iree_host_align(sizeof(add_one_args_t), 8));
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
  packet.header = HSA_PACKET_TYPE_INVALID;
  packet.setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  packet.workgroup_size_x = workgroup_size[0];
  packet.workgroup_size_y = workgroup_size[1];
  packet.workgroup_size_z = workgroup_size[2];
  packet.reserved0 = 0;
  packet.grid_size_x = grid_size[0];
  packet.grid_size_y = grid_size[1];
  packet.grid_size_z = grid_size[2];
  packet.private_segment_size = kernel_info.private_size;
  packet.group_segment_size = kernel_info.group_size;
  packet.kernel_object = kernel_info.handle;
  packet.kernarg_address = kernarg_storage;
  packet.reserved2 = 0;
  packet.completion_signal = dispatch_signal;

  uint16_t packet_header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  uint32_t packet_header_setup = packet_header | (packet.setup << 16);

  uint64_t packet_id = hsa_queue_add_write_index_screlease(gpu_queue, 1);
  while ((packet_id - hsa_queue_load_read_index_acquire(gpu_queue)) >=
         gpu_queue->size) {
    sleep(0);
  }

  hsa_kernel_dispatch_packet_t* packet_ptr =
      (hsa_kernel_dispatch_packet_t*)((uint8_t*)gpu_queue->base_address +
                                      (packet_id & (gpu_queue->size - 1)) * 64);
  nontemporalMemcpy(packet_ptr, &packet, sizeof(packet));
  atomic_store_explicit((volatile atomic_uint*)packet_ptr, packet_header_setup,
                        memory_order_release);

  // value ignored in MULTI cases
  hsa_signal_store_relaxed(gpu_queue->doorbell_signal, packet_id);

  hsa_signal_value_t wait_value =
      hsa_signal_wait_scacquire(dispatch_signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  if (wait_value != 0) {
    fprintf(stderr, "wait failed\n");
  }

  err = hsa_ven_amd_pcs_stop(pc_sampling);
  err = hsa_ven_amd_pcs_flush(pc_sampling);

  for (uint32_t i = 0; i < element_count; ++i) {
    fprintf(stderr, "%u ", buffer[i]);
  }
  fprintf(stderr, "\n");

  err = hsa_amd_memory_pool_free(buffer);

  err = hsa_signal_destroy(dispatch_signal);

  err = hsa_amd_memory_pool_free(kernarg_storage);

  err = hsa_ven_amd_pcs_destroy(pc_sampling);

  err = hsa_queue_destroy(gpu_queue);

  err = hsa_executable_destroy(executable);
  err = hsa_code_object_reader_destroy(object_reader);

  err = hsa_shut_down();

  return 0;
}
