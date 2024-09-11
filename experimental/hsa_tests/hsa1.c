// dispatch hip or opencl kernel and timing

#include <errno.h>
#include <fcntl.h>  // open
#include <fcntl.h>
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

// AMD Signal Kind Enumeration Values.
enum iree_amd_signal_kind_t {
  IREE_AMD_SIGNAL_KIND_INVALID = 0,
  IREE_AMD_SIGNAL_KIND_USER = 1,
  IREE_AMD_SIGNAL_KIND_DOORBELL = -1,
  IREE_AMD_SIGNAL_KIND_LEGACY_DOORBELL = -2
};
typedef int64_t iree_amd_signal_kind64_t;

#define IREE_CL_GLOBAL
#define IREE_CL_ALIGNAS(x) __attribute__((aligned(x)))
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

/* Call ioctl, restarting if it is interrupted */
int hsakmt_ioctl(int fd, unsigned long request, void* arg) {
  int ret;
  do {
    ret = ioctl(fd, request, arg);
  } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
  return ret;
}

#define AMDKFD_IOCTL_BASE 'K'
#define AMDKFD_IO(nr) _IO(AMDKFD_IOCTL_BASE, nr)
#define AMDKFD_IOR(nr, type) _IOR(AMDKFD_IOCTL_BASE, nr, type)
#define AMDKFD_IOW(nr, type) _IOW(AMDKFD_IOCTL_BASE, nr, type)
#define AMDKFD_IOWR(nr, type) _IOWR(AMDKFD_IOCTL_BASE, nr, type)
#define AMDKFD_IOC_GET_CLOCK_COUNTERS \
  AMDKFD_IOWR(0x05, struct kfd_ioctl_get_clock_counters_args)
struct kfd_ioctl_get_clock_counters_args {
  uint64_t gpu_clock_counter;    /* from KFD */
  uint64_t cpu_clock_counter;    /* from KFD */
  uint64_t system_clock_counter; /* from KFD */
  uint64_t system_clock_freq;    /* from KFD */
  uint32_t gpu_id;               /* to KFD */
  uint32_t pad;
};

#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64) || \
    defined(__amd64__) || defined(_M_X64) || defined(_M_AMD64)
#include <xmmintrin.h>
#define IS_X86 1
#endif

__attribute__((always_inline)) static inline void nontemporalMemcpy(
    void* __restrict dst, const void* __restrict src, size_t size) {
  // #if defined(IS_X86)
  // #if defined(__AVX512F__)
  // #pragma unroll
  //   for (size_t i = 0; i != size / sizeof(__m512i); ++i) {
  //     _mm512_stream_si512((__m512i* __restrict)(dst),
  //                         *(const __m512i* __restrict)(src));
  //     src += sizeof(__m512i);
  //     dst += sizeof(__m512i);
  //   }
  //   size = size % sizeof(__m512i);
  // #endif
  // #if defined(__AVX__)
  // #pragma unroll
  //   for (size_t i = 0; i != size / sizeof(__m256i); ++i) {
  //     _mm256_stream_si256((__m256i* __restrict)(dst),
  //                         *(const __m256i* __restrict)(src));
  //     src += sizeof(__m256i);
  //     dst += sizeof(__m256i);
  //   }
  //   size = size % sizeof(__m256i);
  // #endif
  // #pragma unroll
  //   for (size_t i = 0; i != size / sizeof(__m128i); ++i) {
  //     _mm_stream_si128((__m128i* __restrict)(dst),
  //                      *((const __m128i* __restrict)(src)));
  //     src += sizeof(__m128i);
  //     dst += sizeof(__m128i);
  //   }
  //   size = size % sizeof(__m128i);
  // #pragma unroll
  //   for (size_t i = 0; i != size / sizeof(long long); ++i) {
  //     _mm_stream_si64((long long* __restrict)(dst),
  //                     *(const long long* __restrict)(src));
  //     src += sizeof(long long);
  //     dst += sizeof(long long);
  //   }
  //   size = size % sizeof(long long);
  // #pragma unroll
  //   for (size_t i = 0; i != size / sizeof(int); ++i) {
  //     _mm_stream_si32((int* __restrict)(dst), *(const int* __restrict)(src));
  //     src += sizeof(int);
  //     dst += sizeof(int);
  //   }
  // #else
  memcpy(dst, src, size);
  // #endif
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

  // for (uint32_t i = 0; i < gpu_regions.count; ++i) {
  //   hsa_region_t region = gpu_regions.regions[i];
  //   hsa_region_segment_t segment;
  //   err = hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
  //   if (segment == HSA_REGION_SEGMENT_GLOBAL) {
  //     hsa_region_global_flag_t flags;
  //     err = hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS,
  //     &flags); if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
  //       bool host_accessible = false;
  //       err = hsa_region_get_info(
  //           region, (hsa_region_info_t)HSA_AMD_REGION_INFO_HOST_ACCESSIBLE,
  //           &host_accessible);
  //       void* base_address = NULL;
  //       err = hsa_region_get_info(
  //           region, (hsa_region_info_t)HSA_AMD_REGION_INFO_BASE,
  //           &base_address);
  //     }
  //   }
  // }

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
  err = hsa_executable_get_symbol_by_name(
      executable, "add_one_with_timestamp.kd", &agents.gpu_agent, &symbol);
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

  void* kernarg_storage = NULL;
  err = hsa_amd_memory_pool_allocate(kernarg_pool, kernel_info.kernarg_size,
                                     HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
                                     &kernarg_storage);
  err =
      hsa_amd_agents_allow_access(1, &agents.gpu_agent, NULL, kernarg_storage);

  hsa_signal_t dispatch_signal;
  err = hsa_signal_create(1, 0, NULL, &dispatch_signal);
  // err = hsa_amd_signal_create(1, 0, NULL, HSA_AMD_SIGNAL_IPC,
  // &dispatch_signal);

  err = hsa_amd_profiling_set_profiler_enabled(gpu_queue, 1);

  // submit
  // hidden args
  // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/rocm/rocvirtual.cpp#L3134
  // uint8_t kind
  // uint8_t value
  // uint16_t offset
  //
  // or hardcode
  // uint16_t implicit_offset;
  // uint8_t kind_offsets[hidden enum];
  // if (kind_offsets[hidden_block_count_x] != 0xFF) {
  //   kernargs[kind_offsets[hidden_block_count_x]] = block_count_x;
  // }
  // ...
  // keeps data small, no loops just conditional moves
  // could have a bit for all?
  //
  // or bitmap of hidden args
  // bit indicates presence
  // assume dense and ordered because of the way the compiler does things
  //
  // amdgpu-no-implicitarg-ptr is the only way to disable
  // otherwise all are required with a fixed 256b size
  // so just need to find the base offset
  //
  // explicit args always start at 0
  // implicit are explicit + 8-byte aligned
  //
  // third_party/llvm-project/llvm/lib/Target/AMDGPU/AMDGPUHSAMetadataStreamer.cpp
  typedef struct implicit_kernargs_t {
    // Grid dispatch workgroup count.
    // Some languages, such as OpenCL, support a last workgroup in each
    // dimension being partial. This count only includes the non-partial
    // workgroup count. This is not the same as the value in the AQL dispatch
    // packet, which has the grid size in workitems.
    //
    // Represented in metadata as:
    //   hidden_block_count_x
    //   hidden_block_count_y
    //   hidden_block_count_z
    uint32_t block_count[3];  // + 0/4/8

    // Grid dispatch workgroup size.
    // This size only applies to the non-partial workgroups. This is the same
    // value as the AQL dispatch packet workgroup size.
    //
    // Represented in metadata as:
    //   hidden_group_size_x
    //   hidden_group_size_y
    //   hidden_group_size_z
    uint16_t group_size[3];  // + 12/14/16

    // Grid dispatch work group size of the partial work group, if it exists.
    // Any dimension that does not exist must be 0.
    //
    // Represented in metadata as:
    //   hidden_remainder_x
    //   hidden_remainder_y
    //   hidden_remainder_z
    uint16_t remainder[3];  // + 18/20/22

    uint64_t reserved0;  // + 24 hidden_tool_correlation_id
    uint64_t reserved1;  // + 32

    // OpenCL grid dispatch global offset.
    //
    // Represented in metadata as:
    //   hidden_global_offset_x
    //   hidden_global_offset_y
    //   hidden_global_offset_z
    uint64_t global_offset[3];  // + 40/48/56

    // Grid dispatch dimensionality. This is the same value as the AQL
    // dispatch packet dimensionality. Must be a value between 1 and 3.
    //
    // Represented in metadata as:
    //   hidden_grid_dims
    uint16_t grid_dims;  // + 64
  } implicit_kernargs_t;

  uint32_t element_count = 64;
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

  // can directly access agent-specific ticks on the signal
  // could store these as part of the trace provider
  // must call hsa_amd_profiling_set_profiler_enabled to ensure populated
  // request a batch hsa_amd_profiling_convert_tick_to_system_domain?
  // may still want to adjust, but avoid API overheads when converting 1000's
  iree_amd_signal_t* amd_dispatch_signal =
      (iree_amd_signal_t*)dispatch_signal.handle;
  // amd_dispatch_signal->start_ts;
  // amd_dispatch_signal->end_ts;
  uint64_t start_ts = 0;
  err = hsa_amd_profiling_convert_tick_to_system_domain(
      agents.gpu_agent, amd_dispatch_signal->start_ts, &start_ts);
  uint64_t end_ts = 0;
  err = hsa_amd_profiling_convert_tick_to_system_domain(
      agents.gpu_agent, amd_dispatch_signal->end_ts, &end_ts);

  // (end - start) / system_frequency = seconds

  hsa_amd_profiling_dispatch_time_t time;
  err = hsa_amd_profiling_get_dispatch_time(agents.gpu_agent, dispatch_signal,
                                            &time);
  uint64_t system_frequency = 0;
  err = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                            &system_frequency);

  uint64_t system_timestamp = 0;
  err = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &system_timestamp);
  // sleep(10);
  // uint64_t system_timestamp2 = 0;
  // err = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &system_timestamp2);

  uint64_t kernel_time = ((uint64_t)buffer[0]) << 32 | buffer[1];

  // use kmt to query system/gpu time
  // can use to adjust like GpuAgent::TranslateTime
  // https://sourcegraph.com/github.com/ROCm/ROCR-Runtime@909b82d4632b86dff0faadcb19488a43d2108686/-/blob/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp?L2048
  static const char kfd_device_name[] = "/dev/kfd";
  int hsakmt_kfd_fd = open(kfd_device_name, O_RDWR | O_CLOEXEC);
  if (hsakmt_kfd_fd == -1) {
    // result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
    abort();
  }
  struct kfd_ioctl_get_clock_counters_args args;
  int kmt_err =
      hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_GET_CLOCK_COUNTERS, &args);
  if (hsakmt_kfd_fd) {
    close(hsakmt_kfd_fd);
    hsakmt_kfd_fd = -1;
  }

  for (uint32_t i = 0; i < element_count; ++i) {
    fprintf(stderr, "%u ", buffer[i]);
  }
  fprintf(stderr, "\n");

  err = hsa_amd_profiling_set_profiler_enabled(gpu_queue, 0);

  err = hsa_amd_memory_pool_free(buffer);

  err = hsa_signal_destroy(dispatch_signal);

  err = hsa_amd_memory_pool_free(kernarg_storage);

  err = hsa_queue_destroy(gpu_queue);

  err = hsa_executable_destroy(executable);
  err = hsa_code_object_reader_destroy(object_reader);

  err = hsa_shut_down();

  return 0;
}
