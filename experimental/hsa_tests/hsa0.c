// info queries

#include <memory.h>
#include <stdio.h>
#include <string.h>

#include "third_party/hsa-runtime-headers/include/hsa/hsa.h"
#include "third_party/hsa-runtime-headers/include/hsa/hsa_ext_amd.h"

typedef struct {
  uint32_t agent_count;
  hsa_agent_t all_agents[32];
  hsa_agent_t cpu_agent;
  hsa_agent_t gpu_agent;
} agents_t;

static hsa_status_t iterate_agent(hsa_agent_t agent, void* user_data) {
  agents_t* agents = (agents_t*)user_data;
  hsa_status_t err;

  char product_name[64];
  err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME, product_name);

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

static hsa_status_t pcs_config_callback(
    const hsa_ven_amd_pcs_configuration_t* configuration, void* callback_data) {
  fprintf(stderr, "HAVE CONFIG\n");
  return HSA_STATUS_SUCCESS;
}

int main(int argc, char** argv) {
  hsa_status_t err;

  err = hsa_init();

  uint16_t version_major = 0;
  err = hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &version_major);
  uint16_t version_minor = 0;
  err = hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &version_minor);

  uint64_t timestamp_frequency = 0;
  err = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                            &timestamp_frequency);
  uint64_t max_wait = 0;
  err = hsa_system_get_info(HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT, &max_wait);
  hsa_endianness_t endianness = HSA_ENDIANNESS_LITTLE;
  err = hsa_system_get_info(HSA_SYSTEM_INFO_ENDIANNESS, &endianness);
  hsa_machine_model_t machine_model = HSA_MACHINE_MODEL_SMALL;
  err = hsa_system_get_info(HSA_SYSTEM_INFO_MACHINE_MODEL, &machine_model);

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

  //
  char gpu_name[64];
  err = hsa_agent_get_info(agents.gpu_agent, HSA_AGENT_INFO_NAME, gpu_name);
  uint8_t gpu_extensions[128] = {0};
  err = hsa_agent_get_info(agents.gpu_agent, HSA_AGENT_INFO_EXTENSIONS,
                           gpu_extensions);
  bool has_amd_profiler =
      gpu_extensions[64] & (HSA_EXTENSION_AMD_PROFILER - 64);
  if (has_amd_profiler) fprintf(stderr, "HSA_EXTENSION_AMD_PROFILER\n");
  bool has_amd_loader = gpu_extensions[64] & (HSA_EXTENSION_AMD_LOADER - 64);
  if (has_amd_loader) fprintf(stderr, "HSA_EXTENSION_AMD_LOADER\n");
  bool has_amd_aqlprofile =
      gpu_extensions[64] & (HSA_EXTENSION_AMD_AQLPROFILE - 64);
  if (has_amd_aqlprofile) fprintf(stderr, "HSA_EXTENSION_AMD_AQLPROFILE\n");
  bool has_amd_pc_sampling =
      gpu_extensions[64] & (HSA_EXTENSION_AMD_PC_SAMPLING - 64);
  if (has_amd_pc_sampling) fprintf(stderr, "HSA_EXTENSION_AMD_PC_SAMPLING\n");
  char cpu_name[64];
  err = hsa_agent_get_info(agents.cpu_agent, HSA_AGENT_INFO_NAME, cpu_name);
  uint32_t cpu_numa_node = 0;
  err =
      hsa_agent_get_info(agents.cpu_agent, HSA_AGENT_INFO_NODE, &cpu_numa_node);

  memory_pools_t gpu_memory_pools;
  memset(&gpu_memory_pools, 0, sizeof(gpu_memory_pools));
  err = hsa_amd_agent_iterate_memory_pools(
      agents.gpu_agent, iterate_memory_pool, &gpu_memory_pools);
  memory_pools_t cpu_memory_pools;
  memset(&cpu_memory_pools, 0, sizeof(cpu_memory_pools));
  err = hsa_amd_agent_iterate_memory_pools(
      agents.cpu_agent, iterate_memory_pool, &cpu_memory_pools);

  for (uint32_t i = 0; i < gpu_memory_pools.count; ++i) {
    hsa_amd_memory_pool_access_t access =
        HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
    err = hsa_amd_agent_memory_pool_get_info(
        agents.cpu_agent, gpu_memory_pools.pools[i],
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
    uint32_t num_link_hops = 0;
    err = hsa_amd_agent_memory_pool_get_info(
        agents.cpu_agent, gpu_memory_pools.pools[i],
        HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS, &num_link_hops);
    hsa_amd_memory_pool_link_info_t link_infos[32];
    err = hsa_amd_agent_memory_pool_get_info(
        agents.cpu_agent, gpu_memory_pools.pools[i],
        HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO, link_infos);
    err = err;
  }

  for (uint32_t i = 0; i < gpu_memory_pools.count; ++i) {
    hsa_amd_memory_pool_access_t access =
        HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
    err = hsa_amd_agent_memory_pool_get_info(
        agents.gpu_agent, cpu_memory_pools.pools[i],
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
    uint32_t num_link_hops = 0;
    err = hsa_amd_agent_memory_pool_get_info(
        agents.gpu_agent, cpu_memory_pools.pools[i],
        HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS, &num_link_hops);
    hsa_amd_memory_pool_link_info_t link_infos[32];
    err = hsa_amd_agent_memory_pool_get_info(
        agents.gpu_agent, cpu_memory_pools.pools[i],
        HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO, link_infos);
    err = err;
  }

  hsa_ven_amd_pc_sampling_1_00_pfn_t pcs_table;
  err = hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_PC_SAMPLING, 1,
                                             sizeof(pcs_table), &pcs_table);
  // err = pcs_table.hsa_ven_amd_pcs_iterate_configuration(
  //     agents.gpu_agent, pcs_config_callback, NULL);

  err = hsa_shut_down();

  return 0;
}
