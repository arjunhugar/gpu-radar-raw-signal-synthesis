#include "../../include/GPUCodingInterface.h"
#include "../common/export_symbols.h"
#include <cstdio>

LIBAPI
int
clusterInit(
    user_ClusterType sensor_type,
    int n_sensors_per_cluster,
    int* global_sensor_index
)
{
  std::printf("[GPU] clusterInit() entered, n_sensors_per_cluster=%d\n",
              n_sensors_per_cluster);
  std::fflush(stdout);
  return 0;
}

LIBAPI
int
clusterCleanup(void)
{
  return 0;
}

#include <type_traits>
static_assert(std::is_same<decltype(&clusterInit), user_ClusterInit_Fct>::value,
              "Signature of clusterInit differs from expected signature of user_ClusterInit_Fct");
static_assert(std::is_same<decltype(&clusterCleanup), user_ClusterCleanup_Fct>::value,
              "Signature of clusterCleanup differs from expected signature of user_ClusterCleanup_Fct");
