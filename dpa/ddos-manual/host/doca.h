#ifndef _DOCA_H_
#define _DOCA_H_

#include <doca_flow.h>

#include "dpdk.h"

doca_error_t doca_init(struct application_dpdk_config *app_dpdk_config);

#endif  /* _DOCA_H_ */