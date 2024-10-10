#ifndef _QUEUE_H_
#define _QUEUE_H_

#include <doca_error.h>
#include <infiniband/mlx5dv.h>
#include <libflexio/flexio.h>
#include <doca_dev.h>

#include "common.h"

struct app_config;
struct dpa_process_context;

doca_error_t allocate_sq(struct app_config *app_cfg, struct dpa_process_context * ctx);
doca_error_t allocate_rq(struct app_config *app_cfg, struct dpa_process_context * ctx);

void cq_destroy(struct flexio_process *flexio_process, struct flexio_cq *cq, struct app_transfer_cq cq_transf);
void sq_destroy(struct app_config *app_cfg, struct dpa_process_context * ctx);
void rq_destroy(struct app_config *app_cfg, struct dpa_process_context * ctx);
void dev_queues_destroy(struct app_config *app_cfg);

#endif  /* _QUEUE_H_ */