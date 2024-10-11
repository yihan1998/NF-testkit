#ifndef _QUEUE_H_
#define _QUEUE_H_

#include <doca_error.h>
#include <infiniband/mlx5dv.h>
#include <libflexio/flexio.h>
#include <doca_dev.h>

#include "common.h"

/* Reflector configuration structure */
struct app_config {
	char device_name[DOCA_DEVINFO_IBDEV_NAME_SIZE]; /* IB device name */

	/* IB Verbs resources */
	struct ibv_context *ibv_ctx; /* IB device context */
	struct ibv_mr *mr;			/* Memory region */
	struct ibv_pd *pd;	     /* Protection domain */

	int nb_dpa_threads;

	struct flexio_process *flexio_process;	    /* FlexIO process */
	struct flexio_uar *flexio_uar;		    /* FlexIO UAR */
	struct flexio_window *flexio_window;	/* FlexIO window */

	struct dpa_process_context * context[MAX_NB_THREAD];

	/* mlx5dv direct rules resources, used for steering rules */
	struct mlx5dv_dr_domain *rx_domain;
	struct mlx5dv_dr_domain *fdb_domain;

	struct dr_flow_table *rx_flow_table;
	struct dr_flow_table *tx_flow_table;
	struct dr_flow_table *tx_flow_root_table;
};

int allocate_sq(struct app_config *app_cfg, struct dpa_process_context * ctx);
int allocate_rq(struct app_config *app_cfg, struct dpa_process_context * ctx);

void cq_destroy(struct flexio_process *flexio_process, struct flexio_cq *cq, struct app_transfer_cq cq_transf);
void sq_destroy(struct app_config *app_cfg, struct dpa_process_context * ctx);
void rq_destroy(struct app_config *app_cfg, struct dpa_process_context * ctx);
void dev_queues_destroy(struct app_config *app_cfg);

#endif  /* _QUEUE_H_ */