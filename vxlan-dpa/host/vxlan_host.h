#ifndef _VXLAN_HOST_
#define _VXLAN_HOST_

struct dpa_process_context {
	struct dns_filter_data *dev_data;		/* device data */

	struct flexio_event_handler *event_handler; /* Event handler on device */

	struct flexio_mkey *rqd_mkey;
	struct app_transfer_wq rq_transf;

	struct flexio_mkey *sqd_mkey;
	struct app_transfer_wq sq_transf;

	struct flexio_cq *flexio_rq_cq_ptr; /* FlexIO RQ CQ */
	struct flexio_cq *flexio_sq_cq_ptr; /* FlexIO SQ CQ */
	struct flexio_rq *flexio_rq_ptr;    /* FlexIO RQ */
	struct flexio_sq *flexio_sq_ptr;    /* FlexIO SQ */

	/* FlexIO resources */
	flexio_uintptr_t dev_data_daddr;	    /* Data address accessible by the device */
	struct app_transfer_cq rq_cq_transf;
	struct app_transfer_cq sq_cq_transf;

	struct dr_flow_rule *rx_rule;
	struct dr_flow_rule *tx_rule;
	struct dr_flow_rule *tx_root_rule;
};

/* Reflector configuration structure */
struct app_config {
	char device_name[DOCA_DEVINFO_IBDEV_NAME_SIZE]; /* IB device name */

	/* IB Verbs resources */
	struct ibv_context *ibv_ctx; /* IB device context */
	struct ibv_pd *pd;	     /* Protection domain */

	int nb_dpa_threads;

	struct flexio_process *flexio_process;	    /* FlexIO process */
	struct flexio_uar *flexio_uar;		    /* FlexIO UAR */

	struct dpa_process_context * context[MAX_NB_THREAD];

	/* mlx5dv direct rules resources, used for steering rules */
	struct mlx5dv_dr_domain *rx_domain;
	struct mlx5dv_dr_domain *fdb_domain;

	struct dr_flow_table *rx_flow_table;
	struct dr_flow_table *tx_flow_table;
	struct dr_flow_table *tx_flow_root_table;
};

void dev_queues_destroy(struct app_config *app_cfg);

#endif	/* _VXLAN_HOST_ */
