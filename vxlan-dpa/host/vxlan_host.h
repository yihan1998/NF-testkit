#ifndef _VXLAN_HOST_
#define _VXLAN_HOST_

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