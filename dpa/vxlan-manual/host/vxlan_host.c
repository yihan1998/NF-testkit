#include <assert.h>
#include <malloc.h>
#include <signal.h>
#include <unistd.h>

#include <libflexio/flexio.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5_api.h>
#include <libflexio/flexio.h>
#include <bsd/string.h>

#include <doca_argp.h>
#include <doca_log.h>

#include "queue.h"
#include "dpdk.h"

#define MATCH_SIZE 64		  /* DR Matcher size */
#define PRINTF_BUFF_BSIZE (4 * 2048)

static bool force_quit; /* Set to true to terminate the application */
struct flexio_msg_stream *default_stream;

/* FlexIO application, generated by DPACC stub */
extern struct flexio_app *DEV_APP_NAME;

extern flexio_func_t vxlan_device_init;
extern flexio_func_t vxlan_device_event_handler;

void vxlan_steering_rules_destroy(struct app_config *app_cfg);
void vxlan_ibv_device_destroy(struct app_config *app_cfg);
void vxlan_destroy(struct app_config *app_cfg);
void vxlan_device_resources_destroy(struct app_config *app_cfg);
void vxlan_device_destroy(struct app_config *app_cfg);

void print_init(struct app_config *app_cfg) {
	flexio_msg_stream_attr_t attr;
	attr.uar = app_cfg->flexio_uar;
	attr.data_bsize = PRINTF_BUFF_BSIZE;
	attr.sync_mode = FLEXIO_LOG_DEV_SYNC_MODE_SYNC;
	attr.level = FLEXIO_MSG_DEV_INFO;
	attr.stream_name = "Default Stream";
	attr.mgmt_affinity.type = FLEXIO_AFFINITY_NONE;

	printf("Create default stream...\n");
	flexio_status ret = flexio_msg_stream_create(app_cfg->flexio_process, &attr, stdout, NULL, &default_stream);
	if (ret != FLEXIO_STATUS_SUCCESS) {
		printf("Failed to create msg stream\n");
		return;
	}
	
	return;
}

static void signal_handler(int signum)
{
	if (signum == SIGINT || signum == SIGTERM) {
		/* Add additional new lines for output readability */
		printf("\nSignal %d received, preparing to exit\n", signum);
		force_quit = true;
	}
}

static struct ibv_context *open_ibv_device(const char *device_name)
{
	struct ibv_device **dev_list;
	struct ibv_device *dev;
	struct ibv_context *ibv_ctx;
	int dev_idx;

	dev_list = ibv_get_device_list(NULL);
	if (dev_list == NULL) {
		printf("Failed to get device list\n");
		return NULL;
	}

	for (dev_idx = 0; dev_list[dev_idx] != NULL; ++dev_idx) {
		if (!strcmp(ibv_get_device_name(dev_list[dev_idx]), device_name)) {
			break;
        }
    }
	dev = dev_list[dev_idx];
	if (dev == NULL) {
		ibv_free_device_list(dev_list);
		printf("Device %s was not found\n", device_name);
		return NULL;
	}

	ibv_ctx = ibv_open_device(dev);
	if (ibv_ctx == NULL) {
		ibv_free_device_list(dev_list);
		printf("Failed to get device context [%s]\n", device_name);
		return NULL;
	}
	ibv_free_device_list(dev_list);
	return ibv_ctx;
}

int setup_ibv_device(struct app_config *app_cfg)
{
	app_cfg->ibv_ctx = open_ibv_device(app_cfg->device_name);
	if (app_cfg->ibv_ctx == NULL) {
		printf("Failed to open IBV device\n");
		return -1;
    }

	app_cfg->pd = ibv_alloc_pd(app_cfg->ibv_ctx);
	if (app_cfg->pd == NULL) {
		printf("Failed to allocate PD\n");
		ibv_close_device(app_cfg->ibv_ctx);
		return -1;
	}
	return 0;
}

#define SRC_MAC 		(0xa088c2320430)
#define DST_MAC_BASE 	(0xa099c231f700)

int setup_device(struct app_config *app_cfg)
{
	flexio_status result;
	struct flexio_event_handler_attr event_handler_attr = {0};
	event_handler_attr.host_stub_func = vxlan_device_event_handler;
	struct flexio_process_attr process_attr = { 0 };

	/* Create FlexIO Process */
	result = flexio_process_create(app_cfg->ibv_ctx, DEV_APP_NAME, &process_attr, &app_cfg->flexio_process);
	if (result != FLEXIO_STATUS_SUCCESS) {
		printf("Could not create FlexIO process (%d)", result);
		return -1;
	}

	app_cfg->flexio_uar = flexio_process_get_uar(app_cfg->flexio_process);

	for (int i = 0; i < app_cfg->nb_dpa_threads; i++) {
		/* Allocate new DPA context */
		struct dpa_process_context * ctx = (struct dpa_process_context *)calloc(1, sizeof(struct dpa_process_context));
		app_cfg->context[i] = ctx;

		ctx->mac_addr = (uint64_t)DST_MAC_BASE + i;

		event_handler_attr.affinity.type = FLEXIO_AFFINITY_STRICT;
		event_handler_attr.affinity.id = i;
		result = flexio_event_handler_create(app_cfg->flexio_process, &event_handler_attr, &ctx->event_handler);
		if (result != FLEXIO_STATUS_SUCCESS) {
			printf("Could not create event handler (%d)", result);
			return -1;
		}
	}

	return 0;
}

int allocate_device_resources(struct app_config *app_cfg)
{
	int result;
	flexio_status ret;

	for (int i = 0; i < app_cfg->nb_dpa_threads; i++) {
		struct dpa_process_context * ctx = app_cfg->context[i];

		result = allocate_sq(app_cfg, ctx);
		if (result < 0)
			return result;

		result = allocate_rq(app_cfg, ctx);
		if (result < 0)
			return result;
	}

	for (int i = 0; i < app_cfg->nb_dpa_threads; i++) {
		struct dpa_process_context * ctx = app_cfg->context[i];
		ctx->dev_data = (struct host2dev_processor_data *)calloc(1, sizeof(*ctx->dev_data));
		if (ctx->dev_data == NULL) {
			printf("Could not allocate application data memory\n");
			dev_queues_destroy(app_cfg);
			return -1;
		}

		ctx->dev_data->sq_cq_data = ctx->sq_cq_transf;
		ctx->dev_data->sq_data = ctx->sq_transf;
		ctx->dev_data->rq_cq_data = ctx->rq_cq_transf;
		ctx->dev_data->rq_data = ctx->rq_transf;
		ctx->dev_data->thread_index = i;

		ret = flexio_copy_from_host(app_cfg->flexio_process,
						ctx->dev_data,
						sizeof(*ctx->dev_data),
						&ctx->dev_data_daddr);
		if (ret != FLEXIO_STATUS_SUCCESS) {
			printf("Could not copy data to device\n");
			dev_queues_destroy(app_cfg);
			free(ctx->dev_data);
			return -1;
		}
	}

	return 0;
}

void dev_queues_destroy(struct app_config *app_cfg)
{
	for (int i = 0; i < app_cfg->nb_dpa_threads; i++) {
		struct dpa_process_context * ctx = app_cfg->context[i];
		rq_destroy(app_cfg, ctx);
		sq_destroy(app_cfg, ctx);
		cq_destroy(app_cfg->flexio_process, ctx->flexio_rq_cq_ptr, ctx->rq_cq_transf);
		cq_destroy(app_cfg->flexio_process, ctx->flexio_sq_cq_ptr, ctx->sq_cq_transf);
	}
}

int vxlan_run_device_process(struct app_config *app_cfg)
{
	int ret = 0;
	uint64_t rpc_ret_val;
	for (int i = 0; i < app_cfg->nb_dpa_threads; i++) {
		struct dpa_process_context * ctx = app_cfg->context[i];
		ret = flexio_process_call(app_cfg->flexio_process,
					&vxlan_device_init,
					&rpc_ret_val,
					ctx->dev_data_daddr);
		if (ret != FLEXIO_STATUS_SUCCESS) {
			printf("Failed to call init function on device\n");
			return -1;
		}
	}

	return 0;
}

static int create_flow_table(struct mlx5dv_dr_domain *domain,
				      int level,
				      int priority,
				      struct mlx5dv_flow_match_parameters *match_mask,
				      struct dr_flow_table **tbl_out)
{
	uint8_t criteria_enable = 0x1; /* Criteria enabled  */
	struct dr_flow_table *tbl;
	int result;

	tbl = calloc(1, sizeof(*tbl));
	if (tbl == NULL) {
		printf("Failed to allocate memory for dr table\n");
		return -1;
	}

	tbl->dr_table = mlx5dv_dr_table_create(domain, level);
	if (tbl->dr_table == NULL) {
		printf("Failed to create table [%d]\n", errno);
		result = -1;
		goto exit_with_error;
	}

	tbl->dr_matcher = mlx5dv_dr_matcher_create(tbl->dr_table, priority, criteria_enable, match_mask);
	if (tbl->dr_matcher == NULL) {
		printf("Failed to create matcher [%d]\n", errno);
		result = -1;
		goto exit_with_error;
	}

	*tbl_out = tbl;
	return 0;
exit_with_error:
	if (tbl->dr_matcher)
		mlx5dv_dr_matcher_destroy(tbl->dr_matcher);
	if (tbl->dr_table)
		mlx5dv_dr_table_destroy(tbl->dr_table);
	free(tbl);
	return result;
}

static void destroy_table(struct dr_flow_table *tbl)
{
	if (!tbl)
		return;
	if (tbl->dr_table)
		mlx5dv_dr_table_destroy(tbl->dr_table);
	if (tbl->dr_matcher)
		mlx5dv_dr_matcher_destroy(tbl->dr_matcher);
	free(tbl);
}

static void destroy_rule(struct dr_flow_rule *rule)
{
	if (!rule)
		return;
	if (rule->dr_action)
		mlx5dv_dr_action_destroy(rule->dr_action);
	if (rule->dr_rule)
		mlx5dv_dr_rule_destroy(rule->dr_rule);
	free(rule);
}

void mac_to_str(uint64_t mac, char *str) {
    // Extract each byte from the uint64_t MAC address
    uint8_t bytes[6];
    for (int i = 0; i < 6; i++) {
        bytes[5 - i] = (mac >> (i * 8)) & 0xFF;
    }

    // Format the bytes into a string
    sprintf(str, "%02x:%02x:%02x:%02x:%02x:%02x", bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]);
}

int create_rx_table(struct app_config *app_cfg)
{
	struct mlx5dv_flow_match_parameters *match_mask;
	size_t flow_match_size;
	int result;

	flow_match_size = sizeof(*match_mask) + MATCH_SIZE;
	match_mask = (struct mlx5dv_flow_match_parameters *)calloc(1, flow_match_size);
	if (match_mask == NULL) {
		printf("Failed to allocate match mask\n");
		return -1;
	}
	match_mask->match_sz = MATCH_SIZE;
	/* Fill match mask, match on all source mac bits */
	DEVX_SET(dr_match_spec, match_mask->match_buf, dmac_47_16, 0xffffffff);
	DEVX_SET(dr_match_spec, match_mask->match_buf, dmac_15_0, 0xffff);

	result = create_flow_table(app_cfg->rx_domain,
				   0, /* Table level */
				   0, /* Matcher priority */
				   match_mask,
				   &app_cfg->rx_flow_table);

	if (result < 0) {
		printf("Failed to create RX flow table\n");
		mlx5dv_dr_domain_destroy(app_cfg->rx_domain);
		free(match_mask);
		return result;
	}

	free(match_mask);
	return 0;
}

int create_steering_rule_rx(struct app_config *app_cfg, struct dpa_process_context * ctx)
{
	struct mlx5dv_flow_match_parameters *match_mask;
	struct mlx5dv_dr_action *actions[1];
	const int actions_len = 1;
	size_t flow_match_size;
	int result;
	char mac_str[18];  // MAC address string length is 17 characters + null terminator

    mac_to_str(ctx->mac_addr, mac_str);

	flow_match_size = sizeof(*match_mask) + MATCH_SIZE;
	match_mask = (struct mlx5dv_flow_match_parameters *)calloc(1, flow_match_size);
	if (match_mask == NULL) {
		printf("Failed to allocate match mask\n");
		return -1;
	}
	match_mask->match_sz = MATCH_SIZE;

	/* Create rule */
	ctx->rx_rule = calloc(1, sizeof(*ctx->rx_rule));
	if (ctx->rx_rule == NULL) {
		printf("Failed to allocate memory\n");
		result = -1;
		goto exit_with_error;
	}

	/* Action = forward to FlexIO RQ */
	ctx->rx_rule->dr_action = mlx5dv_dr_action_create_dest_devx_tir(flexio_rq_get_tir(ctx->flexio_rq_ptr));
	if (ctx->rx_rule->dr_action == NULL) {
		printf("Failed to create RX rule action [%d]\n", errno);
		result = -1;
		goto exit_with_error;
	}

	actions[0] = ctx->rx_rule->dr_action;
	printf("Match on DST MAC address: %s to queue %d\n", mac_str, flexio_cq_get_cq_num(ctx->flexio_rq_cq_ptr));
	/* Fill rule match, match on source mac address with this value */
	DEVX_SET(dr_match_spec, match_mask->match_buf, dmac_47_16, (ctx->mac_addr) >> 16);
	DEVX_SET(dr_match_spec, match_mask->match_buf, dmac_15_0, (ctx->mac_addr) % (1 << 16));
	ctx->rx_rule->dr_rule = mlx5dv_dr_rule_create(app_cfg->rx_flow_table->dr_matcher, match_mask, actions_len, actions);
	if (ctx->rx_rule->dr_rule == NULL) {
		printf("Failed to create RX rule [%d]\n", errno);
		result = -1;
		goto exit_with_error;
	}
	free(match_mask);
	return 0;

exit_with_error:
	free(match_mask);
	if (ctx->rx_rule) {
		destroy_rule(ctx->rx_rule);
		ctx->rx_rule = NULL;
	}
	destroy_table(app_cfg->rx_flow_table);
	app_cfg->rx_flow_table = NULL;
	mlx5dv_dr_domain_destroy(app_cfg->rx_domain);
	return result;
}

int vxlan_create_steering_rule_rx(struct app_config *app_cfg)
{
	int result;

	app_cfg->rx_domain = mlx5dv_dr_domain_create(app_cfg->ibv_ctx, MLX5DV_DR_DOMAIN_TYPE_NIC_RX);
	if (app_cfg->rx_domain == NULL) {
		printf("Failed to allocate RX domain [%d]\n", errno);
		return -1;
	}

	create_rx_table(app_cfg);
	
	for (int i = 0; i < app_cfg->nb_dpa_threads; i++) {
		struct dpa_process_context * ctx = app_cfg->context[i];
		// printf("Create RX steering rule for thread %d", i);
		result = create_steering_rule_rx(app_cfg, ctx);
		if (result < 0) {
			printf("Failed to create RX steering rule\n");
			return -1;
		}
	}

	return 0;
}

int create_tx_table(struct app_config *app_cfg)
{
	struct mlx5dv_flow_match_parameters *match_mask;
	size_t flow_match_size;
	int result;

	flow_match_size = sizeof(*match_mask) + MATCH_SIZE;
	match_mask = calloc(1, flow_match_size);
	if (match_mask == NULL) {
		printf("Failed to allocate match mask\n");
		return -1;
	}
	match_mask->match_sz = MATCH_SIZE;
	/* Fill match mask, match on all destination mac bits */
	DEVX_SET(dr_match_spec, match_mask->match_buf, smac_47_16, 0xffffffff);
	DEVX_SET(dr_match_spec, match_mask->match_buf, smac_15_0, 0xffff);

	result = create_flow_table(app_cfg->fdb_domain,
				   0, /* Table level */
				   0, /* Matcher priority */
				   match_mask,
				   &app_cfg->tx_flow_root_table);
	if (result < 0) {
		printf("Failed to create TX root flow table\n");
		mlx5dv_dr_domain_destroy(app_cfg->fdb_domain);
		free(match_mask);
		return result;
	}

	result = create_flow_table(app_cfg->fdb_domain,
				   1, /* Table level */
				   0, /* Matcher priority */
				   match_mask,
				   &app_cfg->tx_flow_table);
	if (result < 0) {
		printf("Failed to create Tx flow table\n");
		destroy_table(app_cfg->tx_flow_root_table);
		app_cfg->tx_flow_root_table = NULL;
		mlx5dv_dr_domain_destroy(app_cfg->fdb_domain);
		free(match_mask);
		return result;
	}

	free(match_mask);
	return 0;
}

int create_steering_rule_tx(struct app_config *app_cfg, struct dpa_process_context * ctx)
{
	struct mlx5dv_flow_match_parameters *match_mask;
	struct mlx5dv_dr_action *actions[1];
	size_t flow_match_size;
	int result;

	flow_match_size = sizeof(*match_mask) + MATCH_SIZE;
	match_mask = calloc(1, flow_match_size);
	if (match_mask == NULL) {
		printf("Failed to allocate match mask");
		return -1;
	}
	match_mask->match_sz = MATCH_SIZE;

	/* Jump to entry table rule */
	ctx->tx_root_rule = calloc(1, sizeof(*ctx->tx_root_rule));
	if (ctx->tx_root_rule == NULL) {
		printf("Failed to allocate memory");
		result = -1;
		goto exit_with_error;
	}

	ctx->tx_root_rule->dr_action = mlx5dv_dr_action_create_dest_table(app_cfg->tx_flow_table->dr_table);
	if (ctx->tx_root_rule->dr_action == NULL) {
		printf("Failed to create action jump to table");
		result = -1;
		goto exit_with_error;
	}

	actions[0] = ctx->tx_root_rule->dr_action;
	/* Fill rule match, match on destination mac address with this value */
	DEVX_SET(dr_match_spec, match_mask->match_buf, smac_47_16, (ctx->mac_addr) >> 16);
	DEVX_SET(dr_match_spec, match_mask->match_buf, smac_15_0, (ctx->mac_addr) % (1 << 16));
	ctx->tx_root_rule->dr_rule =
		mlx5dv_dr_rule_create(app_cfg->tx_flow_root_table->dr_matcher, match_mask, 1, actions);
	if (ctx->tx_root_rule->dr_rule == NULL) {
		printf("Failed to create rule jump to table");
		result = -1;
		goto exit_with_error;
	}

	/* Send to wire rule */
	ctx->tx_rule = calloc(1, sizeof(*ctx->tx_rule));
	if (ctx->tx_rule == NULL) {
		printf("Failed to allocate memory");
		result = -1;
		goto exit_with_error;
	}

	ctx->tx_rule->dr_action = mlx5dv_dr_action_create_dest_vport(app_cfg->fdb_domain, 0xFFFF);
	if (ctx->tx_rule->dr_action == NULL) {
		printf("Failed to create action dest vport\n");
		result = -1;
		goto exit_with_error;
	}

	actions[0] = ctx->tx_rule->dr_action;
	DEVX_SET(dr_match_spec, match_mask->match_buf, smac_47_16, (ctx->mac_addr) >> 16);
	DEVX_SET(dr_match_spec, match_mask->match_buf, smac_15_0, (ctx->mac_addr) % (1 << 16));
	ctx->tx_rule->dr_rule = mlx5dv_dr_rule_create(app_cfg->tx_flow_table->dr_matcher, match_mask, 1, actions);
	if (ctx->tx_rule->dr_rule == NULL) {
		printf("Failed to create rule dest vport\n");
		result = -1;
		goto exit_with_error;
	}

	free(match_mask);
	return 0;

exit_with_error:
	free(match_mask);
	if (ctx->tx_root_rule) {
		destroy_rule(ctx->tx_root_rule);
		ctx->tx_root_rule = NULL;
	}
	if (ctx->tx_rule) {
		destroy_rule(ctx->rx_rule);
		ctx->tx_rule = NULL;
	}
	destroy_table(app_cfg->tx_flow_root_table);
	app_cfg->tx_flow_root_table = NULL;
	destroy_table(app_cfg->tx_flow_table);
	app_cfg->tx_flow_table = NULL;
	mlx5dv_dr_domain_destroy(app_cfg->fdb_domain);
	return result;
}

int vxlan_create_steering_rule_tx(struct app_config *app_cfg)
{
	int result;

	app_cfg->fdb_domain = mlx5dv_dr_domain_create(app_cfg->ibv_ctx, MLX5DV_DR_DOMAIN_TYPE_FDB);
	if (app_cfg->fdb_domain == NULL) {
		printf("Failed to allocate FDB domain\n");
		return -1;
	}

	create_tx_table(app_cfg);

	for (int i = 0; i < app_cfg->nb_dpa_threads; i++) {
		struct dpa_process_context * ctx = app_cfg->context[i];
		result = create_steering_rule_tx(app_cfg, ctx);
		if (result < 0) {
			printf("Failed to create RX steering rule\n");
			return -1;
		}
	}

	return 0;
}

int vxlan_run_event_handler(struct app_config *app_cfg) {
	int ret = 0;
	for (int i = 0; i < app_cfg->nb_dpa_threads; i++) {
		struct dpa_process_context * ctx = app_cfg->context[i];
		ret = flexio_event_handler_run(ctx->event_handler, i);
		// ret = flexio_event_handler_run(ctx->event_handler, app_cfg->nb_dpa_threads);
		if (ret != FLEXIO_STATUS_SUCCESS) {
			printf("Failed to run event handler on device\n");
			return -1;
		}
	}

	return 0;
}

void vxlan_steering_rules_destroy(struct app_config *app_cfg)
{
	for (int i = 0; i < app_cfg->nb_dpa_threads; i++) {
		struct dpa_process_context * ctx = app_cfg->context[i];
		if (ctx->rx_rule) {
			destroy_rule(ctx->rx_rule);
			ctx->rx_rule = NULL;
		}
		if (ctx->tx_rule) {
			destroy_rule(ctx->tx_rule);
			ctx->tx_rule = NULL;
		}
		if (ctx->tx_root_rule) {
			destroy_rule(ctx->tx_root_rule);
			ctx->tx_root_rule = NULL;
		}
	}
	if (app_cfg->rx_domain)
		mlx5dv_dr_domain_destroy(app_cfg->rx_domain);
	if (app_cfg->fdb_domain)
		mlx5dv_dr_domain_destroy(app_cfg->fdb_domain);
	if (app_cfg->rx_flow_table) {
		destroy_table(app_cfg->rx_flow_table);
		app_cfg->rx_flow_table = NULL;
	}
	if (app_cfg->tx_flow_table) {
		destroy_table(app_cfg->tx_flow_table);
		app_cfg->tx_flow_table = NULL;
	}
	if (app_cfg->tx_flow_root_table) {
		destroy_table(app_cfg->tx_flow_root_table);
		app_cfg->tx_flow_root_table = NULL;
	}
}

void vxlan_ibv_device_destroy(struct app_config *app_cfg)
{
	ibv_dealloc_pd(app_cfg->pd);
}

void vxlan_destroy(struct app_config *app_cfg)
{
	// /* Destroy matcher and rule */
	vxlan_steering_rules_destroy(app_cfg);

	// /* Destroy WQs */
	vxlan_device_resources_destroy(app_cfg);

	/* Destroy FlexIO resources */
	vxlan_device_destroy(app_cfg);

	/* Destroy DevX and IBV resources */
	vxlan_ibv_device_destroy(app_cfg);

	if (flexio_process_destroy(app_cfg->flexio_process) != FLEXIO_STATUS_SUCCESS)
		printf("Failed to destroy FlexIO process");
	
	/* Close ib device */
	ibv_close_device(app_cfg->ibv_ctx);
}

void vxlan_device_resources_destroy(struct app_config *app_cfg)
{
	dev_queues_destroy(app_cfg);
	for (int i = 0; i < app_cfg->nb_dpa_threads; i++) {
		struct dpa_process_context * ctx = app_cfg->context[i];
		flexio_buf_dev_free(app_cfg->flexio_process, ctx->dev_data_daddr);
		free(ctx->dev_data);
	}
}

void vxlan_device_destroy(struct app_config *app_cfg)
{
	flexio_status ret = FLEXIO_STATUS_SUCCESS;

	for (int i = 0; i < app_cfg->nb_dpa_threads; i++) {
		struct dpa_process_context * ctx = app_cfg->context[i];
		ret |= flexio_event_handler_destroy(ctx->event_handler);
	}

	if (ret != FLEXIO_STATUS_SUCCESS) {
        printf("Failed to destroy FlexIO device");
    }
}

void ibv_device_destroy(struct app_config *app_cfg)
{
	ibv_dealloc_pd(app_cfg->pd);
}

int main(int argc, char ** argv)
{
	int result;
	struct app_config app_cfg = {
		.device_name = "mlx5_0",
		.nb_dpa_threads = 1,
	};

	printf("Init DPDK...\n");
	rte_eal_init(argc, argv);

	printf("Config DPDK...\n");
	config_ports();

	printf("Open IB device and allocate PD...\n");
    result = setup_ibv_device(&app_cfg);
	if (result < 0) {
		return -1;
	}

    printf("Create FlexIO Process and allocate memory...\n");
	/* Create FlexIO Process and allocate memory */
	result = setup_device(&app_cfg);
	if (result < 0) {
		goto ibv_device_cleanup;
    }

	/* Allocate device WQs, CQs and data */
	printf("Allocate device WQs, CQs and data...\n");
	result = allocate_device_resources(&app_cfg);
	if (result < 0)
		goto device_cleanup;

	/* Run init function on device */
	printf("Run init function on device...\n");
	result = vxlan_run_device_process(&app_cfg);
	if (result < 0) {
		printf("Failed to call init function on device\n");
		goto device_resources_cleanup;
	}

	/* Steering rule */
	printf("Create steering rule for RX...\n");
	result = vxlan_create_steering_rule_rx(&app_cfg);
	if (result < 0) {
		printf("Failed to create RX steering rule\n");
		goto device_resources_cleanup;
	}

	printf("Create steering rule for TX...\n");
	result = vxlan_create_steering_rule_tx(&app_cfg);
	if (result < 0) {
		printf("Failed to create TX steering rule\n");
		goto rule_cleanup;
	}

	printf("Run VXLAN handler...\n");
	result = vxlan_run_event_handler(&app_cfg);
	if (result < 0) {
		printf("Failed to run event handler on device\n");
		goto rule_cleanup;
	}

	print_init(&app_cfg);

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);
	printf("Flexio reflector Started\n");
	/* Add an additional new line for output readability */
	printf("Press Ctrl+C to terminate\n");
	while (!force_quit) {
        flexio_msg_stream_flush(default_stream);
		sleep(1);
		run_dpdk_loop();
	}

	vxlan_destroy(&app_cfg);
	return EXIT_SUCCESS;

rule_cleanup:
	vxlan_steering_rules_destroy(&app_cfg);
device_resources_cleanup:
	vxlan_device_resources_destroy(&app_cfg);
device_cleanup:
	vxlan_device_destroy(&app_cfg);
ibv_device_cleanup:
	ibv_device_destroy(&app_cfg);
	return 0;
}