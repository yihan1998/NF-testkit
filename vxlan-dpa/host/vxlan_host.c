#include <assert.h>
#include <malloc.h>
#include <signal.h>
#include <unistd.h>

#include <libflexio/flexio.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5_api.h>
#include <libflexio/flexio.h>

#include <doca_argp.h>
#include <doca_log.h>

#include "queue.h"
#include "vxlan_host.h"

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

int setup_device(struct app_config *app_cfg)
{
	flexio_status result;
	struct flexio_event_handler_attr event_handler_attr = {0};
	event_handler_attr.host_stub_func = dns_filter_device_event_handler;
	struct flexio_process_attr process_attr = { 0 };

	/* Create FlexIO Process */
	result = flexio_process_create(app_cfg->ibv_ctx, dns_filter_device, &process_attr, &app_cfg->flexio_process);
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
		ctx->dev_data = (struct dns_filter_data *)calloc(1, sizeof(*ctx->dev_data));
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
		dns_filter_rq_destroy(app_cfg, ctx);
		dns_filter_sq_destroy(app_cfg, ctx);
		dns_filter_cq_destroy(app_cfg->flexio_process, ctx->flexio_rq_cq_ptr, ctx->rq_cq_transf);
		dns_filter_cq_destroy(app_cfg->flexio_process, ctx->flexio_sq_cq_ptr, ctx->sq_cq_transf);
	}
}

int run_device_process(struct dns_filter_config *app_cfg)
{
	int ret = 0;
	uint64_t rpc_ret_val;
	for (int i = 0; i < app_cfg->nb_dpa_threads; i++) {
		struct dpa_process_context * ctx = app_cfg->context[i];
		ret = flexio_process_call(app_cfg->flexio_process,
					&dns_filter_device_init,
					&rpc_ret_val,
					ctx->dev_data_daddr);
		if (ret != FLEXIO_STATUS_SUCCESS) {
			printf("Failed to call init function on device\n");
			return -1;
		}
	}

	return 0;
}

void device_destroy(struct app_config *app_cfg)
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

void dns_filter_ibv_device_destroy(struct app_config *app_cfg)
{
	ibv_dealloc_pd(app_cfg->pd);
}

int main(int argc, char ** argv)
{
	struct app_config app_cfg = {0};

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
	result = run_device_process(&app_cfg);
	if (result != DOCA_SUCCESS) {
		printf("Failed to call init function on device\n");
		goto device_resources_cleanup;
	}

// rule_cleanup:
// 	steering_rules_destroy(&app_cfg);
device_resources_cleanup:
	device_resources_destroy(&app_cfg);
device_cleanup:
	device_destroy(&app_cfg);
ibv_device_cleanup:
	ibv_device_destroy(&app_cfg);
	return 0;
}