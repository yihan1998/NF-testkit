#include <string.h>
#include <unistd.h>

#include <doca_log.h>
#include <doca_flow.h>

#include "dpdk.h"
#include "doca.h"
#include "flow_common.h"

#define MAX_RSS_QUEUES  16

struct doca_flow_pipe *classifier_pipe[2];
struct doca_flow_pipe *rss_pipe[2];
struct doca_flow_pipe *monitor_pipe[2];
struct doca_flow_pipe_entry *match_entry[2];

struct doca_sha_config doca_sha_cfg = {
	.pci_address = "03:00.0",
};

struct worker_context worker_ctx[NR_CPUS];
__thread struct worker_context * ctx;

doca_error_t open_doca_device_with_pci(const char *pci_addr, tasks_check func, struct doca_dev **retval)
{
	struct doca_devinfo **dev_list;
	uint32_t nb_devs;
	uint8_t is_addr_equal = 0;
	int res;
	size_t i;

	/* Set default return value */
	*retval = NULL;

	res = doca_devinfo_create_list(&dev_list, &nb_devs);
	if (res != DOCA_SUCCESS) {
		printf("Failed to load doca devices list: %s\n", doca_error_get_descr(res));
		return res;
	}

	/* Search */
	for (i = 0; i < nb_devs; i++) {
		res = doca_devinfo_is_equal_pci_addr(dev_list[i], pci_addr, &is_addr_equal);
		if (res == DOCA_SUCCESS && is_addr_equal) {
			/* If any special capabilities are needed */
			if (func != NULL && func(dev_list[i]) != DOCA_SUCCESS)
				continue;

			/* if device can be opened */
			res = doca_dev_open(dev_list[i], retval);
			if (res == DOCA_SUCCESS) {
				doca_devinfo_destroy_list(dev_list);
				return res;
			}
		}
	}

	printf("Matching device not found\n");
	res = DOCA_ERROR_NOT_FOUND;

	doca_devinfo_destroy_list(dev_list);
	return res;
}

doca_error_t doca_sha_init(void) {
	doca_error_t result;

	/* Open DOCA device */
	result = open_doca_device_with_pci(doca_sha_cfg.pci_address, NULL, &doca_sha_cfg.dev);
	if (result != DOCA_SUCCESS) {
		printf("No device matching PCI address found. Reason: %s", doca_error_get_descr(result));
		return result;
	}

	/* Create a DOCA RegEx instance */
	result = doca_sha_create(&(doca_sha_cfg.doca_sha));
	if (result != DOCA_SUCCESS) {
		printf("DOCA SHA creation Failed. Reason: %s", doca_error_get_descr(result));
		doca_dev_close(doca_sha_cfg.dev);
		return DOCA_ERROR_INITIALIZATION;
	}

	/* Start DOCA RegEx */
	result = doca_ctx_start(doca_sha_as_ctx(doca_sha_cfg.doca_sha));
	if (result != DOCA_SUCCESS) {
		printf("Unable to start DOCA RegEx. Reason: %s", doca_error_get_descr(result));
		result = DOCA_ERROR_INITIALIZATION;
		return 0;
	}

	return DOCA_SUCCESS;
}

doca_error_t doca_sha_percore_init(struct doca_sha_ctx * sha_ctx) {
	doca_error_t result;
	char * src_data_buffer, * dst_data_buffer;

	src_data_buffer = (char *)calloc(512, sizeof(char));
	dst_data_buffer = (char *)calloc(512, sizeof(char));

    sha_ctx->src_data_buffer = src_data_buffer;
    sha_ctx->src_data_buffer_len = 512;

	sha_ctx->dst_data_buffer = dst_data_buffer;
    sha_ctx->dst_data_buffer_len = 512;

	result = doca_buf_inventory_create(2, &sha_ctx->buf_inv);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create doca_buf_inventory. Reason: %s\n", doca_error_get_descr(result));
		return 0;
	}

	result = doca_buf_inventory_start(sha_ctx->buf_inv);
	if (result != DOCA_SUCCESS) {
		printf("Unable to start doca_buf_inventory. Reason: %s\n", doca_error_get_descr(result));
		return 0;
	}

	result = doca_mmap_create(&sha_ctx->src_mmap);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create doca_mmap. Reason: %s\n", doca_error_get_descr(result));
		return 0;
	}

	result = doca_mmap_create(&sha_ctx->dst_mmap);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create doca_mmap. Reason: %s\n", doca_error_get_descr(result));
		return 0;
	}

	result = doca_mmap_add_dev(sha_ctx->src_mmap, sha_ctx->dev);
	if (result != DOCA_SUCCESS) {
		printf("Unable to add device to doca_mmap. Reason: %s\n", doca_error_get_descr(result));
		return 0;
	}

	result = doca_mmap_add_dev(sha_ctx->dst_mmap, sha_ctx->dev);
	if (result != DOCA_SUCCESS) {
		printf("Unable to add device to doca_mmap. Reason: %s\n", doca_error_get_descr(result));
		return 0;
	}

	result = doca_mmap_set_memrange(sha_ctx->src_mmap, src_data_buffer, 512);
	if (result != DOCA_SUCCESS) {
		printf("Unable to register src memory with doca_mmap. Reason: %s\n", doca_error_get_descr(result));
		return 0;
	}

	result = doca_mmap_set_memrange(sha_ctx->dst_mmap, dst_data_buffer, 512);
	if (result != DOCA_SUCCESS) {
		printf("Unable to register src memory with doca_mmap. Reason: %s\n", doca_error_get_descr(result));
		return 0;
	}

	// result = doca_mmap_set_memrange(sha_ctx->mmap, sha_ctx->dst_data_buffer, sha_ctx->dst_data_buffer_len);
	// if (result != DOCA_SUCCESS) {
	// 	printf("Unable to register dest memory with doca_mmap. Reason: %s\n", doca_error_get_descr(result));
	// 	return 0;
	// }

	result = doca_mmap_start(sha_ctx->src_mmap);
	if (result != DOCA_SUCCESS) {
		printf("Unable to start doca_mmap. Reason: %s\n", doca_error_get_descr(result));
		return 0;
	}

	result = doca_mmap_start(sha_ctx->dst_mmap);
	if (result != DOCA_SUCCESS) {
		printf("Unable to start doca_mmap. Reason: %s\n", doca_error_get_descr(result));
		return 0;
	}

	if (doca_buf_inventory_buf_get_by_addr(sha_ctx->buf_inv, sha_ctx->src_mmap, sha_ctx->src_data_buffer, sha_ctx->src_data_buffer_len, &sha_ctx->src_buf) != DOCA_SUCCESS) {
        printf("Failed to create inventory buf!\n");
        return 0;
    }

	if (doca_buf_inventory_buf_get_by_addr(sha_ctx->buf_inv, sha_ctx->dst_mmap, sha_ctx->dst_data_buffer, sha_ctx->dst_data_buffer_len, &sha_ctx->dst_buf) != DOCA_SUCCESS) {
        printf("Failed to create inventory buf!\n");
        return 0;
    }

    return DOCA_SUCCESS;
}

int doca_worker_percore_init(void) {
    ctx = &worker_ctx[core_id];
	doca_sha_percore_init(&ctx->sha_ctx);
    return 0;
};

int doca_worker_init(struct worker_context * ctx) {
	doca_error_t result;

	result = doca_pe_create(&ctx->pe);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create progress engine: %s\n", doca_error_get_descr(result));
		return result;
	}

	ctx->sha_ctx.dev = doca_sha_cfg.dev;
	ctx->sha_ctx.doca_sha = doca_sha_cfg.doca_sha;

    /* Add workq to RegEx */
	result = doca_pe_connect_ctx(ctx->pe, doca_sha_as_ctx(ctx->sha_ctx.doca_sha));
	if (result != DOCA_SUCCESS) {
		printf("Failed to connect progress engine to context: %s\n", doca_error_get_descr(result));
		return result;
	}

    return DOCA_SUCCESS;
}

static doca_error_t create_classifier_pipe(struct doca_flow_port *port, struct doca_flow_pipe **pipe)
{
	struct doca_flow_pipe_cfg *pipe_cfg;
	doca_error_t result;

	result = doca_flow_pipe_cfg_create(&pipe_cfg, port);
	if (result != DOCA_SUCCESS) {
		printf("Failed to create doca_flow_pipe_cfg: %s\n", doca_error_get_descr(result));
		return result;
	}

	result = set_flow_pipe_cfg(pipe_cfg, "CLASSIFIER_PIPE", DOCA_FLOW_PIPE_CONTROL, true);
	if (result != DOCA_SUCCESS) {
		printf("Failed to set doca_flow_pipe_cfg: %s\n", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	result = doca_flow_pipe_create(pipe_cfg, NULL, NULL, pipe);
destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	return result;
}

static doca_error_t add_classifier_pipe_entry(struct doca_flow_port *port, int port_id, struct doca_flow_pipe *pipe)
{
    struct doca_flow_match match;
	struct doca_flow_fwd fwd;
	uint8_t priority = 0;
	doca_error_t result;
	struct entries_status status;

	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));
	memset(&status, 0, sizeof(status));

	match.parser_meta.outer_l4_type = DOCA_FLOW_L4_META_TCP;
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = monitor_pipe[port_id];

	result = doca_flow_pipe_control_add_entry(0, priority, pipe,
                &match, NULL, NULL, NULL, NULL, NULL, NULL, &fwd, &status, NULL);
	if (result != DOCA_SUCCESS) {
		printf("Failed to add control pipe entry: %s\n", doca_error_get_descr(result));
		return result;
	}

    result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 1);
    if (result != DOCA_SUCCESS) {
        printf("[%s:%d] Failed to process entries: %s\n", __func__, __LINE__, doca_error_get_descr(result));
        doca_flow_destroy();
        return result;
    }

	return DOCA_SUCCESS;
}

static doca_error_t create_rss_pipe(struct doca_flow_port *port, int port_id, struct doca_flow_pipe **pipe, uint32_t nb_rss_queues)
{
	struct doca_flow_pipe_cfg *pipe_cfg;
	struct doca_flow_match match;
	struct doca_flow_fwd fwd;
	struct entries_status status;
	uint16_t rss_queues[MAX_RSS_QUEUES];
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));
	memset(&status, 0, sizeof(status));

	result = doca_flow_pipe_cfg_create(&pipe_cfg, port);
	if (result != DOCA_SUCCESS) {
		printf("Failed to create doca_flow_pipe_cfg: %s\n", doca_error_get_descr(result));
		return result;
	}

	result = set_flow_pipe_cfg(pipe_cfg, "RSS_PIPE", DOCA_FLOW_PIPE_BASIC, false);
	if (result != DOCA_SUCCESS) {
		printf("Failed to set doca_flow_pipe_cfg: %s\n", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);
	if (result != DOCA_SUCCESS) {
		printf("Failed to set doca_flow_pipe_cfg match: %s\n", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

    for (int i = 0; i < nb_rss_queues; ++i) {
		rss_queues[i] = i;
    }

	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_queues = rss_queues;
	fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP | DOCA_FLOW_RSS_UDP;
	fwd.num_of_queues = nb_rss_queues;

	result = doca_flow_pipe_create(pipe_cfg, &fwd, NULL, pipe);
	if (result != DOCA_SUCCESS) {
		printf("[%s:%d] Failed to create doca flow pipe, err: %s\n", __func__, __LINE__, doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_pipe_add_entry(0, *pipe, &match, NULL, NULL, NULL, 0, &status, NULL);
	if (result != DOCA_SUCCESS) {
		printf("Failed to add count pipe entry: %s\n", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 0);
	if (result != DOCA_SUCCESS) {
		printf("Failed to process count entry: %s\n", doca_error_get_descr(result));
	}

	return result;
destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	return result;
}

static doca_error_t create_monitor_pipe(struct doca_flow_port *port, int port_id, struct doca_flow_pipe **pipe)
{
    struct doca_flow_match match;
	struct doca_flow_actions actions, *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_fwd fwd = {.type = DOCA_FLOW_FWD_CHANGEABLE};
	struct doca_flow_monitor counter;
	struct doca_flow_pipe_cfg *pipe_cfg;
	// uint16_t rss_queues[MAX_RSS_QUEUES];
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&counter, 0, sizeof(counter));

	/* 5 tuple match */
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
	// match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	// match.outer.ip4.src_ip = 0xffffffff;
	// match.outer.ip4.dst_ip = 0xffffffff;
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TCP;
	// match.outer.transport.src_port = 0xffff;
	match.outer.transport.dst_port = 0xffff;

	/* set meta data to match on the egress domain */
	actions.meta.pkt_meta = UINT32_MAX;
	actions_arr[0] = &actions;

	counter.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

	result = doca_flow_pipe_cfg_create(&pipe_cfg, port);
	if (result != DOCA_SUCCESS) {
		printf("Failed to create doca_flow_pipe_cfg: %s\n", doca_error_get_descr(result));
		return result;
	}

	result = set_flow_pipe_cfg(pipe_cfg, "MONITOR_PIPE", DOCA_FLOW_PIPE_BASIC, false);
	if (result != DOCA_SUCCESS) {
		printf("Failed to set doca_flow_pipe_cfg: %s\n", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);
	if (result != DOCA_SUCCESS) {
		printf("Failed to set doca_flow_pipe_cfg match: %s\n", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_actions(pipe_cfg, actions_arr, NULL, NULL, NB_ACTIONS_ARR);
	if (result != DOCA_SUCCESS) {
		printf("Failed to set doca_flow_pipe_cfg monitor: %s\n", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &counter);
	if (result != DOCA_SUCCESS) {
		printf("Failed to set doca_flow_pipe_cfg counter: %s\n", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

    // for (int i = 0; i < nb_rss_queues; ++i) {
	// 	rss_queues[i] = i;
    // }

	// fwd.type = DOCA_FLOW_FWD_RSS;
	// fwd.rss_queues = rss_queues;
	// fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP | DOCA_FLOW_RSS_UDP;
	// fwd.num_of_queues = nb_rss_queues;

	result = doca_flow_pipe_create(pipe_cfg, &fwd, NULL, pipe);
	if (result != DOCA_SUCCESS) {
		printf("[%s:%d] Failed to create doca flow pipe, err: %s\n", __func__, __LINE__, doca_error_get_descr(result));
		return result;
	}

destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	return result;
}

static doca_error_t add_monitor_pipe_entry(struct doca_flow_pipe *pipe, int port_id, struct entries_status *status, struct doca_flow_pipe_entry **entry)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_monitor monitor;
	struct doca_flow_fwd fwd;
	doca_error_t result;
	doca_be16_t dst_port;

	dst_port = rte_cpu_to_be_16(8080);

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));

	match.outer.transport.dst_port = dst_port;

	monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

	actions.meta.pkt_meta = 4;
	actions.action_idx = 0;

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = rss_pipe[port_id];

	result = doca_flow_pipe_add_entry(0, pipe, &match, &actions, &monitor, &fwd, 0, status, entry);
	if (result != DOCA_SUCCESS) {
		printf("[%s:%d] Failed to add entry to pipe: %s", __func__, __LINE__, doca_error_get_descr(result));
		return result;
	}

	dst_port = rte_cpu_to_be_16(1234);

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));

	match.outer.transport.dst_port = dst_port;

	monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

	actions.meta.pkt_meta = 1;
	actions.action_idx = 0;

	fwd.type = DOCA_FLOW_FWD_PORT;
	fwd.port_id = port_id ^ 1;

	result = doca_flow_pipe_add_entry(0, pipe, &match, &actions, &monitor, &fwd, 0, status, entry);
	if (result != DOCA_SUCCESS) {
		printf("[%s:%d] Failed to add entry to pipe: %s", __func__, __LINE__, doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}

doca_error_t doca_init(struct application_dpdk_config *app_dpdk_config)
{
	int nb_ports = app_dpdk_config->port_config.nb_ports;
    int nb_queues = app_dpdk_config->port_config.nb_queues;
	int nb_cores = rte_lcore_count();
	struct flow_resources resource = {.nr_counters = 64};
	uint32_t nr_shared_resources[SHARED_RESOURCE_NUM_VALUES] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_dev *dev_arr[nb_ports];
	struct entries_status status_ingress;
	int num_of_entries_ingress = 2;
	doca_error_t result;
	int port_id;

    printf("Initializing doca flow...\n");

	result = init_doca_flow(nb_queues, "vnf,hws", &resource, nr_shared_resources);
	if (result != DOCA_SUCCESS) {
		printf("Failed to init DOCA Flow: %s\n", doca_error_get_descr(result));
		return result;
	}

    printf("Initializing doca flow ports...\n");

	memset(dev_arr, 0, sizeof(struct doca_dev *) * nb_ports);
	result = init_doca_flow_ports(nb_ports, ports, true, dev_arr);
	if (result != DOCA_SUCCESS) {
		printf("Failed to init DOCA ports: %s\n", doca_error_get_descr(result));
		doca_flow_destroy();
		return result;
	}

    printf("Initializing each port...\n");

	for (port_id = 0; port_id < nb_ports; port_id++) {
		memset(&status_ingress, 0, sizeof(status_ingress));

        printf("Creating rss pipe on port %d...\n", port_id);

		result = create_rss_pipe(ports[port_id], port_id, &rss_pipe[port_id], nb_queues);
		if (result != DOCA_SUCCESS) {
			printf("Failed to create rss pipe: %s\n", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

        printf("Creating monitor pipe on port %d...\n", port_id);

        result = create_monitor_pipe(ports[port_id], port_id, &monitor_pipe[port_id]);
		if (result != DOCA_SUCCESS) {
			printf("Failed to create monitor pipe: %s\n", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

        printf("Adding entry to monitor pipe on port %d...\n", port_id);

		result = add_monitor_pipe_entry(monitor_pipe[port_id], port_id, &status_ingress, &match_entry[port_id]);
		if (result != DOCA_SUCCESS) {
			printf("Failed to add entry to monitor pipe: %s\n", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

        printf("Creating classifier pipe on port %d...\n", port_id);

		result = create_classifier_pipe(ports[port_id], &classifier_pipe[port_id]);
		if (result != DOCA_SUCCESS) {
			printf("Failed to create classifier pipe: %s\n", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

        result = add_classifier_pipe_entry(ports[port_id], port_id, classifier_pipe[port_id]);
        if (result != DOCA_SUCCESS) {
			printf("Failed to add entry to classifier pipe: %s\n", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = doca_flow_entries_process(ports[port_id], 0, DEFAULT_TIMEOUT_US, num_of_entries_ingress);
		if (result != DOCA_SUCCESS) {
			printf("Failed to process entries: %s\n", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		if (status_ingress.nb_processed != num_of_entries_ingress || status_ingress.failure) {
			printf("Failed to process entries: %s\n", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return DOCA_ERROR_BAD_STATE;
		}
	}

    for (int i = 0; i < nb_cores; i++) {
        struct worker_context * ctx = &worker_ctx[i];
        doca_worker_init(ctx);
    }

	return result;
}
