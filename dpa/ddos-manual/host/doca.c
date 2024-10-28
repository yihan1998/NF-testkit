#include <string.h>
#include <unistd.h>

#include <doca_log.h>
#include <doca_flow.h>

#include "flow_common.h"

struct doca_flow_pipe_entry *match_entry[2];

static doca_error_t create_match_pipe(struct doca_flow_port *port, int port_id, struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions, *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_fwd fwd;
#ifdef ENABLE_COUNTER
	struct doca_flow_monitor counter;
#endif	/* ENABLE_COUNTER */
	struct doca_flow_pipe_cfg *pipe_cfg;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));
#ifdef ENABLE_COUNTER
	memset(&counter, 0, sizeof(counter));
#endif	/* ENABLE_COUNTER */

	/* 5 tuple match */
	match.parser_meta.outer_l3_type = DOCA_FLOW_L3_META_IPV4;
	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	// match.outer.ip4.src_ip = 0xffffffff;
	match.outer.ip4.dst_ip = 0xffffffff;
	// match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TRANSPORT;
	// match.outer.transport.src_port = 0xffff;
	// match.outer.transport.dst_port = 0xffff;

	/* set meta data to match on the egress domain */
	actions.meta.pkt_meta = UINT32_MAX;
	actions.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_TRANSPORT;
	actions.outer.transport.src_port = 0xffff;
	actions_arr[0] = &actions;

#ifdef ENABLE_COUNTER
	counter.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
#endif	/* ENABLE_COUNTER */

	result = doca_flow_pipe_cfg_create(&pipe_cfg, port);
	if (result != DOCA_SUCCESS) {
		printf("Failed to create doca_flow_pipe_cfg: %s\n", doca_error_get_descr(result));
		return result;
	}

	result = set_flow_pipe_cfg(pipe_cfg, "MATCH_PIPE", DOCA_FLOW_PIPE_BASIC, true);
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
#ifdef ENABLE_COUNTER
	result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &counter);
	if (result != DOCA_SUCCESS) {
		printf("Failed to set doca_flow_pipe_cfg counter: %s\n", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
#endif	/* ENABLE_COUNTER */

	/* forwarding traffic to other port */
	fwd.type = DOCA_FLOW_FWD_PORT;
	fwd.port_id = port_id ^ 1;

	result = doca_flow_pipe_create(pipe_cfg, &fwd, NULL, pipe);
destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	return result;
}

static doca_error_t add_match_pipe_entry(struct doca_flow_pipe *pipe, 
							struct entries_status *status, 
						   	struct doca_flow_pipe_entry **entry)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	// struct doca_flow_pipe_entry *entry;
	doca_error_t result;

	doca_be32_t dst_ip_addr = BE_IPV4_ADDR(10, 10, 10, 10);
	// doca_be32_t src_ip_addr = BE_IPV4_ADDR(1, 2, 3, 4);
	// doca_be16_t dst_port = rte_cpu_to_be_16(80);
	// doca_be16_t src_port = rte_cpu_to_be_16(1234);

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));

	match.outer.ip4.dst_ip = dst_ip_addr;
	// match.outer.ip4.src_ip = src_ip_addr;
	// match.outer.transport.dst_port = dst_port;
	// match.outer.transport.src_port = src_port;

	actions.meta.pkt_meta = 1;
	// actions.outer.transport.src_port = rte_cpu_to_be_16(1235);
	actions.action_idx = 0;

	result = doca_flow_pipe_add_entry(0, pipe, &match, &actions, NULL, NULL, 0, status, entry);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

doca_error_t doca_init(struct application_dpdk_config *app_dpdk_config)
{
	int nb_ports = 2;
#ifdef ENABLE_COUNTER
	struct flow_resources resource = {.nr_counters = 64};
#else
	struct flow_resources resource = {0};
#endif	/* ENABLE_COUNTER */
	uint32_t nr_shared_resources[SHARED_RESOURCE_NUM_VALUES] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_dev *dev_arr[nb_ports];
	struct doca_flow_pipe *pipe;
	struct entries_status status_ingress;
	int num_of_entries_ingress = 1;
	doca_error_t result;
	int port_id;

	result = init_doca_flow(nb_queues, "vnf,hws", &resource, nr_shared_resources);
	if (result != DOCA_SUCCESS) {
		printf("Failed to init DOCA Flow: %s\n", doca_error_get_descr(result));
		return result;
	}

	memset(dev_arr, 0, sizeof(struct doca_dev *) * nb_ports);
	result = init_doca_flow_ports(app_dpdk_config->port_cfg.nb_queues, ports, true, dev_arr);
	if (result != DOCA_SUCCESS) {
		printf("Failed to init DOCA ports: %s\n", doca_error_get_descr(result));
		doca_flow_destroy();
		return result;
	}

	for (port_id = 0; port_id < nb_ports; port_id++) {
		memset(&status_ingress, 0, sizeof(status_ingress));

		result = create_match_pipe(ports[port_id], port_id, &pipe);
		if (result != DOCA_SUCCESS) {
			printf("Failed to create match pipe: %s\n", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_match_pipe_entry(pipe, &status_ingress, &match_entry[port_id]);
		if (result != DOCA_SUCCESS) {
			printf("Failed to add entry to match pipe: %s\n", doca_error_get_descr(result));
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

	return result;
}
