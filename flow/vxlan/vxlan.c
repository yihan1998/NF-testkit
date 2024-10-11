#include <doca_flow.h>
#include <doca_log.h>

#include <rte_common.h>
#include <rte_eal.h>
#include <rte_flow.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_version.h>

#include "flow_common.h"

static doca_error_t create_control_pipe(struct doca_flow_port *port, struct doca_flow_pipe **control_pipe) {
	struct doca_flow_pipe_cfg pipe_cfg;

	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "CONTROL_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_CONTROL;
	pipe_cfg.attr.is_root = true;
	pipe_cfg.port = port;

	if (doca_flow_pipe_create(&pipe_cfg, NULL, NULL, control_pipe) != DOCA_SUCCESS)
		return -1;
	return DOCA_SUCCESS;
}

static doca_error_t add_control_pipe_entries(struct doca_flow_port *port, struct doca_flow_pipe *control_pipe, struct doca_flow_pipe *udp_pipe) {
	struct doca_flow_match match;
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_entry *entry;
	doca_error_t result;
	uint8_t priority = 0;

	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));

	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
	match.outer.udp.l4_port.dst_port = rte_cpu_to_be_16(0x1234);

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = udp_pipe;

	result = doca_flow_pipe_control_add_entry(0, priority, control_pipe, &match,
						 NULL, NULL, NULL, NULL, NULL, &fwd, NULL, &entry);
	if (result != DOCA_SUCCESS)
		return -1;
	return DOCA_SUCCESS;
}

#define NB_ACTION_ARRAY (1)						/* Used as the size of muti-actions array for DOCA Flow API */

static doca_error_t create_udp_pipe(struct doca_flow_port *port, struct doca_flow_pipe *udp_pipe) {
	struct doca_flow_match match;
	struct doca_flow_actions actions, *actions_arr[NB_ACTION_ARRAY];
	struct doca_flow_fwd fwd;
	struct doca_flow_fwd fwd_miss;
	struct doca_flow_pipe_cfg pipe_cfg;
	struct doca_flow_pipe **pipe;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd_miss));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.match = &match;
	actions_arr[0] = &actions;
	pipe_cfg.actions = actions_arr;
	pipe_cfg.attr.is_root = false;
	pipe_cfg.attr.nb_actions = NB_ACTION_ARRAY;
	pipe_cfg.port = port;

    pipe_cfg.attr.name = "VXLAN_PIPE";
    pipe = udp_pipe;

	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
    match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
    match.outer.udp.l4_port = RTE_BE16(0x1234);

	fwd.type = DOCA_FLOW_FWD_DROP;

	if (doca_flow_pipe_create(&pipe_cfg, &fwd, &fwd_miss, pipe) != DOCA_SUCCESS)
		return -1;

	return 0;
}

int main(int argc, char **argv) {
	doca_error_t result;
    result = create_control_pipe(curr_port_cfg);
    if (result < 0) {
        DOCA_LOG_ERR("Failed building control pipe");
        return -1;
    }

    printf("Add control pipe entries...\n");
    result = add_control_pipe_entries(curr_port_cfg);
    if (result < 0) {
        printf("Failed adding entries to the control pipe\n");
        return -1;
    }
}