#include <stdlib.h>
#include <stdbool.h>
#include <signal.h>
#include <sys/time.h>

#include <doca_flow.h>
#include <doca_log.h>

#include <rte_common.h>
#include <rte_eal.h>
#include <rte_flow.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_version.h>

#include "flow_common.h"

#define PACKET_BURST	64
#define PULL_TIME_OUT 10000						/* Maximum timeout for pulling */

bool force_quit = false;

int nb_ports = 1;
struct doca_flow_port *ports[1];

/* Set match l4 port */
#define SET_L4_PORT(layer, port, value) \
do {\
	if (match.layer.l4_type_ext == DOCA_FLOW_L4_TYPE_EXT_TCP)\
		match.layer.tcp.l4_port.port = (value);\
	else if (match.layer.l4_type_ext == DOCA_FLOW_L4_TYPE_EXT_UDP)\
		match.layer.udp.l4_port.port = (value);\
} while (0)

#define NB_ACTION_ARRAY	(1)

doca_error_t create_rss_pipe(struct doca_flow_port *port, struct doca_flow_pipe **pipe) {
	struct doca_flow_match match;
    struct doca_flow_actions actions;
	struct doca_flow_actions *actions_arr[NB_ACTION_ARRAY];
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;
	struct entries_status status;
    int nb_queues = 1;
	uint16_t rss_queues[nb_queues];
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));
	memset(&status, 0, sizeof(status));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "RSS_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.match = &match;
    actions_arr[0] = &actions;
	pipe_cfg.actions = actions_arr;
	pipe_cfg.attr.nb_actions = NB_ACTION_ARRAY;
	pipe_cfg.port = port;
	pipe_cfg.attr.is_root = false;

    for (int i = 0; i < nb_queues; i++)
        rss_queues[i] = i;

    fwd.type = DOCA_FLOW_FWD_RSS;
    fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4;
    fwd.num_of_queues = nb_queues;
    fwd.rss_queues = rss_queues;

	result = doca_flow_pipe_create(&pipe_cfg, &fwd, NULL, pipe);
	if (result != DOCA_SUCCESS) {
		printf("Failed to create RSS pipe: %s\n", doca_get_error_string(result));
		return result;
	}

	return DOCA_SUCCESS;
}

doca_error_t create_udp_pipe(struct doca_flow_port *port, struct doca_flow_pipe **pipe) {
	struct doca_flow_match match;
	struct doca_flow_fwd fwd_miss;
	struct doca_flow_pipe_cfg pipe_cfg;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&fwd_miss, 0, sizeof(fwd_miss));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "UDP_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.match = &match;
	pipe_cfg.port = port;
	pipe_cfg.attr.is_root = true;

    match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
	match.outer.udp.l4_port.dst_port = rte_cpu_to_be_16(0xffff);

    fwd_miss.type = DOCA_FLOW_FWD_DROP;

	result = doca_flow_pipe_create(&pipe_cfg, NULL, &fwd_miss, pipe);
	if (result != DOCA_SUCCESS) {
		printf("Failed to create UDP pipe: %s\n", doca_get_error_string(result));
		return result;
	}

	return DOCA_SUCCESS;
}

doca_error_t create_drop_pipe(struct doca_flow_port *port, struct doca_flow_pipe **pipe) {
	struct doca_flow_match match;
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;
	struct entries_status status;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));
	memset(&status, 0, sizeof(status));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "DROP_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.match = &match;
	pipe_cfg.port = port;
	pipe_cfg.attr.is_root = false;

    fwd.type = DOCA_FLOW_FWD_DROP;

	result = doca_flow_pipe_create(&pipe_cfg, &fwd, NULL, pipe);
	if (result != DOCA_SUCCESS) {
		printf("Failed to create DROP pipe: %s\n", doca_get_error_string(result));
		return result;
	}

	return DOCA_SUCCESS;
}

static void signal_handler(int signum) {
	if (signum == SIGINT) {
		for (int port_id = 0; port_id < nb_ports; port_id++) {
			if (ports[port_id] != NULL) {
				doca_flow_port_stop(ports[port_id]);
				printf("Flow port stoped!\n");
			}
		}
		doca_flow_destroy();
		printf("DOCA Flow destroyed!\n");
		force_quit = true;
	}
}

static doca_error_t add_udp_pipe_entry(struct doca_flow_port *port, struct doca_flow_pipe *udp_pipe, struct doca_flow_pipe *rss_pipe) {
    struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;
	struct doca_flow_pipe_entry *entry;
	struct entries_status *status;
	int num_of_entries = 1;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	status = (struct entries_status *)calloc(1, sizeof(struct entries_status));

	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.l4_type_ext = DOCA_FLOW_L4_TYPE_EXT_UDP;
	match.outer.udp.l4_port.dst_port = rte_cpu_to_be_16(0x1234);

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = rss_pipe;

	result = doca_flow_pipe_add_entry(0, udp_pipe, &match, &actions, NULL, &fwd, 0, status, &entry);
	if (result != DOCA_SUCCESS) {
		printf("Failed to add pipe entry: %s\n", doca_get_error_string(result));
		free(status);
		return -1;
	}
	result = doca_flow_entries_process(port, 0, PULL_TIME_OUT, num_of_entries);
	if (result != DOCA_SUCCESS) {
		printf("Failed to process pipe entry: %s\n", doca_get_error_string(result));
		return -1;
	}

	if (status->nb_processed != num_of_entries || status->failure) {
		printf("Process failed: %s\n", doca_get_error_string(result));
		return -1;
	}

	return DOCA_SUCCESS;
}

int entrypoint(int argc, char * argv[]) {
    int nb_queues = 1;
	struct doca_flow_resources resource = {0};
	uint32_t nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_MAX] = {0};
	doca_error_t result;
    struct doca_flow_pipe *rss_pipe, *drop_pipe, *udp_pipe;

	signal(SIGINT, signal_handler);

    result = init_doca_flow(nb_queues, "vnf,hws", resource, nr_shared_resources);
	if (result != DOCA_SUCCESS) {
		printf("Failed to init DOCA Flow: %s\n", doca_get_error_string(result));
		return result;
	}

	printf("DOCA flow init!\n");

	result = init_doca_flow_ports(nb_ports, ports, false);
	if (result != DOCA_SUCCESS) {
		printf("Failed to init DOCA ports: %s\n", doca_get_error_string(result));
		doca_flow_destroy();
		return result;
	}

	printf("DOCA flow ports init!\n");

    {
        struct doca_flow_match match;
        struct doca_flow_actions actions;
        struct doca_flow_fwd fwd;
        struct doca_flow_pipe_cfg pipe_cfg;
        struct doca_flow_pipe_entry *entry;
        struct entries_status status = {0};
        int num_of_entries = 1;
        doca_error_t result;

        create_rss_pipe(ports[0], &rss_pipe);
        create_drop_pipe(ports[0], &drop_pipe);
        create_udp_pipe(ports[0], &udp_pipe);

        memset(&match, 0, sizeof(match));
        memset(&actions, 0, sizeof(actions));
        memset(&fwd, 0, sizeof(fwd));
        memset(&pipe_cfg, 0, sizeof(pipe_cfg));

        fwd.type = DOCA_FLOW_FWD_PIPE;
        fwd.next_pipe = drop_pipe;

        result = doca_flow_pipe_add_entry(0, udp_pipe, &match, &actions, NULL, &fwd, 0, &status, &entry);
        if (result != DOCA_SUCCESS) {
            printf("Failed to add pipe entry: %s\n", doca_get_error_string(result));
            free(status);
            return -1;
        }

        /* link between udp and encap */
        result = doca_flow_entries_process(ports[0], 0, PULL_TIME_OUT, num_of_entries);
        if (result != DOCA_SUCCESS) {
            printf("Failed to process pipe entry: %s\n", doca_get_error_string(result));
            return -1;
        }

        if (status.nb_processed != num_of_entries || status.failure) {
            printf("Process failed: %s\n", doca_get_error_string(result));
            return -1;
        }
    }

    bool add_entry = false;
	struct timeval start, curr;
	gettimeofday(&start, NULL);

	while (!force_quit) {
		gettimeofday(&curr, NULL);
        if (curr.tv_sec - start.tv_sec > 10 && !add_entry) {
			printf("Add udp entry!\n");
			add_udp_pipe_entry(ports[0], udp_pipe, rss_pipe);
			add_entry = true;
		}
	}

	return 0;
}