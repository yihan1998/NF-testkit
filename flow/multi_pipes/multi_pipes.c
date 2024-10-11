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
struct doca_flow_pipe *rss_pipe, *drop_pipe, *node2_pipe, *node4_pipe, *udp_pipe, *tcp_pipe;

/* RX queue configuration */
static struct rte_eth_rxconf rx_conf = {
    .rx_thresh = {
        .pthresh = 8,
        .hthresh = 8,
        .wthresh = 4,
    },
    .rx_free_thresh = 32,
#if RTE_VERSION >= RTE_VERSION_NUM(20, 11, 0, 0)
    .rx_deferred_start = 1,
#endif
};

/* TX queue configuration */
static struct rte_eth_txconf tx_conf = {
    .tx_thresh = {
        .pthresh = 36,
        .hthresh = 0,
        .wthresh = 0,
    },
    .tx_free_thresh = 0,
#if RTE_VERSION >= RTE_VERSION_NUM(20, 11, 0, 0)
    .tx_deferred_start = 1,
#endif
};

/* Port configuration */
struct rte_eth_conf port_conf = {
#if RTE_VERSION < RTE_VERSION_NUM(20, 11, 0, 0)
    .rxmode = {
        .mq_mode        = ETH_MQ_RX_NONE,
        .split_hdr_size = 0,
    },
#else
    .rxmode = {
        .mq_mode        = RTE_ETH_MQ_RX_RSS,
    },
    .rx_adv_conf = {
        .rss_conf = {
            .rss_key = NULL,
            .rss_hf =
                RTE_ETH_RSS_IP | RTE_ETH_RSS_TCP | RTE_ETH_RSS_UDP,
        },
    },
#endif
#if RTE_VERSION < RTE_VERSION_NUM(20, 11, 0, 0)
    .txmode = {
        .mq_mode = ETH_MQ_TX_NONE,
        .offloads = (DEV_TX_OFFLOAD_IPV4_CKSUM |
                DEV_TX_OFFLOAD_UDP_CKSUM |
                DEV_TX_OFFLOAD_TCP_CKSUM),
    },
#else
    .txmode = {
        .mq_mode = RTE_ETH_MQ_TX_NONE,
        .offloads = (RTE_ETH_TX_OFFLOAD_IPV4_CKSUM |
                RTE_ETH_TX_OFFLOAD_UDP_CKSUM |
                RTE_ETH_TX_OFFLOAD_TCP_CKSUM),
    },
#endif
};

/* Set match l4 port */
#define SET_L4_PORT(layer, port, value) \
do {\
	if (match.layer.l4_type_ext == DOCA_FLOW_L4_TYPE_EXT_TCP)\
		match.layer.tcp.l4_port.port = (value);\
	else if (match.layer.l4_type_ext == DOCA_FLOW_L4_TYPE_EXT_UDP)\
		match.layer.udp.l4_port.port = (value);\
} while (0)

#define NB_ACTION_ARRAY	(1)

doca_error_t create_udp_pipe(struct doca_flow_port *port) {
	struct doca_flow_match match;
	struct doca_flow_fwd fwd, fwd_miss;
	struct doca_flow_pipe_cfg pipe_cfg;
	struct doca_flow_pipe *pipe;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd_miss));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "UDP_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.match = &match;
	pipe_cfg.port = port;
	pipe_cfg.attr.is_root = true;

    fwd.type = DOCA_FLOW_FWD_DROP;
    fwd_miss.type = DOCA_FLOW_FWD_DROP;

	result = doca_flow_pipe_create(&pipe_cfg, NULL, NULL, &pipe);
	if (result != DOCA_SUCCESS) {
		printf("Failed to create UDP pipe: %s\n", doca_get_error_string(result));
		return result;
	}

	return DOCA_SUCCESS;
}

doca_error_t create_encap_pipe(struct doca_flow_port *port) {
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_actions *actions_arr[NB_ACTION_ARRAY];
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;
	struct doca_flow_pipe *pipe;
	struct entries_status status;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));
	memset(&status, 0, sizeof(status));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));

	pipe_cfg.attr.name = "ENCAP_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	pipe_cfg.match = &match;
	actions_arr[0] = &actions;
	pipe_cfg.actions = actions_arr;
	pipe_cfg.attr.nb_actions = NB_ACTION_ARRAY;
	pipe_cfg.port = port;
	pipe_cfg.attr.is_root = true;

	result = doca_flow_pipe_create(&pipe_cfg, NULL, NULL, &pipe);
	if (result != DOCA_SUCCESS) {
		printf("Failed to create ENCAP pipe: %s\n", doca_get_error_string(result));
		return result;
	}

	return DOCA_SUCCESS;
}

doca_error_t create_rss_pipe(struct doca_flow_port *port) {
	struct doca_flow_match match;
    struct doca_flow_actions actions;
	struct doca_flow_actions *actions_arr[NB_ACTION_ARRAY];
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;
	struct doca_flow_pipe *pipe;
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

	result = doca_flow_pipe_create(&pipe_cfg, &fwd, NULL, &pipe);
	if (result != DOCA_SUCCESS) {
		printf("Failed to create RSS pipe: %s\n", doca_get_error_string(result));
		return result;
	}

	return DOCA_SUCCESS;
}

doca_error_t create_drop_pipe(struct doca_flow_port *port) {
	struct doca_flow_match match;
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg pipe_cfg;
	struct doca_flow_pipe *pipe;
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

	result = doca_flow_pipe_create(&pipe_cfg, &fwd, NULL, &pipe);
	if (result != DOCA_SUCCESS) {
		printf("Failed to create DROP pipe: %s\n", doca_get_error_string(result));
		return result;
	}

	return DOCA_SUCCESS;
}

#define JUMBO_FRAME_LEN (4096 + RTE_ETHER_CRC_LEN + RTE_ETHER_HDR_LEN)

#define JUMBO_ETHER_MTU \
    (JUMBO_FRAME_LEN - RTE_ETHER_HDR_LEN - RTE_ETHER_CRC_LEN) /**< Ethernet MTU. */

#define N_MBUF              8192
#define DEFAULT_MBUF_SIZE	(JUMBO_FRAME_LEN + RTE_PKTMBUF_HEADROOM) /* See: http://dpdk.org/dev/patchwork/patch/4479/ */

#define RX_DESC_DEFAULT    1024
#define TX_DESC_DEFAULT    1024

int port_init(void) {
	struct rte_mempool *pkt_mempool;
	pkt_mempool = rte_pktmbuf_pool_create("pkt_mempool", N_MBUF,
                    RTE_MEMPOOL_CACHE_MAX_SIZE, 0, DEFAULT_MBUF_SIZE, rte_socket_id());
	assert(pkt_mempool != NULL);

	int ret;
    int pid = 0;
    int nb_rx_queue, nb_tx_queue;
	uint16_t mtu, new_mtu;

    nb_rx_queue = nb_tx_queue = 1;

	/* Get Ethernet device info */
	struct rte_eth_dev_info dev_info;
	ret = rte_eth_dev_info_get(pid, &dev_info);
	if (ret != 0) {
		printf("Error during getting device (port %u) info: %s\n", pid, strerror(-ret));
	}

	/* Configure # of RX and TX queue for port */
	ret = rte_eth_dev_configure(pid, nb_rx_queue, nb_tx_queue, &port_conf);
	if (ret < 0) {
		printf("cannot configure device: err=%d, port=%u\n", ret, pid);
	}

	 if (rte_eth_dev_get_mtu(pid, &mtu) != 0) {
            printf("Failed to get MTU for port %u\n", pid);
        }

        new_mtu = JUMBO_ETHER_MTU;

        if (rte_eth_dev_set_mtu(pid, new_mtu) != 0) {
            printf("Failed to set MTU to %u for port %u\n", new_mtu, pid);
        }

        port_conf.rxmode.mtu = new_mtu;

	/* Set up rx queue with pakcet mempool */
	for (int i = 0; i < nb_rx_queue; i++) {
		ret = rte_eth_rx_queue_setup(pid, i, RX_DESC_DEFAULT,
				rte_eth_dev_socket_id(pid), &rx_conf, pkt_mempool);
		if (ret < 0) {
			printf("Rx queue setup failed: err=%d, port=%u\n", ret, pid);
		}
	}

	/* Set up tx queue with pakcet mempool */
	for (int i = 0;i < nb_tx_queue;i++) {
		ret = rte_eth_tx_queue_setup(pid, i, TX_DESC_DEFAULT,
				rte_eth_dev_socket_id(pid), &tx_conf);
		if (ret < 0) {
			printf("Tx queue setup failed: err=%d, port=%u\n", ret, pid);
		}
	}

	printf("Port %d has %d RX queue and %d TX queue\n", pid, nb_rx_queue, nb_tx_queue);

	ret = rte_eth_promiscuous_enable(pid);
	if (ret != 0) {
		printf("rte_eth_promiscuous_enable:err = %d, port = %u\n", ret, (unsigned) pid);
	}

	/* Start Ethernet device */
	ret = rte_eth_dev_start(pid);
	if (ret < 0) {
		printf("rte_eth_dev_start:err = %d, port = %u\n", ret, (unsigned) pid);
	}

    return 0;
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

int main(int argc, char * argv[]) {
    int nb_queues = 1;
	struct doca_flow_resources resource = {0};
	uint32_t nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_MAX] = {0};
	doca_error_t result;
	int nb_packets;
	struct rte_mbuf *packets[PACKET_BURST];

	rte_eal_init(argc, argv);

	port_init();

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

    create_udp_pipe(ports[0]);
    create_encap_pipe(ports[0]);
	create_rss_pipe(ports[0]);
	create_drop_pipe(ports[0]);

	while (!force_quit) {
		nb_packets = rte_eth_rx_burst(0, 0, packets, PACKET_BURST);

		if (nb_packets) {
			printf("Received %u packets\n", nb_packets);
            // for (int i = 0; i < nb_packets; i++) {
            //     printf("Metadata: 0x%x\n", *RTE_FLOW_DYNF_METADATA(packets[i]));
            // }
			rte_pktmbuf_free_bulk(packets, nb_packets);
		}
	}

	return 0;
}