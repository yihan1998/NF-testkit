#include <stdlib.h>
#include <stdbool.h>
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

enum doca_flow_pipe_domain domain = DOCA_FLOW_PIPE_DOMAIN_DEFAULT;

#define PACKET_BURST	64
#define PULL_TIME_OUT 10000						/* Maximum timeout for pulling */

int nb_ports = 1;
struct doca_flow_port *ports[1];
struct doca_flow_pipe *rss_pipe, *udp_pipe, *control_pipe;
bool force_quit = false;

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

doca_error_t pipe_init(int nb_queues) {
	struct doca_flow_resources resource = {0};
	uint32_t nr_shared_resources[DOCA_FLOW_SHARED_RESOURCE_MAX] = {0};
	// struct entries_status status;
	doca_error_t result;
	// int num_of_entries = 1;
	// int port_id;

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

    return result;
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

doca_error_t
add_acl_specific_entry(struct doca_flow_pipe *pipe, int port_id, struct entries_status *status,
		       doca_be32_t src_ip_addr, doca_be32_t dst_ip_addr,
		       doca_be16_t src_port, doca_be16_t dst_port, uint8_t l4_type,
		       doca_be32_t src_ip_addr_mask, doca_be32_t dst_ip_addr_mask,
		       doca_be16_t src_port_mask, doca_be16_t dst_port_mask,
		       uint16_t priority,
		       bool is_allow, enum doca_flow_flags_type flag)
{
	struct doca_flow_match match;
	struct doca_flow_match match_mask;
	struct doca_flow_fwd fwd;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&match_mask, 0, sizeof(match_mask));
	memset(&fwd, 0, sizeof(fwd));

	match_mask.outer.ip4.src_ip = src_ip_addr_mask;
	match_mask.outer.ip4.dst_ip = dst_ip_addr_mask;

	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.ip4.src_ip = src_ip_addr;
	match.outer.ip4.dst_ip = dst_ip_addr;

	if (l4_type == DOCA_FLOW_L4_TYPE_EXT_TCP) {
		match.outer.tcp.l4_port.src_port = src_port;
		match.outer.tcp.l4_port.dst_port = dst_port;
		match_mask.outer.tcp.l4_port.src_port = src_port_mask;
		match_mask.outer.tcp.l4_port.dst_port = dst_port_mask;
	} else {
		match.outer.udp.l4_port.src_port = src_port;
		match.outer.udp.l4_port.dst_port = dst_port;
		match_mask.outer.udp.l4_port.src_port = src_port_mask;
		match_mask.outer.udp.l4_port.dst_port = dst_port_mask;
	}
	match.outer.l4_type_ext = l4_type;

	// if (is_allow) {
			fwd.type = DOCA_FLOW_FWD_PORT;
    		fwd.port_id = port_id ^ 1;
	// } else
	// 	fwd.type = DOCA_FLOW_FWD_DROP;

	result = doca_flow_pipe_acl_add_entry(0, pipe, &match, &match_mask,
			priority, &fwd, flag, status, NULL);

	if (result != DOCA_SUCCESS) {
		printf("Failed to add acl pipe entry: %s", doca_get_error_string(result));
		return result;
	}

	return DOCA_SUCCESS;
}


doca_error_t
add_acl_pipe_entries(struct doca_flow_pipe *pipe, int port_id, struct entries_status *status)
{
	doca_error_t result;

	result = add_acl_specific_entry(pipe, port_id, status,
			BE_IPV4_ADDR(1, 2, 3, 4), BE_IPV4_ADDR(8, 8, 8, 8),
			RTE_BE16(1234), RTE_BE16(80), DOCA_FLOW_L4_TYPE_EXT_TCP,
			RTE_BE32(0xffffffff), RTE_BE32(0xffffffff),
			RTE_BE16(0x00), RTE_BE16(0x0), 10, false,
			DOCA_FLOW_WAIT_FOR_BATCH);
	if (result != DOCA_SUCCESS)
		return result;

	result = add_acl_specific_entry(pipe, port_id, status,
			BE_IPV4_ADDR(172, 20, 1, 4), BE_IPV4_ADDR(192, 168, 3, 4),
			RTE_BE16(1234), RTE_BE16(80), DOCA_FLOW_L4_TYPE_EXT_UDP,
			RTE_BE32(0xffffffff), RTE_BE32(0xffffffff),
			RTE_BE16(0x0), RTE_BE16(3000), 50, true,
			DOCA_FLOW_WAIT_FOR_BATCH);

	if (result != DOCA_SUCCESS)
		return result;

	result = add_acl_specific_entry(pipe, port_id, status,
			BE_IPV4_ADDR(172, 20, 1, 4), BE_IPV4_ADDR(192, 168, 3, 4),
			RTE_BE16(1234), RTE_BE16(80), DOCA_FLOW_L4_TYPE_EXT_TCP,
			RTE_BE32(0xffffffff), RTE_BE32(0xffffffff),
			RTE_BE16(1234), RTE_BE16(0x0), 40, true,
			DOCA_FLOW_WAIT_FOR_BATCH);

	if (result != DOCA_SUCCESS)
		return result;

	result = add_acl_specific_entry(pipe, port_id, status,
			BE_IPV4_ADDR(1, 2, 3, 5), BE_IPV4_ADDR(8, 8, 8, 6),
			RTE_BE16(1234), RTE_BE16(80), DOCA_FLOW_L4_TYPE_EXT_TCP,
			RTE_BE32(0xffffff00), RTE_BE32(0xffffff00),
			RTE_BE16(0xffff), RTE_BE16(80), 20, true,
			DOCA_FLOW_NO_WAIT);

	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

doca_error_t
create_acl_pipe(struct doca_flow_port *port, bool is_root, struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions, *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_pipe_cfg pipe_cfg;
	struct doca_flow_fwd fwd_miss;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&pipe_cfg, 0, sizeof(pipe_cfg));
	memset(&fwd_miss, 0, sizeof(fwd_miss));

	pipe_cfg.attr.name = "ACL_PIPE";
	pipe_cfg.attr.type = DOCA_FLOW_PIPE_ACL;
	pipe_cfg.attr.is_root = is_root;
	pipe_cfg.attr.nb_flows = 10;
	pipe_cfg.attr.domain = domain;
	pipe_cfg.match = &match;
	actions_arr[0] = &actions;
	pipe_cfg.actions = actions_arr;
	pipe_cfg.attr.nb_actions = NB_ACTIONS_ARR;
	pipe_cfg.port = port;

	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.ip4.src_ip = 0xffffffff;
	match.outer.ip4.dst_ip = 0xffffffff;

	match.outer.tcp.l4_port.src_port = 0xffff;
	match.outer.tcp.l4_port.dst_port = 0xffff;

	fwd_miss.type = DOCA_FLOW_FWD_DROP;

	return doca_flow_pipe_create(&pipe_cfg, NULL, &fwd_miss, pipe);
}


int main(int argc, char * argv[]) {
	// int nb_packets;
	// struct rte_mbuf *packets[PACKET_BURST];
	doca_error_t result;
	struct doca_flow_pipe *acl_pipe;
    int port_acl = 0;
	int num_of_entries = 4;
	struct entries_status status;

    memset(&status, 0, sizeof(status));

	rte_eal_init(argc, argv);

	port_init();

    pipe_init(1);

    result = create_acl_pipe(ports[port_acl], true, &acl_pipe);
    if (result != DOCA_SUCCESS) {
        printf("Failed to create acl pipe: %s", doca_get_error_string(result));
        doca_flow_destroy();
        return result;
    }

    result = add_acl_pipe_entries(acl_pipe, port_acl, &status);
    if (result != DOCA_SUCCESS) {
        doca_flow_destroy();
        return result;
    }

    result = doca_flow_entries_process(ports[port_acl], 0, DEFAULT_TIMEOUT_US, num_of_entries);
    if (result != DOCA_SUCCESS) {
        printf("Failed to process entries: %s", doca_get_error_string(result));
        doca_flow_destroy();
        return result;
    }

	return 0;
}
