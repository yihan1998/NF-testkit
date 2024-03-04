#include "net.h"
#include "netfmt.h"

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_errno.h>
#include <rte_config.h>
#include <rte_log.h>
#include <rte_tailq.h>
#include <rte_common.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_memzone.h>
#include <rte_malloc.h>
#include <rte_per_lcore.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_pci.h>
#include <rte_random.h>
#include <rte_timer.h>
#include <rte_ether.h>
#include <rte_ring.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_tcp.h>
#include <rte_version.h>

#include <errno.h>

#define MAX_NR_CORES        16
#define DEFAULT_PKT_BURST   32
#define DEFAULT_RX_DESC     1024
#define DEFAULT_TX_DESC     1024

struct rte_mempool * mempools[MAX_NR_CORES];

struct flow {
    uint32_t src_ip;
    uint32_t src_ip_mask;
    uint32_t dst_ip;
    uint32_t dst_ip_mask;
    uint16_t src_port;
    uint16_t src_port_mask;
    uint16_t dst_port;
    uint16_t dst_port_mask;
};

struct mbuf_table {
	uint16_t next;
	uint16_t len;
	struct rte_mbuf *m_table[DEFAULT_PKT_BURST];
};

__thread int lid;
__thread int qid;
__thread struct rte_mempool * lcore_mempool;
__thread struct mbuf_table rx_mbufs;
__thread struct mbuf_table tx_mbufs;

static struct rte_eth_conf port_conf = {
	.rxmode = {
        .mq_mode    = ETH_MQ_RX_RSS,
        .split_hdr_size = 0,
    },
    .txmode = {
        .mq_mode    = ETH_MQ_TX_NONE,
    },
    .rx_adv_conf = {
		.rss_conf = {
			.rss_key = NULL,
			.rss_hf = RTE_ETH_RSS_IP | RTE_ETH_RSS_UDP | RTE_ETH_RSS_TCP,
		},
	},
};

int net_init(int nb_cores, int nb_rxq, int nb_txq) {
    int argc = 7;
    char core_list[16];

    int ret;
    int pid, nb_mbufs, nb_total, nb_avail;
    char name[RTE_MEMZONE_NAMESIZE];
    uint16_t nb_rxd = DEFAULT_RX_DESC;
    uint16_t nb_txd = DEFAULT_TX_DESC;

    pid = 0;    // Use the first and only port
    nb_total = nb_avail = 0;

    sprintf(core_list, "0-%d", nb_cores - 1);

    char * argv[] = {
        "-l", core_list,
        "-n", "4",
        "-a", "4b:00.0", ""};

    ret = rte_eal_init(argc, argv);
	if (ret < 0) {
		return -1;
    }

    nb_total = rte_eth_dev_count_total();
    nb_avail = rte_eth_dev_count_avail();

    if (nb_total < 1) {
        perror("No available dev!");
        rte_exit(EXIT_FAILURE, "No dev detected! (total: %d)\n", nb_total);
    }

    if (nb_avail != 1) {
        rte_exit(EXIT_FAILURE, "Specify only one dev! (avail: %d)\n", nb_avail);
    }

    nb_mbufs = RTE_MAX(DEFAULT_RX_DESC + DEFAULT_TX_DESC + DEFAULT_PKT_BURST + RTE_MEMPOOL_CACHE_MAX_SIZE, 8192U);

    /* Create mbuf pool for each core */
    for (int i = 0; i < nb_cores; i++) {
        sprintf(name, "mbuf_pool_%d", i);
        mempools[i] = rte_pktmbuf_pool_create(name, nb_mbufs,
            RTE_MEMPOOL_CACHE_MAX_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
        if (!mempools[i]) {
            rte_exit(EXIT_FAILURE, "Cannot init mbuf pool\n");
        }
    }

    /* Initialise each port */
    struct rte_eth_rxconf rxq_conf;
    struct rte_eth_txconf txq_conf;
    struct rte_eth_conf local_port_conf = port_conf;
    struct rte_eth_dev_info dev_info;

    ret = rte_eth_dev_info_get(pid, &dev_info);
    if (ret != 0) {
        rte_exit(EXIT_FAILURE, "Error during getting device (port %u) info: %s\n", pid, strerror(-ret));
    }

    if (nb_rxq > 1 && nb_txq > 1) {
        local_port_conf.rx_adv_conf.rss_conf.rss_key = NULL;
        local_port_conf.rx_adv_conf.rss_conf.rss_hf &= dev_info.flow_type_rss_offloads;
    } else {
        local_port_conf.rx_adv_conf.rss_conf.rss_key = NULL;
        local_port_conf.rx_adv_conf.rss_conf.rss_hf  = 0;
    }

    if (local_port_conf.rx_adv_conf.rss_conf.rss_hf != 0) {
        local_port_conf.rxmode.mq_mode = ETH_MQ_RX_RSS;
    } else {
        local_port_conf.rxmode.mq_mode = ETH_MQ_RX_NONE;
    }

    if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE) {
        local_port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE;
    }

    local_port_conf.rx_adv_conf.rss_conf.rss_hf &= dev_info.flow_type_rss_offloads;
    if (local_port_conf.rx_adv_conf.rss_conf.rss_hf != port_conf.rx_adv_conf.rss_conf.rss_hf) {
        printf("Port %u modified RSS hash function based on hardware support,"
            "requested:%#lx configured:%#lx\n",
            pid,
            port_conf.rx_adv_conf.rss_conf.rss_hf,
            local_port_conf.rx_adv_conf.rss_conf.rss_hf);
    }

    /* Configure the number of queues for a port. */
    ret = rte_eth_dev_configure(pid, nb_rxq, nb_txq, &local_port_conf);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Cannot configure device: err=%d, port=%u\n", ret, pid);
    }
    /* >8 End of configuration of the number of queues for a port. */

    ret = rte_eth_dev_adjust_nb_rx_tx_desc(pid, &nb_rxd, &nb_txd);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Cannot adjust number of descriptors: err=%d, port=%u\n", ret, pid);
    }

    printf("DPDK set up with %d rxq, %d txq\n", nb_rxq, nb_txq);

    /* init ont RX queue and TX queue for each core */
    rxq_conf = dev_info.default_rxconf;
    rxq_conf.offloads = local_port_conf.rxmode.offloads;

    txq_conf = dev_info.default_txconf;
    txq_conf.offloads = local_port_conf.txmode.offloads;

    for (int i = 0; i < nb_cores; i++) {
        /* RX queue setup. 8< */
        ret = rte_eth_rx_queue_setup(pid, i, nb_rxd, rte_eth_dev_socket_id(pid), &rxq_conf, mempools[i]);
        if (ret < 0) {
            rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup:err=%d, port=%u\n", ret, pid);
        }

        ret = rte_eth_tx_queue_setup(pid, i, nb_txd, rte_eth_dev_socket_id(pid), &txq_conf);
        if (ret < 0) {
            rte_exit(EXIT_FAILURE, "rte_eth_tx_queue_setup:err=%d, port=%u\n", ret, pid);
        }
    }
    /* >8 End of queue setup. */

    ret = rte_eth_dev_set_ptypes(pid, RTE_PTYPE_UNKNOWN, NULL, 0);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Failed to disable Ptype parsing:err=%d, port=%u\n", ret, pid);
    }

    /* Start device */
    ret = rte_eth_dev_start(pid);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "rte_eth_dev_start:err=%d, port=%u\n", ret, pid);
    }

    printf("Port %d initialization done\n", pid);
    ret = rte_eth_promiscuous_enable(pid);
    if (ret != 0) {
        rte_exit(EXIT_FAILURE, "rte_eth_promiscuous_enable:err=%s, port=%u\n", rte_strerror(-ret), pid);
    }

    return 0;
}

int net_setup(int lcore_id) {
    lid = lcore_id;
    qid = lid;

    lcore_mempool = mempools[lid];

    rx_mbufs.next = rx_mbufs.len = 0;
    for (int i = 0; i < DEFAULT_PKT_BURST; i++) {
        rx_mbufs.m_table[i] = NULL;
    }

    tx_mbufs.next = tx_mbufs.len = 0;
    for (int i = 0; i < DEFAULT_PKT_BURST; i++) {
        tx_mbufs.m_table[i] = NULL;
    }

    return 0;
}

int net_rx(void) {
    int pid = 0;

    int nb_recv = rte_eth_rx_burst(pid, qid, rx_mbufs.m_table, DEFAULT_PKT_BURST);

    if (nb_recv == 0) {
        return 0;
    }

    rx_mbufs.next = 0;
    rx_mbufs.len = nb_recv;

    return nb_recv;
}

uint8_t * net_get_rxpkt(int * pkt_len) {
    struct rte_mbuf * pkt;

    if (rx_mbufs.next == rx_mbufs.len) {
        /* No more received packet */
        rte_pktmbuf_free_bulk(rx_mbufs.m_table, rx_mbufs.len);
        rx_mbufs.next = rx_mbufs.len = 0;
        return NULL;
    }

    pkt = rx_mbufs.m_table[rx_mbufs.next++];
    *pkt_len = pkt->pkt_len;
	return rte_pktmbuf_mtod(pkt, uint8_t *);
}

int net_tx(void) {
    int total_pkt, pkt_cnt;
    struct rte_mbuf ** pkts;
    int pid = 0;

    pkts = tx_mbufs.m_table;
    total_pkt = pkt_cnt = tx_mbufs.len;

    if (pkt_cnt > 0) {
        int ret;
        do {
            /* Send packets until there is none in TX queue */
            ret = rte_eth_tx_burst(pid, qid, pkts, pkt_cnt);
            pkts += ret;
            pkt_cnt -= ret;
        } while (pkt_cnt > 0);

        tx_mbufs.len = 0;
    }

    return total_pkt;
}

uint8_t * net_get_txpkt(int len) {
    struct rte_mbuf * m;

    if (unlikely(tx_mbufs.len == DEFAULT_PKT_BURST)) {
        printf("TX buffer full\n");
        return NULL;
    }

    m = rte_pktmbuf_alloc(lcore_mempool);
    if (unlikely(m == NULL)) {
        printf("No packet buffers found\n");
        return NULL;
    }

    m->data_len = m->pkt_len = len;
    m->next = NULL;
    m->nb_segs = 1;

    tx_mbufs.m_table[tx_mbufs.len++] = m;

    return rte_pktmbuf_mtod(m, uint8_t *);
}

#define FULL_IP_MASK   0xffffffff /* full mask */
#define EMPTY_IP_MASK  0x0 /* empty mask */

#define FULL_PORT_MASK   0xffff /* full mask */
#define PART_PORT_MASK   0xff00 /* partial mask */
#define EMPTY_PORT_MASK  0x0 /* empty mask */

#define MAX_PATTERN_NUM		4
#define MAX_ACTION_NUM		2

int net_create_tcp_flow(int rx_q, struct flow * fl) {
    int port_id;
    RTE_ETH_FOREACH_DEV(port_id) {
        struct rte_flow_error error;
        struct rte_flow_attr attr;
        struct rte_flow_item pattern[MAX_PATTERN_NUM];
        struct rte_flow_action action[MAX_ACTION_NUM];
        struct rte_flow * flow = NULL;
        struct rte_flow_action_queue queue = { .index = rx_q };
        struct rte_flow_item_ipv4 ip_spec;
        struct rte_flow_item_ipv4 ip_mask;
        struct rte_flow_item_tcp tcp_spec;
        struct rte_flow_item_tcp tcp_mask;
        int res;

        memset(pattern, 0, sizeof(pattern));
        memset(action, 0, sizeof(action));

        /*
        * set the rule attribute.
        * in this case only ingress packets will be checked.
        */
        memset(&attr, 0, sizeof(struct rte_flow_attr));
        attr.ingress = 1;

        /*
        * create the action sequence.
        * one action only,  move packet to queue
        */
        action[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
        action[0].conf = &queue;
        action[1].type = RTE_FLOW_ACTION_TYPE_END;

        /*
        * set the first level of the pattern (ETH).
        */
        pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;

        /*
        * setting the second level of the pattern (IP).
        */
        memset(&ip_spec, 0, sizeof(struct rte_flow_item_ipv4));
        memset(&ip_mask, 0, sizeof(struct rte_flow_item_ipv4));
        ip_spec.hdr.dst_addr = htonl(fl->dst_ip);
        ip_mask.hdr.dst_addr = htonl(fl->dst_ip_mask);
        ip_spec.hdr.src_addr = htonl(fl->src_ip);
        ip_mask.hdr.src_addr = htonl(fl->src_ip_mask);
        pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
        pattern[1].spec = &ip_spec;
        pattern[1].mask = &ip_mask;

        /*
        * setting the third level of the pattern (TCP).
        */
        memset(&tcp_spec, 0, sizeof(struct rte_flow_item_tcp));
        memset(&tcp_mask, 0, sizeof(struct rte_flow_item_tcp));
        tcp_spec.hdr.dst_port = htons(fl->dst_port);
        tcp_mask.hdr.dst_port = htons(fl->dst_port_mask);
        tcp_spec.hdr.src_port = htons(fl->src_port);
        tcp_mask.hdr.src_port = htons(fl->src_port_mask);
        pattern[2].type = RTE_FLOW_ITEM_TYPE_TCP;
        pattern[2].spec = &tcp_spec;
        pattern[2].mask = &tcp_mask;

        /* the final level must be always type end */
        pattern[3].type = RTE_FLOW_ITEM_TYPE_END;

        res = rte_flow_validate(port_id, &attr, pattern, action, &error);
        if (!res) {
retry:
            flow = rte_flow_create(port_id, &attr, pattern, action, &error);
            if (!flow) {
                rte_flow_flush(port_id, &error);
                goto retry;
            }

            uint32_t src_ip_masked, dst_ip_masked;
            uint16_t src_port_masked, dst_port_masked;

            src_ip_masked = fl->src_ip & fl->src_ip_mask;
            dst_ip_masked = fl->dst_ip & fl->dst_ip_mask;
            src_port_masked = fl->src_port & fl->src_port_mask;
            dst_port_masked = fl->dst_port & fl->dst_port_mask;
            printf("Create UDP flow from " IP_STRING ":%u(%x) to " IP_STRING ":%u(%x) by queue %d on port %d\n", \
                        HOST_IP_FMT(src_ip_masked), src_port_masked, src_port_masked, \
                        HOST_IP_FMT(dst_ip_masked), dst_port_masked, dst_port_masked, rx_q, port_id);
        } else {
            printf("Invalid flow rule! msg: %s\n", error.message);
        }
    }

	return 0;
}

int net_create_udp_flow(int rx_q, struct flow * fl) {
    int port_id;
    RTE_ETH_FOREACH_DEV(port_id) {
        struct rte_flow_error error;
        struct rte_flow_attr attr;
        struct rte_flow_item pattern[MAX_PATTERN_NUM];
        struct rte_flow_action action[MAX_ACTION_NUM];
        struct rte_flow * flow = NULL;
        struct rte_flow_action_queue queue = { .index = rx_q };
        struct rte_flow_item_ipv4 ip_spec;
        struct rte_flow_item_ipv4 ip_mask;
        struct rte_flow_item_udp udp_spec;
        struct rte_flow_item_udp udp_mask;
        int res;

        memset(pattern, 0, sizeof(pattern));
        memset(action, 0, sizeof(action));

        /*
        * set the rule attribute.
        * in this case only ingress packets will be checked.
        */
        memset(&attr, 0, sizeof(struct rte_flow_attr));
        attr.ingress = 1;

        /*
        * create the action sequence.
        * one action only,  move packet to queue
        */
        action[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
        action[0].conf = &queue;
        action[1].type = RTE_FLOW_ACTION_TYPE_END;

        /*
        * set the first level of the pattern (ETH).
        */
        pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;

        /*
        * setting the second level of the pattern (IP).
        */
        memset(&ip_spec, 0, sizeof(struct rte_flow_item_ipv4));
        memset(&ip_mask, 0, sizeof(struct rte_flow_item_ipv4));
        ip_spec.hdr.dst_addr = htonl(fl->dst_ip);
        ip_mask.hdr.dst_addr = htonl(fl->dst_ip_mask);
        ip_spec.hdr.src_addr = htonl(fl->src_ip);
        ip_mask.hdr.src_addr = htonl(fl->src_ip_mask);
        pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
        pattern[1].spec = &ip_spec;
        pattern[1].mask = &ip_mask;

        /*
        * setting the third level of the pattern (UDP).
        */
        memset(&udp_spec, 0, sizeof(struct rte_flow_item_udp));
        memset(&udp_mask, 0, sizeof(struct rte_flow_item_udp));
        udp_spec.hdr.dst_port = htons(fl->dst_port);
        udp_mask.hdr.dst_port = htons(fl->dst_port_mask);
        udp_spec.hdr.src_port = htons(fl->src_port);
        udp_mask.hdr.src_port = htons(fl->src_port_mask);
        pattern[2].type = RTE_FLOW_ITEM_TYPE_UDP;
        pattern[2].spec = &udp_spec;
        pattern[2].mask = &udp_mask;

        /* the final level must be always type end */
        pattern[3].type = RTE_FLOW_ITEM_TYPE_END;

        res = rte_flow_validate(port_id, &attr, pattern, action, &error);
        if (!res) {
retry:
            flow = rte_flow_create(port_id, &attr, pattern, action, &error);
            if (!flow) {
                rte_flow_flush(port_id, &error);
                goto retry;
            }

            uint32_t src_ip_masked, dst_ip_masked;
            uint16_t src_port_masked, dst_port_masked;

            src_ip_masked = fl->src_ip & fl->src_ip_mask;
            dst_ip_masked = fl->dst_ip & fl->dst_ip_mask;
            src_port_masked = fl->src_port & fl->src_port_mask;
            dst_port_masked = fl->dst_port & fl->dst_port_mask;
            printf("Create UDP flow from " IP_STRING ":%u(%x) to " IP_STRING ":%u(%x) by queue %d on port %d\n", \
                        HOST_IP_FMT(src_ip_masked), src_port_masked, src_port_masked, \
                        HOST_IP_FMT(dst_ip_masked), dst_port_masked, dst_port_masked, rx_q, port_id);
        } else {
            printf("Invalid flow rule! msg: %s\n", error.message);
        }
    }

	return 0;
}

int net_direct_flow_to_queue(uint16_t qid, uint16_t proto,
                uint32_t src_ip, uint32_t src_ip_mask, uint32_t dst_ip, uint32_t dst_ip_mask, 
                uint16_t src_port, uint16_t src_port_mask,  uint16_t dst_port, uint16_t dst_port_mask) {
    struct flow flow = {
        .src_ip = src_ip,
        .src_ip_mask = src_ip_mask,
        .dst_ip = dst_ip,
        .dst_ip_mask = dst_ip_mask,
        .src_port = src_port,
        .src_port_mask = src_port_mask,
        .dst_port = dst_port,
        .dst_port_mask = dst_port_mask,
    };
    int ret = -EINVAL;

    switch (proto) {
        case SOCK_STREAM:
            ret = net_create_tcp_flow(qid, &flow);
            break;
        case SOCK_DGRAM:
            ret = net_create_udp_flow(qid, &flow);
            break;
        default:
            printf("Unknown protocol %d!\n", proto);
            break;
    }

    return ret;
}
