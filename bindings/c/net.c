#include "net.h"

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

#define MAX_NR_CORES        16
#define DEFAULT_PKT_BURST   32
#define DEFAULT_RX_DESC     1024
#define DEFAULT_TX_DESC     1024

struct rte_mempool * mempools[MAX_NR_CORES];

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
    return 0;
}

uint8_t * net_get_txpkt(int pkt_len) {
    return NULL;
}
