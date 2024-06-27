
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/queue.h>
#include <sys/time.h>
#include <setjmp.h>
#include <stdarg.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>
#include <regex.h>
#include <linux/if_ether.h>
#include <linux/udp.h>

#include <rte_common.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_eal.h>
#include <rte_launch.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_string_fns.h>
#include <rte_ip.h>

#include "config.h"

#define USEC_PER_SEC    1000000L
#define TIMEVAL_TO_USEC(t)  ((t.tv_sec * USEC_PER_SEC) + t.tv_usec)

#define DEFAULT_PKT_BURST   32
#define DEFAULT_RX_DESC     4096
#define DEFAULT_TX_DESC     4096

int nb_cores;

__thread struct rte_ring * fwd_queue;

/* RX queue configuration */
static struct rte_eth_rxconf rx_conf = {
    .rx_thresh = {
        .pthresh = 8,
        .hthresh = 8,
        .wthresh = 4,
    },
    .rx_free_thresh = 32,
};

/* TX queue configuration */
static struct rte_eth_txconf tx_conf = {
    .tx_thresh = {
        .pthresh = 36,
        .hthresh = 0,
        .wthresh = 0,
    },
    .tx_free_thresh = 0,
};

/* Port configuration */
struct rte_eth_conf port_conf = {
    .rxmode = {
        .mq_mode = RTE_ETH_MQ_RX_RSS,
    },
    .txmode = {
        .mq_mode = RTE_ETH_MQ_TX_NONE,
        .offloads = (RTE_ETH_TX_OFFLOAD_IPV4_CKSUM |
                RTE_ETH_TX_OFFLOAD_UDP_CKSUM |
                RTE_ETH_TX_OFFLOAD_TCP_CKSUM),
    },
    .rx_adv_conf = {
        .rss_conf = {
            .rss_key = NULL,
            .rss_hf =
                RTE_ETH_RSS_IP | RTE_ETH_RSS_TCP | RTE_ETH_RSS_UDP,
        },
    },
};

/* Packet mempool for each core */
struct rte_mempool * pkt_mempools[NR_CPUS];

static int config_ports(void) {
    int ret;
    uint16_t portid;
    char name[RTE_MEMZONE_NAMESIZE];
    uint16_t nb_rxd = DEFAULT_RX_DESC;
    uint16_t nb_txd = DEFAULT_TX_DESC;

    for (int i = 0; i < nb_cores; i++) {
        /* Create mbuf pool for each core */
        sprintf(name, "mbuf_pool_%d", i);
        pkt_mempools[i] = rte_pktmbuf_pool_create(name, 8192,
            RTE_MEMPOOL_CACHE_MAX_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE,
            rte_socket_id());
        if (pkt_mempools[i] == NULL) {
            rte_exit(EXIT_FAILURE, "Cannot init mbuf pool\n");
        } else {
            printf("MBUF pool %u: %p...\n", i, pkt_mempools[i]);
        }
    }

    /* Initialise each port */
	RTE_ETH_FOREACH_DEV(portid) {
		struct rte_eth_dev_info dev_info;

        printf("Initializing port %u...", portid);
		fflush(stdout);

		ret = rte_eth_dev_info_get(portid, &dev_info);
		if (ret != 0) {
			rte_exit(EXIT_FAILURE, "Error during getting device (port %u) info: %s\n", portid, strerror(-ret));
        }

		/* Configure the number of queues for a port. */
		ret = rte_eth_dev_configure(portid, nb_cores, nb_cores, &port_conf);
		if (ret < 0) {
			rte_exit(EXIT_FAILURE, "Cannot configure device: err=%d, port=%u\n", ret, portid);
        }
		/* >8 End of configuration of the number of queues for a port. */

        for (int i = 0; i < nb_cores; i++) {
            /* RX queue setup. 8< */
            ret = rte_eth_rx_queue_setup(portid, i, nb_rxd, rte_eth_dev_socket_id(portid), &rx_conf, pkt_mempools[i]);
            if (ret < 0) {
                rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup:err=%d, port=%u\n", ret, portid);
            }

            ret = rte_eth_tx_queue_setup(portid, i, nb_txd, rte_eth_dev_socket_id(portid), &tx_conf);
            if (ret < 0) {
                rte_exit(EXIT_FAILURE, "rte_eth_tx_queue_setup:err=%d, port=%u\n", ret, portid);
            }
        }
		/* >8 End of queue setup. */

		fflush(stdout);

		/* Start device */
		ret = rte_eth_dev_start(portid);
		if (ret < 0) {
			rte_exit(EXIT_FAILURE, "rte_eth_dev_start:err=%d, port=%u\n", ret, portid);
        }

		printf("done\n");
        ret = rte_eth_promiscuous_enable(portid);
        if (ret != 0) {
            rte_exit(EXIT_FAILURE, "rte_eth_promiscuous_enable:err=%s, port=%u\n", rte_strerror(-ret), portid);
        }
    }

    return 0;
}

static int parse_args(int argc, char ** argv) {
	int opt, option_index;
	static struct option lgopts[] = {
		{"crc-strip", 0, 0, 0},
		{NULL, 0, 0, 0}
	};

	while ((opt = getopt_long(argc, argv, "c:h", lgopts, &option_index)) != EOF) {
        switch (opt) {
		case 'c':	/* Number of cores */
            nb_cores = strtol(optarg, NULL, 10);
			break;

		case 'h':	/* print out the help message */
			// pktgen_usage(prgname);
			return -1;

		case 0:	/* crc-strip for all ports */
			break;
		default:
			return -1;
		}
    }
		
    return 0;
}

int ethernet_input(struct rte_mbuf * m, uint8_t * pkt, int pkt_size) {
    struct ethhdr * ethhdr;
    unsigned char tmp;

    ethhdr = (struct ethhdr *)pkt;

    for (int i = 0; i < ETH_ALEN; i++) {
        tmp = ethhdr->h_dest[i];
        ethhdr->h_dest[i] = ethhdr->h_source[i];
        ethhdr->h_source[i] = tmp;
    }

    return 0;
}

static int launch_one_lcore(void * args) {
    int nb_rx, nb_tx;
	uint32_t lid, qid;
    struct rte_mbuf * rx_pkts[DEFAULT_PKT_BURST];
    struct timeval log, curr;
    uint32_t sec_nb_rx, sec_nb_tx;

    lid = rte_lcore_id();
    qid = lid;

    sec_nb_rx = sec_nb_tx = 0;

    gettimeofday(&log, NULL);

    while (1) {
        gettimeofday(&curr, NULL);
        if (curr.tv_sec - log.tv_sec >= 1) {
            printf("CPU %02d| rx: %u/%4.2f (MPS), tx: %u/%4.2f (MPS)\n", rte_lcore_id(), 
                sec_nb_rx, ((double)sec_nb_rx) / (TIMEVAL_TO_USEC(curr) - TIMEVAL_TO_USEC(log)),
                sec_nb_tx, ((double)sec_nb_tx) / (TIMEVAL_TO_USEC(curr) - TIMEVAL_TO_USEC(log)));
            sec_nb_rx = sec_nb_tx = 0;
            gettimeofday(&log, NULL);
        }
        nb_rx = rte_eth_rx_burst(0, qid, rx_pkts, DEFAULT_PKT_BURST);
        if (nb_rx) {
            sec_nb_rx += nb_rx;

            for (int i = 0; i < nb_rx; i++) {
                struct rte_mbuf * rx_pkt = rx_pkts[i];
                int pkt_size = rx_pkt->pkt_len;
                uint8_t * pkt = rte_pktmbuf_mtod(rx_pkt, uint8_t *);
                ethernet_input(rx_pkt, pkt, pkt_size);
            }

            nb_tx = rte_eth_tx_burst(0, qid, rx_pkts, nb_rx);
            sec_nb_tx += nb_tx;
            if (unlikely(nb_tx < nb_rx)) {
                do {
                    rte_pktmbuf_free(rx_pkts[nb_tx]);
                } while (++nb_tx < nb_rx);
            }
        }

    }
    return 0;
}

int main(int argc, char ** argv) {
	int ret;
    if ((ret = rte_eal_init(argc, argv)) < 0) {
		rte_exit(EXIT_FAILURE, "Cannot init EAL: %s\n", rte_strerror(rte_errno));
    }

    argc -= ret;
	argv += ret;

	ret = parse_args(argc, argv);

    config_ports();

    rte_eal_mp_remote_launch(launch_one_lcore, NULL, CALL_MAIN);
    rte_eal_mp_wait_lcore();

    /* clean up the EAL */
	rte_eal_cleanup();
	printf("Bye...\n");

    return 0;
}
