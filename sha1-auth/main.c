
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
#include "doca.h"
#include "skbuff.h"
#include "ethernet.h"
#include "sha1_auth.h"

#define USEC_PER_SEC    1000000L
#define TIMEVAL_TO_USEC(t)  ((t.tv_sec * USEC_PER_SEC) + t.tv_usec)

#define DEFAULT_PKT_BURST   32
#define DEFAULT_RX_DESC     4096
#define DEFAULT_TX_DESC     4096

int nb_cores;

#define NR_VERIFY   6

__thread struct rte_ring * fwd_queue;

__thread EVP_MD_CTX * mdctx;

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
#define	JUMBO_FRAME_MAX_SIZE	3600

/* Port configuration */
struct rte_eth_conf port_conf = {
    .rxmode = {
        .mtu = JUMBO_FRAME_MAX_SIZE - RTE_ETHER_HDR_LEN -
			    RTE_ETHER_CRC_LEN,
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

static int eth_dev_set_mtu_mp(uint16_t port_id, uint16_t mtu) {
    return rte_eth_dev_set_mtu(port_id, mtu);
}

static int config_ports(void) {
    int ret;
    uint16_t portid;
    char name[RTE_MEMZONE_NAMESIZE];
    uint16_t nb_rxd = DEFAULT_RX_DESC;
    uint16_t nb_txd = DEFAULT_TX_DESC;
	uint16_t mtu, new_mtu;

    for (int i = 0; i < nb_cores; i++) {
        /* Create mbuf pool for each core */
        sprintf(name, "mbuf_pool_%d", i);
        pkt_mempools[i] = rte_pktmbuf_pool_create(name, 8192,
            RTE_MEMPOOL_CACHE_MAX_SIZE, 0, 3800,
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

        if (rte_eth_dev_get_mtu(portid, &mtu) != 0) {
            printf("Failed to get MTU for port %u\n", portid);
            return -1;
        }

        new_mtu = JUMBO_FRAME_MAX_SIZE - RTE_ETHER_HDR_LEN + RTE_ETHER_CRC_LEN;

        if (eth_dev_set_mtu_mp(portid, new_mtu) != 0) {
            fprintf(stderr, "Failed to set MTU to %u for port %u\n", new_mtu, portid);
            return -1;
        }

        port_conf.rxmode.mtu = new_mtu;

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

#define FULL_IP_MASK   0xffffffff /* full mask */
#define EMPTY_IP_MASK  0x0 /* empty mask */

#define FULL_PORT_MASK   0xffff /* full mask */
#define PART_PORT_MASK   0xff00 /* partial mask */
#define EMPTY_PORT_MASK  0x0 /* empty mask */

#define MAX_PATTERN_NUM		4
#define MAX_ACTION_NUM		2

void create_flow(int pid, uint16_t start_core, uint16_t nb_queue) {
	int ret, i, j;
	struct rte_flow_error error;
	struct rte_flow_attr attr;
	struct rte_flow_item pattern[MAX_PATTERN_NUM];
	struct rte_flow_action action[MAX_ACTION_NUM];
	struct rte_flow * flow = NULL;
    struct rte_flow_action_rss action_rss;
    uint16_t queue[RTE_MAX_QUEUES_PER_PORT];
	struct rte_flow_item_ipv4 ip_spec;
	struct rte_flow_item_ipv4 ip_mask;
	struct rte_flow_item_udp udp_spec;
	struct rte_flow_item_udp udp_mask;
	int res;
    uint8_t rss_key[64];
    struct rte_eth_rss_conf rss_conf = {
        .rss_key = rss_key,
        .rss_key_len = sizeof(rss_key),
    };

    memset(pattern, 0, sizeof(pattern));
    memset(action, 0, sizeof(action));

    /*
    * set the rule attribute.
    * in this case only ingress packets will be checked.
    */
    memset(&attr, 0, sizeof(struct rte_flow_attr));
    attr.ingress = 1;
    attr.priority = 0;

    /*
    * create the action sequence.
    * one action only,  move packet to queue
    */

    for (i = 0, j = 0; i < nb_queue; ++i)
        queue[j++] = i;

    ret = rte_eth_dev_rss_hash_conf_get(pid, &rss_conf);
    action_rss = (struct rte_flow_action_rss){
            .types = rss_conf.rss_hf,
            .key_len = rss_conf.rss_key_len,
            .queue_num = nb_queue,
            .key = rss_key,
            .queue = queue,
    };

    action[0].type = RTE_FLOW_ACTION_TYPE_RSS;
    action[0].conf = &action_rss;
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
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
    pattern[1].spec = &ip_spec;
    pattern[1].mask = &ip_mask;

    /*
    * setting the third level of the pattern (UDP).
    */
    memset(&udp_spec, 0, sizeof(struct rte_flow_item_udp));
    memset(&udp_mask, 0, sizeof(struct rte_flow_item_udp));
    udp_spec.hdr.dst_port = htons(4321);
    udp_mask.hdr.dst_port = htons(FULL_PORT_MASK);
    pattern[2].type = RTE_FLOW_ITEM_TYPE_UDP;
    pattern[2].spec = &udp_spec;
    pattern[2].mask = &udp_mask;

    /* the final level must be always type end */
    pattern[3].type = RTE_FLOW_ITEM_TYPE_END;

    res = rte_flow_validate(pid, &attr, pattern, action, &error);
    if (!res) {
retry:
        flow = rte_flow_create(pid, &attr, pattern, action, &error);
        if (!flow) {
            rte_flow_flush(pid, &error);
            goto retry;
        }
    } else {
        printf("control: invalid flow rule! msg: %s\n", error.message);
    }
}

int install_flow_rule() {
    uint16_t portid;
	RTE_ETH_FOREACH_DEV(portid) {
        create_flow(portid, rte_get_main_lcore(), NR_VERIFY);
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

static int launch_one_lcore(void * args) {
    int nb_rx, nb_tx;
	uint32_t lid, qid, core_id;
    struct rte_mbuf * rx_pkts[DEFAULT_PKT_BURST];
    struct timeval log, curr;
    uint32_t sec_nb_rx, sec_nb_tx;
    char name[RTE_MEMZONE_NAMESIZE];

    lid = rte_lcore_id();
    core_id = lid - rte_get_main_lcore();
    qid = lid - rte_get_main_lcore();
    ctx = &worker_ctx[core_id];

    sprintf(name, "skb_pool_%d", core_id);
    skb_mp = rte_mempool_lookup(name);
    assert(skb_mp != NULL);

    sprintf(name, "fwd_queue_%d", core_id);
    fwd_queue = rte_ring_create(name, 4096, rte_socket_id(), 0);
    assert(fwd_queue != NULL);

    if((mdctx = EVP_MD_CTX_new()) == NULL) {
        printf("Failed to init EVP context\n");
    }

#ifdef CONFIG_DOCA
    doca_percore_init();
#endif  /* CONFIG_DOCA */

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

    // install_flow_rule();

    skb_init(nb_cores);

#ifdef CONFIG_DOCA
    doca_init();
#endif  /* CONFIG_DOCA */

#ifdef CONFIG_DOCA
    for (int i = 0; i < nb_cores; i++) {
        struct worker_context * ctx = &worker_ctx[i];
        doca_worker_init(ctx);
    }
#endif  /* CONFIG_DOCA */

    rte_eal_mp_remote_launch(launch_one_lcore, NULL, CALL_MAIN);
    rte_eal_mp_wait_lcore();

    /* clean up the EAL */
	rte_eal_cleanup();
	printf("Bye...\n");

    return 0;
}
