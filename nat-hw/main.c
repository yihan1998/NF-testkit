#include <stdlib.h>
#include <sys/types.h>
#include <rte_malloc.h>
#include <rte_flow.h>
#include <rte_ethdev.h>
#include <rte_vxlan.h>
#include <rte_gtp.h>
#include <rte_gre.h>
#include <rte_geneve.h>
#include <rte_common.h>
#include <rte_eal.h>
#include <rte_flow.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_version.h>
#include <rte_errno.h>
#include <rte_ip.h>
#include <linux/if_ether.h>
#include <linux/udp.h>

#include "config.h"
#include "dns_filter.h"

#define NR_HAIRPIN  1

#define DEFAULT_PKT_BURST   32

#define NR_RXD  256
#define NR_TXD  512

#define DEFAULT_RX_DESC    4096
#define DEFAULT_TX_DESC    4096

#define GET_RSS_HF() (RTE_ETH_RSS_IP)

#define IPv4(a, b, c, d)    ((a) << 24) | ((b) << 16) | ((c) << 8) | (d)

enum layer_name {
	L2,
	// L3,
	// L4,
	END
};

static struct rte_flow_item pattern[] = {
	[L2] = { /* ETH type is set since we always start from ETH. */
		.type = RTE_FLOW_ITEM_TYPE_ETH,
		.spec = NULL,
		.mask = NULL,
		.last = NULL },
	// [L3] = {
	// 	.type = RTE_FLOW_ITEM_TYPE_VOID,
	// 	.spec = NULL,
	// 	.mask = NULL,
	// 	.last = NULL },
	// [L4] = {
	// 	.type = RTE_FLOW_ITEM_TYPE_VOID,
	// 	.spec = NULL,
	// 	.mask = NULL,
	// 	.last = NULL },
	[END] = {
		.type = RTE_FLOW_ITEM_TYPE_END,
		.spec = NULL,
		.mask = NULL,
		.last = NULL },
};


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

#define FULL_IP_MASK   0xffffffff /* full mask */
#define EMPTY_IP_MASK  0x0 /* empty mask */

#define FULL_PORT_MASK   0xffff /* full mask */
#define PART_PORT_MASK   0xff00 /* partial mask */
#define EMPTY_PORT_MASK  0x0 /* empty mask */

#define MAX_PATTERN_NUM		4
#define MAX_ACTION_NUM		4

struct rte_flow_action_set_mac src_mac = { .mac_addr = {0x3c, 0xec, 0xef, 0x57, 0x0a, 0x8c} };
struct rte_flow_action_set_mac dst_mac = { .mac_addr = {0x3c, 0xec, 0xef, 0x04, 0x96, 0x64} };

struct rte_flow_action_set_ipv4 src_ip;
struct rte_flow_action_set_ipv4 dst_ip;

struct rte_eth_hairpin_conf hairpin_conf = {
    .peer_count = 1,
    .manual_bind = 0,
    .tx_explicit = 0,
};

/* Packet mempool for each core */
struct rte_mempool * pkt_mempools[NR_CPUS];

static int config_ports(void) {
    int ret;
    uint16_t portid;
    char name[RTE_MEMZONE_NAMESIZE];
    uint16_t nb_rxd = DEFAULT_RX_DESC;
    uint16_t nb_txd = DEFAULT_TX_DESC;
	uint16_t nr_queues = NR_CPUS + NR_HAIRPIN;
    int hairpin_queue, std_queue;

    printf("RX/TX queue: %d, hairpin queue: %d\n", NR_CPUS, NR_HAIRPIN);

    for (int i = 0; i < NR_CPUS; i++) {
        /* Create mbuf pool for each core */
        sprintf(name, "mbuf_pool_%d", i);
        pkt_mempools[i] = rte_pktmbuf_pool_create(name, 8192,
            RTE_MEMPOOL_CACHE_MAX_SIZE, 0, 4300,
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
		ret = rte_eth_dev_configure(portid, nr_queues, nr_queues, &port_conf);
		if (ret < 0) {
			rte_exit(EXIT_FAILURE, "Cannot configure device: err=%d, port=%u\n", ret, portid);
        }
		/* >8 End of configuration of the number of queues for a port. */

        for (int i = 0; i < NR_CPUS; i++) {
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

        for (hairpin_queue = NR_CPUS, std_queue = 0; hairpin_queue < nr_queues; hairpin_queue++, std_queue++) {
            hairpin_conf.peers[0].port = portid;
            hairpin_conf.peers[0].queue = std_queue + NR_CPUS;
            ret = rte_eth_rx_hairpin_queue_setup(portid, hairpin_queue, NR_RXD, &hairpin_conf);
            if (ret != 0)
                rte_exit(EXIT_FAILURE, ":: Hairpin rx queue setup failed: err=%d, port=%u\n", ret, portid);
        }

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

void create_flow(int pid) {
	int ret, i, j;
	struct rte_flow_error error;
	struct rte_flow_attr attr;
	struct rte_flow_item pattern[MAX_PATTERN_NUM];
	struct rte_flow_action action[MAX_ACTION_NUM];
	struct rte_flow * flow = NULL;
    uint16_t queue[RTE_MAX_QUEUES_PER_PORT];
	struct rte_flow_item_ipv4 ip_spec;
	struct rte_flow_item_ipv4 ip_mask;
	struct rte_flow_item_udp udp_spec;
	struct rte_flow_item_udp udp_mask;
	int res;

    memset(pattern, 0, sizeof(pattern));
    memset(action, 0, sizeof(action));

    uint16_t queues[16];
	for (int i = 0; i < NR_HAIRPIN; i++) {
		queues[i] = 0;
	}

	struct rte_flow_action_rss rss = {
        .level = 0, /* RSS should be done on inner header. */
        .queue = queues, /* Set the selected target queues. */
        .queue_num = NR_HAIRPIN, /* The number of queues. */
        .types = GET_RSS_HF() 
    };

    /*
    * set the rule attribute.
    * in this case only ingress packets will be checked.
    */
    memset(&attr, 0, sizeof(struct rte_flow_attr));
    attr.ingress = 1;

    action[0].type = RTE_FLOW_ACTION_TYPE_SET_MAC_SRC;
	action[0].conf = &src_mac;
    action[1].type = RTE_FLOW_ACTION_TYPE_SET_MAC_DST;
	action[1].conf = &dst_mac;
    action[0].type = RTE_FLOW_ACTION_TYPE_SET_IPV4_SRC;
	action[0].conf = &src_ip;
    action[1].type = RTE_FLOW_ACTION_TYPE_SET_IPV4_DST;
	action[1].conf = &dst_ip;
    action[2].type = RTE_FLOW_ACTION_TYPE_RSS;
    action[2].conf = &rss;
    action[3].type = RTE_FLOW_ACTION_TYPE_END;

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
    udp_spec.hdr.dst_port = htons(53);
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

static void
print_mac_addresses(struct rte_mbuf *mbuf) {
    struct rte_ether_hdr *eth_hdr;

    eth_hdr = rte_pktmbuf_mtod(mbuf, struct rte_ether_hdr *);

    char src_mac[RTE_ETHER_ADDR_FMT_SIZE];
    char dst_mac[RTE_ETHER_ADDR_FMT_SIZE];

    rte_ether_format_addr(src_mac, RTE_ETHER_ADDR_FMT_SIZE, &eth_hdr->src_addr);
    rte_ether_format_addr(dst_mac, RTE_ETHER_ADDR_FMT_SIZE, &eth_hdr->dst_addr);

    printf("Source MAC: %s, Destination MAC: %s\n", src_mac, dst_mac);
}

static void handle_dns_packet(struct rte_mbuf * buf) {
    struct ethhdr * ethhdr;
    struct iphdr * iphdr;
    uint16_t iphdr_hlen;
    struct udphdr * udphdr;
	uint16_t ulen, len;
	uint8_t * data;
    uint8_t * pkt = rte_pktmbuf_mtod(buf, uint8_t *);

    ethhdr = (struct ethhdr *)pkt;

    iphdr = (struct iphdr *)&ethhdr[1];
    iphdr_hlen = iphdr->ihl;
    iphdr_hlen <<= 2;

    udphdr = (struct udphdr *)((uint8_t *)iphdr + iphdr_hlen);
    ulen = ntohs(udphdr->len);
	len = ulen - sizeof(struct udphdr);
	data = (uint8_t *)udphdr + sizeof(struct udphdr);

    return parse_dns_query(data, len);
}

static void process_packet(struct rte_mbuf * buf) {
    if (buf->ol_flags & RTE_MBUF_F_RX_FDIR) {
        uint32_t mark = buf->hash.fdir.hi;
        printf("Packet marked with value: %u\n", mark);
        if (mark) {
            return handle_dns_packet(buf);
        }
    }
}

int main(int argc, char ** argv) {
    int ret;
    int pid;
    int nb_rx = 0, nb_tx = 0;
    struct rte_mbuf * pkts_burst[DEFAULT_PKT_BURST];

    if ((ret = rte_eal_init(argc, argv)) < 0) {
		rte_exit(EXIT_FAILURE, "Cannot init EAL: %s\n", rte_strerror(rte_errno));
    }

    argc -= ret;
	argv += ret;

    config_ports();

    load_regex_rules();

    RTE_ETH_FOREACH_DEV(pid) {
        create_flow(pid);
    }

    src_ip.ipv4_addr = rte_cpu_to_be_32(IPv4(10, 10, 0, 2));
    dst_ip.ipv4_addr = rte_cpu_to_be_32(IPv4(8, 8, 8, 8));

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

        nb_rx = rte_eth_rx_burst(0, 0, pkts_burst, DEFAULT_PKT_BURST);
        if (nb_rx) {
            printf("Receive %d packets\n", nb_rx);
            for (int i = 0; i < nb_rx; i++) {
                process_packet(pkts_burst[i]);
            }
            nb_tx = rte_eth_tx_burst(0, qid, rx_pkts, nb_rx);
            if (unlikely(nb_tx < nb_rx)) {
                do {
                    rte_pktmbuf_free(rx_pkts[nb_tx]);
                } while (++nb_tx < nb_rx);
            }
        }
    }

    return 0;
}