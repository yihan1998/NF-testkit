
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

#ifndef CONFIG_NR_CPUS
#define CONFIG_NR_CPUS  1
#endif

#define NR_CPUS CONFIG_NR_CPUS

#define USEC_PER_SEC    1000000L
#define TIMEVAL_TO_USEC(t)  ((t.tv_sec * USEC_PER_SEC) + t.tv_usec)

#define DEFAULT_PKT_BURST   32
#define DEFAULT_RX_DESC     1024
#define DEFAULT_TX_DESC     1024

int nb_cores;

#define MAX_RULES 100
#define MAX_REGEX_LENGTH 256

/* DOCA compatible */
regex_t compiled_rules[MAX_RULES];
int rule_count = 0;

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
    int ret, nb_mbufs;
    uint16_t portid;
    char name[RTE_MEMZONE_NAMESIZE];
    uint16_t nb_rxd = DEFAULT_RX_DESC;
    uint16_t nb_txd = DEFAULT_TX_DESC;

    nb_mbufs = RTE_MAX(nb_rxd + nb_txd + DEFAULT_PKT_BURST + RTE_MEMPOOL_CACHE_MAX_SIZE, 8192U);

    for (int i = 0; i < nb_cores; i++) {
        /* Create mbuf pool for each core */
        sprintf(name, "mbuf_pool_%d", i);
        pkt_mempools[i] = rte_pktmbuf_pool_create(name, nb_mbufs,
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

int load_regex_rules(void) {
    FILE *file = fopen("/local/yihan/NF-testkit/dns-filter/dns_filter_rules.txt", "r");
    if (!file) {
        perror("Failed to open file");
        return -1;
    }

    char regex[MAX_REGEX_LENGTH];
    int ret;
    while (fgets(regex, MAX_REGEX_LENGTH, file)) {
        if (regex[strlen(regex) - 1] == '\n') {
            regex[strlen(regex) - 1] = '\0';  // Remove newline character
        }

        ret = regcomp(&compiled_rules[rule_count], regex, REG_EXTENDED);
        if (ret) {
            fprintf(stderr, "Could not compile regex: %s\n", regex);
            continue;
        }
        rule_count++;
        if (rule_count >= MAX_RULES) break;
    }

    fclose(file);
    return 0;
}

struct dns_header {
    uint16_t id; // Transaction ID
    uint16_t flags; // DNS flags
    uint16_t qdcount; // Number of questions
    uint16_t ancount; // Number of answers
    uint16_t nscount; // Number of authority records
    uint16_t arcount; // Number of additional records
};

int find_matching_rule(const char * domain_name) {
    int result;

    // Iterate through all compiled rules
    for (int i = 0; i < rule_count; i++) {
        result = regexec(&compiled_rules[i], domain_name, 0, NULL, 0);
        if (result == 0) {
            // printf("Match found with rule %d: %s\n", i, domain_name);
            return i;  // Return the index of the first matching rule
        }
    }

    // printf("No match found for: %s\n", domain_name);
    return -1;  // Return -1 if no match is found
}

// Function to print a domain name from a DNS query
void print_domain_name(const unsigned char * buffer, int* position, unsigned char * domain_name) {
    int len = buffer[(*position)++];
    while (len > 0) {
        for (int i = 0; i < len; i++) {
            *(domain_name++) = buffer[(*position)++];
        }
        len = buffer[(*position)++];
        if (len > 0) {
            *(domain_name++) = '.';
        }
    }
}

// Parse and print details from a DNS query
int parse_dns_query(const unsigned char * buffer, int size) {
    unsigned char domain_name[256] = {0};
    // Cast the buffer to the DNS header struct
    // struct dns_header* dns = (struct dns_header*)buffer;
    int position = sizeof(struct dns_header); // Position in the buffer
    print_domain_name(buffer, &position, domain_name);
    if (find_matching_rule((const char *)domain_name) < 0) {
        return -1;
    }
    return 0;
}

int process_packets(struct rte_mbuf ** pkts_burst, int nb_rx) {
    struct rte_mbuf * pkt;
    char * p;
	struct ethhdr * ethhdr;
    struct iphdr * iphdr;
    uint16_t iphdr_hlen;
    uint8_t iphdr_protocol;
    struct udphdr * udphdr;
    uint16_t ulen, len;
	uint8_t * data;

    for (int i = 0; i < nb_rx; i++) {
        pkt = pkts_burst[i];
	    p = rte_pktmbuf_mtod(pkt, char *);

        ethhdr = (struct ethhdr *)p;

        iphdr = (struct iphdr *)&ethhdr[1];
        iphdr_hlen = iphdr->ihl;
        iphdr_hlen <<= 2;
        iphdr_protocol = iphdr->protocol;

        if (iphdr_protocol != IPPROTO_UDP) continue;

        udphdr = (struct udphdr *)((uint8_t *)iphdr + iphdr_hlen);
        if (ntohs(udphdr->dest) != 53) continue;

        ulen = ntohs(udphdr->len);
        len = ulen - sizeof(struct udphdr);
        data = (uint8_t *)udphdr + sizeof(struct udphdr);

        parse_dns_query(data, len);
    }

    return 0;
}

static int launch_one_lcore(void * args) {
    int nb_rx, nb_tx;
	uint32_t lid, qid;
    struct rte_mbuf * pkts_burst[DEFAULT_PKT_BURST];
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

        nb_rx = rte_eth_rx_burst(0, qid, pkts_burst, DEFAULT_PKT_BURST);
        if (nb_rx) {
            sec_nb_rx += nb_rx;

            process_packets(pkts_burst, nb_rx);

            nb_tx = rte_eth_tx_burst(0, qid, pkts_burst, nb_rx);
            sec_nb_tx += nb_tx;
            if (unlikely(nb_tx < nb_rx)) {
                do {
                    rte_pktmbuf_free(pkts_burst[nb_tx]);
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

    load_regex_rules();

    rte_eal_mp_remote_launch(launch_one_lcore, NULL, CALL_MAIN);
    rte_eal_mp_wait_lcore();

    /* clean up the EAL */
	rte_eal_cleanup();
	printf("Bye...\n");

    return 0;
}