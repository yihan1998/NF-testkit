#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

#include "dpdk.h"

#define USEC_PER_SEC    1000000L
#define TIMEVAL_TO_USEC(t)  ((t.tv_sec * USEC_PER_SEC) + t.tv_usec)

#define DEFAULT_PKT_BURST   32
#define DEFAULT_RX_DESC     4096
#define DEFAULT_TX_DESC     4096

int nb_cores;

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
struct rte_mempool * pkt_mempools[8];

void print_ether_hdr(struct rte_ether_hdr * ethhdr) {
    printf("Ethernet Header:\n");
    printf("  Destination MAC " RTE_ETHER_ADDR_PRT_FMT "\n", RTE_ETHER_ADDR_BYTES(&ethhdr->dst_addr));
    printf("  Source MAC " RTE_ETHER_ADDR_PRT_FMT "\n", RTE_ETHER_ADDR_BYTES(&ethhdr->src_addr));
    printf("  EtherType: 0x%04x\n", ntohs(ethhdr->ether_type));
}

void print_ipv4(struct rte_ipv4_hdr * iphdr) {
    char src_ip[INET_ADDRSTRLEN];
    char dst_ip[INET_ADDRSTRLEN];
    struct in_addr src_addr = { .s_addr = iphdr->src_addr };
    struct in_addr dst_addr = { .s_addr = iphdr->dst_addr };

    inet_ntop(AF_INET, &src_addr, src_ip, sizeof(src_ip));
    inet_ntop(AF_INET, &dst_addr, dst_ip, sizeof(dst_ip));

    printf("IP Header:\n");
    printf("  Version: %u\n", iphdr->ihl);
    printf("  Header Length: %u\n", iphdr->version);
    printf("  Type of Service: %d\n", iphdr->type_of_service);
    printf("  Total Length: %d\n", ntohs(iphdr->total_length));
    printf("  Identification: %d\n", ntohs(iphdr->packet_id));
    printf("  Fragment Offset: %d\n", ntohs(iphdr->fragment_offset));
    printf("  Time to Live: %d\n", iphdr->time_to_live);
    printf("  Protocol: %d\n", iphdr->next_proto_id);
    printf("  Header Checksum: 0x%04x\n", ntohs(iphdr->hdr_checksum));
    printf("  Source IP: %s\n", src_ip);
    printf("  Destination IP: %s\n", dst_ip);
}

void print_tcp_header(struct rte_tcp_hdr * tcphdr) {
    printf("TCP Header:\n");
    printf("  Source Port: %d\n", ntohs(tcphdr->src_port));
    printf("  Destination Port: %d\n", ntohs(tcphdr->dst_port));
    printf("  Sequence Number: %u\n", ntohl(tcphdr->sent_seq));
    printf("  Acknowledgment Number: %u\n", ntohl(tcphdr->recv_ack));
    printf("  Data Offset: %d\n", tcphdr->data_off >> 4);
    printf("  Flags: 0x%02x\n", tcphdr->tcp_flags);
    printf("  Window: %d\n", ntohs(tcphdr->rx_win));
    printf("  Checksum: 0x%04x\n", ntohs(tcphdr->cksum));
    printf("  Urgent Pointer: %d\n", ntohs(tcphdr->tcp_urp));
}

void print_udp_header(struct rte_udp_hdr * udphdr) {
    printf("UDP Header:\n");
    printf("  Source Port: %d\n", ntohs(udphdr->src_port));
    printf("  Destination Port: %d\n", ntohs(udphdr->dst_port));
    printf("  Length: %d\n", ntohs(udphdr->dgram_len));
    printf("  Checksum: 0x%04x\n", ntohs(udphdr->dgram_cksum));
}

static doca_error_t
setup_hairpin_queues(uint16_t port_id, uint16_t peer_port_id, uint16_t *reserved_hairpin_q_list, int hairpin_queue_len)
{
	/* Port:
	 *	0. RX queue
	 *	1. RX hairpin queue rte_eth_rx_hairpin_queue_setup
	 *	2. TX hairpin queue rte_eth_tx_hairpin_queue_setup
	 */

	int result = 0, hairpin_q;
	uint16_t nb_tx_rx_desc = 2048;
	uint32_t manual = 1;
	uint32_t tx_exp = 1;
	struct rte_eth_hairpin_conf hairpin_conf = {
	    .peer_count = 1,
	    .manual_bind = !!manual,
	    .tx_explicit = !!tx_exp,
	    .peers[0] = {peer_port_id, 0},
	};

	for (hairpin_q = 0; hairpin_q < hairpin_queue_len; hairpin_q++) {
		// TX
		hairpin_conf.peers[0].queue = reserved_hairpin_q_list[hairpin_q];
		result = rte_eth_tx_hairpin_queue_setup(port_id, reserved_hairpin_q_list[hairpin_q], nb_tx_rx_desc,
						     &hairpin_conf);
		if (result < 0) {
			printf("Failed to setup hairpin queues (%s)\n", doca_error_get_descr(result));
			return DOCA_ERROR_DRIVER;
		}

		// RX
		hairpin_conf.peers[0].queue = reserved_hairpin_q_list[hairpin_q];
		result = rte_eth_rx_hairpin_queue_setup(port_id, reserved_hairpin_q_list[hairpin_q], nb_tx_rx_desc,
						     &hairpin_conf);
		if (result < 0) {
			printf("Failed to setup hairpin queues (%s)\n", doca_error_get_descr(result));
			return DOCA_ERROR_DRIVER;
		}
	}
	return DOCA_SUCCESS;
}

int dpdk_ports_init(struct application_dpdk_config *app_config) {
    int ret;
	doca_error_t result;
    uint16_t portid;
    char name[RTE_MEMZONE_NAMESIZE];
    uint16_t nb_rxd = DEFAULT_RX_DESC;
    uint16_t nb_txd = DEFAULT_TX_DESC;
    const uint16_t rx_rings = app_config->port_config.nb_queues;
	const uint16_t tx_rings = app_config->port_config.nb_queues;
	const uint16_t nb_hairpin_queues = app_config->port_config.nb_hairpin_q;
	uint16_t rss_queue_list[nb_hairpin_queues];
	uint16_t queue_index;

    nb_cores = rte_lcore_count();
	printf("nr cores: %d, nr RX queues: %d, nr TX queues: %d\n", nb_cores, rx_rings, tx_rings);

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

    if (app_config->port_config.enable_mbuf_metadata) {
		ret = rte_flow_dynf_metadata_register();
		if (ret < 0) {
			printf("Metadata register failed, ret=%d", ret);
			return DOCA_ERROR_DRIVER;
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
		ret = rte_eth_dev_configure(portid, rx_rings + nb_hairpin_queues, tx_rings + nb_hairpin_queues, &port_conf);
		if (ret < 0) {
			rte_exit(EXIT_FAILURE, "Cannot configure device: err=%d, port=%u\n", ret, portid);
        }
		/* >8 End of configuration of the number of queues for a port. */

        for (int i = 0; i < rx_rings; i++) {
            /* RX queue setup. 8< */
            ret = rte_eth_rx_queue_setup(portid, i, nb_rxd, rte_eth_dev_socket_id(portid), &rx_conf, pkt_mempools[i]);
            if (ret < 0) {
                rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup:err=%d, port=%u\n", ret, portid);
            }
        }

        for (int i = 0; i < tx_rings; i++) {
            ret = rte_eth_tx_queue_setup(portid, i, nb_txd, rte_eth_dev_socket_id(portid), &tx_conf);
            if (ret < 0) {
                rte_exit(EXIT_FAILURE, "rte_eth_tx_queue_setup:err=%d, port=%u\n", ret, portid);
            }
        }
		/* >8 End of queue setup. */

		fflush(stdout);

        /* Enabled hairpin queue before port start */
        if (nb_hairpin_queues && app_config->port_config.self_hairpin && rte_eth_dev_is_valid_port(portid ^ 1)) {
            /* Hairpin to both self and peer */
            assert((nb_hairpin_queues % 2) == 0);
            for (queue_index = 0; queue_index < nb_hairpin_queues / 2; queue_index++)
                rss_queue_list[queue_index] = app_config->port_config.nb_queues + queue_index * 2;
            result = setup_hairpin_queues(portid, portid, rss_queue_list, nb_hairpin_queues / 2);
            if (result != DOCA_SUCCESS) {
                printf("Cannot hairpin self port %u, ret: %s\n",
                        portid, doca_error_get_descr(result));
                return result;
            }
            for (queue_index = 0; queue_index < nb_hairpin_queues / 2; queue_index++)
                rss_queue_list[queue_index] = app_config->port_config.nb_queues + queue_index * 2 + 1;
            result = setup_hairpin_queues(portid, portid ^ 1, rss_queue_list, nb_hairpin_queues / 2);
            if (result != DOCA_SUCCESS) {
                printf("Cannot hairpin peer port %u, ret: %s\n",
                        portid ^ 1, doca_error_get_descr(result));
                return result;
            }
        } else if (nb_hairpin_queues) {
            /* Hairpin to self or peer */
            for (queue_index = 0; queue_index < nb_hairpin_queues; queue_index++)
                rss_queue_list[queue_index] = app_config->port_config.nb_queues + queue_index;
            if (rte_eth_dev_is_valid_port(portid ^ 1))
                result = setup_hairpin_queues(portid, portid ^ 1, rss_queue_list, nb_hairpin_queues);
            else
                result = setup_hairpin_queues(portid, portid, rss_queue_list, nb_hairpin_queues);
            if (result != DOCA_SUCCESS) {
                printf("Cannot hairpin port %u, ret=%d\n", portid, result);
                return result;
            }
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

int ddos_detect(struct rte_mbuf * m) {

}

int launch_one_lcore(void * args) {
    int portid;
    struct rte_mbuf * rx_pkts[DEFAULT_PKT_BURST];
    int nb_rx, nb_tx;
	int lid = rte_lcore_id();
	int qid = lid;

	while(!force_quit) {
		if (lid == 0)
			flexio_msg_stream_flush(default_stream);

		RTE_ETH_FOREACH_DEV(portid) {
			nb_rx = rte_eth_rx_burst(portid, qid, rx_pkts, DEFAULT_PKT_BURST);
			if (nb_rx) {
				printf("Receive %d packets\n", nb_rx);
				nb_tx = rte_eth_tx_burst(portid, qid, rx_pkts, nb_rx);
				for (int i = 0; i < nb_rx; i++) {
                    struct rte_mbuf * m = rx_pkts[i];
                    uint8_t * pkt = rte_pktmbuf_mtod(m, uint8_t *);
					struct rte_ether_hdr * ethhdr = (struct rte_ether_hdr *)pkt;
                    struct rte_ipv4_hdr * iphdr = (struct rte_ipv4_hdr *)&ethhdr[1];
				// 	print_ether_hdr(ethhdr);
                //     print_ipv4(iphdr);
					if (iphdr->next_proto_id == IPPROTO_TCP) {
                        struct rte_tcp_hdr * tcphdr = (struct rte_tcp_hdr *)&iphdr[1];
                        print_tcp_header(tcphdr);
						if (rte_flow_dynf_metadata_avail() && *RTE_FLOW_DYNF_METADATA(m) == 4) {
							ddos_detect(m);
						}
                    } else if (iphdr->next_proto_id == IPPROTO_UDP) {
                        struct rte_udp_hdr * udphdr = (struct rte_udp_hdr *)&iphdr[1];
                        print_udp_header(udphdr);
                    }
				}
				printf("Send %d packets\n", nb_tx);
				if (unlikely(nb_tx < nb_rx)) {
					do {
						rte_pktmbuf_free(rx_pkts[nb_tx]);
					} while (++nb_tx < nb_rx);
				}
			}
		}
	}

    return 0;
}

/*
 * Unbind port from all its peer ports
 *
 * @port_id [in]: port ID
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
unbind_hairpin_queues(uint16_t port_id)
{
	/* Configure the Rx and Tx hairpin queues for the selected port */
	int result = 0, peer_port, peer_ports_len;
	uint16_t peer_ports[RTE_MAX_ETHPORTS];

	/* unbind current Tx from all peer Rx */
	peer_ports_len = rte_eth_hairpin_get_peer_ports(port_id, peer_ports, RTE_MAX_ETHPORTS, 1);
	if (peer_ports_len < 0) {
		printf("Failed to get hairpin peer Tx ports of port %d, (%d)\n", port_id, peer_ports_len);
		return DOCA_ERROR_DRIVER;
	}

	for (peer_port = 0; peer_port < peer_ports_len; peer_port++) {
		result = rte_eth_hairpin_unbind(port_id, peer_ports[peer_port]);
		if (result < 0) {
			printf("Failed to bind hairpin queues (%d)\n", result);
			return DOCA_ERROR_DRIVER;
		}
	}
	/* unbind all peer Tx from current Rx */
	peer_ports_len = rte_eth_hairpin_get_peer_ports(port_id, peer_ports, RTE_MAX_ETHPORTS, 0);
	if (peer_ports_len < 0) {
		printf("Failed to get hairpin peer Tx ports of port %d, (%d)\n", port_id, peer_ports_len);
		return DOCA_ERROR_DRIVER;
	}
	for (peer_port = 0; peer_port < peer_ports_len; peer_port++) {
		result = rte_eth_hairpin_unbind(peer_ports[peer_port], port_id);
		if (result < 0) {
			printf("Failed to bind hairpin queues (%d)\n", result);
			return DOCA_ERROR_DRIVER;
		}
	}
	return DOCA_SUCCESS;
}

/*
 * Bind port to all the peer ports
 *
 * @port_id [in]: port ID
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
bind_hairpin_queues(uint16_t port_id)
{
	/* Configure the Rx and Tx hairpin queues for the selected port */
	int result = 0, peer_port, peer_ports_len;
	uint16_t peer_ports[RTE_MAX_ETHPORTS];

	/* bind current Tx to all peer Rx */
	peer_ports_len = rte_eth_hairpin_get_peer_ports(port_id, peer_ports, RTE_MAX_ETHPORTS, 1);
	if (peer_ports_len < 0) {
		printf("Failed to get hairpin peer Rx ports of port %d, (%d)\n", port_id, peer_ports_len);
		return DOCA_ERROR_DRIVER;
	}
	for (peer_port = 0; peer_port < peer_ports_len; peer_port++) {
		result = rte_eth_hairpin_bind(port_id, peer_ports[peer_port]);
		if (result < 0) {
			printf("Failed to bind hairpin queues (%d)\n", result);
			return DOCA_ERROR_DRIVER;
		}
	}
	/* bind all peer Tx to current Rx */
	peer_ports_len = rte_eth_hairpin_get_peer_ports(port_id, peer_ports, RTE_MAX_ETHPORTS, 0);
	if (peer_ports_len < 0) {
		printf("Failed to get hairpin peer Tx ports of port %d, (%d)\n", port_id, peer_ports_len);
		return DOCA_ERROR_DRIVER;
	}

	for (peer_port = 0; peer_port < peer_ports_len; peer_port++) {
		result = rte_eth_hairpin_bind(peer_ports[peer_port], port_id);
		if (result < 0) {
			printf("Failed to bind hairpin queues (%d)\n", result);
			return DOCA_ERROR_DRIVER;
		}
	}
	return DOCA_SUCCESS;
}

static void
disable_hairpin_queues(uint16_t nb_ports)
{
	return;
}

static doca_error_t
enable_hairpin_queues(uint8_t nb_ports)
{
	uint16_t port_id;
	uint16_t n = 0;
	doca_error_t result;

	for (port_id = 0; port_id < RTE_MAX_ETHPORTS; port_id++) {
		if (!rte_eth_dev_is_valid_port(port_id))
			/* the device ID  might not be contiguous */
			continue;
		result = bind_hairpin_queues(port_id);
		if (result != DOCA_SUCCESS) {
			printf("Hairpin bind failed on port=%u\n", port_id);
			disable_hairpin_queues(port_id);
			return result;
		}
		if (++n >= nb_ports)
			break;
	}
	return DOCA_SUCCESS;
}

static void
dpdk_ports_fini(struct application_dpdk_config *app_dpdk_config, uint16_t nb_ports)
{
    return;
}

doca_error_t
dpdk_queues_and_ports_init(struct application_dpdk_config *app_dpdk_config)
{
	doca_error_t result;
	int ret = 0;

	/* Check that DPDK enabled the required ports to send/receive on */
	ret = rte_eth_dev_count_avail();
	if (app_dpdk_config->port_config.nb_ports > 0 && ret < app_dpdk_config->port_config.nb_ports) {
		printf("Application will only function with %u ports, num_of_ports=%d\n",
			 app_dpdk_config->port_config.nb_ports, ret);
		return DOCA_ERROR_DRIVER;
	}

	/* Check for available logical cores */
	ret = rte_lcore_count();
	if (app_dpdk_config->port_config.nb_queues > 0 && ret < app_dpdk_config->port_config.nb_queues) {
		printf("At least %u cores are needed for the application to run, available_cores=%d\n",
			 app_dpdk_config->port_config.nb_queues, ret);
		return DOCA_ERROR_DRIVER;
	}
	app_dpdk_config->port_config.nb_queues = ret;

	if (app_dpdk_config->reserve_main_thread)
		app_dpdk_config->port_config.nb_queues -= 1;

	if (app_dpdk_config->port_config.nb_ports > 0) {
		result = dpdk_ports_init(app_dpdk_config);
		if (result != DOCA_SUCCESS) {
			printf("Ports allocation failed\n");
			goto hairpin_queues_cleanup;
		}
	}

	/* Enable hairpin queues */
	if (app_dpdk_config->port_config.nb_hairpin_q > 0) {
		fprintf(stderr, "Enable hairpin queues...\n");
		result = enable_hairpin_queues(app_dpdk_config->port_config.nb_ports);
		if (result != DOCA_SUCCESS)
			goto ports_cleanup;
	}

	return DOCA_SUCCESS;

hairpin_queues_cleanup:
	disable_hairpin_queues(RTE_MAX_ETHPORTS);
ports_cleanup:
	dpdk_ports_fini(app_dpdk_config, RTE_MAX_ETHPORTS);

	return result;
}
