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

int dpdk_ports_init(struct application_dpdk_config *app_config) {
    int ret;
    uint16_t portid;
    char name[RTE_MEMZONE_NAMESIZE];
    uint16_t nb_rxd = DEFAULT_RX_DESC;
    uint16_t nb_txd = DEFAULT_TX_DESC;

    nb_cores = rte_lcore_count();

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

    if (app_config->port_config.enable_mbuf_metadata || app_config->sft_config.enable) {
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

        /* Enabled hairpin queue before port start */
        if (nb_hairpin_queues && app_config->port_config.self_hairpin && rte_eth_dev_is_valid_port(port ^ 1)) {
            /* Hairpin to both self and peer */
            assert((nb_hairpin_queues % 2) == 0);
            for (queue_index = 0; queue_index < nb_hairpin_queues / 2; queue_index++)
                rss_queue_list[queue_index] = app_config->port_config.nb_queues + queue_index * 2;
            result = setup_hairpin_queues(port, port, rss_queue_list, nb_hairpin_queues / 2);
            if (result != DOCA_SUCCESS) {
                printf("Cannot hairpin self port %u, ret: %s\n",
                        port, doca_get_error_string(result));
                return result;
            }
            for (queue_index = 0; queue_index < nb_hairpin_queues / 2; queue_index++)
                rss_queue_list[queue_index] = app_config->port_config.nb_queues + queue_index * 2 + 1;
            result = setup_hairpin_queues(port, port ^ 1, rss_queue_list, nb_hairpin_queues / 2);
            if (result != DOCA_SUCCESS) {
                printf("Cannot hairpin peer port %u, ret: %s\n",
                        port ^ 1, doca_get_error_string(result));
                return result;
            }
        } else if (nb_hairpin_queues) {
            /* Hairpin to self or peer */
            for (queue_index = 0; queue_index < nb_hairpin_queues; queue_index++)
                rss_queue_list[queue_index] = app_config->port_config.nb_queues + queue_index;
            if (rte_eth_dev_is_valid_port(port ^ 1))
                result = setup_hairpin_queues(port, port ^ 1, rss_queue_list, nb_hairpin_queues);
            else
                result = setup_hairpin_queues(port, port, rss_queue_list, nb_hairpin_queues);
            if (result != DOCA_SUCCESS) {
                printf("Cannot hairpin port %u, ret=%d\n", port, result);
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

int run_dpdk_loop(void) {
    int portid;
    struct rte_mbuf * rx_pkts[DEFAULT_PKT_BURST];
    int nb_rx, nb_tx;
	RTE_ETH_FOREACH_DEV(portid) {
        for (int i = 0; i < nb_cores; i++) {
            nb_rx = rte_eth_rx_burst(portid, i, rx_pkts, DEFAULT_PKT_BURST);
            if (nb_rx) {
                printf("Receive %d packets\n", nb_rx);
                nb_tx = rte_eth_tx_burst(portid, i, rx_pkts, nb_rx);
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

doca_error_t
dpdk_queues_and_ports_init(struct application_dpdk_config *app_dpdk_config, int argc, char ** argv)
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
			goto gpu_cleanup;
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
