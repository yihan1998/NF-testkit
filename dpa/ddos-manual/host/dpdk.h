#ifndef _DPDK_H_
#define _DPDK_H_

#include <rte_common.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_eal.h>
#include <rte_launch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>

#include <doca_log.h>

/* Port configuration */
struct application_port_config {
	int nb_ports;			   /* Set on init to 0 for don't care, required ports otherwise */
	uint16_t nb_queues;		   /* Set on init to 0 for don't care, required minimum cores otherwise */
	int nb_hairpin_q;		   /* Set on init to 0 to disable, hairpin queues otherwise */
	uint16_t enable_mbuf_metadata : 1; /* Set on init to 0 to disable, otherwise it will add meta to each mbuf */
	uint16_t self_hairpin : 1;	   /* Set on init to 1 enable both self and peer hairpin */
	uint16_t rss_support : 1;	   /* Set on init to 0 for no RSS support, RSS support otherwise */
	uint16_t lpbk_support : 1;	   /* Enable loopback support */
	uint16_t isolated_mode : 1;	   /* Set on init to 0 for no isolation, isolated mode otherwise */
	uint16_t switch_mode : 1;	   /* Set on init to 1 for switch mode */
};

/* DPDK configuration */
struct application_dpdk_config {
	struct application_port_config port_config; /* DPDK port configuration */
	bool reserve_main_thread;		    /* Reserve lcore for the main thread */
	struct rte_mempool *mbuf_pool;		    /* Will be filled by "dpdk_queues_and_ports_init".
						     * Memory pool that will be used by the DPDK ports
						     * for allocating rte_pktmbuf
						     */
};

extern bool force_quit; /* Set to true to terminate the application */
extern struct flexio_msg_stream *default_stream;

doca_error_t dpdk_queues_and_ports_init(struct application_dpdk_config *app_dpdk_config);
int config_ports(void);
int launch_one_lcore(void * args);

#endif  /* _DPDK_H_ */