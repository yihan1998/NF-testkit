/*
 * Copyright (c) 2023 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <unistd.h>

#include <rte_ethdev.h>

#include <doca_log.h>
#include <doca_flow.h>
#include <doca_flow_ct.h>

#include "flow_ct_common.h"
#include "flow_common.h"

#define PACKET_BURST 128

DOCA_LOG_REGISTER(FLOW_CT_2_PORTS);

/*
 * Create RSS pipe
 *
 * @port [in]: Pipe port
 * @status [in]: User context for adding entry
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t create_rss_pipe(struct doca_flow_port *port,
				    struct entries_status *status,
				    struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_pipe_cfg *cfg;
	struct doca_flow_fwd fwd;
	uint16_t rss_queues[1];
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&fwd, 0, sizeof(fwd));

	result = doca_flow_pipe_cfg_create(&cfg, port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		return result;
	}

	result = set_flow_pipe_cfg(cfg, "RSS_PIPE", DOCA_FLOW_PIPE_BASIC, true);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_match(cfg, &match, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg match: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	/* RSS queue - send matched traffic to queue 0  */
	rss_queues[0] = 0;
	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.rss_queues = rss_queues;
	fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP;
	fwd.num_of_queues = 1;

	result = doca_flow_pipe_create(cfg, &fwd, NULL, pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create RSS pipe: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	doca_flow_pipe_cfg_destroy(cfg);

	/* Match on any packet */
	result = doca_flow_pipe_add_entry(0, *pipe, &match, NULL, NULL, &fwd, 0, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add RSS pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 0);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to process RSS entry: %s", doca_error_get_descr(result));

	return result;

destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(cfg);
	return result;
}

/*
 * Create CT pipe
 *
 * @port [in]: Pipe port
 * @fwd_pipe [in]: Forward pipe pointer
 * @fwd_miss_pipe [in]: Forward miss pipe pointer
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t create_ct_pipe(struct doca_flow_port *port,
				   struct doca_flow_pipe *fwd_pipe,
				   struct doca_flow_pipe *fwd_miss_pipe,
				   struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_match mask;
	struct doca_flow_pipe_cfg *cfg;
	struct doca_flow_fwd fwd;
	struct doca_flow_fwd fwd_miss;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&mask, 0, sizeof(mask));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd));

	result = doca_flow_pipe_cfg_create(&cfg, port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		return result;
	}

	result = set_flow_pipe_cfg(cfg, "CT_PIPE", DOCA_FLOW_PIPE_CT, false);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_match(cfg, &match, &mask);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg match: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = fwd_pipe;

	fwd_miss.type = DOCA_FLOW_FWD_PIPE;
	fwd_miss.next_pipe = fwd_miss_pipe;

	result = doca_flow_pipe_create(cfg, &fwd, &fwd_miss, pipe);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to add CT pipe: %s", doca_error_get_descr(result));
destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(cfg);
	return result;
}

/*
 * Create VxLAN encapsulation pipe
 *
 * @port [in]: Pipe port
 * @status [in]: User context for adding entry
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t create_hairpin_pipe(struct doca_flow_port *port,
					struct entries_status *status,
					struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_actions *actions_list[] = {&actions};
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg *pipe_cfg;
	doca_error_t result;
	uint16_t queue = 2;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));

	result = doca_flow_pipe_cfg_create(&pipe_cfg, port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		return result;
	}

	result = set_flow_pipe_cfg(pipe_cfg, "HAIRPIN_PIPE", DOCA_FLOW_PIPE_BASIC, true);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg match: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_actions(pipe_cfg, actions_list, NULL, NULL, 1);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg actions: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	fwd.type = DOCA_FLOW_FWD_RSS;
	fwd.num_of_queues = 1;
	fwd.rss_queues = &queue;
	fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_UDP;

	result = doca_flow_pipe_create(pipe_cfg, &fwd, NULL, pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create VxLAN Encap pipe: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	doca_flow_pipe_cfg_destroy(pipe_cfg);

	memset(&actions, 0, sizeof(actions));

	result = doca_flow_pipe_add_entry(0, *pipe, &match, &actions, NULL, NULL, 0, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add VxLAN Encap pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 0);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to process UDP entry: %s", doca_error_get_descr(result));

	return result;

destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	return result;
}

/*
 * Create pipe to count packets based on 5 tuple match
 *
 * @port [in]: Pipe port
 * @fwd_pipe [in]: Next pipe pointer
 * @status [in]: User context for adding entry
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t create_count_pipe(struct doca_flow_port *port,
				      struct doca_flow_pipe *fwd_pipe,
				      struct entries_status *status,
				      struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_monitor monitor;
	struct doca_flow_fwd fwd;
	struct doca_flow_fwd fwd_miss;
	struct doca_flow_pipe_cfg *pipe_cfg;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&monitor, 0, sizeof(monitor));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd_miss));

	monitor.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

	result = doca_flow_pipe_cfg_create(&pipe_cfg, port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		return result;
	}

	result = set_flow_pipe_cfg(pipe_cfg, "COUNT_PIPE", DOCA_FLOW_PIPE_BASIC, false);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg match: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &monitor);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg monitor: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = fwd_pipe;

	fwd_miss.type = DOCA_FLOW_FWD_PIPE;
	fwd_miss.next_pipe = fwd_pipe;

	result = doca_flow_pipe_create(pipe_cfg, &fwd, &fwd_miss, pipe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create count pipe: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}
	doca_flow_pipe_cfg_destroy(pipe_cfg);

	result = doca_flow_pipe_add_entry(0, *pipe, &match, NULL, NULL, NULL, 0, status, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add count pipe entry: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_flow_entries_process(port, 0, DEFAULT_TIMEOUT_US, 0);
	if (result != DOCA_SUCCESS)
		DOCA_LOG_ERR("Failed to process count entry: %s", doca_error_get_descr(result));

	return result;

destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	return result;
}

/*
 * Parse UDP packet to update CT tables
 *
 * @packet [in]: Packet to parse
 * @match_o [out]: Origin match struct to fill
 * @match_r [out]: Reply match struct to fill
 */
static void parse_packet(struct rte_mbuf *packet,
			 struct doca_flow_ct_match *match_o,
			 struct doca_flow_ct_match *match_r)
{
	uint8_t *l4_hdr;
	struct rte_ipv4_hdr *ipv4_hdr;
	const struct rte_udp_hdr *udp_hdr;

	ipv4_hdr = rte_pktmbuf_mtod_offset(packet, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));

	match_o->ipv4.src_ip = ipv4_hdr->src_addr;
	match_o->ipv4.dst_ip = ipv4_hdr->dst_addr;
	match_r->ipv4.src_ip = match_o->ipv4.dst_ip;
	match_r->ipv4.dst_ip = match_o->ipv4.src_ip;

	l4_hdr = (typeof(l4_hdr))ipv4_hdr + rte_ipv4_hdr_len(ipv4_hdr);
	udp_hdr = (typeof(udp_hdr))l4_hdr;

	match_o->ipv4.l4_port.src_port = udp_hdr->src_port;
	match_o->ipv4.l4_port.dst_port = udp_hdr->dst_port;
	match_r->ipv4.l4_port.src_port = match_o->ipv4.l4_port.dst_port;
	match_r->ipv4.l4_port.dst_port = match_o->ipv4.l4_port.src_port;

	match_o->ipv4.next_proto = DOCA_FLOW_PROTO_UDP;
	match_r->ipv4.next_proto = DOCA_FLOW_PROTO_UDP;
}

/*
 * Dequeue packets from DPDK queues, parse and update CT tables with new connection 5 tuple
 *
 * @port [in]: Port to which an entry should be inserted
 * @port_id [in]: Port id to which packet can be received
 * @ct_queue [in]: DOCA Flow CT queue number
 * @ct_pipe [in]: Pipe of CT
 * @ct_status [in]: User context for adding CT entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t process_packets(struct doca_flow_port *port,
				    uint16_t port_id,
				    uint16_t ct_queue,
				    struct doca_flow_pipe *ct_pipe,
				    struct entries_status *ct_status)
{
	struct rte_mbuf *packets[PACKET_BURST];
	struct doca_flow_ct_match match_o;
	struct doca_flow_ct_match match_r;
	struct doca_flow_pipe_entry *entry;
	uint32_t flags;
	doca_error_t result;
	int i, entries, nb_packets = 0;
	bool conn_found = false;

	memset(&match_o, 0, sizeof(match_o));
	memset(&match_r, 0, sizeof(match_r));

	DOCA_LOG_INFO("Listening on port %u, please send UDP packet with DIP 1.1.1.1", port_id);
	do {
		nb_packets = rte_eth_rx_burst(port_id, 0, packets, PACKET_BURST);
	} while (nb_packets == 0);

	DOCA_LOG_INFO("%d packets received", nb_packets);

	entries = 0;
	DOCA_LOG_INFO("Sample received %d packets on port %d", nb_packets, port_id);
	for (i = 0; i < PACKET_BURST && i < nb_packets; i++) {
		parse_packet(packets[i], &match_o, &match_r);
		flags = DOCA_FLOW_CT_ENTRY_FLAGS_ALLOC_ON_MISS | DOCA_FLOW_CT_ENTRY_FLAGS_DUP_FILTER_ORIGIN |
			DOCA_FLOW_CT_ENTRY_FLAGS_DUP_FILTER_REPLY;
		/* Allocate CT entry */
		result = doca_flow_ct_entry_prepare(ct_queue,
						    ct_pipe,
						    flags,
						    &match_o,
						    packets[i]->hash.rss,
						    &match_r,
						    packets[i]->hash.rss,
						    &entry,
						    &conn_found);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to prepare CT entry\n");
			return result;
		}

		if (!conn_found) {
			flags = DOCA_FLOW_CT_ENTRY_FLAGS_NO_WAIT | DOCA_FLOW_CT_ENTRY_FLAGS_DIR_ORIGIN |
				DOCA_FLOW_CT_ENTRY_FLAGS_DIR_REPLY;
			result = doca_flow_ct_add_entry(ct_queue,
							ct_pipe,
							flags,
							&match_o,
							&match_r,
							NULL,
							NULL,
							0,
							0,
							0,
							ct_status,
							entry);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("Failed to add CT pipe an entry: %s", doca_error_get_descr(result));
				return result;
			}
			entries++;
		}
	}

	DOCA_LOG_INFO("%d CT connections created", nb_packets);

	while (ct_status->nb_processed != entries) {
		result = doca_flow_entries_process(port, ct_queue, DEFAULT_TIMEOUT_US, 0);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to process Flow CT entries: %s", doca_error_get_descr(result));
			return result;
		}

		if (ct_status->failure) {
			DOCA_LOG_ERR("Flow CT entries process returned with a failure");
			return DOCA_ERROR_BAD_STATE;
		}
	}

	DOCA_LOG_INFO("%d CT connections processed\n", ct_status->nb_processed);

	return DOCA_SUCCESS;
}

/*
 * flow_ct_2_ports
 *
 * @nb_queues [in]: number of queues the sample will use
 * @dev_arr [in]: Flow CT devices
 * @nb_ports [in]: number of ports the sample will use
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t flow_ct_2_ports(uint16_t nb_queues, struct doca_dev *dev_arr[], int nb_ports)
{
	const int nb_entries = 5;
	struct flow_resources resource;
	uint32_t nr_shared_resources[SHARED_RESOURCE_NUM_VALUES] = {0};
	struct doca_flow_pipe *rss_pipes[nb_ports];
	struct doca_flow_pipe *hairpin_pipes[nb_ports];
	struct doca_flow_pipe *count_pipes[nb_ports];
	struct doca_flow_pipe *ct_pipes[nb_ports];
	struct doca_flow_pipe *udp_pipes[nb_ports];
	struct doca_flow_port *ports[nb_ports];
	struct doca_flow_meta o_zone_mask, o_modify_mask, r_zone_mask, r_modify_mask;
	struct entries_status ctrl_status, ct_status;
	uint32_t ct_flags, nb_arm_queues = 1, nb_ctrl_queues = 1, nb_user_actions = 0, nb_ipv4_sessions = 1024,
			   nb_ipv6_sessions = 0; /* On BF2 should always be 0 */
	uint16_t ct_queue = nb_queues;
	doca_error_t result;
	int i;

	memset(&resource, 0, sizeof(resource));
	memset(rss_pipes, 0, sizeof(struct doca_flow_pipe *) * nb_ports);
	memset(hairpin_pipes, 0, sizeof(struct doca_flow_pipe *) * nb_ports);
	memset(count_pipes, 0, sizeof(struct doca_flow_pipe *) * nb_ports);
	memset(ct_pipes, 0, sizeof(struct doca_flow_pipe *) * nb_ports);
	memset(udp_pipes, 0, sizeof(struct doca_flow_pipe *) * nb_ports);

	resource.nr_counters = 1;

	result = init_doca_flow(nb_queues, "switch,hws", &resource, nr_shared_resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA Flow: %s", doca_error_get_descr(result));
		return result;
	}

	/* Dont use zone masking */
	memset(&o_zone_mask, 0, sizeof(o_zone_mask));
	memset(&o_modify_mask, 0, sizeof(o_modify_mask));
	memset(&r_zone_mask, 0, sizeof(r_zone_mask));
	memset(&r_modify_mask, 0, sizeof(r_modify_mask));

	ct_flags = DOCA_FLOW_CT_FLAG_NO_AGING | DOCA_FLOW_CT_FLAG_NO_COUNTER;
	result = init_doca_flow_ct(ct_flags,
				   nb_arm_queues,
				   nb_ctrl_queues,
				   nb_user_actions,
				   NULL,
				   nb_ipv4_sessions,
				   nb_ipv6_sessions,
				   DUP_FILTER_CONN_NUM,
				   false,
				   &o_zone_mask,
				   &o_modify_mask,
				   false,
				   &r_zone_mask,
				   &r_modify_mask);
	if (result != DOCA_SUCCESS) {
		doca_flow_destroy();
		return result;
	}

	memset(ports, 0, sizeof(struct doca_dev *) * nb_ports);
	result = init_doca_flow_ports(nb_ports, ports, false, dev_arr);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA ports: %s", doca_error_get_descr(result));
		doca_flow_ct_destroy();
		doca_flow_destroy();
		return result;
	}

	for (i = 0; i < nb_ports; i++) {
		memset(&ctrl_status, 0, sizeof(ctrl_status));

		result = create_rss_pipe(ports[i], &ctrl_status, &rss_pipes[i]);
		if (result != DOCA_SUCCESS)
			goto cleanup;

		result = create_hairpin_pipe(ports[i], &ctrl_status, &hairpin_pipes[i]);
		if (result != DOCA_SUCCESS)
			goto cleanup;

		result = create_count_pipe(ports[i], rss_pipes[i], &ctrl_status, &count_pipes[i]);
		if (result != DOCA_SUCCESS)
			goto cleanup;

		result = create_ct_pipe(ports[i], hairpin_pipes[i], count_pipes[i], &ct_pipes[i]);
		if (result != DOCA_SUCCESS)
			goto cleanup;

		result = create_ct_root_pipe(ports[i],
					     true,
					     false,
					     DOCA_FLOW_L4_META_UDP,
					     ct_pipes[i],
					     &ctrl_status,
					     &udp_pipes[i]);
		if (result != DOCA_SUCCESS)
			goto cleanup;

		if (ctrl_status.nb_processed != nb_entries || ctrl_status.failure) {
			DOCA_LOG_ERR("Failed to process control path entries");
			result = DOCA_ERROR_BAD_STATE;
			goto cleanup;
		}
	}

	DOCA_LOG_INFO("Please send same UDP packets to see the CT entries being created\n");
	for (i = 0; i < nb_ports; i++) {
		memset(&ct_status, 0, sizeof(ct_status));
		result = process_packets(ports[i], i, ct_queue, ct_pipes[i], &ct_status);
		if (result != DOCA_SUCCESS)
			goto cleanup;
	}

	sleep(3);

cleanup:
	for (i = 0; i < nb_ports; i++) {
		if (udp_pipes[i] != NULL)
			doca_flow_pipe_destroy(udp_pipes[i]);
		if (ct_pipes[i] != NULL)
			doca_flow_pipe_destroy(ct_pipes[i]);
		if (hairpin_pipes[i] != NULL)
			doca_flow_pipe_destroy(hairpin_pipes[i]);
		if (count_pipes[i] != NULL)
			doca_flow_pipe_destroy(count_pipes[i]);
		if (rss_pipes[i] != NULL)
			doca_flow_pipe_destroy(rss_pipes[i]);
	}
	cleanup_procedure(NULL, nb_ports, ports);
	return result;
}
