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

#include <string.h>
#include <unistd.h>

#include <doca_log.h>
#include <doca_flow.h>

#include "flow_common.h"

DOCA_LOG_REGISTER(FLOW_LPM_EM);

#define NB_ACTION_DESC (1)
#define TEST_LPM_EM_TAG 1
#define META_U32_BIT_OFFSET(idx) (offsetof(struct doca_flow_meta, u32[(idx)]) << 3)

/*
 * Create DOCA Flow basic pipe that gets vlan from the packet, sets the value vlan to the register 1
 *
 * @port [in]: port of the pipe
 * @next_pipe [in]: lpm pipe to forward the matched traffic
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t create_main_pipe(struct doca_flow_port *port,
				     struct doca_flow_pipe *next_pipe,
				     struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_monitor counter;
	struct doca_flow_actions actions;
	struct doca_flow_actions *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_fwd fwd;
	struct doca_flow_fwd fwd_miss;
	struct doca_flow_pipe_cfg *pipe_cfg;
	struct doca_flow_action_descs descs;
	struct doca_flow_action_descs *descs_arr[NB_ACTIONS_ARR];
	struct doca_flow_action_desc desc_array[NB_ACTION_DESC] = {0};
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&counter, 0, sizeof(counter));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));
	memset(&fwd_miss, 0, sizeof(fwd_miss));
	memset(&descs, 0, sizeof(descs));

	counter.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;

	actions_arr[0] = &actions;
	descs_arr[0] = &descs;
	descs.nb_action_desc = 1;
	descs.desc_array = desc_array;

	fwd_miss.type = DOCA_FLOW_FWD_DROP;

	/* forwarding traffic to next pipe */
	fwd.type = DOCA_FLOW_FWD_PIPE;
	fwd.next_pipe = next_pipe;

	desc_array[0].type = DOCA_FLOW_ACTION_COPY;
	desc_array[0].field_op.src.field_string = "outer.eth_vlan0.tci";
	desc_array[0].field_op.src.bit_offset = 0;
	desc_array[0].field_op.dst.field_string = "meta.data";
	desc_array[0].field_op.dst.bit_offset = META_U32_BIT_OFFSET(TEST_LPM_EM_TAG);
	desc_array[0].field_op.width = 8;

	result = doca_flow_pipe_cfg_create(&pipe_cfg, port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		return result;
	}

	result = set_flow_pipe_cfg(pipe_cfg, "MAIN_PIPE_COPY_TO_META", DOCA_FLOW_PIPE_BASIC, true);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, NULL);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg match: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &counter);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg monitor: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	result = doca_flow_pipe_cfg_set_actions(pipe_cfg, actions_arr, NULL, descs_arr, NB_ACTIONS_ARR);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg actions: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	result = doca_flow_pipe_create(pipe_cfg, &fwd, &fwd_miss, pipe);

destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	return result;
}

/*
 * Add DOCA Flow pipe entry to the basic pipe that forwards ipv4 traffic to lpm pipe
 *
 * @pipe [in]: pipe of the entry
 * @status [in]: user context for adding entry
 * @entry [out]: result of entry addition
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t add_main_pipe_entry(struct doca_flow_pipe *pipe,
					struct entries_status *status,
					struct doca_flow_pipe_entry **entry)
{
	struct doca_flow_match match;

	memset(&match, 0, sizeof(match));

	return doca_flow_pipe_add_entry(0, pipe, &match, NULL, NULL, NULL, DOCA_FLOW_NO_WAIT, status, entry);
}

/*
 * Add DOCA Flow LPM pipe which performs LPM logic for IPv4 src address and exact-match logic on
 * meta.u32[1], match_mask.tun.vxlan_tun_id and match_mask.inner.eth.dst_mac.
 * Only these fields are available for exact-match logic.
 * To enable the exact-match logic, set any of these fields to full mask.
 *
 *
 * @port [in]: port of the pipe
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t create_lpm_pipe(struct doca_flow_port *port, struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match, match_mask;
	struct doca_flow_actions actions, *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_pipe_cfg *pipe_cfg;
	struct doca_flow_fwd fwd = {.type = DOCA_FLOW_FWD_CHANGEABLE};
	struct doca_flow_monitor counter;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&match_mask, 0, sizeof(match_mask));
	memset(&actions, 0, sizeof(actions));
	memset(&counter, 0, sizeof(counter));

	match.outer.l3_type = DOCA_FLOW_L3_TYPE_IP4;
	match.outer.ip4.src_ip = UINT32_MAX;

	match_mask.meta.u32[1] = UINT32_MAX;
	match_mask.tun.type = DOCA_FLOW_TUN_VXLAN;
	match_mask.tun.vxlan_tun_id = UINT32_MAX;
	memset(match_mask.inner.eth.dst_mac, UINT8_MAX, sizeof(match_mask.inner.eth.dst_mac));

	actions_arr[0] = &actions;

	result = doca_flow_pipe_cfg_create(&pipe_cfg, port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		return result;
	}

	result = set_flow_pipe_cfg(pipe_cfg, "LPM_EM_PIPE", DOCA_FLOW_PIPE_LPM, false);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	result = doca_flow_pipe_cfg_set_match(pipe_cfg, &match, &match_mask);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg match: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	result = doca_flow_pipe_cfg_set_actions(pipe_cfg, actions_arr, NULL, NULL, NB_ACTIONS_ARR);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg actions: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	counter.counter_type = DOCA_FLOW_RESOURCE_TYPE_NON_SHARED;
	result = doca_flow_pipe_cfg_set_monitor(pipe_cfg, &counter);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg counter: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	result = doca_flow_pipe_create(pipe_cfg, &fwd, NULL, pipe);
destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	return result;
}

/*
 * Add DOCA Flow pipe entry to the LPM pipe.
 *
 * @pipe [in]: pipe of the entry
 * @port_id [in]: port ID of the entry
 * @src_ip_addr [in]: src ip address
 * @src_ip_addr_mask [in]: src ip mask
 * @exact_match_meta [in]: value for exact match logic on meta
 * @exact_match_vni [in]: value for exact match logic on vni
 * @exact_match_inner_dmac [in]: pointer to value for exact match logic on inner destination mac
 * @flag [in]: Flow entry will be pushed to hw immediately or not. enum doca_flow_flags_type.
 *	flag DOCA_FLOW_WAIT_FOR_BATCH is using for collecting entries by LPM module
 *	flag DOCA_FLOW_NO_WAIT is using for adding the entry and starting building and offloading
 * @status [in]: user context for adding entry
 * @entry [out]: created entry pointer.
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t add_lpm_one_entry(struct doca_flow_pipe *pipe,
				      uint16_t port_id,
				      doca_be32_t src_ip_addr,
				      doca_be32_t src_ip_addr_mask,
				      uint32_t exact_match_meta,
				      uint32_t exact_match_vni,
				      uint8_t *exact_match_inner_dmac,
				      const enum doca_flow_flags_type flag,
				      struct entries_status *status,
				      struct doca_flow_pipe_entry **entry)
{
	struct doca_flow_match match = {0};
	struct doca_flow_match match_mask = {0};
	struct doca_flow_fwd fwd = {0};
	doca_error_t rc;

	match.outer.ip4.src_ip = src_ip_addr;
	match.meta.u32[1] = exact_match_meta;
	match.tun.vxlan_tun_id = exact_match_vni;
	memcpy(match.inner.eth.dst_mac, exact_match_inner_dmac, sizeof(match.inner.eth.dst_mac));

	match_mask.outer.ip4.src_ip = src_ip_addr_mask;

	if (port_id == UINT16_MAX)
		fwd.type = DOCA_FLOW_FWD_DROP;
	else {
		fwd.type = DOCA_FLOW_FWD_PORT;
		fwd.port_id = port_id ^ 1;
	}

	rc = doca_flow_pipe_lpm_add_entry(0, pipe, &match, &match_mask, NULL, NULL, &fwd, flag, status, entry);
	if (rc != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to add lpm pipe entry: %s", doca_error_get_descr(rc));
		return rc;
	}
	return rc;
}

/*
 * Add DOCA Flow pipe entries to the LPM pipe.
 * one entry with full mask and one with 16 bits mask for vlan 1 and vlan 2
 * and one default entry for each vlan
 *
 * @pipe [in]: pipe of the entry
 * @port_id [in]: port ID of the entry
 * @status [in]: user context for adding entry
 * @entries [out]: created entry pointers.
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t add_lpm_pipe_entries(struct doca_flow_pipe *pipe,
					 uint16_t port_id,
					 struct entries_status *status,
					 struct doca_flow_pipe_entry **entries)
{
	doca_error_t rc;
	uint8_t inner_dmac[6] = {0};

	/* add default entry with 0 bits mask and fwd drop */
	rc = add_lpm_one_entry(pipe,
			       UINT16_MAX, /* indicates forward drop */
			       BE_IPV4_ADDR(0, 0, 0, 0),
			       RTE_BE32(0x00000000),
			       0, /* does not make a difference for a default entry */
			       0,
			       inner_dmac,
			       DOCA_FLOW_WAIT_FOR_BATCH,
			       status,
			       &entries[0]);
	if (rc != DOCA_SUCCESS)
		return rc;

	/* add entry with full mask and fwd port */
	memset(inner_dmac, 1, sizeof(inner_dmac));
	rc = add_lpm_one_entry(pipe,
			       port_id,
			       BE_IPV4_ADDR(1, 2, 3, 4),
			       RTE_BE32(0xffffffff),
			       1,
			       BUILD_VNI(0xabcde1),
			       inner_dmac,
			       DOCA_FLOW_WAIT_FOR_BATCH,
			       status,
			       &entries[1]);
	if (rc != DOCA_SUCCESS)
		return rc;

	memset(inner_dmac, 2, sizeof(inner_dmac));
	rc = add_lpm_one_entry(pipe,
			       port_id,
			       BE_IPV4_ADDR(1, 2, 3, 4),
			       RTE_BE32(0xffffffff),
			       2,
			       BUILD_VNI(0xabcde2),
			       inner_dmac,
			       DOCA_FLOW_WAIT_FOR_BATCH,
			       status,
			       &entries[2]);
	if (rc != DOCA_SUCCESS)
		return rc;

	/* add entry with full mask, but exact-match 3 to fwd drop */
	memset(inner_dmac, 3, sizeof(inner_dmac));
	rc = add_lpm_one_entry(pipe,
			       UINT16_MAX,
			       BE_IPV4_ADDR(1, 2, 3, 4),
			       RTE_BE32(0xffffffff),
			       3,
			       BUILD_VNI(0xabcde3),
			       inner_dmac,
			       DOCA_FLOW_WAIT_FOR_BATCH,
			       status,
			       &entries[3]);
	if (rc != DOCA_SUCCESS)
		return rc;

	/* add entry with 16 bit mask, exact-match 3 and fwd port */
	rc = add_lpm_one_entry(pipe,
			       port_id,
			       BE_IPV4_ADDR(1, 2, 0, 0),
			       RTE_BE32(0xffff0000),
			       3,
			       BUILD_VNI(0xabcde3),
			       inner_dmac,
			       DOCA_FLOW_NO_WAIT,
			       status,
			       &entries[4]);
	if (rc != DOCA_SUCCESS)
		return rc;

	return DOCA_SUCCESS;
}

/*
 * Run flow_lpm_em sample
 *
 * @nb_queues [in]: number of queues the sample will use
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t flow_lpm_em(int nb_queues)
{
	const int nb_ports = 2;
	/* 1 entry for main pipe and 5 entries for LPM pipe */
	const int num_of_entries = 6;
	struct flow_resources resource = {.nr_counters = 64};
	struct doca_dev *dev_arr[nb_ports];
	uint32_t nr_shared_resources[SHARED_RESOURCE_NUM_VALUES] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_flow_pipe *main_pipe;
	struct doca_flow_pipe *lpm_pipe;
	struct entries_status status;
	struct doca_flow_pipe_entry *entries[nb_ports][num_of_entries];
	struct doca_flow_resource_query stats;
	doca_error_t result;
	int port_id, lpm_entry_id;

	result = init_doca_flow(nb_queues, "vnf,hws", &resource, nr_shared_resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA Flow: %s", doca_error_get_descr(result));
		return result;
	}

	memset(dev_arr, 0, sizeof(struct doca_dev *) * nb_ports);
	result = init_doca_flow_ports(nb_ports, ports, true, dev_arr);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init DOCA ports: %s", doca_error_get_descr(result));
		doca_flow_destroy();
		return result;
	}

	for (port_id = 0; port_id < nb_ports; port_id++) {
		memset(&status, 0, sizeof(status));

		result = create_lpm_pipe(ports[port_id], &lpm_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_lpm_pipe_entries(lpm_pipe, port_id, &status, &entries[port_id][1]);
		if (result != DOCA_SUCCESS) {
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = create_main_pipe(ports[port_id], lpm_pipe, &main_pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create main pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_main_pipe_entry(main_pipe, &status, &entries[port_id][0]);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to add entry: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = doca_flow_entries_process(ports[port_id], 0, DEFAULT_TIMEOUT_US, num_of_entries);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to process entries: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		if (status.nb_processed != num_of_entries || status.failure) {
			DOCA_LOG_ERR("Failed to process entries");
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return DOCA_ERROR_BAD_STATE;
		}
	}
	DOCA_LOG_INFO("Wait few seconds for packets to arrive");
	sleep(60);

	for (port_id = 0; port_id < nb_ports; port_id++) {
		result = doca_flow_resource_query_entry(entries[port_id][0], &stats);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Port %d failed to query main pipe entry: %s",
				     port_id,
				     doca_error_get_descr(result));
			return result;
		}
		DOCA_LOG_INFO("Port %d, main pipe entry received %lu packets", port_id, stats.counter.total_pkts);

		for (lpm_entry_id = 1; lpm_entry_id < num_of_entries; lpm_entry_id++) {
			result = doca_flow_resource_query_entry(entries[port_id][lpm_entry_id], &stats);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("Port %d failed to query LPM entry %d: %s",
					     port_id,
					     lpm_entry_id - 1,
					     doca_error_get_descr(result));
				return result;
			}

			DOCA_LOG_INFO("Port %d, LPM entry %d received %lu packets",
				      port_id,
				      lpm_entry_id - 1,
				      stats.counter.total_pkts);
		}
	}

	stop_doca_flow_ports(nb_ports, ports);
	doca_flow_destroy();
	return DOCA_SUCCESS;
}
