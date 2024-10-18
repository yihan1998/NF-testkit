/*
 * Copyright (c) 2022 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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

#include <rte_byteorder.h>

#include <doca_log.h>
#include <doca_flow.h>

#include "flow_common.h"

DOCA_LOG_REGISTER(FLOW_HAIRPIN_VNF);

/*
 * Create DOCA Flow pipe with 5 tuple match that forwards the matched traffic to the other port
 *
 * @port [in]: port of the pipe
 * @port_id [in]: port ID of the pipe
 * @pipe [out]: created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t create_hairpin_pipe(struct doca_flow_port *port, int port_id, struct doca_flow_pipe **pipe)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions, *actions_arr[NB_ACTIONS_ARR];
	struct doca_flow_fwd fwd;
	struct doca_flow_pipe_cfg *pipe_cfg;
	doca_error_t result;

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));
	memset(&fwd, 0, sizeof(fwd));

	actions_arr[0] = &actions;

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
	result = doca_flow_pipe_cfg_set_actions(pipe_cfg, actions_arr, NULL, NULL, NB_ACTIONS_ARR);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set doca_flow_pipe_cfg actions: %s", doca_error_get_descr(result));
		goto destroy_pipe_cfg;
	}

	/* forwarding traffic to other port */
	fwd.type = DOCA_FLOW_FWD_PORT;
	fwd.port_id = port_id ^ 1;

	result = doca_flow_pipe_create(pipe_cfg, &fwd, NULL, pipe);
destroy_pipe_cfg:
	doca_flow_pipe_cfg_destroy(pipe_cfg);
	return result;
}

/*
 * Add DOCA Flow pipe entry to the hairpin pipe
 *
 * @pipe [in]: pipe of the entry
 * @status [in]: user context for adding entry
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
static doca_error_t add_hairpin_pipe_entry(struct doca_flow_pipe *pipe, struct entries_status *status)
{
	struct doca_flow_match match;
	struct doca_flow_actions actions;
	struct doca_flow_pipe_entry *entry;
	doca_error_t result;

	/* example 5-tuple to forward */
	// doca_be32_t dst_ip_addr = BE_IPV4_ADDR(8, 8, 8, 8);
	// doca_be32_t src_ip_addr = BE_IPV4_ADDR(1, 2, 3, 4);
	// doca_be16_t dst_port = rte_cpu_to_be_16(80);
	// doca_be16_t src_port = rte_cpu_to_be_16(1234);

	memset(&match, 0, sizeof(match));
	memset(&actions, 0, sizeof(actions));

	// match.outer.ip4.dst_ip = dst_ip_addr;
	// match.outer.ip4.src_ip = src_ip_addr;
	// match.outer.tcp.l4_port.dst_port = dst_port;
	// match.outer.tcp.l4_port.src_port = src_port;

	result = doca_flow_pipe_add_entry(0, pipe, &match, &actions, NULL, NULL, 0, status, &entry);
	if (result != DOCA_SUCCESS)
		return result;

	return DOCA_SUCCESS;
}

/*
 * Run flow_hairpin_vnf sample
 *
 * @nb_queues [in]: number of queues the sample will use
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t flow_hairpin_vnf(int nb_queues)
{
	int nb_ports = 2;
	struct flow_resources resource = {0};
	uint32_t nr_shared_resources[SHARED_RESOURCE_NUM_VALUES] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_dev *dev_arr[nb_ports];
	struct doca_flow_pipe *pipe;
	struct entries_status status;
	int num_of_entries = 1;
	doca_error_t result;
	int port_id;

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

		result = create_hairpin_pipe(ports[port_id], port_id, &pipe);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to create pipe: %s", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_hairpin_pipe_entry(pipe, &status);
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
	sleep(5);

	result = stop_doca_flow_ports(nb_ports, ports);
	doca_flow_destroy();
	return result;
}
