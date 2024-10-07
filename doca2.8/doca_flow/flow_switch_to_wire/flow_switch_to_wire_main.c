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

/*
 * The application is to verify the unified switch model traffic.
 * It can be used to verify the traffic from wire to wire, wire to
 * vf and vf to vf.
 * User can use different packets to verify different directions of
 * traffic. The incoming traffic can be from wire or vf. It steers
 * the pkt based on the pkt content, so user can send the traffic
 * from different src and check if the pkt goes to the expected
 * destinations.
 * As this sample add the RSS pipe, so doca-flow isolated mode
 * is chosen without any internal RSS pipes.
 * In expert mode, the missed traffic will be tagged with port_id
 * and the traffic will be sent to egress root.
 *
 * It requires 4 ports. It builds the pipe entries as below:
 *
 * Ingress pipe:
 * Entry 0: IP src 1.2.3.4 / TCP src 1234 dst 80 -> egress pipe
 * Entry 1: IP src 1.2.3.5 / TCP src 1234 dst 80 -> vport pipe
 *
 * Egress pipe(test ingress to egress cross domain):
 * Entry 0: IP dst 8.8.8.8 / TCP src 1234 dst 80 -> port 0
 * Entry 1: IP dst 8.8.8.9 / TCP src 1234 dst 80 -> port 1
 * Entry 2: IP dst 8.8.8.10 / TCP src 1234 dst 80 -> port 2
 * Entry 3: IP dst 8.8.8.11 / TCP src 1234 dst 80 -> port 3
 *
 * Vport pipe(test ingress direct to vport):
 * Entry 0: IP dst 8.8.8.8 / TCP src 1234 -> port 0
 * Entry 1: IP dst 8.8.8.9 / TCP src 1234 -> port 1
 * Entry 2: IP dst 8.8.8.10 / TCP src 1234-> port 2
 * Entry 3: IP dst 8.8.8.11 / TCP src 1234-> port 3
 *
 * RSS pipe(test miss traffic port_id get and dst port_id set):
 * Entry 0: IPv4 / TCP -> port 0
 * Entry 0: IPv4 / UDP -> port 1
 * Entry 0: IPv4 / ICMP -> port 2
 */
#include <stdlib.h>

#include <rte_ethdev.h>

#include <doca_argp.h>
#include <doca_dev.h>
#include <doca_flow.h>
#include <doca_log.h>
#include <doca_ctx.h>

#include <dpdk_utils.h>

#include "flow_switch_common.h"

DOCA_LOG_REGISTER(FLOW_SWITCH_TO_WIRE::MAIN);

#define SWITCH_TO_WIRE_PORTS 4

/* Sample's Logic */
doca_error_t flow_switch_to_wire(int nb_queues, int nb_ports, struct flow_switch_ctx *ctx);

/*
 * Sample main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int main(int argc, char **argv)
{
	doca_error_t result;
	struct doca_log_backend *sdk_log;
	int exit_status = EXIT_FAILURE;
	struct application_dpdk_config dpdk_config = {
		.port_config.nb_ports = SWITCH_TO_WIRE_PORTS,
		.port_config.nb_queues = 1,
		.port_config.isolated_mode = 1,
		.port_config.switch_mode = 1,
	};
	struct flow_switch_ctx ctx = {0};
	uint16_t nr_ports;

	/* Register a logger backend */
	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		goto sample_exit;

	/* Register a logger backend for internal SDK errors and warnings */
	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		goto sample_exit;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		goto sample_exit;

	result = doca_argp_init("doca_flow_switch_to_wire", &ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		goto sample_exit;
	}
	result = register_doca_flow_switch_param();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register flow param: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}
	doca_argp_set_dpdk_program(init_flow_switch_dpdk);
	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse sample input: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	result = init_doca_flow_switch_common(&ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init flow switch common: %s", doca_error_get_descr(result));
		goto dpdk_cleanup;
	}

	nr_ports = rte_eth_dev_count_avail();
	if (nr_ports < SWITCH_TO_WIRE_PORTS) {
		DOCA_LOG_ERR("Failed to init - lack of ports, probed:%d, needed:%d", nr_ports, SWITCH_TO_WIRE_PORTS);
		goto sample_exit;
	}

	/* update queues and ports */
	result = dpdk_queues_and_ports_init(&dpdk_config);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to update ports and queues");
		goto dpdk_cleanup;
	}

	/* run sample */
	result = flow_switch_to_wire(dpdk_config.port_config.nb_queues, SWITCH_TO_WIRE_PORTS, &ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("flow_switch_to_wire() encountered an error: %s", doca_error_get_descr(result));
		goto dpdk_ports_queues_cleanup;
	}

	exit_status = EXIT_SUCCESS;

dpdk_ports_queues_cleanup:
	dpdk_queues_and_ports_fini(&dpdk_config);
dpdk_cleanup:
	dpdk_fini();
argp_cleanup:
	doca_argp_destroy();
sample_exit:
	destroy_doca_flow_switch_common(&ctx);
	if (exit_status == EXIT_SUCCESS)
		DOCA_LOG_INFO("Sample finished successfully");
	else
		DOCA_LOG_INFO("Sample finished with errors");
	return exit_status;
}
