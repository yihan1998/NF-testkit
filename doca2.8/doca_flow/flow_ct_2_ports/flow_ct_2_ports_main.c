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

#include <stdlib.h>

#include <doca_argp.h>
#include <doca_log.h>
#include <doca_dpdk.h>

#include <dpdk_utils.h>

#include "flow_ct_common.h"
#include "common.h"

DOCA_LOG_REGISTER(FLOW_CT_2_PORTS::MAIN);

/* Sample's Logic */
doca_error_t flow_ct_2_ports(uint16_t nb_queues, struct doca_dev *dev_arr[], int nb_ports);

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
	struct doca_dev *ct_dev[MAX_PORTS] = {};
	struct ct_config ct_cfg = {0};
	struct application_dpdk_config dpdk_config = {
		.port_config.nb_queues = 2,
		.port_config.isolated_mode = 1,
		.port_config.switch_mode = 1,
		.port_config.nb_hairpin_q = 2,
		.port_config.self_hairpin = 1,
		.reserve_main_thread = false,
	};
	int i;

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

	DOCA_LOG_INFO("Starting the sample");

	result = doca_argp_init("doca_flow_ct_2_ports", &ct_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		goto sample_exit;
	}
	doca_argp_set_dpdk_program(flow_ct_dpdk_init);

	result = flow_ct_register_params();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register Flow Ct sample parameters: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse sample input: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	for (i = 0; i < ct_cfg.n_ports; i++) {
		result = open_doca_device_with_pci(ct_cfg.ct_dev_pci_addr[i], flow_ct_capable, &ct_dev[i]);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to open Flow CT device: %s", doca_error_get_descr(result));
			goto dpdk_cleanup;
		}

		result = doca_dpdk_port_probe(ct_dev[i], FLOW_CT_COMMON_DEVARGS);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to open Flow CT device: %s", doca_error_get_descr(result));
			goto device_cleanup;
		}
	}

	/* update queues and ports */
	dpdk_config.port_config.nb_ports = ct_cfg.n_ports;
	result = dpdk_queues_and_ports_init(&dpdk_config);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to update ports and queues");
		goto device_cleanup;
	}

	/* run sample */
	result = flow_ct_2_ports(dpdk_config.port_config.nb_queues, ct_dev, ct_cfg.n_ports);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Encountered an error: %s", doca_error_get_descr(result));
		goto dpdk_ports_queues_cleanup;
	}

	exit_status = EXIT_SUCCESS;
dpdk_ports_queues_cleanup:
	dpdk_queues_and_ports_fini(&dpdk_config);
device_cleanup:
	for (i = 0; i < ct_cfg.n_ports; i++)
		if (ct_dev[i] != NULL)
			doca_dev_close(ct_dev[i]);
dpdk_cleanup:
	dpdk_fini();
argp_cleanup:
	doca_argp_destroy();
sample_exit:
	if (exit_status == EXIT_SUCCESS)
		DOCA_LOG_INFO("Sample finished successfully");
	else
		DOCA_LOG_INFO("Sample finished with errors");
	return exit_status;
}