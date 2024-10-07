/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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
#include <string.h>

#include <doca_argp.h>
#include <doca_dev.h>
#include <doca_log.h>

#include "dma_common.h"

DOCA_LOG_REGISTER(DMA_COPY_HOST::MAIN);

/* Sample's Logic */
doca_error_t dma_copy_host(const char *pcie_addr,
			   char *src_buffer,
			   size_t src_buffer_size,
			   char *export_desc_file_path,
			   char *buffer_info_file_path);

/*
 * Sample main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int main(int argc, char **argv)
{
	struct dma_config dma_conf;
	char *src_buffer;
	size_t length;
	doca_error_t result;
	struct doca_log_backend *sdk_log;
	int exit_status = EXIT_FAILURE;

	/* Set the default configuration values (Example values) */
	strcpy(dma_conf.pci_address, "b1:00.0");
	strcpy(dma_conf.cpy_txt, "This is a sample piece of text");
	strcpy(dma_conf.export_desc_path, "/tmp/export_desc.txt");
	strcpy(dma_conf.buf_info_path, "/tmp/buffer_info.txt");

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

#ifndef DOCA_ARCH_HOST
	DOCA_LOG_ERR("Sample can run only on the Host");
	goto sample_exit;
#endif

	result = doca_argp_init("doca_dma_copy_host", &dma_conf);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		goto sample_exit;
	}
	result = register_dma_params(true);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register DMA sample parameters: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}
	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse sample input: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	length = strlen(dma_conf.cpy_txt) + 1;
	src_buffer = (char *)malloc(length);
	if (src_buffer == NULL) {
		DOCA_LOG_ERR("Source buffer allocation failed");
		goto argp_cleanup;
	}

	memcpy(src_buffer, dma_conf.cpy_txt, length);

	result = dma_copy_host(dma_conf.pci_address,
			       src_buffer,
			       length,
			       dma_conf.export_desc_path,
			       dma_conf.buf_info_path);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("dma_copy_host() encountered an error: %s", doca_error_get_descr(result));
		goto src_buf_cleanup;
	}

	exit_status = EXIT_SUCCESS;

src_buf_cleanup:
	if (src_buffer != NULL)
		free(src_buffer);
argp_cleanup:
	doca_argp_destroy();
sample_exit:
	if (exit_status == EXIT_SUCCESS)
		DOCA_LOG_INFO("Sample finished successfully");
	else
		DOCA_LOG_INFO("Sample finished with errors");
	return exit_status;
}
