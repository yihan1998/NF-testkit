#include <stdlib.h>

#include "doca.h"

#define WORKQ_DEPTH	128

struct doca_compress_config doca_compress_cfg = {
	.pci_address = "03:00.0",
};

struct worker_context worker_ctx[NR_CPUS];
__thread struct worker_context * ctx;

noreturn doca_error_t sdk_version_callback(void *param, void *doca_config) {
	(void)(param);
	(void)(doca_config);

	printf("DOCA SDK     Version (Compilation): %s\n", doca_version());
	printf("DOCA Runtime Version (Runtime):     %s\n", doca_version_runtime());
	/* We assume that when printing DOCA's versions there is no need to continue the program's execution */
	exit(EXIT_SUCCESS);
}

doca_error_t open_doca_device_with_pci(const char *pci_addr, jobs_check func, struct doca_dev **retval) {
	struct doca_devinfo **dev_list;
	uint32_t nb_devs;
	uint8_t is_addr_equal = 0;
	int res;
	size_t i;

	/* Set default return value */
	*retval = NULL;

	res = doca_devinfo_list_create(&dev_list, &nb_devs);
	if (res != DOCA_SUCCESS) {
		printf("Failed to load doca devices list. Doca_error value: %d\n", res);
		return res;
	}

	/* Search */
	for (i = 0; i < nb_devs; i++) {
		res = doca_devinfo_get_is_pci_addr_equal(dev_list[i], pci_addr, &is_addr_equal);
		if (res == DOCA_SUCCESS && is_addr_equal) {
			/* If any special capabilities are needed */
			if (func != NULL && func(dev_list[i]) != DOCA_SUCCESS)
				continue;

			/* if device can be opened */
			res = doca_dev_open(dev_list[i], retval);
			if (res == DOCA_SUCCESS) {
				doca_devinfo_list_destroy(dev_list);
				return res;
			}
		}
	}

	printf("Matching device not found\n");
	res = DOCA_ERROR_NOT_FOUND;

	doca_devinfo_list_destroy(dev_list);
	return res;
}

doca_error_t doca_compress_init(void) {
	doca_error_t result;

	/* Open DOCA device */
	result = open_doca_device_with_pci(doca_compress_cfg.pci_address, NULL, &doca_compress_cfg.dev);
	if (result != DOCA_SUCCESS) {
		printf("No device matching PCI address found. Reason: %s", doca_get_error_string(result));
		return result;
	}

	/* Create a DOCA RegEx instance */
	result = doca_compress_create(&(doca_compress_cfg.doca_compress));
	if (result != DOCA_SUCCESS) {
		printf("DOCA SHA creation Failed. Reason: %s", doca_get_error_string(result));
		doca_dev_close(doca_compress_cfg.dev);
		return DOCA_ERROR_INITIALIZATION;
	}

	/* Set hw RegEx device to DOCA RegEx */
	result = doca_ctx_dev_add(doca_compress_as_ctx(doca_compress_cfg.doca_compress), doca_compress_cfg.dev);
	if (result != DOCA_SUCCESS) {
		printf("Unable to install SHA device. Reason: %s", doca_get_error_string(result));
		result = DOCA_ERROR_INITIALIZATION;
		return 0;
	}

	/* Start DOCA RegEx */
	result = doca_ctx_start(doca_compress_as_ctx(doca_compress_cfg.doca_compress));
	if (result != DOCA_SUCCESS) {
		printf("Unable to start DOCA RegEx. Reason: %s", doca_get_error_string(result));
		result = DOCA_ERROR_INITIALIZATION;
		return 0;
	}

	return DOCA_SUCCESS;
}

doca_error_t doca_compress_percore_init(struct doca_compress_ctx * compress_ctx) {
	doca_error_t result;
	char * data_buffer;

	data_buffer = (char *)calloc(8192, sizeof(char));

    compress_ctx->src_data_buffer = data_buffer;
    compress_ctx->src_data_buffer_len = 4096;

	compress_ctx->dst_data_buffer = data_buffer + 4096;
    compress_ctx->dst_data_buffer_len = 4096;

	result = doca_buf_inventory_create(NULL, 2, DOCA_BUF_EXTENSION_NONE, &compress_ctx->buf_inv);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create doca_buf_inventory. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_buf_inventory_start(compress_ctx->buf_inv);
	if (result != DOCA_SUCCESS) {
		printf("Unable to start doca_buf_inventory. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_mmap_create(NULL, &compress_ctx->mmap);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create doca_mmap. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_mmap_dev_add(compress_ctx->mmap, compress_ctx->dev);
	if (result != DOCA_SUCCESS) {
		printf("Unable to add device to doca_mmap. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_mmap_set_memrange(compress_ctx->mmap, data_buffer, 8192);
	if (result != DOCA_SUCCESS) {
		printf("Unable to register src memory with doca_mmap. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	// result = doca_mmap_set_memrange(compress_ctx->mmap, compress_ctx->dst_data_buffer, compress_ctx->dst_data_buffer_len);
	// if (result != DOCA_SUCCESS) {
	// 	printf("Unable to register dest memory with doca_mmap. Reason: %s\n", doca_get_error_string(result));
	// 	return 0;
	// }

	result = doca_mmap_start(compress_ctx->mmap);
	if (result != DOCA_SUCCESS) {
		printf("Unable to start doca_mmap. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	if (doca_buf_inventory_buf_by_addr(compress_ctx->buf_inv, compress_ctx->mmap, compress_ctx->src_data_buffer, compress_ctx->src_data_buffer_len, &compress_ctx->src_buf) != DOCA_SUCCESS) {
        printf("Failed to create inventory buf!\n");
        return 0;
    }

	if (doca_buf_inventory_buf_by_addr(compress_ctx->buf_inv, compress_ctx->mmap, compress_ctx->dst_data_buffer, compress_ctx->dst_data_buffer_len, &compress_ctx->dst_buf) != DOCA_SUCCESS) {
        printf("Failed to create inventory buf!\n");
        return 0;
    }

    return DOCA_SUCCESS;
}


int doca_percore_init(void) {
	doca_compress_percore_init(&ctx->compress_ctx);
    return 0;
}

int doca_worker_init(struct worker_context * ctx) {
	doca_error_t result;

    result = doca_workq_create(WORKQ_DEPTH, &ctx->workq);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create work queue. Reason: %s", doca_get_error_string(result));
		return result;
	}
	ctx->compress_ctx.dev = doca_compress_cfg.dev;
	ctx->compress_ctx.doca_compress = doca_compress_cfg.doca_compress;

    /* Add workq to RegEx */
	result = doca_ctx_workq_add(doca_compress_as_ctx(ctx->compress_ctx.doca_compress), ctx->workq);
	if (result != DOCA_SUCCESS) {
		printf("Unable to attach work queue to REGEX. Reason: %s", doca_get_error_string(result));
		return result;
	}

    return DOCA_SUCCESS;
}

int doca_init(void) {
	doca_compress_init();
	return 0;
}