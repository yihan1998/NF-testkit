#include <stdlib.h>

#include "doca.h"

#define WORKQ_DEPTH	128

struct doca_sha_config doca_sha_cfg = {
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

doca_error_t doca_sha_init(void) {
	doca_error_t result;

	/* Open DOCA device */
	result = open_doca_device_with_pci(doca_sha_cfg.pci_address, NULL, &doca_sha_cfg.dev);
	if (result != DOCA_SUCCESS) {
		printf("No device matching PCI address found. Reason: %s", doca_get_error_string(result));
		return result;
	}

	/* Create a DOCA RegEx instance */
	result = doca_sha_create(&(doca_sha_cfg.doca_sha));
	if (result != DOCA_SUCCESS) {
		printf("DOCA SHA creation Failed. Reason: %s", doca_get_error_string(result));
		doca_dev_close(doca_sha_cfg.dev);
		return DOCA_ERROR_INITIALIZATION;
	}

	/* Set hw RegEx device to DOCA RegEx */
	result = doca_ctx_dev_add(doca_sha_as_ctx(doca_sha_cfg.doca_sha), doca_sha_cfg.dev);
	if (result != DOCA_SUCCESS) {
		printf("Unable to install SHA device. Reason: %s", doca_get_error_string(result));
		result = DOCA_ERROR_INITIALIZATION;
		return 0;
	}

	/* Start DOCA RegEx */
	result = doca_ctx_start(doca_sha_as_ctx(doca_sha_cfg.doca_sha));
	if (result != DOCA_SUCCESS) {
		printf("Unable to start DOCA RegEx. Reason: %s", doca_get_error_string(result));
		result = DOCA_ERROR_INITIALIZATION;
		return 0;
	}

	return DOCA_SUCCESS;
}

doca_error_t doca_sha_percore_init(struct doca_sha_ctx * sha_ctx) {
	doca_error_t result;
	char * data_buffer;

	data_buffer = (char *)calloc(8192, sizeof(char));

    sha_ctx->src_data_buffer = data_buffer;
    sha_ctx->src_data_buffer_len = 4096;

	sha_ctx->dst_data_buffer = data_buffer + 4096;
    sha_ctx->dst_data_buffer_len = 4096;

	result = doca_buf_inventory_create(NULL, 2, DOCA_BUF_EXTENSION_NONE, &sha_ctx->buf_inv);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create doca_buf_inventory. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_buf_inventory_start(sha_ctx->buf_inv);
	if (result != DOCA_SUCCESS) {
		printf("Unable to start doca_buf_inventory. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_mmap_create(NULL, &sha_ctx->mmap);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create doca_mmap. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_mmap_dev_add(sha_ctx->mmap, sha_ctx->dev);
	if (result != DOCA_SUCCESS) {
		printf("Unable to add device to doca_mmap. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_mmap_set_memrange(sha_ctx->mmap, data_buffer, 8192);
	if (result != DOCA_SUCCESS) {
		printf("Unable to register src memory with doca_mmap. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	// result = doca_mmap_set_memrange(sha_ctx->mmap, sha_ctx->dst_data_buffer, sha_ctx->dst_data_buffer_len);
	// if (result != DOCA_SUCCESS) {
	// 	printf("Unable to register dest memory with doca_mmap. Reason: %s\n", doca_get_error_string(result));
	// 	return 0;
	// }

	result = doca_mmap_start(sha_ctx->mmap);
	if (result != DOCA_SUCCESS) {
		printf("Unable to start doca_mmap. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	if (doca_buf_inventory_buf_by_addr(sha_ctx->buf_inv, sha_ctx->mmap, sha_ctx->src_data_buffer, sha_ctx->src_data_buffer_len, &sha_ctx->src_buf) != DOCA_SUCCESS) {
        printf("Failed to create inventory buf!\n");
        return 0;
    }

	if (doca_buf_inventory_buf_by_addr(sha_ctx->buf_inv, sha_ctx->mmap, sha_ctx->dst_data_buffer, sha_ctx->dst_data_buffer_len, &sha_ctx->dst_buf) != DOCA_SUCCESS) {
        printf("Failed to create inventory buf!\n");
        return 0;
    }

    return DOCA_SUCCESS;
}


int doca_percore_init(void) {
	doca_sha_percore_init(&ctx->sha_ctx);
    return 0;
}

int doca_worker_init(struct worker_context * ctx) {
	doca_error_t result;

    result = doca_workq_create(WORKQ_DEPTH, &ctx->workq);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create work queue. Reason: %s", doca_get_error_string(result));
		return result;
	}
	ctx->sha_ctx.dev = doca_sha_cfg.dev;
	ctx->sha_ctx.doca_sha = doca_sha_cfg.doca_sha;

    /* Add workq to RegEx */
	result = doca_ctx_workq_add(doca_sha_as_ctx(ctx->sha_ctx.doca_sha), ctx->workq);
	if (result != DOCA_SUCCESS) {
		printf("Unable to attach work queue to REGEX. Reason: %s", doca_get_error_string(result));
		return result;
	}

    return DOCA_SUCCESS;
}

int doca_init(void) {
	doca_sha_init();
	return 0;
}