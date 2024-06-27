#include <stdlib.h>
#include <regex.h>

#include "doca.h"

#define WORKQ_DEPTH	128

struct doca_regex_config doca_regex_cfg = {
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

doca_error_t read_file(char const *path, char **out_bytes, size_t *out_bytes_len) {
	FILE *file;
	char *bytes;

	file = fopen(path, "rb");
	if (file == NULL)
		return DOCA_ERROR_NOT_FOUND;

	if (fseek(file, 0, SEEK_END) != 0) {
		fclose(file);
		return DOCA_ERROR_IO_FAILED;
	}

	long const nb_file_bytes = ftell(file);

	if (nb_file_bytes == -1) {
		fclose(file);
		return DOCA_ERROR_IO_FAILED;
	}

	if (nb_file_bytes == 0) {
		fclose(file);
		return DOCA_ERROR_INVALID_VALUE;
	}

	bytes = malloc(nb_file_bytes);
	if (bytes == NULL) {
		fclose(file);
		return DOCA_ERROR_NO_MEMORY;
	}

	if (fseek(file, 0, SEEK_SET) != 0) {
		free(bytes);
		fclose(file);
		return DOCA_ERROR_IO_FAILED;
	}

	size_t const read_byte_count = fread(bytes, 1, nb_file_bytes, file);

	fclose(file);

	if (read_byte_count != (size_t)nb_file_bytes) {
		free(bytes);
		return DOCA_ERROR_IO_FAILED;
	}

	*out_bytes = bytes;
	*out_bytes_len = read_byte_count;

	return DOCA_SUCCESS;
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

doca_error_t doca_regex_init(void) {
	doca_error_t result;
    char *rules_file_data;
	size_t rules_file_size;

	/* Open DOCA device */
	result = open_doca_device_with_pci(doca_regex_cfg.pci_address, NULL, &doca_regex_cfg.dev);
	if (result != DOCA_SUCCESS) {
		printf("No device matching PCI address found. Reason: %s", doca_get_error_string(result));
		return result;
	}

	/* Create a DOCA RegEx instance */
	result = doca_regex_create(&(doca_regex_cfg.doca_reg));
	if (result != DOCA_SUCCESS) {
		printf("DOCA RegEx creation Failed. Reason: %s", doca_get_error_string(result));
		doca_dev_close(doca_regex_cfg.dev);
		return DOCA_ERROR_INITIALIZATION;
	}

	/* Set hw RegEx device to DOCA RegEx */
	result = doca_ctx_dev_add(doca_regex_as_ctx(doca_regex_cfg.doca_reg), doca_regex_cfg.dev);
	if (result != DOCA_SUCCESS) {
		printf("Unable to install RegEx device. Reason: %s", doca_get_error_string(result));
		result = DOCA_ERROR_INITIALIZATION;
		return 0;
	}
	/* Set matches memory pool to 0 because the app needs to check if there are matches and don't need the matches details  */
	result = doca_regex_set_workq_matches_memory_pool_size(doca_regex_cfg.doca_reg, 0);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create match memory pools. Reason: %s", doca_get_error_string(result));
		return 0;
	}

	/* Start DOCA RegEx */
	result = doca_ctx_start(doca_regex_as_ctx(doca_regex_cfg.doca_reg));
	if (result != DOCA_SUCCESS) {
		printf("Unable to start DOCA RegEx. Reason: %s", doca_get_error_string(result));
		result = DOCA_ERROR_INITIALIZATION;
		return 0;
	}

	result = read_file("/tmp/dns_filter_rules.rof2.binary", &rules_file_data, &rules_file_size);
	if (result != DOCA_SUCCESS) {
		printf("Unable to load rules file content. Reason: %s", doca_get_error_string(result));
		return -1;
	}

	result = doca_regex_set_hardware_compiled_rules(doca_regex_cfg.doca_reg, rules_file_data, rules_file_size);
	if (result != DOCA_SUCCESS) {
		printf("Unable to program rules. Reason: %s", doca_get_error_string(result));
		free(rules_file_data);
		return -1;
	}
	free(rules_file_data);

	return DOCA_SUCCESS;
}

doca_error_t doca_regex_percore_init(struct doca_regex_ctx * regex_ctx) {
	doca_error_t result;

    regex_ctx->data_buffer = (char *)calloc(1024, sizeof(char));
    regex_ctx->data_buffer_len = 1024;

	result = doca_buf_inventory_create(NULL, 1, DOCA_BUF_EXTENSION_NONE, &regex_ctx->buf_inv);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create doca_buf_inventory. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_buf_inventory_start(regex_ctx->buf_inv);
	if (result != DOCA_SUCCESS) {
		printf("Unable to start doca_buf_inventory. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_mmap_create(NULL, &regex_ctx->mmap);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create doca_mmap. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_mmap_dev_add(regex_ctx->mmap, regex_ctx->dev);
	if (result != DOCA_SUCCESS) {
		printf("Unable to add device to doca_mmap. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_mmap_set_memrange(regex_ctx->mmap, regex_ctx->data_buffer, regex_ctx->data_buffer_len);
	if (result != DOCA_SUCCESS) {
		printf("Unable to register memory with doca_mmap. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	result = doca_mmap_start(regex_ctx->mmap);
	if (result != DOCA_SUCCESS) {
		printf("Unable to start doca_mmap. Reason: %s\n", doca_get_error_string(result));
		return 0;
	}

	if (doca_buf_inventory_buf_by_addr(regex_ctx->buf_inv, regex_ctx->mmap, regex_ctx->data_buffer, regex_ctx->data_buffer_len, &regex_ctx->buf) != DOCA_SUCCESS) {
        printf("Failed to create inventory buf!\n");
        return 0;
    }

    return DOCA_SUCCESS;
}


int doca_percore_init(void) {
	doca_regex_percore_init(&ctx->regex_ctx);
    return 0;
}

int doca_worker_init(struct worker_context * ctx) {
	doca_error_t result;

    result = doca_workq_create(WORKQ_DEPTH, &ctx->workq);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create work queue. Reason: %s", doca_get_error_string(result));
		return result;
	}
	ctx->regex_ctx.dev = doca_regex_cfg.dev;
	ctx->regex_ctx.doca_reg = doca_regex_cfg.doca_reg;

    /* Add workq to RegEx */
	result = doca_ctx_workq_add(doca_regex_as_ctx(ctx->regex_ctx.doca_reg), ctx->workq);
	if (result != DOCA_SUCCESS) {
		printf("Unable to attach work queue to REGEX. Reason: %s", doca_get_error_string(result));
		return result;
	}

    return DOCA_SUCCESS;
}

int doca_init(void) {
	doca_regex_init();
	return 0;
}