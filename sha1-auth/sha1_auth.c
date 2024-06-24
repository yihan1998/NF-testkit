#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "doca.h"
#include "sha1_auth.h"

struct sha1pkt {
    uint8_t hash[SHA_DIGEST_LENGTH];
    uint64_t send_start;
    uint64_t completion_time;
    uint32_t len;
    uint8_t data[0];
};

void handleErrors(void) {
    ERR_print_errors_fp(stderr);
    abort();
}

void calculate_sha1(EVP_MD_CTX *mdctx, const unsigned char *data, size_t data_len, unsigned char *hash) {
    if(1 != EVP_DigestInit_ex(mdctx, EVP_sha1(), NULL))
        handleErrors();

    if(1 != EVP_DigestUpdate(mdctx, data, data_len))
        handleErrors();

    unsigned int hash_len;
    if(1 != EVP_DigestFinal_ex(mdctx, hash, &hash_len))
        handleErrors();
}

int verify_pkt(uint8_t * data, int len) {
#if defined(CONFIG_DOCA)
    struct sha1pkt * request;
    int res;
	struct doca_event event = {0};
    void * mbuf_data;
	uint8_t *resp_head;
	struct doca_sha_ctx * sha_ctx = &ctx->sha_ctx;
	union doca_data task_user_data = {0};

    request = (struct sha1pkt *)data;

    memcpy(sha_ctx->src_data_buffer, request->data, request->len);

    doca_buf_get_data(sha_ctx->src_buf, &mbuf_data);
    doca_buf_set_data(sha_ctx->src_buf, mbuf_data, request->len);

    /* Include result in user data of task to be used in the callbacks */
	task_user_data.ptr = &task_result;
	/* Allocate and construct SHA hash task */
	result = doca_sha_task_hash_alloc_init(sha_ctx,
					       DOCA_SHA_ALGORITHM_SHA1,
					       src_doca_buf,
					       dst_doca_buf,
					       task_user_data,
					       &sha_hash_task);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to allocate SHA hash task: %s", doca_error_get_descr(result));
		doca_buf_dec_refcount(dst_doca_buf, NULL);
		doca_buf_dec_refcount(src_doca_buf, NULL);
		sha_cleanup(&resources);
		return result;
	}
	/* Number of tasks submitted to progress engine */
	resources.num_remaining_tasks = 1;

	task = doca_sha_task_hash_as_task(sha_hash_task);
	if (task == NULL) {
		result = DOCA_ERROR_UNEXPECTED;
		DOCA_LOG_ERR("Failed to get DOCA SHA hash task as DOCA task: %s", doca_error_get_descr(result));
		doca_task_free(task);
		doca_buf_dec_refcount(dst_doca_buf, NULL);
		doca_buf_dec_refcount(src_doca_buf, NULL);
		sha_cleanup(&resources);
		return result;
	}

	/* Submit SHA hash task */
	result = doca_task_submit(task);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to submit SHA hash task: %s", doca_error_get_descr(result));
		doca_task_free(task);
		doca_buf_dec_refcount(dst_doca_buf, NULL);
		doca_buf_dec_refcount(src_doca_buf, NULL);
		sha_cleanup(&resources);
		return result;
	}

	resources.run_pe_progress = true;

	/* Wait for all tasks to be completed and for the context to be stopped */
	while (resources.run_pe_progress) {
		if (doca_pe_progress(state->pe) == 0)
			nanosleep(&ts, &ts);
	}

	result = task_result;

    // clock_gettime(CLOCK_MONOTONIC, &end);
    // elapsed_time = (end.tv_sec - start.tv_sec) * 1e9;    // seconds to nanoseconds
    // elapsed_time += (end.tv_nsec - start.tv_nsec);       // add nanoseconds
    // fprintf(stderr, "%.9f\n", elapsed_time);

    doca_buf_get_data(sha_job.resp_buf, (void **)&resp_head);

    // char original[DOCA_SHA1_BYTE_COUNT * 2 + 1] = {0};
	// for (int i = 0; i < DOCA_SHA1_BYTE_COUNT; i++)
	// 	snprintf(original + (2 * i), 3, "%02x", request->hash[i]);
	// printf("Sent SHA is: %s\n", original);

	// char sha_output[DOCA_SHA1_BYTE_COUNT * 2 + 1] = {0};
	// for (int i = 0; i < DOCA_SHA1_BYTE_COUNT; i++)
	// 	snprintf(sha_output + (2 * i), 3, "%02x", resp_head[i]);
	// printf("SHA256 output is: %s\n", sha_output);

    if (memcmp(resp_head, request->hash, DOCA_SHA1_BYTE_COUNT) != 0) {
        printf("Verification failed\n");
    }

    doca_buf_reset_data_len(sha_ctx->dst_buf);

	return 0;
#elif defined(CONFIG_RTE_CRYPTO)

#else
    struct sha1pkt * request;
    unsigned char hash[EVP_MAX_MD_SIZE];

    request = (struct sha1pkt *)data;

    calculate_sha1(mdctx, request->data, request->len, hash);

    if (memcmp(hash, request->hash, DOCA_SHA1_BYTE_COUNT) != 0) {
        printf("Verification failed\n");
    }
#endif
}