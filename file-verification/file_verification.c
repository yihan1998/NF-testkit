#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "doca.h"
#include "file_verification.h"

struct sha256pkt {
    uint8_t hash[SHA256_DIGEST_LENGTH];
    uint64_t send_start;
    uint64_t completion_time;
    uint32_t len;
    uint8_t data[0];
};

void handleErrors(void) {
    ERR_print_errors_fp(stderr);
    abort();
}

void calculate_sha256(EVP_MD_CTX *mdctx, const unsigned char *data, size_t data_len, unsigned char *hash) {
    if(1 != EVP_DigestInit_ex(mdctx, EVP_sha256(), NULL))
        handleErrors();

    if(1 != EVP_DigestUpdate(mdctx, data, data_len))
        handleErrors();

    unsigned int hash_len;
    if(1 != EVP_DigestFinal_ex(mdctx, hash, &hash_len))
        handleErrors();
}

int verify_pkt(uint8_t * data, int len) {
#ifdef CONFIG_DOCA
    struct sha256pkt * request;
    int res;
	struct doca_event event = {0};
    void * mbuf_data;
	uint8_t *resp_head;
	struct doca_sha_ctx * sha_ctx = &ctx->sha_ctx;
	// char orig_sha[DOCA_SHA256_BYTE_COUNT * 2 + 1] = {0};
	// char sha_output[DOCA_SHA256_BYTE_COUNT * 2 + 1] = {0};

    request = (struct sha256pkt *)data;

    memcpy(sha_ctx->src_data_buffer, request->data, request->len);

    doca_buf_get_data(sha_ctx->src_buf, &mbuf_data);
    doca_buf_set_data(sha_ctx->src_buf, mbuf_data, request->len);

    /* Construct sha partial job */
	struct doca_sha_job sha_job = {
		.base = (struct doca_job) {
			.type = DOCA_SHA_JOB_SHA256,
			.flags = DOCA_JOB_FLAGS_NONE,
			.ctx = doca_sha_as_ctx(sha_ctx->doca_sha),
        },
		.resp_buf = sha_ctx->dst_buf,
		.req_buf = sha_ctx->src_buf,
		.flags = DOCA_SHA_JOB_FLAGS_SHA_PARTIAL_FINAL,
	};

    // double elapsed_time;
    // struct timespec start, end;
    // clock_gettime(CLOCK_MONOTONIC, &start);

    res = doca_workq_submit(ctx->workq, (struct doca_job *)&sha_job);
    if (res != DOCA_SUCCESS) {
        printf("Unable to enqueue job. [%s]\n", doca_get_error_string(res));
        return -1;
    }

    do {
		res = doca_workq_progress_retrieve(ctx->workq, &event, DOCA_WORKQ_RETRIEVE_FLAGS_NONE);
	    if (res != DOCA_SUCCESS) {
            if (res == DOCA_ERROR_AGAIN) {
                continue;
            } else {
                printf("Unable to dequeue results. [%s]\n", doca_get_error_string(res));
            }
        }
	} while (res != DOCA_SUCCESS);

    // clock_gettime(CLOCK_MONOTONIC, &end);
    // elapsed_time = (end.tv_sec - start.tv_sec) * 1e9;    // seconds to nanoseconds
    // elapsed_time += (end.tv_nsec - start.tv_nsec);       // add nanoseconds
    // fprintf(stderr, "%.9f\n", elapsed_time);

    doca_buf_get_data(sha_job.resp_buf, (void **)&resp_head);

    // char original[DOCA_SHA256_BYTE_COUNT * 2 + 1] = {0};
	// for (int i = 0; i < DOCA_SHA256_BYTE_COUNT; i++)
	// 	snprintf(original + (2 * i), 3, "%02x", request->hash[i]);
	// printf("Sent SHA is: %s\n", original);

	// char sha_output[DOCA_SHA256_BYTE_COUNT * 2 + 1] = {0};
	// for (int i = 0; i < DOCA_SHA256_BYTE_COUNT; i++)
	// 	snprintf(sha_output + (2 * i), 3, "%02x", resp_head[i]);
	// printf("SHA256 output is: %s\n", sha_output);

    if (memcmp(resp_head, request->hash, DOCA_SHA256_BYTE_COUNT) != 0) {
        printf("Verification failed\n");
    }

    doca_buf_reset_data_len(sha_ctx->dst_buf);

	return 0;
#else
    struct sha256pkt * request;
    unsigned char hash[EVP_MAX_MD_SIZE];

    request = (struct sha256pkt *)data;

    calculate_sha256(mdctx, request->data, request->len, hash);

    if (memcmp(hash, request->hash, DOCA_SHA256_BYTE_COUNT) != 0) {
        printf("Verification failed\n");
    }
#endif
}