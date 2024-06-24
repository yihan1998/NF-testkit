#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <zlib.h>

#include "doca.h"
#include "compression.h"

struct compresspkt {
    uint64_t send_start;
    uint32_t len;
    uint8_t image[0];
};

int compress_pkt(uint8_t * data, int len) {
#ifdef CONFIG_DOCA
    struct compresspkt * request;
    int res;
	struct doca_event event = {0};
    void * mbuf_data;
	// uint8_t *resp_head;
	// size_t data_len;
	struct doca_compress_ctx * compress_ctx = &ctx->compress_ctx;

    request = (struct compresspkt *)data;

    memcpy(compress_ctx->src_data_buffer, request->image, request->len);

    doca_buf_get_data(compress_ctx->src_buf, &mbuf_data);
    doca_buf_set_data(compress_ctx->src_buf, mbuf_data, request->len);

	const struct doca_compress_deflate_job compress_job = {
		.base = (struct doca_job) {
			.type = DOCA_COMPRESS_DEFLATE_JOB,
			.flags = DOCA_JOB_FLAGS_NONE,
			.ctx = doca_compress_as_ctx(compress_ctx->doca_compress),
        },
		.dst_buff = compress_ctx->dst_buf,
		.src_buff = compress_ctx->src_buf,
	};

    // struct timespec start, end;
    // clock_gettime(CLOCK_MONOTONIC, &start);

    res = doca_workq_submit(ctx->workq, (struct doca_job *)&compress_job);
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

    // doca_buf_get_data(compress_job.dst_buff, (void **)&resp_head);
    // doca_buf_get_data_len(compress_job.dst_buff, &data_len);

    doca_buf_reset_data_len(compress_ctx->dst_buf);

	return 0;
#else
    struct compresspkt * request;
    request = (struct compresspkt *)data;

    uLong output_size = compressBound(request->len);
    Bytef * output = (Bytef *)malloc(output_size);
    if (output == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        return -1;
    }

    if (compress((Bytef *)output, (uLongf *)&output_size, (const Bytef *)request->image, request->len) != Z_OK) {
        fprintf(stderr, "Compression failed\n");
        free(output);
    }

    // printf("Compress %u B -> %u B!\n", size, output_size);

    free(output);
    return 0;
#endif
}