#include <infiniband/mlx5_api.h>

#include <assert.h>
#include <bsd/string.h>

#include <doca_log.h>
#include <doca_error.h>

#include "common.h"
#include "queue.h"

#define LOG2VALUE(l) (1UL << (l)) /* 2^l */

static int init_dpa_rq_ring(struct flexio_process *process,
				     flexio_uintptr_t ring_daddr,
				     int log_depth,
				     flexio_uintptr_t data_daddr,
				     int log_chunk_bsize,
				     uint32_t wqd_mkey_id)
{
	struct mlx5_wqe_data_seg *rx_wqes, *dseg;
	size_t data_chunk_bsize;
	size_t ring_bsize;
	int num_of_wqes;
	int result = 0;
	int i;

	num_of_wqes = LOG2VALUE(log_depth);
	ring_bsize = num_of_wqes * sizeof(struct mlx5_wqe_data_seg);
	data_chunk_bsize = LOG2VALUE(log_chunk_bsize);

	rx_wqes = calloc(num_of_wqes, sizeof(struct mlx5_wqe_data_seg));

	if (rx_wqes == NULL) {
		printf("Failed to allocate memory for RQ WQEs\n");
		return -1;
	}

	/* Initialize WQEs' data segment */
	dseg = rx_wqes;

	for (i = 0; i < num_of_wqes; i++) {
		mlx5dv_set_data_seg(dseg, data_chunk_bsize, wqd_mkey_id, data_daddr);
		dseg++;
		data_daddr += data_chunk_bsize;
	}

	/* Copy RX WQEs from host to FlexIO RQ ring */
	if (flexio_host2dev_memcpy(process, rx_wqes, ring_bsize, ring_daddr) != FLEXIO_STATUS_SUCCESS) {
		printf("Failed to copy RX WQEs from host to FlexIO RQ ring\n");
		return -1;
	}

	free(rx_wqes);
	return result;
}

static int create_dpa_mkey(struct flexio_process *process,
				    struct ibv_pd *pd,
				    flexio_uintptr_t daddr,
				    int log_bsize,
				    int access,
				    struct flexio_mkey **mkey)
{
	struct flexio_mkey_attr mkey_attr = {0};

	mkey_attr.pd = pd;
	mkey_attr.daddr = daddr;
	mkey_attr.len = LOG2VALUE(log_bsize);
	mkey_attr.access = access;
	if (flexio_device_mkey_create(process, &mkey_attr, mkey) != FLEXIO_STATUS_SUCCESS) {
		printf("Failed to create MKey\n");
		return -1;
	}

	return 0;
}

static int allocate_dbr(struct flexio_process *process, flexio_uintptr_t *dbr_daddr)
{
	__be32 dbr[2] = {0, 0};

	if (flexio_copy_from_host(process, dbr, sizeof(dbr), dbr_daddr) != FLEXIO_STATUS_SUCCESS) {
		printf("Failed to copy DBR to device memory\n");
		return -1;
	}
	return 0;
}

static int allocate_cq_memory(struct flexio_process *process, int log_depth, struct app_transfer_cq *app_cq)
{
	struct mlx5_cqe64 *cq_ring_src, *cqe;
	size_t ring_bsize;
	int i, num_of_cqes;
	const int log_cqe_bsize = 6; /* CQE size is 64 bytes */
	int result = 0;
	flexio_status ret;

	/* Allocate DB record */
	result = allocate_dbr(process, &app_cq->cq_dbr_daddr);
	if (result < 0) {
		printf("Failed to allocate CQ DB record\n");
		return -1;
	}

	num_of_cqes = LOG2VALUE(log_depth);
	ring_bsize = num_of_cqes * LOG2VALUE(log_cqe_bsize);

	cq_ring_src = calloc(num_of_cqes, LOG2VALUE(log_cqe_bsize));

	if (cq_ring_src == NULL) {
		printf("Failed to allocate CQ ring");
		return -1;
	}

	cqe = cq_ring_src;
	for (i = 0; i < num_of_cqes; i++)
		mlx5dv_set_cqe_owner(cqe++, 1);

	/* Copy CQEs from host to FlexIO CQ ring */
	ret = flexio_copy_from_host(process, cq_ring_src, ring_bsize, &app_cq->cq_ring_daddr);
	free(cq_ring_src);
	if (ret) {
		printf("Failed to allocate CQ ring\n");
		return -1;
	}

	return 0;
}

static int allocate_sq_memory(struct flexio_process *process,
				       int log_depth,
				       int log_data_bsize,
				       struct app_transfer_wq *sq_transf)
{
	const int log_wqe_bsize = 6; /* WQE size is 64 bytes */
	int result;

	if (flexio_buf_dev_alloc(process, LOG2VALUE(log_data_bsize), &sq_transf->wqd_daddr) != FLEXIO_STATUS_SUCCESS) {
		printf("Failed to allocate SQ data buffer\n");
		return -1;
	}

	if (flexio_buf_dev_alloc(process, LOG2VALUE(log_depth + log_wqe_bsize), &sq_transf->wq_ring_daddr) !=
	    FLEXIO_STATUS_SUCCESS) {
		printf("Failed to allocate SQ ring\n");
		return -1;
	}

	result = allocate_dbr(process, &sq_transf->wq_dbr_daddr);
	if (result < 0) {
		printf("Failed to allocate SQ DB record\n");
		return result;
	}

	return 0;
}

int allocate_rq(struct app_config *app_cfg, struct dpa_process_context * ctx)
{
	int result;
	flexio_status ret;
	uint32_t mkey_id;
	uint32_t cq_num;	/* CQ number */
	uint32_t wq_num;	/* WQ number */
	uint32_t log_rqd_bsize; /* SQ data buffer size */

	/* RQ's CQ attributes */
	struct flexio_cq_attr rqcq_attr = {.log_cq_depth = LOG_CQ_RING_DEPTH,
					   .element_type = FLEXIO_CQ_ELEMENT_TYPE_DPA_THREAD,
					   .thread = flexio_event_handler_get_thread(ctx->event_handler),
					   .uar_id = flexio_uar_get_id(app_cfg->flexio_uar),
					   .uar_base_addr = 0};
	/* RQ attributes */
	struct flexio_wq_attr rq_attr = {.log_wq_depth = LOG_RQ_RING_DEPTH, .pd = app_cfg->pd};

	/* Allocate memory for RQ's CQ */
	result = allocate_cq_memory(app_cfg->flexio_process, LOG_CQ_RING_DEPTH, &ctx->rq_cq_transf);
	if (result < 0)
		return result;

	rqcq_attr.cq_dbr_daddr = ctx->rq_cq_transf.cq_dbr_daddr;
	rqcq_attr.cq_ring_qmem.daddr = ctx->rq_cq_transf.cq_ring_daddr;

	/* Create CQ and RQ */
	ret = flexio_cq_create(app_cfg->flexio_process, NULL, &rqcq_attr, &ctx->flexio_rq_cq_ptr);
	if (ret != FLEXIO_STATUS_SUCCESS) {
		printf("Failed to create FlexIO RQ's CQ\n");
		return -1;
	}

	cq_num = flexio_cq_get_cq_num(ctx->flexio_rq_cq_ptr);
	ctx->rq_cq_transf.cq_num = cq_num;
	ctx->rq_cq_transf.log_cq_depth = LOG_RQ_RING_DEPTH;

	log_rqd_bsize = LOG_RQ_RING_DEPTH + LOG_WQ_DATA_ENTRY_BSIZE;

	flexio_buf_dev_alloc(app_cfg->flexio_process, LOG2VALUE(log_rqd_bsize), &ctx->rq_transf.wqd_daddr);
	if (ctx->rq_transf.wqd_daddr == 0) {
		printf("Failed to allocate memory for RQ data buffer\n");
		return -1;
	}

	flexio_buf_dev_alloc(app_cfg->flexio_process,
					LOG2VALUE(LOG_CQ_RING_DEPTH) * sizeof(struct mlx5_wqe_data_seg),
					&ctx->rq_transf.wq_ring_daddr);
	if (ctx->rq_transf.wq_ring_daddr == 0x0) {
		printf("Failed to allocate memory for RQ ring buffer\n");
		return -1;
	}

	result = allocate_dbr(app_cfg->flexio_process, &ctx->rq_transf.wq_dbr_daddr);
	if (result != DOCA_SUCCESS)
		return result;

	/* Create an MKey for RX buffer */
	result = create_dpa_mkey(app_cfg->flexio_process,
				 app_cfg->pd,
				 ctx->rq_transf.wqd_daddr,
				 log_rqd_bsize,
				 IBV_ACCESS_LOCAL_WRITE,
				 &ctx->rqd_mkey);
	if (result != DOCA_SUCCESS)
		return result;

	mkey_id = flexio_mkey_get_id(ctx->rqd_mkey);

	result = init_dpa_rq_ring(app_cfg->flexio_process,
				  ctx->rq_transf.wq_ring_daddr,
				  LOG_CQ_RING_DEPTH,
				  ctx->rq_transf.wqd_daddr,
				  LOG_WQ_DATA_ENTRY_BSIZE,
				  mkey_id);
	if (result != DOCA_SUCCESS)
		return result;

	rq_attr.wq_dbr_qmem.memtype = FLEXIO_MEMTYPE_DPA;
	rq_attr.wq_dbr_qmem.daddr = ctx->rq_transf.wq_dbr_daddr;
	rq_attr.wq_ring_qmem.daddr = ctx->rq_transf.wq_ring_daddr;

	ret = flexio_rq_create(app_cfg->flexio_process, NULL, cq_num, &rq_attr, &ctx->flexio_rq_ptr);
	if (ret != FLEXIO_STATUS_SUCCESS) {
		printf("Failed to create FlexIO RQ\n");
		return -1;
	}

	wq_num = flexio_rq_get_wq_num(ctx->flexio_rq_ptr);
	ctx->rq_transf.wqd_mkey_id = mkey_id;
	ctx->rq_transf.wq_num = wq_num;

	/* Modify RQ's DBR record to count for the number of WQEs */
	__be32 dbr[2];
	uint32_t rcv_counter = LOG2VALUE(LOG_RQ_RING_DEPTH);
	uint32_t send_counter = 0;

	dbr[0] = htobe32(rcv_counter & 0xffff);
	dbr[1] = htobe32(send_counter & 0xffff);

	ret = flexio_host2dev_memcpy(app_cfg->flexio_process, dbr, sizeof(dbr), ctx->rq_transf.wq_dbr_daddr);
	if (ret != FLEXIO_STATUS_SUCCESS) {
		printf("Failed to modify RQ's DBR\n");
		return -1;
	}

	return 0;
}

int allocate_sq(struct app_config *app_cfg, struct dpa_process_context * ctx)
{
	int result;
	flexio_status ret;
	uint32_t cq_num;	/* CQ number */
	uint32_t log_sqd_bsize; /* SQ data buffer size */

	/* SQ's CQ attributes */
	struct flexio_cq_attr sqcq_attr = {.log_cq_depth = LOG_CQ_RING_DEPTH,
					   /* SQ does not need APU CQ */
					   .element_type = FLEXIO_CQ_ELEMENT_TYPE_NON_DPA_CQ,
					   .uar_id = flexio_uar_get_id(app_cfg->flexio_uar),
					   .uar_base_addr = 0};
	/* SQ attributes */
	struct flexio_wq_attr sq_attr = {.log_wq_depth = LOG_SQ_RING_DEPTH,
					 .uar_id = flexio_uar_get_id(app_cfg->flexio_uar),
					 .pd = app_cfg->pd};

	/* Allocate memory for SQ's CQ */
	result = allocate_cq_memory(app_cfg->flexio_process, LOG_CQ_RING_DEPTH, &ctx->sq_cq_transf);
	if (result < 0)
		return result;

	sqcq_attr.cq_dbr_daddr = ctx->sq_cq_transf.cq_dbr_daddr;
	sqcq_attr.cq_ring_qmem.daddr = ctx->sq_cq_transf.cq_ring_daddr;

	/* Create SQ's CQ */
	ret = flexio_cq_create(app_cfg->flexio_process, app_cfg->ibv_ctx, &sqcq_attr, &ctx->flexio_sq_cq_ptr);
	if (ret != FLEXIO_STATUS_SUCCESS) {
		printf("Failed to create FlexIO SQ's CQ\n");
		return -1;
	}

	cq_num = flexio_cq_get_cq_num(ctx->flexio_sq_cq_ptr);
	ctx->sq_cq_transf.cq_num = cq_num;
	ctx->sq_cq_transf.log_cq_depth = LOG_CQ_RING_DEPTH;

	/* Allocate memory for SQ */
	log_sqd_bsize = LOG_WQ_DATA_ENTRY_BSIZE + LOG_SQ_RING_DEPTH;
	result = allocate_sq_memory(app_cfg->flexio_process, LOG_SQ_RING_DEPTH, log_sqd_bsize, &ctx->sq_transf);
	if (result < 0)
		return result;

	sq_attr.wq_ring_qmem.daddr = ctx->sq_transf.wq_ring_daddr;

	ret = flexio_sq_create(app_cfg->flexio_process, NULL, cq_num, &sq_attr, &ctx->flexio_sq_ptr);
	if (ret != FLEXIO_STATUS_SUCCESS) {
		printf("Failed to create FlexIO SQ");
		return -1;
	}

	ctx->sq_transf.wq_num = flexio_sq_get_wq_num(ctx->flexio_sq_ptr);
	/* Create SQ TX MKey */
	result = create_dpa_mkey(app_cfg->flexio_process,
					app_cfg->pd,
					ctx->sq_transf.wqd_daddr,
					log_sqd_bsize,
					IBV_ACCESS_LOCAL_WRITE,
					&ctx->sqd_mkey);
	if (result < 0)
		return result;

	ctx->sq_transf.wqd_mkey_id = flexio_mkey_get_id(ctx->sqd_mkey);

	return 0;
}

void rq_destroy(struct app_config *app_cfg, struct dpa_process_context * ctx)
{
	flexio_status ret = FLEXIO_STATUS_SUCCESS;

	ret |= flexio_rq_destroy(ctx->flexio_rq_ptr);
	ret |= flexio_device_mkey_destroy(ctx->rqd_mkey);
	ret |= flexio_buf_dev_free(app_cfg->flexio_process, ctx->rq_transf.wq_dbr_daddr);
	ret |= flexio_buf_dev_free(app_cfg->flexio_process, ctx->rq_transf.wq_ring_daddr);
	ret |= flexio_buf_dev_free(app_cfg->flexio_process, ctx->rq_transf.wqd_daddr);

	if (ret != FLEXIO_STATUS_SUCCESS)
		printf("Failed to destroy RQ\n");
}

void sq_destroy(struct app_config *app_cfg, struct dpa_process_context * ctx)
{
	flexio_status ret = FLEXIO_STATUS_SUCCESS;

	ret |= flexio_sq_destroy(ctx->flexio_sq_ptr);
	ret |= flexio_device_mkey_destroy(ctx->sqd_mkey);
	ret |= flexio_buf_dev_free(app_cfg->flexio_process, ctx->sq_transf.wq_dbr_daddr);
	ret |= flexio_buf_dev_free(app_cfg->flexio_process, ctx->sq_transf.wq_ring_daddr);
	ret |= flexio_buf_dev_free(app_cfg->flexio_process, ctx->sq_transf.wqd_daddr);

	if (ret != FLEXIO_STATUS_SUCCESS)
		printf("Failed to destroy SQ\n");
}

void cq_destroy(struct flexio_process *flexio_process, struct flexio_cq *cq, struct app_transfer_cq cq_transf)
{
	flexio_status ret = FLEXIO_STATUS_SUCCESS;

	ret |= flexio_cq_destroy(cq);
	ret |= flexio_buf_dev_free(flexio_process, cq_transf.cq_ring_daddr);
	ret |= flexio_buf_dev_free(flexio_process, cq_transf.cq_dbr_daddr);

	if (ret != FLEXIO_STATUS_SUCCESS)
		printf("Failed to destroy CQ\n");
}