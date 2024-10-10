#include <infiniband/mlx5_api.h>

#include <assert.h>
#include <bsd/string.h>

#include <doca_argp.h>
#include <doca_log.h>

#include "utils.h"

#include "common.h"
#include "queue.h"

doca_error_t allocate_rq(struct app_config *app_cfg, struct dpa_process_context * ctx)
{
	doca_error_t result;
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
	if (result != DOCA_SUCCESS)
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

doca_error_t allocate_sq(struct app_config *app_cfg, struct dpa_process_context * ctx)
{
	doca_error_t result;
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
	if (result != DOCA_SUCCESS)
		return result;

	sqcq_attr.cq_dbr_daddr = ctx->sq_cq_transf.cq_dbr_daddr;
	sqcq_attr.cq_ring_qmem.daddr = ctx->sq_cq_transf.cq_ring_daddr;

	/* Create SQ's CQ */
	ret = flexio_cq_create(app_cfg->flexio_process, app_cfg->ibv_ctx, &sqcq_attr, &ctx->flexio_sq_cq_ptr);
	if (ret != FLEXIO_STATUS_SUCCESS) {
		DOCA_LOG_ERR("Failed to create FlexIO SQ's CQ");
		return DOCA_ERROR_DRIVER;
	}

	cq_num = flexio_cq_get_cq_num(ctx->flexio_sq_cq_ptr);
	ctx->sq_cq_transf.cq_num = cq_num;
	ctx->sq_cq_transf.log_cq_depth = LOG_CQ_RING_DEPTH;

	/* Allocate memory for SQ */
	log_sqd_bsize = LOG_WQ_DATA_ENTRY_BSIZE + LOG_SQ_RING_DEPTH;
	result = allocate_sq_memory(app_cfg->flexio_process, LOG_SQ_RING_DEPTH, log_sqd_bsize, &ctx->sq_transf);
	if (result != DOCA_SUCCESS)
		return result;

	sq_attr.wq_ring_qmem.daddr = ctx->sq_transf.wq_ring_daddr;

	ret = flexio_sq_create(app_cfg->flexio_process, NULL, cq_num, &sq_attr, &ctx->flexio_sq_ptr);
	if (ret != FLEXIO_STATUS_SUCCESS) {
		DOCA_LOG_ERR("Failed to create FlexIO SQ");
		return DOCA_ERROR_DRIVER;
	}

	ctx->sq_transf.wq_num = flexio_sq_get_wq_num(ctx->flexio_sq_ptr);
	/* Create SQ TX MKey */
	result = create_dpa_mkey(app_cfg->flexio_process,
					app_cfg->pd,
					ctx->sq_transf.wqd_daddr,
					log_sqd_bsize,
					IBV_ACCESS_LOCAL_WRITE,
					&ctx->sqd_mkey);
	if (result != DOCA_SUCCESS)
		return result;

	ctx->sq_transf.wqd_mkey_id = flexio_mkey_get_id(ctx->sqd_mkey);

	return DOCA_SUCCESS;
}