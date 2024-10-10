#ifndef _VXLAN_COMMON_H_
#define _VXLAN_COMMON_H_

#include <libflexio/flexio.h>

#define MAX_NB_THREAD	256

/* Logarithm ring size */
#define LOG_SQ_RING_DEPTH 7 /* 2^7 entries */
#define LOG_RQ_RING_DEPTH 7 /* 2^7 entries */
#define LOG_CQ_RING_DEPTH 7 /* 2^7 entries */

#define LOG_WQ_DATA_ENTRY_BSIZE 11 /* WQ buffer logarithmic size */

/* Queues index mask, represents the index of the last CQE/WQE in the queue */
#define CQ_IDX_MASK ((1 << LOG_CQ_RING_DEPTH) - 1)
#define RQ_IDX_MASK ((1 << LOG_RQ_RING_DEPTH) - 1)
#define SQ_IDX_MASK ((1 << (LOG_SQ_RING_DEPTH + LOG_SQE_NUM_SEGS)) - 1)
#define DATA_IDX_MASK ((1 << (LOG_SQ_RING_DEPTH)) - 1)

struct app_transfer_cq {
	uint32_t cq_num;
	uint32_t log_cq_depth;
	flexio_uintptr_t cq_ring_daddr;
	flexio_uintptr_t cq_dbr_daddr;
} __attribute__((__packed__, aligned(8)));

struct app_transfer_wq {
	uint32_t wq_num;
	uint32_t wqd_mkey_id;
	flexio_uintptr_t wq_ring_daddr;
	flexio_uintptr_t wq_dbr_daddr;
	flexio_uintptr_t wqd_daddr;
} __attribute__((__packed__, aligned(8)));

/* Collateral structure for transfer host data to device */
struct host2dev_processor_data {
	/* RQ's CQ transfer information. */
	struct app_transfer_cq rq_cq_transf;
	/* RQ transfer information. */
	struct app_transfer_wq rq_transf;
	/* SQ's CQ transfer information. */
	struct app_transfer_cq sq_cq_transf;
	/* SQ transfer information. */
	struct app_transfer_wq sq_transf;
    /* Thread index */
	uint32_t thread_index;
} __attribute__((__packed__, aligned(8)));

struct dpa_process_context {
	struct host2dev_processor_data *dev_data;		/* device data */

	struct flexio_event_handler *event_handler; /* Event handler on device */

	struct flexio_mkey *rqd_mkey;
	struct app_transfer_wq rq_transf;

	struct flexio_mkey *sqd_mkey;
	struct app_transfer_wq sq_transf;

	struct flexio_cq *flexio_rq_cq_ptr; /* FlexIO RQ CQ */
	struct flexio_cq *flexio_sq_cq_ptr; /* FlexIO SQ CQ */
	struct flexio_rq *flexio_rq_ptr;    /* FlexIO RQ */
	struct flexio_sq *flexio_sq_ptr;    /* FlexIO SQ */

	/* FlexIO resources */
	flexio_uintptr_t dev_data_daddr;	    /* Data address accessible by the device */
	struct app_transfer_cq rq_cq_transf;
	struct app_transfer_cq sq_cq_transf;

	struct dr_flow_rule *rx_rule;
	struct dr_flow_rule *tx_rule;
	struct dr_flow_rule *tx_root_rule;
};

#endif  /* _VXLAN_COMMON_H_ */