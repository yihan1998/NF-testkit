#include <stddef.h>
#include <stdint.h>
#include <libflexio-libc/stdio.h>
#include <libflexio-libc/string.h>
#include <libflexio-dev/flexio_dev_err.h>
#include <libflexio-dev/flexio_dev_queue_access.h>
#include <libflexio-libc/string.h>
#include <dpaintrin.h>

#include "../common.h"

/* CQ Context */
struct cq_ctx_t {
	uint32_t cq_number;		  /* CQ number */
	struct flexio_dev_cqe64 *cq_ring; /* CQEs buffer */
	struct flexio_dev_cqe64 *cqe;	  /* Current CQE */
	uint32_t cq_idx;		  /* Current CQE IDX */
	uint8_t cq_hw_owner_bit;	  /* HW/SW ownership */
	uint32_t *cq_dbr;		  /* CQ doorbell record */
};

/* RQ Context */
struct rq_ctx_t {
	uint32_t rq_number;			     /* RQ number */
	struct flexio_dev_wqe_rcv_data_seg *rq_ring; /* WQEs buffer */
	uint32_t *rq_dbr;			     /* RQ doorbell record */
};

/* SQ Context */
struct sq_ctx_t {
	uint32_t sq_number;		   /* SQ number */
	uint32_t sq_wqe_seg_idx;	   /* WQE segment index */
	union flexio_dev_sqe_seg *sq_ring; /* SQEs buffer */
	uint32_t *sq_dbr;		   /* SQ doorbell record */
	uint32_t sq_pi;			   /* SQ producer index */
};

/* SQ data buffer */
struct dt_ctx_t {
	void *sq_tx_buff;     /* SQ TX buffer */
	uint32_t tx_buff_idx; /* TX buffer index */
};

/* The structure of the sample DPA application contains global data that the application uses */
static struct device_context {
	/* Packet count - used for debug message */
	uint64_t packets_count;
	int32_t *host_buffer;

	uint32_t window_id;		/* FlexIO Window ID */
	uint32_t mkey;			/* Memory key for the result matrix */
	uint64_t haddr;			/* Host address for the result matrix */

	uint32_t lkey;				/* Local memory key */
	uint32_t is_initalized;		/* Initialization flag */
	struct cq_ctx_t rq_cq_ctx;	/* RQ CQ */
	struct rq_ctx_t rq_ctx;		/* RQ */
	struct sq_ctx_t sq_ctx;		/* SQ */
	struct cq_ctx_t sq_cq_ctx;	/* SQ CQ */
	struct dt_ctx_t dt_ctx;		/* SQ Data ring */
} __attribute__((__aligned__(64))) dev_ctxs[MAX_NB_THREAD];

static void init_cq(const struct app_transfer_cq app_cq, struct cq_ctx_t *ctx)
{
	ctx->cq_number = app_cq.cq_num;
	ctx->cq_ring = (struct flexio_dev_cqe64 *)app_cq.cq_ring_daddr;
	ctx->cq_dbr = (uint32_t *)app_cq.cq_dbr_daddr;

	ctx->cqe = ctx->cq_ring; /* Points to the first CQE */
	ctx->cq_idx = 0;
	ctx->cq_hw_owner_bit = 0x1;
}

static void init_rq(const struct app_transfer_wq app_rq, struct rq_ctx_t *ctx)
{
	ctx->rq_number = app_rq.wq_num;
	ctx->rq_ring = (struct flexio_dev_wqe_rcv_data_seg *)app_rq.wq_ring_daddr;
	ctx->rq_dbr = (uint32_t *)app_rq.wq_dbr_daddr;
}

static void init_sq(const struct app_transfer_wq app_sq, struct sq_ctx_t *ctx)
{
	ctx->sq_number = app_sq.wq_num;
	ctx->sq_ring = (union flexio_dev_sqe_seg *)app_sq.wq_ring_daddr;
	ctx->sq_dbr = (uint32_t *)app_sq.wq_dbr_daddr;

	ctx->sq_wqe_seg_idx = 0;
	ctx->sq_dbr++;
}

static void *get_next_dte(struct dt_ctx_t *dt_ctx, uint32_t dt_idx_mask, uint32_t log_dt_entry_sz)
{
	uint32_t mask = ((dt_ctx->tx_buff_idx++ & dt_idx_mask) << log_dt_entry_sz);
	char *buff_p = (char *)dt_ctx->sq_tx_buff;

	return buff_p + mask;
}

static void *get_next_sqe(struct sq_ctx_t *sq_ctx, uint32_t sq_idx_mask)
{
	return &sq_ctx->sq_ring[sq_ctx->sq_wqe_seg_idx++ & sq_idx_mask];
}

static void step_cq(struct cq_ctx_t *cq_ctx, uint32_t cq_idx_mask)
{
	cq_ctx->cq_idx++;
	cq_ctx->cqe = &cq_ctx->cq_ring[cq_ctx->cq_idx & cq_idx_mask];
	/* check for wrap around */
	if (!(cq_ctx->cq_idx & cq_idx_mask))
		cq_ctx->cq_hw_owner_bit = !cq_ctx->cq_hw_owner_bit;

	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	flexio_dev_dbr_cq_set_ci(cq_ctx->cq_dbr, cq_ctx->cq_idx);
}

uint64_t vxlan_device_init(uint64_t data);
flexio_dev_event_handler_t vxlan_device_event_handler; /* Event handler function */

uint32_t htonl(uint32_t hostlong);
void swap_macs(char *packet);

__dpa_rpc__ uint64_t vxlan_device_init(uint64_t data)
{
	flexio_dev_print("Initializing device...\n");

	struct host2dev_processor_data *shared_data = (struct host2dev_processor_data *)data;
	struct device_context *dev_ctx = &dev_ctxs[shared_data->thread_index];
	dev_ctx->lkey = shared_data->sq_data.wqd_mkey_id;
	init_cq(shared_data->rq_cq_data, &dev_ctx->rq_cq_ctx);
	init_rq(shared_data->rq_data, &dev_ctx->rq_ctx);
	init_cq(shared_data->sq_cq_data, &dev_ctx->sq_cq_ctx);
	init_sq(shared_data->sq_data, &dev_ctx->sq_ctx);
	dev_ctx->dt_ctx.sq_tx_buff = (void *)shared_data->sq_data.wqd_daddr;
	dev_ctx->dt_ctx.tx_buff_idx = 0;
	dev_ctx->window_id = shared_data->window_id;
	dev_ctx->mkey = shared_data->mkey;
	dev_ctx->haddr = shared_data->haddr;
	dev_ctx->is_initalized = 1;
	return 0;
}

struct ethhdr {
    uint8_t  h_dest[6];
    uint8_t  h_source[6];
    uint16_t h_proto;
};

#define SET_MAC_ADDR(addr, a, b, c, d, e, f)\
do {\
	addr[0] = a & 0xff;\
	addr[1] = b & 0xff;\
	addr[2] = c & 0xff;\
	addr[3] = d & 0xff;\
	addr[4] = e & 0xff;\
	addr[5] = f & 0xff;\
} while (0)

uint32_t htonl(uint32_t hostlong)
{
    return ((hostlong << 24) & 0xFF000000) |
           ((hostlong << 8) & 0x00FF0000) |
           ((hostlong >> 8) & 0x0000FF00) |
           ((hostlong >> 24) & 0x000000FF);
}

uint32_t flow_monitor(struct device_context *dev_ctx, char *out_data, char *in_data, uint32_t in_data_size) {
    uint32_t pkt_size=in_data_size;
	memcpy(out_data, in_data, in_data_size);
    struct ethhdr *out_eth_hdr = (struct ethhdr *)out_data;
	uint8_t temp[6];
    SET_MAC_ADDR(temp,out_eth_hdr->h_source[0],out_eth_hdr->h_source[1],out_eth_hdr->h_source[2],out_eth_hdr->h_source[3],out_eth_hdr->h_source[4],out_eth_hdr->h_source[5]);
    SET_MAC_ADDR(out_eth_hdr->h_source,out_eth_hdr->h_dest[0],out_eth_hdr->h_dest[1],out_eth_hdr->h_dest[2],out_eth_hdr->h_dest[3],out_eth_hdr->h_dest[4],out_eth_hdr->h_dest[5]);
    SET_MAC_ADDR(out_eth_hdr->h_dest,temp[0],temp[1],temp[2],temp[3],temp[4],temp[5]);
	*dev_ctx->host_buffer = *dev_ctx->host_buffer + 1;
	return pkt_size;
}

/* process packet - read it, swap MAC addresses, modify it, create a send WQE and send it back
 *  dtctx - pointer to context of the thread.
 */
static void process_packet(struct flexio_dev_thread_ctx *dtctx, struct device_context *dev_ctx)
{
	/* RX packet handling variables */
	struct flexio_dev_wqe_rcv_data_seg *rwqe;
	/* RQ WQE index */
	uint32_t rq_wqe_idx;
	/* Pointer to RQ data */
	char *rq_data;

	/* TX packet handling variables */
	union flexio_dev_sqe_seg *swqe;
	/* Pointer to SQ data */
	char *sq_data;

	/* Size of the data */
	uint32_t data_sz;

	/* Extract relevant data from the CQE */
	rq_wqe_idx = flexio_dev_cqe_get_wqe_counter(dev_ctx->rq_cq_ctx.cqe);
	data_sz = flexio_dev_cqe_get_byte_cnt(dev_ctx->rq_cq_ctx.cqe);

	/* Get the RQ WQE pointed to by the CQE */
	rwqe = &dev_ctx->rq_ctx.rq_ring[rq_wqe_idx & RQ_IDX_MASK];

	/* Extract data (whole packet) pointed to by the RQ WQE */
	rq_data = flexio_dev_rwqe_get_addr(rwqe);

	/* Take the next entry from the data ring */
	sq_data = get_next_dte(&dev_ctx->dt_ctx, DATA_IDX_MASK, LOG_WQ_DATA_ENTRY_BSIZE);

	// memcpy(sq_data, rq_data, data_sz);
	// swap_macs(sq_data);
	// *dev_ctx->host_buffer = *dev_ctx->host_buffer + 1;
    uint32_t sq_data_size = flow_monitor(dev_ctx,sq_data,rq_data,data_sz);

	/* Take first segment for SQ WQE (3 segments will be used) */
	swqe = get_next_sqe(&dev_ctx->sq_ctx, SQ_IDX_MASK);

	/* Fill out 1-st segment (Control) */
	flexio_dev_swqe_seg_ctrl_set(swqe, dev_ctx->sq_ctx.sq_pi, dev_ctx->sq_ctx.sq_number,
				     MLX5_CTRL_SEG_CE_CQE_ON_CQE_ERROR, FLEXIO_CTRL_SEG_SEND_EN);

	/* Fill out 2-nd segment (Ethernet) */
	swqe = get_next_sqe(&dev_ctx->sq_ctx, SQ_IDX_MASK);
	flexio_dev_swqe_seg_eth_set(swqe, 0, 0, 0, NULL);

	/* Fill out 3-rd segment (Data) */
	swqe = get_next_sqe(&dev_ctx->sq_ctx, SQ_IDX_MASK);
	flexio_dev_swqe_seg_mem_ptr_data_set(swqe, data_sz, dev_ctx->lkey, (uint64_t)sq_data);

	/* Send WQE is 4 WQEBBs need to skip the 4-th segment */
	swqe = get_next_sqe(&dev_ctx->sq_ctx, SQ_IDX_MASK);

	/* Ring DB */
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	flexio_dev_qp_sq_ring_db(dtctx, ++dev_ctx->sq_ctx.sq_pi, dev_ctx->sq_ctx.sq_number);
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	flexio_dev_dbr_rq_inc_pi(dev_ctx->rq_ctx.rq_dbr);
}

void __dpa_global__ vxlan_device_event_handler(uint64_t index)
{
	struct flexio_dev_thread_ctx *dtctx;
    struct device_context *dev_ctx = &dev_ctxs[index];
	flexio_dev_status_t result;

	/* Read the current thread context */
	flexio_dev_get_thread_ctx(&dtctx);

    /* Configure FlexIO Window */
	result = flexio_dev_window_config(dtctx, dev_ctx->window_id, dev_ctx->mkey);
	if (result != FLEXIO_DEV_STATUS_SUCCESS) {
		flexio_dev_print("Failed to configure FlexIO window\n");
    }

    /* Acquire device pointer to host memory */
	result = flexio_dev_window_ptr_acquire(dtctx, dev_ctx->haddr, (flexio_uintptr_t *)&dev_ctx->host_buffer);
	if (result != FLEXIO_DEV_STATUS_SUCCESS) {
		flexio_dev_print("Failed to acquire FlexIO window ptr\n");
    }

	flexio_dev_print("Init packet count %d(%p)\n", *dev_ctx->host_buffer, (void *)dev_ctx->host_buffer);

	// struct flexio_dev_thread_ctx *dtctx;
    // struct device_context *dev_ctx = &dev_ctxs[index];

	/* Read the current thread context */
	flexio_dev_get_thread_ctx(&dtctx);

	/* Poll CQ until the package is received.
	 */
	while (flexio_dev_cqe_get_owner(dev_ctx->rq_cq_ctx.cqe) != dev_ctx->rq_cq_ctx.cq_hw_owner_bit) {
		/* Print the message */
		// flexio_dev_print("Process packet: %ld, seen packet: %d(%p)\n", dev_ctx->packets_count++, *dev_ctx->host_buffer, (void *)dev_ctx->host_buffer);
		/* Update memory to DPA */
		__dpa_thread_fence(__DPA_MEMORY, __DPA_R, __DPA_R);
		/* Process the packet */
		process_packet(dtctx, dev_ctx);
		/* Update RQ CQ */
		step_cq(&dev_ctx->rq_cq_ctx, CQ_IDX_MASK);
	}
	/* Update the memory to the chip */
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	/* Arming cq for next packet */
	flexio_dev_cq_arm(dtctx, dev_ctx->rq_cq_ctx.cq_idx, dev_ctx->rq_cq_ctx.cq_number);

	/* Reschedule the thread */
	flexio_dev_thread_reschedule();
}