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

uint16_t htons(uint16_t hostshort);
uint32_t htonl(uint32_t hostlong);
uint32_t vxlan_encap(char *out_data, char *in_data, uint32_t in_data_size);

__dpa_rpc__ uint64_t vxlan_device_init(uint64_t data)
{
	struct host2dev_processor_data *shared_data = (struct host2dev_processor_data *)data;
	struct device_context *dev_ctx = &dev_ctxs[shared_data->thread_index];
	dev_ctx->lkey = shared_data->sq_data.wqd_mkey_id;
	init_cq(shared_data->rq_cq_data, &dev_ctx->rq_cq_ctx);
	init_rq(shared_data->rq_data, &dev_ctx->rq_ctx);
	init_cq(shared_data->sq_cq_data, &dev_ctx->sq_cq_ctx);
	init_sq(shared_data->sq_data, &dev_ctx->sq_ctx);
	dev_ctx->dt_ctx.sq_tx_buff = (void *)shared_data->sq_data.wqd_daddr;
	dev_ctx->dt_ctx.tx_buff_idx = 0;
	dev_ctx->is_initalized = 1;
	return 0;
}

struct ethhdr {
    uint8_t  h_dest[6];
    uint8_t  h_source[6];
    uint16_t h_proto;
};

struct iphdr {
    uint8_t  ihl:4,
             version:4;
    uint8_t  tos;
    uint16_t tot_len;
    uint16_t id;
    uint16_t frag_off;
    uint8_t  ttl;
    uint8_t  protocol;
    uint16_t check;
    uint32_t saddr;
    uint32_t daddr;
};

struct udphdr {
    uint16_t source;
    uint16_t dest;
    uint16_t len;
    uint16_t check;
};

struct vxlanhdr {
    uint8_t  flags;
    uint8_t  reserved1[3];
    uint32_t vni_reserved2;
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

uint16_t htons(uint16_t hostshort)
{
    return (hostshort << 8) | (hostshort >> 8);
}

uint32_t htonl(uint32_t hostlong)
{
    return ((hostlong << 24) & 0xFF000000) |
           ((hostlong << 8) & 0x00FF0000) |
           ((hostlong >> 8) & 0x0000FF00) |
           ((hostlong >> 24) & 0x000000FF);
}

/* Return size of packet */
uint32_t vxlan_encap(char *out_data, char *in_data, uint32_t in_data_size)
{
    uint32_t pkt_size=in_data_size;
    uint32_t new_hdr_size=sizeof(struct ethhdr)+sizeof(struct iphdr)+sizeof(struct udphdr)+sizeof(struct vxlanhdr);
    memcpy(out_data+new_hdr_size,in_data,in_data_size);
    struct ethhdr *new_eth_hdr = (struct ethhdr *)out_data;
    struct iphdr *new_ip_hdr = (struct iphdr *)&new_eth_hdr[1];
    struct udphdr *new_udp_hdr = (struct udphdr *)&new_ip_hdr[1];
    struct vxlanhdr *vxlan_hdr = (struct vxlanhdr*)&new_udp_hdr[1];
    struct ethhdr *orig_eth_hdr = (struct ethhdr *)in_data;
    struct iphdr *orig_ip_hdr = (struct iphdr *)&orig_eth_hdr[1];
    struct udphdr *orig_udp_hdr = (struct udphdr *)&orig_ip_hdr[1];
	uint8_t *src_mac = orig_eth_hdr->h_source;
	uint8_t *dst_mac = orig_eth_hdr->h_dest;
    new_eth_hdr->h_proto=htons(0x0800);
    SET_MAC_ADDR(new_eth_hdr->h_source,dst_mac[0],dst_mac[1],dst_mac[2],dst_mac[3],dst_mac[4],dst_mac[5]);
    SET_MAC_ADDR(new_eth_hdr->h_dest,src_mac[0],src_mac[1],src_mac[2],src_mac[3],src_mac[4],src_mac[5]);
    new_ip_hdr->version=4;
    new_ip_hdr->ihl=5;
    new_ip_hdr->tot_len=htons(pkt_size+sizeof(struct iphdr)+sizeof(struct udphdr)+sizeof(struct vxlanhdr));
    new_ip_hdr->protocol=0x11;
    new_ip_hdr->saddr=orig_ip_hdr->saddr;
    new_ip_hdr->daddr=orig_ip_hdr->daddr;
    new_udp_hdr->source=orig_udp_hdr->source;
    new_udp_hdr->dest=htons(4789);
    new_udp_hdr->len=htons(pkt_size+sizeof(struct udphdr)+sizeof(struct vxlanhdr));
    vxlan_hdr->flags=htonl(0x08000000);
    vxlan_hdr->vni_reserved2=htonl(0x123456);
	return new_hdr_size+in_data_size;
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

    uint32_t sq_data_size = vxlan_encap(sq_data, rq_data, data_sz);

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
	flexio_dev_swqe_seg_mem_ptr_data_set(swqe, sq_data_size, dev_ctx->lkey, (uint64_t)sq_data);

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

	/* Read the current thread context */
	flexio_dev_get_thread_ctx(&dtctx);

	/* Poll CQ until the package is received.
	 */
	while (flexio_dev_cqe_get_owner(dev_ctx->rq_cq_ctx.cqe) != dev_ctx->rq_cq_ctx.cq_hw_owner_bit) {
		/* Print the message */
		flexio_dev_print("Process packet: %ld\n", dev_ctx->packets_count++);
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
