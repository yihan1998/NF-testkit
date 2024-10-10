#include <libflexio-dev/flexio_dev_err.h>
#include <libflexio-dev/flexio_dev_queue_access.h>
#include <libflexio-libc/string.h>
#include <stddef.h>
#include <dpaintrin.h>

#include "../common.h"

/* Mask for CQ index */
#define CQ_IDX_MASK ((1 << LOG_CQ_DEPTH) - 1)
/* Mask for RQ index */
#define RQ_IDX_MASK ((1 << LOG_RQ_DEPTH) - 1)
/* Mask for SQ index */
#define SQ_IDX_MASK ((1 << (LOG_SQ_DEPTH + LOG_SQE_NUM_SEGS)) - 1)
/* Mask for data index */
#define DATA_IDX_MASK ((1 << (LOG_SQ_DEPTH)) - 1)

/* The structure of the sample DPA application contains global data that the application uses */
static struct device_context {
	/* Packet count - used for debug message */
	uint64_t packets_count;

	uint32_t lkey;				/* Local memory key */
	uint32_t is_initalized;		/* Initialization flag */
	cq_ctx_t rq_cq_ctx;     /* RQ CQ */
	rq_ctx_t rq_ctx;        /* RQ */
	sq_ctx_t sq_ctx;        /* SQ */
	cq_ctx_t sq_cq_ctx;     /* SQ CQ */
	dt_ctx_t dt_ctx;        /* SQ Data ring */
} __attribute__((__aligned__(64))) dev_ctxs[MAX_THREADS];

// /* Initialize the app_ctx structure from the host data.
//  *  data_from_host - pointer host2dev_packet_processor_data from host.
//  */
// static void app_ctx_init(struct host2dev_packet_processor_data *data_from_host)
// {
// 	app_ctx.packets_count = 0;
// 	app_ctx.lkey = data_from_host->sq_transf.wqd_mkey_id;

// 	/* Set context for RQ's CQ */
// 	com_cq_ctx_init(&app_ctx.rq_cq_ctx,
// 			data_from_host->rq_cq_transf.cq_num,
// 			data_from_host->rq_cq_transf.log_cq_depth,
// 			data_from_host->rq_cq_transf.cq_ring_daddr,
// 			data_from_host->rq_cq_transf.cq_dbr_daddr);

// 	/* Set context for RQ */
// 	com_rq_ctx_init(&app_ctx.rq_ctx,
// 			data_from_host->rq_transf.wq_num,
// 			data_from_host->rq_transf.wq_ring_daddr,
// 			data_from_host->rq_transf.wq_dbr_daddr);

// 	/* Set context for SQ */
// 	com_sq_ctx_init(&app_ctx.sq_ctx,
// 			data_from_host->sq_transf.wq_num,
// 			data_from_host->sq_transf.wq_ring_daddr);

// 	/* Set context for SQ's CQ */
// 	com_cq_ctx_init(&app_ctx.sq_cq_ctx,
// 			data_from_host->sq_cq_transf.cq_num,
// 			data_from_host->sq_cq_transf.log_cq_depth,
// 			data_from_host->sq_cq_transf.cq_ring_daddr,
// 			data_from_host->sq_cq_transf.cq_dbr_daddr);

// 	/* Set context for data */
// 	com_dt_ctx_init(&app_ctx.dt_ctx, data_from_host->sq_transf.wqd_daddr);
// }

__dpa_rpc__ uint64_t device_context_init(struct host2dev_processor_data* data)
{
	struct device_context *dev_ctx = &dev_ctxs[data->thread_index];
	dev_ctx->lkey = data->sq_data.wqd_mkey_id;
	init_cq(data->rq_cq_data, &dev_ctx->rqcq_ctx);
	init_rq(data->rq_data, &dev_ctx->rq_ctx);
	init_cq(data->sq_cq_data, &dev_ctx->sqcq_ctx);
	init_sq(data->sq_data, &dev_ctx->sq_ctx);

	dev_ctx->dt_ctx.sq_tx_buff = (void *)shared_data->sq_data.wqd_daddr;
	dev_ctx->dt_ctx.tx_buff_idx = 0;

	dev_ctx->is_initalized = 1;
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

uint8_t h_source[6] = {0xde,0xed,0xbe,0xef,0xab,0xcd};
uint8_t h_dest[6] = {0x10,0x70,0xfd,0xc8,0x94,0x75};
        
/* Return size of packet */
uint32_t vxlan_encap(char *out_data, char *in_data, uint32_t in_data_size) {
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
    new_eth_hdr->h_proto=htons(0x0800);
    SET_MAC_ADDR(new_eth_hdr->h_source,h_source[0],h_source[1],h_source[2],h_source[3],h_source[4],h_source[5]);
    SET_MAC_ADDR(new_eth_hdr->h_dest,h_dest[0],h_dest[1],h_dest[2],h_dest[3],h_dest[4],h_dest[5]);
    new_ip_hdr->version=4;
    new_ip_hdr->ihl=5;
    new_ip_hdr->tot_len=htons(pkt_size+sizeof(struct iphdr)+sizeof(struct udphdr)+sizeof(struct vxlanhdr));
    new_ip_hdr->protocol=IPPROTO_UDP;
    new_ip_hdr->saddr=orig_ip_hdr->saddr;
    new_ip_hdr->daddr=orig_ip_hdr->daddr;
    new_udp_hdr->source=orig_udp_hdr->source;
    new_udp_hdr->dest=htons(4789);
    new_udp_hdr->len=htons(pkt_size+sizeof(struct udphdr)+sizeof(struct vxlanhdr));
    vxlan_hdr->flags=htonl(0x08000000);
    vxlan_hdr->vni_reserved2=htonl(0x123456);
}

/* process packet - read it, swap MAC addresses, modify it, create a send WQE and send it back
 *  dtctx - pointer to context of the thread.
 */
static void process_packet(struct flexio_dev_thread_ctx *dtctx)
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
	rq_wqe_idx = flexio_dev_cqe_get_wqe_counter(app_ctx.rq_cq_ctx.cqe);
	data_sz = flexio_dev_cqe_get_byte_cnt(app_ctx.rq_cq_ctx.cqe);

	/* Get the RQ WQE pointed to by the CQE */
	rwqe = &app_ctx.rq_ctx.rq_ring[rq_wqe_idx & RQ_IDX_MASK];

	/* Extract data (whole packet) pointed to by the RQ WQE */
	rq_data = flexio_dev_rwqe_get_addr(rwqe);

	/* Take the next entry from the data ring */
	sq_data = get_next_dte(&app_ctx.dt_ctx, DATA_IDX_MASK, LOG_WQD_CHUNK_BSIZE);

    uint32_t sq_data_size = vxlan_encap(sq_data, rq_data, data_sz);
#if 0
	/* Copy received packet to sq_data as is */
	memcpy(sq_data, rq_data, data_sz);

	/* swap mac address */
	swap_macs(sq_data);

	/* Primitive validation, that packet is our hardcoded */
	if (data_sz == 65) {
		/* modify UDP payload */
		memcpy(sq_data + 0x2a, "  Event demo***************", 65 - 0x2a);

		/* Set hexadecimal value by the index */
		sq_data[0x2a] = "0123456789abcdef"[app_ctx.dt_ctx.tx_buff_idx & 0xf];
	}
#endif
	/* Take first segment for SQ WQE (3 segments will be used) */
	swqe = get_next_sqe(&app_ctx.sq_ctx, SQ_IDX_MASK);

	/* Fill out 1-st segment (Control) */
	flexio_dev_swqe_seg_ctrl_set(swqe, app_ctx.sq_ctx.sq_pi, app_ctx.sq_ctx.sq_number,
				     MLX5_CTRL_SEG_CE_CQE_ON_CQE_ERROR, FLEXIO_CTRL_SEG_SEND_EN);

	/* Fill out 2-nd segment (Ethernet) */
	swqe = get_next_sqe(&app_ctx.sq_ctx, SQ_IDX_MASK);
	flexio_dev_swqe_seg_eth_set(swqe, 0, 0, 0, NULL);

	/* Fill out 3-rd segment (Data) */
	swqe = get_next_sqe(&app_ctx.sq_ctx, SQ_IDX_MASK);
	flexio_dev_swqe_seg_mem_ptr_data_set(swqe, sq_data_size, app_ctx.lkey, (uint64_t)sq_data);

	/* Send WQE is 4 WQEBBs need to skip the 4-th segment */
	swqe = get_next_sqe(&app_ctx.sq_ctx, SQ_IDX_MASK);

	/* Ring DB */
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	flexio_dev_qp_sq_ring_db(dtctx, ++app_ctx.sq_ctx.sq_pi, app_ctx.sq_ctx.sq_number);
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	flexio_dev_dbr_rq_inc_pi(app_ctx.rq_ctx.rq_dbr);
}

void __dpa_global__ vxlan_device_event_handler(uint64_t index)
{
	struct flexio_dev_thread_ctx *dtctx;
    struct device_context *dev_ctx = &dev_ctxs[index];

	/* Read the current thread context */
	flexio_dev_get_thread_ctx(&dtctx);

	/* Poll CQ until the package is received.
	 */
	while (flexio_dev_cqe_get_owner(app_ctx.rq_cq_ctx.cqe) !=
	       app_ctx.rq_cq_ctx.cq_hw_owner_bit) {
		/* Print the message */
		flexio_dev_print("Process packet: %ld\n", app_ctx.packets_count++);
		/* Update memory to DPA */
		__dpa_thread_fence(__DPA_MEMORY, __DPA_R, __DPA_R);
		/* Process the packet */
		process_packet(dtctx);
		/* Update RQ CQ */
		com_step_cq(&app_ctx.rq_cq_ctx);
	}
	/* Update the memory to the chip */
	__dpa_thread_fence(__DPA_MEMORY, __DPA_W, __DPA_W);
	/* Arming cq for next packet */
	flexio_dev_cq_arm(dtctx, app_ctx.rq_cq_ctx.cq_idx, app_ctx.rq_cq_ctx.cq_number);

	/* Reschedule the thread */
	flexio_dev_thread_reschedule();
}
