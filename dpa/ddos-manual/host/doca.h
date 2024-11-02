#ifndef _DOCA_H_
#define _DOCA_H_

#include <doca_flow.h>
#include <doca_dev.h>
#include <doca_mmap.h>
#include <doca_buf_inventory.h>

#include "dpdk.h"

typedef doca_error_t (*tasks_check)(struct doca_devinfo *);

struct doca_sha_config {
	char pci_address[DOCA_DEVINFO_PCI_ADDR_SIZE];   /* RegEx PCI address to use */
	struct doca_dev * dev;  /* DOCA device */
	struct doca_sha * doca_sha;   /* DOCA SHA interface */
};

struct doca_sha_ctx {
	struct doca_buf_inventory * buf_inv;    /* DOCA buffer inventory */
	char * src_data_buffer;		/* Data buffer */
	size_t src_data_buffer_len;	/* Data buffer length */
	struct doca_buf * src_buf;		/* DOCA buf */
	struct doca_mmap * src_mmap;	/* DOCA mmap */

	char * dst_data_buffer;		/* Data buffer */
	size_t dst_data_buffer_len;	/* Data buffer length */
	struct doca_buf * dst_buf;		/* DOCA buf */
	struct doca_mmap * dst_mmap;	/* DOCA mmap */

	struct doca_dev * dev;		/* DOCA device */
	struct doca_sha * doca_sha;   /* DOCA SHA interface */
};

extern struct doca_sha_config doca_sha_cfg;

struct worker_context {
	struct doca_pe * pe;
    struct doca_sha_ctx sha_ctx;
} __attribute__((aligned(64)));

#define NR_CPUS 16

extern struct worker_context worker_ctx[NR_CPUS];
extern __thread struct worker_context * ctx;

extern int doca_percore_init(void);
extern int doca_worker_init(struct worker_context * ctx);

doca_error_t doca_init(struct application_dpdk_config *app_dpdk_config);

#endif  /* _DOCA_H_ */