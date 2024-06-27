#ifndef _DOCA_H_
#define _DOCA_H_

#include <stdnoreturn.h>

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_flow.h>
#include <doca_regex.h>
#include <doca_mmap.h>
#include <doca_version.h>
#include <doca_log.h>

#include "config.h"

typedef doca_error_t (*jobs_check)(struct doca_devinfo *);

/* DNS configuration structure */
struct doca_regex_config {
	char pci_address[DOCA_DEVINFO_PCI_ADDR_SIZE];   /* RegEx PCI address to use */
	struct doca_dev * dev;  /* DOCA device */
	struct doca_regex * doca_reg;   /* DOCA RegEx interface */
};

struct doca_regex_ctx {
	struct doca_buf_inventory * buf_inv;    /* DOCA buffer inventory */
	char * data_buffer;		/* Data buffer */
	size_t data_buffer_len;	/* Data buffer length */
	struct doca_buf * buf;		/* DOCA buf */
	struct doca_mmap * mmap;	/* DOCA mmap */
	struct doca_dev * dev;		/* DOCA device */
	struct doca_regex * doca_reg;	/* DOCA RegEx interface */
};

extern struct doca_regex_config doca_regex_cfg;

struct worker_context {
	struct doca_workq * workq;
    struct doca_regex_ctx regex_ctx;
} __attribute__((aligned(64)));

extern struct worker_context worker_ctx[NR_CPUS];
extern __thread struct worker_context * ctx;

struct doca_regex_match_metadata {
	struct doca_buf *job_data;		/* Pointer to the data to be scanned with this job */
	struct doca_regex_search_result result;	/* Storage for results */
};

extern int doca_percore_init(void);
extern int doca_worker_init(struct worker_context * ctx);
extern int doca_init(void);

#endif  /* _DOCA_H_ */