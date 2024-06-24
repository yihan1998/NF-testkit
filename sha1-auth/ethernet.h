#ifndef _ETHERNET_H_
#define _ETHERNET_H_

#include <stdint.h>
#include <linux/if_ether.h>

#include "skbuff.h"

#define NET_RX_SUCCESS	0	/* keep 'em coming, baby */
#define NET_RX_DROP		1	/* packet dropped */

#define ETHADDR_COPY(dst, src)  memcpy(dst, src, ETH_ALEN)

extern int ethernet_input(struct rte_mbuf * m, uint8_t * pkt, int pkt_size);

#endif  /* _ETHERNET_H_ */