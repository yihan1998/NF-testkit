#include "ethernet.h"
#include "ip.h"

int ethernet_input(struct rte_mbuf * m, uint8_t * pkt, int pkt_size) {
    struct ethhdr * ethhdr;
    uint16_t proto;
    struct iphdr * iphdr;
    struct sk_buff * skb;
    int ret = NET_RX_SUCCESS;

    skb = alloc_skb(pkt, pkt_size);
    if (!skb) {
		// pr_warn("Failed to allocate new skbuff!\n");
		return NET_RX_DROP;
	}

    skb->m = m;
    skb->ptr = pkt;
    ethhdr = (struct ethhdr *)pkt;
    proto = ntohs(ethhdr->h_proto);

    switch (proto) {
        case ETH_P_IP:
            /* pass to IP layer */
            iphdr = (struct iphdr *)&ethhdr[1];
            ret = ip4_input(skb, iphdr);
            break;
        default:
            break;
    }

    // rte_pktmbuf_free(skb->m);
    free_skb(skb);
    return ret;
}
