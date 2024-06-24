
#include <linux/tcp.h>
#include <linux/udp.h>

#include "ethernet.h"
#include "ip.h"
#include "config.h"
#include "compression.h"

static unsigned int compression_main(struct sk_buff *skb) {
    struct ethhdr * ethhdr;
    struct iphdr * iphdr;
    uint16_t iphdr_hlen;
    struct udphdr * udphdr;
	uint16_t ulen, len;
	uint8_t * data;

    ethhdr = (struct ethhdr *)skb->ptr;

    iphdr = (struct iphdr *)&ethhdr[1];
    iphdr_hlen = iphdr->ihl;
    iphdr_hlen <<= 2;

    udphdr = (struct udphdr *)((uint8_t *)iphdr + iphdr_hlen);
    ulen = ntohs(udphdr->len);
	len = ulen - sizeof(struct udphdr);
	data = (uint8_t *)udphdr + sizeof(struct udphdr);

    return compress_pkt(data, len);
}

int ip4_input(struct sk_buff * skb, struct iphdr * iphdr) {
    uint16_t iphdr_hlen;
    // uint16_t iphdr_len;
    uint8_t iphdr_protocol;
    struct udphdr * udphdr;

    /* obtain IP header length in number of 32-bit words */
    iphdr_hlen = iphdr->ihl;
    /* calculate IP header length in bytes */
    iphdr_hlen <<= 2;
    /* obtain ip length in bytes */
    // iphdr_len = ntohs(iphdr->tot_len);

    iphdr_protocol = iphdr->protocol;

    IPCB(skb)->saddr = iphdr->saddr;
    IPCB(skb)->daddr = iphdr->daddr;

    if (iphdr_protocol == IPPROTO_UDP) {
        udphdr = (struct udphdr *)((uint8_t *)iphdr + iphdr_hlen);
        if (ntohs(udphdr->dest) == 1234) {
    		compression_main(skb);
        }
    }

    return NET_RX_SUCCESS;
}
