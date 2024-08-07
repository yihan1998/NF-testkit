/*
 * Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#include <rte_ether.h>
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include <rte_udp.h>
#include <rte_vxlan.h>

#include <doca_log.h>

#include "vxlan_fwd_pkt.h"


DOCA_LOG_REGISTER(VXLAN_FWD_PKT);

#define GTP_ESPN_FLAGS_ON(p) (p & 0x7)	/* A macro for setting GTP ESPN flags on */
#define GTP_EXT_FLAGS_ON(p) (p & 0x4)	/* A macro for setting GTP EXT flags on */

uint8_t*
vxlan_fwd_pinfo_outer_mac_dst(struct vxlan_fwd_pkt_info *pinfo)
{
	return ((struct rte_ether_hdr *)pinfo->outer.l2)->dst_addr.addr_bytes;
}

uint8_t*
vxlan_fwd_pinfo_outer_mac_src(struct vxlan_fwd_pkt_info *pinfo)
{
	return ((struct rte_ether_hdr *)pinfo->outer.l2)->src_addr.addr_bytes;
}

doca_be32_t
vxlan_fwd_pinfo_outer_ipv4_dst(struct vxlan_fwd_pkt_info *pinfo)
{
	return ((struct rte_ipv4_hdr *)pinfo->outer.l3)->dst_addr;
}

doca_be32_t
vxlan_fwd_pinfo_outer_ipv4_src(struct vxlan_fwd_pkt_info *pinfo)
{
	return ((struct rte_ipv4_hdr *)pinfo->outer.l3)->src_addr;
}

doca_be32_t
vxlan_fwd_pinfo_inner_ipv4_dst(struct vxlan_fwd_pkt_info *pinfo)
{
	return ((struct rte_ipv4_hdr *)pinfo->inner.l3)->dst_addr;
}

doca_be32_t
vxlan_fwd_pinfo_inner_ipv4_src(struct vxlan_fwd_pkt_info *pinfo)
{
	return ((struct rte_ipv4_hdr *)pinfo->inner.l3)->src_addr;
}

/*
 * Extracts the source port address from the packet's info based on layer 4 type
 *
 * @fmt [in]: the packet format as represented in the application
 * @return: source port
 *
 * @NOTE: the returned value is converted to big endian
 */
static doca_be16_t
vxlan_fwd_pinfo_src_port(struct vxlan_fwd_pkt_format *fmt)
{
	switch (fmt->l4_type) {
	case DOCA_PROTO_TCP:
		return ((struct rte_tcp_hdr *)fmt->l4)->src_port;
	case DOCA_PROTO_UDP:
		return ((struct rte_udp_hdr *)fmt->l4)->src_port;
	default:
		return 0;
	}
}

/*
 * Extracts the destination port address from the packet's info based on layer 4 type
 *
 * @fmt [in]: the packet format as represented in the application
 * @return: destination port
 *
 * @NOTE: the returned value is converted to big endian
 */
static doca_be16_t
vxlan_fwd_pinfo_dst_port(struct vxlan_fwd_pkt_format *fmt)
{
	switch (fmt->l4_type) {
	case DOCA_PROTO_TCP:
		return ((struct rte_tcp_hdr *)fmt->l4)->dst_port;
	case DOCA_PROTO_UDP:
		return ((struct rte_udp_hdr *)fmt->l4)->dst_port;
	default:
		return 0;
	}
}

doca_be16_t
vxlan_fwd_pinfo_inner_src_port(struct vxlan_fwd_pkt_info *pinfo)
{
	return vxlan_fwd_pinfo_src_port(&pinfo->inner);
}

doca_be16_t
vxlan_fwd_pinfo_inner_dst_port(struct vxlan_fwd_pkt_info *pinfo)
{
	return vxlan_fwd_pinfo_dst_port(&pinfo->inner);
}

doca_be16_t
vxlan_fwd_pinfo_outer_src_port(struct vxlan_fwd_pkt_info *pinfo)
{
	return vxlan_fwd_pinfo_src_port(&pinfo->outer);
}

doca_be16_t
vxlan_fwd_pinfo_outer_dst_port(struct vxlan_fwd_pkt_info *pinfo)
{
	return vxlan_fwd_pinfo_dst_port(&pinfo->outer);
}

/*
 * Parse the packet and set the packet format as represented in the application
 *
 * @data [in]: packet raw data
 * @len [in]: length of the packet raw data in bytes
 * @l2 [in]: whther or not to set the data pointer in layer 2 field in the packet representation in the application\
 * @fmt [out]: the parsed packet as should be represented in the application fo further processing
 * @return: 0 on success, negative value otherwise
 */
static int
vxlan_fwd_parse_pkt_format(uint8_t *data, int len, bool l2,
			    struct vxlan_fwd_pkt_format *fmt)
{
	struct rte_ether_hdr *eth = NULL;
	struct rte_ipv4_hdr *iphdr;
	int l3_off = 0;
	int l4_off = 0;
	int l7_off = 0;

	fmt->l2 = data;
	if (l2) {
		eth = (struct rte_ether_hdr *)data;
		fmt->l2 = data;
		switch (rte_be_to_cpu_16(eth->ether_type)) {
		case RTE_ETHER_TYPE_IPV4:
			l3_off = sizeof(struct rte_ether_hdr);
			break;
		case RTE_ETHER_TYPE_IPV6:
			l3_off = sizeof(struct rte_ether_hdr);
			fmt->l3_type = IPV6;
			return -1;
		case RTE_ETHER_TYPE_ARP:
			return -1;
		default:
			DOCA_LOG_WARN("Unsupported L2 type %x",
				eth->ether_type);
			return -1;
		}
	}

	iphdr = (struct rte_ipv4_hdr *)(data + l3_off);
	if ((iphdr->version_ihl >> 4) != 4)
		return -1;
	if (iphdr->src_addr == 0 || iphdr->dst_addr == 0)
		return -1;
	fmt->l3 = (data + l3_off);
	fmt->l3_type = IPV4;
	l4_off = l3_off + rte_ipv4_hdr_len(iphdr);
	fmt->l4 = data + l4_off;
	switch (iphdr->next_proto_id) {
	case DOCA_PROTO_TCP:
	{
		struct rte_tcp_hdr *tcphdr =
		    (struct rte_tcp_hdr *)(data + l4_off);

		l7_off = l4_off + ((tcphdr->data_off & 0xf0) >> 2);
		if (l7_off > len)
			return -1;
		fmt->l4_type = DOCA_PROTO_TCP;
		fmt->l7 = (data + l7_off);
		break;
	}
	case DOCA_PROTO_UDP:
	{
		struct rte_udp_hdr *udphdr =
		    (struct rte_udp_hdr *)(data + l4_off);

		l7_off = l4_off + sizeof(*udphdr);
		fmt->l4_type = DOCA_PROTO_UDP;
		if (l7_off > len)
			return -1;
		fmt->l7 = (data + l7_off);
		break;
	}
	default:
		DOCA_LOG_INFO("Unsupported L4 %d\n", iphdr->next_proto_id);
		return -1;
	}
	return 0;
}

/*
 * Parse the packet tunneling info
 *
 * @pinfo [in/out]: the packet representation in the application
 * @return: 0 on success, negative value otherwise
 */
static int
vxlan_fwd_parse_is_tun(struct vxlan_fwd_pkt_info *pinfo)
{
	if (pinfo->outer.l3_type != IPV4)
		return 0;

	if (pinfo->outer.l4_type == DOCA_PROTO_UDP) {
		struct rte_udp_hdr *udphdr =
		    (struct rte_udp_hdr *)pinfo->outer.l4;
		uint8_t *udp_data = pinfo->outer.l4
			+ sizeof(struct rte_udp_hdr);

		switch (rte_cpu_to_be_16(udphdr->dst_port)) {
		case DOCA_VXLAN_DEFAULT_PORT:
		{
			struct rte_vxlan_gpe_hdr *vxlanhdr =
			    (struct rte_vxlan_gpe_hdr *)udp_data;

			if (vxlanhdr->vx_flags & 0x08) {
				/*need to check if this gpe*/
				pinfo->tun_type = DOCA_FLOW_TUN_VXLAN;
				pinfo->tun.vni = vxlanhdr->vx_vni;
				pinfo->tun.l2 = true;
			}
			return sizeof(struct rte_vxlan_gpe_hdr) +
			       sizeof(struct rte_udp_hdr);
		}
		default:
			return 0;
		}
	}
	return 0;
}


int
vxlan_fwd_parse_packet(uint8_t *data, int len,
			struct vxlan_fwd_pkt_info *pinfo)
{
	int off = 0;
	int inner_off = 0;

	if (!pinfo) {
		DOCA_LOG_ERR("Pinfo =%p\n", pinfo);
		return -1;
	}
	pinfo->len = len;
	if (vxlan_fwd_parse_pkt_format(data, len, true, &pinfo->outer))
		return -1;

	off = vxlan_fwd_parse_is_tun(pinfo);
	if (pinfo->tun_type == DOCA_FLOW_TUN_NONE || off < 0)
		return 0;

	switch (pinfo->tun_type) {
	case DOCA_FLOW_TUN_VXLAN:
		inner_off = (pinfo->outer.l4 - data) + off;
		if (vxlan_fwd_parse_pkt_format(data + inner_off,
			len - inner_off, pinfo->tun.l2, &pinfo->inner))
			return -1;
		break;
	default:
		break;
	}
	return 0;
}
