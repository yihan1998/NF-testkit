#ifndef _NET_H_
#define _NET_H_

#include <stdint.h>

/**
 * net_init - Init network module
 * 
 * @param nb_cores: Number of cores to rx/tx packets
 * @param nb_rxq: Number of receive queue
 * @param nb_txq: Number of transmission queue
 * 
 * @return Status: 0/-1
*/
int net_init(int nb_cores, int nb_rxq, int nb_txq);

/**
 * net_setup - Setup local network module
 * 
 * @param lcore_id: Core id
 * 
 * @return Status: 0/-1
*/
int net_setup(int lcore_id);

/**
 * net_rx - Poll receive queue
 * 
 * @return Number of packets received
*/
int net_rx(void);

/**
 * net_get_rxpkt - Get receive packet
 * 
 * @param pkt_len: the length of packet
 * 
 * @return Pointer to the start of packet buffer
*/
uint8_t * net_get_rxpkt(int * pkt_len);

/**
 * net_tx - Flush transmission queue
 * 
 * @return Number of packets transmitted
*/
int net_tx(void);

/**
 * net_get_txpkt - Get transmit packet
 * 
 * @param pkt_len: the length of packet
 * 
 * @return Pointer to the start of packet buffer
*/
uint8_t * net_get_txpkt(int pkt_len);

/**
 * net_direct_flow_to_core - Direct a given flow to a certain core
 * 
 * @param pkt_len: the length of packet
 * 
 * @return Pointer to the start of packet buffer
*/
int net_direct_flow_to_queue(uint16_t qid, uint16_t proto,
                uint32_t src_ip, uint32_t src_ip_mask, uint32_t dst_ip, uint32_t dst_ip_mask, 
                uint16_t src_port, uint16_t src_port_mask,  uint16_t dst_port, uint16_t dst_port_mask);

#endif  // _NET_H_