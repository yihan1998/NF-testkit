#ifndef _NET_H_
#define _NET_H_

#include <stdint.h>

/**
 * net_init - Setup network module
 * 
 * @param nr_cores: Number of cores to rx/tx packets
 * @param nr_rxq: Number of receive queue
 * @param nr_txq: Number of transmission queue
 * 
 * @return Status: 0/-1
*/
int net_init(int nr_cores, int nr_rxq, int nr_txq);

/**
 * net_rx - Poll receive queue
 * 
 * @param qid: RX queue id
 * 
 * @return Number of packets received
*/
int net_rx(int qid);

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
 * @param qid: RX queue id
 * 
 * @return Number of packets transmitted
*/
int net_tx(int qid);

/**
 * net_get_txpkt - Get transmit packet
 * 
 * @param pkt_len: the length of packet
 * 
 * @return Pointer to the start of packet buffer
*/
uint8_t * net_get_txpkt(int pkt_len);

#endif  // _NET_H_