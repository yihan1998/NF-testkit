#include <stdlib.h>

#include <rte_security.h>
#include <rte_ethdev.h>
#include <rte_ip.h>
#include <rte_esp.h>
#include <rte_crypto.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>

#define NB_MBUF 8192
#define NB_RX_DESC 1024
#define NB_TX_DESC 1024
#define IPSEC_SPI 12345

struct rte_mempool *mbuf_pool;
struct rte_security_ctx *sec_ctx;
struct rte_security_session *session;

static struct rte_security_session_conf ipsec_sess_conf;
static struct rte_crypto_sym_xform sym_xform;

// IPsec Security Association (SA) setup function
void setup_ipsec_sa(struct rte_security_ctx *sec_ctx, uint16_t port_id) {
    // Configure IPsec Security Session
    memset(&ipsec_sess_conf, 0, sizeof(struct rte_security_session_conf));
    
    ipsec_sess_conf.action_type = RTE_SECURITY_ACTION_TYPE_INLINE_CRYPTO;
    ipsec_sess_conf.protocol = RTE_SECURITY_IPSEC_SA_PROTO_ESP;
    ipsec_sess_conf.ipsec.options.esn = 0;
    ipsec_sess_conf.ipsec.options.udp_encap = 0;
    ipsec_sess_conf.ipsec.mode = RTE_SECURITY_IPSEC_SA_MODE_TRANSPORT;
    ipsec_sess_conf.ipsec.spi = IPSEC_SPI; // Security Parameter Index (SPI)
    ipsec_sess_conf.ipsec.direction = RTE_SECURITY_IPSEC_SA_DIR_EGRESS;
    ipsec_sess_conf.crypto_xform = &sym_xform;

    // Configure Crypto Transformation for AES-GCM (combined cipher & auth)
    memset(&sym_xform, 0, sizeof(struct rte_crypto_sym_xform));
    sym_xform.type = RTE_CRYPTO_SYM_XFORM_AEAD;
    sym_xform.aead.algo = RTE_CRYPTO_AEAD_AES_GCM;
    sym_xform.aead.op = RTE_CRYPTO_AEAD_OP_ENCRYPT;
    sym_xform.aead.key.length = 16;
    sym_xform.aead.key.data = (uint8_t *)"encryptionkey!!!"; // Key for AES-GCM
    sym_xform.aead.digest_length = 16; // AES-GCM MAC length
    sym_xform.aead.iv.length = 12;     // IV length (12 bytes)

    // Create the security session
    session = rte_security_session_create(sec_ctx, &ipsec_sess_conf, mbuf_pool);
    if (!session) {
        printf("Failed to create security session\n");
        exit(1);
    }
}

// Packet TX function with inline IPsec
static int send_packets(uint16_t port_id, struct rte_mbuf **pkts, uint16_t nb_pkts) {
    for (int i = 0; i < nb_pkts; i++) {
        struct rte_mbuf *pkt = pkts[i];
        // Attach the security session to the packet for inline processing
        if (rte_security_set_pkt_metadata(sec_ctx, session, pkt, NULL)) {
            printf("Failed to attach security session\n");
            continue;
        }
    }
    return 0;
}

int main(int argc, char *argv[]) {
    uint16_t port_id = 0;
    struct rte_eth_conf port_conf = { 0 };
    struct rte_security_ctx *sec_ctx;
    
    // Initialize DPDK EAL
    rte_eal_init(argc, argv);
    
    // Create the mempool
    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NB_MBUF, 0, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());

    // Initialize Ethernet port
    rte_eth_dev_configure(port_id, 1, 1, &port_conf);
    rte_eth_rx_queue_setup(port_id, 0, NB_RX_DESC, rte_eth_dev_socket_id(port_id), NULL, mbuf_pool);
    rte_eth_tx_queue_setup(port_id, 0, NB_TX_DESC, rte_eth_dev_socket_id(port_id), NULL);

    // Start Ethernet device
    rte_eth_dev_start(port_id);

    // Get the security context for the port (IPsec offloading must be supported by the NIC)
    sec_ctx = rte_eth_dev_get_sec_ctx(port_id);
    if (!sec_ctx) {
        printf("NIC does not support inline IPsec\n");
        return -1;
    }

    // Setup IPsec SA
    setup_ipsec_sa(sec_ctx, port_id);

    // Main packet processing loop
    while (1) {
        struct rte_mbuf *pkts[32];
        uint16_t nb_rx, nb_tx;
		nb_rx = rte_eth_rx_burst(port_id, 0, pkts, 32);

        if (nb_rx > 0) {
            send_packets(port_id, pkts, nb_rx);
        	nb_tx = rte_eth_tx_burst(port_id, 0, pkts, nb_rx);
            if (unlikely(nb_tx < nb_rx)) {
                do {
                    rte_pktmbuf_free(pkts[nb_tx]);
                } while (++nb_tx < nb_rx);
            }
        }
    }

    return 0;
}
