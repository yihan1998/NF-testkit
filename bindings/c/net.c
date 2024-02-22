
#include "net.h"

#include <rte_eal.h>

int net_init(int nr_cores, int nr_rxq, int nr_txq) {
    int argc = 7;
    char core_list[16];
    int ret;

    sprintf(core_list, "0-%d", nr_cores - 1);

    char * argv[] = {
        "-l", core_list,
        "-n", "4",
        "-a", "4b:00.0", ""};

    ret = rte_eal_init(argc, argv);
	if (ret < 0) {
		return -1;
    }

    return 0;
}

int net_rx(int qid) {
    return 0;
}

uint8_t * net_get_rxpkt(int * pkt_len) {
    return NULL;
}

int net_tx(int qid) {
    return 0;
}

uint8_t * net_get_txpkt(int pkt_len) {
    return NULL;
}
