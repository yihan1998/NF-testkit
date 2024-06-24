#include "skbuff.h"

__thread struct rte_mempool * skb_mp;

struct sk_buff * alloc_skb(uint8_t * data, unsigned int data_len) {
    struct sk_buff * skb = NULL;

    rte_mempool_get(skb_mp, (void *)&skb);
    if (!skb) {
        return NULL;
    }

    skb->len = data_len;
    skb->ptr = data;

    return skb;
}

void free_skb(struct sk_buff * skb) {
    rte_mempool_put(skb_mp, skb);
}

int skb_init(int nr_cores) {
    char name[RTE_MEMZONE_NAMESIZE];
    struct rte_mempool * mp;

    for (int i = 0; i < nr_cores; i++) {
        sprintf(name, "skb_pool_%d", i);
        mp = rte_mempool_create(name, 2048, sizeof(struct sk_buff) + 1460, RTE_MEMPOOL_CACHE_MAX_SIZE, 0, NULL, NULL, NULL, NULL, rte_socket_id(), 0);
        assert(mp != NULL);
    }

    return 0;
}