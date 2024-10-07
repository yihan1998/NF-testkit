#ifndef _CONFIG_H_
#define _CONFIG_H_

#ifndef CONFIG_NR_CPUS
#define CONFIG_NR_CPUS  1
#endif

#define NR_CPUS CONFIG_NR_CPUS

extern __thread struct rte_ring * fwd_queue;

#endif  /* _CONFIG_H_ */