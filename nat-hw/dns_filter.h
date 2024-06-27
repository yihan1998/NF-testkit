#ifndef _DNS_FILTER_H_
#define _DNS_FILTER_H_

#include <regex.h>

#define MAX_RULES 100
#define MAX_REGEX_LENGTH 256

extern __thread regex_t compiled_rules[MAX_RULES];
extern __thread int rule_count;

extern int load_regex_rules(void);
extern void print_domain_name(const unsigned char * buffer, int* position, unsigned char * domain_name);
extern int parse_dns_query(const unsigned char * buffer, int size);
extern int find_matching_rule(const char * domain_name);

#endif  /* _DNS_FILTER_H_ */