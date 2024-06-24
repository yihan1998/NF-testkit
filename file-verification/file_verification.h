#ifndef _FILE_VERIFICATION_H_
#define _FILE_VERIFICATION_H_

#include <openssl/sha.h>
#include <openssl/evp.h>
#include <openssl/err.h>

extern __thread EVP_MD_CTX * mdctx;

extern int verify_pkt(uint8_t * data, int len);

#endif  /* _DNS_FILTER_H_ */