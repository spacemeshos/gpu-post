#ifndef KECCAK_H
#define KECCAK_H

#include "salsa_kernel.h"

extern "C" void prepare_keccak512(_cudaState *cudaState, const uint8_t *host_pdata, const uint32_t pdata_size);
extern "C" void pre_keccak512(_cudaState *cudaState, int stream, uint64_t nonce, int throughput, uint32_t r);
extern "C" void pre_keccak512_1_1(_cudaState *cudaState, int stream, uint64_t nonce, int throughput);
extern "C" void post_keccak512(_cudaState *cudaState, int stream, uint64_t nonce, int throughput, uint32_t r, uint32_t hash_len_bits);

#endif // #ifndef KECCAK_H
