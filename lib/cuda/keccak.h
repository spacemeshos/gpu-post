#ifndef KECCAK_H
#define KECCAK_H

#include "salsa_kernel.h"

extern "C" void prepare_keccak512(_cudaState *cudaState, const uint8_t *host_pdata);
extern "C" void pre_keccak512(_cudaState *cudaState, int stream, uint64_t nonce, int throughput);
extern "C" void post_keccak512(_cudaState *cudaState, int stream, uint64_t nonce, int throughput);

#endif // #ifndef KECCAK_H
