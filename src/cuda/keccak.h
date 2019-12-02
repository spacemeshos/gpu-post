#ifndef KECCAK_H
#define KECCAK_H

#include "salsa_kernel.h"

extern "C" void prepare_keccak512(_cudaState *cudaState, const uint8_t *host_pdata, const uint32_t pdata_size);
extern "C" void pre_keccak512(_cudaState *cudaState, int stream, uint64_t nonce, int throughput, uint32_t r);
extern "C" void post_keccak512(_cudaState *cudaState, int stream, uint64_t nonce, int throughput, uint32_t r);

#endif // #ifndef KECCAK_H
