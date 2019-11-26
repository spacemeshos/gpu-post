#ifndef SALSA_KERNEL_H
#define SALSA_KERNEL_H

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#ifndef __APPLE__
#include <malloc.h>
#endif
#include <string.h>
#include <cuda_runtime.h>

#include "api_internal.h"

// Define work unit size
#define TOTAL_WARP_LIMIT 4096
#define WU_PER_WARP (32 / THREADS_PER_WU)
#define WU_PER_BLOCK (WU_PER_WARP*WARPS_PER_BLOCK)
#define WU_PER_LAUNCH (GRID_BLOCKS*WU_PER_BLOCK)

// make scratchpad size dependent on N and LOOKUP_GAP
#define SCRATCH   (((N+LOOKUP_GAP-1)/LOOKUP_GAP)*32)

typedef unsigned int uint32_t; // define this as 32 bit type derived from int

static __inline bool IS_SCRYPT_JANE() { return true; }

// If we're in C++ mode, we're either compiling .cu files or scrypt.cpp

#ifdef __NVCC__

/**
 * An pure virtual interface for a CUDA kernel implementation.
 * TODO: encapsulate the kernel launch parameters in some kind of wrapper.
 */
class KernelInterface
{
public:
	virtual void set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V) = 0;
	virtual bool run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int gpu_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int r, unsigned int p, unsigned int batch, unsigned int LOOKUP_GAP, bool benchmark) = 0;

	virtual char get_identifier() = 0;
	virtual int get_major_version() { return 1; }
	virtual int get_minor_version() { return 0; }
	virtual int max_warps_per_block() = 0;
	virtual int threads_per_wu() { return 1; }
	virtual bool support_lookup_gap() { return false; }
	virtual cudaSharedMemConfig shared_mem_config() { return cudaSharedMemBankSizeDefault; }
	virtual cudaFuncCache cache_config() { return cudaFuncCachePreferNone; }
};

// Not performing error checking is actually bad, but...
#define checkCudaErrors(x) x
#define getLastCudaError(x)
#else
class KernelInterface;
#endif // #ifdef __NVCC__

typedef struct {
	int			cuda_id;
	unsigned	context_blocks;
	unsigned	context_wpb;
	bool		context_concurrent;
	bool		keccak_inited;
	KernelInterface * context_kernel;
	uint32_t	*context_idata[2];
	uint32_t	*context_odata[2];
	cudaStream_t context_streams[2];
	uint32_t	*context_X[2];
	uint8_t		*context_L[2];
	cudaEvent_t context_serialize[2];
	uint8_t		*context_labels[2];
} _cudaState;

// CUDA externals
extern _cudaState *initCuda(struct cgpu_info *cgpu, uint32_t N, uint32_t r, uint32_t p);
extern uint32_t *cuda_transferbuffer(_cudaState *cudaState, int stream);
extern uint8_t *cuda_hashbuffer(_cudaState *cudaState, int stream);

extern void cuda_scrypt_serialize(struct cgpu_info *cgpu, _cudaState *cudaState, int stream);
extern void cuda_scrypt_core(struct cgpu_info *cgpu, _cudaState *cudaState, int stream, unsigned int N, unsigned int r, unsigned int p);
extern void cuda_scrypt_done(_cudaState *cudaState, int stream);
extern void cuda_scrypt_DtoH(_cudaState *cudaState, uint8_t *X, int stream);
extern bool cuda_scrypt_sync(struct cgpu_info *cgpu, _cudaState *cudaState, int stream);
extern void cuda_scrypt_flush(_cudaState *cudaState, int stream);

#endif // #ifndef SALSA_KERNEL_H
