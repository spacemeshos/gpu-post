//
// Contains the autotuning logic and some utility functions.
// Note that all CUDA kernels have been moved to other .cu files
//

#include <stdio.h>
#include <map>
#include <algorithm>
#include <ctype.h> // tolower
#include "cuda_helper.h"

#include "salsa_kernel.h"

//#include "nv_kernel.h"
#include "titan_kernel.h"

#include "api_internal.h"

#if defined(_WIN64) || defined(__x86_64__) || defined(__64BIT__)
#define MAXMEM 0x300000000ULL  // 12 GB (the largest Kepler)
#else
#define MAXMEM  0xFFFFFFFFULL  // nearly 4 GB (32 bit limitations)
#endif

// require CUDA 5.5 driver API
#define DMAJ 5
#define DMIN 5

// define some error checking macros
#define DELIMITER '/'
#define __FILENAME__ ( strrchr(__FILE__, DELIMITER) != NULL ? strrchr(__FILE__, DELIMITER)+1 : __FILE__ )

#undef checkCudaErrors
#define checkCudaErrors(gpuId, x) \
do { \
	cudaGetLastError(); \
	x; \
	cudaError_t err = cudaGetLastError(); \
	if (err != cudaSuccess && !g_spacemesh_api_abort_flag) { \
		applog(LOG_ERR, "GPU #%d: Err %d: %s (%s:%d)", gpuId, err, cudaGetErrorString(err), __FILENAME__, __LINE__); \
	} \
} while (0)

int find_optimal_concurency(struct cgpu_info *cgpu, _cudaState *cudaState, KernelInterface* &kernel, uint32_t N, uint32_t r, uint32_t p, bool &concurrent, int &wpb);

_cudaState *initCuda(struct cgpu_info *cgpu, uint32_t N, uint32_t r, uint32_t p, uint32_t hash_len_bits, bool throttled)
{
	_cudaState *cudaState = (_cudaState *)calloc(1, sizeof(_cudaState));
	int GRID_BLOCKS, WARPS_PER_BLOCK;

	checkCudaErrors(cgpu->driver_id, cudaSetDevice(cgpu->driver_id));
	checkCudaErrors(cgpu->driver_id, cudaDeviceSynchronize());
	checkCudaErrors(cgpu->driver_id, cudaDeviceReset());
	checkCudaErrors(cgpu->driver_id, cudaSetDevice(cgpu->driver_id));
	checkCudaErrors(cgpu->driver_id, cudaSetDeviceFlags(cudaDeviceScheduleYield));

	KernelInterface *kernel;
	bool concurrent;
	GRID_BLOCKS = find_optimal_concurency(cgpu, cudaState, kernel, N, r, p, concurrent, WARPS_PER_BLOCK);

	if (GRID_BLOCKS == 0) {
		return 0;
	}

	if (throttled) {
		GRID_BLOCKS = min(GRID_BLOCKS, cgpu->gpu_core_count / 2);
	}

	unsigned int THREADS_PER_WU = kernel->threads_per_wu();
	unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * 32 * r;
	unsigned int labels_size = (WU_PER_LAUNCH * hash_len_bits + 7) / 8;

	// allocate device memory for scrypt_core inputs and outputs
	uint32_t *tmp;
	checkCudaErrors(cgpu->driver_id, cudaMalloc((void **) &tmp, mem_size)); cudaState->context_idata = tmp;
	checkCudaErrors(cgpu->driver_id, cudaMalloc((void **) &tmp, mem_size)); cudaState->context_odata = tmp;

	// allocate pinned host memory for scrypt hashes
	checkCudaErrors(cgpu->driver_id, cudaHostAlloc((void **) &tmp, labels_size, cudaHostAllocDefault)); cudaState->context_L = (uint8_t*)tmp;

	// allocate pinned host memory for scrypt_core input/output
	checkCudaErrors(cgpu->driver_id, cudaHostAlloc((void **) &tmp, 32, cudaHostAllocDefault)); cudaState->context_X = (uint64_t*)tmp;

	checkCudaErrors(cgpu->driver_id, cudaMalloc((void **) &tmp, labels_size)); cudaState->context_labels = (uint8_t*)tmp;

	checkCudaErrors(cgpu->driver_id, cudaMalloc((void **)&tmp, 32)); cudaState->context_solutions = (uint64_t*)tmp;

	cudaState->cuda_id = cgpu->driver_id;
	cudaState->context_kernel = kernel;
	cudaState->context_concurrent = concurrent;
	cudaState->context_blocks = GRID_BLOCKS;
	cudaState->context_wpb = WARPS_PER_BLOCK;
	cgpu->thread_concurrency = WU_PER_LAUNCH;

	return cudaState;
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
	return 128;
}

int find_optimal_concurency(struct cgpu_info *cgpu, _cudaState *cudaState, KernelInterface* &kernel, uint32_t N, uint32_t r, uint32_t p, bool &concurrent, int &WARPS_PER_BLOCK)
{
	int optimal_blocks = 0;

	cudaDeviceProp props;
	checkCudaErrors(cgpu->driver_id, cudaGetDeviceProperties(&props, cgpu->driver_id));
	concurrent = (props.concurrentKernels > 0);

	WARPS_PER_BLOCK = -1;

	kernel = new TitanKernel();

	if (kernel->get_major_version() > props.major || kernel->get_major_version() == props.major && kernel->get_minor_version() > props.minor)
	{
		applog(LOG_ERR, "GPU #%d: FATAL: the '%c' kernel requires %d.%d capability!", cgpu->driver_id, kernel->get_identifier(), kernel->get_major_version(), kernel->get_minor_version());
		return 0;
	}

	// set whatever cache configuration and shared memory bank mode the kernel prefers
	checkCudaErrors(cgpu->driver_id, cudaDeviceSetCacheConfig(kernel->cache_config()));
	checkCudaErrors(cgpu->driver_id, cudaDeviceSetSharedMemConfig(kernel->shared_mem_config()));

	if (cgpu->lookup_gap == 0) {
		cgpu->lookup_gap = 1;
	}

	if (!kernel->support_lookup_gap() && cgpu->lookup_gap > 1) {
		applog(LOG_WARNING, "GPU #%d: the '%c' kernel does not support a lookup gap", cgpu->driver_id, kernel->get_identifier());
		cgpu->lookup_gap = 1;
	}

	// number of threads collaborating on one work unit (hash)
	unsigned int THREADS_PER_WU = kernel->threads_per_wu();
	size_t szPerWarp = r * SCRATCH(N, cgpu->lookup_gap) * WU_PER_WARP * sizeof(uint32_t);
	//applog(LOG_INFO, "WU_PER_WARP=%u, THREADS_PER_WU=%u, LOOKUP_GAP=%u, BACKOFF=%u, SCRATCH=%u", WU_PER_WARP, THREADS_PER_WU, LOOKUP_GAP, BACKOFF, SCRATCH);
	applog(LOG_INFO, "GPU #%d: %d hashes / %.1f MB per warp.", cgpu->driver_id, WU_PER_WARP, double(szPerWarp) / (1024.0 * 1024.0));

	uint32_t *d_V = NULL;
	{
		// compute no. of warps to allocate the largest number producing a single memory block
		// PROBLEM: one some devices, ALL allocations will fail if the first one failed. This sucks.
		int warpmax = (int)min((size_t)TOTAL_WARP_LIMIT, (size_t)(cgpu->gpu_max_alloc / szPerWarp));

		// run a bisection algorithm for memory allocation (way more reliable than the previous approach)
		int best = 0;
		int warp = (warpmax+1)/2;
		int interval = (warpmax+1)/2;
		while (interval > 0)
		{
			cudaGetLastError(); // clear the error state
			cudaMalloc((void **)&d_V, (size_t)(szPerWarp * warp));
			if (cudaGetLastError() == cudaSuccess) {
				checkCudaErrors(cgpu->driver_id, cudaFree(d_V)); d_V = NULL;
				if (warp > best) {
					best = warp;
				}
				if (warp == warpmax) {
					break;
				}
				interval = (interval+1)/2;
				warp += interval;
				if (warp > warpmax) {
					warp = warpmax;
				}
			}
			else {
				interval = interval/2;
				warp -= interval;
				if (warp < 1) {
					warp = 1;
				}
			}
		}

		// back off a bit from the largest possible allocation size
		cudaState->max_warps = ((100 - cgpu->backoff)*best + 50) / 100;

		// now allocate a buffer for determined MAXWARPS setting
		cudaGetLastError(); // clear the error state
		cudaMalloc((void **)&d_V, (size_t)r * SCRATCH(N, cgpu->lookup_gap) * WU_PER_WARP * cudaState->max_warps * sizeof(uint32_t));
		if (cudaGetLastError() == cudaSuccess) {
			for (int i = 0; i < cudaState->max_warps; ++i) {
				cudaState->h_V[i] = d_V + r * SCRATCH(N, cgpu->lookup_gap) * WU_PER_WARP * i;
			}
		}
		else
		{
			applog(LOG_ERR, "GPU #%d: FATAL: Launch config '%s' requires too much memory!", cgpu->driver_id, cgpu->device_config);
			return 0;
		}
	}

	kernel->set_scratchbuf_constants(cudaState->max_warps, cudaState->h_V);

	if (cgpu->device_config != NULL && strcasecmp("auto", cgpu->device_config)) {
		applog(LOG_WARNING, "GPU #%d: Given launch config '%s' does not validate.", cgpu->driver_id, cgpu->device_config);
	}

	// Heuristics to find a good kernel launch configuration

	// base the initial block estimate on the number of multiprocessors
	int device_cores = props.multiProcessorCount * _ConvertSMVer2Cores(props.major, props.minor);

	// defaults, in case nothing else is chosen below
	optimal_blocks = device_cores;
	WARPS_PER_BLOCK = 4;

	while (optimal_blocks > 0 && optimal_blocks * WARPS_PER_BLOCK > cudaState->max_warps) {
		optimal_blocks--;
	}

	applog(LOG_INFO, "GPU #%d: using launch configuration %c%dx%d", cgpu->driver_id, kernel->get_identifier(), optimal_blocks, WARPS_PER_BLOCK);

	if (cudaState->max_warps != optimal_blocks * WARPS_PER_BLOCK)
	{
		cudaState->max_warps = optimal_blocks * WARPS_PER_BLOCK;
		checkCudaErrors(cgpu->driver_id, cudaFree(d_V)); d_V = NULL;

		cudaGetLastError(); // clear the error state
		cudaMalloc((void **)&d_V, (size_t)r * SCRATCH(N, cgpu->lookup_gap) * WU_PER_WARP * cudaState->max_warps * sizeof(uint32_t));
		if (cudaGetLastError() == cudaSuccess) {
			for (int i = 0; i < cudaState->max_warps; ++i) {
				cudaState->h_V[i] = d_V + r * SCRATCH(N, cgpu->lookup_gap) * WU_PER_WARP * i;
			}
			// update pointers to scratch buffer in constant memory after reallocation
			kernel->set_scratchbuf_constants(cudaState->max_warps, cudaState->h_V);
		}
		else
		{
			applog(LOG_ERR, "GPU #%d: Unable to allocate enough memory for launch config '%s'.", cgpu->driver_id, cgpu->device_config);
		}
	}

	return optimal_blocks;
}

void cuda_scrypt_serialize(struct cgpu_info *cgpu, _cudaState *cudaState, int stream)
{
	cudaError_t err = cudaStreamSynchronize(0);
	if (err != cudaSuccess) {
		if (!g_spacemesh_api_abort_flag) {
			applog(LOG_ERR, "GPU #%d: CUDA error `%s` while waiting the kernel.", cudaState->cuda_id, cudaGetErrorString(err));
		}
	}
}

void cuda_scrypt_done(_cudaState *cudaState, int stream)
{
	cudaError_t err = cudaStreamSynchronize(0);
	if (err != cudaSuccess) {
		if (!g_spacemesh_api_abort_flag) {
			applog(LOG_ERR, "GPU #%d: CUDA error `%s` while waiting the kernel.", cudaState->cuda_id, cudaGetErrorString(err));
		}
	}
}

void cuda_scrypt_flush(_cudaState *cudaState, int stream)
{
	cudaError_t err = cudaStreamSynchronize(0);
	if (err != cudaSuccess) {
		if (!g_spacemesh_api_abort_flag) {
			applog(LOG_ERR, "GPU #%d: CUDA error `%s` while waiting the kernel.", cudaState->cuda_id, cudaGetErrorString(err));
		}
	}
}

void cuda_scrypt_core(_cudaState *cudaState, int stream, unsigned int N, unsigned int r, unsigned int p, unsigned int LOOKUP_GAP, unsigned int BATCHSIZE)
{
	unsigned int GRID_BLOCKS = cudaState->context_blocks;
	unsigned int WARPS_PER_BLOCK = cudaState->context_wpb;
	unsigned int THREADS_PER_WU = cudaState->context_kernel->threads_per_wu();

	// setup execution parameters
	dim3 grid(GRID_BLOCKS, 1, 1);
	dim3 threads(THREADS_PER_WU*WU_PER_BLOCK, 1, 1);

	cudaState->context_kernel->run_kernel(grid, threads, WARPS_PER_BLOCK, cudaState->cuda_id,
		0, cudaState->context_idata, cudaState->context_odata,
		N, r, p, BATCHSIZE, LOOKUP_GAP
	);
}

void cuda_scrypt_DtoH(_cudaState *cudaState, uint8_t *X, int stream, uint32_t size)
{
	// copy result from device to host (asynchronously)
	checkCudaErrors(cudaState->cuda_id, cudaMemcpyAsync(X, cudaState->context_labels, size, cudaMemcpyDeviceToHost));
}

void cuda_solutions_DtoH(_cudaState *cudaState, int stream)
{
	// copy result from device to host (asynchronously)
	checkCudaErrors(cudaState->cuda_id, cudaMemcpyAsync(cudaState->context_X, cudaState->context_solutions, 32, cudaMemcpyDeviceToHost));
}

void cuda_solutions_HtoD(_cudaState *cudaState, int stream)
{
	// copy result from device to host (asynchronously)
	checkCudaErrors(cudaState->cuda_id, cudaMemcpyAsync(cudaState->context_solutions, cudaState->context_X, 32, cudaMemcpyHostToDevice));
}

bool cuda_scrypt_sync(struct cgpu_info *cgpu, _cudaState *cudaState, int stream)
{
	cudaError_t err = cudaStreamSynchronize(0);
	if (err != cudaSuccess) {
		if (!g_spacemesh_api_abort_flag) {
			applog(LOG_ERR, "GPU #%d: CUDA error `%s` while waiting the kernel.", cudaState->cuda_id, cudaGetErrorString(err));
		}
		return false;
	}
	return true;
}

uint64_t* cuda_transferbuffer(_cudaState *cudaState, int stream)
{
	return cudaState->context_X;
}

uint8_t* cuda_hashbuffer(_cudaState *cudaState, int stream)
{
	return cudaState->context_L;
}
