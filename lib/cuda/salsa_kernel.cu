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
{ \
	cudaGetLastError(); \
	x; \
	cudaError_t err = cudaGetLastError(); \
	if (err != cudaSuccess && !abort_flag) \
		applog(LOG_ERR, "GPU #%d: Err %d: %s (%s:%d)", gpuId, err, cudaGetErrorString(err), __FILENAME__, __LINE__); \
}

// some globals containing pointers to device memory (for chunked allocation)
// [MAX_GPUS] indexes up to MAX_GPUS threads (0...MAX_GPUS-1)
int       MAXWARPS[MAX_GPUDEVICES];
uint32_t* h_V[MAX_GPUDEVICES][TOTAL_WARP_LIMIT * 64];          // NOTE: the *64 prevents buffer overflow for --keccak
uint32_t  h_V_extra[MAX_GPUDEVICES][TOTAL_WARP_LIMIT * 64];    //       with really large kernel launch configurations

int find_optimal_blockcount(struct cgpu_info *cgpu, KernelInterface* &kernel, uint32_t N, bool &concurrent, int &wpb);

_cudaState *initCuda(struct cgpu_info *cgpu, uint32_t N)
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
	GRID_BLOCKS = find_optimal_blockcount(cgpu, kernel, N, concurrent, WARPS_PER_BLOCK);

	if (GRID_BLOCKS == 0) {
		return 0;
	}

	unsigned int THREADS_PER_WU = kernel->threads_per_wu();
	unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * 32;
//	unsigned int state_size = WU_PER_LAUNCH * sizeof(uint32_t) * 8;
	unsigned int labels_size = WU_PER_LAUNCH;

	// allocate device memory for scrypt_core inputs and outputs
	uint32_t *tmp;
	checkCudaErrors(cgpu->driver_id, cudaMalloc((void **) &tmp, mem_size)); cudaState->context_idata[0] = tmp;
	checkCudaErrors(cgpu->driver_id, cudaMalloc((void **) &tmp, mem_size)); cudaState->context_idata[1] = tmp;
	checkCudaErrors(cgpu->driver_id, cudaMalloc((void **) &tmp, mem_size)); cudaState->context_odata[0] = tmp;
	checkCudaErrors(cgpu->driver_id, cudaMalloc((void **) &tmp, mem_size)); cudaState->context_odata[1] = tmp;

	// allocate pinned host memory for scrypt hashes
	checkCudaErrors(cgpu->driver_id, cudaHostAlloc((void **) &tmp, labels_size, cudaHostAllocDefault)); cudaState->context_L[0] = (uint8_t*)tmp;
	checkCudaErrors(cgpu->driver_id, cudaHostAlloc((void **) &tmp, labels_size, cudaHostAllocDefault)); cudaState->context_L[1] = (uint8_t*)tmp;

	// allocate pinned host memory for scrypt_core input/output
	checkCudaErrors(cgpu->driver_id, cudaHostAlloc((void **) &tmp, mem_size, cudaHostAllocDefault)); cudaState->context_X[0] = tmp;
	checkCudaErrors(cgpu->driver_id, cudaHostAlloc((void **) &tmp, mem_size, cudaHostAllocDefault)); cudaState->context_X[1] = tmp;

	checkCudaErrors(cgpu->driver_id, cudaMalloc((void **) &tmp, labels_size)); cudaState->context_labels[0] = (uint8_t*)tmp;
	checkCudaErrors(cgpu->driver_id, cudaMalloc((void **) &tmp, labels_size)); cudaState->context_labels[1] = (uint8_t*)tmp;

	// create two CUDA streams
	cudaStream_t tmp2;
	checkCudaErrors(cgpu->driver_id, cudaStreamCreate(&tmp2)); cudaState->context_streams[0] = tmp2;
	checkCudaErrors(cgpu->driver_id, cudaStreamCreate(&tmp2)); cudaState->context_streams[1] = tmp2;

	// events used to serialize the kernel launches (we don't want any overlapping of kernels)
	cudaEvent_t tmp4;
	checkCudaErrors(cgpu->driver_id, cudaEventCreateWithFlags(&tmp4, cudaEventDisableTiming)); cudaState->context_serialize[0] = tmp4;
	checkCudaErrors(cgpu->driver_id, cudaEventCreateWithFlags(&tmp4, cudaEventDisableTiming)); cudaState->context_serialize[1] = tmp4;
	checkCudaErrors(cgpu->driver_id, cudaEventRecord(cudaState->context_serialize[1]));

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
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{ 0x10, 8   }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11, 8   }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12, 8   }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13, 8   }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32  }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48  }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class - GK104 = 1536 cores / 8 SMs
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x50, 128 }, // Maxwell First Generation (SM 5.0) GTX750/750Ti
		{ 0x52, 128 }, // Maxwell Second Generation (SM 5.2) GTX980 = 2048 cores / 16 SMs - GTX970 1664 cores / 13 SMs
		{ 0x61, 128 }, // Pascal GeForce (SM 6.1)
		{ -1, -1 },
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	applog(LOG_WARNING, "MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM", major, minor, 128);
	return 128;
}

int find_optimal_blockcount(struct cgpu_info *cgpu, KernelInterface* &kernel, uint32_t N, bool &concurrent, int &WARPS_PER_BLOCK)
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
	unsigned int LOOKUP_GAP = cgpu->lookup_gap;
	unsigned int BACKOFF = cgpu->backoff;
	double szPerWarp = (double)(SCRATCH * WU_PER_WARP * sizeof(uint32_t));
	//applog(LOG_INFO, "WU_PER_WARP=%u, THREADS_PER_WU=%u, LOOKUP_GAP=%u, BACKOFF=%u, SCRATCH=%u", WU_PER_WARP, THREADS_PER_WU, LOOKUP_GAP, BACKOFF, SCRATCH);
	applog(LOG_INFO, "GPU #%d: %d hashes / %.1f MB per warp.", cgpu->driver_id, WU_PER_WARP, szPerWarp / (1024.0 * 1024.0));

	// compute highest MAXWARPS numbers for kernels allowing cudaBindTexture to succeed
	int MW_1D_4 = 134217728 / (SCRATCH * WU_PER_WARP / 4); // for uint4_t textures
	int MW_1D_2 = 134217728 / (SCRATCH * WU_PER_WARP / 2); // for uint2_t textures
	int MW_1D = kernel->get_texel_width() == 2 ? MW_1D_2 : MW_1D_4;

	uint32_t *d_V = NULL;
	{
		// compute no. of warps to allocate the largest number producing a single memory block
		// PROBLEM: one some devices, ALL allocations will fail if the first one failed. This sucks.
		size_t MEM_LIMIT = (size_t)min((unsigned long long)MAXMEM, (unsigned long long)props.totalGlobalMem);
		int warpmax = (int)min((unsigned long long)TOTAL_WARP_LIMIT, (unsigned long long)(MEM_LIMIT / szPerWarp));

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
				if (warp > best) best = warp;
				if (warp == warpmax) break;
				interval = (interval+1)/2;
				warp += interval;
				if (warp > warpmax) warp = warpmax;
			}
			else
			{
				interval = interval/2;
				warp -= interval;
				if (warp < 1) warp = 1;
			}
		}
		// back off a bit from the largest possible allocation size
		MAXWARPS[cgpu->driver_id] = ((100-BACKOFF)*best+50)/100;

		// now allocate a buffer for determined MAXWARPS setting
		cudaGetLastError(); // clear the error state
		cudaMalloc((void **)&d_V, (size_t)SCRATCH * WU_PER_WARP * MAXWARPS[cgpu->driver_id] * sizeof(uint32_t));
		if (cudaGetLastError() == cudaSuccess) {
			for (int i = 0; i < MAXWARPS[cgpu->driver_id]; ++i) {
				h_V[cgpu->driver_id][i] = d_V + SCRATCH * WU_PER_WARP * i;
			}
		}
		else
		{
			applog(LOG_ERR, "GPU #%d: FATAL: Launch config '%s' requires too much memory!", cgpu->driver_id, cgpu->device_config);
			return 0;
		}
	}

	kernel->set_scratchbuf_constants(MAXWARPS[cgpu->driver_id], h_V[cgpu->driver_id]);

	if (cgpu->device_config != NULL && strcasecmp("auto", cgpu->device_config)) {
		applog(LOG_WARNING, "GPU #%d: Given launch config '%s' does not validate.", cgpu->driver_id, cgpu->device_config);
	}

	// Heuristics to find a good kernel launch configuration

	// base the initial block estimate on the number of multiprocessors
	int device_cores = props.multiProcessorCount * _ConvertSMVer2Cores(props.major, props.minor);

	// defaults, in case nothing else is chosen below
	optimal_blocks = 4 * device_cores / WU_PER_WARP;
	WARPS_PER_BLOCK = 2;

	// Based on compute capability, pick a known good block x warp configuration.
	if (props.major >= 3)
	{
		if (props.major == 3 && props.minor == 5) // GK110 (Tesla K20X, K20, GeForce GTX TITAN)
		{
			// TODO: what to do with Titan and Tesla K20(X)?
			// for now, do the same as for GTX 660Ti (2GB)
			optimal_blocks = (int)(optimal_blocks * 0.8809524);
			WARPS_PER_BLOCK = 2;
		}
		else // GK104, GK106, GK107 ...
		{
			if (MAXWARPS[cgpu->driver_id] > (int)(optimal_blocks * 1.7261905) * 2)
			{
				// this results in 290x2 configuration on GTX 660Ti (3GB)
				// but it requires 3GB memory on the card!
				optimal_blocks = (int)(optimal_blocks * 1.7261905);
				WARPS_PER_BLOCK = 2;
			}
			else
			{
				// this results in 148x2 configuration on GTX 660Ti (2GB)
				optimal_blocks = (int)(optimal_blocks * 0.8809524);
				WARPS_PER_BLOCK = 2;
			}
		}
	}
	// 1st generation Fermi (compute 2.0) GF100, GF110
	else if (props.major == 2 && props.minor == 0)
	{
		// this results in a 60x4 configuration on GTX 570
		optimal_blocks = 4 * device_cores / WU_PER_WARP;
		WARPS_PER_BLOCK = 4;
	}
	// 2nd generation Fermi (compute 2.1) GF104,106,108,114,116
	else if (props.major == 2 && props.minor == 1)
	{
		// this results in a 56x2 configuration on GTX 460
		optimal_blocks = props.multiProcessorCount * 8;
		WARPS_PER_BLOCK = 2;
	}

	// in case we run out of memory with the automatically chosen configuration,
	// first back off with WARPS_PER_BLOCK, then reduce optimal_blocks.
	if (WARPS_PER_BLOCK == 3 && optimal_blocks * WARPS_PER_BLOCK > MAXWARPS[cgpu->driver_id]) {
		WARPS_PER_BLOCK = 2;
	}
	while (optimal_blocks > 0 && optimal_blocks * WARPS_PER_BLOCK > MAXWARPS[cgpu->driver_id]) {
		optimal_blocks--;
	}

	applog(LOG_INFO, "GPU #%d: using launch configuration %c%dx%d", cgpu->driver_id, kernel->get_identifier(), optimal_blocks, WARPS_PER_BLOCK);

	{
		if (MAXWARPS[cgpu->driver_id] != optimal_blocks * WARPS_PER_BLOCK)
		{
			MAXWARPS[cgpu->driver_id] = optimal_blocks * WARPS_PER_BLOCK;
			checkCudaErrors(cgpu->driver_id, cudaFree(d_V)); d_V = NULL;

			cudaGetLastError(); // clear the error state
			cudaMalloc((void **)&d_V, (size_t)SCRATCH * WU_PER_WARP * MAXWARPS[cgpu->driver_id] * sizeof(uint32_t));
			if (cudaGetLastError() == cudaSuccess) {
				for (int i = 0; i < MAXWARPS[cgpu->driver_id]; ++i) {
					h_V[cgpu->driver_id][i] = d_V + SCRATCH * WU_PER_WARP * i;
				}
				// update pointers to scratch buffer in constant memory after reallocation
				kernel->set_scratchbuf_constants(MAXWARPS[cgpu->driver_id], h_V[cgpu->driver_id]);
			}
			else
			{
				applog(LOG_ERR, "GPU #%d: Unable to allocate enough memory for launch config '%s'.", cgpu->driver_id, cgpu->device_config);
			}
		}
	}

	return optimal_blocks;
}

void cuda_scrypt_HtoD(_cudaState *cudaState, uint32_t *X, int stream)
{
	unsigned int GRID_BLOCKS = cudaState->context_blocks;
	unsigned int WARPS_PER_BLOCK = cudaState->context_wpb;
	unsigned int THREADS_PER_WU = cudaState->context_kernel->threads_per_wu();
	unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * 32;

	// copy host memory to device
	cudaMemcpyAsync(cudaState->context_idata[stream], X, mem_size, cudaMemcpyHostToDevice, cudaState->context_streams[stream]);
}

void cuda_scrypt_serialize(struct cgpu_info *cgpu, _cudaState *cudaState, int stream)
{
	// if the device can concurrently execute multiple kernels, then we must
	// wait for the serialization event recorded by the other stream
	if (cudaState->context_concurrent) {
		cudaStreamWaitEvent(cudaState->context_streams[stream], cudaState->context_serialize[(stream + 1) & 1], 0);
	}
}

void cuda_scrypt_done(_cudaState *cudaState, int stream)
{
	// record the serialization event in the current stream
	cudaEventRecord(cudaState->context_serialize[stream], cudaState->context_streams[stream]);
}

void cuda_scrypt_flush(_cudaState *cudaState, int stream)
{
	// flush the work queue (required for WDDM drivers)
	cudaStreamSynchronize(cudaState->context_streams[stream]);
}

void cuda_scrypt_core(struct cgpu_info *cgpu, _cudaState *cudaState, int stream, unsigned int N)
{
	unsigned int GRID_BLOCKS = cudaState->context_blocks;
	unsigned int WARPS_PER_BLOCK = cudaState->context_wpb;
	unsigned int THREADS_PER_WU = cudaState->context_kernel->threads_per_wu();
	unsigned int LOOKUP_GAP = cgpu->lookup_gap;

	// setup execution parameters
	dim3 grid(WU_PER_LAUNCH/WU_PER_BLOCK, 1, 1);
	dim3 threads(THREADS_PER_WU*WU_PER_BLOCK, 1, 1);

	cudaState->context_kernel->run_kernel(grid, threads, WARPS_PER_BLOCK, cudaState->cuda_id,
		cudaState->context_streams[stream], cudaState->context_idata[stream], cudaState->context_odata[stream],
		N, cgpu->batchsize, LOOKUP_GAP, opt_benchmark
	);
}

void cuda_scrypt_DtoH(_cudaState *cudaState, uint8_t *X, int stream)
{
	unsigned int GRID_BLOCKS = cudaState->context_blocks;
	unsigned int WARPS_PER_BLOCK = cudaState->context_wpb;
	unsigned int THREADS_PER_WU = cudaState->context_kernel->threads_per_wu();
	// copy result from device to host (asynchronously)
	checkCudaErrors(cudaState->cuda_id, cudaMemcpyAsync(X, cudaState->context_labels[stream], WU_PER_LAUNCH, cudaMemcpyDeviceToHost, cudaState->context_streams[stream]));
}

bool cuda_scrypt_sync(struct cgpu_info *cgpu, _cudaState *cudaState, int stream)
{
	cudaError_t err;
	uint32_t wait_us = 0;

	// this call was replaced by the loop below to workaround the high CPU usage issue
	//err = cudaStreamSynchronize(context_streams[stream][thr_id]);

	while((err = cudaStreamQuery(cudaState->context_streams[stream])) == cudaErrorNotReady) {
		usleep(50); wait_us+=50;
	}

	if (err != cudaSuccess) {
		if (!abort_flag) {
			applog(LOG_ERR, "GPU #%d: CUDA error `%s` while waiting the kernel.", cudaState->cuda_id, cudaGetErrorString(err));
		}
		return false;
	}

	return true;
}

uint32_t* cuda_transferbuffer(_cudaState *cudaState, int stream)
{
	return cudaState->context_X[stream];
}

uint8_t* cuda_hashbuffer(_cudaState *cudaState, int stream)
{
	return cudaState->context_L[stream];
}
