/*
 * scrypt-jane by Andrew M, https://github.com/floodyberry/scrypt-jane
 *
 * Public Domain or MIT License, whichever is easier
 *
 * Adapted to ccminer by tpruvot@github (2015)
 */

#include "api.h"
#include "api_internal.h"

#include "scrypt-jane/scrypt-jane.h"
#include "scrypt-jane/scrypt-jane-portable.h"
#include "scrypt-jane/scrypt-jane-chacha.h"
#include "keccak.h"

#ifdef HAVE_CUDA

#include "salsa_kernel.h"

#define	PREIMAGE_SIZE	128

extern struct device_drv cuda_drv;

static void cuda_shutdown(struct cgpu_info *cgpu);

static int cuda_detect(struct cgpu_info *gpus, int *active)
{
	int most_devices = 0;
	int version = 0, GPU_N = 0;
	cudaError_t err = cudaDriverGetVersion(&version);
	if (err != cudaSuccess) {
		applog(LOG_ERR, "Unable to query CUDA driver version! Is an nVidia driver installed?");
		return -1;
	}

	if (version < CUDART_VERSION) {
		applog(LOG_ERR, "Your system does not support CUDA %d.%d API!", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
		return -1;
	}

	err = cudaGetDeviceCount(&GPU_N);
	if (err != cudaSuccess) {
		applog(LOG_ERR, "Unable to query number of CUDA devices! Is an nVidia driver installed?");
		return -1;
	}

	for (int dev_id = 0; dev_id < GPU_N; dev_id++)
	{
		struct cgpu_info *cgpu = &gpus[*active];
		char vendorname[32] = { 0 };
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, dev_id);

		cgpu->drv = &cuda_drv;
		cgpu->driver_id = dev_id;

		cgpu->cuda_sm = (props.major * 100 + props.minor * 10);
		cgpu->gpu_core_count = (short)props.multiProcessorCount;
		cgpu->gpu_max_alloc = (uint64_t)props.totalGlobalMem;
		cgpu->pci_bus_id = props.pciBusID;
		cgpu->pci_device_id = props.pciDeviceID;

		memcpy(cgpu->name, props.name, min(sizeof(cgpu->name), sizeof(props.name)));
		cgpu->name[sizeof(cgpu->name) - 1] = 0;
		cgpu->device_config = NULL;
		cgpu->backoff = is_windows() ? 12 : 2;
		cgpu->lookup_gap = 1;
		cgpu->batchsize = 1024;

		*active += 1;
		most_devices++;
	}

	return most_devices;
}

static void reinit_cuda_device(struct cgpu_info *gpu)
{
}

static bool cuda_prepare(struct cgpu_info *cgpu, unsigned N, uint32_t r, uint32_t p, uint32_t hash_len_bits, bool throttled)
{
	if (N != cgpu->N || r != cgpu->r || p != cgpu->p || hash_len_bits != cgpu->hash_len_bits) {
		applog(LOG_INFO, "Init GPU thread for GPU %i, platform GPU %i, pci [%d:%d]", cgpu->id, cgpu->driver_id, cgpu->pci_bus_id, cgpu->pci_device_id);
		if (cgpu->device_data) {
			cuda_shutdown(cgpu);
		}
		cgpu->N = N;
		cgpu->r = r;
		cgpu->p = p;
		cgpu->hash_len_bits = hash_len_bits;
		cgpu->device_data = initCuda(cgpu, N, r, p, hash_len_bits, throttled);
		if (!cgpu->device_data) {
			applog(LOG_ERR, "Failed to init GPU, disabling device %d", cgpu->id);
			cgpu->deven = DEV_DISABLED;
			cgpu->status = LIFE_NOSTART;
			return false;
		}
		if (_cudaState *cudaState = (_cudaState *)cgpu->device_data) {
			cudaState->data[0] = new uint32_t[(PREIMAGE_SIZE / 4) * cgpu->thread_concurrency];
			cudaState->data[1] = new uint32_t[(PREIMAGE_SIZE / 4) * cgpu->thread_concurrency];
		}
		applog(LOG_INFO, "initCuda() finished.");
	}

	return true;
}

static bool cuda_init(struct cgpu_info *cgpu)
{
	cgpu->status = LIFE_WELL;
	return true;
}

static int cuda_scrypt_positions(
	struct cgpu_info *cgpu,
	uint8_t *preimage,
	uint64_t start_position,
	uint64_t end_position,
	uint32_t hash_len_bits,
	uint32_t options,
	uint8_t *output,
	uint32_t N,
	uint32_t,
	uint32_t,
	uint64_t *idx_solution,
	struct timeval *tv_start,
	struct timeval *tv_end,
	uint64_t *hashes_computed
)
{
	cgpu->busy = 1;
	if (hashes_computed) {
		*hashes_computed = 0;
	}

	if (cuda_prepare(cgpu, N, 1, 1, hash_len_bits, 0 != (options & SPACEMESH_API_THROTTLED_MODE)))
	{
		_cudaState *cudaState = (_cudaState *)cgpu->device_data;
		int status = SPACEMESH_API_ERROR_NONE;

		if (cgpu->thread_concurrency == 0) {
			cgpu->busy = 0;
			return SPACEMESH_API_ERROR;
		}

		gettimeofday(tv_start, NULL);

		uint8_t *hash[2] = { cuda_hashbuffer(cudaState, 0), cuda_hashbuffer(cudaState, 1) };

		uint64_t n = start_position;
		bool computeLeafs = 0 != (options & SPACEMESH_API_COMPUTE_LEAFS);
		bool computePow = 0 != (options & SPACEMESH_API_COMPUTE_POW);

		uint32_t pdata[32];
		memcpy(pdata, preimage, PREIMAGE_SIZE);
		for (int i = 20; i < 28; i++) {
			pdata[i] = bswap_32(pdata[i]);
		}

		/* byte swap pdata into data[0]/[1] arrays */
		for (int k = 0; k < 2; ++k) {
			for (unsigned i = 0; i < cgpu->thread_concurrency; ++i) memcpy(&cudaState->data[k][(PREIMAGE_SIZE / 4)*i], pdata, PREIMAGE_SIZE);
		}
		prepare_keccak512(cudaState, (uint8_t*)pdata, PREIMAGE_SIZE);

		uint64_t nonce[2];
		uint64_t* cuda_X[2] = { cuda_transferbuffer(cudaState, 0), cuda_transferbuffer(cudaState, 1) };

		int cur = 0, nxt = 1; // streams
		int iteration = 0;
		uint8_t *out = output;
		uint64_t chunkSize = (cgpu->thread_concurrency * hash_len_bits) / 8;
		uint64_t outLength = ((end_position - start_position + 1) * hash_len_bits + 7) / 8;
		uint64_t positions = 0;

		do {
			nonce[nxt] = n;
			cuda_X[nxt][0] = 0xffffffffffffffff;

			// all on gpu

			n += cgpu->thread_concurrency;
			positions += cgpu->thread_concurrency;
			if (g_spacemesh_api_opt_debug && (iteration % 64 == 0)) {
				applog(LOG_DEBUG, "GPU #%d: n=%x", cgpu->driver_id, n);
			}

			cuda_solutions_HtoD(cudaState, nxt);
			cuda_scrypt_serialize(cgpu, cudaState, nxt);

			pre_keccak512_1_1(cudaState, nxt, nonce[nxt], cgpu->thread_concurrency);
			cuda_scrypt_core(cudaState, nxt, N, 1, 1, cgpu->lookup_gap, cgpu->batchsize);

			if (!cuda_scrypt_sync(cgpu, cudaState, nxt)) {
				status = SPACEMESH_API_ERROR;
				break;
			}

			post_keccak512(cudaState, nxt, nonce[nxt], cgpu->thread_concurrency, hash_len_bits);
			cuda_scrypt_done(cudaState, nxt);

			if (computeLeafs) {
				cuda_scrypt_DtoH(cudaState, hash[nxt], nxt, chunkSize);
			}
			cuda_solutions_DtoH(cudaState, nxt);
			if (!cuda_scrypt_sync(cgpu, cudaState, nxt)) {
				status = SPACEMESH_API_ERROR;
				break;
			}

			if (computePow && (cuda_X[nxt][0] != 0xffffffffffffffff)) {
				if (idx_solution) {
					*idx_solution = cuda_X[nxt][0];
				}
				if (!computeLeafs) {
					status = SPACEMESH_API_POW_SOLUTION_FOUND;
					break;
				}
			}

			if (computeLeafs) {
				memcpy(out, hash[nxt], min(chunkSize, outLength));
				out += chunkSize;
				outLength -= chunkSize;
			}

			cur = (cur + 1) & 1;
			nxt = (nxt + 1) & 1;
			++iteration;
		} while (n <= end_position && !g_spacemesh_api_abort_flag);

		gettimeofday(tv_end, NULL);

		cgpu->busy = 0;
		size_t total = end_position - start_position + 1;
		positions = min(total, positions);

		if (hashes_computed) {
			*hashes_computed = positions;
		}

		if (computeLeafs) {
			int usedBits = (positions * hash_len_bits % 8);
			if (usedBits) {
				output[(positions * hash_len_bits) / 8] &= 0xff >> (8 - usedBits);
			}
		}

		if (status) {
			return status;
		}

		return (n <= end_position) ? SPACEMESH_API_ERROR_CANCELED : SPACEMESH_API_ERROR_NONE;
	}

	cgpu->busy = 0;

	return SPACEMESH_API_ERROR;
}

static int64_t cuda_hash(struct cgpu_info *cgpu, uint8_t *pdata, uint8_t *output)
{
	cgpu->busy = 1;

	if (cuda_prepare(cgpu, 512, 1, 1, 256, false))
	{
		_cudaState *cudaState = (_cudaState *)cgpu->device_data;

		if (cgpu->thread_concurrency == 0) {
			cgpu->busy = 0;
			return -1;
		}

		cgpu->thread_concurrency = 128;

		uint8_t *hash = cuda_hashbuffer(cudaState, 0);

		uint64_t n = 0;

		/* byte swap pdata into data[0]/[1] arrays */
		for (unsigned i = 0; i < cgpu->thread_concurrency; ++i) {
			memcpy(&cudaState->data[0][(PREIMAGE_SIZE / 4)*i], pdata, PREIMAGE_SIZE);
		}

		prepare_keccak512(cudaState, pdata, PREIMAGE_SIZE);

		int iteration = 0;
		uint8_t *out = output;
		uint64_t chunkSize = 32 * cgpu->thread_concurrency;
		uint64_t outLength = chunkSize;

		cuda_scrypt_serialize(cgpu, cudaState, 0);
		pre_keccak512_1_1(cudaState, 0, 0, cgpu->thread_concurrency);
		cuda_scrypt_core(cudaState, 0, 512, 1, 1, cgpu->lookup_gap, cgpu->batchsize);

		post_keccak512(cudaState, 0, 0, cgpu->thread_concurrency, 256);
		cuda_scrypt_done(cudaState, 0);

		cuda_scrypt_DtoH(cudaState, hash, 0, chunkSize);
		cuda_scrypt_sync(cgpu, cudaState, 0);

		memcpy(out, hash, min(chunkSize, outLength));

		cgpu->busy = 0;

		return 128;
	}

	cgpu->busy = 0;

	return SPACEMESH_API_ERROR;
}

static int64_t cuda_bit_stream(struct cgpu_info *cgpu, uint8_t *hashes, uint64_t count, uint8_t *output, uint32_t hash_len_bits)
{
	cgpu->busy = 1;

	if (cuda_prepare(cgpu, 512, 1, 1, 256, false))
	{
		_cudaState *cudaState = (_cudaState *)cgpu->device_data;

		if (cgpu->thread_concurrency == 0) {
			cgpu->busy = 0;
			return -1;
		}

		cgpu->thread_concurrency = 128;

		uint8_t *hash = cuda_hashbuffer(cudaState, 0);

		memcpy(hash, hashes, 32 * 128);
		cudaMemcpyAsync(cudaState->context_odata[0], hash, 32 * 128, cudaMemcpyHostToDevice, cudaState->context_streams[0]);

		uint8_t *out = output;
		uint64_t chunkSize = (cgpu->thread_concurrency * hash_len_bits) / 8;
		uint64_t outLength = chunkSize;

		post_labels_copy(cudaState, 0, cgpu->thread_concurrency, hash_len_bits);

		cuda_scrypt_DtoH(cudaState, hash, 0, chunkSize);
		cuda_scrypt_sync(cgpu, cudaState, 0);

		memcpy(out, hash, min(chunkSize, outLength));

		cgpu->busy = 0;

		return chunkSize;
	}

	cgpu->busy = 0;

	return SPACEMESH_API_ERROR;
}

static void cuda_shutdown(struct cgpu_info *cgpu)
{
	if (cgpu->device_data) {
		cudaSetDevice(cgpu->driver_id);
		cudaDeviceSynchronize();
		cudaDeviceReset(); // well, simple way to free ;)
		cgpu->deven = DEV_DISABLED;
		cgpu->status = LIFE_NOSTART;
		if (_cudaState *cudaState = (_cudaState *)cgpu->device_data) {
			delete[] cudaState->data[0];
			delete[] cudaState->data[1];
		}
		free(cgpu->device_data);
		cgpu->device_data = NULL;
	}
}

struct device_drv cuda_drv = {
	SPACEMESH_API_CUDA,
	"cuda",
	"GPU",
	cuda_detect,
	reinit_cuda_device,
	cuda_init,
	cuda_scrypt_positions,
	{ cuda_hash, cuda_bit_stream },
	cuda_shutdown
};

#endif /* HAVE_CUDA */
