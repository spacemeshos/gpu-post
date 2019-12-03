/*
 * scrypt-jane by Andrew M, https://github.com/floodyberry/scrypt-jane
 *
 * Public Domain or MIT License, whichever is easier
 *
 * Adapted to ccminer by tpruvot@github (2015)
 */

#include "api_internal.h"

#include "scrypt-jane/scrypt-jane.h"
#include "scrypt-jane/scrypt-jane-portable.h"
#include "scrypt-jane/scrypt-jane-chacha.h"
#include "keccak.h"

#ifdef HAVE_CUDA

#include "salsa_kernel.h"
#include "cuda.h"

extern struct device_drv cuda_drv;

static void cuda_shutdown(struct cgpu_info *cgpu);

static int cuda_detect(struct cgpu_info *gpus, int *active)
{
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
		cgpu->cuda_mpcount = (short)props.multiProcessorCount;
		cgpu->gpu_max_alloc = (uint64_t)props.totalGlobalMem;
		cgpu->pci_bus_id = props.pciBusID;
		cgpu->pci_device_id = props.pciDeviceID;

		cgpu->name = NULL;
		cgpu->device_config = NULL;
		cgpu->backoff = is_windows() ? 12 : 2;
		cgpu->lookup_gap = 1;
		cgpu->batchsize = 1024;
#if 0
		if (device_name[dev_id]) {
			free(device_name[dev_id]);
			device_name[dev_id] = NULL;
		}

		device_name[dev_id] = strdup(props.name);
#endif
		*active += 1;
	}

	return 0;
}

static void reinit_cuda_device(struct cgpu_info *gpu)
{
}

static bool cuda_prepare(struct cgpu_info *cgpu, unsigned N, uint32_t r, uint32_t p, uint32_t hash_len_bits)
{
	if (N != cgpu->N || r != cgpu->r || p != cgpu->p) {
		applog(LOG_INFO, "Init GPU thread for GPU %i, platform GPU %i, pci [%d:%d]", cgpu->id, cgpu->driver_id, cgpu->pci_bus_id, cgpu->pci_device_id);
		if (cgpu->device_data) {
			cuda_shutdown(cgpu);
		}
		cgpu->device_data = initCuda(cgpu, N, r, p, hash_len_bits);
		if (!cgpu->device_data) {
			applog(LOG_ERR, "Failed to init GPU, disabling device %d", cgpu->id);
			cgpu->deven = DEV_DISABLED;
			cgpu->status = LIFE_NOSTART;
			return false;
		}
		cgpu->N = N;
		cgpu->r = r;
		cgpu->p = p;
		applog(LOG_INFO, "initCuda() finished.");
	}

	return true;
}

static bool cuda_init(struct cgpu_info *cgpu)
{
	cgpu->status = LIFE_WELL;
	return true;
}

#define bswap_32x4(x) ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) \
					 | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))

#define	PREIMAGE_SIZE	72

static int64_t cuda_scrypt_positions(struct cgpu_info *cgpu, uint8_t *pdata, uint64_t start_position, uint64_t end_position, uint8_t hash_len_bits, uint8_t *output, uint32_t N, uint32_t r, uint32_t p, struct timeval *tv_start, struct timeval *tv_end)
{
	if (cuda_prepare(cgpu, N, r, p, hash_len_bits))
	{
		_cudaState *cudaState = (_cudaState *)cgpu->device_data;

		gpulog(LOG_INFO, cgpu->driver_id, "Intensity set to %g, %u cuda threads", throughput2intensity(cgpu->thread_concurrency), cgpu->thread_concurrency);

		if (cgpu->thread_concurrency == 0) {
			return -1;
		}

		gettimeofday(tv_start, NULL);

		uint32_t *data[2] = { new uint32_t[(PREIMAGE_SIZE / 4) * cgpu->thread_concurrency], new uint32_t[(PREIMAGE_SIZE / 4) * cgpu->thread_concurrency] };
		uint8_t *hash[2] = { cuda_hashbuffer(cudaState, 0), cuda_hashbuffer(cudaState, 1) };

		uint64_t n = start_position;

		/* byte swap pdata into data[0]/[1] arrays */
		for (int k = 0; k < 2; ++k) {
			for (unsigned i = 0; i < cgpu->thread_concurrency; ++i) memcpy(&data[k][(PREIMAGE_SIZE / 4)*i], pdata, PREIMAGE_SIZE);
		}
		prepare_keccak512(cudaState, pdata, PREIMAGE_SIZE);

		uint64_t nonce[2];
		uint32_t* cuda_X[2] = { cuda_transferbuffer(cudaState, 0), cuda_transferbuffer(cudaState, 1) };

		int cur = 0, nxt = 1; // streams
		int iteration = 0;
		uint8_t *out = output;
		uint64_t chunkSize = (cgpu->thread_concurrency * hash_len_bits) / 8;
		uint64_t outLength = ((end_position - start_position + 1) * hash_len_bits + 7) / 8;

		do {
			nonce[nxt] = n;

			// all on gpu

			n += cgpu->thread_concurrency;
			if (opt_debug && (iteration % 64 == 0)) {
				applog(LOG_DEBUG, "GPU #%d: n=%x", cgpu->driver_id, n);
			}

			cuda_scrypt_serialize(cgpu, cudaState, nxt);
			if (1 == r && 1 == p) {
				pre_keccak512_1_1(cudaState, nxt, nonce[nxt], cgpu->thread_concurrency);
				cuda_scrypt_core(cgpu, cudaState, nxt, N, r, p);
			}
			else {
				pre_keccak512(cudaState, nxt, nonce[nxt], cgpu->thread_concurrency, r);
				cuda_scrypt_core(cgpu, cudaState, nxt, N, r, p);
			}
			if (!cuda_scrypt_sync(cgpu, cudaState, nxt)) {
				break;
			}

			post_keccak512(cudaState, nxt, nonce[nxt], cgpu->thread_concurrency, r, hash_len_bits);
			cuda_scrypt_done(cudaState, nxt);

			cuda_scrypt_DtoH(cudaState, hash[nxt], nxt, chunkSize);
			if (!cuda_scrypt_sync(cgpu, cudaState, nxt)) {
				break;
			}

			memcpy(out, hash[nxt], min(chunkSize, outLength));
			out += chunkSize;
			outLength -= chunkSize;

			cur = (cur + 1) & 1;
			nxt = (nxt + 1) & 1;
			++iteration;
		} while (n <= end_position && !abort_flag);

		delete[] data[0];
		delete[] data[1];

		gettimeofday(tv_end, NULL);

		return 0;
	}
	return -1;
}

static void cuda_shutdown(struct cgpu_info *cgpu)
{
	if (cgpu->device_data) {
		cudaSetDevice(cgpu->driver_id);
		cudaDeviceSynchronize();
		cudaDeviceReset(); // well, simple way to free ;)
		cgpu->deven = DEV_DISABLED;
		cgpu->status = LIFE_NOSTART;
		free(cgpu->device_data);
		cgpu->device_data = NULL;
	}
}

struct device_drv cuda_drv = {
	DRIVER_CUDA,
	"cuda",
	"GPU",
	cuda_detect,
	reinit_cuda_device,
	cuda_init,
	cuda_scrypt_positions,
	cuda_shutdown
};

#endif /* HAVE_CUDA */
