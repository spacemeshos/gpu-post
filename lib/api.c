#include "api.h"
#include "api_internal.h"

extern void spacemesh_api_init();

int scryptPositions(
    const uint8_t *id, // 32 bytes
    uint64_t start_position,  // e.g. 0 
    uint64_t end_position, // e.g. 49,999
    uint8_t hash_len_bits, // (1...8) for each hash output, the number of prefix bits (not bytes) to copy into the buffer
    const uint8_t *salt,  // 32 bytes
    uint8_t options,  // throttle etc…
    uint8_t *out, // memory buffer large enough to include hash_len_bits * number of requested hashes
    uint32_t N,
    uint32_t R,
    uint32_t P
)
{
	uint32_t data[20]; // align 16
	double t = 0.0;
	struct timeval tv_start;
	struct timeval tv_end;
	struct cgpu_info *cgpu = NULL;
	memcpy(data, id, 32);
	data[8] = 0;
	data[9] = 0;
	memcpy(data + 10, salt, 32);
	data[18] = 0;
	data[19] = 0;
	spacemesh_api_init();
	if (options & SPACEMESH_API_CUDA) {
		cgpu = get_available_gpu_by_type(DRIVER_CUDA);
	}
	if (!cgpu && (options & SPACEMESH_API_OPENCL)) {
		cgpu = get_available_gpu_by_type(DRIVER_OPENCL);
	}
	if (!cgpu && (options & SPACEMESH_API_CPU)) {
		cgpu = get_available_gpu_by_type(DRIVER_CPU);
	}
	if (!cgpu && (0 == (options & (SPACEMESH_API_CUDA | SPACEMESH_API_OPENCL | SPACEMESH_API_CPU)))) {
		cgpu = get_available_gpu();
	}
	if (NULL == cgpu) {
		return -1;
	}
#ifdef _DEBUG
	memset(out, 0, (end_position - start_position + 1));
#endif
	cgpu->drv->scrypt_positions(cgpu, (uint8_t*)data, start_position, end_position, out, N, &tv_start, &tv_end);
	release_gpu(cgpu);
	t = 1e-6 * (tv_end.tv_usec - tv_start.tv_usec) + (tv_end.tv_sec - tv_start.tv_sec);
	printf("Performance: %.0f (%u positions in %.2fs)\n", (end_position - start_position + 1) / t, (end_position - start_position + 1), t);
    return 0;
}

int scryptMany()
{
	spacemesh_api_init();
	return 0;
}

// return to the client the system GPU capabilities. E.g. OPENCL, CUDA/NVIDIA or NONE
int stats()
{
    return 0;
}

// stop all GPU work and don’t fill the passed-in buffer with any more results.
int stop()
{
    return 0;
}

