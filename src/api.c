#include "api.h"
#include "api_internal.h"
#include <string.h>

extern void spacemesh_api_init();

int scryptPositions(
	uint32_t provider_id, // POST compute provider ID
	const uint8_t *id, // 32 bytes
    uint64_t start_position,  // e.g. 0
    uint64_t end_position, // e.g. 49,999
	uint32_t hash_len_bits, // (1...256) for each hash output, the number of prefix bits (not bytes) to copy into the buffer
    const uint8_t *salt,  // 32 bytes
    uint32_t options,  // throttle etc.
    uint8_t *out, // memory buffer large enough to include hash_len_bits * number of requested hashes
    uint32_t N,
    uint32_t R,
    uint32_t P,
	uint64_t *hashes_computed,
	uint64_t *hashes_per_sec
)
{
	uint32_t data[20]; // align 16
	double t = 0.0;
	struct timeval tv_start;
	struct timeval tv_end;
	struct cgpu_info *cgpu = NULL;
	uint64_t hashes_computed_local;

	memcpy(data, id, 32);
	data[8] = 0;
	data[9] = 0;
	memcpy(data + 10, salt, 32);
	data[18] = 0;
	data[19] = 0;

	spacemesh_api_init();

	cgpu = spacemesh_api_get_gpu(provider_id);

	if (NULL == cgpu) {
		return -1;
	}
#ifdef _DEBUG
	memset(out, 0, (end_position - start_position + 1));
#endif

	if (!hashes_computed) {
		hashes_computed = &hashes_computed_local;
	}

	cgpu->drv->scrypt_positions(cgpu, (uint8_t*)data, start_position, end_position, hash_len_bits, options, out, N, R, P, &tv_start, &tv_end, hashes_computed);

	t = 1e-6 * (tv_end.tv_usec - tv_start.tv_usec) + (tv_end.tv_sec - tv_start.tv_sec);

	printf("--------------------------------\n");
	printf("Performance: %.0f (%u positions in %.2fs)\n", *hashes_computed / t, (unsigned)*hashes_computed, t);
	printf("--------------------------------\n");

	if (hashes_per_sec) {
		*hashes_per_sec = *hashes_computed / t;
	}

	return 0;
}

// stop all GPU work and don�t fill the passed-in buffer with any more results.
int stop(uint32_t ms_timeout)
{
	return spacemesh_api_stop(ms_timeout);
}

