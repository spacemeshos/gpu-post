#include "api.h"
#include "api_internal.h"
#include <string.h>

extern void spacemesh_api_init();

static const uint8_t zeros[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

static void * memstr(const void *src, size_t length, const uint8_t *token, int token_length)
{
	for (const uint8_t *cp = (const uint8_t *)src; length >= token_length; cp++, length--) {
		if (0 == memcmp(cp, token, token_length)) {
			return (void*)cp;
		}
	}
	return 0;
}

static void print_hex32(const uint8_t *aSrc)
{
	printf("0x");
	for (int i = 0; i < 32; i++) {
		printf("%02x", aSrc[i]);
	}
}

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
	uint8_t *D,					// Target D for the POW computation. 256 bits.
	uint64_t *idx_solution,		// index of output where output < D if POW compute was on. MAX_UINT64 otherwise.
	uint64_t *hashes_computed,
	uint64_t *hashes_per_sec
)
{
	uint64_t data[16]; // align 16
	double t = 0.0;
	struct timeval tv_start;
	struct timeval tv_end;
	struct cgpu_info *cgpu = NULL;
	uint64_t hashes_computed_local;
	int status;

	spacemesh_api_init();

	if (R != 1 || P != 1) {
		return SPACEMESH_API_ERROR_INVALID_PARAMETER;
	}

	if (0 == (options & (SPACEMESH_API_COMPUTE_LEAFS | SPACEMESH_API_COMPUTE_POW))) {
		return SPACEMESH_API_ERROR_NO_COMPUTE_OPTIONS;
	}

	cgpu = spacemesh_api_get_gpu(provider_id);

	if (NULL == cgpu) {
		return SPACEMESH_API_ERROR_INVALID_PROVIDER_ID;
	}
	if (options & SPACEMESH_API_COMPUTE_LEAFS) {
		if (NULL == out) {
			return SPACEMESH_API_ERROR_INVALID_PARAMETER;
		}
	}
#ifdef _DEBUG
	memset(out, 0, (end_position - start_position + 1));
#endif

	if (!hashes_computed) {
		hashes_computed = &hashes_computed_local;
	}

	memcpy(data, id, 32);
	data[4] = 0;
	memcpy(data + 5, salt, 32);
	data[9] = 0;
	if (options & SPACEMESH_API_COMPUTE_POW) {
		if (NULL == D) {
			return SPACEMESH_API_ERROR_INVALID_PARAMETER;
		}
		memcpy(data + 10, D, 32);
	}

	status = cgpu->drv->scrypt_positions(cgpu, (uint8_t*)data, start_position, end_position, hash_len_bits, options, out, N, R, P, idx_solution, &tv_start, &tv_end, hashes_computed);

	t = 1e-6 * (tv_end.tv_usec - tv_start.tv_usec) + (tv_end.tv_sec - tv_start.tv_sec);
#if 0
	printf("--------------------------------\n");
	printf("id:   "); print_hex32(id); printf("\n");
	printf("salt: "); print_hex32(salt); printf("\n");
	if (NULL != D) {
		printf("D:    "); print_hex32(D); printf("\n");
        }
	printf("from %lu to %lu, length %u, options: %u\n", start_position, end_position, hash_len_bits, options);
	printf("[%d]: status %d, solution %llu, %.0f (%u positions in %.2fs)\n", provider_id, status, idx_solution ? *idx_solution : -1ull, *hashes_computed / t, (unsigned)*hashes_computed, t);
	printf("--------------------------------\n");
#endif
	if (hashes_per_sec) {
		*hashes_per_sec = (uint64_t)(*hashes_computed / t);
	}
#if 0
	if (out) {
		size_t labelsBufferSize = ((end_position - start_position + 1ull) * ((size_t)hash_len_bits) + 7ull) / 8ull;
		if (memstr(out, labelsBufferSize, zeros, 8)) {
			printf("--------------------------------\n");
			printf("[%d]: ZEROS result\n", provider_id);
			printf("--------------------------------\n");
		}
	}
#endif
	return status;
}

int64_t unit_test_hash(uint32_t provider_id, uint8_t *input, uint8_t *hashes)
{
	struct cgpu_info *cgpu = NULL;

	spacemesh_api_init();

	cgpu = spacemesh_api_get_gpu(provider_id);

	if (NULL == cgpu || NULL == cgpu->drv->unit_tests.hash) {
		return -1;
	}

	return cgpu->drv->unit_tests.hash(cgpu, input, hashes);
}

int64_t unit_test_bit_stream(uint32_t provider_id, uint8_t *hashes, uint64_t count, uint8_t *output, uint32_t hash_len_bits)
{
	struct cgpu_info *cgpu = NULL;

	spacemesh_api_init();

	cgpu = spacemesh_api_get_gpu(provider_id);

	if (NULL == cgpu || NULL == cgpu->drv->unit_tests.bit_stream) {
		return -1;
	}

	return cgpu->drv->unit_tests.bit_stream(cgpu, hashes, count, output, hash_len_bits);
}

// stop all GPU work and don�t fill the passed-in buffer with any more results.
int stop(uint32_t ms_timeout)
{
	return spacemesh_api_stop(ms_timeout);
}

