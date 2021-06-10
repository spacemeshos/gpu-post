#ifndef __SPACEMESH_API_H__
#define __SPACEMESH_API_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif	// #ifdef __cplusplus

#ifdef WIN32
# ifdef SPACEMESHAPI_BUILD
#  define SPACEMESHAPI __declspec( dllexport )
# else
#  define SPACEMESHAPI __declspec( dllimport )
# endif
#else
# define SPACEMESHAPI
#endif

#define	SPACEMESH_API_POW_SOLUTION_FOUND		1
#define	SPACEMESH_API_ERROR_NONE				0
#define	SPACEMESH_API_ERROR						-1
#define	SPACEMESH_API_ERROR_TIMEOUT				-2
#define	SPACEMESH_API_ERROR_ALREADY				-3
#define	SPACEMESH_API_ERROR_CANCELED			-4
#define	SPACEMESH_API_ERROR_NO_COMPOTE_OPTIONS	-5
#define	SPACEMESH_API_ERROR_INVALID_PARAMETER	-6
#define	SPACEMESH_API_ERROR_INVALID_PROVIDER_ID	-7

#define	SPACEMESH_API_THROTTLED_MODE	0x00008000

enum {
	SPACEMESH_API_COMPUTE_LEAFS = 1 << 0,
	SPACEMESH_API_COMPUTE_POW = 1 << 1,
};

// Compute API class
typedef enum _ComputeApiClass {
	COMPUTE_API_CLASS_UNSPECIFIED = 0,
	COMPUTE_API_CLASS_CPU = 1, // useful for testing on systems without a cuda or vulkan GPU
	COMPUTE_API_CLASS_CUDA = 2,
	COMPUTE_API_CLASS_VULKAN = 3
} ComputeApiClass;

typedef struct _PostComputeProvider {
	uint32_t id; // 0, 1, 2...
	char model[256]; // e.g. Nvidia GTX 2700
	ComputeApiClass compute_api; // A provided compute api
} PostComputeProvider;

SPACEMESHAPI int scryptPositions(
	uint32_t provider_id,		// POST compute provider ID
	const uint8_t *id,			// 32 bytes
    uint64_t start_position,	// e.g. 0
    uint64_t end_position,		// e.g. 49,999
    uint32_t hash_len_bits,		// (1...256) for each hash output, the number of prefix bits (not bytes) to copy into the buffer
    const uint8_t *salt,		// 32 bytes
    uint32_t options,			// compute leafs and or compute pow
    uint8_t *out,				// memory buffer large enough to include hash_len_bits * number of requested hashes
    uint32_t N,					// scrypt N
    uint32_t R,					// scrypt r
    uint32_t P,					// scrypt p
	uint8_t *D,					// Target D for the POW computation. 32 bytes
	uint64_t *idx_solution,		// index of output where output < D if POW compute was on. MAX_UINT64 otherwise.

	uint64_t *hashes_computed,	//
	uint64_t *hashes_per_sec	//
	);

// stop all GPU work and don't fill the passed-in buffer with any more results.
SPACEMESHAPI int stop(
	uint32_t ms_timeout			// timeout in milliseconds
);

// return non-zero if stop in progress
SPACEMESHAPI int spacemesh_api_stop_inprogress();

// return POST compute providers info
SPACEMESHAPI int spacemesh_api_get_providers(
	PostComputeProvider *providers, // out providers info buffer, if NULL - return count of available providers
	int max_providers			    // buffer size
);

// enable/disable log output
SPACEMESHAPI void spacemesh_api_logging(
	int enable
);

// library shutdown
SPACEMESHAPI void spacemesh_api_shutdown(void);

SPACEMESHAPI int64_t unit_test_hash(uint32_t provider_id, uint8_t *input, uint8_t *hashes);
SPACEMESHAPI int64_t unit_test_bit_stream(uint32_t provider_id, uint8_t *hashes, uint64_t count, uint8_t *output, uint32_t hash_len_bits);

#ifdef __cplusplus
}
#endif	// #ifdef __cplusplus

#endif	// #ifndef __SPACEMESH_API_H__
