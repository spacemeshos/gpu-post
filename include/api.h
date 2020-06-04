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

#define	SPACEMESH_API_ERROR_NONE		0
#define	SPACEMESH_API_ERROR				-1
#define	SPACEMESH_API_ERROR_TIMEOUT		-2
#define	SPACEMESH_API_ERROR_ALREADY		-3

#define	SPACEMESH_API_CPU				0x00000001
#define	SPACEMESH_API_CUDA				0x00000002
#define	SPACEMESH_API_OPENCL			0x00000004
#define	SPACEMESH_API_VULKAN			0x00000008
#define	SPACEMESH_API_GPU				(SPACEMESH_API_CUDA | SPACEMESH_API_OPENCL | SPACEMESH_API_VULKAN)
#define	SPACEMESH_API_ALL				(SPACEMESH_API_CPU | SPACEMESH_API_GPU)

#define	SPACEMESH_API_THROTTLED_MODE	0x00008000

#define SPACEMESH_API_USE_LOCKED_DEVICE	0x00001000

SPACEMESHAPI int scryptPositions(
    const uint8_t *id,			// 32 bytes
    uint64_t start_position,	// e.g. 0 
    uint64_t end_position,		// e.g. 49,999
    uint8_t hash_len_bits,		// (1...8) for each hash output, the number of prefix bits (not bytes) to copy into the buffer
    const uint8_t *salt,		// 32 bytes
    uint32_t options,			// throttle etc.
    uint8_t *out,				// memory buffer large enough to include hash_len_bits * number of requested hashes
    uint32_t N,					// scrypt N
    uint32_t R,					// scrypt r
    uint32_t P					// scrypt p
);

// return to the client the system GPU capabilities. E.g. OPENCL, CUDA/NVIDIA or NONE
SPACEMESHAPI int stats();

// stop all GPU work and don’t fill the passed-in buffer with any more results.
SPACEMESHAPI int stop(
	uint32_t ms_timeout			// timeout in milliseconds
);

// return count of GPUs
SPACEMESHAPI int spacemesh_api_get_gpu_count(
	int type,					// GPU type SPACEMESH_API_CUDA or SPACEMESH_API_OPENCL
	int only_available			// return count of available GPUs only
);

// lock GPU for persistent exclusive use. returned cookie used as options in scryptPositions call
SPACEMESHAPI int spacemesh_api_lock_gpu(
	int type					// GPU type SPACEMESH_API_CUDA or SPACEMESH_API_OPENCL
);

// unlock GPU, locked by previous spacemesh_api_lock_gpu call
SPACEMESHAPI void spacemesh_api_unlock_gpu(
	int cookie					// cookie, returned by previous spacemesh_api_lock_gpu call
);

#ifdef __cplusplus
}
#endif	// #ifdef __cplusplus

#endif	// #ifndef __SPACEMESH_API_H__

