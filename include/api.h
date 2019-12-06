#ifndef __SPACEMESH_API_H__
#define __SPACEMESH_API_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif	// #ifdef __cplusplus

#define	SPACEMESH_API_ERROR_NONE		0
#define	SPACEMESH_API_ERROR				-1
#define	SPACEMESH_API_ERROR_TIMEOUT		-2
#define	SPACEMESH_API_ERROR_ALREADY		-3

#define	SPACEMESH_API_CPU				0x00000001
#define	SPACEMESH_API_CUDA				0x00000002
#define	SPACEMESH_API_OPENCL			0x00000004
#define	SPACEMESH_API_GPU				(SPACEMESH_API_CUDA | SPACEMESH_API_OPENCL)

#define	SPACEMESH_API_THROTTLED_MODE	0x00000008

#define SPACEMESH_API_USE_LOCKED_DEVICE	0x00001000

int scryptPositions(
    const uint8_t *id, // 32 bytes
    uint64_t start_position,  // e.g. 0 
    uint64_t end_position, // e.g. 49,999
    uint8_t hash_len_bits, // (1...8) for each hash output, the number of prefix bits (not bytes) to copy into the buffer
    const uint8_t *salt,  // 32 bytes
    uint32_t options,  // throttle etc.
    uint8_t *out, // memory buffer large enough to include hash_len_bits * number of requested hashes
    uint32_t N,
    uint32_t R,
    uint32_t P
);

int scryptMany();

int stats(); // return to the client the system GPU capabilities. E.g. OPENCL, CUDA/NVIDIA or NONE

int stop(uint32_t ms_timeout); // stop all GPU work and don’t fill the passed-in buffer with any more results.

int spacemesh_api_get_gpu_count(int type, int only_available);

int spacemesh_api_lock_gpu(int type);

void spacemesh_api_unlock_gpu(int cookie);

#ifdef __cplusplus
}
#endif	// #ifdef __cplusplus

#endif	// #ifndef __SPACEMESH_API_H__

