#ifndef __MINER_H__
#define __MINER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <inttypes.h>
#ifdef WIN32
#define	pthread_mutex_lock(cs) EnterCriticalSection(cs)
#define	pthread_mutex_unlock(cs) LeaveCriticalSection(cs)
#else
#include <pthread.h>
#endif
#include <sys/time.h>

#ifdef _MSC_VER
#include <malloc.h>
#define alloca _alloca
#else
#include <alloca.h>
#endif

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#ifdef WIN32
#include "compat.h"
#else
#include "compat/compat.h"
#include <unistd.h>
#include <byteswap.h>
#endif

#ifdef __INTELLISENSE__
/* should be in stdint.h but... */
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef __int16 int8_t;
typedef unsigned __int16 uint8_t;

typedef unsigned __int32 time_t;
typedef char *  va_list;
#endif

enum {
	LOG_ERR,
	LOG_WARNING,
	LOG_NOTICE,
	LOG_INFO,
	LOG_DEBUG,
	/* custom notices */
	LOG_BLUE = 0x10,
	LOG_RAW  = 0x99
};

typedef unsigned char uchar;

#undef unlikely
#undef likely
#if defined(__GNUC__) && (__GNUC__ > 2) && defined(__OPTIMIZE__)
#define unlikely(expr) (__builtin_expect(!!(expr), 0))
#define likely(expr) (__builtin_expect(!!(expr), 1))
#else
#define unlikely(expr) (expr)
#define likely(expr) (expr)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

#ifndef max
# define max(a, b)  ((a) > (b) ? (a) : (b))
#endif
#ifndef min
# define min(a, b)  ((a) < (b) ? (a) : (b))
#endif

#ifndef UINT32_MAX
/* for gcc 4.4 */
#define UINT32_MAX UINT_MAX
#endif

static inline bool is_windows(void) {
#ifdef WIN32
        return 1;
#else
        return 0;
#endif
}

static inline bool is_x64(void) {
#if defined(__x86_64__) || defined(_WIN64) || defined(__aarch64__)
	return 1;
#elif defined(__amd64__) || defined(__amd64) || defined(_M_X64) || defined(_M_IA64)
	return 1;
#else
	return 0;
#endif
}

#if ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
#define WANT_BUILTIN_BSWAP
#else
#define bswap_32(x) ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) \
                   | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#define bswap_64(x) (((uint64_t) bswap_32((uint32_t)((x) & 0xffffffffu)) << 32) \
                   | (uint64_t) bswap_32((uint32_t)((x) >> 32)))
#endif

static inline uint32_t swab32(uint32_t v)
{
#ifdef WANT_BUILTIN_BSWAP
	return __builtin_bswap32(v);
#else
	return bswap_32(v);
#endif
}

static inline uint64_t swab64(uint64_t v)
{
#ifdef WANT_BUILTIN_BSWAP
	return __builtin_bswap64(v);
#else
	return bswap_64(v);
#endif
}

static inline void swab256(void *dest_p, const void *src_p)
{
	uint32_t *dest = (uint32_t *) dest_p;
	const uint32_t *src = (const uint32_t *) src_p;

	dest[0] = swab32(src[7]);
	dest[1] = swab32(src[6]);
	dest[2] = swab32(src[5]);
	dest[3] = swab32(src[4]);
	dest[4] = swab32(src[3]);
	dest[5] = swab32(src[2]);
	dest[6] = swab32(src[1]);
	dest[7] = swab32(src[0]);
}

static inline uint32_t be32dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint32_t)(p[3]) + ((uint32_t)(p[2]) << 8) +
	    ((uint32_t)(p[1]) << 16) + ((uint32_t)(p[0]) << 24));
}

static inline uint32_t le32dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint32_t)(p[0]) + ((uint32_t)(p[1]) << 8) +
	    ((uint32_t)(p[2]) << 16) + ((uint32_t)(p[3]) << 24));
}

static inline void be32enc(void *pp, uint32_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[3] = x & 0xff;
	p[2] = (x >> 8) & 0xff;
	p[1] = (x >> 16) & 0xff;
	p[0] = (x >> 24) & 0xff;
}

static inline void le32enc(void *pp, uint32_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
	p[2] = (x >> 16) & 0xff;
	p[3] = (x >> 24) & 0xff;
}

static inline uint16_t be16dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint16_t)(p[1]) + ((uint16_t)(p[0]) << 8));
}

static inline void be16enc(void *pp, uint16_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[1] = x & 0xff;
	p[0] = (x >> 8) & 0xff;
}

static inline uint16_t le16dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint16_t)(p[0]) + ((uint16_t)(p[1]) << 8));
}

static inline void le16enc(void *pp, uint16_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
}

extern void applog(int prio, const char *fmt, ...);
extern void gpulog(int prio, int thr_id, const char *fmt, ...);
extern void _quit(int status);

#define MAX_GPUDEVICES 16

#define MIN_INTENSITY -10
#define MAX_INTENSITY 20
#define MIN_XINTENSITY 1
#define MAX_XINTENSITY 9999
#define MIN_RAWINTENSITY 1
#define MAX_RAWINTENSITY 2147483647

#define	SCRYPT_FIXED_DATA_LENGTH	64
#define	SCRYPT_POS_DATA_LENGTH		8
#define	SCRYPT_DATA_LENGTH			(SCRYPT_FIXED_DATA_LENGTH + SCRYPT_POS_DATA_LENGTH)

#define LOGBUFSIZ 256

#define quit(status, fmt, ...) do { \
	if (fmt) { \
		char tmp42[LOGBUFSIZ]; \
		snprintf(tmp42, sizeof(tmp42), fmt, ##__VA_ARGS__); \
		applog(LOG_ERR, tmp42); \
	} \
	_quit(status); \
} while (0)

enum dev_enable {
	DEV_ENABLED,
	DEV_DISABLED,
	DEV_RECOVER,
};

enum alive {
	LIFE_WELL,
	LIFE_SICK,
	LIFE_DEAD,
	LIFE_NOSTART,
	LIFE_INIT,
};

struct cgpu_info;

struct device_drv {
	uint32_t type;

	const char *name;
	const char *class_name;

	// DRV-global functions
	int(*drv_detect)(struct cgpu_info *, int *);

	// Device-specific functions
	void(*reinit_device)(struct cgpu_info *);

	bool(*init)(struct cgpu_info *);

	int64_t(*scrypt_positions)(struct cgpu_info *, uint8_t *pdata, uint64_t start_pos, uint64_t end_position, uint8_t hash_len_bits, uint32_t options, uint8_t *out, uint32_t N, uint32_t r, uint32_t p, struct timeval *tv_start, struct timeval *tv_end); // (thr, pdata, start_pos, end_position, out, N, r, p, tv_start, tv_end)

	void(*shutdown)(struct cgpu_info *);

	// Does it need to be free()d?
	bool copy;
};

struct cgpu_info {
	struct device_drv *drv;

	uint32_t pci_bus_id;
	uint32_t pci_device_id;
	int id; // Global GPU index
	int driver_id; // GPU index by driver

	char *name;
	void *device_data;
	char *device_config;
	enum dev_enable deven;
	enum alive status;

	uint32_t gpu_core_count;
	uint64_t gpu_max_alloc;

	uint32_t N;
	uint32_t r;
	uint32_t p;
	int opt_lg, lookup_gap;
	uint32_t block_count;
	uint32_t thread_concurrency;
	size_t buffer_size;

#ifdef HAVE_CUDA
	long cuda_sm;
	int batchsize;
	int backoff;
#endif

#ifdef HAVE_OPENCL
	unsigned int platform;
#endif

	volatile bool available;
	volatile bool busy;
	bool shutdown;
};

extern volatile bool abort_flag;
extern bool opt_debug;
#ifdef WIN32
extern CRITICAL_SECTION applog_lock;
#else
extern pthread_mutex_t applog_lock;
#endif
extern bool have_cuda;
extern bool have_opencl;

#define EXIT_CODE_OK            0
#define EXIT_CODE_USAGE         1
#define EXIT_CODE_POOL_TIMEOUT  2
#define EXIT_CODE_SW_INIT_ERROR 3
#define EXIT_CODE_CUDA_NODEVICE 4
#define EXIT_CODE_CUDA_ERROR    5
#define EXIT_CODE_TIME_LIMIT    0
#define EXIT_CODE_KILLED        7

int spacemesh_api_stop(uint32_t ms_timeout);
int spacemesh_api_stats();
struct cgpu_info * spacemesh_api_get_available_gpu();
struct cgpu_info * spacemesh_api_get_available_gpu_by_type(enum drv_driver type);
struct cgpu_info * spacemesh_api_get_gpu(int id);
int spacemesh_api_get_gpu_count(int type, int only_available);
int spacemesh_api_lock_gpu(int type);
void spacemesh_api_unlock_gpu(int cookie);
void spacemesh_api_release_gpu(struct cgpu_info *cgpu);

#ifdef __cplusplus
}
#endif

#endif /* __MINER_H__ */
