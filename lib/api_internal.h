#ifndef __MINER_H__
#define __MINER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <spacemesh-config.h>

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
#undef HAVE_ALLOCA_H
#undef HAVE_SYSLOG_H
#endif

#ifdef STDC_HEADERS
# include <stdlib.h>
# include <stddef.h>
#else
# ifdef HAVE_STDLIB_H
#  include <stdlib.h>
# endif
#endif

#ifdef HAVE_ALLOCA_H
# include <alloca.h>
#elif !defined alloca
# ifdef __GNUC__
#  define alloca __builtin_alloca
# elif defined _AIX
#  define alloca __alloca
# elif defined _MSC_VER
#  include <malloc.h>
#  define alloca _alloca
# elif !defined HAVE_ALLOCA
void *alloca (size_t);
# endif
#endif

#include "compat.h"

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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
# undef _ALIGN
# define _ALIGN(x) __align__(x)
#endif

#ifdef HAVE_OPENCL
#include "opencl/ocl.h"
#endif /* HAVE_OPENCL */

#ifdef HAVE_SYSLOG_H
#include <syslog.h>
#define LOG_BLUE 0x10
#define LOG_RAW  0x99
#else
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
#endif

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

#ifdef HAVE_SYS_ENDIAN_H
#include <sys/endian.h>
#endif

#if !HAVE_DECL_BE32DEC
static inline uint32_t be32dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint32_t)(p[3]) + ((uint32_t)(p[2]) << 8) +
	    ((uint32_t)(p[1]) << 16) + ((uint32_t)(p[0]) << 24));
}
#endif

#if !HAVE_DECL_LE32DEC
static inline uint32_t le32dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint32_t)(p[0]) + ((uint32_t)(p[1]) << 8) +
	    ((uint32_t)(p[2]) << 16) + ((uint32_t)(p[3]) << 24));
}
#endif

#if !HAVE_DECL_BE32ENC
static inline void be32enc(void *pp, uint32_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[3] = x & 0xff;
	p[2] = (x >> 8) & 0xff;
	p[1] = (x >> 16) & 0xff;
	p[0] = (x >> 24) & 0xff;
}
#endif

#if !HAVE_DECL_LE32ENC
static inline void le32enc(void *pp, uint32_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
	p[2] = (x >> 16) & 0xff;
	p[3] = (x >> 24) & 0xff;
}
#endif

#if !HAVE_DECL_BE16DEC
static inline uint16_t be16dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint16_t)(p[1]) + ((uint16_t)(p[0]) << 8));
}
#endif

#if !HAVE_DECL_BE16ENC
static inline void be16enc(void *pp, uint16_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[1] = x & 0xff;
	p[0] = (x >> 8) & 0xff;
}
#endif

#if !HAVE_DECL_LE16DEC
static inline uint16_t le16dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint16_t)(p[0]) + ((uint16_t)(p[1]) << 8));
}
#endif

#if !HAVE_DECL_LE16ENC
static inline void le16enc(void *pp, uint16_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
}
#endif

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

enum drv_driver {
	DRIVER_CPU = 0,
	DRIVER_OPENCL,
	DRIVER_CUDA,
	DRIVER_MAX
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
	enum drv_driver drv_id;

	const char *dname;
	const char *name;

	// DRV-global functions
	int(*drv_detect)(struct cgpu_info *, int *);

	// Device-specific functions
	void(*reinit_device)(struct cgpu_info *);

	bool(*init)(struct cgpu_info *);

	int64_t(*scrypt_positions)(struct cgpu_info *, uint8_t *pdata, uint64_t start_pos, uint64_t end_position, uint8_t *out, uint32_t N, struct timeval *tv_start, struct timeval *tv_end); // (thr, pdata, start_pos, end_position, out, N, tv_start, tv_end)

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

	uint64_t gpu_max_alloc;

	unsigned N;
	int opt_lg, lookup_gap;
	size_t thread_concurrency;
	size_t buffer_size;

#ifdef HAVE_CUDA
	int cuda_mpcount;
	long cuda_sm;
	char gpu_sn[64];
	char gpu_desc[64];
	int batchsize;
	int backoff;
#endif

#ifdef HAVE_OPENCL
	unsigned int platform;
	const char *kname;
#endif

	bool available;
	bool shutdown;
};

extern bool opt_benchmark;
extern volatile bool abort_flag;
extern bool opt_debug;
#ifdef WIN32
extern CRITICAL_SECTION applog_lock;
#else
extern pthread_mutex_t applog_lock;
#endif
extern bool have_cuda;
extern bool have_opencl;

extern struct cgpu_info g_gpus[MAX_GPUDEVICES];

#define EXIT_CODE_OK            0
#define EXIT_CODE_USAGE         1
#define EXIT_CODE_POOL_TIMEOUT  2
#define EXIT_CODE_SW_INIT_ERROR 3
#define EXIT_CODE_CUDA_NODEVICE 4
#define EXIT_CODE_CUDA_ERROR    5
#define EXIT_CODE_TIME_LIMIT    0
#define EXIT_CODE_KILLED        7

struct cgpu_info * get_available_gpu();
struct cgpu_info * get_available_gpu_by_type(enum drv_driver type);
void release_gpu(struct cgpu_info *cgpu);

void proper_exit(int reason);

#ifdef __cplusplus
}
#endif

#endif /* __MINER_H__ */
