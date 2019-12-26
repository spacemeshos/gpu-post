#include "api.h"
#include "api_internal.h"

#ifdef HAVE_VULKAN
#include "vulkan/driver-vulkan.h"
#endif

#ifdef HAVE_CUDA
#include "cuda/driver-cuda.h"
#endif

#ifdef HAVE_OPENCL
#include "opencl/driver-opencl.h"
#endif

#include "scrypt-jane/scrypt-jane.h"

volatile bool abort_flag = false;
bool opt_debug = false;
#ifdef WIN32
CRITICAL_SECTION applog_lock;
CRITICAL_SECTION gpus_lock;
#else
pthread_mutex_t applog_lock;
pthread_mutex_t gpus_lock;
#endif

static int s_total_devices = 0;
bool have_cuda = false;
bool have_opencl = false;
bool have_vulkan = false;

static struct cgpu_info s_gpus[MAX_GPUDEVICES]; /* Maximum number apparently possible */
static struct cgpu_info s_cpu;

static volatile int api_inited = 0;

extern "C" void spacemesh_api_init()
{
	if (0 == api_inited) {
		api_inited = 1;
#ifdef WIN32
		InitializeCriticalSection(&applog_lock);
		InitializeCriticalSection(&gpus_lock);
#else
		pthread_mutex_init(&applog_lock, NULL);
		pthread_mutex_init(&gpus_lock, NULL);
#endif

		memset(s_gpus, 0, sizeof(s_gpus));

#ifdef HAVE_VULKAN
		vulkan_drv.drv_detect(s_gpus, &s_total_devices);
#endif

#ifdef HAVE_CUDA
		cuda_drv.drv_detect(s_gpus, &s_total_devices);
#endif

#ifdef HAVE_OPENCL
		opencl_drv.drv_detect(s_gpus, &s_total_devices);
#endif

		if (0 == cpu_drv.drv_detect(&s_cpu, NULL)) {
			s_cpu.drv->init(&s_cpu);
			s_cpu.available = true;
		}

		for (int i = 0; i < s_total_devices; ++i) {
			struct cgpu_info *cgpu = &s_gpus[i];
			cgpu->status = LIFE_INIT;

			if (!cgpu->drv->init(cgpu)) {
				continue;
			}

			cgpu->available = true;
		}
	}
}

struct cgpu_info * spacemesh_api_get_available_gpu()
{
	struct cgpu_info *cgpu = NULL;
	pthread_mutex_lock(&gpus_lock);
	for (int i = 0; i < s_total_devices; ++i) {
		if (s_gpus[i].available && (0 != (s_gpus[i].drv->type & SPACEMESH_API_GPU))) {
			cgpu = &s_gpus[i];
			cgpu->available = false;
			break;
		}
	}
	pthread_mutex_unlock(&gpus_lock);
	return cgpu;
}

struct cgpu_info * spacemesh_api_get_gpu(int id)
{
	if (id >= 0 && id < s_total_devices) {
		return &s_gpus[id];
	}
	return NULL;
}

struct cgpu_info * spacemesh_api_get_available_gpu_by_type(int type)
{
	struct cgpu_info *cgpu = NULL;
	pthread_mutex_lock(&gpus_lock);
	for (int i = 0; i < s_total_devices; ++i) {
		if (s_gpus[i].available && (type == s_gpus[i].drv->type)) {
			cgpu = &s_gpus[i];
			cgpu->available = false;
			break;
		}
	}
	pthread_mutex_unlock(&gpus_lock);
	if (NULL == cgpu && (0 != (type & SPACEMESH_API_CPU))) {
		cgpu = &s_cpu;
	}
	return cgpu;
}

void spacemesh_api_release_gpu(struct cgpu_info *cgpu)
{
	pthread_mutex_lock(&gpus_lock);
	cgpu->available = true;
	pthread_mutex_unlock(&gpus_lock);
}

void _quit(int status)
{
//	clean_up();

	exit(status);
}

int spacemesh_api_stats()
{
	int devices = SPACEMESH_API_CPU;
	for (int i = 0; i < s_total_devices; ++i) {
		devices |= s_gpus[i].drv->type;
	}
	return devices;
}

int spacemesh_api_get_gpu_count(int type, int only_available)
{
	int devices = 0;
	if (0 == type) {
		type = SPACEMESH_API_GPU;
	}
	pthread_mutex_lock(&gpus_lock);
	for (int i = 0; i < s_total_devices; ++i) {
		if (0 != (type & s_gpus[i].drv->type)) {
			if (only_available) {
				if (s_gpus[i].available) {
					devices++;
				}
			}
			else {
				devices++;
			}
		}
	}
	pthread_mutex_unlock(&gpus_lock);
	return devices;
}

int spacemesh_api_lock_gpu(int type)
{
	int device = 0;
	if (0 == type) {
		type = SPACEMESH_API_GPU;
	}
	pthread_mutex_lock(&gpus_lock);
	for (int i = 0; i < s_total_devices; ++i) {
		if (s_gpus[i].available && (0 != (type & s_gpus[i].drv->type))) {
			s_gpus[i].available = false;
			device = SPACEMESH_API_USE_LOCKED_DEVICE | (i << 8);
			break;
		}
	}
	pthread_mutex_unlock(&gpus_lock);
	return device;
}

void spacemesh_api_unlock_gpu(int cookie)
{
	pthread_mutex_lock(&gpus_lock);
	int id = (cookie >> 8) & 0x0f;
	if (id >= 0 && id < s_total_devices) {
		s_gpus[id].available = true;
	}
	pthread_mutex_unlock(&gpus_lock);
}

int spacemesh_api_stop(uint32_t ms_timeout)
{
	uint32_t timeout = 0;
	if (abort_flag) {/* already called */
		return SPACEMESH_API_ERROR_ALREADY;
	}

	abort_flag = true;

	while (abort_flag) {
		bool busy = false;
		for (int i = 0; i < s_total_devices; ++i) {
			if (s_gpus[i].busy) {
				busy = true;
				break;
			}
		}
		if (busy) {
			if (timeout >= ms_timeout) {
				abort_flag = false;
				return SPACEMESH_API_ERROR_TIMEOUT;
			}
			usleep(100000);
			timeout += 100;
			continue;
		}
		abort_flag = false;
	}

	return SPACEMESH_API_ERROR_NONE;
}
