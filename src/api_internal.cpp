#include "api.h"
#include "api_internal.h"

#ifdef HAVE_CUDA
#include "cuda/driver-cuda.h"
#endif

#ifdef HAVE_OPENCL
#include "opencl/driver-opencl.h"
#endif

#include "scrypt-jane/scrypt-jane.h"

bool opt_benchmark = false;
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

static struct cgpu_info s_gpus[MAX_GPUDEVICES]; /* Maximum number apparently possible */

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

#ifdef HAVE_CUDA
		cuda_drv.drv_detect(s_gpus, &s_total_devices);
#endif

#ifdef HAVE_OPENCL
		opencl_drv.drv_detect(s_gpus, &s_total_devices);
#endif

		cpu_drv.drv_detect(s_gpus, &s_total_devices);

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

// TODO: wait for available
struct cgpu_info * get_available_gpu()
{
	struct cgpu_info *cgpu = NULL;
	pthread_mutex_lock(&gpus_lock);
	for (int i = 0; i < s_total_devices; ++i) {
		if (s_gpus[i].available) {
			cgpu = &s_gpus[i];
			cgpu->available = false;
			break;
		}
	}
	pthread_mutex_unlock(&gpus_lock);
	return cgpu;
}

struct cgpu_info * get_available_gpu_by_type(enum drv_driver type)
{
	struct cgpu_info *cgpu = NULL;
	pthread_mutex_lock(&gpus_lock);
	for (int i = 0; i < s_total_devices; ++i) {
		if (s_gpus[i].available && (type == s_gpus[i].drv->drv_id)) {
			cgpu = &s_gpus[i];
			cgpu->available = false;
			break;
		}
	}
	pthread_mutex_unlock(&gpus_lock);
	return cgpu;
}

void release_gpu(struct cgpu_info *cgpu)
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
		if (DRIVER_CUDA == s_gpus[i].drv->drv_id) {
			devices |= SPACEMESH_API_CUDA;
		}
		else if (DRIVER_OPENCL == s_gpus[i].drv->drv_id) {
			devices |= SPACEMESH_API_OPENCL;
		}
	}
	return devices;
}

void spacemesh_api_stop()
{
	if (abort_flag) {/* already called */
		return;
	}

	abort_flag = true;
}
