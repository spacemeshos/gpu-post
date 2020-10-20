#include "api.h"
#include "api_internal.h"
#include <stdarg.h>

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

volatile bool g_spacemesh_api_abort_flag = false;
bool g_spacemesh_api_opt_debug = false;
#ifdef WIN32
CRITICAL_SECTION g_spacemesh_api_applog_lock;
#else
pthread_mutex_t g_spacemesh_api_applog_lock;
#endif

static int s_total_devices = 0;
bool g_spacemesh_api_have_cuda = false;
bool g_spacemesh_api_have_opencl = false;
bool g_spacemesh_api_have_vulkan = false;

static struct cgpu_info s_gpus[MAX_GPUDEVICES]; /* Maximum number apparently possible */
static struct cgpu_info s_cpu;

static volatile int api_inited = 0;

extern "C" void spacemesh_api_init()
{
	if (0 == api_inited) {
		api_inited = 1;
#ifdef WIN32
		InitializeCriticalSection(&g_spacemesh_api_applog_lock);
#else
		pthread_mutex_init(&g_spacemesh_api_applog_lock, NULL);
#endif

		memset(s_gpus, 0, sizeof(s_gpus));
		memset(&s_cpu, 0, sizeof(s_cpu));

#ifdef HAVE_VULKAN
		g_spacemesh_api_have_vulkan = vulkan_drv.drv_detect(s_gpus, &s_total_devices) > 0;
#endif

#ifdef HAVE_CUDA
		g_spacemesh_api_have_cuda = cuda_drv.drv_detect(s_gpus, &s_total_devices) > 0;
#endif

#ifdef HAVE_OPENCL
		g_spacemesh_api_have_opencl = opencl_drv.drv_detect(s_gpus, &s_total_devices) > 0;
#endif

		if (cpu_drv.drv_detect(&s_cpu, NULL) > 0) {
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

extern "C" struct cgpu_info * spacemesh_api_get_gpu(int id)
{
	spacemesh_api_init();
	if (id >= 0 && id < s_total_devices) {
		return &s_gpus[id];
	}
	if (id == s_total_devices && s_cpu.available) {
		return &s_cpu;
	}
	return NULL;
}

void _quit(int status)
{
//	clean_up();

	exit(status);
}

extern "C" int spacemesh_api_stop_inprogress()
{
	return g_spacemesh_api_abort_flag;
}

extern "C" int spacemesh_api_stop(uint32_t ms_timeout)
{
	uint32_t timeout = 0;
	if (g_spacemesh_api_abort_flag) {/* already called */
		return SPACEMESH_API_ERROR_ALREADY;
	}

	if (api_inited) {
		g_spacemesh_api_abort_flag = true;

		while (g_spacemesh_api_abort_flag) {
			bool busy = false;
			for (int i = 0; i < s_total_devices; ++i) {
				if (s_gpus[i].busy) {
					busy = true;
					break;
				}
			}
			busy |= s_cpu.busy;
			if (busy) {
				if (timeout >= ms_timeout) {
					g_spacemesh_api_abort_flag = false;
					return SPACEMESH_API_ERROR_TIMEOUT;
				}
				usleep(100000);
				timeout += 100;
				continue;
			}
			g_spacemesh_api_abort_flag = false;
		}
	}

	g_spacemesh_api_abort_flag = false;

	return SPACEMESH_API_ERROR_NONE;
}

extern "C" int spacemesh_api_get_providers(
	PostComputeProvider *providers, // out providers info buffer, if NULL - return count of available providers
	int max_providers			    // buffer size
)
{
	int i;
	int current_providers = 0;

	spacemesh_api_init();

	if (NULL == providers) {
		for (i = 0; i < s_total_devices; i++) {
			if (NULL != s_gpus[i].drv && 0 != (s_gpus[i].drv->type & (SPACEMESH_API_CUDA | SPACEMESH_API_VULKAN))) {
				current_providers++;
			}
		}
		if (s_cpu.available) {
			current_providers++;
		}
		return current_providers;
	}

	for (i = 0; current_providers < max_providers && i < s_total_devices; i++) {
		if (NULL != s_gpus[i].drv && 0 != (s_gpus[i].drv->type & (SPACEMESH_API_CUDA | SPACEMESH_API_VULKAN))) {
			if (0 != (s_gpus[i].drv->type & SPACEMESH_API_CUDA)) {
				providers->compute_api = COMPUTE_API_CLASS_CUDA;
			}
			else {
				providers->compute_api = COMPUTE_API_CLASS_VULKAN;
			}
			providers->id = i;
			memcpy(providers->model, s_gpus[i].name, min(sizeof(providers->model), sizeof(s_gpus[i].name)));
			providers->model[sizeof(providers->model) - 1] = 0;

			providers++;
			current_providers++;
		}
	}

	if (s_cpu.available && current_providers < max_providers) {
		providers->compute_api = COMPUTE_API_CLASS_CPU;
		providers->id = i;
		providers->model[0] = 'C';
		providers->model[1] = 'P';
		providers->model[2] = 'U';
		providers->model[3] = 0;

		current_providers++;
	}

	return current_providers;
}

extern int opt_logs = 0;

void applog(int prio, const char *fmt, ...)
{
	if (opt_logs) {
		va_list ap;

		va_start(ap, fmt);

		{
			const char* color = "";
			const time_t now = time(NULL);
			char *f;
			int len;
			struct tm tm;

			localtime_r(&now, &tm);

			len = 40 + (int)strlen(fmt) + 2;
			f = (char*)alloca(len);
			sprintf(f, "[%d-%02d-%02d %02d:%02d:%02d]%s %s%s\n",
				tm.tm_year + 1900,
				tm.tm_mon + 1,
				tm.tm_mday,
				tm.tm_hour,
				tm.tm_min,
				tm.tm_sec,
				color,
				fmt,
				""
			);
			if (prio == LOG_RAW) {
				// no time prefix, for ccminer -n
				sprintf(f, "%s%s\n", fmt, "");
			}
			pthread_mutex_lock(&g_spacemesh_api_applog_lock);
			vfprintf(stdout, f, ap);	/* atomic write to stdout */
			fflush(stdout);
			pthread_mutex_unlock(&g_spacemesh_api_applog_lock);
		}
		va_end(ap);
	}
}

// Use different prefix if multiple cpu threads per gpu
// Also, auto hide LOG_DEBUG if --debug (-D) is not used
void gpulog(int prio, int dev_id, const char *fmt, ...)
{
	if (opt_logs) {
		char _ALIGN(128) pfmt[128];
		char _ALIGN(128) line[256];
		int len;
		va_list ap;

		if (prio == LOG_DEBUG && !g_spacemesh_api_opt_debug)
			return;

		len = snprintf(pfmt, 128, "GPU #%d: %s", dev_id, fmt);
		pfmt[sizeof(pfmt) - 1] = '\0';

		va_start(ap, fmt);

		if (len && vsnprintf(line, sizeof(line), pfmt, ap)) {
			line[sizeof(line) - 1] = '\0';
			applog(prio, "%s", line);
		}
		else {
			fprintf(stderr, "%s OOM!\n", __func__);
		}

		va_end(ap);
	}
}
