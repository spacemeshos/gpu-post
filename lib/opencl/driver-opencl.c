/*
 * Copyright 2011-2012 Con Kolivas
 * Copyright 2011-2012 Luke Dashjr
 * Copyright 2010 Jeff Garzik
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include <spacemesh-config.h>

#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <sys/types.h>

#ifndef WIN32
#include <sys/resource.h>
#endif

#include "compat.h"
#include "api_internal.h"
#include "driver-opencl.h"
#include "ocl.h"

/* TODO: cleanup externals ********************/

extern bool have_opencl;

/**********************************************/

#ifdef HAVE_OPENCL
struct device_drv opencl_drv;
#endif

#ifdef HAVE_OPENCL

#define CL_SET_ARG(var) status |= clSetKernelArg(*kernel, num++, sizeof(var), (void *)&var)
#define CL_SET_VARG(args, var) status |= clSetKernelArg(*kernel, num++, args * sizeof(uint), (void *)var)

static inline void
be32enc_vect(uint32_t *dst, const uint32_t *src, uint32_t len)
{
	uint32_t i;

	for (i = 0; i < len; i++) {
		dst[i] = bswap_32(src[i]);
	}
}

static void opencl_shutdown(struct cgpu_info *cgpu);

static cl_int queue_scrypt_kernel(_clState *clState, uint8_t *pdata, uint64_t start_pos, uint32_t N)
{
	cl_kernel *kernel = &clState->kernel;
	unsigned int num = 0;
	cl_int status = 0;

	clState->cldata = pdata;
        
	status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, CL_TRUE, 0, 80, clState->cldata, 0, NULL, NULL);

	CL_SET_ARG(clState->CLbuffer0);
	CL_SET_ARG(clState->outputBuffer);
	CL_SET_ARG(clState->padbuffer8);
	CL_SET_ARG(N);
	CL_SET_ARG(start_pos);

	return status;
}

// This is where the number of threads for the GPU gets set - originally 2^I
static void set_threads_hashes(unsigned int vectors, unsigned int compute_shaders, int64_t *hashes, size_t *globalThreads,
					unsigned int minthreads, int *intensity, int *xintensity, int *rawintensity)
{
	unsigned int threads = 0;

	while (threads < minthreads) {
		if (*rawintensity > 0) {
			threads = *rawintensity;
		} else if (*xintensity > 0) {
			threads = compute_shaders * *xintensity;
		} else {
			threads = 1 << *intensity;
		}
		if (threads < minthreads) {
			if (likely(*intensity < MAX_INTENSITY)) {
				(*intensity)++;
			}
			else {
				threads = minthreads;
			}
		}
	}

	*globalThreads = threads;
	*hashes = threads * vectors;
}
#endif /* HAVE_OPENCL */

#ifdef HAVE_OPENCL
static int opencl_detect(struct cgpu_info *gpus, int *active)
{
	cl_int status;
	char pbuff[256];
	cl_uint numDevices;
	cl_uint numPlatforms;
	int most_devices = -1;
	cl_platform_id *platforms;
	cl_platform_id platform = NULL;
	unsigned int i;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	/* If this fails, assume no GPUs. */
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: clGetPlatformsIDs failed (no OpenCL SDK installed?)", status);
		return -1;
	}

	if (numPlatforms == 0) {
		applog(LOG_ERR, "clGetPlatformsIDs returned no platforms (no OpenCL SDK installed?)");
		return -1;
	}

	platforms = (cl_platform_id *)alloca(numPlatforms * sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Getting Platform Ids. (clGetPlatformsIDs)", status);
		return -1;
	}

	for (i = 0; i < numPlatforms; i++) {
		status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuff), pbuff, NULL);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Getting Platform Info. (clGetPlatformInfo)", status);
			return -1;
		}
		platform = platforms[i];
		applog(LOG_INFO, "CL Platform %d vendor: %s", i, pbuff);
		status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pbuff), pbuff, NULL);
		if (status == CL_SUCCESS) {
			applog(LOG_INFO, "CL Platform %d name: %s", i, pbuff);
		}
		status = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(pbuff), pbuff, NULL);
		if (status == CL_SUCCESS) {
			applog(LOG_INFO, "CL Platform %d version: %s", i, pbuff);
		}
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
		if (status != CL_SUCCESS) {
			applog(LOG_INFO, "Error %d: Getting Device IDs (num)", status);
			continue;
		}
		applog(LOG_INFO, "Platform %d devices: %d", i, numDevices);
		if ((int)numDevices > most_devices) {
			most_devices = numDevices;
		}
		if (numDevices) {
			unsigned int j;
			cl_device_id *devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));

			clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
			for (j = 0; j < numDevices && *active < MAX_GPUDEVICES; j++) {
				cl_device_topology_amd topology;
				int status = clGetDeviceInfo(devices[j], CL_DEVICE_TOPOLOGY_AMD, sizeof(cl_device_topology_amd), &topology, NULL);
				if (status == CL_SUCCESS)
				{
					if (topology.raw.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD)
					{
						struct cgpu_info *cgpu = &gpus[*active];
						clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(pbuff), pbuff, NULL);
						applog(LOG_INFO, "\t%i\t%s", j, pbuff);

						cgpu->id = *active;
						cgpu->pci_bus_id = topology.pcie.bus;
						cgpu->pci_device_id = topology.pcie.device;
						cgpu->deven = DEV_ENABLED;
						cgpu->platform = i;
						cgpu->drv = &opencl_drv;
						cgpu->driver_id = j;

						*active += 1;

						have_opencl = true;
					}
				}
			}
			free(devices);
		}
	}

	return most_devices;
}

static void reinit_opencl_device(struct cgpu_info *gpu)
{
//	tq_push(control_thr[gpur_thr_id].q, gpu);
}

static uint32_t *blank_res;

static bool opencl_prepare(struct cgpu_info *cgpu, unsigned N)
{
	if (N != cgpu->N) {
		char name[256];
		strcpy(name, "");
		applog(LOG_INFO, "Init GPU thread for GPU %i, platform GPU %i, pci [%d:%d]", cgpu->id, cgpu->driver_id, cgpu->pci_bus_id, cgpu->pci_device_id);
		if (cgpu->device_data) {
			opencl_shutdown(cgpu);
		}
		cgpu->device_data = initCl(cgpu, name, sizeof(name));
		if (!cgpu->device_data) {
			applog(LOG_ERR, "Failed to init GPU, disabling device %d", cgpu->id);
			cgpu->deven = DEV_DISABLED;
			cgpu->status = LIFE_NOSTART;
			return false;
		}
		if (!cgpu->name) {
			cgpu->name = strdup(name);
		}
		if (!cgpu->kname) {
			cgpu->kname = "scrypt-chacha";
		}
		cgpu->N = N;
		applog(LOG_INFO, "initCl() finished. Found %s", name);
	}
	return true;
}

static bool opencl_init(struct cgpu_info *cgpu)
{
	cgpu->status = LIFE_WELL;
	return true;
}

static int64_t opencl_scrypt_positions(struct cgpu_info *cgpu, uint8_t *pdata, uint64_t start_position, uint64_t end_position, uint8_t *output, uint32_t N, struct timeval *tv_start, struct timeval *tv_end)
{
	if (opencl_prepare(cgpu, N))
	{
		_clState *clState = (_clState *)cgpu->device_data;
		const cl_kernel *kernel = &clState->kernel;

		cl_int status;
		size_t globalThreads[1] = { cgpu->thread_concurrency };
		size_t localThreads[1] = { clState->wsize };

		gettimeofday(tv_start, NULL);

		uint64_t n = start_position;
		size_t outLength = end_position - start_position + 1;

//		set_threads_hashes(1, (unsigned int)clState->compute_shaders, &hashes, globalThreads, (unsigned int)localThreads[0], &cgpu->intensity, &cgpu->xintensity, &cgpu->rawintensity);

		do {
			status = queue_scrypt_kernel(clState, pdata, n, N);
			if (unlikely(status != CL_SUCCESS)) {
				applog(LOG_ERR, "Error: clSetKernelArg of all params failed.");
				return -1;
			}

			status = clEnqueueNDRangeKernel(clState->commandQueue, *kernel, 1, NULL, globalThreads, localThreads, 0, NULL, NULL);

			if (unlikely(status != CL_SUCCESS)) {
				applog(LOG_ERR, "Error %d: Enqueueing kernel onto command queue. (clEnqueueNDRangeKernel)", status);
				return -1;
			}

			n += cgpu->thread_concurrency;

			status = clEnqueueReadBuffer(clState->commandQueue, clState->outputBuffer, CL_TRUE, 0, min(cgpu->thread_concurrency, outLength), output, 0, NULL, NULL);
			if (unlikely(status != CL_SUCCESS)) {
				applog(LOG_ERR, "Error: clEnqueueReadBuffer failed error %d. (clEnqueueReadBuffer)", status);
				return -1;
			}

			/* This finish flushes the readbuffer set with CL_FALSE in clEnqueueReadBuffer */
			clFinish(clState->commandQueue);

			output += cgpu->thread_concurrency;
			outLength -= cgpu->thread_concurrency;

		} while (n <= end_position && !abort_flag);

		gettimeofday(tv_end, NULL);

		return 0;
	}
	return -1;
}

static void opencl_shutdown(struct cgpu_info *cgpu)
{
	_clState *clState = (_clState *)cgpu->device_data;
	if (!clState) {
		clReleaseKernel(clState->kernel);
		clReleaseProgram(clState->program);
		clReleaseCommandQueue(clState->commandQueue);
		clReleaseContext(clState->context);
		free(cgpu->device_data);
		cgpu->device_data = NULL;
	}
}

struct device_drv opencl_drv = {
	DRIVER_OPENCL,
	"opencl",
	"GPU",
	opencl_detect,
	reinit_opencl_device,
	opencl_init,
	opencl_scrypt_positions,
	opencl_shutdown
};
#endif
