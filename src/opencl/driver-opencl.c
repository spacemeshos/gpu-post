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

#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <sys/types.h>

#ifndef WIN32
#include <sys/resource.h>
#endif
#include "api.h"
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

static void opencl_shutdown(struct cgpu_info *cgpu);

static cl_int queue_scrypt_kernel(_clState *clState, uint8_t *pdata, uint64_t start_pos, uint32_t N, int nBuf, uint32_t hash_len_bits, uint32_t r, uint32_t p)
{
	cl_kernel *kernel = &clState->kernel[hash_len_bits];
	unsigned int num = 0;
	cl_int status = 0;

	status = clEnqueueWriteBuffer(clState->commandQueue, clState->CLbuffer0, CL_TRUE, 0, 72, pdata, 0, NULL, NULL);

	CL_SET_ARG(clState->CLbuffer0);
	CL_SET_ARG(clState->outputBuffer[nBuf]);
	CL_SET_ARG(clState->padbuffer8);
	CL_SET_ARG(N);
	CL_SET_ARG(start_pos);
	CL_SET_ARG(r);
	CL_SET_ARG(p);

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
	int most_devices = 0;
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
						clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(cgpu->name), cgpu->name, NULL);
						applog(LOG_INFO, "\t%i\t%s", j, pbuff);

						cgpu->id = *active;
						cgpu->pci_bus_id = topology.pcie.bus;
						cgpu->pci_device_id = topology.pcie.device;
						cgpu->deven = DEV_ENABLED;
						cgpu->platform = i;
						cgpu->drv = &opencl_drv;
						cgpu->driver_id = j;

						*active += 1;

						most_devices++;
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

static bool opencl_prepare(struct cgpu_info *cgpu, unsigned N, uint32_t r, uint32_t p, cl_uint hash_len_bits, bool throttled)
{
	if (N != cgpu->N || r != cgpu->r || p != cgpu->p || hash_len_bits != cgpu->hash_len_bits) {
		cgpu->name[0] = 0;
		applog(LOG_INFO, "Init GPU thread for GPU %i, platform GPU %i, pci [%d:%d]", cgpu->id, cgpu->driver_id, cgpu->pci_bus_id, cgpu->pci_device_id);
		if (cgpu->device_data) {
			opencl_shutdown(cgpu);
		}

		cgpu->N = N;
		cgpu->r = r;
		cgpu->p = p;
		cgpu->hash_len_bits = hash_len_bits;

		cgpu->device_data = initCl(cgpu, hash_len_bits, throttled);
		if (!cgpu->device_data) {
			applog(LOG_ERR, "Failed to init GPU, disabling device %d", cgpu->id);
			cgpu->deven = DEV_DISABLED;
			cgpu->status = LIFE_NOSTART;
			return false;
		}
		applog(LOG_INFO, "initCl() finished. Found %s", cgpu->name);
	}
	return true;
}

static bool opencl_init(struct cgpu_info *cgpu)
{
	cgpu->status = LIFE_WELL;
	return true;
}

#define	USE_ASYNC_BUFFER_READ	1

static int opencl_scrypt_positions(
	struct cgpu_info *cgpu,
	uint8_t *pdata,
	uint64_t start_position,
	uint64_t end_position,
	uint32_t hash_len_bits,
	uint32_t options,
	uint8_t *output,
	uint32_t N,
	uint32_t r,
	uint32_t p,
	uint64_t *idx_solution,
	struct timeval *tv_start,
	struct timeval *tv_end,
	uint64_t *hashes_computed
)
{
	cgpu->busy = 1;
	if (hashes_computed) {
		*hashes_computed = 0;
	}

	if (opencl_prepare(cgpu, N, r, p, hash_len_bits, 0 != (options & SPACEMESH_API_THROTTLED_MODE)))
	{
		_clState *clState = (_clState *)cgpu->device_data;
		const cl_kernel *kernel = &clState->kernel[hash_len_bits];

		cl_int status;
		size_t globalThreads[1] = { cgpu->thread_concurrency };
		size_t localThreads[1] = { clState->wsize };
		bool firstBuffer = true;
		bool running = false;

		gettimeofday(tv_start, NULL);

		uint64_t n = start_position;
		size_t positions = end_position - start_position + 1;
		uint64_t chunkSize = (cgpu->thread_concurrency * hash_len_bits) / 8;
		uint64_t outLength = ((end_position - start_position + 1) * hash_len_bits + 7) / 8;
		uint64_t computedPositions = 0;
		uint8_t *out = output;

		do {
			status = queue_scrypt_kernel(clState, pdata, n, N, (firstBuffer ? 0 : 1), hash_len_bits, r, p);
			if (unlikely(status != CL_SUCCESS)) {
				applog(LOG_ERR, "Error: clSetKernelArg of all params failed.");
				cgpu->busy = 0;
				return -1;
			}

			if (globalThreads[0] > positions) {
				globalThreads[0] = clState->wsize * ((positions + clState->wsize - 1) / clState->wsize);
			}

			status = clEnqueueNDRangeKernel(clState->commandQueue, *kernel, 1, NULL, globalThreads, localThreads, 0, NULL, NULL);

			if (unlikely(status != CL_SUCCESS)) {
				applog(LOG_ERR, "Error %d: Enqueueing kernel onto command queue. (clEnqueueNDRangeKernel)", status);
				cgpu->busy = 0;
				return -1;
			}

			n += cgpu->thread_concurrency;
			computedPositions += cgpu->thread_concurrency;
#if USE_ASYNC_BUFFER_READ
			if (clState->outputEvent[(firstBuffer ? 0 : 1)]) {
				clReleaseEvent(clState->outputEvent[(firstBuffer ? 0 : 1)]);
				clState->outputEvent[(firstBuffer ? 0 : 1)] = NULL;
			}
			status = clEnqueueReadBuffer(clState->commandQueue, clState->outputBuffer[(firstBuffer ? 0 : 1)], CL_FALSE, 0, min(chunkSize, outLength), out, 0, NULL, &clState->outputEvent[(firstBuffer ? 0 : 1)]);
#else
			status = clEnqueueReadBuffer(clState->commandQueue, clState->outputBuffer[(firstBuffer ? 0 : 1)], CL_TRUE, 0, min(chunkSize, outLength), out, 0, NULL, NULL);
#endif
			if (unlikely(status != CL_SUCCESS)) {
				applog(LOG_ERR, "Error: clEnqueueReadBuffer failed error %d. (clEnqueueReadBuffer)", status);
				cgpu->busy = 0;
				return -1;
			}
#if USE_ASYNC_BUFFER_READ
			firstBuffer = !firstBuffer;

			if (running) {
				clWaitForEvents(1, &clState->outputEvent[(firstBuffer ? 0 : 1)]);
			}
			else {
				running = true;
				clFlush(clState->commandQueue);
			}
#else
			clFinish(clState->commandQueue);
#endif
			out += chunkSize;
			outLength -= chunkSize;
			positions -= cgpu->thread_concurrency;

		} while (n <= end_position && !g_spacemesh_api_abort_flag);
#if USE_ASYNC_BUFFER_READ
		if (running) {
			clWaitForEvents(1, &clState->outputEvent[(firstBuffer ? 0 : 1)]);
			clFinish(clState->commandQueue);
			clReleaseEvent(clState->outputEvent[0]);
			clReleaseEvent(clState->outputEvent[1]);
		}
#endif
		gettimeofday(tv_end, NULL);

		cgpu->busy = 0;
		size_t total = end_position - start_position + 1;
		computedPositions = min(total, computedPositions);

		if (hashes_computed) {
			*hashes_computed = computedPositions;
		}

		int usedBits = (computedPositions * hash_len_bits % 8);
		if (usedBits) {
			output[(computedPositions * hash_len_bits) / 8] &= 0xff >> (8 - usedBits);
		}

		return (n <= end_position) ? SPACEMESH_API_ERROR_CANCELED : SPACEMESH_API_ERROR_NONE;
	}

	cgpu->busy = 0;

	return SPACEMESH_API_ERROR;
}

static void opencl_shutdown(struct cgpu_info *cgpu)
{
	_clState *clState = (_clState *)cgpu->device_data;
	if (!clState) {
		if (clState->kernel[1]) {
			clReleaseKernel(clState->kernel[1]);
		}
		if (clState->kernel[2]) {
			clReleaseKernel(clState->kernel[2]);
		}
		if (clState->kernel[3]) {
			clReleaseKernel(clState->kernel[3]);
		}
		if (clState->kernel[4]) {
			clReleaseKernel(clState->kernel[4]);
		}
		if (clState->kernel[5]) {
			clReleaseKernel(clState->kernel[5]);
		}
		if (clState->kernel[6]) {
			clReleaseKernel(clState->kernel[6]);
		}
		if (clState->kernel[7]) {
			clReleaseKernel(clState->kernel[7]);
		}
		if (clState->kernel[8]) {
			clReleaseKernel(clState->kernel[8]);
		}
		clReleaseProgram(clState->program);
		clReleaseCommandQueue(clState->commandQueue);
		clReleaseContext(clState->context);
		free(cgpu->device_data);
		cgpu->device_data = NULL;
	}
}

struct device_drv opencl_drv = {
	SPACEMESH_API_OPENCL,
	"opencl",
	"GPU",
	opencl_detect,
	reinit_opencl_device,
	opencl_init,
	opencl_scrypt_positions,
	{ NULL, NULL },
	opencl_shutdown
};
#endif
