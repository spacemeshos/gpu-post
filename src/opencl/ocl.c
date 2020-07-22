/*
 * Copyright 2011-2012 Con Kolivas
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>

#include <time.h>
#include <sys/time.h>
#ifdef WIN32
#define	PATH_MAX	MAX_PATH
#else
#include <pthread.h>
#endif
#include <sys/stat.h>
#include <unistd.h>

#include "api_internal.h"
#include "ocl.h"
#include "ICD/icd_dispatch.h"
#include "scrypt-chacha-cl.inl"

_clState *initCl(struct cgpu_info *cgpu, cl_uint hash_len_bits, bool throttled)
{
	_clState *clState = calloc(1, sizeof(_clState));
	bool patchbfi = false, prog_built = false;
	cl_platform_id platform = NULL;
	char pbuff[256], vbuff[255];
	cl_platform_id* platforms;
	cl_uint preferred_vwidth;
	cl_device_id *devices;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_int status;

	cl_uint scrypt_mem = 128 * cgpu->r;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Getting Platforms. (clGetPlatformsIDs)", status);
		return NULL;
	}

	platforms = (cl_platform_id *)alloca(numPlatforms*sizeof(cl_platform_id));
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Getting Platform Ids. (clGetPlatformsIDs)", status);
		return NULL;
	}

	if (cgpu->platform >= numPlatforms) {
		applog(LOG_ERR, "Specified platform that does not exist");
		return NULL;
	}

	status = clGetPlatformInfo(platforms[cgpu->platform], CL_PLATFORM_VENDOR, sizeof(pbuff), pbuff, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Getting Platform Info. (clGetPlatformInfo)", status);
		return NULL;
	}
	platform = platforms[cgpu->platform];

	if (platform == NULL) {
		perror("NULL platform found!\n");
		return NULL;
	}

	applog(LOG_INFO, "CL Platform vendor: %s", pbuff);
	status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pbuff), pbuff, NULL);
	if (status == CL_SUCCESS) {
		applog(LOG_INFO, "CL Platform name: %s", pbuff);
	}
	status = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(vbuff), vbuff, NULL);
	if (status == CL_SUCCESS) {
		applog(LOG_INFO, "CL Platform version: %s", vbuff);
	}

	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Getting Device IDs (num)", status);
		return NULL;
	}

	if (numDevices > 0 ) {
		devices = (cl_device_id *)malloc(numDevices*sizeof(cl_device_id));

		/* Now, get the device list data */

		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Getting Device IDs (list)", status);
			return NULL;
		}

		if (cgpu->driver_id < (int)numDevices) {
			status = clGetDeviceInfo(devices[cgpu->driver_id], CL_DEVICE_NAME, sizeof(pbuff), pbuff, NULL);
			if (status != CL_SUCCESS) {
				applog(LOG_ERR, "Error %d: Getting Device Info", status);
				return NULL;
			}
			applog(LOG_INFO, "Selected %i: %s", cgpu->driver_id, pbuff);
		} else {
			applog(LOG_ERR, "Invalid GPU %i", cgpu->driver_id);
			return NULL;
		}
	}
	else {
		return NULL;
	}

	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	clState->context = clCreateContextFromType(cps, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Creating Context. (clCreateContextFromType)", status);
		return NULL;
	}

	/////////////////////////////////////////////////////////////////
	// Create an OpenCL command queue
	/////////////////////////////////////////////////////////////////

	clState->commandQueue = clState->context->dispatch->clCreateCommandQueue(clState->context, devices[cgpu->driver_id], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &status);
	if (status != CL_SUCCESS) { /* Try again without OOE enable */
		clState->commandQueue = clState->context->dispatch->clCreateCommandQueue(clState->context, devices[cgpu->driver_id], 0, &status);
	}
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Creating Command Queue. (clCreateCommandQueue)", status);
		return NULL;
	}

	/* Check for BFI INT support. Hopefully people don't mix devices with
	 * and without it! */
	char * extensions = malloc(1024);
	const char * camo = "cl_amd_media_ops";
	char *find;

	status = clGetDeviceInfo(devices[cgpu->driver_id], CL_DEVICE_EXTENSIONS, 1024, (void *)extensions, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_EXTENSIONS", status);
		return NULL;
	}
	find = strstr(extensions, camo);
	if (find) {
		clState->hasBitAlign = true;
	}
		
	/* Check for OpenCL >= 1.0 support, needed for global offset parameter usage. */
	char * devoclver = malloc(1024);
	const char * ocl10 = "OpenCL 1.0";
	const char * ocl11 = "OpenCL 1.1";

	status = clGetDeviceInfo(devices[cgpu->driver_id], CL_DEVICE_VERSION, 1024, (void *)devoclver, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_VERSION", status);
		return NULL;
	}
	find = strstr(devoclver, ocl10);
	if (!find) {
		clState->hasOpenCL11plus = true;
		find = strstr(devoclver, ocl11);
		if (!find) {
			clState->hasOpenCL12plus = true;
		}
	}

	status = clGetDeviceInfo(devices[cgpu->driver_id], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), (void *)&preferred_vwidth, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT", status);
		return NULL;
	}
	applog(LOG_DEBUG, "Preferred vector width reported %d", preferred_vwidth);

	status = clGetDeviceInfo(devices[cgpu->driver_id], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), (void *)&clState->max_work_size, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_MAX_WORK_GROUP_SIZE", status);
		return NULL;
	}
	applog(LOG_DEBUG, "Max work group size reported %d", (int)(clState->max_work_size));

	size_t compute_units = 0;
	status = clGetDeviceInfo(devices[cgpu->driver_id], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), (void *)&compute_units, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_MAX_COMPUTE_UNITS", status);
		return NULL;
	}
	cgpu->gpu_core_count = (uint32_t)compute_units;
	// AMD architechture got 64 compute shaders per compute unit.
	// Source: http://www.amd.com/us/Documents/GCN_Architecture_whitepaper.pdf
	clState->compute_shaders = compute_units * 64;
	applog(LOG_DEBUG, "Max shaders calculated %d", (int)(clState->compute_shaders));

	status = clGetDeviceInfo(devices[cgpu->driver_id], CL_DEVICE_MAX_MEM_ALLOC_SIZE , sizeof(cl_ulong), (void *)&cgpu->gpu_max_alloc, NULL);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Failed to clGetDeviceInfo when trying to get CL_DEVICE_MAX_MEM_ALLOC_SIZE", status);
		return NULL;
	}
	applog(LOG_DEBUG, "Max mem alloc size is %lu", (long unsigned int)(cgpu->gpu_max_alloc));

	clState->wsize = 64;

	applog(LOG_NOTICE, "GPU %d: selecting lookup gap of 4", cgpu->driver_id);
	cgpu->lookup_gap = 4;

	unsigned int bsize = 1024;
	size_t ipt = (bsize / cgpu->lookup_gap + (bsize % cgpu->lookup_gap > 0));

	// if we do not have TC and we do not have BS, then calculate some conservative numbers
	if (!cgpu->buffer_size) {
		unsigned int base_alloc = (int)(cgpu->gpu_max_alloc * 88 / 100 / 1024 / 1024 / 8) * 8 * 1024 * 1024;
		cgpu->thread_concurrency = (uint32_t)(base_alloc / scrypt_mem / ipt);
		cgpu->buffer_size = base_alloc / 1024 / 1024;
		applog(LOG_DEBUG, "88%% Max Allocation: %u", base_alloc);
		applog(LOG_NOTICE, "GPU %d: selecting buffer_size of %zu", cgpu->driver_id, cgpu->buffer_size);
	}

	if (cgpu->buffer_size) {
		// use the buffer-size to overwrite the thread-concurrency
		cgpu->thread_concurrency = (int)((cgpu->buffer_size * 1024 * 1024) / ipt / scrypt_mem);
		applog(LOG_DEBUG, "GPU %d: setting thread_concurrency to %d based on buffer size %d and lookup gap %d", cgpu->driver_id, (int)(cgpu->thread_concurrency), (int)(cgpu->buffer_size), (int)(cgpu->lookup_gap));
	}

	if (throttled) {
		cgpu->thread_concurrency = min(cgpu->thread_concurrency, 32 * cgpu->gpu_core_count);
		applog(LOG_DEBUG, "GPU %d: setting throttled thread_concurrency to %d based on buffer size %d and lookup gap %d", cgpu->driver_id, (int)(cgpu->thread_concurrency), (int)(cgpu->buffer_size), (int)(cgpu->lookup_gap));
	}

	const char *source = scrypt_chacha_cl;
	size_t sourceSize[] = {sizeof(scrypt_chacha_cl)};

	cl_uint slot, cpnd;

	slot = cpnd = 0;

	/////////////////////////////////////////////////////////////////
	// Load CL file, build CL program object, create CL kernel object
	/////////////////////////////////////////////////////////////////

	clState->program = clCreateProgramWithSource(clState->context, 1, (const char **)&source, sourceSize, &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Loading Binary into cl_program (clCreateProgramWithSource)", status);
		return NULL;
	}

	/* create a cl program executable for all the devices specified */
	char *CompilerOptions = calloc(1, 256);

	sprintf(CompilerOptions, "-D LOOKUP_GAP=%d -D CONCURRENT_THREADS=%d -D WORKSIZE=%d",
			cgpu->lookup_gap, (unsigned int)cgpu->thread_concurrency, (int)clState->wsize);

	applog(LOG_DEBUG, "Setting worksize to %d", (int)(clState->wsize));

	if (clState->hasBitAlign) {
		strcat(CompilerOptions, " -D BITALIGN");
		applog(LOG_DEBUG, "cl_amd_media_ops found, setting BITALIGN");
	}
	else {
		applog(LOG_DEBUG, "cl_amd_media_ops not found, will not set BITALIGN");
	}

	if (!clState->hasOpenCL11plus) {
		strcat(CompilerOptions, " -D OCL1");
	}

	applog(LOG_DEBUG, "CompilerOptions: %s", CompilerOptions);
	status = clBuildProgram(clState->program, 1, &devices[cgpu->driver_id], CompilerOptions , NULL, NULL);
	free(CompilerOptions);

	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Building Program (clBuildProgram)", status);
		size_t logSize;
		status = clGetProgramBuildInfo(clState->program, devices[cgpu->driver_id], CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

		char *log = malloc(logSize);
		status = clGetProgramBuildInfo(clState->program, devices[cgpu->driver_id], CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
		applog(LOG_ERR, "%s", log);
		return NULL;
	} else {
		applog(LOG_DEBUG, "Success: Building Program (clBuildProgram)");
	}

	prog_built = true;

#ifdef __APPLE__
	/* OSX OpenCL breaks reading off binaries with >1 GPU so always build
	 * from source. */
	goto built;
#endif

	applog(LOG_INFO, "Initialising kernel with %s bitalign, 1 vectors and worksize %d",
	       clState->hasBitAlign ? "" : "out", (int)(clState->wsize));

	if (!prog_built) {
		/* create a cl program executable for all the devices specified */
		status = clBuildProgram(clState->program, 1, &devices[cgpu->driver_id], NULL, NULL, NULL);
		if (status != CL_SUCCESS) {
			applog(LOG_ERR, "Error %d: Building Program (clBuildProgram)", status);
			size_t logSize;
			status = clGetProgramBuildInfo(clState->program, devices[cgpu->driver_id], CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

			char *log = malloc(logSize);
			status = clGetProgramBuildInfo(clState->program, devices[cgpu->driver_id], CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
			applog(LOG_ERR, "%s", log);
			return NULL;
		}
	}

	/* get a kernel object handle for a kernel with the given name */
	clState->kernel[8] = clCreateKernel(clState->program, "search8", &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Creating Kernel from program. (clCreateKernel)", status);
		return NULL;
	}

	clState->kernel[7] = clCreateKernel(clState->program, "search7", &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Creating Kernel from program. (clCreateKernel)", status);
		return NULL;
	}

	clState->kernel[6] = clCreateKernel(clState->program, "search6", &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Creating Kernel from program. (clCreateKernel)", status);
		return NULL;
	}

	clState->kernel[5] = clCreateKernel(clState->program, "search5", &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Creating Kernel from program. (clCreateKernel)", status);
		return NULL;
	}

	clState->kernel[4] = clCreateKernel(clState->program, "search4", &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Creating Kernel from program. (clCreateKernel)", status);
		return NULL;
	}

	clState->kernel[3] = clCreateKernel(clState->program, "search3", &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Creating Kernel from program. (clCreateKernel)", status);
		return NULL;
	}

	clState->kernel[2] = clCreateKernel(clState->program, "search2", &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Creating Kernel from program. (clCreateKernel)", status);
		return NULL;
	}

	clState->kernel[1] = clCreateKernel(clState->program, "search1", &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: Creating Kernel from program. (clCreateKernel)", status);
		return NULL;
	}

	size_t bufsize = scrypt_mem * ipt * cgpu->thread_concurrency;

	if (!cgpu->buffer_size) {
		applog(LOG_NOTICE, "GPU %d: bufsize for thread @ %dMB based on TC of %zu", cgpu->driver_id, (int)(bufsize/1048576),cgpu->thread_concurrency);
	} else {
		applog(LOG_NOTICE, "GPU %d: bufsize for thread @ %dMB based on buffer-size", cgpu->driver_id, (int)(cgpu->buffer_size));
		bufsize = (size_t)(cgpu->buffer_size)*(1048576);
	}

	/* Use the max alloc value which has been rounded to a power of
		* 2 greater >= required amount earlier */
	if (bufsize > cgpu->gpu_max_alloc) {
		applog(LOG_WARNING, "Maximum buffer memory device %d supports says %lu", cgpu->driver_id, (long unsigned int)(cgpu->gpu_max_alloc));
		applog(LOG_WARNING, "Your scrypt settings come to %d", (int)bufsize);
	}
	applog(LOG_INFO, "Creating scrypt buffer sized %u", (unsigned int)bufsize);
	clState->padbufsize = bufsize;

	/* This buffer is weird and might work to some degree even if
		* the create buffer call has apparently failed, so check if we
		* get anything back before we call it a failure. */
	clState->padbuffer8 = NULL;
	clState->padbuffer8 = clCreateBuffer(clState->context, CL_MEM_READ_WRITE, bufsize, NULL, &status);
	if (status != CL_SUCCESS && !clState->padbuffer8) {
		applog(LOG_ERR, "Error %d: clCreateBuffer (padbuffer8), decrease TC or increase LG", status);
		return NULL;
	}

	clState->CLbuffer0 = clCreateBuffer(clState->context, CL_MEM_READ_ONLY, 128, NULL, &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: clCreateBuffer (CLbuffer0)", status);
		return NULL;
	}

	cl_uint chunkSize = (cgpu->thread_concurrency * hash_len_bits + 7) / 8;

	clState->outputBuffer[0] = clCreateBuffer(clState->context, CL_MEM_WRITE_ONLY, chunkSize, NULL, &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: clCreateBuffer (outputBuffer)", status);
		return NULL;
	}

	clState->outputBuffer[1] = clCreateBuffer(clState->context, CL_MEM_WRITE_ONLY, chunkSize, NULL, &status);
	if (status != CL_SUCCESS) {
		applog(LOG_ERR, "Error %d: clCreateBuffer (outputBuffer)", status);
		return NULL;
	}

	return clState;
}

