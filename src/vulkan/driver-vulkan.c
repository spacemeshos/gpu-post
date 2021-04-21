#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <sys/types.h>

#ifndef WIN32
#include <sys/resource.h>
#endif
#include "api.h"
#include "api_internal.h"

/**********************************************/

#ifdef HAVE_VULKAN
#include "vulkan-helpers.h"
#include "driver-vulkan.h"
#include "scrypt-chacha-vulkan.inl"

static const uint64_t keccak_round_constants[24] =
{
	0x0000000000000001ull, 0x0000000000008082ull,
	0x800000000000808aull, 0x8000000080008000ull,
	0x000000000000808bull, 0x0000000080000001ull,
	0x8000000080008081ull, 0x8000000000008009ull,
	0x000000000000008aull, 0x0000000000000088ull,
	0x0000000080008009ull, 0x000000008000000aull,
	0x000000008000808bull, 0x800000000000008bull,
	0x8000000000008089ull, 0x8000000000008003ull,
	0x8000000000008002ull, 0x8000000000000080ull,
	0x000000000000800aull, 0x800000008000000aull,
	0x8000000080008081ull, 0x8000000000008080ull,
	0x0000000080000001ull, 0x8000000080008008ull
};

static const uint32_t keccak_rotc[24] =
{
	1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
	27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

static const uint32_t keccak_piln[24] =
{
	10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
	15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

typedef struct {
	int deviceId;
	VkDevice vkDevice;
	VkDeviceMemory gpuLocalMemory;
	VkDeviceMemory gpuSharedMemory;

	VkBuffer gpu_params;
	VkBuffer gpu_constants;

	VkBuffer outputBuffer[2];
	VkBuffer CLbuffer0;
	VkBuffer padbuffer8;

	VkDescriptorSet descriptorSet;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;
	VkSemaphore semaphore;
	VkDescriptorPool descriptorPool;
	VkShaderModule shaderModule;
	VkDescriptorSetLayout descriptorSetLayout;
	VkQueue queue;
	VkFence fence;

	uint32_t alignment;
	uint64_t bufSize;
	uint64_t memConstantSize;
	uint64_t memParamsSize;
	uint64_t memInputSize;
	uint64_t memOutputSize;
	uint64_t sharedMemorySize;
} _vulkanState;

typedef struct AlgorithmConstants {
	uint64_t keccakf_rndc[24];
	uint32_t keccakf_rotc[24];
	uint32_t keccakf_piln[24];
} AlgorithmConstants;

AlgorithmConstants gpuConstants;

typedef struct AlgorithmParams {
	uint64_t global_work_offset;
	uint32_t N;
	uint32_t hash_len_bits;
} AlgorithmParams;

struct device_drv vulkan_drv;

void init_glslang();

static uint64_t alignBuffer(uint64_t size, uint64_t align)
{
	if (align == 1) {
		return size;
	}
	else {
		return (size + align - 1)&(~(align - 1));
	}
}

static _vulkanState *initVulkan(struct cgpu_info *cgpu, char *name, size_t nameSize, uint32_t hash_len_bits, bool throttled, bool copy_only)
{
	_vulkanState *state = calloc(1, sizeof(_vulkanState));

	uint32_t scrypt_mem = 128 * cgpu->r;

	uint32_t computeQueueFamillyIndex = getComputeQueueFamillyIndex(cgpu->driver_id);
	if (computeQueueFamillyIndex < 0) {
		applog(LOG_ERR, "GPU %d: Compute query familly not found\n", cgpu->driver_id);
		return NULL;
	}

	state->deviceId = cgpu->driver_id;
	state->vkDevice = createDevice(cgpu->driver_id, computeQueueFamillyIndex);
	if (NULL == state->vkDevice) {
		applog(LOG_NOTICE, "GPU %d: Create Vulkan device instance failed", cgpu->driver_id);
		return NULL;
	}

	VkDeviceMemory tmpMem = allocateGPUMemory(state->deviceId, state->vkDevice, 1024, true, true);
	VkBuffer tmpBuf = createBuffer(state->vkDevice, computeQueueFamillyIndex, tmpMem, 256, 0);
	state->alignment = (uint32_t)getBufferMemoryRequirements(state->vkDevice, tmpBuf);
	gVulkan.vkDestroyBuffer(state->vkDevice, tmpBuf, NULL);
	gVulkan.vkFreeMemory(state->vkDevice, tmpMem, NULL);

	cgpu->work_size = 64;

	applog(LOG_NOTICE, "GPU %d: selecting lookup gap of 4", cgpu->driver_id);
	cgpu->lookup_gap = 4;

	unsigned int bsize = 1024;
	size_t ipt = (bsize / cgpu->lookup_gap + (bsize % cgpu->lookup_gap > 0));

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
	}

	cgpu->thread_concurrency = min(cgpu->thread_concurrency, /*cgpu->work_size*/ 32 * 1024);
	uint32_t chunkSize = copy_only ? (cgpu->thread_concurrency * 32) : ((cgpu->thread_concurrency * hash_len_bits + 7) / 8);

	applog(LOG_DEBUG, "GPU %d: setting thread_concurrency to %d based on buffer size %d and lookup gap %d", cgpu->driver_id, (int)(cgpu->thread_concurrency), (int)(cgpu->buffer_size), (int)(cgpu->lookup_gap));

	state->bufSize = alignBuffer(scrypt_mem * ipt * cgpu->thread_concurrency, state->alignment);
	state->memConstantSize = alignBuffer(sizeof(AlgorithmConstants), state->alignment);
	state->memParamsSize = alignBuffer(sizeof(AlgorithmParams), state->alignment);
	state->memInputSize = alignBuffer(72, state->alignment);
	state->memOutputSize = alignBuffer(chunkSize, state->alignment);
	state->sharedMemorySize = state->memConstantSize + state->memParamsSize + state->memInputSize + 2 * state->memOutputSize;

	state->gpuLocalMemory = allocateGPUMemory(state->deviceId, state->vkDevice, state->bufSize, true, true);
	state->gpuSharedMemory = allocateGPUMemory(state->deviceId, state->vkDevice, state->sharedMemorySize, false, true);

	state->padbuffer8 = createBuffer(state->vkDevice, computeQueueFamillyIndex, state->gpuLocalMemory, state->bufSize, 0);

	uint64_t o = 0;
	state->gpu_constants = createBuffer(state->vkDevice, computeQueueFamillyIndex, state->gpuSharedMemory, state->memConstantSize, o);
	o += state->memConstantSize;
	state->gpu_params = createBuffer(state->vkDevice, computeQueueFamillyIndex, state->gpuSharedMemory, state->memParamsSize, o);
	o += state->memParamsSize;
	state->CLbuffer0 = createBuffer(state->vkDevice, computeQueueFamillyIndex, state->gpuSharedMemory, state->memInputSize, o);
	o += state->memInputSize;
	state->outputBuffer[0] = createBuffer(state->vkDevice, computeQueueFamillyIndex, state->gpuSharedMemory, state->memOutputSize, o);
	o += state->memOutputSize;
	state->outputBuffer[1] = createBuffer(state->vkDevice, computeQueueFamillyIndex, state->gpuSharedMemory, state->memOutputSize, o);

	gVulkan.vkGetDeviceQueue(state->vkDevice, computeQueueFamillyIndex, 0, &state->queue);

	state->pipelineLayout = bindBuffers(state->vkDevice, &state->descriptorSet, &state->descriptorPool, &state->descriptorSetLayout,
		state->padbuffer8, state->gpu_constants, state->gpu_params, state->CLbuffer0, state->outputBuffer[0], state->outputBuffer[1]
	);

	void *ptr = NULL;
	CHECK_RESULT(gVulkan.vkMapMemory(state->vkDevice, state->gpuSharedMemory, 0, state->memConstantSize, 0, (void **)&ptr), "vkMapMemory", NULL);
	memcpy(ptr, (const void*)&gpuConstants, sizeof(AlgorithmConstants));
	gVulkan.vkUnmapMemory(state->vkDevice, state->gpuSharedMemory);

	VkCommandPoolCreateInfo commandPoolCreateInfo = {
		VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		0,
		VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		computeQueueFamillyIndex
	};
	CHECK_RESULT(gVulkan.vkCreateCommandPool(state->vkDevice, &commandPoolCreateInfo, 0, &state->commandPool), "vkCreateCommandPool", NULL);

	VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
		VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		0,
		state->commandPool,
		VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		1
	};
	CHECK_RESULT(gVulkan.vkAllocateCommandBuffers(state->vkDevice, &commandBufferAllocateInfo, &state->commandBuffer), "vkAllocateCommandBuffers", NULL);

	VkSemaphoreCreateInfo semaphoreCreateInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	semaphoreCreateInfo.pNext = NULL;
	semaphoreCreateInfo.flags = 0;
	CHECK_RESULT(gVulkan.vkCreateSemaphore(state->vkDevice, &semaphoreCreateInfo, NULL, &state->semaphore), "vkCreateSemaphore", NULL);

	VkFenceCreateInfo fenceCreateInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	fenceCreateInfo.pNext = NULL;
	fenceCreateInfo.flags = 0;

	CHECK_RESULT(gVulkan.vkCreateFence(state->vkDevice, &fenceCreateInfo, NULL, &state->fence), "vkCreateFence", NULL);

	char options[256];
	snprintf(options, sizeof(options), "#version 450\n#define LOOKUP_GAP %d\n#define CONCURRENT_THREADS %d\n#define WORKSIZE %d\n#define LABEL_SIZE %d\n",
		cgpu->lookup_gap, (unsigned int)cgpu->thread_concurrency, (int)cgpu->work_size, hash_len_bits);

	state->pipeline = compileShader(state->vkDevice, state->pipelineLayout, &state->shaderModule, scrypt_chacha_comp, options, (int)cgpu->work_size, hash_len_bits, copy_only);
	if (!state->pipeline) {
		return NULL;
	}

	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, 0, 0 };
	CHECK_RESULT(gVulkan.vkBeginCommandBuffer(state->commandBuffer, &commandBufferBeginInfo), "vkBeginCommandBuffer", NULL);

	gVulkan.vkCmdBindPipeline(state->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, state->pipeline);
	gVulkan.vkCmdBindDescriptorSets(state->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, state->pipelineLayout, 0, 1, &state->descriptorSet, 0, 0);
#if 1
	gVulkan.vkCmdDispatch(state->commandBuffer, cgpu->thread_concurrency / cgpu->work_size, 1, 1);
#else
	gVulkan.vkCmdDispatch(state->commandBuffer, 1, 1, 1);
#endif
	CHECK_RESULT(gVulkan.vkEndCommandBuffer(state->commandBuffer), "vkEndCommandBuffer", NULL);

	return state;
}

static int vulkan_detect(struct cgpu_info *gpus, int *active)
{
	int most_devices = 0;

	const VkApplicationInfo applicationInfo = {
		VK_STRUCTURE_TYPE_APPLICATION_INFO,
		0,
		"spacemesh",
		0,
		"",
		0,
		VK_API_VERSION_1_0
	};

	const VkInstanceCreateInfo instanceCreateInfo = {
		VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		0,
		0,
		&applicationInfo,
		0,
		NULL,
		0,
		NULL
	};

	if (initVulkanLibrary()) {
		return 0;
	}

	CHECK_RESULT(gVulkan.vkCreateInstance(&instanceCreateInfo, 0, &gInstance), "vkCreateInstance", 0);

	gPhysicalDeviceCount = 0;
	CHECK_RESULT(gVulkan.vkEnumeratePhysicalDevices(gInstance, &gPhysicalDeviceCount, 0), "vkEnumeratePhysicalDevices", 0);
	if (gPhysicalDeviceCount > 0) {
		gPhysicalDevices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * gPhysicalDeviceCount);
		memset(gPhysicalDevices, 0, sizeof(VkPhysicalDevice) * gPhysicalDeviceCount);
		CHECK_RESULT(gVulkan.vkEnumeratePhysicalDevices(gInstance, &gPhysicalDeviceCount, gPhysicalDevices), "vkEnumeratePhysicalDevices", 0);
		for (unsigned i = 0; i < gPhysicalDeviceCount; i++) {
			struct cgpu_info *cgpu = &gpus[*active];

			VkPhysicalDeviceProperties physicalDeviceProperties;
			gVulkan.vkGetPhysicalDeviceProperties(gPhysicalDevices[i], &physicalDeviceProperties);

			if (0x10DE == physicalDeviceProperties.vendorID) {
				continue;
			}

			memcpy(cgpu->name, physicalDeviceProperties.deviceName, min(sizeof(cgpu->name),sizeof(physicalDeviceProperties.deviceName)));
			cgpu->name[sizeof(cgpu->name) - 1] = 0;

			VkPhysicalDeviceMemoryProperties memoryProperties;
			gVulkan.vkGetPhysicalDeviceMemoryProperties(gPhysicalDevices[i], &memoryProperties);

			cgpu->id = *active;
			cgpu->pci_bus_id = 0;
			cgpu->pci_device_id = 0;
			cgpu->deven = DEV_ENABLED;
			cgpu->drv = &vulkan_drv;
			cgpu->driver_id = i;

			for (unsigned j = 0; j < memoryProperties.memoryTypeCount; j++) {
				VkMemoryType t = memoryProperties.memoryTypes[j];
				if ((t.propertyFlags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
					if (t.heapIndex < memoryProperties.memoryHeapCount) {
						cgpu->gpu_max_alloc = memoryProperties.memoryHeaps[t.heapIndex].size;
						break;
					}
				}
			}

			*active += 1;
			most_devices++;
		}
	} else {
		applog(LOG_ERR, "No graphic cards were found by Vulkan. Use Adrenalin not Crimson and check your drivers with VulkanInfo.");
	}

	memcpy(gpuConstants.keccakf_rndc, keccak_round_constants, sizeof(keccak_round_constants));
	memcpy(gpuConstants.keccakf_rotc, keccak_rotc, sizeof(keccak_rotc));
	memcpy(gpuConstants.keccakf_piln, keccak_piln, sizeof(keccak_piln));

	init_glslang();

	return most_devices;
}

static void reinit_vulkan_device(struct cgpu_info *gpu)
{
}

static void vulkan_shutdown(struct cgpu_info *cgpu);

static bool vulkan_prepare(struct cgpu_info *cgpu, unsigned N, uint32_t r, uint32_t p, uint32_t hash_len_bits, bool throttled, bool copy_only)
{
	if (N != cgpu->N || r != cgpu->r || p != cgpu->p || hash_len_bits != cgpu->hash_len_bits) {
		if (cgpu->device_data) {
			vulkan_shutdown(cgpu);
		}

		cgpu->N = N;
		cgpu->r = r;
		cgpu->p = p;
		cgpu->hash_len_bits = hash_len_bits;

		VkPhysicalDeviceProperties physicalDeviceProperties;
		gVulkan.vkGetPhysicalDeviceProperties(gPhysicalDevices[cgpu->driver_id], &physicalDeviceProperties);
		cgpu->device_data = initVulkan(cgpu, physicalDeviceProperties.deviceName, strlen(physicalDeviceProperties.deviceName), hash_len_bits, throttled, copy_only);
		if (!cgpu->device_data) {
			applog(LOG_ERR, "Failed to init GPU, disabling device %d", cgpu->id);
			cgpu->deven = DEV_DISABLED;
			cgpu->status = LIFE_NOSTART;
			return false;
		}

		applog(LOG_INFO, "initVulkan() finished. Found %s", physicalDeviceProperties.deviceName);
	}
	return true;
}

static bool vulkan_init(struct cgpu_info *cgpu)
{
	cgpu->status = LIFE_WELL;
	return true;
}

static int vulkan_scrypt_positions(
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
	uint64_t *hashes_computed)
{
	cgpu->busy = 1;
	if (hashes_computed) {
		*hashes_computed = 0;
	}

	if (vulkan_prepare(cgpu, N, r, p, hash_len_bits, 0 != (options & SPACEMESH_API_THROTTLED_MODE), false))
	{
		_vulkanState *state = (_vulkanState *)cgpu->device_data;
		AlgorithmParams params;

		gettimeofday(tv_start, NULL);

		uint64_t n = start_position;
		size_t positions = end_position - start_position + 1;
		uint64_t chunkSize = (cgpu->thread_concurrency * hash_len_bits) / 8;
		uint64_t outLength = ((end_position - start_position + 1) * hash_len_bits + 7) / 8;
		uint64_t computedPositions = 0;
		uint8_t *out = output;

		// transfer input to GPU
		char *ptr = NULL;
		uint64_t tfxOrigin = state->memParamsSize + state->memConstantSize;
		CHECK_RESULT(gVulkan.vkMapMemory(state->vkDevice, state->gpuSharedMemory, tfxOrigin, state->memInputSize, 0, (void **)&ptr), "vkMapMemory", 0);
		memcpy(ptr, (const void*)pdata, 72);
		gVulkan.vkUnmapMemory(state->vkDevice, state->gpuSharedMemory);

		params.N = N;
		params.hash_len_bits = hash_len_bits;

		const uint64_t delay = 5ULL * 1000ULL * 1000ULL * 1000ULL;

		tfxOrigin = state->memParamsSize + state->memConstantSize + state->memInputSize;

		do {
			params.global_work_offset = n;
	
			CHECK_RESULT(gVulkan.vkMapMemory(state->vkDevice, state->gpuSharedMemory, state->memConstantSize, state->memParamsSize, 0, (void **)&ptr), "vkMapMemory", 0);
			memcpy(ptr, (const void*)&params, sizeof(params));
			gVulkan.vkUnmapMemory(state->vkDevice, state->gpuSharedMemory);

			n += cgpu->thread_concurrency;
			computedPositions += cgpu->thread_concurrency;
#if 0
			VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &state->commandBuffer, 1, &state->semaphore };
			CHECK_RESULT(gVulkan.vkQueueSubmit(state->queue, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit", 0);
			CHECK_RESULT(gVulkan.vkQueueWaitIdle(state->queue), "vkQueueWaitIdle", 0);
#else
			CHECK_RESULT(gVulkan.vkResetFences(state->vkDevice, 1, &state->fence), "vkResetFences", 0);
			VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &state->commandBuffer, 0, 0 };
			CHECK_RESULT(gVulkan.vkQueueSubmit(state->queue, 1, &submitInfo, state->fence), "vkQueueSubmit", 0);
			VkResult res;
			do {
				uint64_t delay = 5ULL * 1000ULL * 1000ULL * 1000ULL;
				res = gVulkan.vkWaitForFences(state->vkDevice, 1, &state->fence, VK_TRUE, delay);
			} while (res == VK_TIMEOUT);
			gVulkan.vkResetFences(state->vkDevice, 1, &state->fence);
#endif

			uint32_t length = (uint32_t)min(chunkSize, outLength);

			CHECK_RESULT(gVulkan.vkMapMemory(state->vkDevice, state->gpuSharedMemory, tfxOrigin, state->memOutputSize, 0, (void **)&ptr), "vkMapMemory", 0);
			memcpy(out, ptr, length);
			gVulkan.vkUnmapMemory(state->vkDevice, state->gpuSharedMemory);

			out += length;
			outLength -= length;
			positions -= cgpu->thread_concurrency;

		} while (n <= end_position && !g_spacemesh_api_abort_flag);

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

static int64_t vulkan_hash(struct cgpu_info *cgpu, uint8_t *pdata, uint8_t *output)
{
	cgpu->busy = 1;

	if (vulkan_prepare(cgpu, 512, 1, 1, 256, false, false))
	{
		_vulkanState *state = (_vulkanState *)cgpu->device_data;
		AlgorithmParams params;

		cgpu->thread_concurrency = 128;

		uint64_t chunkSize = 32 * cgpu->thread_concurrency;
		uint64_t outLength = chunkSize;
		uint8_t *out = output;

		// transfer input to GPU
		char *ptr = NULL;
		uint64_t tfxOrigin = state->memParamsSize + state->memConstantSize;
		CHECK_RESULT(gVulkan.vkMapMemory(state->vkDevice, state->gpuSharedMemory, tfxOrigin, state->memInputSize, 0, (void **)&ptr), "vkMapMemory", 0);
		memcpy(ptr, (const void*)pdata, 72);
		gVulkan.vkUnmapMemory(state->vkDevice, state->gpuSharedMemory);

		params.N = 512;
		params.hash_len_bits = 256;
		params.global_work_offset = 0;

		const uint64_t delay = 5ULL * 1000ULL * 1000ULL * 1000ULL;

		tfxOrigin = state->memParamsSize + state->memConstantSize + state->memInputSize;

		CHECK_RESULT(gVulkan.vkMapMemory(state->vkDevice, state->gpuSharedMemory, state->memConstantSize, state->memParamsSize, 0, (void **)&ptr), "vkMapMemory", 0);
		memcpy(ptr, (const void*)&params, sizeof(params));
		gVulkan.vkUnmapMemory(state->vkDevice, state->gpuSharedMemory);

#if 0
		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &state->commandBuffer, 1, &state->semaphore };
		CHECK_RESULT(gVulkan.vkQueueSubmit(state->queue, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit", 0);
		CHECK_RESULT(gVulkan.vkQueueWaitIdle(state->queue), "vkQueueWaitIdle", 0);
#else
		CHECK_RESULT(gVulkan.vkResetFences(state->vkDevice, 1, &state->fence), "vkResetFences", 0);
		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &state->commandBuffer, 0, 0 };
		CHECK_RESULT(gVulkan.vkQueueSubmit(state->queue, 1, &submitInfo, state->fence), "vkQueueSubmit", 0);
		VkResult res;
		do {
			uint64_t delay = 5ULL * 1000ULL * 1000ULL * 1000ULL;
			res = gVulkan.vkWaitForFences(state->vkDevice, 1, &state->fence, VK_TRUE, delay);
		} while (res == VK_TIMEOUT);
		gVulkan.vkResetFences(state->vkDevice, 1, &state->fence);
#endif

		uint32_t length = (uint32_t)min(chunkSize, outLength);

		CHECK_RESULT(gVulkan.vkMapMemory(state->vkDevice, state->gpuSharedMemory, tfxOrigin, state->memOutputSize, 0, (void **)&ptr), "vkMapMemory", 0);
		memcpy(out, ptr, length);
		gVulkan.vkUnmapMemory(state->vkDevice, state->gpuSharedMemory);

		cgpu->busy = 0;

		return 128;
	}

	cgpu->busy = 0;

	return SPACEMESH_API_ERROR;
}

static int64_t vulkan_bit_stream(struct cgpu_info *cgpu, uint8_t *hashes, uint64_t count, uint8_t *output, uint32_t hash_len_bits)
{
	cgpu->busy = 1;

	if (vulkan_prepare(cgpu, 512, 1, 1, hash_len_bits, false, true))
	{
		_vulkanState *state = (_vulkanState *)cgpu->device_data;
		AlgorithmParams params;

		cgpu->thread_concurrency = 128;

		uint64_t chunkSize = (cgpu->thread_concurrency * hash_len_bits) / 8;
		uint64_t outLength = chunkSize;
		uint8_t *out = output;

		// transfer input to GPU
		char *ptr = NULL;
		uint64_t tfxOrigin = state->memConstantSize + state->memParamsSize + state->memInputSize + state->memOutputSize;
		CHECK_RESULT(gVulkan.vkMapMemory(state->vkDevice, state->gpuSharedMemory, tfxOrigin, state->memOutputSize, 0, (void **)&ptr), "vkMapMemory", 0);
		memcpy(ptr, (const void*)hashes, 32 * 128);
		gVulkan.vkUnmapMemory(state->vkDevice, state->gpuSharedMemory);

		params.N = 512;
		params.hash_len_bits = hash_len_bits;
		params.global_work_offset = 0;

		const uint64_t delay = 5ULL * 1000ULL * 1000ULL * 1000ULL;

		tfxOrigin = state->memParamsSize + state->memConstantSize + state->memInputSize;

		CHECK_RESULT(gVulkan.vkMapMemory(state->vkDevice, state->gpuSharedMemory, state->memConstantSize, state->memParamsSize, 0, (void **)&ptr), "vkMapMemory", 0);
		memcpy(ptr, (const void*)&params, sizeof(params));
		gVulkan.vkUnmapMemory(state->vkDevice, state->gpuSharedMemory);

#if 0
		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &state->commandBuffer, 1, &state->semaphore };
		CHECK_RESULT(gVulkan.vkQueueSubmit(state->queue, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit", 0);
		CHECK_RESULT(gVulkan.vkQueueWaitIdle(state->queue), "vkQueueWaitIdle", 0);
#else
		CHECK_RESULT(gVulkan.vkResetFences(state->vkDevice, 1, &state->fence), "vkResetFences", 0);
		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &state->commandBuffer, 0, 0 };
		CHECK_RESULT(gVulkan.vkQueueSubmit(state->queue, 1, &submitInfo, state->fence), "vkQueueSubmit", 0);
		VkResult res;
		do {
			uint64_t delay = 5ULL * 1000ULL * 1000ULL * 1000ULL;
			res = gVulkan.vkWaitForFences(state->vkDevice, 1, &state->fence, VK_TRUE, delay);
		} while (res == VK_TIMEOUT);
		gVulkan.vkResetFences(state->vkDevice, 1, &state->fence);
#endif

		uint32_t length = (uint32_t)min(chunkSize, outLength);

		CHECK_RESULT(gVulkan.vkMapMemory(state->vkDevice, state->gpuSharedMemory, tfxOrigin, state->memOutputSize, 0, (void **)&ptr), "vkMapMemory", 0);
		memcpy(out, ptr, length);
		gVulkan.vkUnmapMemory(state->vkDevice, state->gpuSharedMemory);

		cgpu->busy = 0;

		return chunkSize;
	}

	cgpu->busy = 0;

	return SPACEMESH_API_ERROR;
}

static void vulkan_shutdown(struct cgpu_info *cgpu)
{
	_vulkanState *vulkanState = (_vulkanState *)cgpu->device_data;
	if (!vulkanState) {
		gVulkan.vkDestroyPipelineLayout(vulkanState->vkDevice, vulkanState->pipelineLayout, NULL);
		gVulkan.vkDestroyDescriptorSetLayout(vulkanState->vkDevice, vulkanState->descriptorSetLayout, NULL);
		gVulkan.vkDestroyPipeline(vulkanState->vkDevice, vulkanState->pipeline, NULL);

		gVulkan.vkDestroyBuffer(vulkanState->vkDevice, vulkanState->gpu_params, NULL);
		gVulkan.vkDestroyBuffer(vulkanState->vkDevice, vulkanState->gpu_constants, NULL);
		gVulkan.vkDestroyBuffer(vulkanState->vkDevice, vulkanState->outputBuffer[0], NULL);
		gVulkan.vkDestroyBuffer(vulkanState->vkDevice, vulkanState->outputBuffer[1], NULL);
		gVulkan.vkDestroyBuffer(vulkanState->vkDevice, vulkanState->CLbuffer0, NULL);
		gVulkan.vkDestroyBuffer(vulkanState->vkDevice, vulkanState->padbuffer8, NULL);

		gVulkan.vkFreeCommandBuffers(vulkanState->vkDevice, vulkanState->commandPool, 1, &vulkanState->commandBuffer);
		gVulkan.vkDestroyCommandPool(vulkanState->vkDevice, vulkanState->commandPool, NULL);
		gVulkan.vkFreeDescriptorSets(vulkanState->vkDevice, vulkanState->descriptorPool, 1, &vulkanState->descriptorSet);
		gVulkan.vkDestroyDescriptorPool(vulkanState->vkDevice, vulkanState->descriptorPool, NULL);
		gVulkan.vkDestroyShaderModule(vulkanState->vkDevice, vulkanState->shaderModule, NULL);

		gVulkan.vkDestroySemaphore(vulkanState->vkDevice, vulkanState->semaphore, NULL);
		gVulkan.vkDestroyFence(vulkanState->vkDevice, vulkanState->fence, NULL);

		gVulkan.vkFreeMemory(vulkanState->vkDevice, vulkanState->gpuLocalMemory, NULL);
		gVulkan.vkFreeMemory(vulkanState->vkDevice, vulkanState->gpuSharedMemory, NULL);

		gVulkan.vkDestroyDevice(vulkanState->vkDevice, NULL);

		free(cgpu->device_data);
		cgpu->device_data = NULL;
	}
}

struct device_drv vulkan_drv = {
	SPACEMESH_API_VULKAN,
	"vulkan",
	"GPU",
	vulkan_detect,
	reinit_vulkan_device,
	vulkan_init,
	vulkan_scrypt_positions,
	{ vulkan_hash, vulkan_bit_stream },
	vulkan_shutdown
};
#endif
