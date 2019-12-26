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
	VkCommandBuffer vkCommandBuffer;
	VkDescriptorPool descriptorPool;
	VkShaderModule shader_module;
	VkDescriptorSetLayout descriptorSetLayout;
	VkQueue queue;
	VkFence drawFence;
	VkMemoryBarrier memoryBarrier;

	uint32_t alignment;
	bool commandBufferFilled;
} _vulkanState;

typedef struct GpuConstants {
	uint64_t keccakf_rndc[24];
	uint32_t keccakf_rotc[24];
	uint32_t keccakf_piln[24];
} GpuConstants;

GpuConstants gpuConstants;

typedef struct Params {
	uint64_t global_work_offset;
	uint32_t memorySize;
	uint32_t iterations;
	uint32_t mask;
	uint32_t threads;
} Params;

struct device_drv vulkan_drv;

static uint64_t alignBuffer(uint64_t size, uint64_t align)
{
	if (align == 1) {
		return size;
	}
	else {
		return (size + align - 1)&(~(align - 1));
	}
}

static _vulkanState *initVulkan(struct cgpu_info *cgpu, char *name, size_t nameSize, uint32_t hash_len_bits, bool throttled)
{
	_vulkanState *state = calloc(1, sizeof(_vulkanState));

	uint32_t computeQueueFamillyIndex = getComputeQueueFamillyIndex(cgpu->driver_id);

	state->deviceId = cgpu->driver_id;
	state->vkDevice = createDevice(cgpu->driver_id, computeQueueFamillyIndex);

	// Get memory alignment
	VkDeviceMemory tmpMem = allocateGPUMemory(state->deviceId, state->vkDevice, 1024, true, true);
	VkBuffer tmpBuf = createBuffer(state->vkDevice, computeQueueFamillyIndex, tmpMem, 256, 0);
	state->alignment = getBufferMemoryRequirements(state->vkDevice, tmpBuf);
	vkDestroyBuffer(state->vkDevice, tmpBuf, NULL);
	vkFreeMemory(state->vkDevice, tmpMem, NULL);

	// compute memory requirements
	applog(LOG_NOTICE, "GPU %d: selecting lookup gap of 4", cgpu->driver_id);
	cgpu->lookup_gap = 4;

	unsigned int bsize = 1024;
	size_t ipt = (bsize / cgpu->lookup_gap + (bsize % cgpu->lookup_gap > 0));

	// if we do not have TC and we do not have BS, then calculate some conservative numbers
	if (!cgpu->buffer_size) {
		unsigned int base_alloc = (int)(cgpu->gpu_max_alloc * 88 / 100 / 1024 / 1024 / 8) * 8 * 1024 * 1024;
		cgpu->thread_concurrency = (uint32_t)(base_alloc / 128 / ipt);
		cgpu->buffer_size = base_alloc / 1024 / 1024;
		applog(LOG_DEBUG, "88%% Max Allocation: %u", base_alloc);
		applog(LOG_NOTICE, "GPU %d: selecting buffer_size of %zu", cgpu->driver_id, cgpu->buffer_size);
	}

	if (cgpu->buffer_size) {
		// use the buffer-size to overwrite the thread-concurrency
		cgpu->thread_concurrency = (int)((cgpu->buffer_size * 1024 * 1024) / ipt / 128);
		applog(LOG_DEBUG, "GPU %d: setting thread_concurrency to %d based on buffer size %d and lookup gap %d", cgpu->driver_id, (int)(cgpu->thread_concurrency), (int)(cgpu->buffer_size), (int)(cgpu->lookup_gap));
	}

	uint32_t chunkSize = (cgpu->thread_concurrency * hash_len_bits + 7) / 8;

	uint64_t bufSize = alignBuffer(cgpu->buffer_size, state->alignment);
	uint64_t memConstantSize = alignBuffer(sizeof(GpuConstants), state->alignment);
	uint64_t memInputSize = alignBuffer(72, state->alignment);
	uint64_t memOutputSize = alignBuffer(chunkSize, state->alignment);
	uint64_t shared_memory_size = memConstantSize + memInputSize + 2 * memOutputSize;

	state->gpuLocalMemory = allocateGPUMemory(state->deviceId, state->vkDevice, cgpu->buffer_size, true, true);
	state->gpuSharedMemory = allocateGPUMemory(state->deviceId, state->vkDevice, shared_memory_size, false, true);

	// create the internal local buffers
	state->padbuffer8 = createBuffer(state->vkDevice, computeQueueFamillyIndex, state->gpuLocalMemory, cgpu->buffer_size, 0);

	// create the CPU shared buffers
	uint64_t o = 0;
	state->gpu_constants = createBuffer(state->vkDevice, computeQueueFamillyIndex, state->gpuSharedMemory, memConstantSize, o);
	o += memConstantSize;
	state->CLbuffer0 = createBuffer(state->vkDevice, computeQueueFamillyIndex, state->gpuSharedMemory, memInputSize, o);
	o += memInputSize;
	state->outputBuffer[0] = createBuffer(state->vkDevice, computeQueueFamillyIndex, state->gpuSharedMemory, memOutputSize, o);
	o += memOutputSize;
	state->outputBuffer[1] = createBuffer(state->vkDevice, computeQueueFamillyIndex, state->gpuSharedMemory, memOutputSize, o);
	o += memOutputSize;
#if 0
	state->pipelineLayout = bindBuffers(state->vkDevice, state->descriptorSet, state->descriptorPool, state->descriptorSetLayout,
		state->gpu_scratchpadsBuffer1, state->gpu_scratchpadsBuffer2, state->gpu_statesBuffer,
		state->gpu_branchesBuffer, state->gpu_params, state->gpu_constants, state->gpu_inputsBuffer, state->gpu_outputBuffer, state->gpu_debugBuffer);
#endif
	// Transfer constants to GPU
	void *ptr = NULL;
	CHECK_RESULT(vkMapMemory(state->vkDevice, state->gpuSharedMemory, 0, memConstantSize, 0, (void **)&ptr), "vkMapMemory", NULL);
	memcpy(ptr, (const void*)&gpuConstants, sizeof(GpuConstants));
	vkUnmapMemory(state->vkDevice, state->gpuSharedMemory);

	initCommandPool(state->vkDevice, computeQueueFamillyIndex, &state->commandPool);
	state->vkCommandBuffer = createCommandBuffer(state->vkDevice, state->commandPool);

	vkGetDeviceQueue(state->vkDevice, computeQueueFamillyIndex, 0, &state->queue);
	state->memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	state->memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	state->memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	state->commandBufferFilled = false;

	VkFenceCreateInfo fenceInfo;
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.pNext = NULL;
	fenceInfo.flags = 0;
	CHECK_RESULT(vkCreateFence(state->vkDevice, &fenceInfo, NULL, &state->drawFence), "vkCreateFence", NULL);

	return state;
}

static int vulkan_detect(struct cgpu_info *gpus, int *active)
{
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
		VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,								// stype
		0,																	// pNext
		0,																	// flags
		&applicationInfo,													// pApplicationInfo
		0,																	// enabledLayerCount
		NULL,															// ppEnabledLayerNames
		0,																	// enabledExtensionCount
		NULL 															// ppEnabledExtensionNames
	};

	CHECK_RESULT(vkCreateInstance(&instanceCreateInfo, 0, &gInstance), "vkCreateInstance", 0);

	gPhysicalDeviceCount = 0;
	CHECK_RESULT(vkEnumeratePhysicalDevices(gInstance, &gPhysicalDeviceCount, 0), "vkEnumeratePhysicalDevices", 0);
	if (gPhysicalDeviceCount > 0) {
		gPhysicalDevices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * gPhysicalDeviceCount);
		memset(gPhysicalDevices, 0, sizeof(VkPhysicalDevice) * gPhysicalDeviceCount);
		CHECK_RESULT(vkEnumeratePhysicalDevices(gInstance, &gPhysicalDeviceCount, gPhysicalDevices), "vkEnumeratePhysicalDevices", 0);
		for (unsigned i = 0; i < gPhysicalDeviceCount; i++) {
			struct cgpu_info *cgpu = &gpus[*active];

			cgpu->id = *active;
			cgpu->pci_bus_id = 0;
			cgpu->pci_device_id = 0;
			cgpu->deven = DEV_ENABLED;
			cgpu->platform = 0;
			cgpu->drv = &vulkan_drv;
			cgpu->driver_id = i;

			*active += 1;

			have_vulkan = true;
		}
	} else {
		applog(LOG_ERR, "No graphic cards were found by Vulkan. Use Adrenalin not Crimson and check your drivers with VulkanInfo.");
	}

	memcpy(gpuConstants.keccakf_rndc, keccak_round_constants, sizeof(keccak_round_constants));
	memcpy(gpuConstants.keccakf_rotc, keccak_rotc, sizeof(keccak_rotc));
	memcpy(gpuConstants.keccakf_piln, keccak_piln, sizeof(keccak_piln));

	return gPhysicalDeviceCount;
}

static void reinit_vulkan_device(struct cgpu_info *gpu)
{
}

static void vulkan_shutdown(struct cgpu_info *cgpu);

static bool vulkan_prepare(struct cgpu_info *cgpu, unsigned N, uint32_t r, uint32_t p, uint32_t hash_len_bits, bool throttled)
{
	if (N != cgpu->N || r != cgpu->r || p != cgpu->p) {
		if (cgpu->device_data) {
			vulkan_shutdown(cgpu);
		}

		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(gPhysicalDevices[cgpu->driver_id], &physicalDeviceProperties);
		cgpu->device_data = initVulkan(cgpu, physicalDeviceProperties.deviceName, strlen(physicalDeviceProperties.deviceName), hash_len_bits, throttled);
		if (!cgpu->device_data) {
			applog(LOG_ERR, "Failed to init GPU, disabling device %d", cgpu->id);
			cgpu->deven = DEV_DISABLED;
			cgpu->status = LIFE_NOSTART;
			return false;
		}

		cgpu->N = N;
		cgpu->r = r;
		cgpu->p = p;

		applog(LOG_INFO, "initVulkan() finished. Found %s", physicalDeviceProperties.deviceName);
	}
	return true;
}

static bool vulkan_init(struct cgpu_info *cgpu)
{
	cgpu->status = LIFE_WELL;
	return true;
}

static int64_t vulkan_scrypt_positions(struct cgpu_info *cgpu, uint8_t *pdata, uint64_t start_position, uint64_t end_position, uint8_t hash_len_bits, uint32_t options, uint8_t *output, uint32_t N, uint32_t r, uint32_t p, struct timeval *tv_start, struct timeval *tv_end)
{
	cgpu->busy = 1;
	if (vulkan_prepare(cgpu, N, r, p, hash_len_bits, 0 != (options & SPACEMESH_API_THROTTLED_MODE)))
	{
		_vulkanState *vulkanState = (_vulkanState *)cgpu->device_data;

		gettimeofday(tv_start, NULL);

		uint64_t n = start_position;
		size_t positions = end_position - start_position + 1;
		uint64_t chunkSize = (cgpu->thread_concurrency * hash_len_bits) / 8;
		uint64_t outLength = ((end_position - start_position + 1) * hash_len_bits + 7) / 8;

		do {
			n += cgpu->thread_concurrency;

			output += chunkSize;
			outLength -= chunkSize;
			positions -= cgpu->thread_concurrency;

		} while (n <= end_position && !abort_flag);

		gettimeofday(tv_end, NULL);

		cgpu->busy = 0;
		return 0;
	}

	cgpu->busy = 0;
	return -1;
}

static void vulkan_shutdown(struct cgpu_info *cgpu)
{
	_vulkanState *vulkanState = (_vulkanState *)cgpu->device_data;
	if (!vulkanState) {
		vkFreeMemory(vulkanState->vkDevice, vulkanState->gpuLocalMemory, NULL);
		vkFreeMemory(vulkanState->vkDevice, vulkanState->gpuSharedMemory, NULL);

		vkDestroyPipeline(vulkanState->vkDevice, vulkanState->pipeline, NULL);

		vkDestroyPipelineLayout(vulkanState->vkDevice, vulkanState->pipelineLayout, NULL);

		vkDestroyBuffer(vulkanState->vkDevice, vulkanState->gpu_params, NULL);
		vkDestroyBuffer(vulkanState->vkDevice, vulkanState->gpu_constants, NULL);
		vkDestroyBuffer(vulkanState->vkDevice, vulkanState->outputBuffer[0], NULL);
		vkDestroyBuffer(vulkanState->vkDevice, vulkanState->outputBuffer[1], NULL);
		vkDestroyBuffer(vulkanState->vkDevice, vulkanState->CLbuffer0, NULL);
		vkDestroyBuffer(vulkanState->vkDevice, vulkanState->padbuffer8, NULL);

		vkDestroyCommandPool(vulkanState->vkDevice, vulkanState->commandPool, NULL);
		vkDestroyDescriptorPool(vulkanState->vkDevice, vulkanState->descriptorPool, NULL);
		vkDestroyShaderModule(vulkanState->vkDevice, vulkanState->shader_module, NULL);
		vkDestroyDescriptorSetLayout(vulkanState->vkDevice, vulkanState->descriptorSetLayout, NULL);

		free(cgpu->device_data);
		cgpu->device_data = NULL;
	}
}

struct device_drv vulkan_drv = {
	SPACEMESH_API_OPENCL,
	"vulkan",
	"GPU",
	vulkan_detect,
	reinit_vulkan_device,
	vulkan_init,
	vulkan_scrypt_positions,
	vulkan_shutdown
};
#endif
