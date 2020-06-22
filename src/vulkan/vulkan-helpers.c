#include "api.h"
#include "api_internal.h"
#include <string.h>

/**********************************************/

#ifdef HAVE_VULKAN
#include "vulkan-helpers.h"

Vulkan gVulkan = { NULL };
VkInstance gInstance = NULL;
VkPhysicalDevice* gPhysicalDevices = NULL;
uint32_t gPhysicalDeviceCount = 0;

#if defined(_MSC_VER) || defined(_WIN32) || defined(_WIN64)
#define	LOAD_VULKAN_FUNCTION(func)	gVulkan.func = (PFN_##func)GetProcAddress(gVulkan.library, #func); if (!gVulkan.func) return -1
#else
#include <dlfcn.h>
#define	LOAD_VULKAN_FUNCTION(func)	gVulkan.func = (PFN_##func)dlsym(gVulkan.library, #func); if (!gVulkan.func) return -1
#endif

int initVulkanLibrary()
{
	if (!gVulkan.library) {
#if defined( __linux__ )
		gVulkan.library = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
		if (gVulkan.library == nullptr) {
			gVulkan.library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
		}
#elif defined( __APPLE__ )
		gVulkan.library = dlopen("libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
#elif defined( _WIN32 )
		gVulkan.library = LoadLibraryA("vulkan-1.dll");
#else
#error unsupported platform
#endif
	}
	if (!gVulkan.library) {
		return -1;
	}

	LOAD_VULKAN_FUNCTION(vkCreateInstance);
	LOAD_VULKAN_FUNCTION(vkGetPhysicalDeviceQueueFamilyProperties);
	LOAD_VULKAN_FUNCTION(vkCreateDevice);
	LOAD_VULKAN_FUNCTION(vkGetPhysicalDeviceMemoryProperties);
	LOAD_VULKAN_FUNCTION(vkAllocateMemory);
	LOAD_VULKAN_FUNCTION(vkCreateBuffer);
	LOAD_VULKAN_FUNCTION(vkBindBufferMemory);
	LOAD_VULKAN_FUNCTION(vkCreateDescriptorSetLayout);
	LOAD_VULKAN_FUNCTION(vkCreatePipelineLayout);
	LOAD_VULKAN_FUNCTION(vkCreateDescriptorPool);
	LOAD_VULKAN_FUNCTION(vkAllocateDescriptorSets);
	LOAD_VULKAN_FUNCTION(vkUpdateDescriptorSets);
	LOAD_VULKAN_FUNCTION(vkGetBufferMemoryRequirements);
	LOAD_VULKAN_FUNCTION(vkCreateShaderModule);
	LOAD_VULKAN_FUNCTION(vkCreateComputePipelines);
	LOAD_VULKAN_FUNCTION(vkDestroyBuffer);
	LOAD_VULKAN_FUNCTION(vkFreeMemory);
	LOAD_VULKAN_FUNCTION(vkGetDeviceQueue);
	LOAD_VULKAN_FUNCTION(vkMapMemory);
	LOAD_VULKAN_FUNCTION(vkUnmapMemory);
	LOAD_VULKAN_FUNCTION(vkCreateCommandPool);
	LOAD_VULKAN_FUNCTION(vkAllocateCommandBuffers);
	LOAD_VULKAN_FUNCTION(vkCreateSemaphore);
	LOAD_VULKAN_FUNCTION(vkBeginCommandBuffer);
	LOAD_VULKAN_FUNCTION(vkCmdBindPipeline);
	LOAD_VULKAN_FUNCTION(vkCmdBindDescriptorSets);
	LOAD_VULKAN_FUNCTION(vkCmdDispatch);
	LOAD_VULKAN_FUNCTION(vkEndCommandBuffer);
	LOAD_VULKAN_FUNCTION(vkEnumeratePhysicalDevices);
	LOAD_VULKAN_FUNCTION(vkGetPhysicalDeviceProperties);
	LOAD_VULKAN_FUNCTION(vkQueueSubmit);
	LOAD_VULKAN_FUNCTION(vkQueueWaitIdle);
	LOAD_VULKAN_FUNCTION(vkDestroyPipelineLayout);
	LOAD_VULKAN_FUNCTION(vkDestroyDescriptorSetLayout);
	LOAD_VULKAN_FUNCTION(vkDestroyPipeline);
	LOAD_VULKAN_FUNCTION(vkDestroyCommandPool);
	LOAD_VULKAN_FUNCTION(vkDestroyDescriptorPool);
	LOAD_VULKAN_FUNCTION(vkDestroyShaderModule);
	LOAD_VULKAN_FUNCTION(vkDestroySemaphore);

	return 0;
}

int getComputeQueueFamillyIndex(uint32_t index)
{
	if (index >= gPhysicalDeviceCount) {
		applog(LOG_ERR, "Card index %u not found\n", index);
		return -1;
	}
	uint32_t queueFamilyPropertiesCount = 0;
	gVulkan.vkGetPhysicalDeviceQueueFamilyProperties(gPhysicalDevices[index], &queueFamilyPropertiesCount, 0);
	VkQueueFamilyProperties* const queueFamilyProperties = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * queueFamilyPropertiesCount);
	gVulkan.vkGetPhysicalDeviceQueueFamilyProperties(gPhysicalDevices[index], &queueFamilyPropertiesCount, queueFamilyProperties);
	int ret = -1;
	for (unsigned int i = 0; i< queueFamilyPropertiesCount; i++) {
		if (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			ret = i;
		}
	}
	free(queueFamilyProperties);
	return ret;
}

VkDevice createDevice(int index, uint32_t computeQueueFamillyIndex)
{
	const float queuePrioritory = 1.0f;
	const VkDeviceQueueCreateInfo deviceQueueCreateInfo = {
		VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		0,
		0,
		computeQueueFamillyIndex,
		1,
		&queuePrioritory
	};

	static VkPhysicalDeviceFeatures enabledFeatures;
	memset(&enabledFeatures, 0, sizeof(enabledFeatures));
#if 0
	enabledFeatures.shaderInt64 = VK_TRUE;
#endif
	const char * deviceExtensions[1];
	uint32_t extensionsCount = 0;
#if 0
	if (forceAMD) {
		extensionsCount = 1;
		deviceExtensions[0] = "VK_AMD_shader_info";
	}
#endif
	const VkDeviceCreateInfo deviceCreateInfo = {
		VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		0,
		0,
		1,
		&deviceQueueCreateInfo,
		0,
		0,
		extensionsCount,
		deviceExtensions,
		&enabledFeatures
	};

	VkDevice vulkanDevice;
	CHECK_RESULT(gVulkan.vkCreateDevice(gPhysicalDevices[index], &deviceCreateInfo, 0, &vulkanDevice), "vkCreateDevice", NULL);

	return vulkanDevice;
}

VkDeviceMemory allocateGPUMemory(int index,  VkDevice vkDevice, const VkDeviceSize memorySize, char isLocal, bool isFatal)
{
	VkPhysicalDeviceMemoryProperties properties;
	gVulkan.vkGetPhysicalDeviceMemoryProperties(gPhysicalDevices[index], &properties);

	// set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
	uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

	VkMemoryPropertyFlags flag;
	if (isLocal) flag = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	else flag = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT /*| VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT*/;
	for (uint32_t k = 0; k < properties.memoryTypeCount; k++) {
		if ((flag & properties.memoryTypes[k].propertyFlags) == flag && memorySize < properties.memoryHeaps[properties.memoryTypes[k].heapIndex].size) {
			memoryTypeIndex = k;
			break;
		}
	}

	VkResult ret = (memoryTypeIndex == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS);
	if (ret != VK_SUCCESS) {
		applog(LOG_ERR,	"Cannot allocated %u kB GPU memory type for GPU index %u\n", (unsigned)(memorySize / 1024), index);
		return NULL;
	}

	const VkMemoryAllocateInfo memoryAllocateInfo = {
		VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		0,
		memorySize,
		memoryTypeIndex
	};

	VkDeviceMemory memory;
	CHECK_RESULT(gVulkan.vkAllocateMemory(vkDevice, &memoryAllocateInfo, 0, &memory), "vkAllocateMemory", NULL);

	return memory;
}

VkBuffer createBuffer(VkDevice vkDevice, uint32_t computeQueueFamillyIndex, VkDeviceMemory memory, VkDeviceSize bufferSize, VkDeviceSize offset)
{
	// 4Gb limit on AMD and Nvidia
	if (bufferSize >= 0x100000000) bufferSize = 0xffffffff;

	const VkBufferCreateInfo bufferCreateInfo = {
		VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		0,
		0,
		bufferSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		VK_SHARING_MODE_EXCLUSIVE,
		1,
		&computeQueueFamillyIndex
	};

	VkBuffer buffer;
	CHECK_RESULT(gVulkan.vkCreateBuffer(vkDevice, &bufferCreateInfo, 0, &buffer), "vkCreateBuffer", NULL);
	CHECK_RESULT(gVulkan.vkBindBufferMemory(vkDevice, buffer, memory, offset), "vkBindBufferMemory", NULL);

	return buffer;
}

VkPipelineLayout bindBuffers(VkDevice vkDevice, VkDescriptorSet *descriptorSet, VkDescriptorPool *descriptorPool, VkDescriptorSetLayout *descriptorSetLayout, VkBuffer b0, VkBuffer b1, VkBuffer b2, VkBuffer b3, VkBuffer b4, VkBuffer b5)
 {
	VkPipelineLayout pipelineLayout;
	const uint32_t nBuffers = 6;
	VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[6] = {
		{ 0,		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,		1,		VK_SHADER_STAGE_COMPUTE_BIT,		0 },
		{ 1,		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,		1,		VK_SHADER_STAGE_COMPUTE_BIT,		0 },
		{ 2,		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,		1,		VK_SHADER_STAGE_COMPUTE_BIT,		0 },
		{ 3,		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,		1,		VK_SHADER_STAGE_COMPUTE_BIT,		0 },
		{ 4,		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,		1,		VK_SHADER_STAGE_COMPUTE_BIT,		0 },
		{ 5,		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,		1,		VK_SHADER_STAGE_COMPUTE_BIT,		0 }
	};

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		0,
		0,
		nBuffers,
		descriptorSetLayoutBindings
	};

	CHECK_RESULT(gVulkan.vkCreateDescriptorSetLayout(vkDevice, &descriptorSetLayoutCreateInfo, 0, descriptorSetLayout), "vkCreateDescriptorSetLayout", NULL);

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
		VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		0,
		0,
		1,
		descriptorSetLayout,
		0,
		0,
	};

	CHECK_RESULT(gVulkan.vkCreatePipelineLayout(vkDevice, &pipelineLayoutCreateInfo, 0, &pipelineLayout), "vkCreatePipelineLayout", NULL);

	VkDescriptorPoolSize descriptorPoolSize = {
		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		nBuffers
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
		VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		0,
		0,
		1,
		1,
		&descriptorPoolSize
	};

	CHECK_RESULT(gVulkan.vkCreateDescriptorPool(vkDevice, &descriptorPoolCreateInfo, 0, descriptorPool), "vkCreateDescriptorPool", NULL);

	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		0,
		*descriptorPool,
		1,
		descriptorSetLayout
	};
	CHECK_RESULT(gVulkan.vkAllocateDescriptorSets(vkDevice, &descriptorSetAllocateInfo, descriptorSet), "vkAllocateDescriptorSets", NULL);

	VkDescriptorBufferInfo descriptorBufferInfo0 = { b0, 0, 	VK_WHOLE_SIZE };
	VkDescriptorBufferInfo descriptorBufferInfo1 = { b1, 0, 	VK_WHOLE_SIZE };
	VkDescriptorBufferInfo descriptorBufferInfo2 = { b2, 0, 	VK_WHOLE_SIZE };
	VkDescriptorBufferInfo descriptorBufferInfo3 = { b3, 0, 	VK_WHOLE_SIZE };
	VkDescriptorBufferInfo descriptorBufferInfo4 = { b4, 0, 	VK_WHOLE_SIZE };
	VkDescriptorBufferInfo descriptorBufferInfo5 = { b5, 0, 	VK_WHOLE_SIZE };
	
	VkWriteDescriptorSet writeDescriptorSet[6] = {
		{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,	0,	*descriptorSet,	0,	0,	1,	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,	0,	&descriptorBufferInfo0,	0 },
		{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,	0,	*descriptorSet,	1,	0,	1,	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,	0,	&descriptorBufferInfo1,	0 },
		{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,	0,	*descriptorSet,	2,	0,	1,	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,	0,	&descriptorBufferInfo2,	0 },
		{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,	0,	*descriptorSet,	3,	0,	1,	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,	0,	&descriptorBufferInfo3,	0 },
		{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,	0,	*descriptorSet,	4,	0,	1,	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,	0,	&descriptorBufferInfo4,	0 },
		{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,	0,	*descriptorSet,	5,	0,	1,	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,	0,	&descriptorBufferInfo5,	0 }
	};

	gVulkan.vkUpdateDescriptorSets(vkDevice, nBuffers, writeDescriptorSet, 0, 0);

	return pipelineLayout;
}

uint64_t getBufferMemoryRequirements(VkDevice vkDevice, VkBuffer b)
{
	VkMemoryRequirements req;
	gVulkan.vkGetBufferMemoryRequirements(vkDevice, b, &req);
	return req.alignment;
}

VkPipeline loadShader(VkDevice vkDevice, VkPipelineLayout pipelineLayout, VkShaderModule *shader_module, const char * spirv_file_name)
{
	uint32_t *shader;
	size_t shader_size;

	FILE *fp = fopen(spirv_file_name, "rb");
	if (fp == NULL) {
		applog(LOG_ERR, "SPIR-V program %s not found\n", spirv_file_name);
		return NULL;
	}
	fseek(fp, 0, SEEK_END);
	shader_size = (size_t)(ftell(fp) * sizeof(char));
	fseek(fp, 0, SEEK_SET);

	shader = (uint32_t*)malloc(shader_size + 1);
	memset(shader, 0, shader_size + 1);
	size_t read_size = fread(shader, sizeof(char), shader_size, fp);
	if (read_size != shader_size) {
		free(shader);
		applog(LOG_ERR, "Failed to read shader %s!\n", spirv_file_name);
		return NULL;
	}

	VkShaderModuleCreateInfo shaderModuleCreateInfo = {
		VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		0,
		0,
		shader_size,
		shader
	};

	CHECK_RESULT(gVulkan.vkCreateShaderModule(vkDevice, &shaderModuleCreateInfo, 0, shader_module), "vkCreateShaderModule", NULL);

	VkComputePipelineCreateInfo computePipelineCreateInfo = {
		VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		0,
		0,
		{
			VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			0,
			0,
			VK_SHADER_STAGE_COMPUTE_BIT,
			*shader_module,
			"main",
			0
		},
		pipelineLayout,
		0,
		0
	};

	VkPipeline pipeline;
	CHECK_RESULT(gVulkan.vkCreateComputePipelines(vkDevice, 0, 1, &computePipelineCreateInfo, 0, &pipeline), "vkCreateComputePipelines", NULL);

	free(shader);
	fclose(fp);
	return pipeline;
}

#endif

