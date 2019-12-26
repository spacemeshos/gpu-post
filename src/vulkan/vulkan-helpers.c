#include "api.h"
#include "api_internal.h"

/**********************************************/

#ifdef HAVE_VULKAN
#include "vulkan-helpers.h"

VkInstance gInstance = NULL;
VkPhysicalDevice* gPhysicalDevices = NULL;
uint32_t gPhysicalDeviceCount = 0;

int getComputeQueueFamillyIndex(uint32_t index)
{
	if (index >= gPhysicalDeviceCount) {
		applog(LOG_ERR, "Card index %u not found\n", index);
		return -1;
	}
	uint32_t queueFamilyPropertiesCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(gPhysicalDevices[index], &queueFamilyPropertiesCount, 0);
	VkQueueFamilyProperties* const queueFamilyProperties = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * queueFamilyPropertiesCount);
	vkGetPhysicalDeviceQueueFamilyProperties(gPhysicalDevices[index], &queueFamilyPropertiesCount, queueFamilyProperties);
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
	enabledFeatures.shaderInt64 = VK_TRUE;
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
	CHECK_RESULT(vkCreateDevice(gPhysicalDevices[index], &deviceCreateInfo, 0, &vulkanDevice), "vkCreateDevice", NULL);

	return vulkanDevice;
}

VkDeviceMemory allocateGPUMemory(int index,  VkDevice vkDevice, const VkDeviceSize memorySize, char isLocal, bool isFatal)
{
	VkPhysicalDeviceMemoryProperties properties;
	vkGetPhysicalDeviceMemoryProperties(gPhysicalDevices[index], &properties);

	// set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
	uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

	VkMemoryPropertyFlags flag;
	if (isLocal) flag = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	else flag = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT /*| VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT*/;
	for (uint32_t k = 0; k < properties.memoryTypeCount; k++) {
		if (properties.memoryTypes[k].propertyFlags == flag && memorySize < properties.memoryHeaps[properties.memoryTypes[k].heapIndex].size) {
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
	CHECK_RESULT(vkAllocateMemory(vkDevice, &memoryAllocateInfo, 0, &memory), "vkAllocateMemory", NULL);

	return memory;
}

void initCommandPool(VkDevice vkDevice, uint32_t computeQueueFamillyIndex, VkCommandPool *commandPool)
{
	VkCommandPoolCreateInfo commandPoolCreateInfo = {
		VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		0,
		VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		computeQueueFamillyIndex
	};

	CHECK_RESULT_NORET(vkCreateCommandPool(vkDevice, &commandPoolCreateInfo, 0, commandPool), "vkCreateCommandPool");
}

VkCommandBuffer createCommandBuffer(VkDevice vkDevice, VkCommandPool commandPool)
{
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
		VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		0,
		commandPool,
		VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		1
	};

	VkCommandBuffer commandBuffer;
	CHECK_RESULT(vkAllocateCommandBuffers(vkDevice, &commandBufferAllocateInfo, &commandBuffer), "vkAllocateCommandBuffers", NULL);

	return commandBuffer;
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
	CHECK_RESULT(vkCreateBuffer(vkDevice, &bufferCreateInfo, 0, &buffer), "vkCreateBuffer", NULL);
	CHECK_RESULT(vkBindBufferMemory(vkDevice, buffer, memory, offset), "vkBindBufferMemory", NULL);

	return buffer;
}

VkPipelineLayout bindBuffer(VkDevice vkDevice, VkDescriptorSet *descriptorSet, VkDescriptorPool *descriptorPool, VkDescriptorSetLayout *descriptorSetLayout, VkBuffer b0)
{
	VkPipelineLayout pipelineLayout;
	uint32_t nb_Buffers = 1;
	VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[1] = {
		{ 0,		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,		1,		VK_SHADER_STAGE_COMPUTE_BIT,		0 },
	};

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		0,
		0,
		nb_Buffers,
		descriptorSetLayoutBindings
	};

	CHECK_RESULT(vkCreateDescriptorSetLayout(vkDevice, &descriptorSetLayoutCreateInfo, 0, descriptorSetLayout), "vkCreateDescriptorSetLayout", NULL);

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
		VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		0,
		0,
		1,
		descriptorSetLayout,
		0,
		0
	};

	CHECK_RESULT(vkCreatePipelineLayout(vkDevice, &pipelineLayoutCreateInfo, 0, &pipelineLayout), "vkCreatePipelineLayout", NULL);

	VkDescriptorPoolSize descriptorPoolSize = {
		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		nb_Buffers
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
		VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		0,
		0,
		1,
		1,
		&descriptorPoolSize
	};

	CHECK_RESULT(vkCreateDescriptorPool(vkDevice, &descriptorPoolCreateInfo, 0, descriptorPool), "vkCreateDescriptorPool", NULL);

	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		0,
		*descriptorPool,
		1,
		descriptorSetLayout
	};
	CHECK_RESULT(vkAllocateDescriptorSets(vkDevice, &descriptorSetAllocateInfo, descriptorSet), "vkAllocateDescriptorSets", NULL);

	VkDescriptorBufferInfo descriptorBufferInfo0 = { b0, 0, 	VK_WHOLE_SIZE };
	
	VkWriteDescriptorSet writeDescriptorSet[1] = {
		{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,	0,	*descriptorSet,	0,	0,	1,	VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,	0,	&descriptorBufferInfo0,	0 },
	};

	vkUpdateDescriptorSets(vkDevice, nb_Buffers, writeDescriptorSet, 0, 0);

	return pipelineLayout;
}

uint32_t getBufferMemoryRequirements(VkDevice vkDevice, VkBuffer b)
{
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(vkDevice, b, &req);
	return req.alignment;
}

#endif

