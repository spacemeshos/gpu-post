#ifndef	_SPACEMESH_VULKAN_VULKAN_HELPERS_H__
#define	_SPACEMESH_VULKAN_VULKAN_HELPERS_H__

#define	VK_NO_PROTOTYPES 1
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#define CHECK_RESULT(result,msg,errorRet) \
  if (VK_SUCCESS != (result)) {\
	applog(LOG_ERR, "Failure in %s at %u %s  ErrCode=%d\n", msg, __LINE__, __FILE__, result); \
	return (errorRet);\
  }

#define CHECK_RESULT_WITH_ACTION(result,msg,errorRet,action) \
  if (VK_SUCCESS != (result)) {\
	applog(LOG_ERR, "Failure in %s at %u %s  ErrCode=%d\n", msg, __LINE__, __FILE__, result); \
	action; \
	return (errorRet);\
  }

#define CHECK_RESULT_NORET(result,msg) \
  if (VK_SUCCESS != (result)) {\
	applog(LOG_ERR, "Failure in %s at %u %s  ErrCode=%d\n", msg,__LINE__, __FILE__, result); \
	return;\
  }

#define	DECLARE_VULKAN_FUNCTION(func)	PFN_##func	func

typedef struct _Vulkan {
	void *library;

	DECLARE_VULKAN_FUNCTION(vkCreateInstance);
	DECLARE_VULKAN_FUNCTION(vkDestroyInstance);
	DECLARE_VULKAN_FUNCTION(vkGetPhysicalDeviceQueueFamilyProperties);
	DECLARE_VULKAN_FUNCTION(vkCreateDevice);
	DECLARE_VULKAN_FUNCTION(vkGetPhysicalDeviceMemoryProperties);
	DECLARE_VULKAN_FUNCTION(vkAllocateMemory);
	DECLARE_VULKAN_FUNCTION(vkCreateBuffer);
	DECLARE_VULKAN_FUNCTION(vkBindBufferMemory);
	DECLARE_VULKAN_FUNCTION(vkCreateDescriptorSetLayout);
	DECLARE_VULKAN_FUNCTION(vkCreatePipelineLayout);
	DECLARE_VULKAN_FUNCTION(vkCreateDescriptorPool);
	DECLARE_VULKAN_FUNCTION(vkAllocateDescriptorSets);
	DECLARE_VULKAN_FUNCTION(vkUpdateDescriptorSets);
	DECLARE_VULKAN_FUNCTION(vkGetBufferMemoryRequirements);
	DECLARE_VULKAN_FUNCTION(vkCreateShaderModule);
	DECLARE_VULKAN_FUNCTION(vkCreateComputePipelines);
	DECLARE_VULKAN_FUNCTION(vkDestroyBuffer);
	DECLARE_VULKAN_FUNCTION(vkFreeMemory);
	DECLARE_VULKAN_FUNCTION(vkGetDeviceQueue);
	DECLARE_VULKAN_FUNCTION(vkMapMemory);
	DECLARE_VULKAN_FUNCTION(vkUnmapMemory);
	DECLARE_VULKAN_FUNCTION(vkCreateCommandPool);
	DECLARE_VULKAN_FUNCTION(vkAllocateCommandBuffers);
	DECLARE_VULKAN_FUNCTION(vkCreateSemaphore);
	DECLARE_VULKAN_FUNCTION(vkCreateFence);
	DECLARE_VULKAN_FUNCTION(vkBeginCommandBuffer);
	DECLARE_VULKAN_FUNCTION(vkCmdBindPipeline);
	DECLARE_VULKAN_FUNCTION(vkCmdBindDescriptorSets);
	DECLARE_VULKAN_FUNCTION(vkCmdDispatch);
	DECLARE_VULKAN_FUNCTION(vkEndCommandBuffer);
	DECLARE_VULKAN_FUNCTION(vkEnumeratePhysicalDevices);
	DECLARE_VULKAN_FUNCTION(vkGetPhysicalDeviceProperties);
	DECLARE_VULKAN_FUNCTION(vkGetPhysicalDeviceProperties2);
	DECLARE_VULKAN_FUNCTION(vkQueueSubmit);
	DECLARE_VULKAN_FUNCTION(vkQueueWaitIdle);
	DECLARE_VULKAN_FUNCTION(vkDestroyPipelineLayout);
	DECLARE_VULKAN_FUNCTION(vkDestroyDescriptorSetLayout);
	DECLARE_VULKAN_FUNCTION(vkDestroyPipeline);
	DECLARE_VULKAN_FUNCTION(vkDestroyCommandPool);
	DECLARE_VULKAN_FUNCTION(vkDestroyDescriptorPool);
	DECLARE_VULKAN_FUNCTION(vkDestroyShaderModule);
	DECLARE_VULKAN_FUNCTION(vkDestroySemaphore);
	DECLARE_VULKAN_FUNCTION(vkDestroyFence);
	DECLARE_VULKAN_FUNCTION(vkWaitForFences);
	DECLARE_VULKAN_FUNCTION(vkResetFences);
	DECLARE_VULKAN_FUNCTION(vkFreeDescriptorSets);
	DECLARE_VULKAN_FUNCTION(vkFreeCommandBuffers);
	DECLARE_VULKAN_FUNCTION(vkDestroyDevice);
} Vulkan;

#undef	DECLARE_VULKAN_FUNCTION

#ifdef __cplusplus
extern "C" {
#endif

extern Vulkan gVulkan;
extern VkInstance gInstance;
extern VkPhysicalDevice* gPhysicalDevices;
extern uint32_t gPhysicalDeviceCount;

int initVulkanLibrary();
void doneVulkanLibrary();
int getComputeQueueFamilyIndex(uint32_t index);
VkDevice createDevice(int index, uint32_t computeQueueFamilyIndex);
VkDeviceMemory allocateGPUMemory(int index,  VkDevice vkDevice, const VkDeviceSize memorySize, char isLocal, bool isFatal);
VkBuffer createBuffer(VkDevice vkDevice, uint32_t computeQueueFamilyIndex, VkDeviceMemory memory, VkDeviceSize bufferSize, VkDeviceSize offset);
VkPipelineLayout bindBuffers(VkDevice vkDevice, VkDescriptorSet *descriptorSet, VkDescriptorPool *descriptorPool, VkDescriptorSetLayout *descriptorSetLayout, VkBuffer b0, VkBuffer b1, VkBuffer b2, VkBuffer b3, VkBuffer b4, VkBuffer b5);
uint64_t getBufferMemoryRequirements(VkDevice vkDevice, VkBuffer b);
VkPipeline loadShaderFromFile(VkDevice vkDevice, VkPipelineLayout pipelineLayout, VkShaderModule *shader_module, const char * spirv_file_name);
VkPipeline loadShader(VkDevice vkDevice, VkPipelineLayout pipelineLayout, VkShaderModule *shader_module, uint32_t workSize, uint32_t labelSize);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif	// _SPACEMESH_VULKAN_VULKAN_HELPERS_H__
