#ifndef	_SPACEMESH_VULKAN_VULKAN_HELPERS_H__
#define	_SPACEMESH_VULKAN_VULKAN_HELPERS_H__

#include <vulkan/vulkan.h>

#define CHECK_RESULT(result,msg,errorRet) \
  if (VK_SUCCESS != (result)) {\
	applog(LOG_ERR, "Failure in %s at %u %s  ErrCode=%d\n", msg, __LINE__, __FILE__, result); \
	return (errorRet);\
  }

#define CHECK_RESULT_NORET(result,msg) \
  if (VK_SUCCESS != (result)) {\
	applog(LOG_ERR, "Failure in %s at %u %s  ErrCode=%d\n", msg,__LINE__, __FILE__, result); \
	return;\
  }

extern VkInstance gInstance;
extern VkPhysicalDevice* gPhysicalDevices;
extern uint32_t gPhysicalDeviceCount;

int getComputeQueueFamillyIndex(uint32_t index);
VkDevice createDevice(int index, uint32_t computeQueueFamillyIndex);
VkDeviceMemory allocateGPUMemory(int index,  VkDevice vkDevice, const VkDeviceSize memorySize, char isLocal, bool isFatal);
VkBuffer createBuffer(VkDevice vkDevice, uint32_t computeQueueFamillyIndex, VkDeviceMemory memory, VkDeviceSize bufferSize, VkDeviceSize offset);
VkPipelineLayout bindBuffers(VkDevice vkDevice, VkDescriptorSet *descriptorSet, VkDescriptorPool *descriptorPool, VkDescriptorSetLayout *descriptorSetLayout, VkBuffer b0, VkBuffer b1, VkBuffer b2, VkBuffer b3, VkBuffer b4, VkBuffer b5);
uint64_t getBufferMemoryRequirements(VkDevice vkDevice, VkBuffer b);
VkPipeline loadShader(VkDevice vkDevice, VkPipelineLayout pipelineLayout, VkShaderModule *shader_module, const char * spirv_file_name);
VkPipeline compileShader(VkDevice vkDevice, VkPipelineLayout pipelineLayout, VkShaderModule *shader_module, const char *glsl_source, const char *options);

#endif	// _SPACEMESH_VULKAN_VULKAN_HELPERS_H__
