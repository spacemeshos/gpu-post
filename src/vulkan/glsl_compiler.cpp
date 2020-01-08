/* Copyright (c) 2019, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include <glslang/Public/ShaderLang.h>

#include <SPIRV/GLSL.std.450.h>
#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/ResourceLimits.h>
#include <glslang/Include/ShHandle.h>
#include <glslang/Include/revision.h>
#include <glslang/OSDependent/osinclude.h>

#include "api_internal.h"

#define CHECK_RESULT(result,msg,errorRet) \
  if (VK_SUCCESS != (result)) {\
	applog(LOG_ERR, "Failure in %s at %u %s  ErrCode=%d\n", msg,__LINE__, __FILE__, result); \
	return (errorRet);\
  }

inline EShLanguage FindShaderLanguage(VkShaderStageFlagBits stage)
{
	switch (stage)
	{
		case VK_SHADER_STAGE_VERTEX_BIT:
			return EShLangVertex;

		case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
			return EShLangTessControl;

		case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
			return EShLangTessEvaluation;

		case VK_SHADER_STAGE_GEOMETRY_BIT:
			return EShLangGeometry;

		case VK_SHADER_STAGE_FRAGMENT_BIT:
			return EShLangFragment;

		case VK_SHADER_STAGE_COMPUTE_BIT:
			return EShLangCompute;

		default:
			return EShLangVertex;
	}
}

bool compile_to_spirv(
	VkShaderStageFlagBits stage,
	const std::vector<uint8_t> &glsl_source,
	const std::string &entry_point,
	std::vector<std::uint32_t> &spirv,
	std::string &info_log)
{
	// Initialize glslang library.
	glslang::InitializeProcess();

	EShMessages messages = static_cast<EShMessages>(EShMsgDefault | EShMsgVulkanRules | EShMsgSpvRules);

	EShLanguage language = FindShaderLanguage(stage);
	std::string source   = std::string(glsl_source.begin(), glsl_source.end());

	const char *file_name_list[1] = {""};
	const char *shader_source     = reinterpret_cast<const char *>(source.data());

	glslang::TShader shader(language);
	shader.setStringsWithLengthsAndNames(&shader_source, nullptr, file_name_list, 1);
	shader.setEntryPoint(entry_point.c_str());
	shader.setSourceEntryPoint(entry_point.c_str());

	if (!shader.parse(&glslang::DefaultTBuiltInResource, 100, false, messages))
	{
		info_log = std::string(shader.getInfoLog()) + "\n" + std::string(shader.getInfoDebugLog());
		return false;
	}

	// Add shader to new program object.
	glslang::TProgram program;
	program.addShader(&shader);

	// Link program.
	if (!program.link(messages))
	{
		info_log = std::string(program.getInfoLog()) + "\n" + std::string(program.getInfoDebugLog());
		return false;
	}

	// Save any info log that was generated.
	if (shader.getInfoLog())
	{
		info_log += std::string(shader.getInfoLog()) + "\n" + std::string(shader.getInfoDebugLog()) + "\n";
	}

	if (program.getInfoLog())
	{
		info_log += std::string(program.getInfoLog()) + "\n" + std::string(program.getInfoDebugLog());
	}

	glslang::TIntermediate *intermediate = program.getIntermediate(language);

	// Translate to SPIRV.
	if (!intermediate)
	{
		info_log += "Failed to get shared intermediate code.\n";
		return false;
	}

	spv::SpvBuildLogger logger;

	glslang::GlslangToSpv(*intermediate, spirv, &logger);

	info_log += logger.getAllMessages() + "\n";

	// Shutdown glslang library.
	glslang::FinalizeProcess();

	return true;
}

extern "C" VkPipeline compileShader(VkDevice vkDevice, VkPipelineLayout pipelineLayout, VkShaderModule *shader_module, const char * file_name)
{
	size_t shader_size;
	std::vector<uint8_t> buffer;

	FILE *fp = fopen(file_name, "rb");
	if (fp == NULL) {
		applog(LOG_ERR, "Program %s not found\n", file_name);
		return NULL;
	}
	fseek(fp, 0, SEEK_END);
	shader_size = (size_t)(ftell(fp) * sizeof(char));
	fseek(fp, 0, SEEK_SET);

	buffer.resize(shader_size + 1, 0);
	size_t read_size = fread(buffer.data(), sizeof(char), shader_size, fp);
	fclose(fp);
	if (read_size != shader_size) {
		applog(LOG_ERR, "Failed to read shader %s!\n", file_name);
		return NULL;
	}

	std::vector<uint32_t> spirv;
	std::string           info_log;

	// Compile the GLSL source
	if (!compile_to_spirv(VK_SHADER_STAGE_COMPUTE_BIT, buffer, "main", spirv, info_log))
	{
		applog(LOG_ERR, "Failed to compile shader, Error: %s\n", info_log.c_str());
		return VK_NULL_HANDLE;
	}

	VkShaderModuleCreateInfo shaderModuleCreateInfo = {
		VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		0,
		0,
		spirv.size() * sizeof(uint32_t),
		spirv.data()
	};

	CHECK_RESULT(vkCreateShaderModule(vkDevice, &shaderModuleCreateInfo, 0, shader_module), "vkCreateShaderModule", NULL);

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
	CHECK_RESULT(vkCreateComputePipelines(vkDevice, 0, 1, &computePipelineCreateInfo, 0, &pipeline), "vkCreateComputePipelines", NULL);

	return pipeline;
}

