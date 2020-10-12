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
#include <stdarg.h>

#include "api_internal.h"
#undef max
#undef min
#include "vulkan-helpers.h"

#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
#include <MoltenVKGLSLToSPIRVConverter/GLSLToSPIRVConverter.h>
#else
#include <SPIRV/GlslangToSpv.h>
#endif

#include <glslang/Include/ResourceLimits.h>

static void init_resources(TBuiltInResource &Resources)
{
	Resources.maxLights = 32;
	Resources.maxClipPlanes = 6;
	Resources.maxTextureUnits = 32;
	Resources.maxTextureCoords = 32;
	Resources.maxVertexAttribs = 64;
	Resources.maxVertexUniformComponents = 4096;
	Resources.maxVaryingFloats = 64;
	Resources.maxVertexTextureImageUnits = 32;
	Resources.maxCombinedTextureImageUnits = 80;
	Resources.maxTextureImageUnits = 32;
	Resources.maxFragmentUniformComponents = 4096;
	Resources.maxDrawBuffers = 32;
	Resources.maxVertexUniformVectors = 128;
	Resources.maxVaryingVectors = 8;
	Resources.maxFragmentUniformVectors = 16;
	Resources.maxVertexOutputVectors = 16;
	Resources.maxFragmentInputVectors = 15;
	Resources.minProgramTexelOffset = -8;
	Resources.maxProgramTexelOffset = 7;
	Resources.maxClipDistances = 8;
	Resources.maxComputeWorkGroupCountX = 65535;
	Resources.maxComputeWorkGroupCountY = 65535;
	Resources.maxComputeWorkGroupCountZ = 65535;
	Resources.maxComputeWorkGroupSizeX = 1024;
	Resources.maxComputeWorkGroupSizeY = 1024;
	Resources.maxComputeWorkGroupSizeZ = 64;
	Resources.maxComputeUniformComponents = 1024;
	Resources.maxComputeTextureImageUnits = 16;
	Resources.maxComputeImageUniforms = 8;
	Resources.maxComputeAtomicCounters = 8;
	Resources.maxComputeAtomicCounterBuffers = 1;
	Resources.maxVaryingComponents = 60;
	Resources.maxVertexOutputComponents = 64;
	Resources.maxGeometryInputComponents = 64;
	Resources.maxGeometryOutputComponents = 128;
	Resources.maxFragmentInputComponents = 128;
	Resources.maxImageUnits = 8;
	Resources.maxCombinedImageUnitsAndFragmentOutputs = 8;
	Resources.maxCombinedShaderOutputResources = 8;
	Resources.maxImageSamples = 0;
	Resources.maxVertexImageUniforms = 0;
	Resources.maxTessControlImageUniforms = 0;
	Resources.maxTessEvaluationImageUniforms = 0;
	Resources.maxGeometryImageUniforms = 0;
	Resources.maxFragmentImageUniforms = 8;
	Resources.maxCombinedImageUniforms = 8;
	Resources.maxGeometryTextureImageUnits = 16;
	Resources.maxGeometryOutputVertices = 256;
	Resources.maxGeometryTotalOutputComponents = 1024;
	Resources.maxGeometryUniformComponents = 1024;
	Resources.maxGeometryVaryingComponents = 64;
	Resources.maxTessControlInputComponents = 128;
	Resources.maxTessControlOutputComponents = 128;
	Resources.maxTessControlTextureImageUnits = 16;
	Resources.maxTessControlUniformComponents = 1024;
	Resources.maxTessControlTotalOutputComponents = 4096;
	Resources.maxTessEvaluationInputComponents = 128;
	Resources.maxTessEvaluationOutputComponents = 128;
	Resources.maxTessEvaluationTextureImageUnits = 16;
	Resources.maxTessEvaluationUniformComponents = 1024;
	Resources.maxTessPatchComponents = 120;
	Resources.maxPatchVertices = 32;
	Resources.maxTessGenLevel = 64;
	Resources.maxViewports = 16;
	Resources.maxVertexAtomicCounters = 0;
	Resources.maxTessControlAtomicCounters = 0;
	Resources.maxTessEvaluationAtomicCounters = 0;
	Resources.maxGeometryAtomicCounters = 0;
	Resources.maxFragmentAtomicCounters = 8;
	Resources.maxCombinedAtomicCounters = 8;
	Resources.maxAtomicCounterBindings = 1;
	Resources.maxVertexAtomicCounterBuffers = 0;
	Resources.maxTessControlAtomicCounterBuffers = 0;
	Resources.maxTessEvaluationAtomicCounterBuffers = 0;
	Resources.maxGeometryAtomicCounterBuffers = 0;
	Resources.maxFragmentAtomicCounterBuffers = 1;
	Resources.maxCombinedAtomicCounterBuffers = 1;
	Resources.maxAtomicCounterBufferSize = 16384;
	Resources.maxTransformFeedbackBuffers = 4;
	Resources.maxTransformFeedbackInterleavedComponents = 64;
	Resources.maxCullDistances = 8;
	Resources.maxCombinedClipAndCullDistances = 8;
	Resources.maxSamples = 4;
	Resources.maxMeshOutputVerticesNV = 256;
	Resources.maxMeshOutputPrimitivesNV = 512;
	Resources.maxMeshWorkGroupSizeX_NV = 32;
	Resources.maxMeshWorkGroupSizeY_NV = 1;
	Resources.maxMeshWorkGroupSizeZ_NV = 1;
	Resources.maxTaskWorkGroupSizeX_NV = 32;
	Resources.maxTaskWorkGroupSizeY_NV = 1;
	Resources.maxTaskWorkGroupSizeZ_NV = 1;
	Resources.maxMeshViewCountNV = 4;
	Resources.limits.nonInductiveForLoops = 1;
	Resources.limits.whileLoops = 1;
	Resources.limits.doWhileLoops = 1;
	Resources.limits.generalUniformIndexing = 1;
	Resources.limits.generalAttributeMatrixVectorIndexing = 1;
	Resources.limits.generalVaryingIndexing = 1;
	Resources.limits.generalSamplerIndexing = 1;
	Resources.limits.generalVariableIndexing = 1;
	Resources.limits.generalConstantMatrixVectorIndexing = 1;
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
	const std::string &glsl_source,
	const std::string &entry_point,
	std::vector<std::uint32_t> &spirv,
	std::string &info_log)
{
#if 1
	EShLanguage language = FindShaderLanguage(stage);

	glslang::TShader shader(language);
	glslang::TProgram program;
	const char *shaderStrings[1];
	TBuiltInResource Resources = {};
	init_resources(Resources);

	// Enable SPIR-V and Vulkan rules when parsing GLSL
	EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);

	shaderStrings[0] = glsl_source.c_str();
	shader.setStrings(shaderStrings, 1);

	if (!shader.parse(&Resources, 100, false, messages)) {
		info_log = std::string(shader.getInfoLog()) + "\n" + std::string(shader.getInfoDebugLog());
		return false;  // something didn't work
	}

	program.addShader(&shader);

	//
	// Program-level processing...
	//

	if (!program.link(messages)) {
		info_log = std::string(shader.getInfoLog()) + "\n" + std::string(shader.getInfoDebugLog());
		return false;
	}

	glslang::GlslangToSpv(*program.getIntermediate(language), spirv);
#else
	// Initialize glslang library.
	glslang::InitializeProcess();

	EShMessages messages = static_cast<EShMessages>(EShMsgDefault | EShMsgVulkanRules | EShMsgSpvRules);

	EShLanguage language = FindShaderLanguage(stage);

	const char *file_name_list[1] = {""};
	const char *shader_source     = reinterpret_cast<const char *>(glsl_source.data());

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
#endif
	return true;
}

extern "C" bool loadSource(const char * file_name, std::vector<uint8_t> &buffer)
{
	size_t shader_size;

	FILE *fp = fopen(file_name, "rb");
	if (fp == NULL) {
		applog(LOG_ERR, "Program %s not found\n", file_name);
		return false;
	}
	fseek(fp, 0, SEEK_END);
	shader_size = (size_t)(ftell(fp) * sizeof(char));
	fseek(fp, 0, SEEK_SET);

	buffer.resize(shader_size + 1, 0);
	size_t read_size = fread(buffer.data(), sizeof(char), shader_size, fp);
	fclose(fp);
	if (read_size != shader_size) {
		applog(LOG_ERR, "Failed to read shader %s!\n", file_name);
		return false;
	}

	return true;
}

static void stdprintf(std::string &out, const char *format, ...)
{
	char buffer[256];
	va_list argList;
	va_start(argList, format);
	vsnprintf(buffer, sizeof(buffer), format, argList);
	out += buffer;
}

struct GlCodeWritter
{
	GlCodeWritter(int aLabelSize) : _label_size(aLabelSize)
	{
		_label_words_size = (_label_size + 31) / 32;
		_use_word_copy = 0 == (_label_size % 32);
		_label_last_words_size = _label_size % 32;
	}

	void outLabel(uint32_t i, std::string &out)
	{
		if (_use_word_copy) {
			for (int current_word = 0; current_word < _label_words_size; current_word++) {
				stdprintf(out, "outputBuffer0[i++] = labels[%d];\n", i * _label_words_size + current_word);
			}
		}
		else {
			int current_word = 0;
			if (_label_words_size > 1) {
				if (32 == _available) {
					while (current_word < (_label_words_size - 1)) {
						stdprintf(out, "outputBuffer0[i++] = labels[%d];\n", i * _label_words_size + current_word);
						current_word++;
					}
					_current_nonce = 0;
				}
				else {
					while (current_word < (_label_words_size - 1)) {
						stdprintf(out, "nonce.%c |= labels[%d] << %d;\n", 'x' + _current_nonce, i * _label_words_size + current_word, (32 - _available));
						stdprintf(out, "outputBuffer0[i++] = nonce.%c;\n", 'x' + _current_nonce);
						inc_current_nonce();
						stdprintf(out, "nonce.%c = labels[%d] >> %d;\n", 'x' + _current_nonce, i * _label_words_size + current_word, _available);
						current_word++;
					}
				}
			}
			if (_label_last_words_size > _available) {
				stdprintf(out, "nonce.%c |= labels[%d] << %d;\n", 'x' + _current_nonce, i * _label_words_size + current_word, (32 - _available));
				stdprintf(out, "outputBuffer0[i++] = nonce.%c;\n", 'x' + _current_nonce);
				inc_current_nonce();
				stdprintf(out, "nonce.%c = labels[%d] >> %d;\n", 'x' + _current_nonce, i * _label_words_size + current_word, _available);
				_available = 32 - (_label_last_words_size - _available);
			}
			else {
				if (_available == 32) {
					stdprintf(out, "nonce.%c = labels[%d];\n", 'x' + _current_nonce, i * _label_words_size + current_word);
				}
				else {
					stdprintf(out, "nonce.%c |= labels[%d] << %d;\n", 'x' + _current_nonce, i * _label_words_size + current_word, (32 - _available));
				}
				_available -= _label_last_words_size;
				if (0 == _available) {
					stdprintf(out, "outputBuffer0[i++] = nonce.%c;\n", 'x' + _current_nonce);
					_available = 32;
					inc_current_nonce();
				}
			}
		}
	}

	void inc_current_nonce()
	{
		if (_current_nonce) {
			_current_nonce = 0;
		}
		else {
			_current_nonce = 1;
		}
	}

	int _label_size;
	int _label_words_size;
	int _label_last_words_size;
	int _current_nonce = 0;
	int _available = 32;
	bool _use_word_copy;
};

extern "C" VkPipeline compileShader(VkDevice vkDevice, VkPipelineLayout pipelineLayout, VkShaderModule *shader_module, const char *glsl_source, const char *options, int work_size, int hash_len_bits)
{
	std::vector<uint32_t> spirv;
	std::string           info_log;
	std::string           source;

	if (options) {
		source = options;
		source += '\n';
	}
	source.append(glsl_source);

	GlCodeWritter writter(hash_len_bits);
	std::string packer;

	if (writter._label_words_size == 1) {
		if (hash_len_bits == 32) {
			source += "labels[lid] = hmac_pw.outer.state4[0].x;\n";
		}
		else {
			stdprintf(source, "labels[lid] = hmac_pw.outer.state4[0].x & 0x%08x;\n", (1 << hash_len_bits) - 1);
		}
	}
	else {
		std::string last;
		stdprintf(source, "tmp = lid * %u;\n", writter._label_words_size);
		if (writter._label_words_size > 1) {
			source += "labels[tmp++] = hmac_pw.outer.state4[0].x;\n";
			last = "labels[tmp] = hmac_pw.outer.state4[0].y";
		}
		if (writter._label_words_size > 2) {
			source += "labels[tmp++] = hmac_pw.outer.state4[0].y;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[0].z";
		}
		if (writter._label_words_size > 3) {
			source += "labels[tmp++] = hmac_pw.outer.state4[0].z;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[0].w";
		}
		if (writter._label_words_size > 4) {
			source += "labels[tmp++] = hmac_pw.outer.state4[0].w;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[1].x";
		}
		if (writter._label_words_size > 5) {
			source += "labels[tmp++] = hmac_pw.outer.state4[1].x;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[1].y";
		}
		if (writter._label_words_size > 6) {
			source += "labels[tmp++] = hmac_pw.outer.state4[1].y;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[1].z";
		}
		if (writter._label_words_size > 7) {
			source += "labels[tmp++] = hmac_pw.outer.state4[1].z;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[1].w";
		}
		if (writter._label_words_size > 8) {
			source += "labels[tmp++] = hmac_pw.outer.state4[1].w;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[2].x";
		}
		if (writter._label_words_size > 9) {
			source += "labels[tmp++] = hmac_pw.outer.state4[2].x;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[2].y";
		}
		if (writter._label_words_size > 10) {
			source += "labels[tmp++] = hmac_pw.outer.state4[2].y;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[2].z";
		}
		if (writter._label_words_size > 11) {
			source += "labels[tmp++] = hmac_pw.outer.state4[2].z;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[2].w";
		}
		if (writter._label_words_size > 12) {
			source += "labels[tmp++] = hmac_pw.outer.state4[2].w;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[3].x";
		}
		if (writter._label_words_size > 13) {
			source += "labels[tmp++] = hmac_pw.outer.state4[3].x;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[3].y";
		}
		if (writter._label_words_size > 14) {
			source += "labels[tmp++] = hmac_pw.outer.state4[3].y;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[3].z";
		}
		if (writter._label_words_size > 15) {
			source += "labels[tmp++] = hmac_pw.outer.state4[3].z;\n";
			last = "labels[tmp++] = hmac_pw.outer.state4[3].w";
		}
		source += last;
		if (0 == hash_len_bits % 32) {
			source += ";\n";
		}
		else {
			stdprintf(source, " & 0x%08x;\n", (1 << (hash_len_bits % 32)) - 1);
		}
	}

	source += "barrier();\n";
	source += "if (0 == lid) {\n";
	if (work_size == 128) {
		source += "i = (gid * LABEL_SIZE) >> 6;\n";
	}
	else {
		source += "i = (gid * LABEL_SIZE) >> 5;\n";
	}

	for (uint32_t i = 0; i < work_size; i++) {
		writter.outLabel(i, packer);
	}

	source.append(packer);
	source += "}}\n";

	// Compile the GLSL source
	if (!compile_to_spirv(VK_SHADER_STAGE_COMPUTE_BIT, source, "main", spirv, info_log))
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

	CHECK_RESULT(gVulkan.vkCreateShaderModule(vkDevice, &shaderModuleCreateInfo, 0, shader_module), "vkCreateShaderModule", VK_NULL_HANDLE);

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
	CHECK_RESULT(gVulkan.vkCreateComputePipelines(vkDevice, 0, 1, &computePipelineCreateInfo, 0, &pipeline), "vkCreateComputePipelines", VK_NULL_HANDLE);

	return pipeline;
}

extern "C" void init_glslang()
{
	glslang::InitializeProcess();
}

extern "C" void finalize_glslang()
{
	glslang::FinalizeProcess();
}

