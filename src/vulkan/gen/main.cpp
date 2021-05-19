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

#include <fstream>
#include <string>
#include <vector>
#include <stdarg.h>

#include "zlib.h"

#define	VK_NO_PROTOTYPES 1
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
#include <MoltenVKGLSLToSPIRVConverter/GLSLToSPIRVConverter.h>
#else
#include <SPIRV/GlslangToSpv.h>
#endif

#include <glslang/Include/ResourceLimits.h>

#include "scrypt-chacha-vulkan.inl"

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

extern "C" bool compileShaderToSpirV(const char *glsl_source, const char *options, int work_size, int hash_len_bits, bool copy_only, std::vector<uint32_t> &spirv)
{
	std::string           info_log;
	std::string           source;

	if (options) {
		source = options;
		source += '\n';
	}
	source.append(glsl_source);

	GlCodeWritter writter(hash_len_bits);
	std::string packer;

	if (copy_only) {
		if (writter._label_words_size == 1) {
			if (hash_len_bits == 32) {
				source += "labels[lid] = outputBuffer1[gid * 8 + 0];\n";
			}
			else {
				stdprintf(source, "labels[lid] = outputBuffer1[gid * 8 + 0] & 0x%08x;\n", (1 << hash_len_bits) - 1);
			}
		}
		else {
			std::string last;
			stdprintf(source, "tmp = lid * %u;\n", writter._label_words_size);
			if (writter._label_words_size > 1) {
				source += "labels[tmp++] = outputBuffer1[gid * 8 + 0];\n";
				last = "labels[tmp] = outputBuffer1[gid * 8 + 1]";
			}
			if (writter._label_words_size > 2) {
				source += "labels[tmp++] = outputBuffer1[gid * 8 + 1];\n";
				last = "labels[tmp++] = outputBuffer1[gid * 8 + 2]";
			}
			if (writter._label_words_size > 3) {
				source += "labels[tmp++] = outputBuffer1[gid * 8 + 2];\n";
				last = "labels[tmp++] = outputBuffer1[gid * 8 + 3]";
			}
			if (writter._label_words_size > 4) {
				source += "labels[tmp++] = outputBuffer1[gid * 8 + 3];\n";
				last = "labels[tmp++] = outputBuffer1[gid * 8 + 4]";
			}
			if (writter._label_words_size > 5) {
				source += "labels[tmp++] = outputBuffer1[gid * 8 + 4];\n";
				last = "labels[tmp++] = outputBuffer1[gid * 8 + 5]";
			}
			if (writter._label_words_size > 6) {
				source += "labels[tmp++] = outputBuffer1[gid * 8 + 5];\n";
				last = "labels[tmp++] = outputBuffer1[gid * 8 + 6]";
			}
			if (writter._label_words_size > 7) {
				source += "labels[tmp++] = outputBuffer1[gid * 8 + 6];\n";
				last = "labels[tmp++] = outputBuffer1[gid * 8 + 7]";
			}
			source += last;
			if (0 == hash_len_bits % 32) {
				source += ";\n";
			}
			else {
				stdprintf(source, " & 0x%08x;\n", (1 << (hash_len_bits % 32)) - 1);
			}
		}
	}
	else {
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
			source += last;
			if (0 == hash_len_bits % 32) {
				source += ";\n";
			}
			else {
				stdprintf(source, " & 0x%08x;\n", (1 << (hash_len_bits % 32)) - 1);
			}
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
		printf("Failed to compile shader, Error: %s\n", info_log.c_str());
		return false;
	}

	return true;
}

extern "C" void init_glslang()
{
	glslang::InitializeProcess();
}

extern "C" void finalize_glslang()
{
	glslang::FinalizeProcess();
}

bool saveToFile(const char *aFileName, const void *aData, size_t aSize)
{
	std::ofstream out(aFileName, std::ios::binary | std::ios::out);
	if (!out.fail()) {
		return out.write(reinterpret_cast<const char*>(aData), aSize).tellp() == static_cast<std::streampos>(aSize);
	}
	return false;
}

int main(int argc, char **argv)
{
	init_glslang();

	std::vector<uint8_t> out((4 + 4 * 256) * sizeof(uint32_t));
	uint32_t *header = (uint32_t *)out.data();

	header[0] = 1;
	header[1] = 64;
	header[2] = 12;
	header[3] = 256;

	for (int hash_len_bits = 1; hash_len_bits <= 256; hash_len_bits++) {
		char options[256];
		std::vector<uint32_t> spirv;
		
		snprintf(options, sizeof(options), "#version 450\n#define LOOKUP_GAP %d\n#define WORKSIZE %d\n#define LABEL_SIZE %d\n",	4, 64, hash_len_bits);

		if (compileShaderToSpirV(scrypt_chacha_comp, options, 64, hash_len_bits, false, spirv)) {
//			char filename[128];
//			snprintf(filename, sizeof(filename), "kernel-%02d-%03d.spirv", 64, hash_len_bits);
//			saveToFile(filename, spirv.data(), spirv.size() * sizeof(uint32_t));
			uLongf destLen = spirv.size() * sizeof(uint32_t);
			std::vector<Bytef> dst(destLen);
			if (Z_OK != compress2(dst.data(), &destLen, (Bytef*)spirv.data(), destLen, 9)) {
				return 1;
			}
			header[hash_len_bits * 4 + 0] = hash_len_bits;
			header[hash_len_bits * 4 + 1] = spirv.size() * sizeof(uint32_t);
			header[hash_len_bits * 4 + 2] = destLen;
			header[hash_len_bits * 4 + 3] = out.size();

			printf("64:%03u %u -> %u\n", header[hash_len_bits * 4 + 0], header[hash_len_bits * 4 + 1], header[hash_len_bits * 4 + 2]);

			out.insert(out.end(), dst.begin(), dst.begin() + destLen);
			header = (uint32_t *)out.data();
		}
	}

	if (FILE *h = fopen("vulkan-shaders-vault.inl", "wb")) {
		fprintf(h, "uint8_t vulkan_shaders_vault[] = {\r\n");
		uint8_t *src = out.data();
		for (uint32_t length = (uint32_t)out.size(); length > 0; length -= std::min<uint32_t>(32, length)) {
			for (int i = 0; i < std::min<uint32_t>(32, length); i++) {
				fprintf(h, "0x%02x, ", *src++);
			}
			fprintf(h, "\n");
		}

		fprintf(h, "};\r\n");

		fclose(h);
	}

	finalize_glslang();

	return 0;
}

