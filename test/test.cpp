#include "../include/api.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#ifdef WIN32
#include <conio.h>
#endif

#ifndef min
# define min(a, b)  ((a) < (b) ? (a) : (b))
#endif

static void print(uint8_t *data)
{
	for (int i = 0; i < 32; i++) {
		printf("%02x ", data[i]);
	}
	printf("\n");
}

void do_benchmark(int aLabelSize, int aLabelsCount)
{
	uint8_t id[32];
	uint8_t salt[32];

	int providersCount = spacemesh_api_get_providers(NULL, 0);

	srand(time(nullptr));
	for (int i = 0; i < sizeof(id); i++) {
		id[i] = rand();
	}
	for (int i = 0; i < sizeof(salt); i++) {
		salt[i] = rand();
	}

	if (providersCount > 0) {
		PostComputeProvider *providers = (PostComputeProvider *)malloc(providersCount * sizeof(PostComputeProvider));

		if (spacemesh_api_get_providers(providers, providersCount) == providersCount) {
			uint8_t *out = (uint8_t *)malloc((uint64_t(aLabelsCount) * uint64_t(aLabelSize) + 7ull) / 8ull);
			if (out == NULL) {
				printf("Buffer allocation error\n");
				return;
			}
			for (int i = 0; i < providersCount; i++) {
				if (providers[i].compute_api != COMPUTE_API_CLASS_CPU) {
					uint64_t hashes_computed;
					uint64_t hashes_per_sec;
					scryptPositions(providers[i].id, id, 0, aLabelsCount - 1, aLabelSize, salt, 0, out, 512, 1, 1, &hashes_computed, &hashes_per_sec);
					printf("%s: %u hashes, %u h/s\n", providers[i].model, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);
				}
			}
			free(out);
		}
		free(providers);
	}
}

void do_test(int aLabelSize, int aLabelsCount)
{
	int referenceLabelsCount = aLabelsCount;
	uint8_t id[32];
	uint8_t salt[32];
	int providersCount = spacemesh_api_get_providers(NULL, 0);

	srand(time(nullptr));
	for (int i = 0; i < sizeof(id); i++) {
		id[i] = rand();
	}
	for (int i = 0; i < sizeof(salt); i++) {
		salt[i] = rand();
	}

	if (referenceLabelsCount > 128 * 1024) {
		referenceLabelsCount = 128 * 1024;
	}

	if (providersCount > 0) {
		PostComputeProvider *providers = (PostComputeProvider *)malloc(providersCount * sizeof(PostComputeProvider));

		if (spacemesh_api_get_providers(providers, providersCount) == providersCount) {
			int i;
			size_t labelsBufferSize = (size_t(aLabelsCount) * size_t(aLabelSize) + 7ull) / 8ull;
			size_t labelsBufferAllignedSize = ((labelsBufferSize + 31ull) / 32ull) * 32ull;
			uint8_t *out = (uint8_t *)malloc(providersCount * labelsBufferAllignedSize);
			uint8_t *referenceLabels = nullptr;
			uint64_t hashes_computed;
			uint64_t hashes_per_sec;
			bool checkOuitput = false;
			int cpu;

			// Find CPU provider and compute reference labels
			for (i = 0; i < providersCount; i++) {
				if (providers[i].compute_api == COMPUTE_API_CLASS_CPU) {
					referenceLabels = out + i * labelsBufferAllignedSize;
					memset(referenceLabels, 0, labelsBufferSize);
					scryptPositions(providers[i].id, id, 0, referenceLabelsCount - 1, aLabelSize, salt, 0, referenceLabels, 512, 1, 1, &hashes_computed, &hashes_per_sec);
					printf("%s: %u hashes, %u h/s\n", providers[i].model, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);
					cpu = i;
					checkOuitput = true;
					break;
				}
			}

			for (i = 0; i < providersCount; i++) {
				if (providers[i].compute_api != COMPUTE_API_CLASS_CPU) {
					uint8_t *labels = out + i * labelsBufferAllignedSize;
					memset(labels, 0, labelsBufferSize);
					scryptPositions(providers[i].id, id, 0, aLabelsCount - 1, aLabelSize, salt, 0, labels, 512, 1, 1, &hashes_computed, &hashes_per_sec);
					printf("%s: %u hashes, %u h/s\n", providers[i].model, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);
					if (checkOuitput) {
#if 0
						if (0 != memcmp(referenceLabels, labels, labelsBufferSize)) {
							printf("WRONG result for label size %d from provider %d [%s]\n", labelSize, i, providers[i].model);
						}
#else
						uint8_t *cpuSrc = referenceLabels;
						uint8_t *gpuSrc = labels;
						volatile unsigned errors = 0;
						static volatile struct {
							unsigned	pos;
							uint8_t		cpu;
							uint8_t		gpu;
						} errorInfo[2048];
						size_t referencelabelsBufferSize = (size_t(referenceLabelsCount) * size_t(aLabelSize)) / 8ull;
						for (unsigned pos = 0; pos < referencelabelsBufferSize; pos++) {
							if (cpuSrc[pos] != gpuSrc[pos]) {
								if (errors < 2048) {
									errorInfo[errors].pos = pos;
									errorInfo[errors].cpu = cpuSrc[pos];
									errorInfo[errors].gpu = gpuSrc[pos];
								}
								errors++;
							}
						}
						if (errors) {
							printf("WRONG result for label size %d (%u) from provider %d [%s]\n", aLabelSize, errors, i, providers[i].model);
						}
#endif
					}
				}
			}

			free(out);
		}

		free(providers);
	}
}

int main(int argc, char **argv)
{
	bool runBenchmark = false;
	bool runTest = false;
	int labelSize = 8;
	int labelsCount = 128 * 1024;
	if (argc == 1) {
		printf("Usage:\n");
		return 0;
	}
	for (int i = 1; i < argc; i++) {
		if (0 == strcmp(argv[i], "--benchmark") || 0 == strcmp(argv[i], "-b")) {
			runBenchmark = true;
		}
		else if (0 == strcmp(argv[i], "--test") || 0 == strcmp(argv[i], "-t")) {
			runTest = true;
		}
		else if (0 == strcmp(argv[i], "--label-size") || 0 == strcmp(argv[i], "-s")) {
			i++;
			if (i < argc) {
				labelSize = atoi(argv[i]);
				if (labelsCount < 1) {
					labelSize = 1;
				}
				else if (labelSize > 256) {
					labelSize = 256;
				}
			}
		}
		else if (0 == strcmp(argv[i], "--labels-count") || 0 == strcmp(argv[i], "-n")) {
			i++;
			if (i < argc) {
				labelsCount = atoi(argv[i]);
				if (labelsCount < 1) {
					labelsCount = 250000;
				}
				else if (labelsCount > 32 * 1024 * 1024) {
					labelsCount = 32 * 1024 * 1024;
				}
			}
		}
	}
	if (runBenchmark) {
		printf("Benchmark: Label size: %u, count %u, buffer %.1fM\n", labelSize, labelsCount, ((uint64_t(labelsCount) * uint64_t(labelSize) + 7ull) / 8ull) / (1024.0*1024));
		do_benchmark(labelSize, labelsCount);
		return 0;
	}
	if (runTest) {
		printf("Test: Label size: %u, count %u, buffer %.1fM\n", labelSize, labelsCount, ((uint64_t(labelsCount) * uint64_t(labelSize) + 7ull) / 8ull) / (1024.0 * 1024));
		do_test(labelSize, labelsCount);
		return 0;
	}
#ifdef WIN32
	printf("\nPress any key to continue...\n");
	_getch();
#endif
	return 0;
}
