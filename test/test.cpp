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

#define	DO_COMPARE_RESULTS	0
#define	LABELS_COUNT	2500000
//#define	LABELS_COUNT	249999
//#define	LABELS_COUNT	32768
//#define	LABELS_COUNT	(3*32*1024 - 1)
#define	LABEL_SIZE		16

static void print(uint8_t *data)
{
	for (int i = 0; i < 32; i++) {
		printf("%02x ", data[i]);
	}
	printf("\n");
}

void do_benchmark(int aLabelSize, int aLabelsCount)
{
	uint8_t id[32] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
	uint8_t salt[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

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

int main(int argc, char **argv)
{
	bool runBenchmark = false;
	int labelSize = 8;
	int labelsCount = 250000;
	if (argc == 1) {
		printf("Usage:\n");
		return 0;
	}
	for (int i = 1; i < argc; i++) {
		if (0 == strcmp(argv[i], "--benchmark") || 0 == strcmp(argv[i], "-b")) {
			runBenchmark = true;
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
		printf("Label size: %u, count %u, buffer %.1fM\n", labelSize, labelsCount, ((uint64_t(labelsCount) * uint64_t(labelSize) + 7ull) / 8ull) / (1024.0*1024));
		do_benchmark(labelSize, labelsCount);
		return 0;
	}
#if 0
	uint8_t id[32] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
	uint8_t salt[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t *out[4];
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
			int i;
			uint8_t *out = (uint8_t *)malloc(providersCount * LABELS_COUNT * 32);
			bool checkOuitput = false;
			int cpu;

			for (uint8_t labelSize = 37; labelSize > 29; labelSize--) {
#if LABELS_COUNT <= 250000
				for (i = 0; i < providersCount; i++) {
					if (providers[i].compute_api == COMPUTE_API_CLASS_CPU) {
						memset(out + i * LABELS_COUNT, 0, LABELS_COUNT);
						scryptPositions(providers[i].id, id, 0, LABELS_COUNT - 1, labelSize, salt, 0, out + i * 32 * LABELS_COUNT, 512, 1, 1, NULL, NULL);
						cpu = i;
						checkOuitput = true;
						break;
					}
				}
#endif
				for (i = 0; i < providersCount; i++) {
					if (providers[i].compute_api != COMPUTE_API_CLASS_CPU) {
						memset(out + i * LABELS_COUNT, 0, LABELS_COUNT);
						scryptPositions(providers[i].id, id, 0, LABELS_COUNT - 1, labelSize, salt, 0, out + i * 32 * LABELS_COUNT, 512, 1, 1, NULL, NULL);
						if (checkOuitput) {
#if 0
							if (0 != memcmp(out + cpu * LABELS_COUNT, out + i * LABELS_COUNT, (LABELS_COUNT * labelSize + 7) / 8)) {
								printf("WRONG result for label size %d from provider %d [%s]\n", labelSize, i, providers[i].model);
							}
#else
							uint8_t *cpuSrc = out + cpu * 32 * LABELS_COUNT;
							uint8_t *gpuSrc = out + i * 32 * LABELS_COUNT;
							volatile unsigned errors = 0;
							static volatile struct {
								unsigned	pos;
								uint8_t		cpu;
								uint8_t		gpu;
							} errorInfo[2048];
							for (unsigned pos = 0; pos < ((LABELS_COUNT * labelSize + 7) / 8); pos++) {
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
								printf("WRONG result for label size %d (%u) from provider %d [%s]\n", labelSize, errors, i, providers[i].model);
							}
#endif
						}
					}
				}
			}

			free(out);
		}

		free(providers);
	}
#endif
#ifdef WIN32
	printf("\nPress any key to continue...\n");
	_getch();
#endif
	return 0;
}
