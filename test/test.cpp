#include "../include/api.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef WIN32
#include <conio.h>
#endif

#ifndef min
# define min(a, b)  ((a) < (b) ? (a) : (b))
#endif

#define	DO_COMPARE_RESULTS	0
#define	LABELS_COUNT	25000
#define	LABEL_SIZE		8

static void print(uint8_t *data)
{
	for (int i = 0; i < 32; i++) {
		printf("%02x ", data[i]);
	}
	printf("\n");
}

int main()
{
	uint8_t id[32] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
	uint8_t salt[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t *out[4];
	int providersCount = spacemesh_api_get_providers(NULL, 0);

	if (providersCount > 0) {
		PostComputeProvider *providers = (PostComputeProvider *)malloc(providersCount * sizeof(PostComputeProvider));

		if (spacemesh_api_get_providers(providers, providersCount) == providersCount) {
			int i;
			uint8_t *out = (uint8_t *)malloc(providersCount * LABELS_COUNT);
			bool checkOuitput = false;
			int cpu;

#if LABELS_COUNT <= 250000
			for (i = 0; i < providersCount; i++) {
				if (providers[i].compute_api == COMPUTE_API_CLASS_CPU) {
					memset(out + i * LABELS_COUNT, 0, LABELS_COUNT);
					scryptPositions(providers[i].id, id, 0, LABELS_COUNT - 1, LABEL_SIZE, salt, 0, out + i * LABELS_COUNT, 512, 1, 1, NULL, NULL);
					cpu = i;
					checkOuitput = true;
					break;
				}
			}
#endif
			for (i = 0; i < providersCount; i++) {
				if (providers[i].compute_api != COMPUTE_API_CLASS_CPU) {
					memset(out + i * LABELS_COUNT, 0, LABELS_COUNT);
					scryptPositions(providers[i].id, id, 0, LABELS_COUNT - 1, LABEL_SIZE, salt, 0, out + i * LABELS_COUNT, 512, 1, 1, NULL, NULL);
					if (checkOuitput) {
						if (0 != memcmp(out + i * LABELS_COUNT, out + cpu * LABELS_COUNT, LABELS_COUNT)) {
							printf("WRONG result from provider %d [%s]\n", i, providers[i].model);
						}
					}
				}
			}
			free(out);
		}

		free(providers);
	}

#ifdef WIN32
	printf("\nPress any key to continue...\n");
	_getch();
#endif

	return 0;
}
