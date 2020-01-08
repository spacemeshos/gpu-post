#include "../include/api.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef WIN32
#include <conio.h>
#endif

#define	LABELS_COUNT	25000

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
	uint8_t *out[3];

	out[0] = (uint8_t *)malloc(LABELS_COUNT);
	memset(out[0], 0, LABELS_COUNT);

	out[1] = (uint8_t *)malloc(LABELS_COUNT);
	memset(out[1], 0, LABELS_COUNT);

	out[2] = (uint8_t *)malloc(LABELS_COUNT);
	memset(out[2], 0, LABELS_COUNT);

	scryptPositions(id, 0, LABELS_COUNT - 1, 8, salt, SPACEMESH_API_CPU, out[0], 512, 1, 1);
	scryptPositions(id, 0, LABELS_COUNT - 1, 8, salt, SPACEMESH_API_CUDA/*   | SPACEMESH_API_THROTTLED_MODE*/, out[1], 512, 1, 1);
	scryptPositions(id, 0, LABELS_COUNT - 1, 8, salt, SPACEMESH_API_VULKAN/* | SPACEMESH_API_THROTTLED_MODE*/, out[2], 512, 1, 1);
//	scryptPositions(id, 0, LABELS_COUNT - 1, 8, salt, SPACEMESH_API_OPENCL/* | SPACEMESH_API_THROTTLED_MODE*/, out[2], 512, 1, 1);

	print(out[0]);
	print(out[1]);
	print(out[2]);

	if (0 != memcmp(out[0], out[1], LABELS_COUNT)) {
		printf("WRONG result from CUDA\n");
	}

	if (0 != memcmp(out[0], out[2], LABELS_COUNT)) {
		printf("WRONG result from OpenCL:\n");
	}

	if (int cookie = spacemesh_api_lock_gpu(SPACEMESH_API_CUDA)) {
		for (int i = 0; i < 5; i++) {
			scryptPositions(id, 0, LABELS_COUNT - 1, 8, salt, cookie, out[1], 512, 1, 1);
			if (0 != memcmp(out[0], out[1], LABELS_COUNT)) {
				printf("WRONG result from CUDA\n");
			}
		}
		spacemesh_api_unlock_gpu(cookie);
	}

	if (int cookie = spacemesh_api_lock_gpu(SPACEMESH_API_OPENCL)) {
		for (int i = 0; i < 5; i++) {
			scryptPositions(id, 0, LABELS_COUNT - 1, 8, salt, cookie, out[2], 512, 1, 1);
			if (0 != memcmp(out[0], out[2], LABELS_COUNT)) {
				printf("WRONG result from OpenCL:\n");
			}
		}
		spacemesh_api_unlock_gpu(cookie);
	}

	free(out[0]);
	free(out[1]);
	free(out[2]);

#ifdef WIN32
	printf("\nPress any key to continue...\n");
	_getch();
#endif
	
	return 0;
}

