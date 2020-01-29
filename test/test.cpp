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

#define	LABELS_COUNT	25000000

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

	out[0] = (uint8_t *)malloc(LABELS_COUNT);
	memset(out[0], 0, LABELS_COUNT);

	out[1] = (uint8_t *)malloc(LABELS_COUNT);
	memset(out[1], 0, LABELS_COUNT);

	out[2] = (uint8_t *)malloc(LABELS_COUNT);
	memset(out[2], 0, LABELS_COUNT);

	out[3] = (uint8_t *)malloc(LABELS_COUNT);
	memset(out[3], 0, LABELS_COUNT);

#if LABELS_COUNT <= 250000
	scryptPositions(id, 0, LABELS_COUNT - 1, 8, salt, SPACEMESH_API_CPU, out[0], 512, 1, 1);
#endif
	scryptPositions(id, 0, LABELS_COUNT - 1, 8, salt, SPACEMESH_API_CUDA/*   | SPACEMESH_API_THROTTLED_MODE*/, out[1], 512, 1, 1);
	scryptPositions(id, 0, LABELS_COUNT - 1, 8, salt, SPACEMESH_API_OPENCL/* | SPACEMESH_API_THROTTLED_MODE*/, out[2], 512, 1, 1);

	print(out[0]);
	print(out[1]);
	print(out[2]);

	if (0 != memcmp(out[0], out[1], LABELS_COUNT)) {
		printf("WRONG result from CUDA\n");
	}

	if (0 != memcmp(out[0], out[2], LABELS_COUNT)) {
		printf("WRONG result from OpenCL\n");
	}

	int cookie1 = spacemesh_api_lock_gpu(SPACEMESH_API_VULKAN);
	int cookie2 = spacemesh_api_lock_gpu(SPACEMESH_API_VULKAN);

	if (cookie1) {
		scryptPositions(id, 0, LABELS_COUNT - 1, 8, salt, cookie1, out[2], 512, 1, 1);
		print(out[2]);
		if (0 != memcmp(out[1], out[2], LABELS_COUNT)) {
			printf("WRONG result from NVIDIA Vulkan\n");
		}
	}

	if (cookie2) {
		scryptPositions(id, 0, LABELS_COUNT - 1, 8, salt, cookie2, out[3], 512, 1, 1);
		print(out[3]);
		if (0 != memcmp(out[1], out[3], LABELS_COUNT)) {
			printf("WRONG result from AMD Vulkan\n");
		}
	}

	if (cookie1) {
		spacemesh_api_unlock_gpu(cookie1);
	}

	if (cookie2) {
		spacemesh_api_unlock_gpu(cookie2);
	}

	free(out[0]);
	free(out[1]);
	free(out[2]);
	free(out[3]);

#ifdef WIN32
	printf("\nPress any key to continue...\n");
	_getch();
#endif
	
	return 0;
}

