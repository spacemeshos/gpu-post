#include "test.hpp"
#include "test-vectors.h"
#include <memory>

static void printHex(uint8_t *data, uint32_t length)
{
	while (length >= 32) {
		for (int i = 0; i < 32; i++) {
			printf("0x%02x, ", data[i]);
		}
		printf("\n");
		length -= 32;
		data += 32;
	}
}

int do_unit_tests()
{
	uint32_t input[20]; // align 16
	int providersCount = spacemesh_api_get_providers(NULL, 0);

	memset(input, 0, sizeof(input));

	if (providersCount > 0) {
		std::unique_ptr<PostComputeProvider> providers_holder((PostComputeProvider *)malloc(providersCount * sizeof(PostComputeProvider)));
		uint8_t hashes[128][32];
		PostComputeProvider *providers = providers_holder.get();

		if (spacemesh_api_get_providers(providers, providersCount) == providersCount) {
			int i;
			for (i = 0; i < providersCount; i++) {
				memset(hashes, 0, sizeof(hashes));
				if (128 != unit_test_hash(providers[i].id, (uint8_t*)input, (uint8_t*)hashes)) {
					printf("[%s]: error compute hashes\n", providers[i].model);
					continue;
				}
				;
				if (memcmp(test_vector_hashes, hashes, sizeof(hashes))) {
					printf("[%s]: hash test WRONG\n", providers[i].model);
					return 1;
				}
				else {
					printf("[%s]: hash test OK\n", providers[i].model);
				}
				{
					uint8_t stream[128 * 32];
					for (uint32_t label_length = 1; label_length <= 256; label_length++) {
						int64_t expected_stream_length = (128 * label_length + 7) / 8;
						memset(stream, 0, sizeof(stream));
						int64_t output_stream_length = unit_test_bit_stream(providers[i].id, (uint8_t*)hashes, 128, stream, label_length);
						if (expected_stream_length != output_stream_length) {
							printf("[%s]: %d bits stream test WRONG\n", providers[i].model, label_length);
							return 1;
						}
						else {
							printf("[%s]: %d bits stream test OK\n", providers[i].model, label_length);
						}
					}
				}
			}
		}

		return 0;
	}

	return 1;
}
