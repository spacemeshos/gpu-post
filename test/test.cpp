#include "test.hpp"
#ifdef WIN32
#include <conio.h>
#endif
#include "test-vectors.h"

void do_unit_tests();
void do_integration_tests();
int test_variable_label_length();
int test_variable_labels_count();
int test_of_concurrency();
int test_of_cancelation();

#define	MAX_CPU_LABELS_COUNT	(9 * 128 * 1024)

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

void do_test(int aLabelSize, int aLabelsCount, int aReferenceProvider, bool aPrintResult)
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

			if (aReferenceProvider < 0 || aReferenceProvider >= providersCount) {
				if (referenceLabelsCount > MAX_CPU_LABELS_COUNT) {
					referenceLabelsCount = MAX_CPU_LABELS_COUNT;
				}
				// Find CPU provider and compute reference labels
				for (i = 0; i < providersCount; i++) {
					if (providers[i].compute_api == COMPUTE_API_CLASS_CPU) {
						referenceLabels = out + i * labelsBufferAllignedSize;
						memset(referenceLabels, 0, labelsBufferSize);
						scryptPositions(providers[i].id, id, 0, referenceLabelsCount - 1, aLabelSize, salt, 0, referenceLabels, 512, 1, 1, &hashes_computed, &hashes_per_sec);
						printf("%s: %u hashes, %u h/s\n", providers[i].model, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);
						aReferenceProvider = i;
						checkOuitput = true;
						break;
					}
				}
			}
			else {
				referenceLabels = out + aReferenceProvider * labelsBufferAllignedSize;
				memset(referenceLabels, 0, labelsBufferSize);
				scryptPositions(providers[aReferenceProvider].id, id, 0, referenceLabelsCount - 1, aLabelSize, salt, 0, referenceLabels, 512, 1, 1, &hashes_computed, &hashes_per_sec);
				printf("%s: %u hashes, %u h/s\n", providers[aReferenceProvider].model, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);
				checkOuitput = true;
			}

			for (i = 0; i < providersCount; i++) {
				if (i != aReferenceProvider && providers[i].compute_api != COMPUTE_API_CLASS_CPU) {
					uint8_t *labels = out + i * labelsBufferAllignedSize;
					memset(labels, 0, labelsBufferSize);
					scryptPositions(providers[i].id, id, 0, aLabelsCount - 1, aLabelSize, salt, 0, labels, 512, 1, 1, &hashes_computed, &hashes_per_sec);
					printf("%s: %u hashes, %u h/s\n", providers[i].model, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);
					if (checkOuitput) {
						size_t referencelabelsBufferSize = (size_t(referenceLabelsCount) * size_t(aLabelSize)) / 8ull;
						if (0 != memcmp(referenceLabels, labels, referencelabelsBufferSize)) {
							printf("WRONG result for label size %d from provider %d [%s]\n", aLabelSize, i, providers[i].model);
							if (aPrintResult) {
								const uint8_t *ref = referenceLabels;
								const uint8_t *res = labels;
								for (size_t i = 0; i < referencelabelsBufferSize / 32; i++) {
									for (int j = 0; j < 32; j++, ref++, res++) {
										if (*ref == *res) {
											printf("%02x=%02x ", *ref, *res);
										}
										else {
											printf("%02x!%02x ", *ref, *res);
										}
									}
									printf("\n");
								}
							}
						}
					}
				}
			}

			free(out);
		}

		free(providers);
	}
}

const char * getProviderClassString(ComputeApiClass aClass)
{
	switch (aClass) {
	case COMPUTE_API_CLASS_UNSPECIFIED:
		return "UNSPECIFIED";
	case COMPUTE_API_CLASS_CPU:
		return "CPU";
	case COMPUTE_API_CLASS_CUDA:
		return "CUDA";
	case COMPUTE_API_CLASS_VULKAN:
		return "VULKAN";
	default:
		return "INVALID";
	}
}

void do_providers_list()
{
	int providersCount = spacemesh_api_get_providers(NULL, 0);

	if (providersCount > 0) {
		PostComputeProvider *providers = (PostComputeProvider *)malloc(providersCount * sizeof(PostComputeProvider));

		if (spacemesh_api_get_providers(providers, providersCount) == providersCount) {
			printf("Available POST compute providers:\n");
			for (int i = 0; i < providersCount; i++) {
				printf("%3d: [%s] %s\n", i, getProviderClassString(providers[i].compute_api), providers[i].model);
			}
		}

		free(providers);
	}
	else {
		printf("There are no POST computation providers available.\n");
	}
}

bool do_test_vector(const TestVector *aTestVector, bool aPrintResult)
{
	bool ok = false;
	int providersCount = spacemesh_api_get_providers(NULL, 0);

	printf("Check test vector...\n");

	if (providersCount > 0) {
		PostComputeProvider *providers = (PostComputeProvider *)malloc(providersCount * sizeof(PostComputeProvider));

		if (spacemesh_api_get_providers(providers, providersCount) == providersCount) {
			for (int i = 0; i < providersCount; i++) {
				if (providers[i].compute_api == COMPUTE_API_CLASS_CPU) {
					const size_t labelsBufferSize = (size_t(aTestVector->labelsCount) * size_t(aTestVector->labelSize) + 7ull) / 8ull;
					uint8_t *out;
					uint64_t hashes_computed;
					uint64_t hashes_per_sec;

					out = (uint8_t*)calloc(1, labelsBufferSize);

					scryptPositions(providers[i].id, aTestVector->id, 0, aTestVector->labelsCount - 1, aTestVector->labelSize, aTestVector->salt, 0, out, 512, 1, 1, &hashes_computed, &hashes_per_sec);
					printf("Test vector: %s: %u hashes, %u h/s\n", providers[i].model, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);

					if (0 != memcmp(aTestVector->result, out, labelsBufferSize)) {
						printf("WRONG result for label size %d from provider %d [%s]\n", aTestVector->labelSize, i, providers[i].model);
						if (aPrintResult) {
							const uint8_t *ref = aTestVector->result;
							const uint8_t *res = out;
							for (size_t i = 0; i < labelsBufferSize / 32; i++) {
								for (int j = 0; j < 32; j++, ref++, res++) {
									if (*ref == *res) {
										printf("%02x=%02x ", *ref, *res);
									}
									else {
										printf("%02x!%02x ", *ref, *res);
									}
								}
								printf("\n");
							}
						}
					}
					else {
						ok = true;
						printf("OK result for label size %d from provider %d [%s]\n", aTestVector->labelSize, i, providers[i].model);
					}

					free(out);

					break;
				}
			}
		}

		free(providers);
	}
	else {
		printf("There are no POST computation providers available.\n");
	}

	return ok;
}

void create_test_vector()
{
	int providersCount = spacemesh_api_get_providers(NULL, 0);

	printf("Create test vector...\n");

	if (providersCount > 0) {
		PostComputeProvider *providers = (PostComputeProvider *)malloc(providersCount * sizeof(PostComputeProvider));

		if (spacemesh_api_get_providers(providers, providersCount) == providersCount) {
			for (int i = 0; i < providersCount; i++) {
				if (providers[i].compute_api == COMPUTE_API_CLASS_CPU) {
					static const uint32_t testLabelsCount = 64 * 1024;
					static const uint32_t labelSize = 1;
					static const size_t labelsBufferSize = (size_t(testLabelsCount) * size_t(labelSize) + 7ull) / 8ull;
					uint8_t id[32];
					uint8_t salt[32];
					uint8_t vector[labelsBufferSize];
					uint64_t hashes_computed;
					uint64_t hashes_per_sec;

					memset(id, 0, sizeof(id));
					memset(salt, 0, sizeof(salt));
					memset(vector, 0, sizeof(vector));

					scryptPositions(providers[i].id, id, 0, testLabelsCount - 1, labelSize, salt, 0, vector, 512, 1, 1, &hashes_computed, &hashes_per_sec);
					printf("Test vector: %s: %u hashes, %u h/s\n", providers[i].model, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);

					const uint8_t *src = vector;
					
					for (uint32_t length = sizeof(vector); length > 0; length -= std::min<uint32_t>(32, length)) {
						for (int i = 0; i < 32; i++) {
							printf("0x%02x, ", *src++);
						}
						printf("\n");
					}
					break;
				}
			}
		}

		free(providers);
	}
	else {
		printf("There are no POST computation providers available.\n");
	}
}

int main(int argc, char **argv)
{
	bool runBenchmark = false;
	bool runTest = false;
	bool createTestVector = false;
	bool checkTestVector = false;
	int labelSize = 8;
	int labelsCount = MAX_CPU_LABELS_COUNT;
	int referenceProvider = -1;
	bool printDataCompare = false;

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
		else if (0 == strcmp(argv[i], "--test-vector-create")) {
			createTestVector = true;
		}
		else if (0 == strcmp(argv[i], "--test-vector-check")) {
			checkTestVector = true;
		}
		else if (0 == strcmp(argv[i], "--list") || 0 == strcmp(argv[i], "-l")) {
			do_providers_list();
			return 0;
		}
		else if (0 == strcmp(argv[i], "--unit-tests") || 0 == strcmp(argv[i], "-u")) {
			do_unit_tests();
			return 0;
		}
		else if (0 == strcmp(argv[i], "--integration-tests") || 0 == strcmp(argv[i], "-i")) {
			do_integration_tests();
			return 0;
		}
		else if (0 == strcmp(argv[i], "--integration-test-length") || 0 == strcmp(argv[i], "-il")) {
			return test_variable_label_length();
		}
		else if (0 == strcmp(argv[i], "--integration-test-labels") || 0 == strcmp(argv[i], "-in")) {
			return test_variable_labels_count();
		}
		else if (0 == strcmp(argv[i], "--integration-test-concurrency") || 0 == strcmp(argv[i], "-ip")) {
			return test_of_concurrency();
		}
		else if (0 == strcmp(argv[i], "--integration-test-cancelation") || 0 == strcmp(argv[i], "-ic")) {
			return test_of_cancelation();
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
		else if (0 == strcmp(argv[i], "--reference-provider") || 0 == strcmp(argv[i], "-r")) {
			i++;
			if (i < argc) {
				referenceProvider = atoi(argv[i]);
			}
		}
		else if (0 == strcmp(argv[i], "--print") || 0 == strcmp(argv[i], "-p")) {
			printDataCompare = true;
		}
	}
	if (createTestVector) {
		create_test_vector();
		return 0;
	}
	if (checkTestVector) {
		return do_test_vector(&test_vector_1_64k, printDataCompare) ? 0 : 1;
	}
	if (runBenchmark) {
		printf("Benchmark: Label size: %u, count %u, buffer %.1fM\n", labelSize, labelsCount, ((uint64_t(labelsCount) * uint64_t(labelSize) + 7ull) / 8ull) / (1024.0*1024));
		do_benchmark(labelSize, labelsCount);
		return 0;
	}
	if (runTest) {
		printf("Test: Label size: %u, count %u, buffer %.1fM\n", labelSize, labelsCount, ((uint64_t(labelsCount) * uint64_t(labelSize) + 7ull) / 8ull) / (1024.0 * 1024));
		do_test(labelSize, labelsCount, referenceProvider, printDataCompare);
		return 0;
	}
#ifdef WIN32
	printf("\nPress any key to continue...\n");
	_getch();
#endif
	return 0;
}
