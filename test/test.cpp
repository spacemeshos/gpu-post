#include "test.hpp"
#ifdef WIN32
#include <conio.h>
#endif
#include "test-vectors.h"
#include <memory>

int do_unit_tests();
void do_integration_tests();
int test_variable_label_length();
int test_variable_labels_count();
int test_of_concurrency();
int test_of_cancelation();

#define	MAX_CPU_LABELS_COUNT	(1 * 128 * 1024)
#define MAX_LABELS_COUNT_API_CALL (32 * 1024 * 1024)

static uint8_t s_id[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static uint8_t s_salt[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

static const uint8_t zeros[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

void * memstr(const void *src, size_t length, const uint8_t *token, int token_length)
{
	for (const uint8_t *cp = (const uint8_t *)src; length >= token_length; cp++, length--) {
		if (0 == memcmp(cp, token, token_length)) {
			return (void*)cp;
		}
	}
	return 0;
}

void print_hex32(const uint8_t *aSrc)
{
	printf("0x");
	for (int i = 0; i < 32; i++) {
		printf("%02x", aSrc[i]);
	}
}

inline bool isHexDigits(char aChar)
{
	return (aChar >= '0' && aChar <= '9') || (aChar >= 'a' && aChar <= 'f') || (aChar >= 'A' && aChar <= 'F');
}

inline char hexToInt(char aChar)
{
	if (aChar >= '0' && aChar <= '9') {
		return aChar - '0';
	}
	else if (aChar >= 'a' && aChar <= 'f') {
		return 10 + (aChar - 'a');
	}
	else if (aChar >= 'A' && aChar <= 'F') {
		return 10 + (aChar - 'A');
	}
	return 0;
}

int hex2bin(const char *aSrc, unsigned char *aDst, size_t aDstLen)
{
	if (aSrc[0] == '0' && aSrc[1] == 'x') {
		aSrc += 2;
	}
	if (aDstLen >= strlen(aSrc) / 2) {
		if (0 == aSrc[0]) {
			return 0;
		}
		aDstLen = 0;
		const char *cp = aSrc;
		while (*cp) {
			if (!isHexDigits(cp[0]) || !isHexDigits(cp[1])) {
				return -2;
			}
			*aDst = (hexToInt(cp[0]) << 4) | hexToInt(cp[1]);
			aDst++;
			cp += 2;
			aDstLen++;
		}
		return (int)aDstLen;
	}
	return -1;
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
				if (providers[i].compute_api != COMPUTE_API_CLASS_CPU)
				{
					uint64_t hashes_computed;
					uint64_t hashes_per_sec;
					int status = scryptPositions(providers[i].id, id, 0, aLabelsCount - 1, aLabelSize, salt, SPACEMESH_API_COMPUTE_LEAFS, out, 512, 1, 1, NULL, NULL, &hashes_computed, &hashes_per_sec);
					printf("%s: status %d, %u hashes, %u h/s\n", providers[i].model, status, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);
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
						uint64_t idx_solution = -1;
						uint8_t D[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
						referenceLabels = out + i * labelsBufferAllignedSize;
						memset(referenceLabels, 0, labelsBufferSize);
						scryptPositions(providers[i].id, id, 0, referenceLabelsCount - 1, aLabelSize, salt, SPACEMESH_API_COMPUTE_LEAFS, referenceLabels, 512, 1, 1, D, &idx_solution, &hashes_computed, &hashes_per_sec);
						printf("%s: %u hashes, %u h/s\n", providers[i].model, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);
						aReferenceProvider = i;
						checkOuitput = true;
						break;
					}
				}
			}
			else {
				uint64_t idx_solution = -1;
				uint8_t D[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
				referenceLabels = out + aReferenceProvider * labelsBufferAllignedSize;
				memset(referenceLabels, 0, labelsBufferSize);
				scryptPositions(providers[aReferenceProvider].id, id, 0, referenceLabelsCount - 1, aLabelSize, salt, SPACEMESH_API_COMPUTE_LEAFS, referenceLabels, 512, 1, 1, D, &idx_solution, &hashes_computed, &hashes_per_sec);
				printf("%s: %u hashes, %u h/s\n", providers[aReferenceProvider].model, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);
				checkOuitput = true;
			}

			for (i = 0; i < providersCount; i++) {
				if (i != aReferenceProvider && providers[i].compute_api != COMPUTE_API_CLASS_CPU) {
					uint64_t idx_solution = -1;
					uint8_t D[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
					uint8_t *labels = out + i * labelsBufferAllignedSize;
					memset(labels, 0, labelsBufferSize);
					scryptPositions(providers[i].id, id, 0, aLabelsCount - 1, aLabelSize, salt, SPACEMESH_API_COMPUTE_LEAFS, labels, 512, 1, 1, D, &idx_solution, &hashes_computed, &hashes_per_sec);
					printf("%s: %u hashes, %u h/s\n", providers[i].model, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);
					if (memstr(labels, labelsBufferSize, zeros, 8)) {
						printf("ZEROS result\n");
					}
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

// compute pos and compute pow - the core lib use case here
void test_core(int aLabelsCount, unsigned aDiff, unsigned aSeed, int labelSize)
{
	uint8_t id[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	uint8_t salt[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	int providersCount = spacemesh_api_get_providers(NULL, 0);

	if (aSeed) {
		if (aSeed == -1) {
			srand(time(nullptr));
		}
		else {
			srand(aSeed);
		}

		for (int i = 0; i < sizeof(id); i++) {
			id[i] = rand();
		}
		for (int i = 0; i < sizeof(salt); i++) {
			salt[i] = rand();
		}
	}

	uint8_t D[32] = { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };

	if (aDiff) {
		int i = 0;
		while (aDiff >= 8) {
			D[i] = 0;
			i++;
			aDiff -= 8;
		}
		if (aDiff) {
			D[i] = (1 << (8 - aDiff)) - 1;
		}
	}

	printf("Target D: ");
	print_hex32(D);
	printf("\n");

	if (providersCount > 0) {
		PostComputeProvider *providers = (PostComputeProvider *)malloc(providersCount * sizeof(PostComputeProvider));

		if (spacemesh_api_get_providers(providers, providersCount) == providersCount) {
			uint32_t cpu_id = -1;
			for (int i = 0; i < providersCount; i++) {
				if (providers[i].compute_api == COMPUTE_API_CLASS_CPU) {
					cpu_id = providers[i].id;
					break;
				}
			}
			for (int i = 0; i < providersCount; i++) {
				if (providers[i].compute_api != COMPUTE_API_CLASS_CPU)
				{
					uint64_t labels_per_iter = aLabelsCount;
					int iters = 1;

					if (aLabelsCount > MAX_LABELS_COUNT_API_CALL) {
						labels_per_iter = MAX_LABELS_COUNT_API_CALL;
						iters = aLabelsCount / MAX_LABELS_COUNT_API_CALL;
					}
					printf("Labels+Pow Task. Iters: %u, Total labels: %d, Labels per iter: %lu. Label size (bits): %u\n", iters,  aLabelsCount, labels_per_iter, labelSize);

					const uint64_t labelsBufferSize = (uint64_t(labels_per_iter) * uint64_t(labelSize) + 7ull) / 8ull;
					uint8_t *out = (uint8_t *)malloc(labelsBufferSize);

					printf("buffer size: %0.1f MiB\n", labelsBufferSize / (1024.0*1024));

					uint64_t idx = 0;
					uint64_t idx_solution = -1ull;
					uint64_t hashes_computed;
					uint64_t hashes_per_sec;

					for (int j=0; j < iters; j++) {

						hashes_computed = 0;
						hashes_per_sec = 0;

						if (idx_solution == -1ull) {
							printf("Compute labels and look for a pow solution... Iteration: %d\n", j);
							int status = scryptPositions(providers[i].id, id, idx, idx + labels_per_iter - 1, labelSize, salt, SPACEMESH_API_COMPUTE_LEAFS | SPACEMESH_API_COMPUTE_POW, out, 512, 1, 1, D, &idx_solution, &hashes_computed, &hashes_per_sec);

							if (status != SPACEMESH_API_ERROR_NONE && status != SPACEMESH_API_POW_SOLUTION_FOUND) {
								printf("Compute error: %u\n", status);
							}

							if (hashes_computed != labels_per_iter) {
								printf ("Compute error: hashes computed: %lu, hashes expected: %lu\n", hashes_computed, labels_per_iter);
							}

							printf("Labels + pow: requested %lu labels at index %lu. Computed hashes: %lu, (%lu h/s)\n", labels_per_iter, idx, hashes_computed, hashes_per_sec);

							if (idx_solution != -1ull) {
								printf("Found pow solution at index %lu\n", idx_solution);
							} else {
								printf("Pow solution not found in this iteration.\n");
							}
						} else {
							// solution was found - compute labels only and don't overwrite hash
							printf("Compute labels only... Iteration: %d\n", j);

							uint64_t idx_temp = -1;
							int status = scryptPositions(providers[i].id, id, idx, idx + labels_per_iter - 1, labelSize, salt, SPACEMESH_API_COMPUTE_LEAFS, out, 512, 1, 1, D, &idx_temp, &hashes_computed, &hashes_per_sec);

							if (status != SPACEMESH_API_ERROR_NONE && status != SPACEMESH_API_POW_SOLUTION_FOUND) {
								printf("Compute returned an error: %u", status);
							}

							printf("Compute labels only. Requested: %lu labels at index %lu. hHshes computed: %lu. (%lu h/s). Solution index:  %lu\n", labels_per_iter, idx, hashes_computed, hashes_per_sec, idx_solution);
						}

						// start index of next iteration
						idx += labels_per_iter;
					}

					// finished leaves computation - we need to continue look for a pow solution until one is found, if was not found yet
					while (idx_solution == -1ull) {
						hashes_computed = 0;
						hashes_per_sec = 0;

						printf("Calling pow compute...\n");

						int status = scryptPositions(providers[i].id, id, idx, idx + labels_per_iter - 1, labelSize, salt, SPACEMESH_API_COMPUTE_POW, out, 512, 1, 1, D, &idx_solution, &hashes_computed, &hashes_per_sec);

						printf("Compute pow only at index: %lu. hashes computed: %lu (%lu h/s)\n", idx, hashes_computed, hashes_per_sec);

						switch(status) {
							case SPACEMESH_API_POW_SOLUTION_FOUND:
								break;

							case SPACEMESH_API_ERROR_NONE:
								if (hashes_computed != labels_per_iter) {
									printf("Error: compute diff than expected: hashes computed: %lu, labels requested: %lu\n", hashes_computed, labels_per_iter);
								}
								break;
							default:
								printf("Error %d. Hashes computed: %lu, (%lu h/s)\n", status, hashes_computed, hashes_per_sec);
								break;
						}

						// set index for next iteration
						idx += labels_per_iter;
					}

				 	if (idx_solution != -1ull) {

						printf("Pow solution at %u\n", (uint32_t)idx_solution);

						// compute 256 hash at solution index:
						uint8_t hash[32];
						scryptPositions(cpu_id, id, idx_solution, idx_solution, 256, salt, SPACEMESH_API_COMPUTE_LEAFS, hash, 512, 1, 1, NULL, NULL, &hashes_computed, &hashes_per_sec);

						printf("D: ");
						print_hex32(D);
						printf("\n");
						printf("H: ");
						print_hex32(hash);
						printf("\n");
					} else {
						printf("error: no pow solution found");
					}

				}
			}
		}
		free(providers);
	}
}

int do_test_pow(uint64_t aStartPos, int aLabelsCount, unsigned aDiff, unsigned aSeed, int aProviderId, uint64_t aSolutionIdx)
{
	int providersCount = spacemesh_api_get_providers(NULL, 0);

	if (aSeed) {
		if (aSeed == -1) {
			srand(time(nullptr));
		}
		else {
			srand(aSeed);
		}

		for (int i = 0; i < sizeof(s_id); i++) {
			s_id[i] = rand();
		}
		for (int i = 0; i < sizeof(s_salt); i++) {
			s_salt[i] = rand();
		}
	}

	if (providersCount > 0) {
		std::unique_ptr<PostComputeProvider> providers_holder((PostComputeProvider *)malloc(providersCount * sizeof(PostComputeProvider)));
		PostComputeProvider *providers = providers_holder.get();

		if (spacemesh_api_get_providers(providers, providersCount) == providersCount) {
			uint64_t idx_solution = -1;
			uint8_t D[32] = { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
			if (aDiff) {
				int i = 0;
				while (aDiff >= 8) {
					D[i] = 0;
					i++;
					aDiff -= 8;
				}
				if (aDiff) {
					D[i] = (1 << (8 - aDiff)) - 1;
				}
			}
			printf("Target D: ");
			print_hex32(D);
			printf("\n");
			uint32_t cpu_id = -1;
			for (int i = 0; i < providersCount; i++) {
				if (providers[i].compute_api == COMPUTE_API_CLASS_CPU) {
					cpu_id = providers[i].id;
					break;
				}
			}
			for (int i = 0; i < providersCount; i++) {
				if (providersCount == 1 || providers[i].compute_api != COMPUTE_API_CLASS_CPU)
				{
					if (-1 != aProviderId && aProviderId != providers[i].id) {
						continue;
					}
					uint64_t hashes_computed;
					uint64_t hashes_per_sec;
					printf("%s: ", providers[i].model);
					int status = scryptPositions(providers[i].id, s_id, aStartPos, aStartPos + aLabelsCount - 1, 8, s_salt, SPACEMESH_API_COMPUTE_POW, NULL, 512, 1, 1, D, &idx_solution, &hashes_computed, &hashes_per_sec);
					switch (status) {
					case SPACEMESH_API_POW_SOLUTION_FOUND:
						printf("%u hashes, %u h/s, solution at %u\n", (uint32_t)hashes_computed, (uint32_t)hashes_per_sec, (uint32_t)idx_solution);
						if (-1 != cpu_id) {
							uint8_t hash[32];
							scryptPositions(cpu_id, s_id, idx_solution, idx_solution, 256, s_salt, SPACEMESH_API_COMPUTE_LEAFS, hash, 512, 1, 1, NULL, NULL, &hashes_computed, &hashes_per_sec);
							printf("id: ");
							print_hex32(s_id);
							printf("\n");
							printf("salt: ");
							print_hex32(s_salt);
							printf("\n");
							printf("D: ");
							print_hex32(D);
							printf("\n");
							printf("H: ");
							print_hex32(hash);
							printf("\n");
						}
						break;
					case SPACEMESH_API_ERROR_NONE:
						printf("%u hashes, %u h/s, solution not found\n", (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);
						break;
					default:
						printf("error %d, %u hashes, %u h/s\n", status, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);
					}
					if (aSolutionIdx != 0xffffffffffffffffull) {
						if (idx_solution != aSolutionIdx) {
							printf("WRONG solution index %ull, expected %ull\n", idx_solution, aSolutionIdx);
							return 1;
						}
					}
				}
			}

			return 0;
		}
	}

	return 1;
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
				printf("%3d: [%s] %s\n", providers[i].id, getProviderClassString(providers[i].compute_api), providers[i].model);
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
		std::unique_ptr<PostComputeProvider> providers_holder((PostComputeProvider *)malloc(providersCount * sizeof(PostComputeProvider)));
		PostComputeProvider *providers = providers_holder.get();

		if (spacemesh_api_get_providers(providers, providersCount) == providersCount) {
			for (int i = 0; i < providersCount; i++) {
				if (providers[i].compute_api == COMPUTE_API_CLASS_CPU) {
					const size_t labelsBufferSize = (size_t(aTestVector->labelsCount) * size_t(aTestVector->labelSize) + 7ull) / 8ull;
					uint64_t hashes_computed;
					uint64_t hashes_per_sec;
					uint64_t idx_solution = -1;
					uint8_t D[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

					std::unique_ptr<uint8_t> out((uint8_t*)calloc(1, labelsBufferSize));

					scryptPositions(providers[i].id, aTestVector->id, 0, aTestVector->labelsCount - 1, aTestVector->labelSize, aTestVector->salt, SPACEMESH_API_COMPUTE_LEAFS, out.get(), 512, 1, 1, D, &idx_solution, &hashes_computed, &hashes_per_sec);
					printf("Test vector: %s: %u hashes, %u h/s\n", providers[i].model, (uint32_t)hashes_computed, (uint32_t)hashes_per_sec);

					if (0 != memcmp(aTestVector->result, out.get(), labelsBufferSize)) {
						printf("WRONG result for label size %d from provider %d [%s]\n", aTestVector->labelSize, i, providers[i].model);
						if (aPrintResult) {
							const uint8_t *ref = aTestVector->result;
							const uint8_t *res = out.get();
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

					break;
				}
			}
		}
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
					uint64_t idx_solution = -1;
					uint8_t D[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

					memset(id, 0, sizeof(id));
					memset(salt, 0, sizeof(salt));
					memset(vector, 0, sizeof(vector));

					scryptPositions(providers[i].id, id, 0, testLabelsCount - 1, labelSize, salt, SPACEMESH_API_COMPUTE_LEAFS, vector, 512, 1, 1, D, &idx_solution, &hashes_computed, &hashes_per_sec);
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

void print_usage() {
	printf("Usage:\n");
	printf("--list               or -l	print available providers\n");
	printf("--benchmark          or -b	run benchmark\n");
	printf("--core               or -c	test the core library use case\n");
	printf("--test               or -t	run basic test\n");
	printf("--test-vector-check		run a CPU test and compare with test-vector\n");
	printf("--test-pow           or -tp 	test pow computation\n");
	printf("--unit-tests         or -u 	run unit tests\n");
	printf("--integration-tests  or -i  	run integration tests\n");
	printf("--label-size         or -s	<1-256>	set label size [1-256]\n");
	printf("--labels-count       or -n	<1-32M>	set labels count [up to 32M]\n");
	printf("--reference-provider or -r	<id> the result of this provider will be used as a reference [default - CPU]\n");
	printf("--print              or -p	print detailed data comparison report for incorrect results\n");
	printf("--pow-diff           or -d 	<0-256> count of leading zero bits in target D value [default - 16]\n");
	printf("--srand-seed         or -ss	<unsigned int> set srand seed value for POW test: 0 - use zero id/seed [default], -1 - use random value\n");
}

int main(int argc, char **argv)
{
	bool runBenchmark = false;
	bool runTest = false;
	bool runTestPow = false;
	bool runTestCore = false;
	bool createTestVector = false;
	bool checkTestVector = false;
	int labelSize = 8;
	int labelsCount = MAX_CPU_LABELS_COUNT;
	int powDiff = 16;
	int referenceProvider = -1;
	unsigned srand_seed = 0;
	bool printDataCompare = false;
	uint64_t startPos = 0;
	uint64_t solutionIdx = 0xffffffffffffffff;

	if (argc == 1) {
		print_usage();
		return 0;
	}
	for (int i = 1; i < argc; i++) {
		if (0 == strcmp(argv[i], "--benchmark") || 0 == strcmp(argv[i], "-b")) {
			runBenchmark = true;
		}
		else if (0 == strcmp(argv[i], "--test") || 0 == strcmp(argv[i], "-t")) {
			runTest = true;
		}
		else if (0 == strcmp(argv[i], "--test-pow") || 0 == strcmp(argv[i], "-tp")) {
			runTestPow = true;
		}
		else if (0 == strcmp(argv[i], "--core") || 0 == strcmp(argv[i], "-c")) {
			runTestCore = true;
		}
		else if (0 == strcmp(argv[i], "--test-vector-create")) {
			createTestVector = true;
		}
		else if (0 == strcmp(argv[i], "--test-vector-check")) {
			checkTestVector = true;
		}
		else if (0 == strcmp(argv[i], "--list") || 0 == strcmp(argv[i], "-l")) {
			do_providers_list();
//			spacemesh_api_shutdown();
			return 0;
		}
		else if (0 == strcmp(argv[i], "--unit-tests") || 0 == strcmp(argv[i], "-u")) {
			return do_unit_tests();
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
		else if (0 == strcmp(argv[i], "--pow-diff") || 0 == strcmp(argv[i], "-d")) {
			i++;
			if (i < argc) {
				powDiff = atoi(argv[i]);
				if (powDiff < 0) {
					powDiff = 0;
				}
				else if (powDiff > 256) {
					powDiff = 256;
				}
			}
		}
		else if (0 == strcmp(argv[i], "--label-size") || 0 == strcmp(argv[i], "-s")) {
			i++;
			if (i < argc) {
				labelSize = atoi(argv[i]);
				if (labelSize < 1) {
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
					labelsCount = MAX_LABELS_COUNT_API_CALL;
				}
			}
		}
		else if (0 == strcmp(argv[i], "--reference-provider") || 0 == strcmp(argv[i], "-r")) {
			i++;
			if (i < argc) {
				referenceProvider = atoi(argv[i]);
			}
		}
		else if (0 == strcmp(argv[i], "--srand-seed") || 0 == strcmp(argv[i], "-ss")) {
			i++;
			if (i < argc) {
				srand_seed = strtoul(argv[i], NULL, 10);
			}
		}
		else if (0 == strcmp(argv[i], "--start-pos") || 0 == strcmp(argv[i], "-sp")) {
			i++;
			if (i < argc) {
				startPos = strtoull(argv[i], NULL, 10);
			}
		}
		else if (0 == strcmp(argv[i], "--solution-idx") || 0 == strcmp(argv[i], "-si")) {
			i++;
			if (i < argc) {
				solutionIdx = strtoull(argv[i], NULL, 10);
			}
		}
		else if (0 == strcmp(argv[i], "-id")) {
			i++;
			if (i < argc) {
				uint8_t buffer[32];
				if (32 == hex2bin(argv[i], buffer, sizeof(buffer))) {
					memcpy(s_id, buffer, 32);
				}
			}
		}
		else if (0 == strcmp(argv[i], "--salt")) {
			i++;
			if (i < argc) {
				uint8_t buffer[32];
				if (32 == hex2bin(argv[i], buffer, sizeof(buffer))) {
					memcpy(s_salt, buffer, 32);
				}
			}
		}
		else if (0 == strcmp(argv[i], "--print") || 0 == strcmp(argv[i], "-p")) {
			printDataCompare = true;
		}
		else if (0 == strcmp(argv[i], "--logs")) {
			spacemesh_api_logging(1);
		}
		else if (0 == strcmp(argv[i], "--false")) {
			return 1;
		}
		else if (0 == strcmp(argv[i], "--true")) {
			return 0;
		}
		else {
			printf("Unknown options: %s\n", argv[i]);
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
		printf("Test LEAFS: Label size: %u, count %u, buffer %.1fM\n", labelSize, labelsCount, ((uint64_t(labelsCount) * uint64_t(labelSize) + 7ull) / 8ull) / (1024.0 * 1024));
		do_test(labelSize, labelsCount, referenceProvider, printDataCompare);
		return 0;
	}
	if (runTestPow) {
		printf("Test POW: count %u\n", labelsCount);
		do_test_pow(startPos, labelsCount, powDiff, srand_seed, referenceProvider, solutionIdx);
		return 0;
	}
	if (runTestCore) {
		printf("Test POS+POW core use case: count %u\n", labelsCount);
		test_core(labelsCount, powDiff, srand_seed, labelSize);
		return 0;
	}
#ifdef WIN32
	printf("\nPress any key to continue...\n");
	_getch();
#endif
	return 0;
}
