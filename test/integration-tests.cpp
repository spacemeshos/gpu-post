#include "test.hpp"
#include <memory>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>

struct Provider
{
	typedef std::shared_ptr<Provider> Ptr;
	typedef std::vector<Ptr> Vector;

	int						provider_id;
	PostComputeProvider		info;
	uint8_t					id[32];
	uint8_t					salt[32];
	std::vector<uint8_t>	labels;
	uint64_t				hashes_computed;
	std::thread				_thread;
	std::atomic_uint32_t	*_counter = nullptr;
	uint64_t				_start_pos;
	uint64_t				_end_pos;
	uint32_t				_label_length;

	Provider(int aProviderId, const PostComputeProvider &aProviderInfo)
		: provider_id(aProviderId)
		, info(aProviderInfo)
	{
		memset(id, 0, sizeof(id));
		memset(salt, 0, sizeof(salt));
	}

	/* compute labels */
	int compute(uint64_t start_pos, uint64_t end_pos, uint32_t label_length) {
		uint64_t idx_solution = -1;
		uint8_t D[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		size_t labels_count = end_pos - start_pos + 1;
		size_t output_length = (labels_count * label_length + 7) / 8;
		labels.resize(output_length);
		return scryptPositions(provider_id, id, start_pos, end_pos, label_length, salt, SPACEMESH_API_COMPUTE_LEAFS, labels.data(), 512, 1, 1, D, &idx_solution, &hashes_computed, NULL);
	}

	/* compute labels in separate thread */
	void post(uint64_t start_pos, uint64_t end_pos, uint32_t label_length, std::atomic_uint32_t *counter) {
		_start_pos = start_pos;
		_end_pos = end_pos;
		_label_length = label_length;
		_counter = counter;
		_thread = std::thread(&Provider::background, this);
		if (std::thread::id() != _thread.get_id()) {
			if (counter) {
				(*counter)++;
			}
		}
	}

	/* compare computed labels by this provider with computed labels by aRef provider */
	bool equals(const Provider &aRef) const {
		if (labels.size() == aRef.labels.size()) {
			return 0 == memcmp(labels.data(), aRef.labels.data(), labels.size());
		}
		return false;
	}

	/* thread proc */
	void background() {
		compute(_start_pos, _end_pos, _label_length);
		if (_counter) {
			(*_counter)--;
			_counter = nullptr;
		}
	}

	/* wait for thread end */
	void join() {
		if (_thread.joinable()) {
			_thread.join();
		}
	}
};

/* get providers list */
static std::pair<Provider::Ptr, Provider::Vector> getProviders()
{
	Provider::Ptr cpu;
	Provider::Vector gpus;

	int providersCount = spacemesh_api_get_providers(NULL, 0);

	if (providersCount > 0) {
		std::vector<PostComputeProvider> providers(providersCount);

		if (spacemesh_api_get_providers(providers.data(), providersCount) == providersCount) {
			for (int i = 0; i < providersCount; i++) {
				Provider::Ptr provider(new Provider(providers[i].id, providers[i]));
				if (providers[i].compute_api == COMPUTE_API_CLASS_CPU) {
					cpu = provider;
				}
				else {
					gpus.push_back(provider);
				}
			}
		}
	}

	return {cpu, gpus};
}

/* compute labels with various length and compare with others providers */
int test_variable_label_length()
{
	static const uint32_t cLabelsCount = 256;
	std::pair<Provider::Ptr, Provider::Vector> providers{getProviders()};

	for (uint32_t label_length = 1; label_length <= 256; label_length++) {
		bool ok = true;
		printf("Label length %d: ", label_length);
		if (providers.first) {
			if (SPACEMESH_API_ERROR_NONE != providers.first->compute(0, cLabelsCount - 1, label_length)) {
				printf("[CPU]: Error compute labels\n");
				return 1;
			}
		}
		for (auto provider : providers.second) {
			if (SPACEMESH_API_ERROR_NONE != provider->compute(0, cLabelsCount - 1, label_length)) {
				printf("[%s]: Error compute labels\n", provider->info.model);
				ok = false;
			}
		}
		if (providers.first) {
			for (auto provider : providers.second) {
				if (!provider->equals(*providers.first)) {
					printf("[%s]: WRONG result for label length %d\n", provider->info.model, label_length);
					ok = false;
				}
			}
		}
		if (providers.second.size() > 1) {
			for (Provider::Vector::iterator ref = providers.second.begin(); providers.second.end() != ref; ref++) {
				for (Provider::Vector::iterator provider = ref + 1; providers.second.end() != provider; provider++) {
					if (!(*provider)->equals(**ref)) {
						printf("[%s]: compare error with [%s] for label length %d\n", (*provider)->info.model, (*ref)->info.model, label_length);
						ok = false;
					}
				}
			}
		}
		if (ok) {
			printf("OK\n");
		}
		else {
			return 1;
		}
	}

	return 0;
}

/* compute various labels count with various length and compare with others providers */
int test_variable_labels_count()
{
	static const uint32_t cLabelsCount = 16;
	std::pair<Provider::Ptr, Provider::Vector> providers{ getProviders() };

	for (uint32_t label_length = 1; label_length <= 256; label_length++) {
		for (uint32_t labels_count = 1; labels_count <= cLabelsCount; labels_count++) {
			bool ok = true;
			printf("Label length %d, labels count: %d ", label_length, labels_count);
			if (providers.first) {
				if (SPACEMESH_API_ERROR_NONE != providers.first->compute(0, labels_count - 1, label_length)) {
					printf("[CPU]: Error compute labels\n");
					return -1;
				}
			}
			for (auto provider : providers.second) {
				if (SPACEMESH_API_ERROR_NONE != provider->compute(0, labels_count - 1, label_length)) {
					printf("[%s]: Error compute labels\n", provider->info.model);
					ok = false;
				}
			}
			if (providers.first) {
				for (auto provider : providers.second) {
					if (!provider->equals(*providers.first)) {
						printf("[%s]: WRONG result for label length %d\n", provider->info.model, label_length);
						ok = false;
					}
				}
			}
			if (providers.second.size() > 1) {
				for (Provider::Vector::iterator ref = providers.second.begin(); providers.second.end() != ref; ref++) {
					for (Provider::Vector::iterator provider = ref + 1; providers.second.end() != provider; provider++) {
						if (!(*provider)->equals(**ref)) {
							printf("[%s]: compare error with [%s] for label length %d\n", (*provider)->info.model, (*ref)->info.model, label_length);
							ok = false;
						}
					}
				}
			}
			if (ok) {
				printf("OK\n");
			}
			else {
				return 1;
			}
		}
	}

	return 0;
}

/* run all providers in parallel and compare computed labels */
int test_of_concurrency()
{
	std::pair<Provider::Ptr, Provider::Vector> providers{ getProviders() };

	if (providers.second.size() > 1) {
		std::atomic_uint32_t counter(0);
		for (auto provider : providers.second) {
			printf("Run %s\n", provider->info.model);
			provider->post(0, 32 * 1024 * 1024 - 1, 1, &counter);
		}

		while (counter.load()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			printf(".");
		}

		for (auto provider : providers.second) {
			provider->join();
		}

		printf("\nCompute done, compare: ");

		bool ok = true;
		for (Provider::Vector::iterator ref = providers.second.begin(); providers.second.end() != ref; ref++) {
			for (Provider::Vector::iterator provider = ref + 1; providers.second.end() != provider; provider++) {
				if (!(*provider)->equals(**ref)) {
					printf("[%s]: compare error with [%s] for label length %d\n", (*provider)->info.model, (*ref)->info.model, 1);
					ok = false;
				}
			}
		}

		if (ok) {
			printf("OK\n");
		}
		else {
			return 1;
		}
	}

	return 0;
}

/* run labels computation for provider, cancel it and check if tail result is right */
int test_of_cancellation()
{
	std::pair<Provider::Ptr, Provider::Vector> providers{ getProviders() };

	for (auto provider : providers.second) {
		for (int i = 1; i < 4; i++) {
			std::atomic_uint32_t counter(0);
			printf("Run (%d) %s\n", i, provider->info.model);
			provider->post(0, 32 * 1024 * 1024 - 1, 1, &counter);
			std::this_thread::sleep_for(std::chrono::seconds(5));
			stop(10000);
			while (counter.load()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				printf(".");
			}
			provider->join();
			if (providers.first && provider->hashes_computed > 128) {
				uint64_t start_pos = ((provider->hashes_computed / 128) - 1) * 128;
				providers.first->compute(start_pos, start_pos + 127, 1);
				if (0 != memcmp(providers.first->labels.data(), provider->labels.data() + (start_pos / 8), providers.first->labels.size())) {
					printf("[%s]: WRONG result for cancel\n", provider->info.model);
					return 1;
				}
				else {
					printf("[%s]: cancel OK\n", provider->info.model);
				}
			}
		}
	}

	return 0;
}

/* run all integration tests */
void do_integration_tests()
{
	// test variable label length
	test_variable_label_length();

	// test variable labels count
	test_variable_labels_count();

	// test of concurrency
	test_of_concurrency();

	// test of cancellation
	test_of_cancellation();
}
