#include <stdio.h>
#include <memory.h>
#include <string.h>
#include <unistd.h>
#include <map>

// include thrust
#ifndef __cplusplus
#include <thrust/version.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#else
#include <ctype.h>
#endif

#include "api_internal.h"
#include "nvml.h"
#include "cuda_runtime.h"

static bool substringsearch(const char *haystack, const char *needle, int &match)
{
	int hlen = (int) strlen(haystack);
	int nlen = (int) strlen(needle);
	for (int i=0; i < hlen; ++i)
	{
		if (haystack[i] == ' ') continue;
		int j=0, x = 0;
		while(j < nlen)
		{
			if (haystack[i+x] == ' ') {++x; continue;}
			if (needle[j] == ' ') {++j; continue;}
			if (needle[j] == '#') return ++match == needle[j+1]-'0';
			if (tolower(haystack[i+x]) != tolower(needle[j])) break;
			++j; ++x;
		}
		if (j == nlen) return true;
	}
	return false;
}

// since 1.8.3
double throughput2intensity(uint32_t throughput)
{
	double intensity = 0.;
	uint32_t ws = throughput;
	uint8_t i = 0;
	while (ws > 1 && i++ < 32)
		ws = ws >> 1;
	intensity = (double) i;
	if (i && ((1U << i) < throughput)) {
		intensity += ((double) (throughput-(1U << i)) / (1U << i));
	}
	return intensity;
}

// Check (and reset) last cuda error, and report it in logs
void cuda_log_lasterror(int thr_id, const char* func, int line)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		gpulog(LOG_WARNING, thr_id, "%s:%d %s", func, line, cudaGetErrorString(err));
}

// Clear any cuda error in non-cuda unit (.c/.cpp)
void cuda_clear_lasterror()
{
	cudaGetLastError();
}

// Zeitsynchronisations-Routine von cudaminer mit CPU sleep
// Note: if you disable all of these calls, CPU usage will hit 100%
typedef struct { double value[8]; } tsumarray;
cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id)
{
	cudaError_t result = cudaSuccess;
	if (abort_flag)
		return result;
	if (situation >= 0)
	{
		static std::map<int, tsumarray> tsum;

		double a = 0.95, b = 0.05;
		if (tsum.find(situation) == tsum.end()) { a = 0.5; b = 0.5; } // faster initial convergence

		double tsync = 0.0;
		double tsleep = 0.95 * tsum[situation].value[thr_id];
		if (cudaStreamQuery(stream) == cudaErrorNotReady)
		{
			usleep((useconds_t)(1e6*tsleep));
			struct timeval tv_start, tv_end;
			gettimeofday(&tv_start, NULL);
			result = cudaStreamSynchronize(stream);
			gettimeofday(&tv_end, NULL);
			tsync = 1e-6 * (tv_end.tv_usec-tv_start.tv_usec) + (tv_end.tv_sec-tv_start.tv_sec);
		}
		if (tsync >= 0) tsum[situation].value[thr_id] = a * tsum[situation].value[thr_id] + b * (tsleep+tsync);
	}
	else
		result = cudaStreamSynchronize(stream);
	return result;
}
