#ifndef TITAN_KERNEL_H
#define TITAN_KERNEL_H

#include "salsa_kernel.h"

class TitanKernel : public KernelInterface
{
public:
	TitanKernel();

	virtual void set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V);
	virtual bool run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int gpu_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int r, unsigned int p, unsigned int batch, unsigned int LOOKUP_GAP);

	virtual char get_identifier() { return 't'; }
	virtual int get_major_version() { return 5; }
	virtual int get_minor_version() { return 0; }

	virtual int threads_per_wu() { return 4; }
	virtual bool support_lookup_gap() { return true; }
	virtual cudaFuncCache cache_config() { return cudaFuncCachePreferL1; }

protected:
	uint32_t	_prev_N = 0;
	uint32_t	_prev_r = 0;
};

#endif // #ifndef TITAN_KERNEL_H
