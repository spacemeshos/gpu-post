#ifndef	__SPACEMESH_CUDA_CUDA_H__
#define	__SPACEMESH_CUDA_CUDA_H__

double throughput2intensity(uint32_t throughput);

void cuda_log_lasterror(int thr_id, const char* func, int line);
void cuda_clear_lasterror();
#define CUDA_LOG_ERROR() cuda_log_lasterror(thr_id, __func__, __LINE__)

#endif	/* __SPACEMESH_CUDA_CUDA_H__ */
