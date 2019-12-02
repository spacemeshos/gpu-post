#ifndef __DEVICE_GPU_H__
#define __DEVICE_GPU_H__

#include "api_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

extern bool have_opencl;

extern struct device_drv opencl_drv;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* __DEVICE_GPU_H__ */
