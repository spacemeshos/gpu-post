#ifndef __SPACEMESH_UTIL_H__
#define	__SPACEMESH_UTIL_H__


#ifdef __APPLE__
struct cgsem {
	int pipefd[2];
};

typedef struct cgsem cgsem_t;
#elif defined(WIN32)
#include <Windows.h>
struct cgsem {
	HANDLE	handle;
};

typedef struct cgsem cgsem_t;
#else
#include <semaphore.h>
typedef sem_t cgsem_t;
#endif

void cgsem_init(cgsem_t *cgsem);
void cgsem_post(cgsem_t *cgsem);
void cgsem_wait(cgsem_t *cgsem);
void cgsem_destroy(cgsem_t *cgsem);

#endif	/* __SPACEMESH_UTIL_H__ */
