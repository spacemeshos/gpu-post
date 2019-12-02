/*
 * Copyright 2010 Jeff Garzik
 * Copyright 2012-2014 pooler
 * Copyright 2014 ccminer team
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include <spacemesh-config.h>

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdarg.h>
#include <string.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <time.h>
#include "api_internal.h"
#include "util.h"

extern bool opt_debug_diff;

bool opt_tracegpu = false;

void applog(int prio, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);

#ifdef HAVE_SYSLOG_H
	if (use_syslog) {
		va_list ap2;
		char *buf;
		int len;

		/* custom colors to syslog prio */
		if (prio > LOG_DEBUG) {
			switch (prio) {
				case LOG_BLUE: prio = LOG_NOTICE; break;
			}
		}

		va_copy(ap2, ap);
		len = vsnprintf(NULL, 0, fmt, ap2) + 1;
		va_end(ap2);
		buf = (char*) alloca(len);
		if (vsnprintf(buf, len, fmt, ap) >= 0)
			syslog(prio, "%s", buf);
	}
#else
	if (0) {}
#endif
	else {
		const char* color = "";
		const time_t now = time(NULL);
		char *f;
		int len;
		struct tm tm;

		localtime_r(&now, &tm);

		len = 40 + (int) strlen(fmt) + 2;
		f = (char*) alloca(len);
		sprintf(f, "[%d-%02d-%02d %02d:%02d:%02d]%s %s%s\n",
			tm.tm_year + 1900,
			tm.tm_mon + 1,
			tm.tm_mday,
			tm.tm_hour,
			tm.tm_min,
			tm.tm_sec,
			color,
			fmt,
			""
		);
		if (prio == LOG_RAW) {
			// no time prefix, for ccminer -n
			sprintf(f, "%s%s\n", fmt, "");
		}
		pthread_mutex_lock(&applog_lock);
		vfprintf(stdout, f, ap);	/* atomic write to stdout */
		fflush(stdout);
		pthread_mutex_unlock(&applog_lock);
	}
	va_end(ap);
}

// Use different prefix if multiple cpu threads per gpu
// Also, auto hide LOG_DEBUG if --debug (-D) is not used
void gpulog(int prio, int dev_id, const char *fmt, ...)
{
	char _ALIGN(128) pfmt[128];
	char _ALIGN(128) line[256];
	int len;
	va_list ap;

	if (prio == LOG_DEBUG && !opt_debug)
		return;

	len = snprintf(pfmt, 128, "GPU #%d: %s", dev_id, fmt);
	pfmt[sizeof(pfmt)-1]='\0';

	va_start(ap, fmt);

	if (len && vsnprintf(line, sizeof(line), pfmt, ap)) {
		line[sizeof(line)-1]='\0';
		applog(prio, "%s", line);
	} else {
		fprintf(stderr, "%s OOM!\n", __func__);
	}

	va_end(ap);
}

/* cgminer specific wrappers for true unnamed semaphore usage on platforms
* that support them and for apple which does not. We use a single byte across
* a pipe to emulate semaphore behaviour there. */
#ifdef __APPLE__
void cgsem_init(cgsem_t *cgsem)
{
	int flags, fd, i;

	if (pipe(cgsem->pipefd) == -1)
		quit(1, "Failed pipe in cgsem_init");

	/* Make the pipes FD_CLOEXEC to allow them to close should we call
	* execv on restart. */
	for (i = 0; i < 2; i++) {
		fd = cgsem->pipefd[i];
		flags = fcntl(fd, F_GETFD, 0);
		flags |= FD_CLOEXEC;
		if (fcntl(fd, F_SETFD, flags) == -1)
			quit(1, "Failed to fcntl in cgsem_init");
	}
}

void cgsem_post(cgsem_t *cgsem)
{
	const char buf = 1;
	int ret;

	ret = write(cgsem->pipefd[1], &buf, 1);
	if (unlikely(ret == 0))
		applog(LOG_WARNING, "Failed to write in cgsem_post");
}

void cgsem_wait(cgsem_t *cgsem)
{
	char buf;
	int ret;

	ret = read(cgsem->pipefd[0], &buf, 1);
	if (unlikely(ret == 0))
		applog(LOG_WARNING, "Failed to read in cgsem_wait");
}

void cgsem_destroy(cgsem_t *cgsem)
{
	close(cgsem->pipefd[1]);
	close(cgsem->pipefd[0]);
}
#elif defined(WIN32)
void cgsem_init(cgsem_t *cgsem)
{
	cgsem->handle = CreateSemaphore(NULL, 0, 0x10000000, NULL);

	if (NULL == cgsem->handle) {
		quit(1, "Failed to sem_init in cgsem_init");
	}
}

void cgsem_post(cgsem_t *cgsem)
{
	if (!ReleaseSemaphore(cgsem->handle, 1, NULL)) {
		quit(1, "Failed to sem_post in cgsem_post");
	}
}

void cgsem_wait(cgsem_t *cgsem)
{
	if (WAIT_OBJECT_0 != WaitForSingleObject(cgsem->handle, INFINITE)) {
		quit(1, "Failed to sem_wait in cgsem_wait");
	}
}

void cgsem_destroy(cgsem_t *cgsem)
{
	if (NULL != cgsem->handle) {
		CloseHandle(cgsem->handle);
		cgsem->handle = NULL;
	}
}
#else
void cgsem_init(cgsem_t *cgsem)
{
	if (sem_init(cgsem, 0, 0))
		quit(1, "Failed to sem_init in cgsem_init");
}

void cgsem_post(cgsem_t *cgsem)
{
	if (unlikely(sem_post(cgsem)))
		quit(1, "Failed to sem_post in cgsem_post");
}

void cgsem_wait(cgsem_t *cgsem)
{
	if (unlikely(sem_wait(cgsem)))
		quit(1, "Failed to sem_wait in cgsem_wait");
}

void cgsem_destroy(cgsem_t *cgsem)
{
	sem_destroy(cgsem);
}
#endif
