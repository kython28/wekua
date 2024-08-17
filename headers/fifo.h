#ifndef WEKUA_FIFO_H
#define WEKUA_FIFO_H

#include <stdint.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _w_fifo {
	pthread_cond_t cond;
	pthread_mutex_t lock;
	void *data;
	uint64_t qsize;
} *wfifo;

wfifo wekuaAllocFIFO(void);
void *wekuaFIFOGet(wfifo fifo);
uint8_t wekuaFIFOisEmpty(wfifo fifo) __attribute__ ((warn_unused_result));
uint8_t wekuaFIFOisnotEmpty(wfifo fifo) __attribute__ ((warn_unused_result));
void wekuaFIFOPut(wfifo fifo, void *data);
void wekuaFreeFIFO(wfifo fifo);

#ifdef __cplusplus
}
#endif
#endif

