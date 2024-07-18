#include "../headers/fifo.h"
#include <stdlib.h>

struct linked_list_node {
	void *data;
	void *next;
};

struct linked_list {
	struct linked_list_node *first;
	struct linked_list_node *last;
};

wfifo wekuaAllocFIFO(){
	wfifo fifo = calloc(1, sizeof(struct _w_fifo));
	if (pthread_cond_init(&fifo->cond, NULL) != 0){
		free(fifo);
		return NULL;
	}
	if (pthread_mutex_init(&fifo->lock, NULL) != 0){
		pthread_cond_destroy(&fifo->cond);
		free(fifo);
		return NULL;
	}
	fifo->data = calloc(1, sizeof(struct linked_list));
	return fifo;
}

void *wekuaFIFOGet(wfifo fifo){
	void *data;
	pthread_mutex_t *lock = &fifo->lock;
	pthread_cond_t *cond = &fifo->cond;
	struct linked_list *queue = fifo->data;
	struct linked_list_node *node, *next_node, **head;

	pthread_mutex_lock(lock);
	head = &queue->first;
	while (!head[0]){
		pthread_cond_wait(cond, lock);
	}
	node = head[0];
	data = node->data;
	next_node = node->next;

	queue->first = next_node;
	if (next_node == NULL){
		queue->last = NULL;
	}
	free(node);
	fifo->qsize--;
	pthread_mutex_unlock(lock);
	return data;
}

void wekuaFIFOPut(wfifo fifo, void *data){
	pthread_mutex_t *lock = &fifo->lock;
	pthread_cond_t *cond = &fifo->cond;
	struct linked_list *queue = fifo->data;
	struct linked_list_node *node, **node_last;

	pthread_mutex_lock(lock);
	node = calloc(1, sizeof(struct linked_list_node));
	node_last = &queue->last;
	if (node_last[0] == NULL){
		queue->first = node;
	}else{
		node_last[0]->next = node;
	}
	node_last[0] = node;
	node->data = data;
	fifo->qsize++;
	pthread_cond_signal(cond);
	pthread_mutex_unlock(lock);
}

uint8_t wekuaFIFOisEmpty(wfifo fifo){
	pthread_mutex_t *lock = &fifo->lock;
	uint8_t ret = 0;
	pthread_mutex_lock(lock);
	if (fifo->qsize == 0) ret = 1;
	pthread_mutex_unlock(lock);
	return ret;
}

uint8_t wekuaFIFOisnotEmpty(wfifo fifo){
	pthread_mutex_t *lock = &fifo->lock;
	uint8_t ret = 0;
	pthread_mutex_lock(lock);
	if (fifo->qsize > 0) ret = 1;
	pthread_mutex_unlock(lock);
	return ret;
}

void wekuaFreeFIFO(wfifo fifo){
	if (wekuaFIFOisnotEmpty(fifo)) return;

	pthread_mutex_destroy(&fifo->lock);
	pthread_cond_destroy(&fifo->cond);

	free(fifo->data);
	free(fifo);
}