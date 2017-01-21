#ifndef COMMON_DEFINIION_H
#define COMMON_DEFINIION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil.h"

#ifdef __INTELLISENSE__
#define __launch_bounds__(a,b)
void __syncthreads(void);
void __threadfence(void);
int __mul24(int, int);
#endif

#define DEBUG 1

/* define constant cuda parameters */
#define WARP_SIZE (32)
#define BLOCK_SIZE (256)
#define SHARED_SIZE_LIMIT (512)

#define THREADS_PER_BLOCK (blockDim.x)
#define WARPS_PER_BLOCK ((THREADS_PER_BLOCK-1)/WARP_SIZE + 1)
#define BLOCKS_PER_GRID (gridDim.x)

#define BID	(blockIdx.x)
#define TID	(threadIdx.x)	/*thread ID in current block*/
#define WID (TID / WARP_SIZE)
#define WTID (TID % WARP_SIZE)

#define GTID (BID * THREADS_PER_BLOCK + TID)	/*global thread ID*/
#define GWID (GTID / WARP_SIZE)
#define GWTID (GTID % WARP_SIZE)

#define TOTAL_THREADS (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

#define SHARED_MEMORY_PER_PROCESSOR (8*1024)
#define TOTAL_PROCESSORS (14)

/* define short functions */
#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
				    }} while(0)

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define GPUMALLOC(pointer, size) CUDA_SAFE_CALL(cudaMalloc( pointer, size))
#define CPUMALLOC(pointer, size) CUDA_SAFE_CALL(cudaMallocHost (pointer, size))

#define CPUFREE(pointer) if(pointer!=NULL) CUDA_SAFE_CALL(cudaFreeHost (pointer))
#define GPUFREE(pointer) CUDA_SAFE_CALL( cudaFree( pointer) )

#define TOGPU(dev_pointer,hos_pointer, size)  CUDA_SAFE_CALL(cudaMemcpy(dev_pointer,hos_pointer, size, cudaMemcpyHostToDevice))
#define FROMGPU(hos_pointer, dev_pointer, size)  CUDA_SAFE_CALL(cudaMemcpy(hos_pointer, dev_pointer, size, cudaMemcpyDeviceToHost))
#define GPUTOGPU(dev_to, dev_from, size)  CUDA_SAFE_CALL(cudaMemcpy(dev_to, dev_from, size, cudaMemcpyDeviceToDevice))

#endif