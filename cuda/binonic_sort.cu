#ifndef BITONIC_SORT_CU
#define BITONIC_SORT_CU

#include "../define/common.h"

__device__ inline void compare(
	index_t &keyA, index_t &valA,
	index_t &keyB, index_t &valB,
	unsigned short dir
	)
{
	index_t t;

	if ((keyA > keyB) == dir)
	{
		t = keyA;
		keyA = keyB;
		keyB = t;
		t = valA;
		valA = valB;
		valB = t;
	}
}


__global__ void bitonic_sort_shared(
	index_t *dev_destKey, index_t *dev_destVal,
	index_t *dev_srcKey, index_t *dev_srcVal,
	index_t len, index_t dir
	)
{
	//Shared memory storage for one or more short vectors
	__shared__ index_t s_key[SHARED_SIZE_LIMIT];
	__shared__ index_t s_val[SHARED_SIZE_LIMIT];

	//Offset to the beginning of subbatch and load data
	dev_srcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	dev_srcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	dev_destKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	dev_destVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	s_key[threadIdx.x + 0] = dev_srcKey[0];
	s_val[threadIdx.x + 0] = dev_srcVal[0];
	s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = dev_srcKey[(SHARED_SIZE_LIMIT / 2)];
	s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = dev_srcVal[(SHARED_SIZE_LIMIT / 2)];

	for (index_t size = 2; size < len; size <<= 1)
	{
		//Bitonic merge
		index_t ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

		for (index_t stride = size / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();
			index_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
			Comparator(
				s_key[pos + 0], s_val[pos + 0],
				s_key[pos + stride], s_val[pos + stride],
				ddd
				);
		}
	}

	//ddd == dir for the last bitonic merge step
	{
		for (index_t stride = len / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();
			index_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
			Comparator(
				s_key[pos + 0], s_val[pos + 0],
				s_key[pos + stride], s_val[pos + stride],
				dir
				);
		}
	}

	__syncthreads();
	dev_destKey[0] = s_key[threadIdx.x + 0];
	dev_destVal[0] = s_val[threadIdx.x + 0];
	dev_destKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
	dev_destVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

#endif