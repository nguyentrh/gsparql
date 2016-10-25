#ifndef BITONIC_SORT_CU
#define BITONIC_SORT_CU

#include "../define/cuda_primitive.h"

#define SHARED_SIZE_LIMIT (1024)

inline int getNP2(int n) {
	int np2 = 1;
	while (np2 < n && np2 % BLOCK_SIZE != 0) np2 <<= 1;
	return np2;
}

/* Kernel that copies data from an 'src' array and fill the rest with a default 'value' */
__global__ void copyAndFill(
	index_t*	dev_dest, 
	index_t*	dev_src, 
	int			size, 
	index_t		value
	){

	const unsigned int threadId = GTID;

	if (threadId < size) {
		dev_dest[threadId] = dev_src[threadId];
	}
	else {
		dev_dest[threadId] = value;
	}
}

// swap two <key, value> pairs
__device__ inline void swap(
	index_t &keyA,
	index_t &valA,
	index_t &keyB,
	index_t &valB
	) {
	
	index_t t;

	t = keyA;
	keyA = keyB;
	keyB = t;

	t = valA;
	valA = valB;
	valB = t;
}

// compare and swap two <key, value> pairs
__device__ inline void compAndSwap(
	index_t &keyA,
	index_t &valA,
	index_t &keyB,
	index_t &valB,
	bool dir
	){
	
	if (keyA == keyB) {
		if ((valA < valB) == dir) {
			swap(keyA, valA, keyB, valB);
		}
	}
	else if ((keyA < keyB) == dir) {
		swap(keyA, valA, keyB, valB);
	}

}

/* Monolithic bitonic sort kernel for very short arrays fitting into shared memory */
__global__ void bitonicSortBlocked(
	index_t	*dev_destKey,
	index_t *dev_destVal,
	index_t *dev_srcKey,
	index_t *dev_srcVal,
	int		destLen,
	int		srcLen,
	bool	dir
	){

	// Shared memory storage for one or more short vectors
	__shared__ index_t s_key[SHARED_SIZE_LIMIT];
	__shared__ index_t s_val[SHARED_SIZE_LIMIT];
		
	// Copy values to shared memory
	if (TID < srcLen) {
		s_key[TID] = dev_srcKey[TID];
		s_val[TID] = dev_srcVal[TID];
	}

	if (TID + SHARED_SIZE_LIMIT / 2 < srcLen) {
		s_key[TID + SHARED_SIZE_LIMIT / 2] = dev_srcKey[TID + SHARED_SIZE_LIMIT / 2];
		s_val[TID + SHARED_SIZE_LIMIT / 2] = dev_srcVal[TID + SHARED_SIZE_LIMIT / 2];
	}
	
	// Build bitonic arrays
	for (int size = 2; size < srcLen; size <<= 1)
	{
		for (int stride = size / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();

			int pos = 2 * TID - TID % stride;
			bool ddd = dir ^ ((TID & (size / 2)) != 0);

			compAndSwap( s_key[pos], s_val[pos], s_key[pos + stride], s_val[pos + stride], ddd);
		}
	}

	//ddd == dir for the last bitonic merge step
	for (int stride = srcLen / 2; stride > 0; stride >>= 1) {
		__syncthreads();
		
		int pos = 2 * TID - TID % stride;
		compAndSwap(s_key[pos], s_val[pos], s_key[pos + stride], s_val[pos + stride], dir);
	}

	__syncthreads();

	if (TID < destLen) {
		dev_destKey[TID] = s_key[TID];
		dev_destVal[TID] = s_val[TID];
	}

	if (TID + SHARED_SIZE_LIMIT / 2 < destLen) {
		dev_destKey[TID + SHARED_SIZE_LIMIT / 2] = s_key[TID + SHARED_SIZE_LIMIT / 2];
		dev_destVal[TID + SHARED_SIZE_LIMIT / 2] = s_val[TID + SHARED_SIZE_LIMIT / 2];
	}
}

/* bitonic sort */
extern "C" void bitonicSort(
	index_t*	dev_key, 
	index_t*	dev_value, 
	int			numRecords, 
	int			dir) {
	
	// extend the size of the sorted array to 2^n
	int np2Size = getNP2(numRecords);

	int blocksPerGrid = np2Size / BLOCK_SIZE;

	// declair two temp 'key', 'value' arrays for sorting computation in gpu
	index_t* dev_tempKey;
	index_t* dev_tempVal;
	GPUMALLOC(&dev_tempKey, np2Size * sizeof(index_t));
	GPUMALLOC(&dev_tempVal, np2Size * sizeof(index_t));

	// copy key, value arrays to two temp arrays
	index_t defaultVal = dir == 0 ? UINT_MAX : 0;
	copyAndFill GPUPARAM(blocksPerGrid, BLOCK_SIZE) (dev_tempKey, dev_key, numRecords, defaultVal);
	copyAndFill GPUPARAM(blocksPerGrid, BLOCK_SIZE) (dev_tempVal, dev_value, numRecords, defaultVal);

	if (np2Size <= SHARED_SIZE_LIMIT) { // For small-size arrays, we execute an kernel which utilizes shared memory
		bitonicSortBlocked GPUPARAM(blocksPerGrid, BLOCK_SIZE) (dev_key, dev_value, dev_tempKey, dev_tempVal, numRecords, np2Size, (dir == 1));
	}
	else { // For large arrays
		// adsadada
	}

	// free data
	GPUFREE(dev_tempKey);
	GPUFREE(dev_tempVal);
}

#endif