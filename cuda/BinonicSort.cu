#ifndef BITONIC_SORT_CU
#define BITONIC_SORT_CU

#include "../define/GPUPrimitive.h"
#include "../define/GPUCommon.h"

using namespace sparql;

#define SHARED_SIZE_LIMIT (32)

inline int getNP2(int n) {
	int np2 = 1;
	while (np2 < n || np2 % BLOCK_SIZE != 0) np2 <<= 1;
	return np2;
}

/* Kernel that copies data from an array to another array and fill the rest with a default value */
__global__ void copy_fill(
	char*	dev_dest,
	char*	dev_src, 
	int		elementNum,
	int		elementSize,
	char*	defaultValue
	){
	const unsigned int threadId = GTID;

	// offset to the beginning of element
	dev_dest += threadId * elementSize;
	dev_src += threadId * elementSize;
	
	if (threadId < elementNum) {
		memcpy(dev_dest, dev_src, elementSize);
	}
	else {
		memcpy(dev_dest, defaultValue, elementSize);
	}
}

// swap two <key, value> pairs
__device__ inline void swap(
	char* keyA,
	char* valA,
	char* keyB,
	char* valB,
	int sizeKey,
	int sizeVal
	) {
	
	char temp[8]; // temporary values

	// swap two keys
	memcpy(temp, keyA, sizeKey);
	memcpy(keyA, keyB, sizeKey);
	memcpy(keyB, temp, sizeKey);

	// swap two values
	memcpy(temp, valA, sizeVal);
	memcpy(valA, valB, sizeVal);
	memcpy(valB, temp, sizeVal);
}

// compare and swap two <key, value> pairs
__device__ inline void compare_swap(
	char* keyA,
	char* valA,
	char* keyB,
	char* valB,
	int sizeKey,
	int sizeVal,
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
	int	*dev_destKey,
	int *dev_destVal,
	int *dev_srcKey,
	int *dev_srcVal,
	int		destLen,
	int		srcLen,
	bool	dir
	){

	// Shared memory storage for one or more short vectors
	__shared__ int s_key[SHARED_SIZE_LIMIT];
	__shared__ int s_val[SHARED_SIZE_LIMIT];
		
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

__global__ void bitonicSortShared(
	int *dev_key,
	int *dev_val
	){
	//Shared memory storage for current subarray
	__shared__ int s_key[SHARED_SIZE_LIMIT];
	__shared__ int s_val[SHARED_SIZE_LIMIT];

	// Copy values to shared memory
	int globalId = BID * SHARED_SIZE_LIMIT + TID;
	s_key[TID] = dev_key[globalId];
	s_val[TID] = dev_val[globalId];
	s_key[TID + BLOCK_SIZE] = dev_key[globalId + BLOCK_SIZE];
	s_val[TID + BLOCK_SIZE] = dev_val[globalId + BLOCK_SIZE];

	for (int size = 2; size < SHARED_SIZE_LIMIT; size <<= 1)
	{
		for (int stride = size / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();
			
			int pos = 2 * TID - TID % stride;
			bool ddd = (TID & (size / 2)) != 0;
			compAndSwap(s_key[pos], s_val[pos], s_key[pos + stride], s_val[pos + stride], ddd);
		}
	}

	//Odd / even arrays of SHARED_SIZE_LIMIT elements
	//sorted in opposite directions
	bool ddd = BID & 1;
	{
		for (int stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();
			int pos = 2 * TID - TID % stride;
			compAndSwap(s_key[pos], s_val[pos], s_key[pos + stride], s_val[pos + stride], ddd);
		}
	}

	__syncthreads();
	dev_key[globalId] = s_key[TID];
	dev_val[globalId] = s_val[TID];
	dev_key[globalId + BLOCK_SIZE] = s_key[TID + BLOCK_SIZE];
	dev_val[globalId + BLOCK_SIZE] = s_val[TID + BLOCK_SIZE];
}

/* bitonic sort */
extern "C" void bitonicSort(
	int*	dev_key, 
	int*	dev_value, 
	int		numRecords, 
	int		dir) {
	
	// extend the size of the sorted array to 2^n
	int np2Size = getNP2(numRecords);
	printf("np2Size: %d\n", np2Size);

	int blocksPerGrid = np2Size / BLOCK_SIZE;
	printf("blocksPerGrid: %d\n", blocksPerGrid);

	// declair two temp 'key', 'value' arrays for sorting computation in gpu
	int* dev_tempKey;
	int* dev_tempVal;
	GPUMALLOC(&dev_tempKey, np2Size * sizeof(int));
	GPUMALLOC(&dev_tempVal, np2Size * sizeof(int));

	// copy key, value arrays to two temp arrays
	int value = dir == 0 ? INT_MAX : 0;

	copy GPUPARAM(blocksPerGrid, BLOCK_SIZE) (dev_tempKey, dev_key, numRecords, value);
	CUT_CHECK_ERROR("copyAndFill");

	copy GPUPARAM(blocksPerGrid, BLOCK_SIZE) (dev_tempVal, dev_value, numRecords, value);
	CUT_CHECK_ERROR("copyAndFill");

	if (np2Size <= SHARED_SIZE_LIMIT) { // For small-size arrays, we execute an kernel which utilizes shared memory
		bitonicSortBlocked GPUPARAM(blocksPerGrid, BLOCK_SIZE) (dev_key, dev_value, dev_tempKey, dev_tempVal, numRecords, np2Size, (dir == 1));
		CUT_CHECK_ERROR("bitonicSortBlocked");
	}
	else { // For large arrays
		bitonicSortShared GPUPARAM(blocksPerGrid, BLOCK_SIZE) (dev_tempKey, dev_tempVal);
		CUT_CHECK_ERROR("bitonicSortShared");

		for (int size = 2 * SHARED_SIZE_LIMIT; size <= np2Size; size <<= 1) {
			for (int stride = size / 2; stride > 0; stride >>= 1) {

			}
		}

		GPUTOGPU(dev_key, dev_tempKey, sizeof(int) * numRecords);
		GPUTOGPU(dev_value, dev_tempVal, sizeof(int) * numRecords);
	}

	// free data
	GPUFREE(dev_tempKey);
	GPUFREE(dev_tempVal);
}

#endif