#ifndef BITONIC_SORT_CU
#define BITONIC_SORT_CU

#include "..\define\common.h"

#define SHARED_SIZE_LIMIT (32)

inline int getNP2(int n) {
	int np2 = 1;
	while (np2 < n || np2 % BLOCK_SIZE != 0) np2 <<= 1;
	return np2;
}


/// copy arrays
template <typename T>
__global__ void copy_array(T* d_dest, T* d_src, int size, T val) {
	const unsigned int threadId = GTID;

	if (threadId < size) {
		d_dest[threadId] = d_src[threadId];
	}
	else {
		d_dest[threadId] = val;
	}
}

// swap two <key, value> pairs
template <typename Key, typename Val>
__device__ inline void swap(
	Key keyA,
	Val valA,
	Key keyB,
	Val valB ) {

	Key keyTemp;
	// swap two keys
	keyTemp = keyA;
	keyA = keyB;
	keyB = keyTemp;

	Val valTemp;
	// swap two values
	valTemp = valA;
	valA = valB;
	valB = valTemp;
}

// compare and swap two <key, value> pairs
template <typename Key, typename Val, bool dir>
__device__ inline void compare_swap(
	Key keyA,
	Val valA,
	Key keyB,
	Val valB ){

	if (keyA == keyB) {
		if ((valA < valB) == dir) {
			swap(keyA, valA, keyB, valB);
		}
	}
	else if ((keyA < keyB) == dir) {
		swap(keyA, valA, keyB, valB);
	}
}


/* bitonic sort */
template<typename Key, typename Val>
extern "C" void bitonicSort(
	Key*	d_key,
	Val*	d_value,
	int		len,
	int		dir) {

	// extend the size of the sorted array to 2^n
	int np2Size = getNP2(len);

	Key* d_tempKey;
	Val* d_tempVal;

	if (np2Size != len) {
		GPUMALLOC(&d_tempKey, np2Size * sizeof(Key));
		GPUMALLOC(&d_tempVal, np2Size * sizeof(Val));

		int value = (dir == 0) ? INT_MAX : 0;

	}

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