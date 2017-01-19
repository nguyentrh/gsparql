#include "../inc/device_func.h"
#include "../inc/primitives.h"

using namespace gsparql;

// find positions of elements of key-value pairs 2 in the merged array
template <typename Key, typename Value>
__global__ void locateWrite(
	Key* d_key1, Value* d_value1, int len1,
	Key* d_key2, Value* d_value2, int len2,
	Key* d_keyOut, Value* d_valueOut) {

	// copy chunk1 to shared memory (a segment of array1 with BLOCK_SIZE elements)
	const int threadId = GTID;
	__shared__ Key s_key1[BLOCK_SIZE];
	__shared__ Key s_value1[BLOCK_SIZE];

	__shared__ volatile int s_start2;
	__shared__ volatile int s_stop2;
	__shared__ volatile int s_len1;

	if (threadId < len1) {
		s_key1[TID] = d_key1[threadId];
		s_value1[TID] = d_value1[threadId];
	}
	__syncthreads();

	// binary search the begin and end indexes of chunk2, BAD (thread divergence)
	if (WTID == 0) {
		if (WID == 0) {
			s_start2 = lowerBound<Key, Value>(d_key2, d_value2, len2, s_key1[0], s_value1[0]);
		}

		if (WID == 1) {
			s_len1 = min(len1 - BID * BLOCK_SIZE, BLOCK_SIZE);
			s_stop2 = upperBound<Key, Value>(d_key2, d_value2, len2, s_key1[s_len1 - 1], s_value1[s_len1 - 1]);
		}
	}
	__syncthreads();

	// write elements of chunk2 to output array
	for (int i = s_start2 + TID; i < s_stop2; i += BLOCK_SIZE) {
		Key key2 = d_key2[i]; // coalesced read, GOOD
		Value val2 = d_value2[i];

		int offset = lowerBound<Key, Value>(s_key1, s_value1, s_len1, key2, val2); // binary search on shared memory, OKAY
		offset += i;

		d_keyOut[offset] = key2; // random write, BAD
		d_valueOut[offset] = val2;
	}
}

// merge two non-overlapping sorted arrays
template <typename Key, typename Value>
int mergeSorted(Key* d_key1, Value* d_value1, int size1,
	Key* d_key2, Value* d_value2, int size2, Key* d_keyOut, Value* d_valueOut) {

	// write array 2 to output array
	int blocksPerGrid = (size1 - 1) / BLOCK_SIZE + 1;

	locateWrite<Key, Value> <<<blocksPerGrid, BLOCK_SIZE>>> (d_key1, d_value1, size1,
		d_key2, d_value2, size2, d_keyOut, d_valueOut);
	CUT_CHECK_ERROR("locateWrite");

	// write array 1 to output array
	blocksPerGrid = (size2 - 1) / BLOCK_SIZE + 1;
	locateWrite<Key, Value> <<<blocksPerGrid, BLOCK_SIZE>>> (d_key2, d_value2, size2,
		d_key1, d_value1, size1, d_keyOut, d_valueOut);
	CUT_CHECK_ERROR("locateWrite");

	return size1 + size2;
}