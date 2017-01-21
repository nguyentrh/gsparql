#pragma once

#include <utility>
#include "../inc/define.h"
#include "../inc/device_func.h"
#include "scan.cuh"

namespace gsparql {
	namespace join {

		/// count number of merged results
		template <typename Key>
		__global__ void mergeJoinSharedCount(
			Key* d_keyA, int lenA,
			Key* d_keyB, int lenB,
			int* d_count) {

			int pos = GTID;

			__shared__ Key s_keyA[BLOCK_SIZE];

			if (pos < lenA) {
				s_keyA[TID] = d_keyA[pos];
			}
			__syncthreads();

			__shared__ volatile int s_startB, s_stopB, s_lenB, s_lenA;

			// binary search the begin and end indexes of the corresponding chunk of arrayB, BAD (thread divergence)
			if (TID == 0) {
				s_startB = util::lowerBound<Key>(d_keyB, lenB, s_keyA[0]);
				s_lenA = min(lenA - BID * BLOCK_SIZE, BLOCK_SIZE);
				s_stopB = util::upperBound<Key>(d_keyB, lenB, s_keyA[s_lenA - 1]);
			}
			__syncthreads();

			int count = 0;
			for (int i = s_startB + TID; i < s_stopB; i += BLOCK_SIZE) {
				Key key = d_keyB[i];

				int lo = util::lowerBound<Key>(s_keyA, s_lenA, key);
				int hi = util::upperBound<Key>(s_keyA, s_lenA, key);

				count += (hi - lo);
			}

			d_count[pos] = count;
		}

		/// merge two arbitrary arrays
		template <typename Key, typename ValueA, typename ValueB>
		__global__ void mergeJoinSharedWrite(
			Key* d_keyA, ValueA* d_valueA, int lenA,
			Key* d_keyB, ValueB* d_valueB, int lenB,
			int* d_offset, ValueA* d_keyC, ValueB* d_valueC) {

			int pos = GTID;

			__shared__ Key s_keyA[BLOCK_SIZE];
			__shared__ ValueA s_valueA[BLOCK_SIZE];

			if (pos < lenA) {
				s_keyA[TID] = d_keyA[pos];
				s_valueA[TID] = d_valueA[pos];
			}
			__syncthreads();

			__shared__ volatile int s_startB, s_stopB, s_lenB, s_lenA;

			if (TID == 0) {
				s_startB = util::lowerBound<Key>(d_keyB, lenB, s_keyA[0]);
				s_lenA = min(lenA - BID * BLOCK_SIZE, BLOCK_SIZE);
				s_stopB = util::upperBound<Key>(d_keyB, lenB, s_keyA[s_lenA - 1]);
				printf("block %d s_startB %d key %d\n", BID, s_startB, d_keyB[s_startB]);
				printf("block %d s_stopB %d\n", BID, s_stopB);
			}
			__syncthreads();

			int offset = d_offset[pos];

			for (int i = s_startB + TID; i < s_stopB; i += BLOCK_SIZE) {
				Key key = d_keyB[i];
				ValueB val = d_valueB[i];

				int lo = util::lowerBound<Key>(s_keyA, s_lenA, key);
				int hi = util::upperBound<Key>(s_keyA, s_lenA, key);

				for (int j = lo; j < hi; j++) {
					d_keyC[offset] = s_valueA[j];
					d_valueC[offset] = val;
					offset++;
				}
			}
		}

		template <typename Key, typename ValueA, typename ValueB>
		int mergeSortedJoin(
			Key* d_keyA, ValueA* d_valueA, int lenA,
			Key* d_keyB, ValueB* d_valueB, int lenB,
			std::pair<ValueA*, ValueB*>* d_output){

			if (lenA > lenB) {
				int* d_count;
				GPUMALLOC(&d_count, lenA * sizeof(int));
				
				int blocksPerGrid = (lenA - 1) / BLOCK_SIZE + 1;
				mergeJoinSharedCount<Key> <<<blocksPerGrid, BLOCK_SIZE>>>(d_keyA, lenA, d_keyB, lenB, d_count);
				CUT_CHECK_ERROR("mergeJoinSharedCount (arrA)");

				int* d_offset;
				GPUMALLOC(&d_offset, lenA * sizeof(int));
				
				int res = scan::exclusiveScan(d_count, d_offset, lenA);
				if (res != 0) {
					GPUMALLOC(&d_output->first, res * sizeof(ValueA));
					GPUMALLOC(&d_output->second, res * sizeof(ValueB));

					mergeJoinSharedWrite<Key, ValueA, ValueB> <<<blocksPerGrid, BLOCK_SIZE >>>(
						d_keyA, d_valueA, lenA,
						d_keyB, d_valueB, lenB,
						d_offset, d_output->first, d_output->second);
					CUT_CHECK_ERROR("mergeJoinSharedWrite (arrA)");
				}

				GPUFREE(d_offset);
				GPUFREE(d_count);

				return res;
			}
			else {
				int* d_count;
				GPUMALLOC(&d_count, lenB * sizeof(int));

				int blocksPerGrid = (lenB - 1) / BLOCK_SIZE + 1;
				mergeJoinSharedCount<Key> <<<blocksPerGrid, BLOCK_SIZE >>>(d_keyB, lenB, d_keyA, lenA, d_count);
				CUT_CHECK_ERROR("mergeJoinSharedCount (arrB)");

				int* d_offset;
				GPUMALLOC(&d_offset, lenB * sizeof(int));
				int res = scan::exclusiveScan(d_count, d_offset, lenB);

				if (res != 0) {
					GPUMALLOC(&d_output->first, res * sizeof(ValueA));
					GPUMALLOC(&d_output->second, res * sizeof(ValueB));

					mergeJoinSharedWrite<Key, ValueB, ValueA> <<<blocksPerGrid, BLOCK_SIZE >>>(
						d_keyB, d_valueB, lenB,
						d_keyA, d_valueA, lenA,
						d_offset, d_output->second, d_output->first);
					CUT_CHECK_ERROR("mergeJoinSharedWrite (arrB)");
				}

				GPUFREE(d_offset);
				GPUFREE(d_count);

				return res;
			}
		}
	}
}