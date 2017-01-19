#pragma once

#include "define.h"
#include "device_func.h"
#include <stdio.h>

namespace gsparql {
	namespace merge {

		/// merge two arrays (stored in shared memory) by using one block
		template <typename Key, typename Value>
		inline __device__ void merge(
			Key* s_keyA, Value* s_valueA, int lenA,
			Key* s_keyB, Value* s_valueB, int lenB,
			Key* d_keyC, Value* d_valueC) {

			if (TID < lenB) {
				Key key = s_keyB[TID];
				Value value = s_valueB[TID];

				// binary search on shared memory, OKAY
				int destB = util::lowerBound<Key, Value>(s_keyA, s_valueA, lenA, key, value) + TID;

				// random write, BAD
				d_keyC[destB] = key;
				d_valueC[destB] = value;
			}

			if (TID < lenA) {
				Key key = s_keyA[TID];
				Value value = s_valueA[TID];

				// binary search on shared memory, OKAY
				int destA = util::upperBound<Key, Value>(s_keyB, s_valueB, lenB, key, value) + TID;

				// random write, BAD
				d_keyC[destA] = key;
				d_valueC[destA] = value;
			}
		}

		/// merge two arrays when at least one array has size smaller than BLOCK_SIZE
		template <typename Key, typename Value>
		__global__ void mergeSharedSmall(
			Key* d_keyA, Value* d_valueA, int lenA,
			Key* d_keyB, Value* d_valueB, int lenB,
			Key* d_keyC, Value* d_valueC){

			__shared__ Key s_keyA[BLOCK_SIZE];
			__shared__ Value s_valueA[BLOCK_SIZE];

			__shared__ Key s_keyB[BLOCK_SIZE];
			__shared__ Value s_valueB[BLOCK_SIZE];

			__shared__ volatile int s_startA, s_stopA, s_lenB, s_lenA;

			const int startB = BID * (BLOCK_SIZE - 1);

			// copy array 1 to shared memory
			if (TID < lenA) {
				s_keyA[TID] = d_keyA[TID];
				s_valueA[TID] = d_valueA[TID];
			}

			// copy segments of array 2 to shared memory, each thread processes a segment
			if (startB + TID < lenB) {
				s_keyB[TID] = d_keyB[startB + TID];
				s_valueB[TID] = d_valueB[startB + TID];
			}
			__syncthreads();

			if (TID == 0) {
				s_startA = BID == 0 ? 0 : util::lowerBound<Key, Value>(s_keyA, s_valueA, lenA, s_keyB[0], s_valueB[0]);
				s_lenB = min(lenB - startB, BLOCK_SIZE);
				s_stopA = BID == BLOCKS_PER_GRID - 1 ? lenA : util::lowerBound<Key, Value>(s_keyA, s_valueA, lenA, s_keyB[s_lenB - 1], s_valueB[s_lenB - 1]);
				s_lenA = s_stopA - s_startA;
			}
			__syncthreads();

			merge<Key, Value>(
				s_keyA + s_startA, s_valueA + s_startA, s_lenA,
				s_keyB, s_valueB, s_lenB,
				d_keyC + (s_startA + startB), d_valueC + (s_startA + startB));
		}

		/// merge two arbitrary arrays
		template <typename Key, typename Value>
		__global__ void mergeSharedNaive(
			Key* d_keyA, Value* d_valueA, int lenA,
			Key* d_keyB, Value* d_valueB, int lenB,
			Key* d_keyC, Value* d_valueC) {

			const int startA = BID * (BLOCK_SIZE - 1);

			// copy BLOCK_SIZE-sized chunk of arrayA to shared memory
			__shared__ Key s_keyA[BLOCK_SIZE];
			__shared__ Value s_valueA[BLOCK_SIZE];

			__shared__ volatile int s_startB, s_stopB, s_lenA, s_lenB;

			if (startA + TID < lenA) {
				s_keyA[TID] = d_keyA[startA + TID];
				s_valueA[TID] = d_valueA[startA + TID];
			}
			__syncthreads();

			// binary search the begin and end indexes of the corresponding chunk of arrayB, BAD (thread divergence)
			if (TID == 0) {
				s_startB = BID == 0 ? 0 : util::upperBound<Key, Value>(d_keyB, d_valueB, lenB, s_keyA[0], s_valueA[0]);
				s_lenA = min(lenA - startA, BLOCK_SIZE);
				s_stopB = BID == BLOCKS_PER_GRID - 1 ? lenB : util::upperBound<Key, Value>(d_keyB, d_valueB, lenB, s_keyA[s_lenA - 1], s_valueA[s_lenA - 1]);
				s_lenB = s_stopB - s_startB;
			}
			__syncthreads();


			// split chunkB into BLOCK_SIZE-sized sub-chunks
			int round = (s_lenB - 1) / BLOCK_SIZE + 1;

			__shared__ Key s_keyB[BLOCK_SIZE];
			__shared__ Value s_valueB[BLOCK_SIZE];

			__shared__ volatile int s_subStartA, s_subStopA, s_subLenA, s_subLenB;

			for (int i = 0; i < round; i++) {
				// copy sub-chunk of arrayB to shared memory
				int subStartB = s_startB + i * BLOCK_SIZE;

				if (subStartB + TID < s_stopB) {
					s_keyB[TID] = d_keyB[subStartB + TID];
					s_valueB[TID] = d_valueB[subStartB + TID];
				}
				__syncthreads();

				if (TID == 0) {
					s_subStartA = i == 0 ? 0 : util::lowerBound<Key, Value>(s_keyA, s_valueA, s_lenA, s_keyB[0], s_valueB[0]);
					s_subLenB = min(s_stopB - subStartB, BLOCK_SIZE);
					s_subStopA = i == round - 1 ? s_lenA : util::lowerBound<Key, Value>(s_keyA, s_valueA, s_lenA, s_keyB[s_subLenB - 1], s_valueB[s_subLenB - 1]);
					s_subLenA = s_subStopA - s_subStartA;
				}
				__syncthreads();

				merge<Key, Value>(
					s_keyA + s_subStartA, s_valueA + s_subStartA, s_subLenA,
					s_keyB, s_valueB, s_subLenB,
					d_keyC + (startA + s_subStartA + subStartB), d_valueC + (startA + s_subStartA + subStartB));
				__syncthreads();
			}

		}

		// merge two non-overlapping sorted arrays
		template <typename Key, typename Value>
		int mergeSorted(Key* d_keyA, Value* d_valueA, int sizeA,
			Key* d_keyB, Value* d_valueB, int sizeB, Key* d_keyC, Value* d_valueC) {

			// for the case of sizeA or sizeB is smaller than BLOCK_SIZE
			if (sizeA <= BLOCK_SIZE || sizeB <= BLOCK_SIZE) {
				if (sizeA <= BLOCK_SIZE) {
					int blocksPerGrid = (sizeB - 1) / BLOCK_SIZE + 1;
					mergeSharedSmall<Key, Value> <<<blocksPerGrid, BLOCK_SIZE >>> (d_keyA, d_valueA, sizeA,
						d_keyB, d_valueB, sizeB, d_keyC, d_valueC);
					CUT_CHECK_ERROR("mergeSharedSmall (arrA)");
				}
				else {
					int blocksPerGrid = (sizeA - 1) / BLOCK_SIZE + 1;
					mergeSharedSmall<Key, Value> <<<blocksPerGrid, BLOCK_SIZE >>>(d_keyB, d_valueB, sizeB,
						d_keyA, d_valueA, sizeA, d_keyC, d_valueC);
					CUT_CHECK_ERROR("mergeSharedSmall (arrB)");
				}
			}
			// for arbitrary size of array1 and array2
			else {
				if (sizeB < sizeA) {
					int blocksPerGrid = (sizeA - 1) / BLOCK_SIZE + 1;
					mergeSharedNaive<Key, Value> <<<blocksPerGrid, BLOCK_SIZE >>> (d_keyA, d_valueA, sizeA,
						d_keyB, d_valueB, sizeB, d_keyC, d_valueC);
					CUT_CHECK_ERROR("mergeSharedNaive (arrA)");
				}
				else {
					int blocksPerGrid = (sizeB - 1) / BLOCK_SIZE + 1;
					mergeSharedNaive<Key, Value> <<<blocksPerGrid, BLOCK_SIZE >>>(d_keyB, d_valueB, sizeB,
						d_keyA, d_valueA, sizeA, d_keyC, d_valueC);
					CUT_CHECK_ERROR("mergeSharedNaive (arrB)");
				}
			}

			return sizeA + sizeB;
		}
	}
}