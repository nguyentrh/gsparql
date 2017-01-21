#pragma once

#include "../inc/define.h"
#include "../inc/device_func.h"

#define UMUL(a, b) __umul24((a), (b))
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

namespace gsparql {
	namespace sort {

		inline int getNP2(int n) {
			int np2 = 1;
			while (np2 < n || np2 % SHARED_SIZE_LIMIT != 0) np2 <<= 1;
			return np2;
		}

		/// compare and swap key-value pairs
		template <typename Key, typename Value>
		__device__ inline void comparator(
			Key& keyA, Value& valA,
			Key& keyB, Value& valB,
			int dir){

			if (keyA == keyB) {
				if ((valA > valB) == dir) {
					util::swap<Key, Value>(keyA, valA, keyB, valB);
				}
			}
			else if ((keyA > keyB) == dir) {
				util::swap<Key, Value>(keyA, valA, keyB, valB);
			}

		}

		/* Monolithic bitonic sort kernel for very short arrays fitting into shared memory */
		template <typename Key, typename Value>
		__global__ void bitonicSortBlocked(
			Key *d_key, Value *d_val,
			int len, int dir ) {

			//Shared memory storage for one or more short vectors
			__shared__ Key s_key[SHARED_SIZE_LIMIT];
			__shared__ Value s_val[SHARED_SIZE_LIMIT];

			//Offset to the beginning of subbatch and load data
			d_key += BID * SHARED_SIZE_LIMIT + TID;
			d_val += BID * SHARED_SIZE_LIMIT + TID;
			s_key[TID] = d_key[0];
			s_val[TID] = d_val[0];
			s_key[TID + (SHARED_SIZE_LIMIT / 2)] = d_key[(SHARED_SIZE_LIMIT / 2)];
			s_val[TID + (SHARED_SIZE_LIMIT / 2)] = d_val[(SHARED_SIZE_LIMIT / 2)];

			for (int size = 2; size < len; size <<= 1) {
				
				//Bitonic merge
				int ddd = dir ^ ((TID & (size / 2)) != 0);

				for (int stride = size / 2; stride > 0; stride >>= 1) {
					__syncthreads();

					int pos = 2 * TID - (TID & (stride - 1));
					comparator<Key, Value>(
						s_key[pos], s_val[pos],
						s_key[pos + stride], s_val[pos + stride],
						ddd);
				}
			}

			//ddd == dir for the last bitonic merge step
			for (int stride = len / 2; stride > 0; stride >>= 1) {
				__syncthreads();
				int pos = 2 * TID - (TID & (stride - 1));
				comparator<Key, Value>(
					s_key[pos], s_val[pos],
					s_key[pos + stride], s_val[pos + stride],
					dir);
			}

			__syncthreads();
			d_key[0] = s_key[TID];
			d_val[0] = s_val[TID];
			d_key[(SHARED_SIZE_LIMIT / 2)] = s_key[TID + (SHARED_SIZE_LIMIT / 2)];
			d_val[(SHARED_SIZE_LIMIT / 2)] = s_val[TID + (SHARED_SIZE_LIMIT / 2)];
		}

		/// build bitonic array in shared memory
		template <typename Key, typename Value>
		__global__ void bitonicSortShared(Key *d_key, Value *d_val){

			//Shared memory storage for current subarray
			__shared__ Key s_key[SHARED_SIZE_LIMIT];
			__shared__ Value s_val[SHARED_SIZE_LIMIT];

			//Offset to the beginning of subarray and load data
			d_key += BID * SHARED_SIZE_LIMIT + TID;
			d_val += BID * SHARED_SIZE_LIMIT + TID;
			s_key[TID] = d_key[0];
			s_val[TID] = d_val[0];
			s_key[TID + (SHARED_SIZE_LIMIT / 2)] = d_key[(SHARED_SIZE_LIMIT / 2)];
			s_val[TID + (SHARED_SIZE_LIMIT / 2)] = d_val[(SHARED_SIZE_LIMIT / 2)];

			for (int size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) {
				//Bitonic merge
				int ddd = (TID & (size / 2)) != 0;

				for (int stride = size / 2; stride > 0; stride >>= 1) {
					__syncthreads();
					int pos = 2 * TID - (TID & (stride - 1));
					comparator<Key, Value>(
						s_key[pos], s_val[pos],
						s_key[pos + stride], s_val[pos + stride],
						ddd);
				}
			}

			//Odd / even arrays of SHARED_SIZE_LIMIT elements
			//sorted in opposite directions
			int ddd = BID & 1;
			for (int stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
				__syncthreads();
				int pos = 2 * TID - (TID & (stride - 1));
				comparator<Key, Value>(
					s_key[pos], s_val[pos],
					s_key[pos + stride], s_val[pos + stride],
					ddd);
			}

			__syncthreads();
			d_key[0] = s_key[TID];
			d_val[0] = s_val[TID];
			d_key[(SHARED_SIZE_LIMIT / 2)] = s_key[TID + (SHARED_SIZE_LIMIT / 2)];
			d_val[(SHARED_SIZE_LIMIT / 2)] = s_val[TID + (SHARED_SIZE_LIMIT / 2)];
		}

		// Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
		template <typename Key, typename Value>
		__global__ void bitonicMergeGlobal(
			Key *d_key, Value *d_val, int len, 
			int size, int stride, int dir){

			int global_comparatorI = GTID;
			int        comparatorI = global_comparatorI & (len / 2 - 1);

			//Bitonic merge
			int ddd = dir ^ ((comparatorI & (size / 2)) != 0);
			int pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

			Key keyA = d_key[pos];
			Value valA = d_val[pos];
			Key keyB = d_key[pos + stride];
			Value valB = d_val[pos + stride];

			comparator<Key, Value>( keyA, valA, keyB, valB, ddd);

			d_key[pos] = keyA;
			d_val[pos] = valA;
			d_key[pos + stride] = keyB;
			d_val[pos + stride] = valB;
		}

		//Combined bitonic merge steps for
		//size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
		template <typename Key, typename Value>
		__global__ void bitonicMergeShared(
			Key *d_key, Value *d_val, int len, 
			int size, int dir) {

			//Shared memory storage for current subarray
			__shared__ int s_key[SHARED_SIZE_LIMIT];
			__shared__ int s_val[SHARED_SIZE_LIMIT];

			d_key += BID * SHARED_SIZE_LIMIT + TID;
			d_val += BID * SHARED_SIZE_LIMIT + TID;
			s_key[TID] = d_key[0];
			s_val[TID] = d_val[0];
			s_key[TID + (SHARED_SIZE_LIMIT / 2)] = d_key[(SHARED_SIZE_LIMIT / 2)];
			s_val[TID + (SHARED_SIZE_LIMIT / 2)] = d_val[(SHARED_SIZE_LIMIT / 2)];

			//Bitonic merge
			int comparatorI = UMAD(BID, THREADS_PER_BLOCK, TID) & ((len / 2) - 1);
			int ddd = dir ^ ((comparatorI & (size / 2)) != 0);

			for (int stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
			{
				__syncthreads();
				int pos = 2 * TID - (TID & (stride - 1));

				comparator<Key, Value>(
					s_key[pos], s_val[pos],
					s_key[pos + stride], s_val[pos + stride],
					ddd);
			}

			__syncthreads();
			d_key[0] = s_key[TID];
			d_val[0] = s_val[TID];
			d_key[(SHARED_SIZE_LIMIT / 2)] = s_key[TID + (SHARED_SIZE_LIMIT / 2)];
			d_val[(SHARED_SIZE_LIMIT / 2)] = s_val[TID + (SHARED_SIZE_LIMIT / 2)];
		}


		template <typename Key, typename Value>
		__global__ void fill(Key* d_key, Value* d_value, int size, Key key, Value val) {
			int pos = GTID;
			if (pos < size) {
				d_key[pos] = key;
				d_value[pos] = val;
			}
		}

		template <typename Key, typename Value, Key maxKey, Value maxVal>
		void bitonicSort(Key* d_key, Value* d_value, int len) {
			// extend the size of the sorted array to 2^n
			
			int np2Size = getNP2(len);
			
			Key* d_tempKey;
			Value* d_tempVal;
			GPUMALLOC(&d_tempKey, np2Size * sizeof(Key));
			GPUMALLOC(&d_tempVal, np2Size * sizeof(Value));

			GPUTOGPU(d_tempKey, d_key, len * sizeof(Key));
			GPUTOGPU(d_tempVal, d_value, len * sizeof(Value));

			if (len < np2Size) {
				int diff = np2Size - len;
				int blocksPerGrid = (diff - 1) / BLOCK_SIZE + 1;
				fill<Key, Value> <<< blocksPerGrid, BLOCK_SIZE >>>(d_tempKey + len, d_tempVal + len, diff, maxKey, maxVal);
				CUT_CHECK_ERROR("fill (tempArr)");
			}

			int dir = 1;
			int blocksPerGrid = np2Size / SHARED_SIZE_LIMIT;

			if (np2Size <= SHARED_SIZE_LIMIT) { // For small-size arrays, we execute an kernel which utilizes shared memory
				bitonicSortBlocked<Key, Value> <<< blocksPerGrid, BLOCK_SIZE >>>(
					d_tempKey, d_tempVal, np2Size, dir);
				CUT_CHECK_ERROR("bitonicSortBlocked (tempArr)");
			}
			else { // For large arrays
				bitonicSortShared<Key, Value> <<< blocksPerGrid, BLOCK_SIZE >>>(d_tempKey, d_tempVal);
				CUT_CHECK_ERROR("bitonicSortBlocked (tempArr)");

				for (unsigned size = 2 * SHARED_SIZE_LIMIT; size <= np2Size; size <<= 1) {
					for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
						if (stride >= SHARED_SIZE_LIMIT) {
							bitonicMergeGlobal<Key, Value> <<< blocksPerGrid, BLOCK_SIZE >>>(
								d_tempKey, d_tempVal, np2Size, size, stride, dir);
							CUT_CHECK_ERROR("bitonicMergeGlobal (tempArr)");
						}
						else
						{
							bitonicMergeShared <Key, Value> <<< blocksPerGrid, BLOCK_SIZE >>>(
								d_tempKey, d_tempVal, np2Size, size, dir);
							CUT_CHECK_ERROR("bitonicMergeShared (tempArr)");
							break;
						}
					}
				}
			}

			GPUTOGPU(d_key, d_tempKey, len * sizeof(Key));
			GPUTOGPU(d_value, d_tempVal, len * sizeof(Value));

			GPUFREE(d_tempKey);
			GPUFREE(d_tempVal);
		}
	}
}