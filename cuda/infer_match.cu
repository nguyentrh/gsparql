#ifndef CUDA_SMJ_INFERENCE_CU
#define CUDA_SMJ_INFERENCE_CU

#include "../define/common.h"
#include "../define/cuda_primitive.h"

namespace gsparql {

	#define KERNEL_COUNT_INFERENCES(grid,block) merge_join_count<<<grid,block>>>
	#define KERNEL_WRITE_INFERENCES(grid,block) merge_join_write<<<grid,block>>>


	/* find the lower index of 'key' in S relation */
	__device__ int lowerBound(index_t* dev_keyR, int lenR, index_t key) {
		int lo = 0;
		int hi = lenR - 1;
		int mid;
		while (lo <= hi) {
			mid = lo + (hi - lo) / 2;

			if (dev_keyR[mid] < key)
				lo = mid + 1;
			else
				hi = mid - 1;
		}

		if (lo + 1 < lenR && dev_keyR[lo] < key) {
			return lo + 1;
		}

		return lo;
	}

	/* find the upper index of 'key' in S relation */
	__device__ int upperBound(index_t* dev_keyR, int lenR, index_t key) {
		int lo = 0;
		int hi = lenR - 1;
		int mid;
		while (lo <= hi) {
			mid = lo + (hi - lo) / 2;

			if (dev_keyR[mid] > key)
				hi = mid - 1;
			else
				lo = mid + 1;
		}

		if (hi != -1 && dev_keyR[hi] > key) {
			return hi - 1;
		}

		return hi;
	}

	/* count number of matches of keyR in a thread */
	__global__ void merge_join_count(
		index_t* dev_keyL, int lenL, 
		index_t* dev_keyR, int lenR, 
		int* dev_count
		) {
		
		const int threadId = GTID;
		
		__shared__ index_t s_keyL[BLOCK_SIZE]; // store keys of Left relation
		__shared__ volatile int s_startR; // start index of matched Right relation
		__shared__ volatile int s_stopR; // stop index of matched Right relation
		__shared__ volatile int s_numL;
		
		// load a block of rel R to shared memory
		if (threadId < lenL) {
			s_keyL[TID] = dev_keyL[threadId]; 
		}
		__syncthreads();

		// find start and stop indexes in rel S, based on the shared data in current thread block
		if (WTID == 0) {
			if (WID == 0) {
				s_startR = lowerBound(dev_keyR, lenR, s_keyL[0]);
			}

			if (WID == 1) {
				s_numL = min(lenL - BID * BLOCK_SIZE, BLOCK_SIZE);
				s_stopR = upperBound(dev_keyR, lenR, s_keyL[s_numL - 1]);
			}
		}
		__syncthreads();

		int numResult = 0;
		for (int i = s_startR + TID; i < s_stopR; i += BLOCK_SIZE) { // each thread hold a key in rel S, coaslesed access
			index_t keyR = dev_keyR[i];
			int lo = lowerBound(s_keyL, s_numL, keyR);
			int hi = upperBound(s_keyL, s_numL, keyR);

			if (hi >= lo) {
				numResult += (hi - lo) + 1;
			}
		}

		dev_count[threadId] = numResult;
	}

	/* write matches of keyR in a thread */
	__global__ void merge_join_write(
		index_t* dev_keyL, index_t* dev_valueL, int lenL,
		index_t* dev_keyR, index_t* dev_valueR, int lenR, 
		int* dev_sum, index_t* dev_newKey, index_t* dev_newValue
		) {

		const int threadId = GTID;

		__shared__ index_t s_keyL[BLOCK_SIZE]; // store keyR of rel R
		__shared__ index_t s_valueL[BLOCK_SIZE]; // store valueR of rel R
		__shared__ volatile int s_startR;
		__shared__ volatile int s_stopR;
		__shared__ volatile int s_numL;

		// load a block of rel R to shared memory
		if (threadId < lenL) {
			s_keyL[TID] = dev_keyL[threadId];
			s_valueL[TID] = dev_valueL[threadId];
		}
		__syncthreads();

		// find start and stop indexes in rel S, based on the shared data in current thread block
		if (WTID == 0) {
			if (WID == 0) {
				s_startR = lowerBound(dev_keyR, lenR, s_keyL[0]);
			}

			if (WID == 1) {
				s_numL = min(lenL - BID * BLOCK_SIZE, BLOCK_SIZE);
				s_stopR = upperBound(dev_keyR, lenR, s_keyL[s_numL - 1]);
			}
		}
		__syncthreads();

		int base = dev_sum[threadId];

		for (int i = s_startR + TID; i < s_stopR; i += BLOCK_SIZE) { // each thread hold a key in rel S, coaslesed access
			index_t keyR = dev_keyR[i];
			index_t valueR = dev_valueR[i];

			int lo = lowerBound(s_keyL, s_numL, keyR);

			for (int j = lo; j < s_numL; j++)
				if (s_keyL[j] == keyR) {
					dev_newKey[base] = s_valueL[j];
					dev_newValue[base] = valueR;
					base++;
				}
				else break;
		}
	}

	/* infer new triples based on the rule format (?b ?p1 ?a) x (?b ?p2 ?c) -> (?a ?p3 ?c)  */
	int inferMergeRules(PropTable* dev_pL, PropTable* dev_pR, PropTable* dev_new) {
		int numResult;

		int lenL = dev_pL->tupleCount;
		int lenR = dev_pR->tupleCount;

		int* dev_count;
		GPUMALLOC(&dev_count, lenL * sizeof(int));

		int blocksPerGrid = (lenL - 1) / BLOCK_SIZE + 1;
		//KERNEL_COUNT_INFERENCES(blocksPerGrid, BLOCK_SIZE) (dev_pL->key, lenL, dev_pR->key, lenR, dev_count);

		int* dev_sum;
		GPUMALLOC(&dev_sum, lenL * sizeof(int));
		numResult = prefexSum(dev_count, dev_sum, lenL);

		if (numResult == 0) return 0;

		

		return numResult;
	}
}

#endif