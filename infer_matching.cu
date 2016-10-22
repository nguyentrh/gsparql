#ifndef CUDA_SMJ_INFERENCE_CU
#define CUDA_SMJ_INFERENCE_CU

#include "common.h"

namespace gsparql {

	/* infer new triples based on the rule format (?a ?p1 ?b) x (?b ?p2 ?c) -> (?a ?p3 ?c)  */
	/* use sorted merge join (?b reverse(?p1) ?a) x (?b ?p2 ?c) -> (?a ?p3 ?c)  */
	

	__device__ int lowerBound(long* dev_keyS, int lenS, long key) {
		int lo = 0;
		int hi = lenS - 1;
		int mid;
		while (lo <= hi) {
			mid = lo + (hi - lo) / 2;

			if (dev_keyS[mid] < key)
				lo = mid + 1;
			else
				hi = mid - 1;
		}

		return lo;
	}

	__device__ int upperBound(long* dev_keyS, int lenS, long key) {
		int lo = 0;
		int hi = lenS - 1;
		int mid;
		while (lo <= hi) {
			mid = lo + (hi - lo) / 2;

			if (dev_keyS[mid] > key)
				hi = mid - 1;
			else
				lo = mid + 1;
		}

		return hi;
	}

	__global__ void merge_join_count(long* dev_keyR, int lenR, long* dev_keyS, int lenS, int* dev_count) {
		
		__shared__ long s_keyR[BLOCK_SIZE]; // store keys of relation R
		
		const int threadId = GTID;
		if (threadId < lenR) {
			s_keyR[TID] = dev_keyR[threadId]; 
			__syncthreads();
		}


	}
}

#endif