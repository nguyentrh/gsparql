#ifndef CUDA_PRIMITIVE_H
#define CUDA_PRIMITIVE_H

#include "common.h"

/* parallel prefix sum on gpus */
extern "C" int prefexSum(int* dev_inArr, int* dev_outArr, int numRecords);

/* bitonic sort property table*/
extern "C" void bitonicSort(index_t* dev_key, index_t* dev_value, int numRecords, int dir);

#endif