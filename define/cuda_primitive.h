#ifndef CUDA_PRIMITIVE_H
#define CUDA_PRIMITIVE_H

/* parallel prefix sum on gpus */
int prefexSum(int* dev_inArr, int* dev_outArr, int numRecords);

#endif