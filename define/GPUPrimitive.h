#ifndef CUDA_PRIMITIVE_H
#define CUDA_PRIMITIVE_H

namespace sparql {

	/* parallel prefix sum on gpus */
	extern "C" int prefixSum(int* dev_in, int* dev_out, int size);

}

#endif