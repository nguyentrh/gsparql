#ifndef DEVICE_FUNCTION_H
#define DEVICE_FUNCTION_H

#include "common.h"

namespace sparql {

	// compare 2 values 
	__device__ bool compare(char* a, char* b, int type, int op);

	__device__ int size_of(int type);

}

#endif