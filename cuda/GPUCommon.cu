#ifndef CUDA_DEVICE_FUNCTION_CU
#define CUDA_DEVICE_FUNCTION_CU

#include "../define/GPUCommon.h"

using namespace sparql;

/* define short functions */
#define COMPARE(ai, bi, op) if (op == sparql::EQ) return ai == bi; \
				else if (op == sparql::NEQ) return ai != bi;			\
				else if (op == sparql::GTH) return ai > bi;			\
				else if (op == sparql::LTH) return ai < bi;			\
				else if (op == sparql::GEQ) return ai >= bi;			\
				else if (op == sparql::LEQ) return ai <= bi;			\
				else return false

/* compare the values of two elements with the same type */
__device__ bool compare(char* a, char* b, int type, int op) {
	switch (type)
	{
	case sparql::URI:
	case sparql::STRING:
	case sparql::INT:
		int ai, bi;
		memcpy(&ai, a, sizeof(int));
		memcpy(&bi, b, sizeof(int));
		COMPARE(ai, bi, op);

	case sparql::FLOAT:
		float ai, bi;
		memcpy(&ai, a, sizeof(float));
		memcpy(&bi, b, sizeof(float));
		COMPARE(ai, bi, op);

	case sparql::DOUBLE:
		double ai, bi;
		memcpy(&ai, a, sizeof(double));
		memcpy(&bi, b, sizeof(double));
		COMPARE(ai, bi, op);

	case sparql::LONG:
		long ai, bi;
		memcpy(&ai, a, sizeof(long));
		memcpy(&bi, b, sizeof(long));
		COMPARE(ai, bi, op);

	case sparql::BOOL:
		char ai = *a;
		char bi = *b;
		COMPARE(ai, bi, op);

	default:
		return false;
	}
}

/* return the size of datatype in bytes */
__device__ int size_of(int type) {
	switch (type)
	{
	case sparql::URI:
	case sparql::STRING:
	case sparql::INT:
		return sizeof(int);

	case sparql::FLOAT:
		return sizeof(float);

	case sparql::DOUBLE:
		return sizeof(double);

	case sparql::LONG:
		return sizeof(long);

	case sparql::BOOL:
		return sizeof(char);

	default:
		return -1;
	}
}

#endif