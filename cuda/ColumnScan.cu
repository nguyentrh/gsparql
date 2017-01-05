#ifndef COLUMN_SCAN_CU
#define COLUMN_SCAN_CU

#include "../define/GPUCommon.h"

using namespace sparql;

/* filter a column based on conditions */
__global__ void filterColumn(char* column, int length, int dataSize, int dataType, char* con,  int op, int* flags) {
	unsigned int threadId = GTID;

	if (threadId < length) {
		bool res = compare(column + dataSize * threadId, con, dataSize, dataType, op);
		flags[threadId] = (res == true ? 1 : 0);
	}
	else {
		flags[threadId] = 0;
	}
}

/* scan column and return list of offsets */
int scanColumn(Column* d_block, char* d_condition, int dataType, int opType, int* d_offsets) {
	int blocksPerGrid = d_block->tupleNum / BLOCK_SIZE;
	if (d_block->tupleNum % BLOCK_SIZE != 0) blocksPerGrid++;

	int* d_flags;
	GPUMALLOC(&d_flags, d_block->tupleNum * sizeof(int));
	
	int dataSize = d_block->blockSize / d_block->tupleNum;
	filterColumn <<<blocksPerGrid, BLOCK_SIZE>>> (d_block->content, d_block->tupleNum, dataSize, dataType, d_condition, opType, d_flags);
	CUT_CHECK_ERROR("filterColumn");

	int count = prefixSum(d_flags, d_offsets, dataSize);

	GPUFREE(d_flags);
	return count;
}

#endif