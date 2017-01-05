#ifndef COMMON_DEFINIION_H
#define COMMON_DEFINIION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil.h"
#include "Timer.h"

#ifdef __INTELLISENSE__
#define __launch_bounds__(a,b)
void __syncthreads(void);
void __threadfence(void);
int __mul24(int, int);
#endif

#define DEBUG 1

/* define constant cuda parameters */
#define WARP_SIZE (32)
#define BLOCK_SIZE (16)

#define THREADS_PER_BLOCK (blockDim.x)
#define WARPS_PER_BLOCK ((THREADS_PER_BLOCK-1)/WARP_SIZE + 1)
#define BLOCKS_PER_GRID (gridDim.x)

#define BID	(blockIdx.x)
#define TID	(threadIdx.x)	/*thread ID in current block*/
#define WID (TID / WARP_SIZE)
#define WTID (TID % WARP_SIZE)

#define GTID (BID * THREADS_PER_BLOCK + TID)	/*global thread ID*/
#define GWID (GTID / WARP_SIZE)
#define GWTID (GTID % WARP_SIZE)

#define TOTAL_THREADS (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

#define SHARED_MEMORY_PER_PROCESSOR (8*1024)
#define TOTAL_PROCESSORS (14)

/* define short functions */
#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
			    }} while(0)

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#define GPUMALLOC(pointer, size) CUDA_SAFE_CALL(cudaMalloc( pointer, size))
#define CPUMALLOC(pointer, size) CUDA_SAFE_CALL(cudaMallocHost (pointer, size))

#define CPUFREE(pointer) if(pointer!=NULL) CUDA_SAFE_CALL(cudaFreeHost (pointer))
#define GPUFREE(pointer) CUDA_SAFE_CALL( cudaFree( pointer) )

#define TOGPU(dev_pointer,hos_pointer, size)  CUDA_SAFE_CALL(cudaMemcpy(dev_pointer,hos_pointer, size, cudaMemcpyHostToDevice))
#define FROMGPU(hos_pointer, dev_pointer, size)  CUDA_SAFE_CALL(cudaMemcpy(hos_pointer, dev_pointer, size, cudaMemcpyDeviceToHost))
#define GPUTOGPU(dev_to, dev_from, size)  CUDA_SAFE_CALL(cudaMemcpy(dev_to, dev_from, size, cudaMemcpyDeviceToDevice))

#define GPUPARAM(grid, block) <<<grid, block>>>

namespace sparql {

	//---------------------------------------------------------------------------
	enum
	{
		/* data type supported in schema */
		URI,
		STRING,
		INT,
		LONG,
		DOUBLE,
		FLOAT,
		BOOL,
		NONE,

		/* supported relation in exp */
		EQ,
		GTH,
		LTH,
		GEQ,
		LEQ,
		NEQ,

		/* rule types */
		COPY,
		INV,
		TRANS,
		JOIN_SS,
		JOIN_SO,
		JOIN_SP,
		JOIN_PS,
		JOIN_PO,
		JOIN_PP,
		JOIN_OS,
		JOIN_OO,
		JOIN_OP,

		/* data format */
		RAW,
		COMPRESSED
	};

	//---------------------------------------------------------------------------
	struct Triple {
		int subjectType;
		int predicateType;
		int objectType;

		char subject[8];
		char predicate[8];
		char object[8];
	};

	struct RuleNode {
		int ruleId;
		int ruleType;				/* the types of rules */
		int unaryBinary;			/* only support unary and binary rules for simplicity */

		struct TripleNode* tail1;	/* first triple in the LHS of the rule */
		struct TripleNode* tail2;	/* second triple in the LHS of the rule */

		struct TripleNode* head;	/* triple in the RHS of the rules */
	};

	struct TripleNode {
		int tripleId;
		Triple* triple;				/* triple description of node */

		int headRuleNum;				/* number of rules use triple as LHS triple */
		struct RuleNode* headRules;		/* set of rules */

		int tailRuleNum;				/* number of rules use triple as RHS triple */
		struct RuleNode* tailRules;		/* set of tail rules */
	};

	struct ReasonTree {
		int tripleNodeNum;
		int ruleNodeNum;

		int rootTripleId;

		TripleNode* tripleNodes;
		RuleNode* ruleNodes;
	};

	//---------------------------------------------------------------------------
	struct Column {
		long tupleNum;		/* the total number of tuples in this column */
		long dataSize;     /* the size of this column in bytes */

		int dataType;		/* the type of this column content */
		int format;         /* the format of this column */

		char* content;		/* the content of this column */

		Column() : tupleNum(0), dataSize(0), dataType(URI), format(RAW), content(NULL){}
		~Column() { if (dataSize > 0) free(content); }
	};

	struct PropTable {
		int predicateId;

		struct Column* subject;
		struct Column* object;

		int sorted;						/* 0: un-sorted; 1: sorted by subject; 2: sorted by object */

		PropTable() : predicateId(0), subject(NULL), object(NULL){}
		~PropTable() {
			if (subject != NULL) delete subject;
			if (object != NULL) delete object;
		}
	};

	struct TripleStore {
		int predicateNum;
		int datatypeNum;

		struct PropTable** propTables;

		TripleStore() : predicateNum(0), datatypeNum(0), propTables(NULL) {}
		~TripleStore() {
			if (predicateNum > 0) {
				for (int i = 0; i < predicateNum; i++) {
					delete[] propTables[i];
				}
				delete[] propTables;
			}
		}
	};

}

#endif