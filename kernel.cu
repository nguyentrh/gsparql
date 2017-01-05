#include "define\common.h"
#include <time.h>       /* time */
#include <iostream>

using namespace std;
using namespace sparql;

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Name:                          %s\n", devProp.name);
	printf("Total global memory:           %u\n", devProp.totalGlobalMem);
	printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %u\n", devProp.memPitch);
	printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n", devProp.clockRate);
	printf("Total constant memory:         %u\n", devProp.totalConstMem);
	printf("Texture alignment:             %u\n", devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}

bool setDevice() {
	// Number of CUDA devices
	int devCount = 0;
	cudaGetDeviceCount(&devCount);

	if (DEBUG) {
		printf("CUDA Device Query...\n");
		printf("There are %d CUDA devices.\n", devCount);
	}

	if (devCount == 0) return false;

	// Iterate through devices
	int devIdx;
	int maxSMCount = 0;
	for (int i = 0; i < devCount; ++i)
	{
		// Get device properties
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		
		if (DEBUG) {
			printf("\nCUDA Device #%d\n", i);
			printDevProp(devProp);
		}

		if (maxSMCount < devProp.multiProcessorCount) {
			maxSMCount = devProp.multiProcessorCount;
			devIdx = i;
		}
	}

	CUDA_SAFE_CALL(cudaSetDevice(devIdx));
	return true;
}

TripleStore* readStore(char* folder, char* dbname, int propNum, int typeNum) {
	char filename[256];
	TripleStore* store = new TripleStore();
	store->predicateNum = propNum;
	store->datatypeNum = typeNum;

	store->propTables = new PropTable*[propNum];
	
	for (int i = 0; i < propNum; i++) {
		store->propTables[i] = new PropTable[typeNum];

		for (int j = 0; j < typeNum; j++) {
			// load subject column
			sscanf(filename, "%s/%s_S_%d_%d.COL", folder, dbname, propNum, typeNum);
			Column* subject = new Column();
			if (read(subject, filename)) { store->propTables[i][j].subject = subject; }
			else { delete subject; }

			// load object column
			sscanf(filename, "%s/%s_O_%d_%d.COL", folder, dbname, propNum, typeNum);
			Column* object = new Column();
			if (read(object, filename)) { store->propTables[i][j].object = object; }
			else { delete object; }
		}
	}

	return store;
}

int main() {
	if (setDevice() == false) return -1;

	char* folder = "";
	char* dbname = "";
	int propNum = 1; 
	int typeNum = 1;
	
	// load triples from files to main memory
	TripleStore* store = readStore(folder, dbname, propNum, typeNum);

	

	delete store;
	return 0;
}