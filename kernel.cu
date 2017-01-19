#include <time.h>       /* time */
#include <iostream>
#include "inc\define.h"
#include "inc\merge.cuh"
#include "inc\Timer.h"
#include "test\mergeTest.h"

using namespace std;
using namespace gsparql;

#define TEST_SIZE1 400
#define TEST_SIZE2 500

int main() {
	int key1[TEST_SIZE1];
	int val1[TEST_SIZE1];
	int key2[TEST_SIZE2];
	int val2[TEST_SIZE2];

	for (int i = 0; i < TEST_SIZE1; i++) {
		key1[i] = i * 2 + 2;
		val1[i] = key1[i] * 2;
	}

	for (int i = 0; i < TEST_SIZE2; i++) {
		key2[i] = i * 4 + 1;
		val2[i] = key2[i] * 2;
	}

	/*
	cout << "Array 1:\n";
	for (int i = 0; i < TEST_SIZE; i++) {
		cout << key1[i] << "|" << val1[i] << "  ";
	}
	cout << endl;

	cout << "Array 2:\n";
	for (int i = 0; i < TEST_SIZE; i++) {
		cout << key2[i] << "|" << val2[i] << "  ";
	}
	cout << endl;
	*/

	int* d_key1;
	GPUMALLOC(&d_key1, TEST_SIZE1 * sizeof(int));
	TOGPU(d_key1, key1, TEST_SIZE1 * sizeof(int));

	int* d_val1;
	GPUMALLOC(&d_val1, TEST_SIZE1 * sizeof(int));
	TOGPU(d_val1, val1, TEST_SIZE1 * sizeof(int));

	int* d_key2;
	GPUMALLOC(&d_key2, TEST_SIZE2 * sizeof(int));
	TOGPU(d_key2, key2, TEST_SIZE2 * sizeof(int));

	int* d_val2;
	GPUMALLOC(&d_val2, TEST_SIZE2 * sizeof(int));
	TOGPU(d_val2, val2, TEST_SIZE2 * sizeof(int));

	int outSize = TEST_SIZE1 + TEST_SIZE2;
	int* outKey1 = new int[outSize];
	int* outVal1 = new int[outSize];

	int* d_outKey;
	GPUMALLOC(&d_outKey, outSize * sizeof(int));
	int* d_outVal;
	GPUMALLOC(&d_outVal, outSize * sizeof(int));

	cout << "start\n";
	
	Timer timer;

	timer.start();
	merge::mergeSorted<int, int>(d_key1, d_val1, TEST_SIZE1, d_key2, d_val2, TEST_SIZE2, d_outKey, d_outVal);
	timer.stop();

	cout << "time: " << timer.getElapsedTimeInMilliSec() << " ms\n";
	
	FROMGPU(outKey1, d_outKey, outSize * sizeof(int));
	FROMGPU(outVal1, d_outVal, outSize * sizeof(int));

	int* outKey2 = new int[outSize];
	int* outVal2 = new int[outSize];

	timer.start();
	mergeSortedCPU<int, int>(key1, val1, TEST_SIZE1, key2, val2, TEST_SIZE2, outKey2, outVal2);
	timer.stop();

	cout << "time: " << timer.getElapsedTimeInMilliSec() << " ms\n";

	if (compareArray<int, int>(outKey1, outVal1, outKey2, outVal2, outSize) == true) {
		cout << "Verify OK\n";
	}
	else {
		cout << "Verify failed\n";
	}
	
	//for (int i = 0; i < OUT_SIZE; i++) {
	//	cout << outKey1[i] << "  ";
	//}
	//cout << endl;

	GPUFREE(d_key1);
	GPUFREE(d_val1);
	GPUFREE(d_key2);
	GPUFREE(d_val2);
	GPUFREE(d_outKey);
	GPUFREE(d_outVal);
	
	delete[] outKey1;
	delete[] outKey2;
	delete[] outVal1;
	delete[] outVal2;

	return 0;
}