#include <time.h>       /* time */
#include <iostream>
#include "inc\define.h"
#include "cuda\merge.cuh"
#include "cuda\merge_join.cuh"
#include "cuda\sort.cuh"
#include "inc\Timer.h"
#include "test\cpu_test.h"
#include "thrust\merge.h"
#include "thrust\device_ptr.h"

using namespace std;
using namespace gsparql;

#define TEST_SIZE1 900
#define TEST_SIZE2 45000

void testMerge() {
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
	cout << "gpu time: " << timer.getElapsedTimeInMilliSec() << " ms\n";

	int* outKey2 = new int[outSize];
	int* outVal2 = new int[outSize];

	timer.start();
	test::mergeSortedCPU<int, int>(key1, val1, TEST_SIZE1, key2, val2, TEST_SIZE2, outKey2, outVal2);
	timer.stop();
	cout << "cpu time: " << timer.getElapsedTimeInMilliSec() << " ms\n";

	FROMGPU(outKey1, d_outKey, outSize * sizeof(int));
	FROMGPU(outVal1, d_outVal, outSize * sizeof(int));

	if (test::compareArray<int, int>(outKey1, outVal1, outKey2, outVal2, outSize) == true) {
		cout << "Verify OK\n";
	}
	else {
		cout << "Verify failed\n";
	}

	thrust::device_ptr<int> d_keyPtr1(d_key1);
	thrust::device_ptr<int> d_keyPtr2(d_key2);
	thrust::device_ptr<int> d_valPtr1(d_val1);
	thrust::device_ptr<int> d_valPtr2(d_val2);
	thrust::device_ptr<int> d_outKeyPtr(d_outKey);
	thrust::device_ptr<int> d_outValPtr(d_outVal);

	timer.start();
	thrust::merge_by_key(d_keyPtr1, d_keyPtr1 + TEST_SIZE1, d_keyPtr2, d_keyPtr2 + TEST_SIZE2, d_valPtr1, d_valPtr2, d_outKeyPtr, d_outValPtr);
	timer.stop();
	cout << "thrust time: " << timer.getElapsedTimeInMilliSec() << " ms\n";

	FROMGPU(outKey1, d_outKey, outSize * sizeof(int));
	FROMGPU(outVal1, d_outVal, outSize * sizeof(int));

	if (test::compareArray<int, int>(outKey1, outVal1, outKey2, outVal2, outSize) == true) {
		cout << "Verify OK\n";
	}
	else {
		cout << "Verify failed\n";
	}

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

}

int main() {
	testMerge();

	return 0;
}