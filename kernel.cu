#include "define\cuda_primitive.h"
#include <time.h>       /* time */
#include <iostream>

using namespace std;

int main() {
	int size = 16;

	index_t* h_key = new index_t[size];
	index_t* h_value = new index_t[size];
	for (int i = 0; i < size; i++) {
		h_key[i] = rand() % 50;
		h_value[i] = rand() % 50;

		cout << h_key[i] << "(" << h_value[i] << ") ";
	}
	cout << endl;

	index_t* d_key;
	index_t* d_value;

	GPUMALLOC(&d_key, size * sizeof(index_t));
	GPUMALLOC(&d_value, size * sizeof(index_t));

	TOGPU(d_key, h_key, size * sizeof(index_t));
	TOGPU(d_value, h_value, size * sizeof(index_t));

	bitonicSort(d_key, d_value, size, 0);


	FROMGPU(h_key, d_key, size * sizeof(index_t));
	FROMGPU(h_value, d_value, size * sizeof(index_t));

	for (int i = 0; i < size; i++) {
		cout << h_key[i] << "(" << h_value[i] << ") ";
	}
	cout << endl;

	GPUFREE(d_key);
	GPUFREE(d_value);
	
	delete[] h_key;
	delete[] h_value;
}