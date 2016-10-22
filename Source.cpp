#include <iostream>

using namespace std;

int lowerBound(long* dev_keyS, int lenS, long key) {
	int lo = 0;
	int hi = lenS-1;
	int mid;
	while (lo <= hi) {
		mid = lo + (hi - lo) / 2;

		if (dev_keyS[mid] > key)
			hi = mid - 1;
		else
			lo = mid + 1;
	}

	return hi;
}

int main() {
	int size = 5;
	long* arr = new long[5] {2, 2, 4, 5, 6};

	for (int i = 0; i < size; i++)
		cout << arr[i] << endl;

	cout << lowerBound(arr, size, 1);
}