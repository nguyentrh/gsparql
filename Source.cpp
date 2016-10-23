#include <iostream>

using namespace std;

// find the first index of 'key' in S relation
int lowerBound(long* dev_keyS, int lenS, long key) {
	int lo = 0;
	int hi = lenS - 1;
	int mid;
	while (lo <= hi) {
		mid = lo + (hi - lo) / 2;

		if (dev_keyS[mid] < key)
			lo = mid + 1;
		else
			hi = mid - 1;
	}

	if (lo + 1 < lenS && dev_keyS[lo] < key) {
		return lo + 1;
	}

	return lo;
}

// find the last index of 'key' in S relation
int upperBound(long* dev_keyS, int lenS, long key) {
	int lo = 0;
	int hi = lenS - 1;
	int mid;
	while (lo <= hi) {
		mid = lo + (hi - lo) / 2;

		if (dev_keyS[mid] > key)
			hi = mid - 1;
		else
			lo = mid + 1;
	}

	if (hi != -1 && dev_keyS[hi] > key) {
		return hi - 1;
	}

	return hi;
}

int main() {
	int size = 5;
	long* arr = new long[5] {2, 2, 4, 5, 6};

	//for (int i = 0; i < size; i++)
	//	cout << arr[i] << endl;

	cout << lowerBound(arr, size, 7) << endl;
	cout << upperBound(arr, size, 7) << endl;

}