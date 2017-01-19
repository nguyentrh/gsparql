#include "..\inc\device_func.h"

using namespace gsparql;

/// compare key-value pares
template <typename Key, typename Value>
__device__ int compare(Key k1, Value v1, Key k2, Value v2) {
	if (k1 < k2) return -1;
	if (k1 > k2) return 1;

	if (v1 < v2) return -1;
	if (v1 > v2) return 1;

	return 0;
}

/// swap keys
template <typename Key>
__device__ void swap(Key &k1, Key &k2) {
	Key keyTemp;
	// swap two keys
	keyTemp = k1;
	k1 = k2;
	k2 = keyTemp;
}

/// swap key-value pairs
template <typename Key, typename Value>
__device__ void swap(Key &k1, Value &v1, Key &k2, Value &v2) {
	Key keyTemp;
	// swap two keys
	keyTemp = k1;
	k1 = k2;
	k2 = keyTemp;

	Value valTemp;
	// swap two values
	valTemp = v1;
	v1 = v2;
	v2 = valTemp;
}

/// find the lowerbound of a key in an array
template <typename Key>
__device__ int lowerBound(Key* d_arr, int len, Key key) {
	int lo = 0;
	int hi = len - 1;
	int mid;
	while (lo <= hi) {
		mid = lo + (hi - lo) / 2;

		if (d_arr[mid] < key)
			lo = mid + 1;
		else
			hi = mid - 1;
	}

	if (lo + 1 < len && d_arr[lo] < key) {
		return lo + 1;
	}

	return lo;
}

/// find the lowerbound of a key-val pair in an array
template <typename Key, typename Value>
__device__ int lowerBound(Key* d_keys, Value* d_vals, int len, Key key, Value val) {
	int lo = 0;
	int hi = len - 1;
	int mid;
	while (lo <= hi) {
		mid = lo + (hi - lo) / 2;

		if (compare<Key, Value>(d_keys[mid], d_vals[mid], key, val) < 0)
			lo = mid + 1;
		else
			hi = mid - 1;
	}

	if (lo + 1 < len && compare<Key, Value>(d_keys[lo], d_vals[lo], key, val) < 0) {
		return lo + 1;
	}

	return lo;
}

/// find the upperbound of a key in an array
template <typename Key>
__device__ int upperBound(Key* d_arr, int len, Key key) {
	int lo = 0;
	int hi = len - 1;
	int mid;
	while (lo <= hi) {
		mid = lo + (hi - lo) / 2;

		if (d_arr[mid] > key)
			hi = mid - 1;
		else
			lo = mid + 1;
	}

	if (hi != -1 && d_arr[hi] > key) {
		return hi;
	}

	return hi + 1;
}

/// find the upperbound of a key-val pair in an array
template <typename Key, typename Value>
__device__ int upperBound(Key* d_keys, Value* d_vals, int len, Key key, Value val) {
	int lo = 0;
	int hi = len - 1;
	int mid;
	while (lo <= hi) {
		mid = lo + (hi - lo) / 2;

		if (compare<Key, Value>(d_keys[mid], d_vals[mid], key, val) > 0)
			hi = mid - 1;
		else
			lo = mid + 1;
	}

	if (hi != -1 && compare<Key, Value>(d_keys[hi], d_vals[hi], key, val) > 0) {
		return hi;
	}

	return hi + 1;
}

/// binary search on an array, return -1 if not found
template <typename Key>
__device__ int binarySearch(Key* d_arr, int len, Key key) {
	int lo = 0;
	int hi = len - 1;
	int mid;
	while (lo <= hi) {
		mid = lo + (hi - lo) / 2;

		Key curr = d_arr[mid];

		if (curr > key)
			hi = mid - 1;
		else if (curr < key)
			lo = mid + 1;
		else return mid;
	}

	return -1;
}

template <typename Key, typename Value>
__device__ int binarySearch(Key* d_keys, Value* d_vals, int len, Key key, Value val) {
	int lo = 0;
	int hi = len - 1;
	int mid;
	while (lo <= hi) {
		mid = lo + (hi - lo) / 2;

		int comp = compare<Key, Value>(d_keys[mid], d_vals[mid], key, val);

		if (comp > 0)
			hi = mid - 1;
		else if (comp < 0)
			lo = mid + 1;
		else return mid;
	}

	return -1;
}