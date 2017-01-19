#pragma once

#include "define.h"

namespace gsparql {
	namespace util {

		/// compare key-value pares
		template <typename Key, typename Value>
		static inline __device__ __host__ int compare(Key keyA, Value valA, Key keyB, Value valB) {
			if (keyA < keyB) return -1;
			if (keyA > keyB) return 1;

			if (valA < valB) return -1;
			if (valA > valB) return 1;

			return 0;
		}

		/// swap keys
		template <typename Key>
		static inline __device__ __host__ void swap(Key &keyA, Key &keyB) {
			Key keyTemp;
			// swap two keys
			keyTemp = keyA;
			keyA = keyB;
			keyB = keyTemp;
		}

		/// swap key-value pairs
		template <typename Key, typename Value>
		static inline __device__ __host__ void swap(Key &keyA, Value &valA, Key &keyB, Value &valB) {
			Key keyTemp;
			// swap two keys
			keyTemp = keyA;
			keyA = keyB;
			keyB = keyTemp;

			Value valTemp;
			// swap two values
			valTemp = valA;
			valA = valB;
			valB = valTemp;
		}

		/// find the lowerbound of a key in an array
		template <typename Key>
		static inline __device__ __host__ int lowerBound(Key* d_arr, int len, Key key) {
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
		static inline __device__ __host__ int lowerBound(Key* d_keys, Value* d_vals, int len, Key key, Value val) {
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
		static inline __device__ __host__ int upperBound(Key* d_arr, int len, Key key) {
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
		static inline __device__ __host__ int upperBound(Key* d_keys, Value* d_vals, int len, Key key, Value val) {
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
		static inline __device__ __host__ int binarySearch(Key* d_arr, int len, Key key) {
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
		static inline __device__ __host__ int binarySearch(Key* d_keys, Value* d_vals, int len, Key key, Value val) {
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
	}
}