#pragma once

#include <stdio.h>

using namespace std;

namespace gsparql {

// merge two non-overlapping sorted arrays
	template <typename Key, typename Value>
	int mergeSortedCPU(Key* key1, Value* value1, int size1,
		Key* key2, Value* value2, int size2, Key* keyOut, Value* valueOut) {

		int i = 0;
		int j = 0;
		int k = 0;

		while (i < size1 && j < size2) {
			int cmp = util::compare<Key, Value>(key1[i], value1[i], key2[j], value2[j]);
			if (cmp < 0) {
				keyOut[k] = key1[i];
				valueOut[k] = value1[i];
				i++;
				k++;
			}
			else if (cmp > 0) {
				keyOut[k] = key2[j];
				valueOut[k] = value2[j];
				j++;
				k++;
			}
			else {
				keyOut[k] = key1[j];
				valueOut[k] = value1[j];
				j++;
				i++;
				k++;
			}
		}

		while (i < size1) {
			keyOut[k] = key1[i];
			valueOut[k] = value1[i];
			i++;
			k++;
		}

		while (j < size2) {
			keyOut[k] = key2[j];
			valueOut[k] = value2[j];
			j++;
			k++;
		}

		return k;
	}

	template <typename Key, typename Value>
	bool compareArray(Key* key1, Value* value1,
		Key* key2, Value* value2, int size) {
		for (int i = 0; i < size; i++) {
			if (util::compare<Key, Value>(key1[i], value1[i], key2[i], value2[i]) != 0) {
				printf("index %d k1 %d v1 %d k2 %d v2 %d\n", i, key1[i], value1[i], key2[i], value2[i]);
				return false;
			}
		}

		return true;
	}
}