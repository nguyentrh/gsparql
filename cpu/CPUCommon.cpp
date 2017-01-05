#include "../define/common.h"

using namespace sparql;

bool read(Column* col, char* input) {
	FILE* fp;
	fp = fopen(input, "r");

	if (fp == NULL) return false;

	fread(&col->tupleNum, sizeof(long), 1, fp);
	fread(&col->dataSize, sizeof(long), 1, fp);
	fread(&col->dataType, sizeof(int), 1, fp);
	fread(&col->format, sizeof(int), 1, fp);
	
	col->content = (char*)malloc(col->dataSize);
	CHECK_POINTER(col->content);

	fread(col->content, sizeof(char), col->dataSize, fp);

	return true;
}

bool write(Column* col, char* input) {
	FILE* fp;
	fp = fopen(input, "w");

	if (fp == NULL) return false;

	fwrite(&col->tupleNum, sizeof(long), 1, fp);
	fwrite(&col->dataSize, sizeof(long), 1, fp);
	fwrite(&col->dataType, sizeof(int), 1, fp);
	fwrite(&col->format, sizeof(int), 1, fp);

	fwrite(col->content, sizeof(char), col->dataSize, fp);

	return true;
}
