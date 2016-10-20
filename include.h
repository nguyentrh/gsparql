#ifndef COMMON_STRUCT_H
#define COMMON_STRUCT_H

namespace gsparql {

	enum
	{
		/* data type supported in schema */
		URI,
		STRING,
		INT,
		LONG,
		DOUBLE,
		FLOAT
	};
	
	struct SubjectObject
	{
		long tupleCount; /* the number of tuples in the relation */
		int attrType[2];
		int attrSize[2];
		char** content;
	};
}

#endif