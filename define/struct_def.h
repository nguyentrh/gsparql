#ifndef COMMON_STRUCT_H
#define COMMON_STRUCT_H

/* define data types */
typedef unsigned int index_t;

namespace gsparql {
	enum
	{
		/* data type supported in schema */
		URI,
		STRING,
		INT,
		LONG,
		DOUBLE,
		FLOAT,
		BOOL

	};
	
	struct PropTable /* property_table = {key (URI), value (data_type)} */
	{
		long tupleCount; /* the number of tuples in the relation */
		int valueType;
		int valueSize;

		index_t* key;
		char* value;
	};

	struct JoinTable {
		long tupleCount;
	};


}

#endif