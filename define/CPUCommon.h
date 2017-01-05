#ifndef CPU_COMMON_H
#define CPU_COMMON_H

#include "common.h"

namespace sparql {

	/* read column content from a binary file */
	extern "C" bool read(Column* col, char* input);

	/* write column content to a binary file */
	extern "C" bool write(Column* col, char* input);
}

#endif