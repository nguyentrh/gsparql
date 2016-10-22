#ifndef COMMON_DEFINIION_H
#define COMMON_DEFINIION_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cutil.h"
#include "constant_define.h"
#include "struct_define.h"

#ifdef __INTELLISENSE__
#define __launch_bounds__(a,b)
void __syncthreads(void);
void __threadfence(void);
int __mul24(int, int);
#endif

#endif