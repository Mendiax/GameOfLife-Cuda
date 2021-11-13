#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"

#define maxInt  unsigned long long int;

namespace
{
	bool* dev_cellsStatusIn_p;
	bool* dev_cellsStatusOut_p;
	bool* cellsStatusBuffer_p;
	unsigned long int cellsStatusLength;
}
namespace gpu
{

	__global__ void calculateKernel(bool* boxesStatusIn, bool* boxesStatusOut, unsigned long long int sizeOfArray);

	void flipCellStatus(unsigned long long int x);

	cudaError_t calculateWithCuda(bool* statusArray);

	cudaError_t mallocMemory(unsigned long long int arrayLength);

	void freeMemory();
};

#endif
