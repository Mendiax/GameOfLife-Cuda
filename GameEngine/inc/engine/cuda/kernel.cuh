#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"

struct GpuData
{
	bool* dev_cellsStatusIn_p = 0;
	bool* dev_cellsStatusOut_p = 0;
	bool* cellsStatusBuffer_p = 0;
	unsigned long long int cellsStatusLength;
	unsigned long long int cellsStatusRowLength;
	bool* lifeArray = 0;
	bool* deathArray = 0;
};
namespace gpu
{

	__global__ void calculateKernel(bool* boxesStatusIn, bool* boxesStatusOut, unsigned long long int row, unsigned long long int sizeOfArray, bool* lifeArray, bool* deathArray);

	void flipCellStatus(unsigned long long int x, GpuData& gpu);

	cudaError_t calculateWithCuda(bool* statusArray, GpuData& gpu);

	cudaError_t getCellArray(bool* statusArray, GpuData& gpu);

	cudaError_t mallocMemory(GpuData& gpu);

	void setGameRules(GpuData& gpu, bool* lifeArray, bool* deathArray);

	void freeMemory(GpuData& gpu);
};

#endif
