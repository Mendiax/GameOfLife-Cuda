#include <engine/cuda/kernel.cuh>
#include <engine/board.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <malloc.h>
#include <iostream>

__device__ uint16_t rowCount;
__device__ uint16_t sizeOfBoxArray;


__device__ void moveUp(int* id)
{
	if (*id + rowCount < sizeOfBoxArray)
		*id += rowCount;
	else
		*id %= rowCount;
}
__device__ void moveDown(int* id)
{
	if (*id - rowCount >= 0)
		*id -= rowCount;
	else
		*id -= rowCount + sizeOfBoxArray;
}
__device__ void moveLeft(int* id)
{
	if (0 == *id % rowCount)
		*id = *id - 1 + rowCount;
	else
		*id -= 1;
}
__device__ void moveRight(int* id)
{
	if (rowCount - 1 == *id % rowCount)
		*id = *id + 1 - rowCount;
	else
		*id += 1;
}

__global__ void gpu::calculateKernel(bool* boxesStatusIn, bool* boxesStatusOut, unsigned long long int row, unsigned long long int sizeOfArray, bool* lifeArray, bool* deathArray)
{
	rowCount = row;
	sizeOfBoxArray = sizeOfArray;
	__shared__ int boxId;
	__shared__ unsigned int sum;
	__shared__ bool state;
	__shared__ int id[8];

	if (threadIdx.x == 0)
	{
		boxId = blockIdx.x;
		state = boxesStatusIn[boxId];
		sum = 0;
	}

	__syncthreads();

	id[threadIdx.x] = boxId;



	__shared__  bool surroundState[8];
	switch (threadIdx.x)
	{
	case 0:
		moveLeft(&id[threadIdx.x]);
		moveUp(&id[threadIdx.x]);
		break;
	case 1:
		moveUp(&id[threadIdx.x]);
		break;
	case 2:
		moveRight(&id[threadIdx.x]);
		moveUp(&id[threadIdx.x]);
		break;
	case 3:
		moveLeft(&id[threadIdx.x]);
		break;
	case 4:
		moveRight(&id[threadIdx.x]);
		break;
	case 5:
		moveLeft(&id[threadIdx.x]);
		moveDown(&id[threadIdx.x]);
		break;
	case 6:
		moveDown(&id[threadIdx.x]);
		break;
	case 7:
		moveRight(&id[threadIdx.x]);
		moveDown(&id[threadIdx.x]);
		break;
	}
	if (id[threadIdx.x] >= sizeOfArray || id[threadIdx.x] < 0)
		return;

	surroundState[threadIdx.x] = boxesStatusIn[id[threadIdx.x]];

	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (bool state : surroundState)
		{
			sum += state;
		}
		if (state)
		{
			if (lifeArray[sum])
				boxesStatusOut[boxId] = true;
			else
				boxesStatusOut[boxId] = false;
		}
		else
		{
			if (deathArray[sum])
				boxesStatusOut[boxId] = true;
			else
				boxesStatusOut[boxId] = false;
		}
	}
}

void gpu::flipCellStatus(unsigned long long int x, GpuData& gpu)
{
	cudaMemcpy(gpu.cellsStatusBuffer_p, gpu.dev_cellsStatusIn_p, gpu.cellsStatusLength * sizeof(bool), cudaMemcpyDeviceToHost);
	gpu.cellsStatusBuffer_p[x] = !gpu.cellsStatusBuffer_p[x];
	cudaMemcpy(gpu.dev_cellsStatusIn_p, gpu.cellsStatusBuffer_p, gpu.cellsStatusLength * sizeof(bool), cudaMemcpyHostToDevice);
}

void gpu::setGameRules(GpuData& gpu, bool* lifeArray, bool* deathArray)
{
	cudaMemcpy(gpu.lifeArray, lifeArray, 9 * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu.deathArray, deathArray, 9 * sizeof(bool), cudaMemcpyHostToDevice);
}

void gpu::freeMemory(GpuData& gpu)
{
	fprintf(stderr, "cuda free memory!");
	cudaFree(gpu.dev_cellsStatusOut_p);
	cudaFree(gpu.dev_cellsStatusIn_p);
	cudaFree(gpu.lifeArray);
	cudaFree(gpu.deathArray);
	// nie działa idk czemu
	if (gpu.cellsStatusBuffer_p) {
		free(gpu.cellsStatusBuffer_p);
		gpu.cellsStatusBuffer_p = 0;
	}
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
}

cudaError_t gpu::mallocMemory(GpuData& gpu)
{
	gpu.cellsStatusBuffer_p = (bool*)malloc(gpu.cellsStatusLength * sizeof(bool));
	memset(gpu.cellsStatusBuffer_p, 0, gpu.cellsStatusLength);

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		freeMemory(gpu);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&gpu.dev_cellsStatusIn_p, gpu.cellsStatusLength * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freeMemory(gpu);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&gpu.lifeArray, 9 * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freeMemory(gpu);
		return cudaStatus;
	}

	bool* statusArray = (bool*)calloc(gpu.cellsStatusLength, sizeof(bool));
	cudaStatus = cudaMemcpy(gpu.lifeArray, statusArray, 9 * sizeof(bool), cudaMemcpyHostToDevice);
	free(statusArray);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeMemory(gpu);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&gpu.deathArray, 9 * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freeMemory(gpu);
		return cudaStatus;
	}


	cudaStatus = cudaMemcpy(gpu.deathArray, statusArray, 9 * sizeof(bool), cudaMemcpyHostToDevice);
	free(statusArray);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeMemory(gpu);
		return cudaStatus;
	}



	cudaStatus = cudaMalloc((void**)&gpu.dev_cellsStatusOut_p, gpu.cellsStatusLength * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freeMemory(gpu);
		return cudaStatus;
	}

	statusArray = (bool*)calloc(gpu.cellsStatusLength, sizeof(bool));
	cudaStatus = cudaMemcpy(gpu.dev_cellsStatusIn_p, statusArray, gpu.cellsStatusLength * sizeof(bool), cudaMemcpyHostToDevice);
	free(statusArray);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeMemory(gpu);
		return cudaStatus;
	}

	return cudaSuccess;
}

cudaError_t gpu::calculateWithCuda(bool* statusArray, GpuData& gpu)
{
	cudaError_t cudaStatus;
	/* for debug
	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	std::cout << "using " << properties.multiProcessorCount << " multiprocessors" << std::endl;
	std::cout << "max threads per processor: " << properties.maxThreadsPerMultiProcessor << std::endl;*/
	std::cout << gpu.dev_cellsStatusIn_p << ", " << gpu.dev_cellsStatusOut_p << ", " << gpu.cellsStatusRowLength << ", " << gpu.cellsStatusLength << ", " << gpu.lifeArray << ", " << gpu.deathArray << std::endl;//*/
	calculateKernel << <gpu.cellsStatusLength, 8 >> > (gpu.dev_cellsStatusIn_p, gpu.dev_cellsStatusOut_p, gpu.cellsStatusRowLength, gpu.cellsStatusLength, gpu.lifeArray, gpu.deathArray);


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		freeMemory(gpu);
		return cudaStatus;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		freeMemory(gpu);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(statusArray, gpu.dev_cellsStatusOut_p, gpu.cellsStatusLength * sizeof(bool), cudaMemcpyDeviceToHost);

	bool* buf = gpu.dev_cellsStatusIn_p;
	gpu.dev_cellsStatusIn_p = gpu.dev_cellsStatusOut_p;
	gpu.dev_cellsStatusOut_p = buf;

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeMemory(gpu);
		return cudaStatus;
	}

	return cudaStatus;
}

cudaError_t gpu::getCellArray(bool* statusArray, GpuData& gpu) {
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(statusArray, gpu.dev_cellsStatusIn_p, gpu.cellsStatusLength * sizeof(bool), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeMemory(gpu);
		return cudaStatus;
	}

	return cudaStatus;
}
