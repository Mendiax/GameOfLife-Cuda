#include <engine/cuda/kernel.cuh>
#include <engine/board.h>


//#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <malloc.h>
#include <iostream>

__global__ void gpu::calculateKernel(bool* boxesStatusIn, bool* boxesStatusOut, unsigned long long int sizeOfArray)
{
	
}


void gpu::flipCellStatus(unsigned long long int x)
{
	cudaMemcpy(cellsStatusBuffer_p, dev_cellsStatusIn_p, cellsStatusLength * sizeof(bool), cudaMemcpyDeviceToHost);
	cellsStatusBuffer_p[x] = !cellsStatusBuffer_p[x];
	cudaMemcpy(dev_cellsStatusIn_p, cellsStatusBuffer_p, cellsStatusLength * sizeof(bool), cudaMemcpyHostToDevice);
}

void gpu::freeMemory()
{
	fprintf(stderr, "cuda free memory!");
	cudaFree(dev_cellsStatusOut_p);
	cudaFree(dev_cellsStatusIn_p);
	free(cellsStatusBuffer_p);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
}

cudaError_t gpu::mallocMemory(unsigned long long int size)
{
	
	cellsStatusBuffer_p = (bool*)calloc(size, sizeof(bool));
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		freeMemory();
		return cudaStatus;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_cellsStatusIn_p, size * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freeMemory();
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_cellsStatusOut_p, size * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freeMemory();
		return cudaStatus;
	}

	bool* statusArray = (bool*)calloc(size, sizeof(bool));
	cudaStatus = cudaMemcpy(dev_cellsStatusIn_p, statusArray, size * sizeof(bool), cudaMemcpyHostToDevice);
	free(statusArray);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeMemory();
		return cudaStatus;
	}

	return cudaSuccess;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t gpu::calculateWithCuda(bool* statusArray)
{
	cudaError_t cudaStatus;
	unsigned long long int size = cellsStatusLength;
	calculateKernel << <size, 8 >> > (dev_cellsStatusIn_p, dev_cellsStatusOut_p, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		freeMemory();
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		freeMemory();
		return cudaStatus;
	}
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(statusArray, dev_cellsStatusOut_p, size * sizeof(bool), cudaMemcpyDeviceToHost);

	bool* buf = dev_cellsStatusIn_p;
	dev_cellsStatusIn_p = dev_cellsStatusOut_p;
	dev_cellsStatusOut_p = buf;

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeMemory();
		return cudaStatus;
	}

	return cudaStatus;
}
