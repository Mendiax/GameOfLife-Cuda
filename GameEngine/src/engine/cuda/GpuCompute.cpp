#include <iostream>
#include <engine/cuda/GpuCompute.h>
#include <engine/BoardComputeInterface.h>
#include <engine/cuda/kernel.cuh>

GpuCompute::~GpuCompute()
{
	freeMemory();
}


BciError_E GpuCompute::flipCellStatus(uint32_t x, uint32_t y)
{
	uint64_t id = this->board->getCellId(x, y);
	gpu::flipCellStatus(id, gpuData);
	gpu::getCellArray(board->getBoardArray(), gpuData);
	return BciError_E::BCI_OK;
}

BciError_E GpuCompute::calculateBoxes()
{
	cudaError_t cudaStatus;
	cudaStatus = gpu::calculateWithCuda(board->getBoardArray(), gpuData);
	if (cudaStatus == cudaSuccess)
		return BciError_E::BCI_OK;
	else
		return BciError_E::BCI_ERROR;
}

BciError_E GpuCompute::mallocMemory()
{
	gpu::mallocMemory(gpuData);
	return BciError_E::BCI_OK;
}

void GpuCompute::freeMemory()
{
	gpu::freeMemory(gpuData);
	board = 0;
}

void GpuCompute::setBoard(Board* board_p)
{
	if(board)
		freeMemory();
	this->board = board_p;
	//set gpu data
	gpuData.cellsStatusLength = board->getSize();
	gpuData.cellsStatusRowLength = board->getWidth();
	
	std::cout 
		<< gpuData.dev_cellsStatusIn_p << ", "
		<< gpuData.dev_cellsStatusOut_p << ", " 
		<< gpuData.cellsStatusRowLength << ", "
		<< gpuData.cellsStatusLength << std::endl;
	mallocMemory();
	gpu::setGameRules(gpuData, board->getLifeArray(), board->getDeathArray());
}
