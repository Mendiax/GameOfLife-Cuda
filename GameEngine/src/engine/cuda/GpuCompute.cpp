#include <iostream>
#include <engine/cuda/GpuCompute.h>
#include <engine/BoardComputeInterface.h>
#include <engine/cuda/kernel.cuh>

GpuCompute::~GpuCompute()
{
	freeMemory();
}


BciError GpuCompute::flipCellStatus(uint32_t x, uint32_t y)
{
	uint64_t id = this->board->getCellId(x, y);
	board->getBoardArray()[id] = !board->getBoardArray()[id];
	gpu::flipCellStatus(id);
	return BciError::OK;
}

BciError GpuCompute::calculateBoxes()
{
	gpu::calculateWithCuda(board->getBoardArray());
	return BciError::OK;
}

BciError GpuCompute::mallocMemory()
{
	gpu::mallocMemory(board->getSize());
	return BciError::OK;
}

void GpuCompute::freeMemory()
{
	gpu::freeMemory();
	board = nullptr;
}

void GpuCompute::setBoard(Board* board_p)
{
	if(board != nullptr)
		freeMemory();
	this->board = board_p;
	mallocMemory();
}
