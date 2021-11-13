#include <iostream>
#include <engine/cuda/GpuCompute.h>
#include <engine/BoardComputeInterface.h>

BciError GpuCompute::flipBox(uint64_t x) noexcept(true)
{
	
}

BciError GpuCompute::calculateBoxes() noexcept(true)
{
	return BciError::OK;
}

BciError GpuCompute::mallocMemory(Board* board_p) noexcept(true)
{
	return BciError::OK;
}

void GpuCompute::freeMemory() noexcept(true)
{

}

void GpuCompute::setBoard(Board* board_p) noexcept(true)
{
	freeMemory();
	this->board = board_p;
	mallocMemory(board_p);
}
