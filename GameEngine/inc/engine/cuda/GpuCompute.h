#ifndef GPUCOMPUTE_H
#define GPUCOMPUTE_H

#include <engine/BoardComputeInterface.h>
#include <engine/cuda/kernel.cuh>

class GpuCompute :
	public BoardComputeInterface
{
private:
	Board* board = 0;
	GpuData gpuData;

	/*free memory if not needed*/
	void freeMemory() override;

	/*allocate memory and set size*/
	BciError_E mallocMemory() override;

public:
	/*allocate memory, link board */
	GpuCompute(Board* board_p)
	{
		setBoard(board_p);
	}

	~GpuCompute();

	/*set board*/
	void setBoard(Board* board_p) override;

	/*calculate next state of each cell from board and save in board*/
	BciError_E calculateBoxes() override;

	/*flip state of box with index of x, y*/
	BciError_E flipCellStatus(uint32_t x, uint32_t y) override;
};

#endif
