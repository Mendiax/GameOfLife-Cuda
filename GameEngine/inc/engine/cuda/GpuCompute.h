#ifndef GPUCOMPUTE_H
#define GPUCOMPUTE_H

#include <engine/BoardComputeInterface.h>

class GpuCompute :
	public BoardComputeInterface
{
private:
	Board* board;

	/*free memory if not needed*/
	void freeMemory() override;

	/*allocate memory and set size*/
	BciError mallocMemory() override;

public:
	/*allocate memory, link board */
	GpuCompute(Board* board_p)
		: board(board_p)
	{
		mallocMemory();
	}

	~GpuCompute();

	/*set board*/
	void setBoard(Board* board_p) override;

	/*calculate next state of each cell from board and save in board*/
	BciError calculateBoxes() override;

	/*flip state of box with index of x, y*/
	BciError flipCellStatus(uint32_t x, uint32_t y) override;
};

#endif
