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
	BciError mallocMemory(Board* board_p) override;

public:
	/*allocate memory, link board */
	GpuCompute(Board* board_p)
		: board(board_p)
	{
		mallocMemory(board_p);
	}

	/*set board*/
	void setBoard(Board* board_p) override;

	/*calculate next state of each cell from board and save in board*/
	BciError calculateBoxes() override;

	/*flip state of box with index of x*/
	BciError flipBox(uint64_t x) override;
};

#endif
