#include "pch.h"
#include <engine/cuda/GpuCompute.h>
#include <engine/BoardComputeInterface.h>
#include <iostream>

TEST(CUDA, Flip) {
	Board testBoard(5,5);
	std::cout << "board init" << std::endl;
	testBoard.print();
	GpuCompute compute(&testBoard);
	compute.flipCellStatus(2, 2);
	std::cout << "board test" << std::endl;
	testBoard.print();
	EXPECT_TRUE(testBoard.getCell(2, 2));
}