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
	//EXPECT_TRUE(false);

}

template <typename t>
bool equal(t* arr1, t* arr2, uint16_t n) {
	for (uint16_t i = 0; i < n; i++) {
		if (arr1[i] != arr2[i]) return false;
	}
	return true;
}

TEST(CUDA, simple_cross_test)
{
	Board* board = new Board(5, 5);
	GpuCompute* gpu = new GpuCompute(board);

	gpu->flipCellStatus(1, 2);
	gpu->flipCellStatus(2, 2);
	gpu->flipCellStatus(3, 2);
	EXPECT_TRUE(board->getCell(1, 2));
	EXPECT_TRUE(board->getCell(2, 2));
	EXPECT_TRUE(board->getCell(3, 2));

	Board* expec = new Board(5, 5);
	std::cout << "before:\n";

	board->print();
	gpu->calculateBoxes();
	std::cout << "after:\n";
	board->print();
	
	expec->getBoardArray()[expec->getCellId(2,1)] = 1;
	expec->getBoardArray()[expec->getCellId(2,2)] = 1;
	expec->getBoardArray()[expec->getCellId(2,3)] = 1;
	std::cout << "expect:\n";
	expec->print();

	EXPECT_TRUE(equal(board->getBoardArray(), expec->getBoardArray(), board->getSize()));

	free(expec);
	delete gpu;
	delete board;
}