#include "pch.h"
#include <engine/cuda/GpuCompute.h>
#include <engine/BoardComputeInterface.h>
#include <iostream>
#include <engine/cuda/kernel.cuh>


namespace CUDA {
	void testMove(Board* board, uint32_t x, uint32_t y, int moveX, int moveY) {
		unsigned long long int row = board->getWidth();
		unsigned long long int size = board->getSize();
		unsigned long long int id, newId;
		//uint32_t x2, y2;
		id = board->getCellId(x, y);
		newId = id;
		if (moveY > 0)
			gpu::moveUp(&newId, row, size);
		if (moveY < 0)
			gpu::moveDown(&newId, row, size);
		if (moveX > 0)
			gpu::moveRight(&newId, row, size);
		if (moveX < 0)
			gpu::moveLeft(&newId, row, size);
		if (newId != board->getCellId((x + moveX) % board->getWidth(), (y + moveY) % board->getHeight())) {
			std::cout << x << " " << y << " " << moveX << " " << moveY << std::endl;
			std::cout << newId << "=?" << board->getCellId((x + moveX) % board->getWidth(), (y + moveY) % board->getHeight()) << std::endl;
		}
		EXPECT_TRUE(newId == board->getCellId((x + board->getWidth() + moveX) % board->getWidth(), (y + board->getHeight() + moveY) % board->getHeight()));

	}

	TEST(Functions, move_up)
	{

		Board* board = new Board(5, 5);
		for (uint32_t x = 0; x < board->getWidth(); x++) {
			for (uint32_t y = 0; y < board->getHeight(); y++) {
				testMove(board, x, y, 0, 1);
			}
		}
		delete board;
	}

	TEST(Functions, move_down)
	{

		Board* board = new Board(5, 5);
		for (uint32_t x = 0; x < board->getWidth(); x++) {
			for (uint32_t y = 0; y < board->getHeight(); y++) {
				testMove(board, x, y, 0, -1);
			}
		}
		delete board;
	}

	TEST(Functions, move_left)
	{

		Board* board = new Board(5, 5);
		for (uint32_t x = 0; x < board->getWidth(); x++) {
			for (uint32_t y = 0; y < board->getHeight(); y++) {
				testMove(board, x, y, -1, 0);
			}
		}
		delete board;
	}

	TEST(Functions, move_right)
	{

		Board* board = new Board(5, 5);
		for (uint32_t x = 0; x < board->getWidth(); x++) {
			for (uint32_t y = 0; y < board->getHeight(); y++) {
				testMove(board, x, y, 1, 0);
			}
		}
		delete board;
	}

	TEST(Functions, move_all)
	{

		Board* board = new Board(5, 5);
		for (uint32_t x = 0; x < board->getWidth(); x++) {
			for (uint32_t y = 0; y < board->getHeight(); y++) {
				for (int dx = -1; dx <= 1; dx++) {
					for (int dy = -1; dy <= 1; dy++) {
						testMove(board, x, y, dx, dy);
					}
				}
			}
		}
		delete board;
	}

	TEST(Compute, Flip) {
		Board testBoard(5, 5);
		std::cout << "board init" << std::endl;
		testBoard.print();
		GpuCompute compute(&testBoard);
		compute.flipCellStatus(2, 2);
		std::cout << "board test" << std::endl;
		testBoard.print();
		EXPECT_TRUE(testBoard.getCell(2, 2));
	}

	template <typename t>
	bool equal(t* arr1, t* arr2, uint16_t n) {
		for (uint16_t i = 0; i < n; i++) {
			if (arr1[i] != arr2[i]) return false;
		}
		return true;
	}

	TEST(Compute, simple_cross_test)
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

		expec->getBoardArray()[expec->getCellId(2, 1)] = 1;
		expec->getBoardArray()[expec->getCellId(2, 2)] = 1;
		expec->getBoardArray()[expec->getCellId(2, 3)] = 1;
		std::cout << "expect:\n";
		expec->print();

		EXPECT_TRUE(equal(board->getBoardArray(), expec->getBoardArray(), board->getSize()));

		free(expec);
		delete gpu;
		delete board;
	}
}
