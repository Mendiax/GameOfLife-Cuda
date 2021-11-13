#ifndef BOARD_H
#define BOARD_H

#include <cstdint>
/**
* Used to store state of all cells. It is read only, so modifing board should be only made by BoardCoputeInterface!
*/
class Board
{
	uint32_t rowCount = 0;
	uint32_t collumCount = 0;

	uint64_t cellsArraySize;
	bool* cellsArray_p;

public:
	/**
	* main constructor
	* @param w - width of a board, h - height of a board
	*/
	Board(uint32_t w, uint32_t h);

	/**
	* main deconstructor
	*/
	~Board();

	/**
	* @return pointer to array with state of cells
	*/
	bool* getBoardArray();

	/**
	* @return size of board
	*/
	uint64_t getSize();

	/**
	* @return number of collums
	*/
	uint32_t getWidth();

	/**
	* @return number of rows
	*/
	uint32_t getHeight();

	/**
	* @return index of cellsArray[i][j]
	*/
	uint64_t getCellId(uint32_t i, uint32_t j);

	/**
	* @return cellsArray[i][j]
	*/
	bool getCell(uint32_t i, uint32_t j);

	/**
	* prints board
	*/
	void print();
};

#endif