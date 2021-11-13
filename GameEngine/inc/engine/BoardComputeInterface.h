#ifndef BOARDCOPUTEINTERFACE_H
#define BOARDCOPUTEINTERFACE_H

#include "Board.h"
/**
* Enum for error checking for BoardComputeInterface
*/
enum class BciError {
	OK,
	ERROR
};

/**
* Class for computing end changing Board class (@see class Board). 
*/
class BoardComputeInterface
{
protected:
	Board* board;

	/*free memory that is not needed*/
	virtual void freeMemory() = 0;

	/*allocate memory and set size*/
	virtual BciError mallocMemory() = 0;

	BoardComputeInterface(){
		board = nullptr;
	}

public:

	/*allocate memory, link board */
	BoardComputeInterface(Board* board_p);


	/*set board*/
	virtual void setBoard(Board* board_p) = 0;

	/*calculate next state of each cell from board and save in board*/
	virtual BciError calculateBoxes() = 0;

	/*flip state of box with index of x, y*/
	virtual BciError flipCellStatus(uint32_t x, uint32_t y) = 0;

};

#endif



