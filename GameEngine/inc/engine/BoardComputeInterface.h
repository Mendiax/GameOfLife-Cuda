#ifndef BOARDCOPUTEINTERFACE_H
#define BOARDCOPUTEINTERFACE_H

#include <engine/board.h>
/**
* Enum for error checking for BoardComputeInterface
*/

enum class BciError_E {
	BCI_OK,
	BCI_ERROR
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
	virtual BciError_E mallocMemory() = 0;

	BoardComputeInterface(){
		board = nullptr;
	}

public:

	/*allocate memory, link board */
	BoardComputeInterface(Board* board_p);


	/*set board*/
	virtual void setBoard(Board* board_p) = 0;

	/*calculate next state of each cell from board and save in board*/
	virtual BciError_E calculateBoxes() = 0;

	/*flip state of box with index of x, y*/
	virtual BciError_E flipCellStatus(uint32_t x, uint32_t y) = 0;

};

#endif



