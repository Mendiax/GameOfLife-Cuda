#include <engine/board.h>
#include <iostream>

Board::Board(uint32_t w, uint32_t h)
{
	this->rowCount = w;
	this->collumCount = h;
	this->cellsArraySize = (uint64_t)w * (uint64_t)h;
	this->cellsArray_p = new bool[this->cellsArraySize];
}

Board::~Board() {
	delete[] this->cellsArray_p;
}

bool* Board::getBoardArray() {
	return this->cellsArray_p;
}

uint64_t Board::getSize() {
	return this->cellsArraySize;
}

uint32_t Board::getWidth() {
	return this->rowCount;
}

uint32_t Board::getHeight() {
	return this->collumCount;
}

uint32_t Board::getCell(uint32_t i, uint32_t j) {
	return  this->cellsArray_p[j * getWidth() + i];
}

void Board::print() {
	std::cout << "Printig board array" << std::endl;
	for (uint64_t i = 0; i < getHeight(); i++) {
		for (uint64_t j = 0; j < getWidth(); j++) {
			std::cout << cellsArray_p[j];
		}
		std::cout << std::endl;
	}
}