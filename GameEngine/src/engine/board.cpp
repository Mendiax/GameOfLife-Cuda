#include <engine/board.h>
#include <iostream>
#include <cassert>

Board::Board(uint32_t w, uint32_t h, bool* lifeArray, bool* deathArray)
{
	this->rowCount = w;
	this->collumCount = h;
	this->cellsArraySize = (uint64_t)w * (uint64_t)h;
	this->cellsArray_p = (bool*)calloc(this->cellsArraySize, sizeof(bool));
	this->lifeArray = lifeArray;
    this->deathArray = deathArray;
}

Board::~Board() {
	free(this->cellsArray_p);
}

bool* Board::getBoardArray() {
	return this->cellsArray_p;
}

uint64_t Board::getSize() {
	return this->cellsArraySize;
}
bool* Board::getLifeArray() {
	return this->lifeArray;
}

bool* Board::getDeathArray() {
	return this->deathArray;
}

uint32_t Board::getWidth() {
	return this->rowCount;
}

uint32_t Board::getHeight() {
	return this->collumCount;
}

uint64_t Board::getCellId(uint32_t i, uint32_t j) {
	assert(i < rowCount);
	assert(j < collumCount);
	return  static_cast<uint64_t>(j) * getWidth() + i;
}

bool Board::getCell(uint32_t i, uint32_t j) {
	return  this->cellsArray_p[getCellId(i,j)];
}

void Board::getCordsFromId(uint64_t i, uint32_t* x, uint32_t* y) {
	assert(i < getSize());
	*x = 0;
	*y = 0;
	while (i > getWidth())
	{
		i -= getWidth();
		*y += 1;
	}
	*x = i;
}

void Board::print() {
	std::cout << "Printig board array" << std::endl;
	for (uint64_t j = 0; j < getWidth(); j++) {
		for (uint64_t i = 0; i < getHeight(); i++) {
			std::cout << getCell(i,j);
		}
		std::cout << std::endl;
	}
}