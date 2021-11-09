#include <iostream>
#include <iostream>

#include <engine/board.h>
#include <engine/cuda/kernel.cuh>
#include <graphic/test.h>

using namespace std;

int main(void)
{
	cout << "Game of Life" << endl;
	opengltest();
	mainCuda();//test
	cout << "Reading settings from file" << endl;
	uint32_t boardWidth, boardHeight;

	//change it later
	boardWidth = 10;
	boardHeight = 10;

	cout << "OK" << endl;

	cout << "Initializing memory" << endl;
	Board board(boardWidth, boardHeight);
	cout << "OK" << endl;

	cout << "Initializing CUDA" << endl;

	cout << "OK" << endl;

	board.print();
	return 0;
}