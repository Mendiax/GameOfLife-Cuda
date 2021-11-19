#include <iostream>
#include <filesystem>
#include <windows.h>
#include <fstream>
#include <engine/board.h>
#include <engine/cuda/kernel.cuh>
#include <graphic/test.h>
#include <graphic/painter.h>

using namespace std;

std::string GetExeFileName()
{
	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	std::string f = std::string(buffer);
	return f.substr(0, f.find_last_of("\\/"));
}

int main(void)
{
	cout << "Game of Life" << endl;
	//opengltest();
	//mainCuda();
	cout << "Reading settings from file" << endl;
	uint32_t boardWidth, boardHeight;

	ifstream file(GetExeFileName() + "\\inputData.txt");
	if (file.is_open()) {
		std::string line;
		file >> boardWidth;
		file >> boardHeight;

		cout << "board width: " << boardWidth << endl;
		cout << "board height: " << boardHeight << endl;

		file.close();
	}
	else {
		cout << "Input file missing";
		return 1;
	}

	cout << "OK" << endl;

	cout << "Initializing memory" << endl;
	Board board(boardWidth, boardHeight);
	cout << "OK" << endl;

	cout << "Initializing CUDA" << endl;

	cout << "OK" << endl;

	Painter painter;

	painter.paint(board);

	return 0;
}

