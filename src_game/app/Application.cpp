#include <iostream>
#include <filesystem>
#include <windows.h>
#include <fstream>
#include <engine/cuda/GpuCompute.h>
#include <graphic/painter.h>
#include <chrono>

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
	cout << "Reading settings from file" << endl;
	uint32_t boardWidth, boardHeight;
	bool* lifeArray = (bool*)calloc(9, sizeof(bool));
	bool* deathArray = (bool*)calloc(9, sizeof(bool));

	ifstream file(GetExeFileName() + "\\inputData.txt");
	if (file.is_open()) {
		
		file >> boardWidth;
		file >> boardHeight;

		cout << "board width: " << boardWidth << endl;
		cout << "board height: " << boardHeight << endl;
		std::string lifeString;
		std::string deathString;
		file >> lifeString;
		file >> deathString;
		for (int i = 0; i < 9; i++) {
			lifeArray[i] = lifeString[i] == '1';
			deathArray[i] = deathString[i] == '1';
		}
		cout << "lifeArray: " << lifeString << endl;
		cout << "deathArray: " << deathString << endl;


		file.close();
	}
	else {
		cout << "Input file missing";
		return 1;
	}

	cout << "OK" << endl;

	cout << "Initializing memory" << endl;
	
	lifeArray[8] = 1;

	Board board(boardWidth, boardHeight, lifeArray, deathArray);
	cout << "OK" << endl;

	cout << "Initializing CUDA" << endl;
	GpuCompute gpu(&board);
	cout << "OK" << endl;

	Painter painter(board.getWidth(), board.getHeight());
	//start timer
	auto start = std::chrono::steady_clock::now();
	double time = 1.0;
	while (!painter.paint(board)) {

		bool isPressed;
		int cellX, cellY;
		painter.getPress(isPressed, cellX, cellY);

		// Test
		if (isPressed) {
			std::cout << "Cell (" << cellX << ", " << cellY << ") was pressed" << std::endl;
			gpu.flipCellStatus(cellX, cellY);
		}

		if (painter.isStarted())
		{
			auto end = std::chrono::steady_clock::now();
			std::chrono::duration<double> elapsed_seconds = end - start;
			if (elapsed_seconds.count() >= time)
			{
				auto startUpdate = std::chrono::high_resolution_clock::now();
				if (gpu.calculateBoxes() == BciError_E::BCI_ERROR)
					break;

				auto endUpdate = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> elapsed_seconds_Update = endUpdate - startUpdate;
				std::cout << "update takes: " << elapsed_seconds_Update.count() * 1000.0 << " millis\n";
				start = std::chrono::steady_clock::now();
			}
		}
	}
	free(lifeArray);
	free(deathArray);
	return 0;
}

