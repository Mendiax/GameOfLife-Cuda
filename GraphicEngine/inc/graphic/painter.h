#ifndef PAINTER_H
#define PAINTER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "../../GameEngine/inc/engine/board.h"

/**
* Painter is used to create window and paint a board in openGL
*/
class Painter
{
public:
	Painter(int width, int height);
	~Painter();

	// methods used by the user
	int paint(Board& board);
	void getPress(bool& isPressed, int& cellX, int& cellY);
	bool isStarted();

	// methods used by GLFW library
	void setMouseX(float x);
	void setMouseY(float y);
	void setDirection(int direction);
	void press();
	void zoom(int zoomValue);
	void startStop();

private:
	GLFWwindow* window;

	void createWindow();
	void createShaders();
	void createVertices();
	void createBuffers();
	void createCallbacks();

	const int width;
	const int height;
	const int screen_width = 800;
	const int screen_height = 600;

	unsigned int shaderProgram;

	unsigned int vertexBufferObject;
	unsigned int vertexArrayObject;

	const float defaultRed      = 0.47f;
	const float defaultGreen    = 0.43f;
	const float defaultBlue     = 0.00f;

	const float aliveRed        = 0.90f;
	const float aliveGreen      = 0.00f;
	const float aliveBlue       = 0.00f;

	const float backgroundRed   = 1.00f;
	const float backgroundGreen = 0.98f;
	const float backgroundBlue  = 0.00f;

	const float backgroundAlpha = 1.0f;

	float margin = 0.0f;
	float* vertices = 0;

	double zoomValue = 1.0;
	
	float move = 0.015625; // 1.0 / 2.0^6
	float dx = 0;
	float dy = 0;

	int mouseX = screen_width / 2;
	int mouseY = screen_height / 2;
	int verticiesLength = 0;
	int zoomLocation = 0;
	int moveLocation = 0;

	bool isPressed = false;
	bool start = true;
};

#endif