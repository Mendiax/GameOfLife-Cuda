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
	/**
	* Constructor arguments are width and height of board
	*/
	Painter(int width, int height);
	~Painter();

	/**
	* Use this method to paint a board
	*/
	int paint(Board& board);

	void setMouseX(float x);
	void setMouseY(float y);

	void setDirection(int direction);

	void press();
	void getPress(bool& isPressed, int& cellX, int& cellY);

	void zoom(int zoomValue);

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

	int mouseX = screen_width / 2.0;
	int mouseY = screen_height / 2.0;

	float* vertices = 0;
	int verticiesLength;

	bool isPressed = false;

	double zoomValue = 1.0;

	int zoomLocation;
	int moveLocation;

	float move = 0.015625; // 1.0 / 2.0^6
	float dx = 0;
	float dy = 0;
};

#endif