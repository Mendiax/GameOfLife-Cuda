#ifndef PAINTER_H
#define PAINTER_H

#include <GLFW/glfw3.h>
#include <graphic/test.h>
#include <GLFW/glfw3.h>

#include "../../GameEngine/inc/engine/board.h"

class Painter
{
public:
	Painter();
	void paint(Board board);

private:
	GLFWwindow* window;
	void paintSquare(float x, float y, float x2, float y2, float red, float green, float blue);
};

#endif