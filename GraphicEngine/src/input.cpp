#include "input/input.h"
#include <iostream>

/**
* Nie mo¿na zastosowaæ metody klasy Painter jako argumentu glfwSetCursorPosCallback.
* glfwSetCursorPosCallback oczekuje zwyk³ej funkcji. Dlatego ta funkcja wysy³a do obiektu painter wspó³rzêdne myszki.
* Je¿eli wiesz jak zrobiæ z "cursorPositionCallback" metodê klasy Painter, w taki sposób, ¿eby mo¿na by³o wywo³aæ
* glfwSetCursorPosCallback(window, cursorPositionCallback) to to poprawiê.
*/
void cursorPositionCallback(GLFWwindow* window, double xPos, double yPos) {
	Painter* painter = (Painter*)(glfwGetWindowUserPointer(window));

	painter->setMouseX(xPos);
	painter->setMouseY(yPos);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		Painter* painter = (Painter*)(glfwGetWindowUserPointer(window));
		painter->press();
	}
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	Painter* painter = (Painter*)(glfwGetWindowUserPointer(window));
	painter->zoom(yoffset);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action != GLFW_PRESS && key != GLFW_KEY_LEFT && key != GLFW_KEY_RIGHT && key != GLFW_KEY_UP && key != GLFW_KEY_DOWN)
	{
		return;
	}

	Painter* painter = (Painter*)(glfwGetWindowUserPointer(window));
	painter->setDirection(key);
}