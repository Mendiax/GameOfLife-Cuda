#include "input/input.h"
#include <iostream>

/**
* Nie mo�na zastosowa� metody klasy Painter jako argumentu glfwSetCursorPosCallback.
* glfwSetCursorPosCallback oczekuje zwyk�ej funkcji. Dlatego ta funkcja wysy�a do obiektu painter wsp�rz�dne myszki.
* Je�eli wiesz jak zrobi� z "cursorPositionCallback" metod� klasy Painter, w taki spos�b, �eby mo�na by�o wywo�a�
* glfwSetCursorPosCallback(window, cursorPositionCallback) to to poprawi�.
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