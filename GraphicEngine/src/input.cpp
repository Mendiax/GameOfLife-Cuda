#include "input/input.h"
#include <iostream>

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

	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
		Painter* painter = (Painter*)(glfwGetWindowUserPointer(window));
		painter->startStop();
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