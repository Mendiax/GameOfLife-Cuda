#include <GLFW/glfw3.h>
#include <graphic/test.h>
#include <GLFW/glfw3.h>

#include "../inc/graphic/painter.h"
#include <iostream>

Painter::Painter() {
	/* Initialize the library */
	if (!glfwInit())
		return;

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);
}

void Painter::paint(Board board) {
	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(window))
	{
		/* Render here */
		glClear(GL_COLOR_BUFFER_BIT);
		float x0, y0, windowWidth, windowHeight, tileWidth, tileHeight, margin;

		windowWidth = windowHeight = 2;

		x0 = -windowWidth / 2;
		y0 = -windowHeight / 2;

		margin = 0.01;

		tileWidth = windowWidth / board.getWidth();
		tileHeight = windowHeight / board.getHeight();

		glBegin(GL_TRIANGLES);

		board.getBoardArray()[10] = 0;

		for (int i = 0; i < board.getWidth(); i++) {
			for (int j = 0; j < board.getHeight(); j++) {

				float red = 0.5f;
				float green = 0.5f;
				float blue = board.getBoardArray()[i * board.getWidth() + j] ? 1.0f : 0.0f;

				paintSquare(x0 + i * tileWidth + margin, y0 + j * tileWidth + margin, x0 + (i + 1) * tileWidth - margin, y0 + (j + 1) * tileWidth - margin, red, green, blue);
			}
		}

		glEnd();

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();
	}

	glfwTerminate();
}

void Painter::paintSquare(float x, float y, float x2, float y2, float red, float green, float blue) {
	glColor3f(red, green, blue);

	glVertex3f(x, y, 0);
	glVertex3f(x, y2, 0);
	glVertex3f(x2, y2, 0);

	glVertex3f(x2, y2, 0);
	glVertex3f(x2, y, 0);
	glVertex3f(x, y, 0);
}