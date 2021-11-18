#include "../inc/graphic/painter.h"
#include <iostream>
#include <stdexcept>

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

Painter::Painter(int width, int height) : width(width), height(height) {
	createWindow();

	createBuffers();

	createShaders();

	createVertices();
}

Painter::~Painter() {
	delete[] vertices;

	glfwTerminate();
}

int Painter::paint(Board& board) {
	int shouldClose = glfwWindowShouldClose(window);
	
	if (!shouldClose)
	{
		for (int i = 0; i < board.getSize(); i++) {
			if (board.getBoardArray()[i]) {
				for (int k = 0; k < 6; k++) {
					vertices[6 * i * 3 * 2 + 3 + 6 * k] = aliveRed;
					vertices[6 * i * 3 * 2 + 4 + 6 * k] = aliveGreen;
					vertices[6 * i * 3 * 2 + 5 + 6 * k] = aliveBlue;
				}
				
			}
			else {
				for (int k = 0; k < 6; k++) {
					vertices[6 * i * 3 * 2 + 3 + 6 * k] = defaultRed;
					vertices[6 * i * 3 * 2 + 4 + 6 * k] = defaultGreen;
					vertices[6 * i * 3 * 2 + 5 + 6 * k] = defaultBlue;
				}
			}
		}

		glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * verticiesLength, vertices, GL_DYNAMIC_DRAW);


		glClearColor(backgroundRed, backgroundGreen, backgroundBlue, backgroundAlpha);
		glClear(GL_COLOR_BUFFER_BIT);


		glUseProgram(shaderProgram);
		glBindVertexArray(vertexArrayObject);
		glDrawArrays(GL_TRIANGLES, 0, verticiesLength / 6);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return shouldClose;
}

void Painter::createWindow()
{
	if (!glfwInit()) {
		throw std::exception("Failed to init GLFW.");
	}

	window = glfwCreateWindow(screen_width, screen_height, "Game of life", NULL, NULL);

	if (!window)
	{
		throw std::exception("Failed to create GLFW window.");
		glfwTerminate();

	}

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw std::exception("Failed to initialize GLAD.");
	}

	glViewport(0, 0, screen_width, screen_height);

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
}

void Painter::createVertices()
{
	verticiesLength = width * height * 9 * 2 * 2; 
	vertices = new float[verticiesLength];

	const float screenWidth  = 2.0f;
	const float screenHeight = 2.0f;

	const float cellWidth  = screenWidth / width;
	const float cellHeight = screenHeight / height;

	const float margin = 0.01f;

	const float x0 = -screenWidth / 2.0f;
	const float y0 = -screenHeight / 2.0f;

	int index = 0;

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {

			vertices[index + 0] = x0 + i * cellWidth  + margin;
			vertices[index + 1] = y0 + j * cellHeight + margin;
			vertices[index + 2] = 0;
			
			vertices[index + 3] = defaultRed;
			vertices[index + 4] = defaultGreen;
			vertices[index + 5] = defaultBlue;

			vertices[index + 6] = x0 + (i + 1) * cellWidth - margin;
			vertices[index + 7] = y0 + j * cellHeight + margin;
			vertices[index + 8] = 0;

			vertices[index + 9] = defaultRed;
			vertices[index + 10] = defaultGreen;
			vertices[index + 11] = defaultBlue;

			vertices[index + 12] = x0 + (i + 1) * cellWidth - margin;
			vertices[index + 13] = y0 + (j + 1) * cellHeight - margin;
			vertices[index + 14] = 0;

			vertices[index + 15] = defaultRed;
			vertices[index + 16] = defaultGreen;
			vertices[index + 17] = defaultBlue;

			vertices[index + 18] = x0 + i * cellWidth + margin;
			vertices[index + 19] = y0 + j * cellHeight + margin;
			vertices[index + 20] = 0;

			vertices[index + 21] = defaultRed;
			vertices[index + 22] = defaultGreen;
			vertices[index + 23] = defaultBlue;

			vertices[index + 24] = x0 + i * cellWidth + margin;
			vertices[index + 25] = y0 + (j + 1) * cellHeight - margin;
			vertices[index + 26] = 0;

			vertices[index + 27] = defaultRed;
			vertices[index + 28] = defaultGreen;
			vertices[index + 29] = defaultBlue;

			vertices[index + 30] = x0 + (i + 1) * cellWidth - margin;
			vertices[index + 31] = y0 + (j + 1) * cellHeight - margin;
			vertices[index + 32] = 0;

			vertices[index + 33] = defaultRed;
			vertices[index + 34] = defaultGreen;
			vertices[index + 35] = defaultBlue;

			index += 36;
		}
	}
}
void Painter::createBuffers()
{
	glGenBuffers(1, &vertexBufferObject);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);

	glGenVertexArrays(1, &vertexArrayObject);
	glGenBuffers(1, &vertexBufferObject);
	glBindVertexArray(vertexArrayObject);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void Painter::createShaders()
{
	const char* vertexShaderSource = "#version 330 core\n"
		"layout (location = 0) in vec3 aPos;\n"
		"layout (location = 1) in vec3 aColor;\n"
		"out vec3 ourColor;\n"
		"void main()\n"
		"{\n"
		"   gl_Position = vec4(aPos, 1.0);\n"
		"   ourColor = aColor;\n"
		"}\0";

	const char* fragmentShaderSource = "#version 330 core\n"
		"out vec4 FragColor;\n"
		"in vec3 ourColor;\n"
		"void main()\n"
		"{\n"
		"   FragColor = vec4(ourColor, 1.0f);\n"
		"}\n\0";

	unsigned int vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);

	int  success;
	char infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	unsigned int fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);

	shaderProgram = glCreateProgram();

	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glUseProgram(shaderProgram);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}