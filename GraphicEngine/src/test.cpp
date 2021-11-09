//#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <graphic/test.h>

#include <GLFW/glfw3.h>

void opengltest()
{
    GLFWwindow* window;

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

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);
        float x, y, x2, y2;
        x = y = -0.5;
        x2 = y2 = 0.5;
        glBegin(GL_TRIANGLES);
        glVertex3f(x, y, 0);
        glVertex3f(x, y2, 0);
        glVertex3f(x2, y2, 0);

        glVertex3f(x2, y2, 0);
        glVertex3f(x2, y, 0);
        glVertex3f(x, y, 0);
        glEnd();

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
}