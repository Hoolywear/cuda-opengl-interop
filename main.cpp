#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error: %s\n", description);
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

static void win_size_callback(GLFWwindow *window, int width, int height) {
    glfwSetWindowSize(window,width,height);
}

int main(void)
{
    bool fullscreen = false;
    GLFWwindow *window;
    GLFWmonitor *monitor;
    int win_w = 800, win_h = 600; // window dimensions in pixels

    fprintf(stdout, "Starting GLFW %s\n", glfwGetVersionString());
    
    /* Set error callback function */
    glfwSetErrorCallback(error_callback);

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 1 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

    
    if (fullscreen) {
        // initialize fullscreen display
        monitor = glfwGetPrimaryMonitor();

        const GLFWvidmode *mode = glfwGetVideoMode( monitor );

        // Hinting these properties lets us use "borderless full screen" mode.
        glfwWindowHint( GLFW_RED_BITS, mode->redBits );
        glfwWindowHint( GLFW_GREEN_BITS, mode->greenBits );
        glfwWindowHint( GLFW_BLUE_BITS, mode->blueBits );
        glfwWindowHint( GLFW_REFRESH_RATE, mode->refreshRate );

        // use 'desktop' resolution for window size to get a full screen borderless window
        win_w = mode->width;
        win_h = mode->height;
    }
    

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(win_w, win_h, "Extended OpenGL Init", monitor, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    /* Set callback functions */
    glfwSetKeyCallback(window, key_callback);
    glfwSetWindowSizeCallback(window, win_size_callback);

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1); // Enable vsync
    
    
    /* Initialize GLAD */
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        glfwTerminate();
        return -1;
    }

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Poll for and process events */
        glfwPollEvents();

        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

    }

    glfwTerminate();
    return 0;
}
