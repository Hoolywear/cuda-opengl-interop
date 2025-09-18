/* INCLUDES */

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>

#include "callbacks.h"
#include "gl_setup.h"
#include "cuda_kernels.cuh"


bool init(GLFWwindow **windowPtr, int win_w, int win_h) {
    // Set error callback
    glfwSetErrorCallback(error_callback);

    // Initialize GLFW
    if (!glfwInit())
        return false;
    
    // OpenGL hints for version support
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Create GLFW window
    *windowPtr = glfwCreateWindow(win_w, win_h, "Extended OpenGL Init", nullptr, nullptr);
    if (!*windowPtr) {
        glfwTerminate();
        return false;
    }

    // Set callbacks
    glfwSetKeyCallback(*windowPtr, key_callback);
    glfwSetWindowSizeCallback(*windowPtr, window_size_callback);
    
    // Set OpenGL context
    glfwMakeContextCurrent(*windowPtr);
    gladLoadGL();
    glfwSwapInterval(1); // vsync
    
    // Initialize GLAD (Copilot-generated try catch block)
    try {
        init_glad();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        glfwTerminate();
        return false;
    }
    
    // default initialization
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    return true;
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

}

int main()
{
    // initial window dimensions in pixels
    int win_w = 800, win_h = 600;
    // window and monitor
    GLFWwindow *window = nullptr;

    std::printf("Starting GLFW %s\n", glfwGetVersionString());

    // Initialize the entire program (GLFW, OpenGL context, GLAD)
    if (!init(&window, win_w, win_h)) {
        return -1;
    }
    
    std::printf("init() function run properly, executing main loop...\n");

    // create VBO
    // createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    // run the cuda part
    // runCuda(&cuda_vbo_resource);

    while (!glfwWindowShouldClose(window)) {
        // Run loop instructions
        display();
        // Window routines (swap buffers, retrieve events from e.g. key presses or mouse interactions)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    std::printf("Terminating program...\n");

    glfwTerminate();
    return 0;
}