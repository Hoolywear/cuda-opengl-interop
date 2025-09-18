#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>

#include "callbacks.h"
#include "gl_setup.h"
#include "cuda_kernels.cuh"

int main()
{
    bool fullscreen = false;
    GLFWwindow* window = nullptr;
    GLFWmonitor* monitor = nullptr;
    int win_w = 800, win_h = 600; // window dimensions in pixels

    std::printf("Starting GLFW %s\n", glfwGetVersionString());

    // Set error callback
    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    if (fullscreen) {
        monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
        win_w = mode->width;
        win_h = mode->height;
    }

    window = glfwCreateWindow(win_w, win_h, "Extended OpenGL Init", monitor, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwSetKeyCallback(window, key_callback);
    glfwSetWindowSizeCallback(window, window_size_callback);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    try {
        init_glad();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        glfwTerminate();
        return -1;
    }

    // Test CUDA dummy kernel
    int cuda_status = launch_dummy_kernel();
    if (cuda_status != 0) {
        std::fprintf(stderr, "CUDA dummy kernel failed (code %d)\n", cuda_status);
    } else {
        std::printf("CUDA dummy kernel executed successfully.\n");
    }

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}
