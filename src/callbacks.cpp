#include "callbacks.h"
#include <cstdio>

void error_callback(int /*error*/, const char* description) {
    std::fprintf(stderr, "GLFW Error: %s\n", description);
}

void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void window_size_callback(GLFWwindow* window, int width, int height) {
    // For now, just update the framebuffer size; you might later adjust viewport, etc.
    glfwSetWindowSize(window, width, height);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}