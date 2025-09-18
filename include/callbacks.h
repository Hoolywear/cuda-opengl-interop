#pragma once
#include <GLFW/glfw3.h>

// Key input handler
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

// Window size callback (resize handling)
void window_size_callback(GLFWwindow* window, int width, int height);

// Error callback for GLFW
void error_callback(int error, const char* description);
