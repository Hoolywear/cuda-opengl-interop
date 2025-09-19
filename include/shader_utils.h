#pragma once
#include <string>
#include <glad/glad.h>

// Compile a single shader. Returns 0 on failure (logs to stderr).
GLuint compileShader(GLenum type, const char* src);
// Link a program from already compiled shaders (owned by caller until link). 0 on failure.
GLuint linkProgram(GLuint vs, GLuint fs);
// Convenience: compile vertex+fragment sources and link. Returns 0 on failure.
GLuint createProgramFromSource(const char* vsSrc, const char* fsSrc);
// Compile and link program from vertex+fragment shader file paths.
GLuint createProgramFromFiles(const char* vertexPath, const char* fragmentPath);
// Load file into string (returns empty string on error)
std::string loadTextFile(const char* path);
