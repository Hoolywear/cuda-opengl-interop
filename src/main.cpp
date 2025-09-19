/* INCLUDES */

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>

#include "callbacks.h"
#include "gl_setup.h"
#include "cuda_kernels.cuh"
#include "cuda_utils.cuh"

/* Modern VBO/VAO structure (no fixed-function pipeline) */
struct buffer_t {
    GLuint vbo = 0;
    GLuint vao = 0; // holds attribute state
    struct cudaGraphicsResource* cuda_resource = nullptr;
    float3* dptr = nullptr; // mapped device pointer (generic)
};

// Shader utilities
#include "shader_utils.h"

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
    glfwSwapInterval(1); // vsync

    // Initialize GLAD (gladLoadGLLoader wrapper) once context current.
    try { init_glad(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        glfwTerminate();
        return false;
    }
    
    // default initialization
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    return true;
}

void createBuf(buffer_t &buf, unsigned int vbo_res_flags, unsigned int mesh_width, unsigned int mesh_height) {
    // We will store 16 bytes per vertex: 3 floats position (12 bytes) + 4 unsigned bytes color (RGBA) (4 bytes)
    const unsigned int stride = 12;
    const size_t sizeBytes = static_cast<size_t>(mesh_width) * mesh_height * stride;

    glGenVertexArrays(1, &buf.vao);
    glBindVertexArray(buf.vao);

    glGenBuffers(1, &buf.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, buf.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeBytes, nullptr, GL_DYNAMIC_DRAW);

    // Position attribute (location 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    // // Color attribute (location 1) normalized
    // glEnableVertexAttribArray(1);
    // glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, stride, (void*)12);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&buf.cuda_resource, buf.vbo, vbo_res_flags));
}

void deleteBuf(buffer_t &buf) {
    if (buf.cuda_resource) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(buf.cuda_resource));
        buf.cuda_resource = nullptr;
    }
    if (buf.vbo) {
        glDeleteBuffers(1, &buf.vbo);
        buf.vbo = 0;
    }
    if (buf.vao) {
        glDeleteVertexArrays(1, &buf.vao);
        buf.vao = 0;
    }
    buf.dptr = nullptr;
}

int main(int argc, char** argv) {
    // initial window dimensions in pixels
    const int win_w = 800, win_h = 600;
    const int mesh_w = 16, mesh_h = 16;
    // window and monitor
    GLFWwindow *window = nullptr;

    std::printf("Starting GLFW %s\n", glfwGetVersionString());

    // Initialize the entire program (GLFW, OpenGL context, GLAD)
    if (!init(&window, win_w, win_h)) {
        return -1;
    }
    
    std::printf("init() function run properly, executing main loop...\n");

    // Initialize CUDA device (assumes device 0 suitable). For robust apps, query.
    int dev = 0; CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("Using CUDA device %d: %s\n", dev, prop.name);

    // CREATE VBOS HERE
    buffer_t buf1;
    createBuf(buf1, cudaGraphicsMapFlagsWriteDiscard, mesh_w, mesh_h);

    GLuint program = createProgramFromFiles("shaders/basic.vert", "shaders/basic.frag");
    if (!program) {
        std::fprintf(stderr, "Failed to build shader program. Exiting.\n");
        glfwTerminate();
        return -1;
    }

    // int frameCount = 0;
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Run loop instructions (for now, use the loop instead of the helper function)
        // display();

        // Map/update with CUDA (placeholder) ---------------------------------
        CUDA_CHECK(cudaGraphicsMapResources(1, &buf1.cuda_resource, 0));
        size_t num_bytes = 0;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&buf1.dptr, &num_bytes, buf1.cuda_resource));
        // TODO: Launch CUDA kernel to fill (x,y,z,rgba)
        launch_kernel(buf1.dptr, mesh_w, mesh_h);
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &buf1.cuda_resource, 0));

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(program);
        glBindVertexArray(buf1.vao);
        glPointSize(4.0f);
        glDrawArrays(GL_POINTS, 0, mesh_w * mesh_h);
        glBindVertexArray(0);
        glUseProgram(0);

        // Window routines (swap buffers, retrieve events from e.g. key presses or mouse interactions)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup program before termination
    std::printf("Cleaning up...\n");
    
    deleteBuf(buf1);
    // Cleanup shader program (After VAO/VBO deletion is safe, program does not own buffer objects)
    if (program) glDeleteProgram(program);

    std::printf("Terminating program...\n");

    glfwTerminate();
    return 0;
}