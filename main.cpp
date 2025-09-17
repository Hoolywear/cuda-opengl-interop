#include <glad/glad.h>
#include <GLFW/glfw3.h>

int main(void)
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwSwapInterval(1); // Enable vsync


    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    /* Initialize GLAD */
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        glfwTerminate();
        return -1;
    }

    float points[] = {
   0.0f,  0.5f,  0.0f,
   0.5f, -0.5f,  0.0f,
  -0.5f, -0.5f,  0.0f
};

const char* vertex_shader =
"#version 410 core\n"
"in vec3 vp;"
"void main() {"
"  gl_Position = vec4( vp, 1.0 );"
"}";

const char* fragment_shader =
"#version 410 core\n"
"out vec4 frag_colour;"
"void main() {"
"  frag_colour = vec4( 0.5, 0.0, 0.5, 1.0 );"
"}";

    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);

    GLuint vao = 0;
    glGenVertexArrays( 1, &vao );
    glBindVertexArray( vao );
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0, nullptr);

GLuint vs = glCreateShader( GL_VERTEX_SHADER );
glShaderSource( vs, 1, &vertex_shader, NULL );
glCompileShader( vs );
GLuint fs = glCreateShader( GL_FRAGMENT_SHADER );
glShaderSource( fs, 1, &fragment_shader, NULL );
glCompileShader( fs );

GLuint shader_program = glCreateProgram();
glAttachShader( shader_program, fs );
glAttachShader( shader_program, vs );
glLinkProgram( shader_program );


    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Poll for and process events */
        glfwPollEvents();

        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram( shader_program );
        glBindVertexArray( vao );
        glDrawArrays( GL_TRIANGLES, 0, 3 );

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

    }

    glfwTerminate();
    return 0;
}
