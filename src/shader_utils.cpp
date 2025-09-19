#include "shader_utils.h"
#include <vector>
#include <cstdio>
#include <fstream>
#include <streambuf>

std::string loadTextFile(const char* path){
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if(!f) return {};
    std::string data;
    f.seekg(0, std::ios::end);
    data.resize(static_cast<size_t>(f.tellg()));
    f.seekg(0, std::ios::beg);
    f.read(&data[0], data.size());
    return data;
}

GLuint compileShader(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok=0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok){
        GLint len=0; glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len+1, '\0');
        glGetShaderInfoLog(s, len, nullptr, log.data());
        std::fprintf(stderr, "Shader compile error (%u):\n%s\n", type, log.data());
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint linkProgram(GLuint vs, GLuint fs){
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){
        GLint len=0; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len+1, '\0');
        glGetProgramInfoLog(p, len, nullptr, log.data());
        std::fprintf(stderr, "Program link error:\n%s\n", log.data());
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

GLuint createProgramFromSource(const char* vsSrc, const char* fsSrc){
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    if(!vs) return 0;
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
    if(!fs){ glDeleteShader(vs); return 0; }
    GLuint prog = linkProgram(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

GLuint createProgramFromFiles(const char* vertexPath, const char* fragmentPath){
    std::string vs = loadTextFile(vertexPath);
    std::string fs = loadTextFile(fragmentPath);
    if(vs.empty()){
        std::fprintf(stderr, "Vertex shader file empty or missing: %s\n", vertexPath);
        return 0;
    }
    if(fs.empty()){
        std::fprintf(stderr, "Fragment shader file empty or missing: %s\n", fragmentPath);
        return 0;
    }
    return createProgramFromSource(vs.c_str(), fs.c_str());
}
