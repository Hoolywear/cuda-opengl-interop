#version 410 core
layout(location=0) in vec3 inPos;
layout(location=1) in vec4 inColor;
out vec4 vColor;
void main(){
    gl_Position = vec4(inPos,1.0);
    vColor = vec4(inPos,1.0); // placeholder coloring based on position
    gl_PointSize = 4.0;
}
