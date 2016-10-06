/*
Copyright 2016 Tommi M. Tykkälä

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "shader.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>

char *Shader::loadSource(const char* filename)
{
    FILE *fp;
    char *content = NULL;

    int count=0;

    if (filename != NULL) {
        fp = fopen(filename,"rt");

        if (fp != NULL) {

            fseek(fp, 0, SEEK_END);
            count = ftell(fp);
            rewind(fp);

            if (count > 0) {
                content = (char *)malloc(sizeof(char) * (count+1));
                count = fread(content,sizeof(char),count,fp);
                content[count] = '\0';
            }
            fclose(fp);
        }
    }
    return content;
}


int Shader::unloadSource(GLcharARB** ShaderSource)
{
   if (*ShaderSource != 0) delete[] *ShaderSource;
   *ShaderSource = 0;
}

Shader::Shader(const char *vsFilename, const char *psFilename) {
    // Step 1: Create a vertex & fragment shader object
    vs = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
    fs = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
    // Step 2: Load source code strings into shaders
    GLcharARB* vsSource; unsigned long vsLen; vsSource = loadSource(vsFilename);
    GLcharARB* psSource; unsigned long psLen; psSource = loadSource(psFilename);

    //printf("vsSource: %s\n",vsSource); fflush(stdin); fflush(stdout);

    glShaderSourceARB(vs, 1, (const GLcharARB**)&vsSource, NULL);
    glShaderSourceARB(fs, 1, (const GLcharARB**)&psSource, NULL);
    // release ASCII GLSL code from memory
    unloadSource(&vsSource);
    unloadSource(&psSource);

    // Step 3: Compile the vertex, fragment shaders.
    GLint compiled;

    glCompileShaderARB(vs);
    glGetObjectParameterivARB(vs, GL_COMPILE_STATUS, &compiled);
    if (compiled) {
        printf("compilation succeeds: %s\n",vsFilename);
    } else {
        printf("compilation fails: %s\n",vsFilename);
        GLint maxLength = 0;
        glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &maxLength);
        //The maxLength includes the NULL character
        std::vector<GLchar> infoLog(maxLength);
        glGetShaderInfoLog(vs, maxLength, &maxLength, &infoLog[0]);
        for (int i = 0; i < maxLength; i++) printf("%c",infoLog[i]);
    }

    glCompileShaderARB(fs);
    glGetObjectParameterivARB(fs, GL_COMPILE_STATUS, &compiled);
    if (compiled) {
        printf("compilation succeeds: %s\n",psFilename);
    } else {
        printf("compilation fails: %s\n",psFilename);
        GLint maxLength = 0;
        glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &maxLength);
        //The maxLength includes the NULL character
        std::vector<GLchar> infoLog(maxLength);
        glGetShaderInfoLog(fs, maxLength, &maxLength, &infoLog[0]);
        for (int i = 0; i < maxLength; i++) printf("%c",infoLog[i]);
    }

    // Step 4: Create a program object
    program = glCreateProgramObjectARB();
    // Step 5: Attach the two compiled shaders
    glAttachObjectARB(program, vs);
    glAttachObjectARB(program, fs);

    glBindAttribLocation(program, 1, "inputVertex");
    glBindAttribLocation(program, 2, "inputColor");
    glBindAttribLocation(program, 3, "inputNormal");

    // Step 6: Link the program object
    glLinkProgramARB(program);

    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked) {
        printf("shader linking succeeded.\n");
    } else {
        printf("shader linking failed.\n");
        GLint maxLength = GL_INFO_LOG_LENGTH;
        std::vector<GLchar> infoLog(GL_INFO_LOG_LENGTH);
        glGetProgramInfoLog(program,GL_INFO_LOG_LENGTH,&maxLength,&infoLog[0]);
        for (int i = 0; i < maxLength; i++) printf("%c",infoLog[i]);
    }

     printf("vertex attrib index: %d\n", glGetAttribLocation(program, "inputVertex"));    
     printf("color attrib index: %d\n",  glGetAttribLocation(program, "inputColor"));
     printf("normal attrib index: %d\n", glGetAttribLocation(program, "inputNormal"));
     printf("uniform light pos index: %d\n", glGetUniformLocation(program,"globalLight"));


    GLint active_attribs, max_length;

    glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &active_attribs);
    glGetProgramiv(program, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &max_length);
    printf("%d active inputs for shader (%s,%s)\n",active_attribs,vsFilename,psFilename);
    fflush(stdin); fflush(stdout);
}

void Shader::bind() {
    // Step 7: Finally, install program object as part of current state
    glUseProgramObjectARB(program);
}

void Shader::unbind() {
    glUseProgramObjectARB(0);
}

int Shader::getAttrib(const char *name) {
    return glGetAttribLocation(program, name);
}

void Shader::setUniformVec4(const char *name, const float *vec4) {
    GLint index = glGetUniformLocation(program,name);
    glUniform4fv(index, 1, vec4);
}


void Shader::release() {
    glDeleteShader(vs);
    glDeleteShader(fs);
    glDeleteProgram(program);
}

Shader::~Shader() {

}
