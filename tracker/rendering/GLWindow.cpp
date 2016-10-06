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
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <GL/glew.h> // GLEW Library
//#include <GL/gl.h>	// OpenGL32 Library
//#include <GL/glu.h>	// GLU32 Library

#include "GLWindow.h"
#include <camera/Camera.hpp>

using namespace customCameraTools;

GLWindow::GLWindow( int x0, int y0, int w, int h,	void (*renderFunc)())
{
	this->x0 = x0;
	this->y0 = y0;
	this->w = w;
	this->h = h;
	this->camera = NULL;
	this->renderFunc = renderFunc;
}

GLWindow::~GLWindow()
{

}

void GLWindow::render() {
	if (camera == NULL) {
		glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1.0f,1.0f,-1.0f,1.0f, 0.1f, 20000.0f);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glScalef(1,-1,1);
	} else {
		camera->activate();
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(camera->getProjectionMatrix());
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(camera->getModelViewMatrix());
	}
	glViewport(x0, y0, w, h);

	renderFunc();
}

void GLWindow::setCamera( Camera *camera )
{
	this->camera = camera;
}
