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


#include <GL/glew.h> // GLEW Library
#include "GLWindow.h"


GLWindow::GLWindow( float x0, float y0, float w, float h,	void (*renderFunc)(float x0, float y0, float w, float h))
{
	this->x0 = x0;
	this->y0 = y0;
	this->w = w;
	this->h = h;
	this->renderFunc = renderFunc;
}

GLWindow::~GLWindow()
{

}

void GLWindow::render() {
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
//  glOrtho(0,w,0,h, 0.1f, 100000.0f);
  glOrtho(-1.0f,1.0f,-1.0f,1.0f, 0.1f, 100000.0f);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glScalef(1,-1,1);
//  glViewport(0, 0, w, h);
  glViewport(x0, y0, w, h);
  renderFunc(x0,y0,w,h);
}

