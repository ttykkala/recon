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
#pragma once

#include <GL/glew.h> // GLEW Library
#include <GL/gl.h>	// OpenGL32 Library
#include <GL/glu.h>	// GLU32 Library

#include <SDL.h>
#include <SDL_ttf.h>

TTF_Font *font = NULL;
unsigned int renderTextRefCount = 0;

class RenderText {
private:
	void closeFont(TTF_Font *font) {
		if (font != NULL) {
			TTF_CloseFont(font);
		}
	}
	TTF_Font *loadFont(char *name, int size)
	{    
		TTF_Font *font = TTF_OpenFont(name, size);
		if (font == NULL) {
			fprintf(stderr,"Failed to open Font %s: %s\n", name, TTF_GetError());
			return NULL;
		}
		return font;
	}
public:
	int initFonts() {
		if (TTF_Init() < 0) {
			fprintf(stderr,"Couldn't initialize SDL TTF: %s\n", SDL_GetError());
			return 0;
		}
		font = loadFont("fonts\\blackWolf.ttf",64);
		renderTextRefCount = 0;
		return 1;
	}
	void releaseFonts() {
		if (font != NULL) closeFont(font);
		font = NULL; renderTextRefCount = 0;
		TTF_Quit();		
	}

	unsigned int texID;
	int w,h;
	RenderText() { texID = 0; w = 0; h = 0; if (renderTextRefCount == 0) {RenderText::initFonts();} renderTextRefCount++; }
	void updateText(char *text, const GLubyte& R, const GLubyte& G, const GLubyte& B) {
		SDL_Color Color = {R, G, B};
		SDL_Surface *message = TTF_RenderText_Blended(const_cast<TTF_Font*>(font), text, Color);
		if (texID != 0) glDeleteTextures(1,&texID);
		glGenTextures(1,&texID);
		glBindTexture(GL_TEXTURE_2D,texID);
		glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,message->w,message->h, 0, GL_BGRA, GL_UNSIGNED_BYTE, message->pixels);
		w = message->w;
		h = message->h;
		SDL_FreeSurface(message);
	}
	void render(const double& X, const double& Y, const double& Z) {
		if (texID == 0) return;
		glBindTexture(GL_TEXTURE_2D,texID);
		float scaleX = 0.35f*float(w)/320.0f;
		float scaleY = 0.35f*float(h)/240.0f;
		glBegin(GL_QUADS);
		glTexCoord2d(0, 1); glVertex3d(X, Y+scaleY, Z);
		glTexCoord2d(1, 1); glVertex3d(X+scaleX, Y+scaleY, Z);
		glTexCoord2d(1, 0); glVertex3d(X+scaleX, Y, Z);
		glTexCoord2d(0, 0); glVertex3d(X, Y, Z);
		glEnd();
	}
	~RenderText() {
		if (texID != 0) glDeleteTextures(1,&texID); renderTextRefCount--;
		if (renderTextRefCount <=0) RenderText::releaseFonts();
	}
};
