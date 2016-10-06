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

#include <ft2build.h>
#include <freetype/freetype.h>

class TrueTypeText {
public:
	float h;
    unsigned int *textures;
    unsigned int list_base;
	int next_p2(int a);
	TrueTypeText();
    void make_dlist(FT_Face face, char ch, unsigned int list_base, unsigned int *tex_base);
	void init(const char *fname, unsigned int h);
	void clean();
	void pushScreenCoordinateMatrix(int *scrWidth, int *scrHeight);
	void pop_projection_matrix();
    void printTTF(float x, float y, float z, float scaleX, float scaleY, bool flipY, const char *text);
};


