/*
Copyright 2016 Tommi M. Tykk�l�

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

#include <TrueTypeText.h>
#include <GL/glew.h>
#include <ft2build.h>
#include <freetype.h>
#include <ftglyph.h>
#include <ftoutln.h>
#include <fttrigon.h>
#include <vector>
#include <string>

using std::vector;
using std::string;

int TrueTypeText::next_p2(int a) {
		int rval=1;
		while(rval<a) rval<<=1;
		return rval;
}

TrueTypeText::TrueTypeText() {};

void TrueTypeText::make_dlist(FT_Face face, char ch, GLuint list_base, GLuint *tex_base) {
		if (FT_Load_Glyph(face,FT_Get_Char_Index(face,ch),FT_LOAD_DEFAULT)) printf("FT_Load_Glyph failed!\n");
		FT_Glyph glyph;
		if (FT_Get_Glyph(face->glyph,&glyph)) printf("FT_GET_Glyph failed");
		FT_Glyph_To_Bitmap(&glyph,ft_render_mode_normal,0,1);
		FT_BitmapGlyph bitmap_glyph = (FT_BitmapGlyph)glyph;
		FT_Bitmap &bitmap = bitmap_glyph->bitmap;
		int width = next_p2(bitmap.width);
		int height = next_p2(bitmap.rows);
        GLubyte *expanded_data = new GLubyte[2*width*height];
        glLoadIdentity();
        for (int j=0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                expanded_data[2*(i+j*width)] = expanded_data[2*(i+j*width)+1] = (i>=int(bitmap.width) || j>= int(bitmap.rows)) ? 0 : bitmap.buffer[i+bitmap.width*j];
            }
        }
        glBindTexture(GL_TEXTURE_2D, tex_base[ch]);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_LUMINANCE_ALPHA,GL_UNSIGNED_BYTE,expanded_data);
        delete[] expanded_data;

        glLoadIdentity();
        glNewList(list_base+ch,GL_COMPILE);
        glBindTexture(GL_TEXTURE_2D,tex_base[ch]);
        glPushMatrix();
        glTranslated(bitmap_glyph->left,0,0);
        glTranslated(0,int(bitmap_glyph->top)-int(bitmap.rows),0);

        float x = (float)bitmap.width/(float)width;
        float y = (float)bitmap.rows/(float)height;
        glBegin(GL_QUADS);
        glTexCoord2d(0,0); glVertex2f(0,float(bitmap.rows));
        glTexCoord2d(0,y); glVertex2f(0,0);
        glTexCoord2d(x,y); glVertex2f((float)bitmap.width,0);
        glTexCoord2d(x,0); glVertex2f((float)bitmap.width,(float)bitmap.rows);
        glEnd();
        glPopMatrix();
        glTranslatef((GLfloat)(face->glyph->advance.x>>6),0,0);
        glEndList();
}

void TrueTypeText::init(const char *fname, unsigned int h) {
        textures = new GLuint[128];
		this->h = float(h);
		FT_Library library;
		if (FT_Init_FreeType(&library)) printf("FT_Init_FreeType failed!\n");
		FT_Face face;
		if (FT_New_Face(library, fname, 0, &face)) printf("loading font failed?\n");
        FT_Set_Char_Size(face, h<<6, h<<6,72,72);
        list_base=glGenLists(128);
        glGenTextures(128,textures);
        for (unsigned char i = 0; i < 128; i++)
			make_dlist(face,i,list_base,textures);
		FT_Done_Face(face);
		FT_Done_FreeType(library);
	}

void TrueTypeText::clean() {
        glDeleteLists(list_base,128);
        glDeleteTextures(128,textures);
		delete[] textures;
	}

void TrueTypeText::pushScreenCoordinateMatrix(int *scrWidth, int *scrHeight) {
	glPushAttrib(GL_TRANSFORM_BIT);
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT,viewport);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(viewport[0],viewport[2],viewport[1],viewport[3]);
	glPopAttrib();
	*scrWidth = viewport[2]-viewport[0]+1;
	*scrHeight = viewport[3]-viewport[1]+1;
}
void TrueTypeText::pop_projection_matrix() {
	glPushAttrib(GL_TRANSFORM_BIT);
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();
}

void TrueTypeText::printTTF(float x, float y, float z, float scaleX, float scaleY, bool flipY, const char *text) { //const char *fmt, ...) {
   // glPushAttrib(GL_LIST_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TRANSFORM_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_DEPTH_BUFFER_BIT);
    const TrueTypeText &ft_font = *this;
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

    float ySign = 1.0f;
    if (flipY) {
        ySign = -1.0f;
    }

    GLuint font = ft_font.list_base;
    float h = ft_font.h/.63f;
    const char *start_line = text;
    vector<string> lines;
    const char *c = text;
    for (c = text; *c; c++) {
        if (*c == '\n') {
            string line;
            for (const char *n=start_line; n<c; n++) line.append(1,*n);
            lines.push_back(line);
            start_line=c+1;
        }
    }
    if (start_line) {
        string line;
        for (const char *n=start_line;n<c; n++) line.append(1,*n);
        lines.push_back(line);
    }
    glMatrixMode(GL_MODELVIEW);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);

    glListBase(font);    
    glPushMatrix();
    for (size_t i = 0; i < lines.size();i++) {
        glPushMatrix();
        glLoadIdentity();
        glTranslatef(x,y+h*i,z);
        glScalef(scaleX,ySign*scaleY,1);
        glCallLists((GLsizei)lines[i].length(),GL_UNSIGNED_BYTE,lines[i].c_str());
        glPopMatrix();
    }
    glPopMatrix();
    //glPopAttrib();
};


