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

#include <math.h>
#include <stdlib.h>
#include <Renderer.h>
#include <Shaders.h>
#include <time.h>
#include <teapot.h>
#include <string.h>

// two helper functions for clean release code:
#define SAFE_RELEASE(x) { if (x != NULL) {delete x; x = NULL;} }
#define SAFE_RELEASE_ARRAY(x) { if (x != NULL) {delete[] x; x = NULL;} }

Renderer::Renderer() :
    mNumVertices(0),
    mGridReso(0),
    mPosVbo(0),
    mPos2dVbo(0),
    mColorVbo(0),
    mIndexBuffer(0),
    mPointSize(0.005f),
    mWindowW(0),
    mWindowH(0),
    m_downSample(2),
    m_spriteAlpha(0.1f),
    m_lightPos(5.0f, 5.0f, -5.0f),
    m_lightTarget(0.0f, 0.0f, 0.0f),
    m_lightColor(1.0f, 1.0f, 0.5f),
    m_colorAttenuation(0.1f, 0.2f, 0.3f),
    m_cameraPos(0,0,0)
{
    identity3x3(&K[0]);
    identity3x3(&Knew[0]);
    identity4x4(&cameraTransform[0]);
    identity4x4(&cameraPose[0]);

 //   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  //  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0f);

	m_rgbd2DProg       = new GLSLProgram(passThruVS, rgbd2DPS);
    m_colorVertexDepthProg  = new GLSLProgram(colorVS,colorDepthPS);
    m_combine2DProg    = new GLSLProgram(passThruVS, vignette2DPS);
    m_texShader        = new GLSLProgram(colorVS, texPS);
	m_textureProg       = new GLSLProgram(floorVS, floorPS);
		

    //m_autoTexShader    = new GLSLProgram(autoTexRGBDVS, autoTexRGBDPS);
    //m_gouraudShader    = new GLSLProgram(gouraudRGBDVS, gouraudRGBDPS);
    //m_rgba2DProg       = new GLSLProgram(passThruVS, rgba2DPS);
    //m_weightFusionProg = new GLSLProgram(passThruVS, weightFusionPS);
    //m_weight2DProg     = new GLSLProgram(passThruVS, weight2DPS);
     m_depth2DProg      = new GLSLProgram(passThruVS, depth2DPS);
    //m_depthVertexDepthProg  = new GLSLProgram(colorVS,depthDepthPS);
}


Renderer::~Renderer()
{   
	printf("releasing shaders..\n");
    SAFE_RELEASE(m_rgbd2DProg);
    SAFE_RELEASE(m_depth2DProg);
	SAFE_RELEASE(m_textureProg);	
    SAFE_RELEASE(m_combine2DProg);
    SAFE_RELEASE(m_colorVertexDepthProg);
    SAFE_RELEASE(m_texShader);
}
// display texture to screen
void Renderer::displayTexture(GLuint tex)
{
    m_displayTexProg->enable();
    m_displayTexProg->bindTexture("tex", tex, GL_TEXTURE_2D, 0);
    drawQuad();
    m_displayTexProg->disable();
}

void displayText( float x, float y, float r, float g, float b, const char *string ) {
    /*glPushMatrix();
    glLoadIdentity();
    glEnable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
    int j = strlen( string );
    glColor4f( r, g, b, 1.0f );
    glRasterPos2f( x, y );
    for( int i = 0; i < j; i++ ) {
        glutBitmapCharacter( GLUT_BITMAP_TIMES_ROMAN_24, string[i] );
    }
    glPopMatrix();
    glEnable(GL_DEPTH_TEST);*/
}

void Renderer::drawText(int x0, int y0, int width, int height, char *text, float xPos, float yPos, float r, float g, float b) {
 /*   glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_LIGHTING);
    glDisable(GL_ALPHA_TEST);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1e8f);
    glViewport(x0, y0, width,height);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glPopMatrix();

    if (text != NULL) {
        displayText( xPos, yPos, r,g,b, text);
    }
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);*/
}

void Renderer::drawTestScene3D(GLuint boxTex, float minDist, float maxDist) {
   // resetView3d(offX,offY,mWindowW,mWindowH);
    // draw floor
//    vec3f lightColor(1.0f, 1.0f, 0.8f);
    m_textureProg->enable();
    //m_floorProg->bindTexture("tex", floorTex, GL_TEXTURE_2D, 0);
//    m_floorProg->setUniformfv("lightPosEye", getLightPositionEyeSpace(), 3);
//    m_floorProg->setUniformfv("lightColor", lightColor, 3);
/*
    glColor3f(1.0, 1.0, 1.0);
    glNormal3f(0.0, 1.0, 0.0);
    glBegin(GL_QUADS);
    {
        float s = 2000.f;
        float rep = 20.f;
        glTexCoord2f(0.f, rep);
        glVertex3f(-s, 0, s);
        glTexCoord2f(rep, rep);
        glVertex3f(s, 0, s);
        glTexCoord2f(rep, 0.f);
        glVertex3f(s, 0, -s);
        glTexCoord2f(0.f, 0.f);
        glVertex3f(-s, 0, -s);
    }
    glEnd();
*/
    m_textureProg->bindTexture("tex", boxTex, GL_TEXTURE_2D, 0);
    m_textureProg->setUniform1f("minDepth", minDist);
    m_textureProg->setUniform1f("maxDepth", maxDist);

    Eigen::Vector3f halfDim;
    halfDim(0) = m_cubeDim(0)/2;
    halfDim(1) = m_cubeDim(1)/2;
    halfDim(2) = m_cubeDim(2)/2;

    glPushMatrix();
    glTranslatef(m_cubeOrigin(0),m_cubeOrigin(1),m_cubeOrigin(2));
    drawTeapot(halfDim(0));
    glPopMatrix();
    //drawTeapot(m_cubeOrigin,halfDim);
//    drawBox(m_cubeOrigin,halfDim);
    m_textureProg->disable();
/*
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glColor4f(1,1,1,1);
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    drawBox(m_cubeOrigin,m_cubeDim);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    */
}
// display image to the screen as textured quad
void Renderer::displayImage(GLuint texture, float x0, float y0, float dimx, float dimy, float r, float g, float b, float a)
{
    resetView2d(x0,y0,dimx,dimy);

    glBindTexture(GL_TEXTURE_2D, texture);    
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_LIGHTING);
    glEnable(GL_ALPHA_TEST);

    //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    //if (flip)
    glScalef(1,-1,1);

    glEnable(GL_COLOR_MATERIAL);
    glColor4f(r,g,b,a);

    float t = 1.0f;
    float sx = dimx/2;
    float sy = dimy/2;
    x0 += sx;
    y0 += sy;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(x0-sx, y0-sy, -0.5);
    glTexCoord2f(t, 0.0);
    glVertex3f(x0+sx, y0-sy, -0.5);
    glTexCoord2f(t, t);
    glVertex3f(x0+sx, y0+sy, -0.5);
    glTexCoord2f(0.0, t);
    glVertex3f(x0-sx, y0+sy, -0.5);
    glEnd();

    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}

void Renderer::drawFilledCircle(float x0, float y0, float radius, float r, float g, float b) {
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glEnable(GL_COLOR_MATERIAL);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_DEPTH_TEST);

    glColor3f(r,g,b);
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(x0,y0,-1);
    for (int i = 0; i < 30; i++) {
        float t = float(i)/29.0f;
        float x = sin(t*M_PI*2);
        float y = cos(t*M_PI*2);
        glVertex3f(x0+x*radius,y0+y*radius,-1);
    }
    glEnd();
}

void Renderer::drawCursor(float x0, float y0, float dimX, float dimY) {
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_TEXTURE_2D);

    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glEnable(GL_COLOR_MATERIAL);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_DEPTH_TEST);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(0,0,0);
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(x0,y0,-1);
    for (int i = 0; i < 30; i++) {
        float t = float(i)/29.0f;
        float x = sin(t*M_PI*2);
        float y = cos(t*M_PI*2);
        glVertex3f(x0+x*dimX,y0+y*dimY,-1);
    }
    glEnd();

    float dimX2 = dimX*2.0f/3.0f;
    float dimY2 = dimY*2.0f/3.0f;
    glColor3f(1,140.0f/255.0f,0);
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(x0,y0,-1);
    for (int i = 0; i < 30; i++) {
        float t = float(i)/29.0f;
        float x = sin(t*M_PI*2);
        float y = cos(t*M_PI*2);
        glVertex3f(x0+x*dimX2,y0+y*dimY2,-0.99f);
    }
    glEnd();



    glPopMatrix();
    glPopAttrib();
}

void rotatePoint(Eigen::Vector2f &p, float angle) {
    Eigen::Vector2f b;
    b(0) = p(0)*cos(angle)-p(1)*sin(angle);
    b(1) = p(0)*sin(angle)+p(1)*cos(angle);
    p = b;
}

void Renderer::vignettePass(GLuint rgbTex, GLuint vigTex, bool flipY) {
    glPushAttrib(GL_ENABLE_BIT);
    glEnable(GL_TEXTURE_2D);

    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    m_combine2DProg->enable();
    m_combine2DProg->bindTexture("rgbTex", rgbTex, GL_TEXTURE_2D, 0);
    m_combine2DProg->bindTexture("vignetteTex", vigTex, GL_TEXTURE_2D, 1);

    float sX =  1.0f;
    float sY =  1.0f;//*(4.0f/3.0f);
    float flip = 0.0f;
    if (flipY) flip = 1;

    Eigen::Vector2f p0(-sX,-sY);
    Eigen::Vector2f p1( sX,-sY);
    Eigen::Vector2f p2( sX, sY);
    Eigen::Vector2f p3(-sX, sY);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0f-flip);
    glVertex3f(p0(0), p0(1), 0.5);
    glTexCoord2f(1.0f, 1.0-flip);
    glVertex3f(p1(0), p1(1), 0.5);
    glTexCoord2f(1.0f, flip);
    glVertex3f(p2(0), p2(1), 0.5);
    glTexCoord2f(0.0,  flip);
    glVertex3f(p3(0), p3(1), 0.5);
    glEnd();
    m_combine2DProg->disable();
    glPopAttrib();
   // glEnable(GL_LIGHTING);
   // glEnable(GL_DEPTH_TEST);

}

void Renderer::displayRGBDImage(GLuint rgbTex, bool flipY, bool rot90, float zAngle, bool binaryAlpha)
{
    glPushAttrib(GL_ENABLE_BIT);
    glEnable(GL_TEXTURE_2D);

    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    m_rgbd2DProg->enable();
    m_rgbd2DProg->bindTexture("rgbTex", rgbTex, GL_TEXTURE_2D, 0);
    float thresholdAlpha = 0.0f;
    if (binaryAlpha) {
        thresholdAlpha = 1.0f;
    }
    //printf("threshold alpha set to : %f\n",thresholdAlpha);
    m_rgbd2DProg->setUniform1f("thresholdAlpha",thresholdAlpha);

    float sX =  1.0f;
    float sY =  1.0f;//*(4.0f/3.0f);
    float flip = 0.0f;
    if (flipY) flip = 1;

    float radAngle = deg2rad(zAngle);

    Eigen::Vector2f p0(-sX,-sY); rotatePoint(p0,radAngle);
    Eigen::Vector2f p1( sX,-sY); rotatePoint(p1,radAngle);
    Eigen::Vector2f p2( sX, sY); rotatePoint(p2,radAngle);
    Eigen::Vector2f p3(-sX, sY); rotatePoint(p3,radAngle);

    glBegin(GL_QUADS);
    if (!rot90) {
        glTexCoord2f(0.0, 1.0f-flip);
        glVertex3f(p0(0), p0(1), 0.5);
        glTexCoord2f(1.0f, 1.0-flip);
        glVertex3f(p1(0), p1(1), 0.5);
        glTexCoord2f(1.0f, flip);
        glVertex3f(p2(0), p2(1), 0.5);
        glTexCoord2f(0.0,  flip);
        glVertex3f(p3(0), p3(1), 0.5);
    } else {
        glTexCoord2f(0.0,  flip);
        glVertex3f(p0(0), p0(1), 0.5);
        glTexCoord2f(0.0, 1.0f-flip);
        glVertex3f(p1(0), p1(1), 0.5);
        glTexCoord2f(1.0f, 1.0-flip);
        glVertex3f(p2(0), p2(1), 0.5);
        glTexCoord2f(1.0f, flip);
        glVertex3f(p3(0), p3(1), 0.5);
    }
    glEnd();
    m_rgbd2DProg->disable();
    glPopAttrib();
   // glEnable(GL_LIGHTING);
   // glEnable(GL_DEPTH_TEST);
}

void Renderer::displayRGBDImage_old(GLuint rgbTex, GLuint depthTex, int flip)
{
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glEnable(GL_LIGHTING);

    m_depth2DProg->enable();
    m_depth2DProg->bindTexture("rgbTex", rgbTex, GL_TEXTURE_2D, 0);
    m_depth2DProg->bindTexture("depthTex", depthTex, GL_TEXTURE_2D, 1);
    m_depth2DProg->setUniform1f("flip",float(flip));

    glPushMatrix();
    float s = 1.0f;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0f);
    glVertex3f(-s, -s, 0.5);
    glTexCoord2f(1.0f, 0.0);
    glVertex3f(s, -s, 0.5);
    glTexCoord2f(1.0f, 1);
    glVertex3f(s, s, 0.5);
    glTexCoord2f(0.0, 1.0f);
    glVertex3f(-s, s, 0.5);
    glEnd();
    glPopMatrix();

    m_depth2DProg->disable();

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
}


float clamp(float val, int minval, float maxval) {
    if (val < minval) val=minval;
    if (val > maxval) val=maxval;
    return val;
}

void renderSphereBall(float radius, int verticalSlices, int horizontalSlices, bool flipWorld, float cutPercentage, float radScale) {
    glBegin(GL_QUADS);
    float zSign = -1.0f;
    if (flipWorld) zSign = 1.0f;
    for (float y = 0; y < verticalSlices; y++) {
        for (float x = 0; x < cutPercentage*float(horizontalSlices); x++) {
            float uf0 = (x+0)/float(horizontalSlices-1);
            float uf1 = (x+1)/float(horizontalSlices-1);
            float vf0 = (y+0)/float(verticalSlices-1);
            float vf1 = (y+1)/float(verticalSlices-1);

            float phi0,phi1;
            //if (!flipWorld)
			{

                phi0 = M_PI*(1-uf0);
                phi1 = M_PI*(1-uf1);

//                phi0 = M_PI*uf0;
//                phi1 = M_PI*uf1;

            } /*else {
				phi0 = M_PI*uf0;
				phi1 = M_PI*uf1;            	
            }*/
            float theta0   = 2*M_PI*vf0;
            float theta1   = 2*M_PI*vf1;

            float vx,vy,vz,texU,texV,angle,radius2d;
            vx = cos(theta0)*sin(phi0);
            vy = sin(theta0)*sin(phi0);
            vz = cos(phi0);
            angle = atan2(vy,vx);
            //radius2d = sqrtf(vy*vy+vx*vx); /*if (flipWorld) radius2d = (1-radius2d);*/ radius2d *= radScale;
			radius2d = sqrtf(vy*vy+vx*vx); if (flipWorld) radius2d = (1-radius2d); radius2d *= radScale;
            texU = clamp(0.5f+sin(angle)*radius2d,0,1);
            texV = clamp(0.5f+cos(angle)*radius2d,0,1);
            glTexCoord2f(texU,texV);
            glVertex3f(zSign*vx*radius,vy*radius,zSign*vz*radius);

            vx = cos(theta0)*sin(phi1);
            vy = sin(theta0)*sin(phi1);
            vz = cos(phi1);
            angle = atan2(vy,vx);
            //radius2d = sqrtf(vy*vy+vx*vx); /*if (flipWorld) radius2d = (1-radius2d);*/ radius2d *= radScale;
			radius2d = sqrtf(vy*vy+vx*vx); if (flipWorld) radius2d = (1-radius2d); radius2d *= radScale;
			texU = clamp(0.5f+sin(angle)*radius2d,0,1);
            texV = clamp(0.5f+cos(angle)*radius2d,0,1);
            glTexCoord2f(texU,texV);
            glVertex3f(zSign*vx*radius,vy*radius,zSign*vz*radius);

            vx = cos(theta1)*sin(phi1);
            vy = sin(theta1)*sin(phi1);
            vz = cos(phi1);
            angle = atan2(vy,vx);
            //radius2d = sqrtf(vy*vy+vx*vx); /*if (flipWorld) radius2d = (1-radius2d);*/ radius2d *= radScale;
			radius2d = sqrtf(vy*vy+vx*vx); if (flipWorld) radius2d = (1-radius2d); radius2d *= radScale;
			texU = clamp(0.5f+sin(angle)*radius2d,0,1);
            texV = clamp(0.5f+cos(angle)*radius2d,0,1);
            glTexCoord2f(texU,texV);
            glVertex3f(zSign*vx*radius,vy*radius,zSign*vz*radius);

            vx = cos(theta1)*sin(phi0);
            vy = sin(theta1)*sin(phi0);
            vz = cos(phi0);
            angle = atan2(vy,vx);
            //radius2d = sqrtf(vy*vy+vx*vx); /*if (flipWorld) radius2d = (1-radius2d);*/ radius2d *= radScale;
			radius2d = sqrtf(vy*vy+vx*vx); if (flipWorld) radius2d = (1-radius2d); radius2d *= radScale;

			texU = clamp(0.5f+sin(angle)*radius2d,0,1);
            texV = clamp(0.5f+cos(angle)*radius2d,0,1);
            glTexCoord2f(texU,texV);
            glVertex3f(zSign*vx*radius,vy*radius,zSign*vz*radius);
        }
    }
    glEnd();

}

void Renderer::renderSphere(GLuint rgbTex, float radius, int verticalSlices, int horizontalSlices, bool flipWorld, float sphereAmount, float radScale, float vShift)
{

    glPushAttrib(GL_ENABLE_BIT);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glDisable(GL_ALPHA_TEST);
    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadIdentity();
    glLoadMatrixf(&mtx[0]);
    //glRotatef(12.0f,0,1,0);
    //glRotatef(90.0f,1,0,0);
    glRotatef(-100.0f,1,0,0);
    glRotatef(90.0f,0,0,1);
    //    glMultMatrixf(&cameraTransform[0]);
    glTranslatef(0,0,vShift);
    m_texShader->enable();
	//printf("rgbTex: %d\n",rgbTex);
    m_texShader->bindTexture("rgbTex", rgbTex, GL_TEXTURE_2D, 0);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    renderSphereBall(radius,verticalSlices,horizontalSlices,flipWorld,sphereAmount,radScale);
    m_texShader->disable();

    glPopMatrix();
    glPopAttrib();
}

void Renderer::renderFrontSphere(GLuint rgbTex, float radius, int verticalSlices, int horizontalSlices, bool flipWorld, float sphereAmount, float radScale, float vShift, float yRot)
{

    glPushAttrib(GL_ENABLE_BIT);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_COLOR_MATERIAL);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadIdentity();
    glLoadMatrixf(&mtx[0]);
    //glRotatef(12.0f,0,1,0);
    glRotatef(yRot,1,0,0);
    glRotatef(180.0f,0,1,0);
    glRotatef(90.0f,0,0,1);
//    glMultMatrixf(&cameraTransform[0]);
    glTranslatef(0,0,vShift);
    m_texShader->enable();
    //printf("rgbTex: %d\n",rgbTex);
    m_texShader->bindTexture("rgbTex", rgbTex, GL_TEXTURE_2D, 0);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    renderSphereBall(radius,verticalSlices,horizontalSlices,flipWorld,sphereAmount,radScale);
    m_texShader->disable();

    glPopMatrix();
    glPopAttrib();
}

void Renderer::renderFrontWireSphere(float radius, int verticalSlices, int horizontalSlices, bool flipWorld, float sphereAmount, float radScale, float vShift, float yRot)
{
    glPushAttrib(GL_ENABLE_BIT);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glDisable(GL_ALPHA_TEST);
    glEnable(GL_DEPTH_TEST);
    glLineWidth(2.0f);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadIdentity();
    glLoadMatrixf(&mtx[0]);
    //glRotatef(12.0f,0,1,0);
    glRotatef(yRot,1,0,0);
    glRotatef(180.0f,0,1,0);
    glRotatef(90.0f,0,0,1);
//    glMultMatrixf(&cameraTransform[0]);
    glTranslatef(0,0,vShift);

    glDisable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    glColor4f(0,1,0,1);
    renderSphereBall(radius,verticalSlices,horizontalSlices,flipWorld,sphereAmount,radScale);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);


    glPopMatrix();
    glPopAttrib();
}


void Renderer::renderWireSphere(float radius, int verticalSlices, int horizontalSlices, bool flipWorld, float sphereAmount, float radScale, float vShift)
{
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glDisable(GL_ALPHA_TEST);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadIdentity();
    glLoadMatrixf(&mtx[0]);
    glRotatef(90.0f,1,0,0);
    glTranslatef(0,0,vShift);
    //glScalef(0.99f,0.99f,0.99f);
    glDisable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    glColor4f(0,1,0,1);
    renderSphereBall(radius,verticalSlices,horizontalSlices,flipWorld,sphereAmount,radScale);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

    glPopMatrix();
    glPopAttrib();
}

void Renderer::displayWeightImage(GLuint rgbdTex, float initialWeight, int width, int height, float minDepth, float maxDepth, bool flipY) {
    glPushMatrix();
    glLoadIdentity();

    m_weight2DProg->enable();
    m_weight2DProg->bindTexture("rgbdTex", rgbdTex, GL_TEXTURE_2D, 0);
    m_weight2DProg->setUniform1f("minDepth",minDepth);
    m_weight2DProg->setUniform1f("maxDepth",maxDepth);
    m_weight2DProg->setUniform1f("initialWeight",initialWeight);

    float sX =  1.0f;
    float sY =  1.0f;
    float flip = 0.0f;
    if (flipY) flip = 1;

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0f-flip);
    glVertex3f(-sX, -sY, 0.5);
    glTexCoord2f(1.0f, 1.0-flip);
    glVertex3f(sX, -sY, 0.5);
    glTexCoord2f(1.0f, flip);
    glVertex3f(sX, sY, 0.5);
    glTexCoord2f(0.0,  flip);
    glVertex3f(-sX, sY, 0.5);
    glEnd();
    m_weight2DProg->disable();
    glPopMatrix();
}


void Renderer::combineRGBDLayers(GLuint rgbdTex0, GLuint weightTex0, GLuint rgbdTex1, GLuint weightTex1, int width, int height, float minDepth, float maxDepth, bool flipY, float zepsilon)
{
    glPushMatrix();
    glLoadIdentity();

    m_rgba2DProg->enable();
    m_rgba2DProg->bindTexture("rgbdTex0", rgbdTex0, GL_TEXTURE_2D, 0);
    m_rgba2DProg->bindTexture("rgbdTex1", rgbdTex1, GL_TEXTURE_2D, 1);
    m_rgba2DProg->bindTexture("weightTex0", weightTex0, GL_TEXTURE_2D, 2);
    m_rgba2DProg->bindTexture("weightTex1", weightTex1, GL_TEXTURE_2D, 3);
    m_rgba2DProg->setUniform1f("zepsilon",zepsilon);
    float sX =  1.0f;
    float sY =  1.0f;
    float flip = 0.0f;
    if (flipY) flip = 1;

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0f-flip);
    glVertex3f(-sX, -sY, 0.5);
    glTexCoord2f(1.0f, 1.0-flip);
    glVertex3f(sX, -sY, 0.5);
    glTexCoord2f(1.0f, flip);
    glVertex3f(sX, sY, 0.5);
    glTexCoord2f(0.0,  flip);
    glVertex3f(-sX, sY, 0.5);
    glEnd();
    m_rgba2DProg->disable();
    glPopMatrix();
}

void Renderer::combineWeightLayers(GLuint rgbdTex0, GLuint weightTex0, GLuint rgbdTex1, GLuint weightTex1, int width, int height, float minDepth, float maxDepth, bool flipY, float zepsilon)
{
    glPushMatrix();
    glLoadIdentity();

    m_weightFusionProg->enable();
    m_weightFusionProg->bindTexture("rgbdTex0", rgbdTex0, GL_TEXTURE_2D, 0);
    m_weightFusionProg->bindTexture("rgbdTex1", rgbdTex1, GL_TEXTURE_2D, 1);
    m_weightFusionProg->bindTexture("weightTex0", weightTex0, GL_TEXTURE_2D, 2);
    m_weightFusionProg->bindTexture("weightTex1", weightTex1, GL_TEXTURE_2D, 3);
    m_weightFusionProg->setUniform1f("zepsilon",zepsilon);
    float sX =  1.0f;
    float sY =  1.0f;
    float flip = 0.0f;
    if (flipY) flip = 1;

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0f-flip);
    glVertex3f(-sX, -sY, 0.5);
    glTexCoord2f(1.0f, 1.0-flip);
    glVertex3f(sX, -sY, 0.5);
    glTexCoord2f(1.0f, flip);
    glVertex3f(sX, sY, 0.5);
    glTexCoord2f(0.0,  flip);
    glVertex3f(-sX, sY, 0.5);
    glEnd();
    m_weightFusionProg->disable();
    glPopMatrix();
}

void Renderer::displayDepthImage(GLuint depthTex, int width, int height, int flip)
{
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_LIGHTING);
    glDisable(GL_ALPHA_TEST);
    //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glPushMatrix();
    glLoadIdentity();

    m_depth2DProg->enable();
    m_depth2DProg->bindTexture("depthTex", depthTex, GL_TEXTURE_2D, 1);
    m_depth2DProg->setUniform1f("flip",float(flip));

    float s = 1.0f;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0f);
    glVertex3f(-s, -s, 0.5);
    glTexCoord2f(1.0f, 1.0);
    glVertex3f(s, -s, 0.5);
    glTexCoord2f(1.0f, 0);
    glVertex3f(s, s, 0.5);
    glTexCoord2f(0.0, 0.0f);
    glVertex3f(-s, s, 0.5);
    glEnd();
    m_depth2DProg->disable();
    glPopMatrix();
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
}

void Renderer::setCameraMatrix(float *modelView) {
    memcpy(&cameraTransform[0],modelView,sizeof(float)*16);
    invertRT4(modelView,&cameraPose[0]);
    copy3x3(&cameraPose[0],&R[0]);
    m_cameraPos(0) = cameraPose[3];
    m_cameraPos(1) = cameraPose[7];
    m_cameraPos(2) = cameraPose[11];
}

void Renderer::setCameraMatrix(Eigen::Matrix4f &modelView) {
	cameraTransform[0] = modelView(0,0);  cameraTransform[1] = modelView(0,1);  cameraTransform[2] = modelView(0,2);  cameraTransform[3] = modelView(0,3);
	cameraTransform[4] = modelView(1,0);  cameraTransform[5] = modelView(1,1);  cameraTransform[6] = modelView(1,2);  cameraTransform[7] = modelView(1,3);
	cameraTransform[8] = modelView(2,0);  cameraTransform[9] = modelView(2,1);  cameraTransform[10] = modelView(2,2); cameraTransform[11] = modelView(2,3);
	cameraTransform[12] = modelView(3,0); cameraTransform[13] = modelView(3,1); cameraTransform[14] = modelView(3,2); cameraTransform[15] = modelView(3,3);
	//memcpy(&cameraTransform[0],modelView,sizeof(float)*16);
	invertRT4(&cameraTransform[0],&cameraPose[0]);
	copy3x3(&cameraPose[0],&R[0]);
    m_cameraPos(0) = cameraPose[3];
    m_cameraPos(1) = cameraPose[7];
    m_cameraPos(2) = cameraPose[11];
}

void Renderer::setCameraPose(Eigen::Matrix4f &modelView) {
	cameraPose[0] = modelView(0,0); cameraPose[1] = modelView(0,1); cameraPose[2] = modelView(0,2); cameraPose[3] = modelView(0,3);
	cameraPose[4] = modelView(1,0); cameraPose[5] = modelView(1,1); cameraPose[6] = modelView(1,2); cameraPose[7] = modelView(1,3);
	cameraPose[8] = modelView(2,0); cameraPose[9] = modelView(2,1); cameraPose[10] = modelView(2,2); cameraPose[11] = modelView(2,3);
	cameraPose[12] = modelView(3,0); cameraPose[13] = modelView(3,1); cameraPose[14] = modelView(3,2); cameraPose[15] = modelView(3,3);
	//memcpy(&cameraTransform[0],modelView,sizeof(float)*16);
	invertRT4(&cameraPose[0],&cameraTransform[0]);
	copy3x3(&cameraPose[0],&R[0]);
    m_cameraPos(0) = cameraPose[3];
    m_cameraPos(1) = cameraPose[7];
    m_cameraPos(2) = cameraPose[11];
}

float *Renderer::getCameraMatrix() {
    return &cameraTransform[0];
}

float *Renderer::getPoseMatrix() {
    return &cameraPose[0];
}

void Renderer::render(int offX, int offY, int viewportW, int viewportH, char *text)
{
    resetView3d(offX,offY,viewportW,viewportH);
    //glColor3f(1.0, 0.0, 0.0);

    //glDisable(GL_TEXTURE_2D);
    //glDisable(GL_LIGHTING);
    //glEnable(GL_COLOR_MATERIAL);
    //glColor4f(1,1,1,1);
    //glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    //drawBox(m_cubeOrigin,m_cubeDim);
    //glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

    if (text != NULL) {
        resetView2d(offX,offY,viewportW,viewportH);
        displayText( viewportW*0.1f,viewportH*0.9f, 1.0f,1.0f,1.0f, text);
    }

    glEnable(GL_LIGHTING);
    resetView3d(0,0,mWindowW,mWindowH);
    //glutReportErrors();
}

void Renderer::renderLidar(unsigned int vbo, int pointCount, float pointSize, float minDist, float maxDist) {
    if (vbo == 0) return;

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadIdentity();
    glLoadMatrixf(&mtx[0]);

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_LIGHTING);
    glDepthMask(GL_TRUE);
    glPointSize(pointSize);
    glColor4f(1,1,1,1);

    m_colorVertexDepthProg->enable();
    m_colorVertexDepthProg->setUniform1f("minDepth", minDist);
    m_colorVertexDepthProg->setUniform1f("maxDepth", maxDist);

    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_COLOR_ARRAY );
    float *ptrBase = 0;
    glVertexPointer(3,GL_FLOAT,6*sizeof(float),ptrBase); ptrBase+=3;
    glColorPointer(3,GL_FLOAT,6*sizeof(float),ptrBase);
    glDrawArrays(GL_POINTS,0,pointCount);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    m_colorVertexDepthProg->disable();
/*
    nv::vec3f origin,dim,color;
    origin.x = bbox->corner[0];
    origin.y = bbox->corner[1];
    origin.z = bbox->corner[2];
    dim.x    = bbox->dim[0];
    dim.y    = bbox->dim[1];
    dim.z    = bbox->dim[2];
    color.x  = 1;
    color.y  = 0;
    color.z  = 0;

    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    drawBox(origin,dim,color);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
*/
    glPopMatrix();
}
/*
void SmokeRenderer::renderLidarDepth(LidarDataset &lidarData, float minDist, float maxDist) {
    if (lidarData.vbo < 0) return;

    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadMatrixf(&mtx[0]);

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_LIGHTING);
    glDepthMask(GL_TRUE);
    glPointSize(2.0f);
    glColor4f(1,1,1,1);

    m_depthVertexDepthProg->enable();
    m_depthVertexDepthProg->setUniform1f("minDepth", minDist);
    m_depthVertexDepthProg->setUniform1f("maxDepth", maxDist);

    glBindBuffer( GL_ARRAY_BUFFER, lidarData.vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_COLOR_ARRAY );
    float *ptrBase = 0;
    glVertexPointer(3,GL_FLOAT,6*sizeof(float),ptrBase); ptrBase+=3;
    glColorPointer(3,GL_FLOAT,6*sizeof(float),ptrBase);
    glDrawArrays(GL_POINTS,0,lidarData.pointCount);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    m_depthVertexDepthProg->disable();

    nv::vec3f origin,dim;
    origin.x = lidarData.center[0];
    origin.y = lidarData.center[1];
    origin.z = lidarData.center[2];
    dim.x    = lidarData.extents[0];
    dim.y    = lidarData.extents[1];
    dim.z    = lidarData.extents[2];


    glPopMatrix();
}
*/

void Renderer::renderTexturedVbo(Keyframe *kf, Calibration &calib)
{
    unsigned int vbo = kf->vbo;
    unsigned int ibo = kf->ibo;
    int depthWidth   = kf->width;
    int depthHeight  = kf->height;
    int texWidth     = kf->rgbImage.cols;
    int texHeight    = kf->rgbImage.rows;
    GLuint rgbTex    = kf->rgbTex;

    if (vbo < 0 || ibo < 0 || texWidth < 1 || texHeight < 1 || depthWidth < 1 || depthHeight < 1) return;
    float w = float(texWidth); float h = float(texHeight);
    glColor3f(1.0, 0.0, 0.0);

    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadMatrixf(&mtx[0]);
    //glScalef(1,1,-1);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glColor4f(1,1,1,1);
    glFrontFace(GL_CCW);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    //glDisable(GL_CULL_FACE);
    float invAspect = float(texWidth)/float(texHeight);
    double *Krgb = calib.getKR();
    double *KcR  = calib.getKcR();
    float *TLR = &calib.getCalibData()[TLR_OFFSET];

    m_autoTexShader->enable();
    m_autoTexShader->setUniform1f("minDepth", calib.getMinDist()); //printf("%f %f\n",calib.getMinDist(),calib.getMaxDist());
    m_autoTexShader->setUniform1f("maxDepth", calib.getMaxDist());
    m_autoTexShader->setUniformMatrix4fv("Tbaseline",TLR,true);
    m_autoTexShader->setUniform1f("fx",fabs(Krgb[0]));
    m_autoTexShader->setUniform1f("fy",fabs(Krgb[4]*invAspect)); // remove aspect ratio based scaling (texture coords will be [0,1] x [0,1])
    m_autoTexShader->setUniform1f("cx",fabs(Krgb[2]));
    m_autoTexShader->setUniform1f("cy",fabs(Krgb[5]*invAspect)); // remove aspect ratio based scaling (texture coords will be [0,1] x [0,1])
    m_autoTexShader->setUniform1f("kc0",KcR[0]);
    m_autoTexShader->setUniform1f("kc1",KcR[1]);
    m_autoTexShader->setUniform1f("kc2",KcR[2]);
    m_autoTexShader->setUniform1f("kc3",KcR[3]);
    m_autoTexShader->setUniform1f("kc4",KcR[4]);
    m_autoTexShader->bindTexture("tex", rgbTex, GL_TEXTURE_2D, 0);

    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_NORMAL_ARRAY );
    float *ptrBase = 0;
    glVertexPointer(3,GL_FLOAT,6*sizeof(float),ptrBase); ptrBase+=3;
    glNormalPointer(GL_FLOAT,6*sizeof(float),ptrBase);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_QUADS, kf->numElements, GL_UNSIGNED_INT, (GLvoid*) 0);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    m_autoTexShader->disable();
    glPopMatrix();
}

void Renderer::renderGouraudVbo(Keyframe *kf, Calibration &calib, float *lightDir)
{
    unsigned int vbo = kf->vbo;
    unsigned int ibo = kf->ibo;
    int depthWidth   = kf->width;
    int depthHeight  = kf->height;
    int texWidth     = kf->rgbImage.cols;
    int texHeight    = kf->rgbImage.rows;

    if (vbo < 0 || ibo < 0 || texWidth < 1 || texHeight < 1 || depthWidth < 1 || depthHeight < 1) return;
    float w = float(texWidth); float h = float(texHeight);
    glColor3f(1.0, 0.0, 0.0);

    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadMatrixf(&mtx[0]);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);
    glColor4f(1,1,1,1);
    glFrontFace(GL_CCW);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);

    m_gouraudShader->enable();
    m_gouraudShader->setUniform1f("minDepth", calib.getMinDist());
    m_gouraudShader->setUniform1f("maxDepth", calib.getMaxDist());
    m_gouraudShader->setUniform3f("lightDir",lightDir[0],lightDir[1],lightDir[2]);

    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_NORMAL_ARRAY );
    float *ptrBase = 0;
    glVertexPointer(3,GL_FLOAT,6*sizeof(float),ptrBase); ptrBase+=3;
    glNormalPointer(GL_FLOAT,6*sizeof(float),ptrBase);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_QUADS, kf->numElements, GL_UNSIGNED_INT, (GLvoid*) 0);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    m_gouraudShader->disable();
    glPopMatrix();
}

void Renderer::renderTexturedMesh(ProjectData *pointCloud, int texWidth, int texHeight, Calibration &calib, GLuint rgbTex)
{
    float w = float(texWidth); float h = float(texHeight);
    glColor3f(1.0, 0.0, 0.0);

    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadMatrixf(&mtx[0]);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
    glColor4f(1,1,1,1);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	float invAspect = float(texWidth)/float(texHeight);
	double *Krgb = calib.getKR();
	double *KcR  = calib.getKcR();
	float *TLR = &calib.getCalibData()[TLR_OFFSET];

	m_autoTexShader->enable();
	m_autoTexShader->setUniform1f("minDepth", calib.getMinDist());
	m_autoTexShader->setUniform1f("maxDepth", calib.getMaxDist());
	m_autoTexShader->setUniformMatrix4fv("Tbaseline",TLR,true);
	m_autoTexShader->setUniform1f("fx",Krgb[0]);
	m_autoTexShader->setUniform1f("fy",Krgb[4]*invAspect); // remove aspect ratio based scaling (texture coords will be [0,1] x [0,1])
	m_autoTexShader->setUniform1f("cx",Krgb[2]);
	m_autoTexShader->setUniform1f("cy",Krgb[5]*invAspect); // remove aspect ratio based scaling (texture coords will be [0,1] x [0,1])
	m_autoTexShader->setUniform1f("kc0",KcR[0]);
	m_autoTexShader->setUniform1f("kc1",KcR[1]);
	m_autoTexShader->setUniform1f("kc2",KcR[2]);
	m_autoTexShader->setUniform1f("kc3",KcR[3]);
	m_autoTexShader->setUniform1f("kc4",KcR[4]);
	m_autoTexShader->bindTexture("tex", rgbTex, GL_TEXTURE_2D, 0);

//genProjectionMtx(Kin,texWidth,texHeight,nearZ,farZ,&P[0]);

    glBegin(GL_QUADS);
    for (int j = 1; j < (texHeight-1); j++) {
        for (int i = 1; i < (texWidth-1); i++) {
            int offset = i+j*texWidth;
            ProjectData &p0 = pointCloud[offset]; //float n0 = randGrid[offset];
            ProjectData &p1 = pointCloud[offset+texWidth]; //float n1 = randGrid[offset+texWidth];
            ProjectData &p2 = pointCloud[offset+1+texWidth]; //float n2 = randGrid[offset+1+texWidth];
            ProjectData &p3 = pointCloud[offset+1]; //float n3 = randGrid[offset+1];

            if (p0.magGrad < 0) continue;
            if (p1.magGrad < 0) continue;
            if (p2.magGrad < 0) continue;
            if (p3.magGrad < 0) continue;

			float minZ = MIN(MIN(MIN(p0.pz,p1.pz),p2.pz),p3.pz);
			float maxZ = MAX(MAX(MAX(p0.pz,p1.pz),p2.pz),p3.pz);

            if (maxZ >= -300.0f || fabs(maxZ-minZ) > 150.0f) continue;
			glVertex3f(p0.px,p0.py,p0.pz);
			glVertex3f(p1.px,p1.py,p1.pz);
			glVertex3f(p2.px,p2.py,p2.pz);
			glVertex3f(p3.px,p3.py,p3.pz);
        }
    }
    glEnd();

	m_autoTexShader->disable();
    glPopMatrix();
  //  glDisable(GL_BLEND);
  /*
    if (text != NULL) {
        resetView2d(offX,offY,viewportW,viewportH);
        displayText( viewportW*0.01f,viewportH*0.95f, 1.0f,1.0f,1.0f, text);
    }*/
    //glutReportErrors();
}

void Renderer::draw3DPoints(Eigen::MatrixXf &featurePoints3D, Eigen::MatrixXf &statusPoints3D, float r, float g, float b)
{
	glColor3f(1.0, 0.0, 0.0);
	glPushMatrix();
	float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
	glLoadMatrixf(&mtx[0]);

	glDisable(GL_TEXTURE_2D);
	glEnable(GL_COLOR_MATERIAL);
	//glEnable(GL_DEPTH_TEST);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	glDisable(GL_LIGHTING);

	glPointSize(6.0f);
	glBegin(GL_POINTS);
	for (size_t i = 0; i < featurePoints3D.cols(); i++) {
		if (statusPoints3D(0,i) < 0) continue;//glColor4f(1,0,0,1);
		else {
			float rr = float(int(statusPoints3D(0,i)*65536)%255)/255.0f;
			float gg = float(int(statusPoints3D(0,i)*12241121)%255)/255.0f;
			float bb = float(int(statusPoints3D(0,i)*110041042)%255)/255.0f;
			glColor4f(rr,gg,bb,1);
		}
		glVertex3f(featurePoints3D(0,i),featurePoints3D(1,i),featurePoints3D(2,i));
	}
	glEnd();
	glPopMatrix();
}
// create an OpenGL texture
GLuint Renderer::createTexture(GLenum target, int w, int h, GLint internalformat, GLenum format)
{
    GLuint texid;
    glGenTextures(1, &texid);
    glBindTexture(target, texid);

    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_FLOAT, 0);
    return texid;
}

/*
// create buffers for off-screen rendering
void SmokeRenderer::createBuffers(int w, int h)
{
    if (w == mWindowW && h == mWindowH) return;

    if (m_imageFbo)
    {
        glDeleteTextures(1, &m_imageTex);
        glDeleteTextures(1, &m_depthTex);
        delete m_imageFbo;
    }

    // create fbo for image buffer
    GLint format = GL_RGBA16F_ARB;
    //GLint format = GL_LUMINANCE16F_ARB;
    //GLint format = GL_RGBA8;
    m_imageTex = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, format, GL_RGBA);
    m_depthTex = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, GL_DEPTH_COMPONENT24_ARB, GL_DEPTH_COMPONENT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    m_imageFbo = new FramebufferObject();
    m_imageFbo->AttachTexture(GL_TEXTURE_2D, m_imageTex, GL_COLOR_ATTACHMENT0_EXT);
    m_imageFbo->AttachTexture(GL_TEXTURE_2D, m_depthTex, GL_DEPTH_ATTACHMENT_EXT);
    m_imageFbo->IsValid();
}*/

void Renderer::genProjectionMtx(float *Kin, float texWidth, float texHeight, float nearZ, float farZ, float *P) {
    P[0]  = -2*Kin[0]/texWidth;  P[1]  = 0;                   P[2]  = 1-(2*Kin[2]/texWidth);        P[3] = 0;
    P[4]  = 0.0f;                P[5]  = 2*Kin[4]/texHeight;  P[6]  = -1+2*Kin[5]/texHeight;        P[7] = 0;
	P[8]  = 0.0f;                P[9]  = 0.0f;                P[10] = (farZ+nearZ)/(nearZ-farZ);    P[11] = 2*nearZ*farZ/(nearZ-farZ);
	P[12] = 0.0f;                P[13] = 0.0f;                P[14] = -1;                           P[15] = 0;
}

void Renderer::setProjectionMatrix(float *Kin, float texWidth, float texHeight, float nearZ, float farZ) {
    float P[16],PT[16];
    //if (Kin != this->Knew)
//    memcpy(this->Knew,Kin,sizeof(float)*9);
    for (int i = 0; i < 9; i++) Knew[i] = Kin[i];
//    this->Knew[0] /= 1.5;
//    this->Knew[4] /= 1.5;
//    float KK[9];
//    for (int i = 0; i < 9; i++) KK[i] = Kin[i]*4;
//    dumpMatrix("K",KK,3,3);
    this->zNear    = nearZ;
    this->zFar     = farZ;
    this->mWindowW = texWidth;
    this->mWindowH = texHeight;
    genProjectionMtx(Kin,texWidth,texHeight,nearZ,farZ,&P[0]);
    transpose4x4(&P[0],&PT[0]);
   // dumpMatrix("P",P,4,4);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glLoadMatrixf(&PT[0]);
    glMatrixMode(GL_MODELVIEW);
}

void Renderer::resetView3d(int x0, int y0, int w, int h) {
    //createBuffers(w, h);
    setProjectionMatrix(Knew,w,h,zNear,zFar);
  //  printf("%d %d %d %d\n",x0,y0,w,h);
    glViewport(x0, y0, w, h);
}

void Renderer::resetView2d(int x0, int y0, int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0,w,0,h,0.1f,1e6f);
    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
    //glScalef(1,-1,1);
    glViewport(x0,y0,w,h);
}

float *Renderer::getK() {
    return &Knew[0];
}


void Renderer::drawQuad()
{
    float s=1.0f;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-s, -s);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(s, -s);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(s,  s);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-s,  s);
    glEnd();
}

void Renderer::drawBox(const Eigen::Vector3f &origin, const Eigen::Vector3f &dim, const Eigen::Vector3f &color)
{

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadIdentity();
    glLoadMatrixf(&mtx[0]);

    glPushAttrib(GL_LIST_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TRANSFORM_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_DEPTH_BUFFER_BIT);
    glLineWidth(2.0f);
    glDisable(GL_TEXTURE_2D);

    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    glEnable(GL_COLOR_MATERIAL);

    float hx = dim(0)+1e-4f;
    float hy = dim(1)+1e-4f;
    float hz = dim(2)+1e-4f;
    float x0 = origin(0);
    float y0 = origin(1);
    float z0 = origin(2);

    glBegin(GL_QUADS);
    // Front Face
    glColor3f(color(0),color(1),color(2));
    glVertex3f( x0,     y0,  hz+z0);  // Bottom Left Of The Texture and Quad
    glVertex3f( hx+x0,  y0,  hz+z0);  // Bottom Right Of The Texture and Quad
    glVertex3f( hx+x0,  hy+y0,  hz+z0);  // Top Right Of The Texture and Quad
    glVertex3f( x0,     hy+y0,  hz+z0);  // Top Left Of The Texture and Quad
    // Back Face
    glVertex3f( x0,    y0,    z0);  // Bottom Right Of The Texture and Quad
    glVertex3f( x0,    hy+y0, z0);  // Top Right Of The Texture and Quad
    glVertex3f( hx+x0, hy+y0, z0);  // Top Left Of The Texture and Quad
    glVertex3f( hx+x0, y0,    z0);  // Bottom Left Of The Texture and Quad
    // Top Face
    glVertex3f(x0,      hy+y0,  z0);  // Top Left Of The Texture and Quad
    glVertex3f(x0,      hy+y0,  hz+z0);  // Bottom Left Of The Texture and Quad
    glVertex3f( hx+x0,  hy+y0,  hz+z0);  // Bottom Right Of The Texture and Quad
    glVertex3f( hx+x0,  hy+y0,  z0);  // Top Right Of The Texture and Quad
    // Bottom Face
    glVertex3f( x0,     y0, z0);  // Top Right Of The Texture and Quad
    glVertex3f( hx+x0,  y0, z0);  // Top Left Of The Texture and Quad
    glVertex3f( hx+x0,  y0, hz+z0);  // Bottom Left Of The Texture and Quad
    glVertex3f( x0,     y0, hz+z0);  // Bottom Right Of The Texture and Quad
    // Right face
    glVertex3f( hx+x0,  y0,    z0);  // Bottom Right Of The Texture and Quad
    glVertex3f( hx+x0,  hy+y0, z0);  // Top Right Of The Texture and Quad
    glVertex3f( hx+x0,  hy+y0,  hz+z0);  // Top Left Of The Texture and Quad
    glVertex3f( hx+x0,  y0,     hz+z0);  // Bottom Left Of The Texture and Quad
    // Left Face
    glVertex3f(x0,  y0,    z0);  // Bottom Left Of The Texture and Quad
    glVertex3f(x0,  y0,    hz+z0);  // Bottom Right Of The Texture and Quad
    glVertex3f(x0,  hy+y0, hz+z0);  // Top Right Of The Texture and Quad
    glVertex3f(x0,  hy+y0, z0);  // Top Left Of The Texture and Quad
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    glPopAttrib();
    glPopMatrix();
}

void Renderer::renderPose(Eigen::Matrix4f &T, float fovXDeg, float fovYDeg, float aspect, float r, float g, float b, float len) {
	glPushMatrix();
	float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
	glLoadMatrixf(&mtx[0]);
    glPushAttrib(GL_LIST_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TRANSFORM_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_DEPTH_BUFFER_BIT);
    glLineWidth(2.0f);
    glDisable(GL_TEXTURE_2D);
    glBegin(GL_LINES);

	Eigen::Matrix4f iT = T.inverse();

	float o[3]; o[0] =      iT(0,3); o[1] =     iT(1,3); o[2] =      iT(2,3);
	float z[3]; z[0] =  -len*iT(0,2); z[1] = -len*iT(1,2); z[2] = -len*iT(2,2);
    float sX = aspect*tan(3.141592653f*fovXDeg/360.0f)*len;
    float sY = tan(3.141592653f*fovYDeg/360.0f)*len;

	float u[3]; u[0] = sX*iT(0,0); u[1] = sX*iT(1,0); u[2] = sX*iT(2,0);
	float v[3]; v[0] = sY*iT(0,1); v[1] = sY*iT(1,1); v[2] = sY*iT(2,1);

    float x0[3]; x0[0] = o[0]+z[0]-u[0]-v[0]; x0[1] = o[1]+z[1]-u[1]-v[1]; x0[2] = o[2]+z[2]-u[2]-v[2];
    float x1[3]; x1[0] = o[0]+z[0]+u[0]-v[0]; x1[1] = o[1]+z[1]+u[1]-v[1]; x1[2] = o[2]+z[2]+u[2]-v[2];
    float x2[3]; x2[0] = o[0]+z[0]+u[0]+v[0]; x2[1] = o[1]+z[1]+u[1]+v[1]; x2[2] = o[2]+z[2]+u[2]+v[2];
    float x3[3]; x3[0] = o[0]+z[0]-u[0]+v[0]; x3[1] = o[1]+z[1]-u[1]+v[1]; x3[2] = o[2]+z[2]-u[2]+v[2];

    glColor3f(r,g,b);

    glVertex3f(x0[0],x0[1],x0[2]); glVertex3f(x1[0],x1[1],x1[2]);
    glVertex3f(x1[0],x1[1],x1[2]); glVertex3f(x2[0],x2[1],x2[2]);
    glVertex3f(x2[0],x2[1],x2[2]); glVertex3f(x3[0],x3[1],x3[2]);
    glVertex3f(x3[0],x3[1],x3[2]); glVertex3f(x0[0],x0[1],x0[2]);

    glVertex3f(o[0],o[1],o[2]); glVertex3f(x0[0],x0[1],x0[2]);
    glVertex3f(o[0],o[1],o[2]); glVertex3f(x1[0],x1[1],x1[2]);
    glVertex3f(o[0],o[1],o[2]); glVertex3f(x2[0],x2[1],x2[2]);
    glVertex3f(o[0],o[1],o[2]); glVertex3f(x3[0],x3[1],x3[2]);

    glEnd();
    glPopAttrib();
	glPopMatrix();

}

void Renderer::renderPose(float *Traw, float fovY, float aspect, float r, float g, float b, float len) {

    Eigen::Matrix4f T;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            T(j,i) = Traw[i+j*4];
        }
    }

    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadMatrixf(&mtx[0]);
    glPushAttrib(GL_LIST_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TRANSFORM_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_DEPTH_BUFFER_BIT);
    glLineWidth(2.0f);
    glDisable(GL_TEXTURE_2D);
    glBegin(GL_LINES);

    Eigen::Matrix4f iT = T.inverse();

    float o[3]; o[0] =      iT(0,3); o[1] =     iT(1,3); o[2] =      iT(2,3);
    float z[3]; z[0] =   len*iT(0,2); z[1] =  len*iT(1,2); z[2] =  len*iT(2,2);
    float sX = aspect*tan(3.141592653f*fovY/360.0f)*len;
    float sY = tan(3.141592653f*fovY/360.0f)*len;

    float u[3]; u[0] = sX*iT(0,0); u[1] = sX*iT(1,0); u[2] = sX*iT(2,0);
    float v[3]; v[0] = sY*iT(0,1); v[1] = sY*iT(1,1); v[2] = sY*iT(2,1);

    float x0[3]; x0[0] = o[0]+z[0]-u[0]-v[0]; x0[1] = o[1]+z[1]-u[1]-v[1]; x0[2] = o[2]+z[2]-u[2]-v[2];
    float x1[3]; x1[0] = o[0]+z[0]+u[0]-v[0]; x1[1] = o[1]+z[1]+u[1]-v[1]; x1[2] = o[2]+z[2]+u[2]-v[2];
    float x2[3]; x2[0] = o[0]+z[0]+u[0]+v[0]; x2[1] = o[1]+z[1]+u[1]+v[1]; x2[2] = o[2]+z[2]+u[2]+v[2];
    float x3[3]; x3[0] = o[0]+z[0]-u[0]+v[0]; x3[1] = o[1]+z[1]-u[1]+v[1]; x3[2] = o[2]+z[2]-u[2]+v[2];

    glColor3f(r,g,b);

    glVertex3f(x0[0],x0[1],x0[2]); glVertex3f(x1[0],x1[1],x1[2]);
    glVertex3f(x1[0],x1[1],x1[2]); glVertex3f(x2[0],x2[1],x2[2]);
    glVertex3f(x2[0],x2[1],x2[2]); glVertex3f(x3[0],x3[1],x3[2]);
    glVertex3f(x3[0],x3[1],x3[2]); glVertex3f(x0[0],x0[1],x0[2]);

    glVertex3f(o[0],o[1],o[2]); glVertex3f(x0[0],x0[1],x0[2]);
    glVertex3f(o[0],o[1],o[2]); glVertex3f(x1[0],x1[1],x1[2]);
    glVertex3f(o[0],o[1],o[2]); glVertex3f(x2[0],x2[1],x2[2]);
    glVertex3f(o[0],o[1],o[2]); glVertex3f(x3[0],x3[1],x3[2]);

    glEnd();
    glPopAttrib();
    glPopMatrix();

}

void Renderer::drawMesh(unsigned int vbo, int npoints, unsigned ibo, int nfaces3) {
    if (vbo == 0) return;
    glPushAttrib(GL_LIST_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TRANSFORM_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadIdentity();
    glLoadMatrixf(&mtx[0]);

//    glTranslatef(-5,-0.5f,0);

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    // Create light components
    float ambientLight[] = { 0.2f, 0.2f, 0.2f, 1.0f };
    float diffuseLight[] = { 0.8f, 0.8f, 0.8, 1.0f };
    float specularLight[] = { 0.5f, 0.5f, 0.5f, 1.0f };
    float position[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    position[0] = m_cameraPos(0); position[1] = m_cameraPos(1); position[2] = m_cameraPos(2);

    // Assign created components to GL_LIGHT0
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);
    glLightfv(GL_LIGHT0, GL_POSITION, position);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glBindBuffer( GL_ARRAY_BUFFER, vbo );
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_NORMAL_ARRAY );
    float *ptrBase = 0;
    glVertexPointer(3,GL_FLOAT,6*sizeof(float),ptrBase); ptrBase+=3;
    glNormalPointer(GL_FLOAT,6*sizeof(float),ptrBase);

    // Index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    glColor4f(1,1,1,1);
    // glDrawArrays(GL_POINTS,0,npoints);
    // draw 6 quads using offset of index array
   // glDrawElements(GL_QUADS, 24, GL_UNSIGNED_BYTE, 0);
   // glPushMatrix();
    //glScalef(10,10,10);
    glDrawElements(
         GL_TRIANGLES,      // mode
         nfaces3*3,    // count
         GL_UNSIGNED_INT,   // type
         (void*)0           // element array buffer offset
     );
     glPopMatrix();
     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
     glDisableClientState(GL_VERTEX_ARRAY);
     glDisableClientState(GL_NORMAL_ARRAY);
     glBindBuffer( GL_ARRAY_BUFFER, 0);
     glDisable(GL_LIGHTING);
     glPopAttrib();
}
