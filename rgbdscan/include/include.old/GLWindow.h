#pragma once

class GLWindow {
public:
    GLWindow(float x0, float y0, float w, float h, void (*renderFunc)(float x0, float y0, float w, float h));
	~GLWindow();
	void render();
    float x0,y0,w,h;
private:
    void (*renderFunc)(float x0, float y0, float w, float h);
};
