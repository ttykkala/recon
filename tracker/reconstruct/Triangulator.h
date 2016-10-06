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

#define __TRIANGULATOR_H__

#include "MagicSoftware/WmlRay3.h"
#define NUM_VBUFFERS 32

class VertexBuffer;
//#define double double

class Triangulator 
{
private:
	double K1[9];  //K1
	double K2[9];  //K2
	double T1[12]; //RT
	double T2[12]; //RT
	void generateRay(double *K, double *cameraMatrix, double x, double y, Wml::Ray3<double> &ray);
	void intersectRays(Wml::Ray3<double> &ray1,Wml::Ray3<double> &ray2, double *res);
	VertexBuffer *vbuffer[NUM_VBUFFERS];
public:
	Triangulator(double *K1, double *T1, double *K2, double *T2);
	~Triangulator();
	void setVertexBuffer(int slot, VertexBuffer *vbuffer) { this->vbuffer[slot] = vbuffer; }
	void triangulate(int slot,double x1, double y1, double x2, double y2,unsigned char colorR = 255,unsigned char colorG = 255,unsigned char colorB = 255,unsigned char colorR2 = 255,unsigned char colorG2 = 255,unsigned char colorB2 = 255);
	void triangulateSingle(double x1, double y1, double x2, double y2, double *x, double *y, double *z);
};
