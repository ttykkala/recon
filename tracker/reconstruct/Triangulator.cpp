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
#include "Triangulator.h"
#include "basic_math.h"
#include <rendering/VertexBuffer.h>

Triangulator::Triangulator(double *K1, double *T1, double *K2, double *T2 )
{
	for (int i = 0; i < NUM_VBUFFERS; i++) vbuffer[i] = NULL;
	memcpy(this->K1,K1,sizeof(double)*9);
	memcpy(this->K2,K2,sizeof(double)*9);
	for (int j = 0; j < 3; j++)
		for (int i = 0; i < 4; i++) {
			this->T1[i+j*4] = T1[i+j*4];
			this->T2[i+j*4] = T2[i+j*4];
		}
}


void Triangulator::generateRay(double *K, double *cameraMatrix, double x, double y, Wml::Ray3<double> &ray)
{
	float originX = float(cameraMatrix[3]);
	float originY = float(cameraMatrix[7]);
	float originZ = float(cameraMatrix[11]);
	ray.SetOrigin(originX,originY,originZ);

	double P[12];
	projectionMatrix(K,cameraMatrix,P);

	// compute pseudo-inverse
	double invP[12];
	pseudoInverse3x4(P, invP);

	double dir[4];
	dir[0] = 0; dir[1] = 0; dir[2] = 0; dir[3] = 1;
	// generate homogeneous point
	double p[3]; p[0] = x; p[1] = y; p[2] = 1;
	transformVector(invP, p, 4, 3, dir);
	if (dir[3]*dir[3]>0) {
		dir[0] /= dir[3];
		dir[1] /= dir[3];
		dir[2] /= dir[3];
	} /*else {
		dir[0] = -dir[0];
		dir[1] = -dir[1];
		dir[2] = -dir[2];
	}*/

	dir[0] -= originX;
	dir[1] -= originY;
	dir[2] -= originZ;
	normalize(dir);
	/*if (dir[2] > 0) {
		dir[0] = -dir[0]; dir[1] = -dir[1]; dir[2] = -dir[2];
	}*/
	ray.SetDirection(dir[0],dir[1],dir[2]);
}

void Triangulator::intersectRays(Wml::Ray3<double> &ray1,Wml::Ray3<double> &ray2, double *res)
{
	double a[3],b[3],c[3],d[3],s,t;
	double sample1[3],sample2[3];

	a[0] = ray1.Origin().X(); a[1] = ray1.Origin().Y(); a[2] = ray1.Origin().Z();
	b[0] = ray1.Direction().X(); b[1] = ray1.Direction().Y(); b[2] = ray1.Direction().Z();
	c[0] = ray2.Origin().X(); c[1] = ray2.Origin().Y(); c[2] = ray2.Origin().Z();
	d[0] = ray2.Direction().X(); d[1] = ray2.Direction().Y(); d[2] = ray2.Direction().Z();

	normalize(b);
	normalize(d);

	t = ((-dot3(a,b)+dot3(b,c))*dot3(b,d)+dot3(a,d)-dot3(c,d))/(1-dot3(b,d)*dot3(b,d));
	s = (dot3(b,d)*t-dot3(a,b)+dot3(b,c));
/* TODO sometimes t < 0 and s < 0 are valid!
	if (t <= 0 || s <= 0) {
		res[0] = 0;
		res[1] = 0;
		res[2] = 0;
		return;
	}*/
	//assert(t > 0 && s > 0);

	sample1[0] = a[0]+s*b[0];
	sample1[1] = a[1]+s*b[1];
	sample1[2] = a[2]+s*b[2];

	sample2[0] = c[0]+t*d[0];
	sample2[1] = c[1]+t*d[1];
	sample2[2] = c[2]+t*d[2];

	res[0] = (sample1[0]+sample2[0])/2.0f;
	res[1] = (sample1[1]+sample2[1])/2.0f;
	res[2] = (sample1[2]+sample2[2])/2.0f;

	return;
}

void Triangulator::triangulateSingle(double x1, double y1, double x2, double y2, double *x, double *y, double *z)
{
	double xxx[3];
	Wml::Ray3<double> ray1;
	generateRay(K1,T1,x1,y1,ray1);
	Wml::Ray3<double> ray2;
	generateRay(K2,T2,x2,y2,ray2);
	intersectRays(ray1,ray2,xxx); *x = xxx[0]; *y = xxx[1]; *z = xxx[2];
}

void Triangulator::triangulate( int activeBuffer, double x1, double y1, double x2, double y2, unsigned char colorR,unsigned char colorG,unsigned char colorB,unsigned char colorR2,unsigned char colorG2,unsigned char colorB2)
{
	assert(vbuffer[activeBuffer] != NULL);
	Wml::Ray3<double> ray1;
	generateRay(K1,T1,x1,y1,ray1);
	Wml::Ray3<double> ray2;
	generateRay(K2,T2,x2,y2,ray2);
	double xyz[3];
	intersectRays(ray1,ray2,xyz);
	vbuffer[activeBuffer]->addVertex((float)xyz[0],(float)xyz[1],(float)xyz[2],colorR,colorG,colorB,colorR2,colorG2,colorB2);
}

Triangulator::~Triangulator()
{
}
