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
#include <vector>
#include <tinyxml.h>

using namespace std;

class RectangleElement {
private:
	float vertices[3*4];
	int   indices[4];
	float angleX;
	float angleY;
	float scaleX;
	float scaleY;
	float x;
	float y;
	float z;
	float color[4];
	float borderColor[4];
public:
	RectangleElement() { indices[0] = 0; indices[1] = 1; indices[2] = 2; indices[3] = 3; scaleX = 1; scaleY = 1; angleX = 0; angleY = 0; x = 0; y = 0; z = 0;}
	void setVertex0(float x, float y, float z) { vertices[0*3+0] = x; vertices[0*3+1] = y; vertices[0*3+2] = z; }
	void setVertex1(float x, float y, float z) { vertices[1*3+0] = x; vertices[1*3+1] = y; vertices[1*3+2] = z; }
	void setVertex2(float x, float y, float z) { vertices[2*3+0] = x; vertices[2*3+1] = y; vertices[2*3+2] = z; }
	void setVertex3(float x, float y, float z) { vertices[3*3+0] = x; vertices[3*3+1] = y; vertices[3*3+2] = z; }
	void setAngleX(float x) { angleX = x; }
	void setAngleY(float y) { angleY = y; }
	void updateScaleX(float x) { scaleX += x; if (scaleX < 1e-7f) scaleX = 1e-7f; }
	void updateScaleY(float y) { scaleY += y; if (scaleY < 1e-7f) scaleY = 1e-7f; } //if (y < 1e-7f) y = 1e-7f; scaleY = baseScaleY*y; }
	float getScaleX() { return scaleX; }
	float getScaleY() { return scaleY; }
	void setScaleX(float x) { scaleX = x; }
	void setScaleY(float y) { scaleY = y; }
	void setPosition(float x, float y, float z) { this->x = x; this->y = y; this->z = z; }
	void setColor(float r, float g, float b, float a) { color[0] = r; color[1] = g; color[2] = b; color[3] = a;}
	void setBorderColor(float r, float g, float b, float a) { borderColor[0] = r; borderColor[1] = g; borderColor[2] = b; borderColor[3] = a;}
	float *getVertex0() { return &vertices[indices[0]*3]; }
	float *getVertex1() { return &vertices[indices[1]*3]; }
	float *getVertex2() { return &vertices[indices[2]*3]; }
	float *getVertex3() { return &vertices[indices[3]*3]; }
	float *getVertices() { return &vertices[0]; }
	int   *getIndices() { return &indices[0]; }
	float *getColor() { return &color[0];}
	float *getBorderColor() { return &borderColor[0];}
	float getX() { return x; }
	float getY() { return y; }
	float getZ() { return z; }
	float getAngleX() { return angleX; }
	float getAngleY() { return angleY; }
	void updateAngleX(float inc) { angleX += inc; }
	void updateAngleY(float inc) { angleY += inc; }
	void updateX(float inc) { x += inc; }
	void updateY(float inc) { y += inc; }
	void updateZ(float inc) { z += inc; }
};

class PolygonModel {
private:
	vector<RectangleElement> m_rectangles;
	int selectedID;
public:
	void addRectangle(RectangleElement r) {
		r.setColor(0,1,0,0.5f);
		m_rectangles.push_back(r);
	}
	PolygonModel() {
		selectedID = -1;
	}
	~PolygonModel() {
		saveXML();
		savePLY("cadModel.ply");
	}
	void selectNext() {
		if (selectedID == -1) selectedID = 0;
		else selectedID = (selectedID+1)%m_rectangles.size();
	}
	void selectPrev() {
		if (selectedID == -1) selectedID = 0;
		else selectedID = (selectedID+m_rectangles.size()-1)%m_rectangles.size();
	}
	void drawRectangle(RectangleElement &rectum, bool selected = true) {
		glPushMatrix();
		glDisable(GL_TEXTURE_2D);
		glLineWidth(2.0f);
		glEnable(GL_COLOR_MATERIAL);
		glColor4fv(rectum.getColor());
		glTranslatef(rectum.getX(),rectum.getY(),rectum.getZ());
		glRotatef(rectum.getAngleX(),1,0,0);
		glRotatef(rectum.getAngleY(),0,1,0);
		glScalef(rectum.getScaleX(),rectum.getScaleY(),1);	
		if (selected) {
			glBegin(GL_QUADS);
			glVertex3fv(rectum.getVertex0());
			glVertex3fv(rectum.getVertex1());
			glVertex3fv(rectum.getVertex2());
			glVertex3fv(rectum.getVertex3());
			glEnd();
		}

		if (selected) glColor4f(1,1,1,1);
		else glColor4fv(rectum.getBorderColor());
		glBegin(GL_LINES);
		glVertex3fv(rectum.getVertex0()); glVertex3fv(rectum.getVertex1());
		glVertex3fv(rectum.getVertex1()); glVertex3fv(rectum.getVertex2());
		glVertex3fv(rectum.getVertex2()); glVertex3fv(rectum.getVertex3());
		glVertex3fv(rectum.getVertex3()); glVertex3fv(rectum.getVertex0());
		glEnd();
		glEnable(GL_TEXTURE_2D);
		glPopMatrix();
	}
	bool isSelected() {
		if (selectedID <= -1) return false;
		return true;
	}
	RectangleElement *getSelected() {
		return &m_rectangles[selectedID];
	}
	void selectNone() {
		selectedID = -1;
	}
	void deleteSelected() {
		if (selectedID == -1) return;
		vector<RectangleElement>::iterator i;
		int index = 0;
		for (i = m_rectangles.begin(); i != m_rectangles.end(); i++,index++) {
			if (index == selectedID) {
				m_rectangles.erase(i);
				return;				
			}
		}
		if (selectedID >= m_rectangles.size()) selectedID--;
		selectNext();
	}

	void drawRectangles() {
		vector<RectangleElement>::iterator i;
		int index = 0;
		for (i = m_rectangles.begin(); i != m_rectangles.end(); i++,index++) {
			bool selected = false;
			if (index == selectedID) selected = true;
			drawRectangle(*i,selected); 
		}
	}
	void saveRectangle(RectangleElement &rectum, TiXmlElement *node) 
	{
		float *color = rectum.getColor();
		float *borderColor = rectum.getBorderColor();
		float angleX = rectum.getAngleX();
		float angleY = rectum.getAngleY();
		float sX = rectum.getScaleX();
		float sY = rectum.getScaleY();
		float tX = rectum.getX();
		float tY = rectum.getY();
		float tZ = rectum.getZ();

		TiXmlElement *vNode = NULL;
		float *v = NULL;

		vNode = new TiXmlElement( "Vertex" );  
		v = rectum.getVertex0(); 
		vNode->SetAttribute("id",0);
		vNode->SetDoubleAttribute("x",v[0]);
		vNode->SetDoubleAttribute("y",v[1]);
		vNode->SetDoubleAttribute("z",v[2]);
		node->LinkEndChild( vNode );  

		vNode = new TiXmlElement( "Vertex" );  
		vNode->SetAttribute("id",1);
		v = rectum.getVertex1(); 
		vNode->SetDoubleAttribute("x",v[0]);
		vNode->SetDoubleAttribute("y",v[1]);
		vNode->SetDoubleAttribute("z",v[2]);
		node->LinkEndChild( vNode );  

		vNode = new TiXmlElement( "Vertex" );  
		vNode->SetAttribute("id",2);
		v = rectum.getVertex2(); 
		vNode->SetDoubleAttribute("x",v[0]);
		vNode->SetDoubleAttribute("y",v[1]);
		vNode->SetDoubleAttribute("z",v[2]);
		node->LinkEndChild( vNode );  

		vNode = new TiXmlElement( "Vertex" );  
		vNode->SetAttribute("id",3);
		v = rectum.getVertex3(); 
		vNode->SetDoubleAttribute("x",v[0]);
		vNode->SetDoubleAttribute("y",v[1]);
		vNode->SetDoubleAttribute("z",v[2]);
		node->LinkEndChild( vNode );		
			
		node->SetDoubleAttribute("scaleX", sX);
		node->SetDoubleAttribute("scaleY", sY);
		node->SetDoubleAttribute("angleX", angleX);
		node->SetDoubleAttribute("angleY", angleY);
	 	node->SetDoubleAttribute("tX", tX);
		node->SetDoubleAttribute("tY", tY);
		node->SetDoubleAttribute("tZ", tZ);
		node->SetDoubleAttribute("colorR",color[0]);
		node->SetDoubleAttribute("colorG",color[1]);
		node->SetDoubleAttribute("colorB",color[2]);	
		node->SetDoubleAttribute("colorA",color[3]);

		node->SetDoubleAttribute("borderColorR",borderColor[0]);
		node->SetDoubleAttribute("borderColorG",borderColor[1]);
		node->SetDoubleAttribute("borderColorB",borderColor[2]);	
		node->SetDoubleAttribute("borderColorA",borderColor[3]);
	}

	void loadRectangle(TiXmlElement *pElem, RectangleElement &rectum) {
		double scaleX,scaleY;
		double tX,tY,tZ;
		double angleX,angleY;
		double colorR,colorG,colorB,colorA;
		
		pElem->QueryDoubleAttribute("scaleX",&scaleX);
		pElem->QueryDoubleAttribute("scaleY",&scaleY);
		pElem->QueryDoubleAttribute("angleX",&angleX);
		pElem->QueryDoubleAttribute("angleY",&angleY);
		pElem->QueryDoubleAttribute("tX",&tX);
		pElem->QueryDoubleAttribute("tY",&tY);
		pElem->QueryDoubleAttribute("tZ",&tZ);
		pElem->QueryDoubleAttribute("colorR",&colorR);
		pElem->QueryDoubleAttribute("colorG",&colorG);
		pElem->QueryDoubleAttribute("colorB",&colorB);
		pElem->QueryDoubleAttribute("colorA",&colorA);
		rectum.setColor(colorR,colorG,colorB,colorA);
		pElem->QueryDoubleAttribute("borderColorR",&colorR);
		pElem->QueryDoubleAttribute("borderColorG",&colorG);
		pElem->QueryDoubleAttribute("borderColorB",&colorB);
		pElem->QueryDoubleAttribute("borderColorA",&colorA);
		rectum.setBorderColor(colorR,colorG,colorB,colorA);
		rectum.setPosition(tX,tY,tZ);
		rectum.setAngleX(angleX);
		rectum.setAngleY(angleY);
		rectum.setScaleX(scaleX);
		rectum.setScaleY(scaleY);

		TiXmlElement *vElem=pElem->FirstChildElement("Vertex");	
		float *vdata = rectum.getVertices(); int *indices = rectum.getIndices();	
		int index = 0;
		while (vElem != NULL) {
			double val;
			vElem->QueryDoubleAttribute("x",&val); vdata[index*3+0] = val;
			vElem->QueryDoubleAttribute("y",&val); vdata[index*3+1] = val;
			vElem->QueryDoubleAttribute("z",&val); vdata[index*3+2] = val;
			indices[index] = index;
			vElem=vElem->NextSiblingElement();
			index++;
		} 


	}

	void loadXML(const char *pFileName) {
		TiXmlDocument doc(pFileName);
		if (!doc.LoadFile()) return;

		TiXmlHandle hDoc(&doc);
		TiXmlElement* pElem;
		TiXmlHandle hRoot(0);
		std::string name;
		// block: name
		{
			pElem=hDoc.FirstChildElement().Element();
			// should always have a valid root but handle gracefully if it does
			if (!pElem) return;
			name=pElem->Value();
			// save this for later
			hRoot=TiXmlHandle(pElem);
		}
		m_rectangles.clear();
		pElem=hRoot.FirstChild("Rectangle").Element();
			
		while (pElem != NULL) {
			RectangleElement r;
			loadRectangle(pElem,r);
			m_rectangles.push_back(r); 
			pElem=pElem->NextSiblingElement();
		} 
	}
	void savePLY(const char *fn) {
		FILE *f = fopen(fn,"wb");
		fprintf(f,"ply\n");
		fprintf(f,"format ascii 1.0\n");
		fprintf(f,"element vertex %d\n",4*m_rectangles.size());
		fprintf(f,"property float x\n");
		fprintf(f,"property float y\n");
		fprintf(f,"property float z\n");
		fprintf(f,"element face %d\n",m_rectangles.size()*4);
		fprintf(f,"property list uchar uint vertex_indices\n");
		fprintf(f,"end_header\n");
		vector<RectangleElement>::iterator i;
		for (i = m_rectangles.begin(); i != m_rectangles.end(); i++) {
			RectangleElement &rectum = *i;
			float *v0 = rectum.getVertex0();
			float *v1 = rectum.getVertex1();
			float *v2 = rectum.getVertex2();
			float *v3 = rectum.getVertex3();
			fprintf(f,"%f %f %f\n",v0[0],v0[1],v0[2]);
			fprintf(f,"%f %f %f\n",v1[0],v1[1],v1[2]);
			fprintf(f,"%f %f %f\n",v2[0],v2[1],v2[2]);
			fprintf(f,"%f %f %f\n",v3[0],v3[1],v3[2]);
		}
		int baseIndex = 0;
		for (i = m_rectangles.begin(); i != m_rectangles.end(); i++) {
			fprintf(f,"3 %d %d %d\n",baseIndex+0,baseIndex+1,baseIndex+2);
			fprintf(f,"3 %d %d %d\n",baseIndex+0,baseIndex+2,baseIndex+3);
			fprintf(f,"3 %d %d %d\n",baseIndex+2,baseIndex+1,baseIndex+0);
			fprintf(f,"3 %d %d %d\n",baseIndex+3,baseIndex+2,baseIndex+0);
			baseIndex+=4;		
		}
		fclose(f);
	}
	void saveXML() {
		TiXmlDocument doc;  
		TiXmlElement* msg;
	 	TiXmlDeclaration* decl = new TiXmlDeclaration( "1.0", "", "" );			
		doc.LinkEndChild( decl );  
 
		TiXmlElement * root = new TiXmlElement( "CADModel" );  
		doc.LinkEndChild( root );  
 
		vector<RectangleElement>::iterator i;
		for (i = m_rectangles.begin(); i != m_rectangles.end(); i++) {
			msg = new TiXmlElement( "Rectangle" );  
			saveRectangle(*i,msg); 
			root->LinkEndChild( msg );  
		} 
		doc.SaveFile( "cadModel.xml" );  
	}
};
