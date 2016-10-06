/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#ifndef MARCHING_CUBES_INCLUDED
#define MARCHING_CUBES_INCLUDED
#include <vector>
#include "Geometry.h"

class Square{
public:
	const static unsigned int CORNERS=4,EDGES=4,NEIGHBORS=4;

    static int AntipodalCornerIndex(int idx){
        int x,y;
        FactorCornerIndex(idx,x,y);
        return CornerIndex( (x+1)%2 , (y+1)%2 );
    }
    static int CornerIndex(int x,int y){return (y<<1)|x;}
    static void FactorCornerIndex(int idx,int& x,int& y){
        x=(idx>>0)%2;
        y=(idx>>1)%2;
    }
    static int EdgeIndex(int orientation,int i){
        switch(orientation){
            case 0: // x
                if(!i)	{return  0;} // (0,0) -> (1,0)
                else	{return  2;} // (0,1) -> (1,1)
            case 1: // y
                if(!i)	{return  3;} // (0,0) -> (0,1)
                else	{return  1;} // (1,0) -> (1,1)
        };
        return -1;
    }
    static void FactorEdgeIndex(int idx,int& orientation,int& i){
        switch(idx){
            case 0: case 2:
                orientation=0;
                i=idx/2;
                return;
            case 1: case 3:
                orientation=1;
                i=((idx/2)+1)%2;
                return;
        };
    }
    static void EdgeCorners(int idx,int& c1,int& c2){
        int orientation,i;
        FactorEdgeIndex(idx,orientation,i);
        switch(orientation){
            case 0:
                c1=CornerIndex(0,i);
                c2=CornerIndex(1,i);
                break;
            case 1:
                c1=CornerIndex(i,0);
                c2=CornerIndex(i,1);
                break;
        };
    }
    static int ReflectEdgeIndex(int idx,int edgeIndex){
        int orientation=edgeIndex%2;
        int o,i;
        FactorEdgeIndex(idx,o,i);
        if(o!=orientation){return idx;}
        else{return EdgeIndex(o,(i+1)%2);}
    }
    static int ReflectCornerIndex(int idx,int edgeIndex){
        int orientation=edgeIndex%2;
        int x,y;
        FactorCornerIndex(idx,x,y);
        switch(orientation){
            case 0:	return CornerIndex((x+1)%2,y);
            case 1:	return CornerIndex(x,(y+1)%2);
        };
        return -1;
    }
};

class Cube{
public:
	const static unsigned int CORNERS=8,EDGES=12,NEIGHBORS=6;

//	static int  CornerIndex			(int x,int y,int z);
    static void FactorCornerIndex	(int idx,int& x,int& y,int& z) {
        x=(idx>>0)%2;
        y=(idx>>1)%2;
        z=(idx>>2)%2;
    }
    //static int  EdgeIndex			(int orientation,int i,int j);
    static void FactorEdgeIndex		(int idx,int& orientation,int& i,int &j){
        orientation=idx>>2;
        i = (idx&1);
        j = (idx&2)>>1;
    }

    static int CornerIndex(int x,int y,int z){return (z<<2)|(y<<1)|x;}
    static int EdgeIndex(int orientation,int i,int j){return (i | (j<<1))|(orientation<<2);}
    static int FaceIndex(int x,int y,int z){
        if		(x<0)	{return  0;}
        else if	(x>0)	{return  1;}
        else if	(y<0)	{return  2;}
        else if	(y>0)	{return  3;}
        else if	(z<0)	{return  4;}
        else if	(z>0)	{return  5;}
        else			{return -1;}
    }
    static int FaceIndex(int dir,int offSet){return (dir<<1)|offSet;}

    static void FactorFaceIndex(int idx,int& x,int& y,int& z){
        x=y=z=0;
        switch(idx){
            case 0:		x=-1;	break;
            case 1:		x= 1;	break;
            case 2:		y=-1;	break;
            case 3:		y= 1;	break;
            case 4:		z=-1;	break;
            case 5:		z= 1;	break;
        };
    }
    static void FactorFaceIndex(int idx,int& dir,int& offSet){
        dir  = idx>>1;
        offSet=idx &1;
    }

    static int FaceAdjacentToEdges(int eIndex1,int eIndex2){
        int f1,f2,g1,g2;
        FacesAdjacentToEdge(eIndex1,f1,f2);
        FacesAdjacentToEdge(eIndex2,g1,g2);
        if(f1==g1 || f1==g2){return f1;}
        if(f2==g1 || f2==g2){return f2;}
        return -1;
    }

    static void FacesAdjacentToEdge(int eIndex,int& f1Index,int& f2Index){
        int orientation,i1,i2;
        FactorEdgeIndex(eIndex,orientation,i1,i2);
        i1<<=1;
        i2<<=1;
        i1--;
        i2--;
        switch(orientation){
            case 0:
                f1Index=FaceIndex( 0,i1, 0);
                f2Index=FaceIndex( 0, 0,i2);
                break;
            case 1:
                f1Index=FaceIndex(i1, 0, 0);
                f2Index=FaceIndex( 0, 0,i2);
                break;
            case 2:
                f1Index=FaceIndex(i1, 0, 0);
                f2Index=FaceIndex( 0,i2, 0);
                break;
        };
    }
    static void EdgeCorners(int idx,int& c1,int& c2){
        int orientation,i1,i2;
        FactorEdgeIndex(idx,orientation,i1,i2);
        switch(orientation){
            case 0:
                c1=CornerIndex(0,i1,i2);
                c2=CornerIndex(1,i1,i2);
                break;
            case 1:
                c1=CornerIndex(i1,0,i2);
                c2=CornerIndex(i1,1,i2);
                break;
            case 2:
                c1=CornerIndex(i1,i2,0);
                c2=CornerIndex(i1,i2,1);
                break;
        };
    }
    static void FaceCorners(int idx,int& c1,int& c2,int& c3,int& c4){
        int i=idx%2;
        switch(idx/2){
        case 0:
            c1=CornerIndex(i,0,0);
            c2=CornerIndex(i,1,0);
            c3=CornerIndex(i,0,1);
            c4=CornerIndex(i,1,1);
            return;
        case 1:
            c1=CornerIndex(0,i,0);
            c2=CornerIndex(1,i,0);
            c3=CornerIndex(0,i,1);
            c4=CornerIndex(1,i,1);
            return;
        case 2:
            c1=CornerIndex(0,0,i);
            c2=CornerIndex(1,0,i);
            c3=CornerIndex(0,1,i);
            c4=CornerIndex(1,1,i);
            return;
        }
    }
    static int AntipodalCornerIndex(int idx){
        int x,y,z;
        FactorCornerIndex(idx,x,y,z);
        return CornerIndex((x+1)%2,(y+1)%2,(z+1)%2);
    }
    static int FaceReflectFaceIndex(int idx,int faceIndex){
        if(idx/2!=faceIndex/2){return idx;}
        else{
            if(idx%2)	{return idx-1;}
            else		{return idx+1;}
        }
    }
    static int FaceReflectEdgeIndex(int idx,int faceIndex){
        int orientation=faceIndex/2;
        int o,i,j;
        FactorEdgeIndex(idx,o,i,j);
        if(o==orientation){return idx;}
        switch(orientation){
            case 0:	return EdgeIndex(o,(i+1)%2,j);
            case 1:
                switch(o){
                    case 0:	return EdgeIndex(o,(i+1)%2,j);
                    case 2:	return EdgeIndex(o,i,(j+1)%2);
                };
            case 2:	return EdgeIndex(o,i,(j+1)%2);
        };
        return -1;
    }
    static int FaceReflectCornerIndex(int idx,int faceIndex){
        int orientation=faceIndex/2;
        int x,y,z;
        FactorCornerIndex(idx,x,y,z);
        switch(orientation){
            case 0:	return CornerIndex((x+1)%2,y,z);
            case 1:	return CornerIndex(x,(y+1)%2,z);
            case 2: return CornerIndex(x,y,(z+1)%2);
        };
        return -1;
    }
    static int EdgeReflectCornerIndex(int idx,int edgeIndex){
        int orientation,x,y,z;
        FactorEdgeIndex(edgeIndex,orientation,x,y);
        FactorCornerIndex(idx,x,y,z);
        switch(orientation){
            case 0:	return CornerIndex( x     ,(y+1)%2,(z+1)%2);
            case 1:	return CornerIndex((x+1)%2, y     ,(z+1)%2);
            case 2:	return CornerIndex((x+1)%2,(y+1)%2, z     );
        };
        return -1;
    }
    static int EdgeReflectEdgeIndex(int edgeIndex){
        int o,i1,i2;
        FactorEdgeIndex(edgeIndex,o,i1,i2);
        return Cube::EdgeIndex(o,(i1+1)%2,(i2+1)%2);
    }
};

class MarchingSquares{
	static double Interpolate(double v1,double v2);
	static void SetVertex(int e,const double values[Square::CORNERS],double iso);
public:
	const static unsigned int MAX_EDGES=2;
	static const int edgeMask[1<<Square::CORNERS];
	static const int edges[1<<Square::CORNERS][2*MAX_EDGES+1];
	static double vertexList[Square::EDGES][2];

	static int GetIndex(const double values[Square::CORNERS],double iso);
	static int IsAmbiguous(const double v[Square::CORNERS],double isoValue);
	static int AddEdges(const double v[Square::CORNERS],double isoValue,Edge* edges);
	static int AddEdgeIndices(const double v[Square::CORNERS],double isoValue,int* edges);
};

class MarchingCubes
{
	static void SetVertex(int e,const double values[Cube::CORNERS],double iso);
	static int GetFaceIndex(const double values[Cube::CORNERS],double iso,int faceIndex);

	static void SetVertex(int e,const float values[Cube::CORNERS],float iso);
	static int GetFaceIndex(const float values[Cube::CORNERS],float iso,int faceIndex);

	static int GetFaceIndex(int mcIndex,int faceIndex);
public:
	static double Interpolate(double v1,double v2);
	static float Interpolate(float v1,float v2);
	const static unsigned int MAX_TRIANGLES=5;
	static const int edgeMask[1<<Cube::CORNERS];
	static const int triangles[1<<Cube::CORNERS][3*MAX_TRIANGLES+1];
	static const int cornerMap[Cube::CORNERS];
	static double vertexList[Cube::EDGES][3];

	static int AddTriangleIndices(int mcIndex,int* triangles);

	static int GetIndex(const double values[Cube::CORNERS],double iso);
	static int IsAmbiguous(const double v[Cube::CORNERS],double isoValue,int faceIndex);
	static int HasRoots(const double v[Cube::CORNERS],double isoValue);
	static int HasRoots(const double v[Cube::CORNERS],double isoValue,int faceIndex);
	static int AddTriangles(const double v[Cube::CORNERS],double isoValue,Triangle* triangles);
	static int AddTriangleIndices(const double v[Cube::CORNERS],double isoValue,int* triangles);

	static int GetIndex(const float values[Cube::CORNERS],float iso);
	static int IsAmbiguous(const float v[Cube::CORNERS],float isoValue,int faceIndex);
	static int HasRoots(const float v[Cube::CORNERS],float isoValue);
	static int HasRoots(const float v[Cube::CORNERS],float isoValue,int faceIndex);
	static int AddTriangles(const float v[Cube::CORNERS],float isoValue,Triangle* triangles);
	static int AddTriangleIndices(const float v[Cube::CORNERS],float isoValue,int* triangles);

	static int IsAmbiguous(int mcIndex,int faceIndex);
	static int HasRoots(int mcIndex);
	static int HasFaceRoots(int mcIndex,int faceIndex);
	static int HasEdgeRoots(int mcIndex,int edgeIndex);
};
#endif //MARCHING_CUBES_INCLUDED
