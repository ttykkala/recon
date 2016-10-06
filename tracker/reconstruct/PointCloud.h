/********************************************************************
	CNRS Copyright (C) 2011.
	This software is owned by CNRS. No unauthorized distribution of the code is allowed.
	
	Address:
	CNRS/I3S
	2000 Route des Lucioles
	06903 Sophia-Antipolis
	
	created:	15:3:2011   20:03
	filename: 	d:\projects\phd-project\kyklooppi\realtime\utils\reconstruct\PointCloud.h
	author:		Tommi Tykkala
	purpose:	Point cloud for each mipmap layer
*********************************************************************/
#pragma once

class PointCloud {
public:
	bool differentialsComputed;
	int nSamplePoints;
	int nSupportPoints;
	int nMaxPoints;
	// main memory pointers
	float *pointsR0; float *supportR0;
	float *pointsL0; float *supportL0;
	float *pointsR1; float *supportR1;
	float *pointsL1; float *supportL1;
	// cuda device pointers
	float *pointsR0Dev; float *supportR0Dev;
	float *pointsL0Dev; float *supportL0Dev;
	float *pointsR1Dev; float *supportR1Dev; 
	float *pointsL1Dev; float *supportL1Dev;
	// main memory differentials
	float *pointsR0_dx1; 
	float *pointsR0_dx2;
	float *pointsR0_dx3;
	float *pointsR0_dx4;
	float *pointsR0_dx5;
	float *pointsR0_dx6;
	float *pointsR0_dxd;

	PointCloud(int numMaxPoints);
	~PointCloud();
	void addRefPoint(float xR, float yR, float xL, float yL);
	void addRefPointSupport(float xR, float yR, float xL, float yL);
	void reset();
	void updateRefDevice();
};
