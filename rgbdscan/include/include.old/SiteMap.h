#pragma once

#include <vector>
#include <map>
#include <Eigen/Geometry>
//#include <Eigen/StdVector>

class Site {
private:
public:
	Site(float xx, float yy, float ww, float hh, int fframe);
    float x,y;
    float w,h;
    int frame;
    float videoAngleSpeedX;
    float videoAngleSpeedY;
	float cameraPos[3];
	float cameraRot[3];
	float cameraPosLag[3];
	float cameraRotLag[3];
	float modelView[16];	
};

typedef struct {
	float x,y,z;
} TNODE;

class SiteMap {
public:
    SiteMap();
    ~SiteMap();
    void getTrajectoryPose(float videoFrame, Eigen::Vector2f &trajPos, Eigen::Vector2f &trajOrigin,float &coneAngle, float &coneAngleBase);
    int init(const char *filename, int videoFrameCount, int imageWidth, int imageHeight, int iconWidth, int iconHeight);
    std::map< std::string, std::vector<TNODE> >trajectories;
	std::vector<Site*> sites;
    std::string activeTrajectory;
    unsigned int getFirstFrame();
    unsigned int getLastFrame();
    unsigned int getFrameCount();
    void reverseDirection(float &videoFrame);

 };
