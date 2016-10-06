#pragma once
#include <string.h>

class Keyframe
{
    public:
        Keyframe();
        ~Keyframe();
        void setPose(float *Tin, float fovXin, float aspect);
        void setFov(float fovXin);
        void setCenter(float cxin, float cyin);
        void setLensDistortion(float *kcin);
        float *getPoseGL();
        float *getPose();
        float getFov();
        void setAspectRatio(float aspect);
        float getAspectRatio();
    private:
        float T[16];
        float Tgl[16];
        float cx,cy;
        float kc[5];
        float fovY;
        float aspectRatio;
};
