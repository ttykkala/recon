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


#include <tracker/basic_math.h>
#include <calib/calib.h>
#include <tinyxml.h>
#include <opencv2/opencv.hpp>
using namespace cv;

void parseDoubles(const char *str, int nDoubles, double *data)
{
	int stringIndex = 0;
	int arrayIndex = 0;

	int nRest = nDoubles;
	while (nRest>0) {
		double val = atof(&str[stringIndex]);
		data[arrayIndex] = val;
		if (nRest>1)
			while(str[stringIndex] != ' ') { stringIndex++;}
		stringIndex++;
		arrayIndex++;
		nRest--;
	}
}

float Calibration::getFovX_R() {
    float f = fabs(KR[0]);
    // note: principal point is assumed to be in the middle of the screen!
    return 2.0f*rad2deg(atan(0.5f/f));
}

float Calibration::getFovY_R() {
    float f = fabs(KR[4]);
    float aspect = (3.0f/4.0f);
    return 2.0f*rad2deg(atan(0.5f*aspect/f));
}

float Calibration::getFovX_L() {
    float f = fabs(KL[0]);
    // note: principal point is assumed to be in the middle of the screen!
    return 2.0f*rad2deg(atan(0.5f/f));
}


float Calibration::getFovY_L() {
    float f = fabs(KL[4]);
    float aspect = (3.0f/4.0f);
    return 2.0f*rad2deg(atan(0.5f*aspect/f));
}


void transpose(double *P, double *M, int nRows, int nCols)
{
	for (int j = 0; j < nCols; j++)
		for (int i = 0; i < nRows; i++)
			M[i+j*nRows] = P[j+i*nCols];
}

void Calibration::setupCalibDataBuffer(int width, int height) {
    this->width = width;
    this->height = height;
    identity3x3(&calibData[KR_OFFSET]);
    for (int i = 0; i < 6; i++) calibData[KR_OFFSET+i] = (float)(width*KR[i]);
    inverse3x3(&calibData[KR_OFFSET],&calibData[iKR_OFFSET]); // precompute inverse
    for (int i = 0; i < 8; i++) calibData[KcR_OFFSET+i] = (float)kcR[i];
    for (int i = 0; i < 16; i++) {calibData[TLR_OFFSET+i] = (float)TLR[i]; calibData[TRL_OFFSET+i] = (float)TRL[i]; }

    identity3x3(&calibData[KL_OFFSET]);
    for (int i = 0; i < 6; i++) calibData[KL_OFFSET+i] = (float)(width*KL[i]);
    inverse3x3(&calibData[KL_OFFSET],&calibData[iKL_OFFSET]); // precompute inverse

    calibData[C0_OFFSET] = (float)c0;
    calibData[C1_OFFSET] = (float)c1;
    //calibData[ALPHA0_OFFSET] = (float)alpha0;
    //calibData[ALPHA1_OFFSET] = (float)alpha1;
    calibData[MIND_OFFSET] = (float)minDist;
    calibData[MAXD_OFFSET] = (float)maxDist;
}

void Calibration::copyCalib(Calibration &extCalib) {

	memcpy(&KL[0],extCalib.getKL(),sizeof(double)*9);
	memcpy(&KR[0],extCalib.getKR(),sizeof(double)*9);
	memcpy(&TLR[0],extCalib.getTLR(),sizeof(double)*16);
    invertRT4(TLR,TRL);
    memcpy(&kcR[0],extCalib.getKcR(),sizeof(double)*8);
    memcpy(&kcL[0],extCalib.getKcL(),sizeof(double)*8);

    if (calibData == NULL) {
        calibData = new float[CALIB_SIZE];
        //beta = &calibData[BETA_OFFSET];
    }    
	//memcpy(&calibData[0],extCalib->getCalibData(),sizeof(float)*CALIB_SIZE);
	c0 = extCalib.getC0();
	c1 = extCalib.getC1();
//	alpha0 = extCalib.getAlpha0();
//	alpha1 = extCalib.getAlpha1();
	maxDist = extCalib.getMaxDist();
	minDist = extCalib.getMinDist();
	width = extCalib.width;
	height = extCalib.height;
	useXYOffset = extCalib.isOffsetXY();
    setupCalibDataBuffer(width,height);
}
/*
int Calibration::initOulu(const char *fileName, bool silent) {
    if (!silent) printf("loading %s\n",fileName);

    const int COLOR_CAMERA_IDX = 0;
    std::stringstream s;
    cv::FileStorage fs;
    cv::Mat m;

    fs.open(fileName,cv::FileStorage::READ);

    int width=0,height=0;
    s << "rsize" << (COLOR_CAMERA_IDX+1);
    fs[s.str().c_str()] >> m;
    height = (int)m.at<float>(0);
    width = (int)m.at<float>(1);

    printf("yml reso: %d x %d\n",width,height);

    cv::Matx33f rK;
    s.str("");
    s << "rK" << (COLOR_CAMERA_IDX+1);
    fs[s.str().c_str()] >> m;
    rK = m;

    KR[6] = 0; KR[7] = 0; KR[8] = 1;
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 3; i++) {
            KR[i+j*3] = rK(j,i)/float(width);
        }
    } KR[0] = -KR[0];
    if (!silent) dumpMatrix("KR",KR,3,3);
    //printf("%f %f %f\n%f %f %f\n%f %f %f\n", rK(0,0),rK(0,1),rK(0,2),rK(1,0),rK(1,1),rK(1,2),rK(2,0),rK(2,1),rK(2,2));

    float rkc[5];
    s.str("");
    s << "rkc" << (COLOR_CAMERA_IDX+1);
    fs[s.str().c_str()] >> m;
    memcpy(rkc,m.data,sizeof(rkc)); for (int i = 0; i < 5; i++) kcR[i] = (double)rkc[i];
    if (!silent) printf("%e %e %e %e %e\n",kcR[0],kcR[1],kcR[2],kcR[3],kcR[4]);

    float color_error_var = 0.0f;
    fs["color_error_var"] >> m;
    color_error_var = m.at<float>(0,COLOR_CAMERA_IDX);
    if (!silent) printf("color_error_var: %f\n",color_error_var);

    cv::Matx33f dK;
    fs["dK"] >> m;
    dK = m;
    KL[6] = 0; KL[7] = 0; KL[8] = 1;
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 3; i++) {
            KL[i+j*3] = dK(j,i)/float(width);
        }
    } KL[0] = -KL[0];
    if (!silent) dumpMatrix("KL",KL,3,3);

    float dkc[5]; float dkcMag = 0.0f;
    fs["dkc"] >> m;
    memcpy(dkc,m.data,sizeof(dkc)); for (int i = 0; i < 5; i++) { kcL[i] = (double)dkc[i]; dkcMag += fabs(dkc[i]); }
    if (!silent) printf("%e %e %e %e %e\n",kcL[0],kcL[1],kcL[2],kcL[3],kcL[4]);
    if (dkcMag > 0.0f) {
        printf("WARNING! IR image lens distotion coeffs set! NOT SUPPORTED! \n");
    }


    cv::Matx33f dR;
    fs["dR"] >> m;
    dR = m;
    identity4x4(TLR);
    TLR[0] = dR(0,0); TLR[1] = dR(0,1); TLR[2]  = dR(0,2);
    TLR[4] = dR(1,0); TLR[5] = dR(1,1); TLR[6]  = dR(2,1);
    TLR[8] = dR(2,0); TLR[9] = dR(2,1); TLR[10] = dR(2,2);

    cv::Matx31f dT;
    fs["dt"] >> m;
    dT = m;
    TLR[3]  = dT(0)*1000.0f;
    TLR[7]  = dT(1)*1000.0f;
    TLR[11] = dT(2)*1000.0f;

    if (!silent) dumpMatrix("TLR",TLR,4,4);
    invertRT4(TLR,TRL);

    float dc[2];
    fs["dc"] >> m;
    memcpy(dc,m.data,sizeof(dc)); c0 = double(dc[0])/1000.0f; c1 = double(dc[1])/1000.0f;
    if (!silent) printf("c0: %f, c1: %f\n",c0,c1);

    float dc_alpha[2];
    fs["dc_alpha"] >> m;
    //memcpy(dc_alpha,m.data,sizeof(dc)); alpha0 = double(dc_alpha[0]); alpha1 = double(dc_alpha[1]);
    //if (!silent) printf("alpha0: %f, alpha1: %f\n",alpha0,alpha1);


    fs["dc_beta"] >> m;

    //if (m.rows*m.cols == 0) {
    //    for (int j = 0; j < height; j++)
    //        for (int i = 0; i < width; i++)
    //            beta[i+j*width] = 0.0f;
    //    printf("identity beta.\n");
    //} else {
    //    for (int j = 0; j < height; j++)
    //        for (int i = 0; i < width; i++)
    //            beta[i+j*width] = m.at<float>(j,i);
    //    printf("found beta.\n");
    //}

    float disp_error_var;
    fs["depth_error_var"] >> m;
    disp_error_var = m.at<float>(0);
    if (!silent) printf("disp_error_var: %f\n",disp_error_var);
    printf("(L) fovx: %f, fovy:%f (R) fovx: %f, fovy:%f\n",getFovX_L(),getFovY_L(),getFovX_R(),getFovY_R());
    useXYOffset = false; // use disparity image grid for 2D points
    return 1;
}
*/
bool Calibration::fileExists(const char *fn) {
    FILE *f = fopen(fn,"rb");
    if (f == NULL) return false;
    fclose(f);
    return true;
}

/*
int rodrigues2( const CvMat* src, CvMat* dst, CvMat* jacobian )
{
    int depth, elem_size;
    int i, k;
    double J[27];
    CvMat matJ = cvMat( 3, 9, CV_64F, J );

    if( !CV_IS_MAT(src) )
        CV_Error( !src ? CV_StsNullPtr : CV_StsBadArg, "Input argument is not a valid matrix" );

    if( !CV_IS_MAT(dst) )
        CV_Error( !dst ? CV_StsNullPtr : CV_StsBadArg,
        "The first output argument is not a valid matrix" );

    depth = CV_MAT_DEPTH(src->type);
    elem_size = CV_ELEM_SIZE(depth);

    if( depth != CV_32F && depth != CV_64F )
        CV_Error( CV_StsUnsupportedFormat, "The matrices must have 32f or 64f data type" );

    if( !CV_ARE_DEPTHS_EQ(src, dst) )
        CV_Error( CV_StsUnmatchedFormats, "All the matrices must have the same data type" );

    if( jacobian )
    {
        if( !CV_IS_MAT(jacobian) )
            CV_Error( CV_StsBadArg, "Jacobian is not a valid matrix" );

        if( !CV_ARE_DEPTHS_EQ(src, jacobian) || CV_MAT_CN(jacobian->type) != 1 )
            CV_Error( CV_StsUnmatchedFormats, "Jacobian must have 32fC1 or 64fC1 datatype" );

        if( (jacobian->rows != 9 || jacobian->cols != 3) &&
            (jacobian->rows != 3 || jacobian->cols != 9))
            CV_Error( CV_StsBadSize, "Jacobian must be 3x9 or 9x3" );
    }

    if( src->cols == 1 || src->rows == 1 )
    {
        double rx, ry, rz, theta;
        int step = src->rows > 1 ? src->step / elem_size : 1;

        if( src->rows + src->cols*CV_MAT_CN(src->type) - 1 != 3 )
            CV_Error( CV_StsBadSize, "Input matrix must be 1x3, 3x1 or 3x3" );

        if( dst->rows != 3 || dst->cols != 3 || CV_MAT_CN(dst->type) != 1 )
            CV_Error( CV_StsBadSize, "Output matrix must be 3x3, single-channel floating point matrix" );

        if( depth == CV_32F )
        {
            rx = src->data.fl[0];
            ry = src->data.fl[step];
            rz = src->data.fl[step*2];
        }
        else
        {
            rx = src->data.db[0];
            ry = src->data.db[step];
            rz = src->data.db[step*2];
        }
        theta = sqrt(rx*rx + ry*ry + rz*rz);

        if( theta < DBL_EPSILON )
        {
            cvSetIdentity( dst );

            if( jacobian )
            {
                memset( J, 0, sizeof(J) );
                J[5] = J[15] = J[19] = -1;
                J[7] = J[11] = J[21] = 1;
            }
        }
        else
        {
            const double I[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

            double c = cos(theta);
            double s = sin(theta);
            double c1 = 1. - c;
            double itheta = theta ? 1./theta : 0.;

            rx *= itheta; ry *= itheta; rz *= itheta;

            double rrt[] = { rx*rx, rx*ry, rx*rz, rx*ry, ry*ry, ry*rz, rx*rz, ry*rz, rz*rz };
            double _r_x_[] = { 0, -rz, ry, rz, 0, -rx, -ry, rx, 0 };
            double R[9];
            CvMat matR = cvMat( 3, 3, CV_64F, R );

            // R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
            // where [r_x] is [0 -rz ry; rz 0 -rx; -ry rx 0]
            for( k = 0; k < 9; k++ )
                R[k] = c*I[k] + c1*rrt[k] + s*_r_x_[k];

            cvConvert( &matR, dst );

            if( jacobian )
            {
                double drrt[] = { rx+rx, ry, rz, ry, 0, 0, rz, 0, 0,
                                  0, rx, 0, rx, ry+ry, rz, 0, rz, 0,
                                  0, 0, rx, 0, 0, ry, rx, ry, rz+rz };
                double d_r_x_[] = { 0, 0, 0, 0, 0, -1, 0, 1, 0,
                                    0, 0, 1, 0, 0, 0, -1, 0, 0,
                                    0, -1, 0, 1, 0, 0, 0, 0, 0 };
                for( i = 0; i < 3; i++ )
                {
                    double ri = i == 0 ? rx : i == 1 ? ry : rz;
                    double a0 = -s*ri, a1 = (s - 2*c1*itheta)*ri, a2 = c1*itheta;
                    double a3 = (c - s*itheta)*ri, a4 = s*itheta;
                    for( k = 0; k < 9; k++ )
                        J[i*9+k] = a0*I[k] + a1*rrt[k] + a2*drrt[i*9+k] +
                                   a3*_r_x_[k] + a4*d_r_x_[i*9+k];
                }
            }
        }
    }
    else if( src->cols == 3 && src->rows == 3 )
    {
        double R[9], U[9], V[9], W[3], rx, ry, rz;
        CvMat matR = cvMat( 3, 3, CV_64F, R );
        CvMat matU = cvMat( 3, 3, CV_64F, U );
        CvMat matV = cvMat( 3, 3, CV_64F, V );
        CvMat matW = cvMat( 3, 1, CV_64F, W );
        double theta, s, c;
        int step = dst->rows > 1 ? dst->step / elem_size : 1;

        if( (dst->rows != 1 || dst->cols*CV_MAT_CN(dst->type) != 3) &&
            (dst->rows != 3 || dst->cols != 1 || CV_MAT_CN(dst->type) != 1))
            CV_Error( CV_StsBadSize, "Output matrix must be 1x3 or 3x1" );

        cvConvert( src, &matR );
        if( !cvCheckArr( &matR, CV_CHECK_RANGE+CV_CHECK_QUIET, -100, 100 ) )
        {
            cvZero(dst);
            if( jacobian )
                cvZero(jacobian);
            return 0;
        }

        cvSVD( &matR, &matW, &matU, &matV, CV_SVD_MODIFY_A + CV_SVD_U_T + CV_SVD_V_T );
        cvGEMM( &matU, &matV, 1, 0, 0, &matR, CV_GEMM_A_T );

        rx = R[7] - R[5];
        ry = R[2] - R[6];
        rz = R[3] - R[1];

        s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
        c = (R[0] + R[4] + R[8] - 1)*0.5;
        c = c > 1. ? 1. : c < -1. ? -1. : c;
        theta = acos(c);

        if( s < 1e-5 )
        {
            double t;

            if( c > 0 )
                rx = ry = rz = 0;
            else
            {
                t = (R[0] + 1)*0.5;
                rx = sqrt(MAX(t,0.));
                t = (R[4] + 1)*0.5;
                ry = sqrt(MAX(t,0.))*(R[1] < 0 ? -1. : 1.);
                t = (R[8] + 1)*0.5;
                rz = sqrt(MAX(t,0.))*(R[2] < 0 ? -1. : 1.);
                if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R[5] > 0) != (ry*rz > 0) )
                    rz = -rz;
                theta /= sqrt(rx*rx + ry*ry + rz*rz);
                rx *= theta;
                ry *= theta;
                rz *= theta;
            }

            if( jacobian )
            {
                memset( J, 0, sizeof(J) );
                if( c > 0 )
                {
                    J[5] = J[15] = J[19] = -0.5;
                    J[7] = J[11] = J[21] = 0.5;
                }
            }
        }
        else
        {
            double vth = 1/(2*s);

            if( jacobian )
            {
                double t, dtheta_dtr = -1./s;
                // var1 = [vth;theta]
                // var = [om1;var1] = [om1;vth;theta]
                double dvth_dtheta = -vth*c/s;
                double d1 = 0.5*dvth_dtheta*dtheta_dtr;
                double d2 = 0.5*dtheta_dtr;
                // dvar1/dR = dvar1/dtheta*dtheta/dR = [dvth/dtheta; 1] * dtheta/dtr * dtr/dR
                double dvardR[5*9] =
                {
                    0, 0, 0, 0, 0, 1, 0, -1, 0,
                    0, 0, -1, 0, 0, 0, 1, 0, 0,
                    0, 1, 0, -1, 0, 0, 0, 0, 0,
                    d1, 0, 0, 0, d1, 0, 0, 0, d1,
                    d2, 0, 0, 0, d2, 0, 0, 0, d2
                };
                // var2 = [om;theta]
                double dvar2dvar[] =
                {
                    vth, 0, 0, rx, 0,
                    0, vth, 0, ry, 0,
                    0, 0, vth, rz, 0,
                    0, 0, 0, 0, 1
                };
                double domegadvar2[] =
                {
                    theta, 0, 0, rx*vth,
                    0, theta, 0, ry*vth,
                    0, 0, theta, rz*vth
                };

                CvMat _dvardR = cvMat( 5, 9, CV_64FC1, dvardR );
                CvMat _dvar2dvar = cvMat( 4, 5, CV_64FC1, dvar2dvar );
                CvMat _domegadvar2 = cvMat( 3, 4, CV_64FC1, domegadvar2 );
                double t0[3*5];
                CvMat _t0 = cvMat( 3, 5, CV_64FC1, t0 );

                cvMatMul( &_domegadvar2, &_dvar2dvar, &_t0 );
                cvMatMul( &_t0, &_dvardR, &matJ );

                // transpose every row of matJ (treat the rows as 3x3 matrices)
                CV_SWAP(J[1], J[3], t); CV_SWAP(J[2], J[6], t); CV_SWAP(J[5], J[7], t);
                CV_SWAP(J[10], J[12], t); CV_SWAP(J[11], J[15], t); CV_SWAP(J[14], J[16], t);
                CV_SWAP(J[19], J[21], t); CV_SWAP(J[20], J[24], t); CV_SWAP(J[23], J[25], t);
            }

            vth *= theta;
            rx *= vth; ry *= vth; rz *= vth;
        }

        if( depth == CV_32F )
        {
            dst->data.fl[0] = (float)rx;
            dst->data.fl[step] = (float)ry;
            dst->data.fl[step*2] = (float)rz;
        }
        else
        {
            dst->data.db[0] = rx;
            dst->data.db[step] = ry;
            dst->data.db[step*2] = rz;
        }
    }

    if( jacobian )
    {
        if( depth == CV_32F )
        {
            if( jacobian->rows == matJ.rows )
                cvConvert( &matJ, jacobian );
            else
            {
                float Jf[3*9];
                CvMat _Jf = cvMat( matJ.rows, matJ.cols, CV_32FC1, Jf );
                cvConvert( &matJ, &_Jf );
                cvTranspose( &_Jf, jacobian );
            }
        }
        else if( jacobian->rows == matJ.rows )
            cvCopy( &matJ, jacobian );
        else
            cvTranspose( &matJ, jacobian );
    }

    return 1;
}
*/

int Calibration::init(const char *fn, bool silent) {

    if (calibData == NULL) {
        calibData = new float[CALIB_SIZE];
        //beta = &calibData[BETA_OFFSET];
        zpolyCoeffs = &calibData[ZPOLY_DATA];
    }

    char fileName[512]; strcpy(fileName,fn);

    int len = strlen(fileName);
    if (len < 4) { printf("invalid calib filename : %s\n",fileName); fflush(stdout); return 0; }

    minDist = 300;
    maxDist = 7000;

    // check if calibration is given in oulu compatible yml format
   // if (fileName[len-3] == 'y' && fileExists(fileName)) { return initOulu(fileName,silent); }

    if (!fileExists(fileName)) {
        // enforce xml file!
        fileName[len-3] = 'x';
        printf("file %s did not exist, loading %s instead!\n",fn,fileName);
    }

    if (!silent) printf("loading %s\n",fileName);

    TiXmlDocument *doc = new TiXmlDocument;
    bool loadOkay = doc->LoadFile(fileName,TIXML_DEFAULT_ENCODING);

    if (!loadOkay) {
        printf("problem loading %s!\n", fileName);
        fflush(stdin); fflush(stdout);
        return 0;
        assert(0);
    }
    m_filename = new char[512]; memset(m_filename,0,512); strncpy(m_filename,fileName,512);//strdup(fileName);

    cv::Mat omMat(3, 1, CV_64F, Scalar(0));
	cv::Mat TMat(3, 1, CV_64F, Scalar(0));
	cv::Mat RMat(3, 3, CV_64F, Scalar(0));
    //Mat JMat(3, 3, CV_64F, Scalar(0));

    TiXmlHandle hDoc(doc);
    TiXmlElement* pElem;
    TiXmlHandle hRoot(0);

    double B=0,b=0;

    //get root element 'TodoList', set to hRoot
    pElem = hDoc.FirstChildElement().Element();
    if(!pElem) { printf("no xml root found!\n"); assert(0); }
    hRoot = TiXmlHandle(pElem);

    //printf("filename: %s\n",fileName);
    pElem=hRoot.FirstChild("KL").Element();
    if (pElem!=0){
        if (!silent) printf("parsing KL\n");
        const char *sizeStr = pElem->Attribute("size");
        int sX=0,sY=0;
        sscanf(sizeStr,"%d %d",&sX,&sY);
        assert(sX == 3 && sY == 3);
        double tmpK[9];
        parseDoubles(pElem->GetText(),sX*sY,tmpK);
        transpose(tmpK, KL, 3,3);
        //if (!silent) dumpMatrix("KL=",KL,3,3);
    } else { printf("KL not found!\n"); fflush(stdout); return 0; }

    pElem=hRoot.FirstChild("KR").Element();
    if (pElem!=0){
        if (!silent) printf("parsing KR\n");
        const char *sizeStr = pElem->Attribute("size");
        int sX=0,sY=0;
        sscanf(sizeStr,"%d %d",&sX,&sY);
        assert(sX == 3 && sY == 3);
        double tmpK[9];
        parseDoubles(pElem->GetText(),sX*sY,tmpK);
        transpose(tmpK, KR, 3,3);
        //if (!silent) dumpMatrix("KR=",KR,3,3);
    } else { printf("KR not found!\n");  fflush(stdout); return 0; }

    memset(kcR,0,sizeof(kcR));
    memset(kcL,0,sizeof(kcL));

    // for backward compatibility:
    pElem=hRoot.FirstChild("kc_right").Element();
    if (pElem!=0){
        if (!silent) printf("parsing kc_right\n");
        const char *sizeStr = pElem->Attribute("size");
        int sX=0,sY=0;
        sscanf(sizeStr,"%d %d",&sX,&sY);
        assert((sX > 0) && (sX <= 8) && sY == 1);
        parseDoubles(pElem->GetText(),sX*sY,kcR);
        if (!silent) {
            printf("kcR: \n");
            for (int i = 0; i < sX*sY;i++) printf("%e\n",kcR[i]);
         //   printf("%e %e %e %e %e\n",kcR[0],kcR[1],kcR[2],kcR[3],kcR[4]);
        }
    }

    pElem=hRoot.FirstChild("B").Element();
    if (pElem!=0){
        if (!silent) printf("parsing B\n");
        const char *sizeStr = pElem->Attribute("size");
        int sX=0,sY=0;
        sscanf(sizeStr,"%d %d",&sX,&sY);
        assert(sX == 1 && sY == 1);
        parseDoubles(pElem->GetText(),sX*sY,&B);
        //if (!silent) printf("B: %e\n",B);
    }

    pElem=hRoot.FirstChild("b").Element();
    if (pElem!=0){
        if (!silent) printf("parsing b\n");
        const char *sizeStr = pElem->Attribute("size");
        int sX=0,sY=0;
        sscanf(sizeStr,"%d %d",&sX,&sY);
        assert(sX == 1 && sY == 1);
        parseDoubles(pElem->GetText(),sX*sY,&b);
       // if (!silent) printf("b: %e\n",b);
    }
    // for backward compatibility:
    identity4x4(TLR);
    identity4x4(TRL);

    pElem=hRoot.FirstChild("om").Element();
    if (pElem!=0){
        const char *sizeStr = pElem->Attribute("size");
        int sX=0,sY=0;
        sscanf(sizeStr,"%d %d",&sX,&sY);
        assert(sX == 3 && sY == 1);
        //		printf("%d %d\n",sX,sY);
        parseDoubles(pElem->GetText(),sX*sY,(double*)omMat.ptr());
        //printf("parsing om\n");
        //		printf("om: %e %e %e\n",omMat.at<double>(0),omMat.at<double>(1),omMat.at<double>(2));
        Mat jacobianDummy;
        //omMat.
        Rodrigues(omMat,RMat,jacobianDummy);
        //printf("%e %e %e\n%e %e %e\n%e %e %e\n",RMat.at<double>(cvPoint(0,0)),RMat.at<double>(cvPoint(1,0)),RMat.at<double>(cvPoint(2,0)),RMat.at<double>(cvPoint(0,1)),RMat.at<double>(cvPoint(1,1)),RMat.at<double>(cvPoint(2,1)),RMat.at<double>(cvPoint(0,2)),RMat.at<double>(cvPoint(1,2)),RMat.at<double>(cvPoint(2,2)));
        TLR[0] = RMat.at<double>(cv::Point(0,0)); TLR[1] = RMat.at<double>(cv::Point(1,0)); TLR[2]  = RMat.at<double>(cv::Point(2,0));
        TLR[4] = RMat.at<double>(cv::Point(0,1)); TLR[5] = RMat.at<double>(cv::Point(1,1)); TLR[6]  = RMat.at<double>(cv::Point(2,1));
        TLR[8] = RMat.at<double>(cv::Point(0,2)); TLR[9] = RMat.at<double>(cv::Point(1,2)); TLR[10] = RMat.at<double>(cv::Point(2,2));
        invertRT4(TLR,TRL);
    } else { printf("om not found!\n");  fflush(stdout); return 0; }

    pElem=hRoot.FirstChild("T").Element();
    if (pElem!=0){
        if (!silent) printf("parsing T\n");
        const char *sizeStr = pElem->Attribute("size");
        int sX=0,sY=0;
        sscanf(sizeStr,"%d %d",&sX,&sY);
        assert(sX == 3 && sY == 1);
        parseDoubles(pElem->GetText(),sX*sY,(double*)TMat.ptr());
        TLR[3]  = TMat.at<double>(0);
        TLR[7]  = TMat.at<double>(1);
        TLR[11] = TMat.at<double>(2);
        //if (!silent) dumpMatrix("Tb=",TLR,4,4);
        //transpose(tmpT12, TRL, 4,4);
        invertRT4(TLR,TRL);
    } else { printf("om not found!\n"); fflush(stdout); return 0; }

    pElem=hRoot.FirstChild("zpoly").Element();
    if (pElem!=0) {
        const char *sizeStr = pElem->Attribute("size");
        int sX=0,sY=0;
        sscanf(sizeStr,"%d %d",&sX,&sY);
        if (sX*sY > 5) {
            printf("zpoly has illegal dimensions (%d x %d)!\n",sX,sY); fflush(stdout);
        }
        numZPolyCoeffs = sX*sY;
        double zpolyCoeffsDouble[16];
        parseDoubles(pElem->GetText(),numZPolyCoeffs,zpolyCoeffsDouble);
        dumpMatrix("zpoly=",zpolyCoeffsDouble,sY,sX);

        for (int i = 0; i < numZPolyCoeffs; i++) zpolyCoeffs[i] = zpolyCoeffsDouble[i];
        dumpMatrix("zpolyf=",zpolyCoeffs,1,numZPolyCoeffs);
    }

    double scale = 1.0f / (8 * b * fabs(KL[0])*640.0);
    c1 = -scale;
    c0 = B*640.0f*scale;

    // default disparity distortion coeffs (exponential correction)
    //alpha0 = 2.316537f;
    //alpha1 = 0.003942f;
/*
    // default distortion is zero
    for (int j = 0; j < 480; j++) {
        for (int i = 0; i < 640; i++) {
            beta[i+j*640] = 0.0f;
        }
    }*/
    useXYOffset = true; // IR image grid in use, (-4,-3) image offset required
    delete doc;
    return 1;
}

void Calibration::resetVariables() {
    calibData = NULL;
    m_filename = NULL;
    //beta = NULL;
    zpolyCoeffs = NULL;
    numZPolyCoeffs = 0;
    memset(kcR,0,sizeof(kcR));
    memset(kcL,0,sizeof(kcL));
   // printf("sizeof kcr: %d bytes.\n",sizeof(kcR));
}


Calibration::Calibration(const char *fileName, bool silent) {
    resetVariables();
    if (!init(fileName,silent)) {
        printf("calib loading failed!\n"); fflush(stdout);
    }
}

Calibration::Calibration() {
    resetVariables();
}


Calibration::~Calibration() {
    if (calibData != NULL) delete[] calibData;
    if (m_filename != NULL) { /*printf("calib files was : %s\n",m_filename); fflush(stdout);*/ delete[] m_filename;}
};


int Calibration::getNumPolyCoeffs() {
    return numZPolyCoeffs;
}

float *Calibration::getPolyCoeffs() {
    return zpolyCoeffs;
}


void Calibration::saveZPolynomial(int polyOrder, float *coeffs) {
    if (m_filename == NULL) { printf("calib: can not save zmap correction, filename is not defined!\n"); return; }
    TiXmlDocument *doc = new TiXmlDocument;
    bool loadOkay = doc->LoadFile(m_filename,TIXML_DEFAULT_ENCODING);

    if (!loadOkay) {
        printf("calib: can not save zmap correction %s is not found!\n", m_filename); fflush(stdout); return;
    }

    TiXmlHandle hDoc(doc);
    TiXmlElement* pElem;
    TiXmlHandle hRoot(0);

    //get root element 'TodoList', set to hRoot
    TiXmlElement *root = hDoc.FirstChildElement().Element();
    if(!root) { printf("no xml root found!\n"); fflush(stdout); return; }
    hRoot = TiXmlHandle(root);

    // prepare updated strings:
    char sizeStr[512]; sprintf(sizeStr,"%d %d",polyOrder+1,1);
    char coeffStr[512]; memset(coeffStr,0,512);
    for (int i = 0; i < polyOrder+1; i++) { char buf[512]; sprintf(buf,"%e ",coeffs[i]); strcat(coeffStr,buf); }

    // remove all zpolys:
    while (1) {
        pElem=hRoot.FirstChild("zpoly").Element();
        if (pElem != 0) root->RemoveChild(pElem);
        else break;
    }

    printf("zpoly not found, adding..!\n"); fflush(stdout);
    TiXmlElement *element = new TiXmlElement("zpoly");
    element->SetAttribute("size",sizeStr);
    TiXmlText * text = new TiXmlText( coeffStr );
    element->LinkEndChild( text );
    root->LinkEndChild( element );

    doc->SaveFile();
    delete doc;
}
