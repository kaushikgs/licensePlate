#include "mser.h"

using namespace cv;
using namespace extrema;

//add MSER ellipses, only MSERmin
void computeMSEREllipses(Mat &inputImage, vector<ellipseParameters> &MSEREllipses, extrema::ExtremaParams &p, double scale_factor){

    //Create an image that can be given as input to MSER code by Matas
    extrema::ExtremaImage im;
    im.width = inputImage.cols;
    im.height = inputImage.rows;
    im.channels = inputImage.channels();
    im.data = inputImage.data; //Might need a type case to unsigned char *
    
    if (im.channels<3)
        p.preprocess = extrema::PREPROCESS_CHANNEL_none;
    else
        p.preprocess = extrema::PREPROCESS_CHANNEL_intensity;
    
    extrema::RLEExtrema result;
    result = extrema::getRLEExtrema(p, im);
    
    //result.MSERmin.insert(result.MSERmin.end(), result.MSERplus.begin(), result.MSERplus.end());
    //convRleToRect(result.MSERmin, MSERRects, im.width, im.height);
    extrema::convRleToEll(result.MSERmin, MSEREllipses, scale_factor);
    return;
}

//add MSER ellipses and rectangles to MSEREllipses and MSERRects
void computeMSERRLEs(Mat &inputImage, vector<RLERegion> &mserRLEs, vector<Rect> &mserRects, extrema::ExtremaParams &p, double scale_factor){

    //Create an image that can be given as input to MSER code by Matas
    extrema::ExtremaImage im;
    im.width = inputImage.cols;
    im.height = inputImage.rows;
    im.channels = inputImage.channels();
    im.data = inputImage.data; //Might need a type case to unsigned char *
    
    if (im.channels<3)
        p.preprocess = extrema::PREPROCESS_CHANNEL_none;
    else
        p.preprocess = extrema::PREPROCESS_CHANNEL_intensity;
    
    extrema::RLEExtrema result;
    result = extrema::getRLEExtrema(p, im);
    
    // mserRLEs.insert(mserRLEs.end(), result.MSERmin.begin(), result.MSERmin.end());
    mserRLEs.insert(mserRLEs.end(), result.MSERplus.begin(), result.MSERplus.end());
    convRleToRect(mserRLEs, mserRects);
    return;
}

// void computeMSER2(Mat &inputImage, vector<ellipseParameters> &MSEREllipses){
// void computeMSER2(Mat &inputImage, vector<Rect> &MSERRects){
//     extrema::ExtremaParams p;
//     p.preprocess = 0;
//     p.max_area = 0.01;
//     p.min_size = 30;
//     p.min_margin = 10;
//     p.relative = 0;
//     p.verbose = 0;
//     p.debug = 0;
//     double scale_factor = 1.0;
//     scale_factor = scale_factor * 2; /* compensate covariance matrix */

//     //Create an image that can be given as input to MSER code by Matas
//     extrema::ExtremaImage im;
//     im.width = inputImage.cols;
//     im.height = inputImage.rows;
//     im.channels = inputImage.channels();
//     im.data = inputImage.data; //Might need a type case to unsigned char *
    
//     if (im.channels<3)
//         p.preprocess = extrema::PREPROCESS_CHANNEL_none;
//     else
//         p.preprocess = extrema::PREPROCESS_CHANNEL_intensity;
    
//     extrema::RLEExtrema result;
//     result = extrema::getRLEExtrema(p, im);

//     // vector<ellipseParameters> MSERMinEllipses;
//     // vector<ellipseParameters> MSERPlusEllipses;
    
//     result.MSERmin.insert(result.MSERmin.end(), result.MSERplus.begin(), result.MSERplus.end());
//     convRleToRect(result.MSERmin, MSERRects, im.width, im.height);
//     //convRleToRect(result.MSERplus, MSERRects, im.width, im.height);
//     // extrema::convRleToEll(result.MSERmin, MSERMinEllipses, scale_factor);
//     // extrema::convRleToEll(result.MSERplus, MSERPlusEllipses, scale_factor);

//     // MSEREllipses.insert(MSEREllipses.end(), MSERMinEllipses.begin(), MSERMinEllipses.end());
//     // MSEREllipses.insert(MSEREllipses.end(), MSERPlusEllipses.begin(), MSERPlusEllipses.end());

//     return;
// }

// //add MSER ellipses and rectangles to MSEREllipses and MSERRects
// void computeMSERChar(Mat &inputImage, vector<ellipseParameters> &MSEREllipses, vector<Rect> &MSERRects){

//     extrema::ExtremaParams p;
//     p.preprocess = 0;
//     p.max_area = 0.9;   //changed from 0.01
//     p.min_size = 3;
//     p.min_margin = 1;
//     p.relative = 0;
//     p.verbose = 0;
//     p.debug = 0;
//     double scale_factor = 1.0;
//     scale_factor = scale_factor * 2; /* compensate covariance matrix */

//     //Create an image that can be given as input to MSER code by Matas
//     extrema::ExtremaImage im;
//     im.width = inputImage.cols;
//     im.height = inputImage.rows;
//     im.channels = inputImage.channels();
//     im.data = inputImage.data; //Might need a type case to unsigned char *
    
//     if (im.channels<3)
//         p.preprocess = extrema::PREPROCESS_CHANNEL_none;
//     else
//         p.preprocess = extrema::PREPROCESS_CHANNEL_intensity;
    
//     extrema::RLEExtrema result;
//     result = extrema::getRLEExtrema(p, im);

//     vector<extrema::RLERegion> allMSERs;
//     allMSERs.insert(allMSERs.end(), result.MSERplus.begin(), result.MSERplus.end());
//     allMSERs.insert(allMSERs.end(), result.MSERmin.begin(), result.MSERmin.end());
    
//     //result.MSERmin.insert(result.MSERmin.end(), result.MSERplus.begin(), result.MSERplus.end());
//     extrema::convRleToEll(allMSERs, MSEREllipses, scale_factor);    
//     convRleToRect(allMSERs, MSERRects, im.width, im.height);
//     return;
// }

//get corners of the rectangle surrounding MSER
void convRleToRect(vector<extrema::RLERegion> &MSER, vector<Rect> &rects){
    for(int i=0;i<MSER.size();i++){
        extrema::RLERegion r=MSER[i];
        Point topLeft, bottomRight;
        
            //remember that x=>col and y=>row
        topLeft.x=INT_MAX;
        topLeft.y=INT_MAX;
        bottomRight.x=INT_MIN;
        bottomRight.y=INT_MIN;
        for(int j=0;j<r.rle.size();j++){
                //condition to get extremas
            if(r.rle[j].line<topLeft.y){
                topLeft.y=r.rle[j].line;
            }
            if(r.rle[j].line>bottomRight.y){
                bottomRight.y=r.rle[j].line;
            }
            if(r.rle[j].col1<topLeft.x){
                topLeft.x=r.rle[j].col1;
            }
            if(r.rle[j].col2>bottomRight.x){
                bottomRight.x=r.rle[j].col2;
            }
        }
        bottomRight.x = bottomRight.x + 1;  //bottomright is exclusive
        bottomRight.y = bottomRight.y + 1;
        // int rewidth = bottomRight.x - topLeft.x;
        // int reheight = bottomRight.y - topLeft.y;
        // topLeft.x = max(topLeft.x - rewidth/2, 0);
        // topLeft.y = max(topLeft.y - reheight/2, 0);
        // bottomRight.x = min(bottomRight.x + rewidth/2, imwidth-1);
        // bottomRight.y = min(bottomRight.y + reheight/2, imheight-1);
        rects.push_back(Rect(topLeft, bottomRight));
    }
}
