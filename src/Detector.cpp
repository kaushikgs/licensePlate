#include <opencv2/core/core.hpp>
#include <fstream>
#include <vector>
#include <string>
// #include <sys/stat.h>
#include "mser.h"
#include "Convnet.h"
#include "Detector.h"

#define PI 3.14159265

using namespace std;
using namespace cv;
using namespace caffe;
using namespace extrema;

Detector::Detector(string modelPath, string weightsPath, string meanPath, int regionWidth, int regionHeight)
 : batchSize(64), regionSize(regionWidth, regionHeight), convnet(modelPath, weightsPath)
{
    setMean(meanPath);
}

//* Load the mean file in custom format. */
void Detector::setMean(const string& meanPath) {
    float meanDbl[3];
    ifstream meanFile(meanPath);
    meanFile >> meanDbl[0] >> meanDbl[1] >> meanDbl[2];
    meanFile.close();
    Scalar channelMeanRGB(meanDbl[0], meanDbl[1], meanDbl[2]);
    mean = cv::Mat(regionSize, CV_32FC3, channelMeanRGB);
}

void Detector::genMSEREllipses(Mat &image, vector<ellipseParameters> &mserEllipses){
    extrema::ExtremaParams p;
    p.preprocess = 0;
    p.max_area = 0.01;
    p.min_size = 30;
    p.min_margin = 10;
    p.relative = 0;
    p.verbose = 0;
    p.debug = 0;
    double scale_factor = 1.0;
    scale_factor = scale_factor * 2; /* compensate covariance matrix */

    computeMSEREllipses(image, mserEllipses, p, scale_factor);
}

bool Detector::liesInside(Mat &img, RotatedRect &rect){
    Rect bound = rect.boundingRect();

    if(bound.x < 0 || bound.y < 0){
        return false;
    }
    if(bound.x + bound.width > img.cols || bound.y + bound.height > img.rows){
        return false;
    }
    return true;
}

void Detector::filterNConvertEllipses(Mat &image, vector<ellipseParameters> &mserEllipses, vector<RotatedRect> &mserBoxes){
    int minArea = 500;
    int maxArea = 1000000;

    for(unsigned int i=0; i<mserEllipses.size(); i++){
        ellipseParameters tempEll = mserEllipses[i];
        float ellArea = PI * tempEll.axes.width * tempEll.axes.height;
        if(ellArea > minArea && ellArea < maxArea && (tempEll.angle < 25 || tempEll.angle > 155) && tempEll.axes.height > 0 && (float)tempEll.axes.width/(float)tempEll.axes.height > 1.5 && (float)tempEll.axes.width/(float)tempEll.axes.height < 10 )//Potential Number Plates
        {
            RotatedRect mserRect(tempEll.center, Size(tempEll.axes.width*4, tempEll.axes.height*4), tempEll.angle); //double the region
            if (liesInside(image, mserRect)){
                mserBoxes.push_back(mserRect);
            }
        }
    }
}

void Detector::genRegions(Mat &image, vector<RotatedRect> &numPlateBoxes){
    vector<ellipseParameters> mserEllipses;
    genMSEREllipses(image, mserEllipses);
    filterNConvertEllipses(image, mserEllipses, numPlateBoxes);
    //TODO: yellowchannel
}

Mat cropRegion(Mat image, RotatedRect rect){
    Mat M, rotated, cropped;
    float angle = rect.angle;
    Size rect_size = rect.size;
    Rect bound = rect.boundingRect();
    Mat boundMat(image, bound);
    
    Point center(rect.center.x - bound.x, rect.center.y - bound.y);
    M = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(boundMat, rotated, M, boundMat.size(), INTER_CUBIC);
    getRectSubPix(rotated, rect_size, center, cropped);
    return cropped;
}

Mat Detector::preprocessMat(Mat &input){
    Mat resized, rescaled, normalized;
    resize(input, resized, regionSize);
    resized.convertTo(rescaled, CV_32FC3, 1.0/255);
    subtract(rescaled, mean, normalized);
    return normalized;
}

void Detector::detectNumPlates(Mat &image, vector<Mat> &numPlateImgs, vector<RotatedRect> &numPlateBoxes){
    vector<RotatedRect> mserBoxes;
    genRegions(image, mserBoxes);
    Mat imageCopy = image.clone();
    
    int numBatches = ceil( ((float) mserBoxes.size()) / batchSize);
    vector<Mat> batchMats;
    
    for(int batchNo = 0, readNo = 0, writeNo = 0; batchNo < numBatches; batchNo++){
        int curBatchSize;
        if(batchNo == (numBatches - 1) && ((mserBoxes.size() % batchSize) != 0))
            curBatchSize = mserBoxes.size() % batchSize;
        else
            curBatchSize = batchSize;

        batchMats.clear();
        for(int i = 0; i < curBatchSize; i++){
            Mat candidateMat = cropRegion(image, mserBoxes[readNo]);
            imwrite(string("debugFiles/detect/b4processing_") + to_string(readNo) + ".jpg", candidateMat);  //DEBUG
            batchMats.push_back( preprocessMat(candidateMat));
            readNo++;
        }

        vector<float> batchScores = convnet.scoreBatch(batchMats);
        for(int i=0; i < curBatchSize; i++){
            if(batchScores[2*i] > batchScores[2*i+1]){
                // numPlateImgs.push_back(batchMats[i]);
                Mat numPlateImg = cropRegion(imageCopy, mserBoxes[writeNo]);
                imwrite(string("debugFiles/detect/numPlateImg_") + to_string(writeNo) + ".jpg", numPlateImg);   //DEBUG
                numPlateImgs.push_back( numPlateImg);
                numPlateBoxes.push_back(mserBoxes[writeNo]);
            }
            writeNo++;
        }
    }
    return;
}