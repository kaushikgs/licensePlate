#ifndef READER_H
#define READER_H

#include <opencv2/ml/ml.hpp>
#include "mser.h"
#include "Convnet.h"

class Reader{
    int batchSize;
    cv::Size regionSize;
	Convnet convnet;
    cv::Mat mean;
    int numClasses;
    int numDetections;  //DEBUG
    float threshold;
    CvSVM svm;
    vector<float> svmMean, svmStd;

    void setMean(const std::string &meanPath);
    cv::Mat preprocessMat(cv::Mat &input);
    cv::Mat makeMatFrmRLE(extrema::RLERegion &region, cv::Rect &boundBox);
    void svmFilter(cv::Mat &img, std::vector<cv::Rect> &allRects, std::vector<int> &srcIdxs, std::vector<int> &dstIdxs);
    void genMSERRLEs(cv::Mat &image, string imageName, std::vector<extrema::RLERegion> &mserRLEs, std::vector<cv::Rect> &mserBoxes);
    
public:
    Reader(string configPath);
    std::string readNumPlate(cv::Mat &numPlateImg, string imageName);
};

#endif