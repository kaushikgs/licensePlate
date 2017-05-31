#ifndef READER_H
#define READER_H

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

    void setMean(const std::string &meanPath);
    cv::Mat preprocessMat(cv::Mat &input);
    void genMSERRLEs(cv::Mat &image, std::string imageName, std::vector<extrema::RLERegion> &mserRLEs, std::vector<cv::Rect> &mserBoxes);
    cv::Mat makeMatFrmRLE(extrema::RLERegion &region, cv::Rect &boundBox);
    
public:
    Reader(string configPath);
    std::string readNumPlate(cv::Mat &numPlateImg, string imageName);
};

#endif