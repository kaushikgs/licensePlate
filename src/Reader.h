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
    cv::Mat makeMatFrmRLE(extrema::RLERegion &region, cv::Rect &boundBox);
    
public:
    Reader(string configPath);
    std::string readNumPlate(cv::Mat &numPlateImg);
};

#endif