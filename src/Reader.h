#ifndef READER_H
#define READER_H

#include "mser.h"
#include "Convnet.h"

class Reader{
	std::string meanPath;
    int batchSize;
    cv::Size regionSize;
	Convnet convnet;
    cv::Mat mean;
    int numClasses;

    void setMean(const std::string &meanPath);
    cv::Mat preprocessMat(cv::Mat &input);
    cv::Mat makeMatFrmRLE(extrema::RLERegion &region, cv::Rect &boundBox);

public:
    Reader(std::string modelPath, std::string weightsPath, std::string meanPath, int regionWidth, int regionHeight);
    std::string readNumPlate(cv::Mat &numPlateImg);
};

#endif