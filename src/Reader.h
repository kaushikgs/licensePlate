#ifndef READER_H
#define READER_H

#include "mser.h"
#include "Convnet.h"

struct Candidate;

class Reader{
    int batchSize;
    cv::Size regionSize;
	Convnet convnet;
    cv::Mat mean;
    int numClasses;
    int numDetections;  //DEBUG
    float threshold;
    ofstream xmlFile;

    void setMean(const std::string &meanPath);
    cv::Mat preprocessMat(cv::Mat &input);
    void genMSERRLEs(cv::Mat &image, std::string imageName, bool  doubled, std::vector<extrema::RLERegion> &mserRLEs, std::vector<cv::Rect> &mserBoxes);
    cv::Mat makeMatFrmRLE(extrema::RLERegion &region, cv::Rect &boundBox);
    void writeXML(std::string imageName, float scale, std::vector<Candidate> &detections);
    
public:
    Reader(string configPath);
    void initXML();
    void closeXML();
    std::string readNumPlate(cv::Mat &numPlateImg, string imageName, bool doubled);
};

#endif