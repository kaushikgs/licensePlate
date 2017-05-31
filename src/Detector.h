#ifndef DETECTOR_H
#define DETECTOR_H

#include "libExtrema.h"
#include "mser.h"
#include "Convnet.h"

class Detector{
    int batchSize;
    cv::Size regionSize;
    Convnet convnet;
    cv::Mat mean;
    float threshold;

    void setMean(const std::string &meanPath);
    void genMSEREllipses(cv::Mat &image, std::vector<ellipseParameters> &mserEllipses);
    void filterNConvertEllipses(cv::Mat &image, std::vector<ellipseParameters> &mserEllipses, std::vector<cv::RotatedRect> &mserBoxes);
    void genRegions(cv::Mat &image, std::vector<cv::RotatedRect> &candidateRegions);
    cv::Mat preprocessMat(cv::Mat &input);

public:
    Detector(string configPath);
    void detectNumPlates(cv::Mat &image, string imageName, std::vector<cv::Mat> &numPlateImgs, std::vector<cv::RotatedRect> &numPlateBoxes, std::vector<bool> &doubled);
};

#endif