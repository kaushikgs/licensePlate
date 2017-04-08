#ifndef DETECTOR_H
#define DETECTOR_H

#include "libExtrema.h"
#include "mser.h"
#include "Convnet.h"

class Detector{
    std::string meanPath;
    int batchSize;
    cv::Size regionSize;
    Convnet convnet;
    cv::Mat mean;

    void setMean(const std::string &meanPath);
    void genMSEREllipses(cv::Mat &image, std::vector<ellipseParameters> &mserEllipses);
    bool liesInside(cv::Mat &img, cv::RotatedRect &rect);
    void filterNConvertEllipses(cv::Mat &image, std::vector<ellipseParameters> &mserEllipses, std::vector<cv::RotatedRect> &mserBoxes);
    void genRegions(cv::Mat &image, std::vector<cv::RotatedRect> &candidateRegions);
    cv::Mat preprocessMat(cv::Mat &input);

public:
    Detector(std::string modelPath, std::string weightsPath, std::string meanPath, int regionWidth, int regionHeight);
    void detectNumPlates(cv::Mat &image, std::vector<cv::Mat> &numPlateImgs, std::vector<cv::RotatedRect> &numPlateBoxes);
};

#endif