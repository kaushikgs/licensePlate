#ifndef MSER_H
#define MSER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "libExtrema.h" 

void computeMSEREllipses(cv::Mat &inputImage, std::vector<ellipseParameters> &MSEREllipses, extrema::ExtremaParams &p, double scale_factor);
void computeMSERRLEs(cv::Mat &inputImage, std::vector<extrema::RLERegion> &mserRLEs, vector<cv::Rect> &mserRects, extrema::ExtremaParams &p, double scale_factor);

// void computeMSER2(cv::Mat &inputImage, std::vector<cv::Rect> &MSEREllipses);
// void computeMSERChar(cv::Mat &inputImage, std::vector<ellipseParameters> &MSEREllipses, std::vector<cv::Rect> &MSERRects);

void convRleToRect(std::vector<extrema::RLERegion> &MSER, std::vector<cv::Rect> &rects);

#endif