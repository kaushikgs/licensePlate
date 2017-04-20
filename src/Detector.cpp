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

Detector::Detector(string configPath) {
    string modelPath, weightsPath, meanPath;
    int regionWidth, regionHeight;
    ifstream configFile(configPath);
    configFile >> modelPath >> weightsPath >> meanPath;
    configFile >> regionWidth >> regionHeight >> batchSize >> threshold;
    configFile.close();

    regionSize = Size(regionWidth, regionHeight);
    convnet = Convnet(modelPath, weightsPath);
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

Mat extractYellowChannel(Mat &inputImage){
    Mat yellowImage(Size(inputImage.cols, inputImage.rows), CV_8UC1, Scalar(0));

    for(int i=0; i<inputImage.rows; i++){
        for(int j=0; j<inputImage.cols; j++){
            int yellowValue = 0;
            int blueValue = (int) inputImage.at<Vec3b>(i,j)[0];
            int greenValue = (int) inputImage.at<Vec3b>(i,j)[1];
            int redValue = (int) inputImage.at<Vec3b>(i,j)[2];
            int rgValue = (redValue + greenValue)/2;
            int sumColors = blueValue + greenValue + redValue;
            if((float)redValue/(float)sumColors > 0.35 && (float)greenValue/(float)sumColors > 0.35 && (float)blueValue/(float)sumColors < 0.3 && sumColors > 200){
                yellowValue = min(255, 255*(redValue + greenValue)/2/sumColors);
            }
            yellowImage.at<uchar>(i,j) = yellowValue;
        }
    }
    return yellowImage;
}

void Detector::genMSEREllipses(Mat &image, vector<ellipseParameters> &mserEllipses){
    extrema::ExtremaParams p;
    p.preprocess = 0;
    p.max_area = 0.03;
    p.min_size = 30;
    p.min_margin = 10;
    p.relative = 0;
    p.verbose = 0;
    p.debug = 0;
    double scale_factor = 1.0;
    scale_factor = scale_factor * 2; /* compensate covariance matrix */

    computeMSEREllipses(image, mserEllipses, p, scale_factor);
    Mat yellowChannel=extractYellowChannel(image);
    computeMSEREllipses(yellowChannel, mserEllipses, p, scale_factor);
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
    // int minArea = 500;
    // int maxArea = 1000000;
    int minArea = (image.rows * image.cols) / 10000;
    int maxArea = image.rows * image.cols;


    for(unsigned int i=0; i<mserEllipses.size(); i++){
        ellipseParameters tempEll = mserEllipses[i];
        float ellArea = PI * tempEll.axes.width * tempEll.axes.height;
        if(ellArea > minArea && ellArea < maxArea && /*(tempEll.angle < 25 || tempEll.angle > 155) && tempEll.axes.height > 0 && (float)tempEll.axes.width/(float)tempEll.axes.height > 1.5 &&*/ (float)tempEll.axes.width/(float)tempEll.axes.height < 10 )//Potential Number Plates
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
}

void remapRects(float scale, vector<RotatedRect> &allRects, vector<RotatedRect> &scaledRects){
    if (scale > 1) cout << "Scale is " << scale;
    for(RotatedRect r : allRects){
        RotatedRect temp(Point(ceil((r.center.x)/scale), ceil((r.center.y)/scale)), Size(floor((r.size.width-1)/scale)-2, floor((r.size.height-1)/scale)-2), r.angle);
        scaledRects.push_back(temp);
    }
}

Mat cropRegion(Mat &image, RotatedRect &rect){
    Mat M, rotated, cropped;
    while(rect.angle > 90){
        rect.angle = rect.angle-180;
    }
    while(rect.angle <=-90){
        rect.angle = rect.angle+180;
    }

    if(rect.angle > 45){
        int temp = rect.size.width;
        rect.size.width = rect.size.height;
        rect.size.height = temp;
        rect.angle = rect.angle - 90;
    }
    else if(rect.angle < -45){
        int temp = rect.size.width;
        rect.size.width = rect.size.height;
        rect.size.height = temp;
        rect.angle = rect.angle + 90;
    }
    
    Rect bound = rect.boundingRect();

    int pad = 0;
    if(bound.width > bound.height){
        pad = bound.width;
    }
    else{
        pad = bound.height;
    }
    
    Mat boundMat(image, bound);
    copyMakeBorder( boundMat, boundMat, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0,0,0) );
    
    Point center(rect.center.x - bound.x + pad, rect.center.y - bound.y + pad);
    M = getRotationMatrix2D(center, rect.angle, 1.0);
    warpAffine(boundMat, rotated, M, boundMat.size(), INTER_CUBIC);
    getRectSubPix(rotated, rect.size, center, cropped);
    return cropped;
}

void writeMat(Mat &mat){
    static int num = 0;
    ofstream file(string("debugFiles/mats/candidate_") + to_string(num) + ".mat");
    for(int i=0; i<mat.rows; i++){
        for(int j=0; j<mat.cols; j++){
            file << "(" << mat.at<Vec3f>(i,j)[0] << "," << mat.at<Vec3f>(i,j)[1] << "," << mat.at<Vec3f>(i,j)[2] << ") ";
        }
        file << endl;
    }
    file.close();
    num++;
}

Mat Detector::preprocessMat(Mat &input){
    Mat resized_bgr, resized_rgb, rescaled, normalized;
    resize(input, resized_bgr, regionSize);
    cv::cvtColor(resized_bgr, resized_rgb, CV_BGR2RGB);
    resized_rgb.convertTo(rescaled, CV_32FC3, 1.0/255);
    subtract(rescaled, mean, normalized);
    // writeMat(rescaled);
    // writeMat(mean);
    // writeMat(normalized);
    return normalized;
}

void Detector::detectNumPlates(Mat &inputImage, vector<Mat> &numPlateImgs, vector<RotatedRect> &numPlateBoxes){
    vector<RotatedRect> smalBoxes, mserBoxes;
    Mat smalImage;

    int maxSize = 750;
    float scale = -1;
    if(inputImage.rows > maxSize || inputImage.cols> maxSize){
        if(inputImage.rows > inputImage.cols)
            scale = (float) maxSize/inputImage.rows;
        else
            scale = (float) maxSize/inputImage.cols;
        Size dsize(round(scale*inputImage.cols), round(scale*inputImage.rows));
        resize(inputImage, smalImage, dsize);
    }

    genRegions(smalImage, smalBoxes);
    
    if (scale!=-1){
        remapRects(scale, smalBoxes, mserBoxes);
    }
    else{
        mserBoxes.insert(mserBoxes.end(), smalBoxes.begin(), smalBoxes.end());
    }

    Mat imageCopy = inputImage.clone();
    
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
            Mat candidateMat = cropRegion(inputImage, mserBoxes[readNo]);
            // imwrite(string("debugFiles/detect/b4processing_") + to_string(readNo) + ".jpg", candidateMat);  //DEBUG
            batchMats.push_back( preprocessMat(candidateMat));
            readNo++;
        }

        vector<float> batchScores = convnet.scoreBatch(batchMats);
        for(int i=0; i < curBatchSize; i++){
            if(batchScores[2*i] > threshold){
                Mat numPlateImg = cropRegion(imageCopy, mserBoxes[writeNo]);
                // imwrite(string("debugFiles/detect/numPlateImg_") + to_string(writeNo) + ".jpg", numPlateImg);   //DEBUG
                // RotatedRect fullRect = mserBoxes[writeNo];
                // RotatedRect halfRect;
                // halfRect.center = fullRect.center;
                // halfRect.size = Size(fullRect.size.width/2, fullRect.size.height/2);
                // halfRect.angle = fullRect.angle;
                // Mat numPlateImg = cropRegion(imageCopy, halfRect);
                numPlateImgs.push_back( numPlateImg);
                // numPlateBoxes.push_back( halfRect);
                numPlateBoxes.push_back(mserBoxes[writeNo]);
            }
            writeNo++;
        }
    }
    return;
}