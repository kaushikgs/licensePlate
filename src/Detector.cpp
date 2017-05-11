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

void adjustRect(RotatedRect &rect){
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
}

bool liesInside(Mat &img, RotatedRect &rect){
    Rect bound = rect.boundingRect();

    if(bound.x < 0 || bound.y < 0){
        return false;
    }
    if(bound.x + bound.width > img.cols || bound.y + bound.height > img.rows){
        return false;
    }
    return true;
}

//return true if either right half or left half or both lie inside
bool partialLiesInside(Mat &img, RotatedRect &rect){
    if(liesInside(img, rect)){
        return true;
    }

    RotatedRect quatrRect = rect;
    quatrRect.size.width = rect.size.width/2;
    quatrRect.size.height = rect.size.height/2;
    if(!liesInside(img, quatrRect)){
        return false;
    }

    double angleRadians = ((double) rect.angle * CV_PI) /180;
    float costheta = cos(angleRadians);
    float sintheta = sin(angleRadians);
    Size halfSize(rect.size.width/2, rect.size.height);

    Point rightCenter;
    rightCenter.x = rect.center.x + (rect.size.width * costheta)/4;
    rightCenter.y = rect.center.y + (rect.size.width * sintheta)/4;
    RotatedRect rightHalf(rightCenter, halfSize, rect.angle);
    if(liesInside(img, rightHalf)){
        return true;
    }

    Point leftCenter;
    leftCenter.x = rect.center.x - (rect.size.width * costheta)/4;
    leftCenter.y = rect.center.y - (rect.size.width * sintheta)/4;
    RotatedRect leftHalf(leftCenter, halfSize, rect.angle);
    if(liesInside(img, leftHalf)){
        return true;
    }

    return false;
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
            adjustRect(mserRect);
            if (partialLiesInside(image, mserRect)){
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

//pads, rotates and returns new center
Point rotateMat(Mat &image, Point &center, float angle, Mat &rotated){
    int pad = 0;
    if(image.cols > image.rows){
        pad = image.cols;
    }
    else{
        pad = image.rows;
    }
    
    Mat boundMat;
    copyMakeBorder( image, boundMat, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0,0,0) );
    Point newCenter(center.x + pad, center.y + pad);
    Mat M = getRotationMatrix2D(newCenter, angle, 1.0);
    warpAffine(boundMat, rotated, M, boundMat.size(), INTER_CUBIC);
    return newCenter;
}

//if the entire region is present, return it
//else, one of the halves should be present - then make other half as mirror image
//of the known half, concatenate and return
Mat cropRegion(Mat &image, RotatedRect &rect){
    Mat rotated, cropped;
    
    Rect bound = rect.boundingRect();

    Point center, newCenter;
    if(liesInside(image, rect)){
        Mat boundMat(image, bound);
        center = Point(rect.center.x - bound.x, rect.center.y - bound.y);
        newCenter = rotateMat(boundMat, center, rect.angle, rotated);
        getRectSubPix(rotated, rect.size, newCenter, cropped);
        return cropped;
    }

    else{   //one of the halves should be in the image
        double angleRadians = ((double) rect.angle * CV_PI) /180;
        float costheta = cos(angleRadians);
        float sintheta = sin(angleRadians);
        
        Point leftCenter, rightCenter;
        rightCenter.x = rect.center.x + (rect.size.width * costheta)/4;
        rightCenter.y = rect.center.y + (rect.size.width * sintheta)/4;
        leftCenter.x = rect.center.x - (rect.size.width * costheta)/4;
        leftCenter.y = rect.center.y - (rect.size.width * sintheta)/4;

        Size halfSize(rect.size.width/2, rect.size.height);
        RotatedRect rightHalf(rightCenter, halfSize, rect.angle);
        RotatedRect leftHalf(leftCenter, halfSize, rect.angle);

        Mat rightMat, leftMat;
        if(liesInside(image, rightHalf)){
            rightMat = cropRegion(image, rightHalf);
            flip(rightMat, leftMat, 1);
        }
        else if(liesInside(image, leftHalf)){
            leftMat = cropRegion(image, leftHalf);
            flip(leftMat, rightMat, 1);
        }
        else{
            cout << "MSER box without eithr half filtered" << endl;
        }
        hconcat(leftMat, rightMat, cropped);
        return cropped;
    }
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
    Mat imageCopy = inputImage.clone();
    
    if(inputImage.rows > maxSize || inputImage.cols> maxSize){
        if(inputImage.rows > inputImage.cols)
            scale = (float) maxSize/inputImage.rows;
        else
            scale = (float) maxSize/inputImage.cols;
        Size dsize(round(scale*inputImage.cols), round(scale*inputImage.rows));
        resize(inputImage, smalImage, dsize);
    }
    else
    	smalImage = inputImage;

    genRegions(smalImage, smalBoxes);
    
    if (scale!=-1){
        remapRects(scale, smalBoxes, mserBoxes);
    }
    else{
        mserBoxes.insert(mserBoxes.end(), smalBoxes.begin(), smalBoxes.end());
    }
    
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
#ifdef DEBUG
            imwrite(string("debugFiles/detect/b4processing_") + to_string(readNo) + ".jpg", candidateMat);
#endif /* DEBUG */
            batchMats.push_back( preprocessMat(candidateMat));
            readNo++;
        }

        vector<float> batchScores = convnet.scoreBatch(batchMats);
        for(int i=0; i < curBatchSize; i++){
            if(batchScores[2*i] > threshold){
                RotatedRect fullRect = mserBoxes[writeNo];
                if(liesInside(imageCopy, fullRect)){
                    Mat numPlateImg = cropRegion(imageCopy, fullRect);
                    numPlateImgs.push_back( numPlateImg);
                    numPlateBoxes.push_back( fullRect);
                }
                else{
                    RotatedRect halfRect = fullRect;
                    halfRect.size.width = fullRect.size.width/2;
                    halfRect.size.height = fullRect.size.height/2;
                    Mat numPlateImg = cropRegion(imageCopy, halfRect);
                    numPlateImgs.push_back( numPlateImg);
                    numPlateBoxes.push_back( halfRect);
                }
            }
            writeNo++;
        }
    }
    return;
}
