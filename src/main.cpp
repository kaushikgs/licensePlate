//TODO: Check if msers being created match with torch

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include "Convnet.h"
#include "Detector.h"
#include "Reader.h"

using namespace std;
using namespace cv;

void drawResult(Mat &img, RotatedRect &box, string str){
    Scalar color(rand()%255, rand()%255, rand()%255);
    Point2f corners[4];
    box.points(corners);
    int thickness = ceil(img.cols/1000.0);
    for(int i=0; i<4; i++){
        line(img, corners[i], corners[(i+1)%4], color, thickness);
    }
    putText(img, str, corners[1], FONT_HERSHEY_SIMPLEX, thickness, color, thickness);
}

char imdisplay(Mat dispImg, string windowName = "Display"){
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, dispImg);
    char k = waitKey(0);
    destroyWindow(windowName);
    return k;
}

void displayResult(Mat &image, string windowName){
    Mat dispImage;
    int maxDispSize = 1000;
    float scale = -1;
    if(image.rows > maxDispSize || image.cols> maxDispSize){
        if(image.rows > image.cols)
            scale = (float) maxDispSize/image.rows;
        else
            scale = (float) maxDispSize/image.cols;
        Size dispSize(round(scale*image.cols), round(scale*image.rows));
        resize(image, dispImage, dispSize);
    }
    else{
        dispImage = image;
    }
    imdisplay(dispImage, windowName);
}

int main(int argc, char **argv){
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <image path>" << std::endl;
        return 1;
    }
    chrono::steady_clock::time_point start_t = chrono::steady_clock::now();
    srand(1);
    string imagePath = argv[1];
    size_t lastindex = imagePath.find_last_of("/");
    string imageName = imagePath.substr(lastindex);

    system("rm debugFiles/detect/*");   //DEBUG
    system("rm debugFiles/result/*"); //DEBUG
    system("rm -r debugFiles/read/*");   //DEBUG
    // system("rm -r debugFiles/mats/*");   //DEBUG

    Detector detector("detectConfig.txt");
    Reader reader("readConfig.txt");
    Mat image = imread(imagePath);

    chrono::steady_clock::time_point init_t = chrono::steady_clock::now();
    vector<Mat> numPlateImgs;
    vector<RotatedRect> numPlateBoxes;
    vector<bool> doubled;
    detector.detectNumPlates(image, imageName, numPlateImgs, numPlateBoxes, doubled);
    chrono::steady_clock::time_point detected_t = chrono::steady_clock::now();

    vector<string> numPlateStrs;
    int detNo = 0;
    for(Mat numPlateImg : numPlateImgs){    //TODO: use iterator for performance
        string numPlateStr = reader.readNumPlate( numPlateImg, imageName, doubled[detNo]);
        numPlateStrs.push_back(numPlateStr);
        detNo++;
    }
    chrono::steady_clock::time_point read_t = chrono::steady_clock::now();

    for(int boxNo=0; boxNo < numPlateBoxes.size(); boxNo++){
        RotatedRect numPlateBox = numPlateBoxes[boxNo];
        string numPlateStr = numPlateStrs[boxNo];
        drawResult(image, numPlateBox, numPlateStr);
    }
    chrono::steady_clock::time_point drawn_t = chrono::steady_clock::now();

    // displayResult(image, imagePath);
    imwrite("debugFiles/result/" + imageName + "_result.jpg", image);

    cout << numPlateImgs.size() << " number plates found in " << imagePath << endl;
    cout << "Initialisation: " << chrono::duration_cast<std::chrono::milliseconds> (init_t - start_t).count() << " ms" << endl;
    cout << "Detection: " << chrono::duration_cast<std::chrono::milliseconds> (detected_t - init_t).count() << " ms" << endl;
    cout << "Recognition: " << chrono::duration_cast<std::chrono::milliseconds> (read_t - detected_t).count() << " ms" << endl;
    cout << "Drawing results: " << chrono::duration_cast<std::chrono::milliseconds> (drawn_t - read_t).count() << " ms" << endl;
}
