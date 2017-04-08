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

struct Configuration{
    string detectModelPath, detectWeightsPath, detectMeanPath;
    string readModelPath, readWeightsPath, readMeanPath;
    int detectRegionWidth, detectRegionHeight, readRegionWidth, readRegionHeight;

    Configuration(string configPath){
        ifstream configFile(configPath);
        configFile >> detectModelPath >> detectWeightsPath >> detectMeanPath;
        configFile >> detectRegionWidth >> detectRegionHeight;
        configFile >> readModelPath >> readWeightsPath >> readMeanPath;
        configFile >> readRegionWidth >> readRegionHeight;
        configFile.close();
    }
};

void drawResult(Mat &img, RotatedRect &box, string str){
    Scalar color(rand()%255, rand()%255, rand()%255);
    Point2f corners[4];
    box.points(corners);
    for(int i=0; i<4; i++){
        line(img, corners[i], corners[(i+1)%4], color, 5);
    }
    putText(img, str, corners[1], FONT_HERSHEY_SIMPLEX, 4, color, 4);
}

char imdisplay(Mat dispImg, string windowName = "Display"){
    Size oldSize = dispImg.size();
    Size newSize(oldSize.width/3, oldSize.height/3);
    resize(dispImg, dispImg, newSize);
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, dispImg);
    char k = waitKey(0);
    destroyWindow(windowName);
    return k;
}

int main(int argc, char **argv){
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <image path>" << std::endl;
        return 1;
    }
    chrono::steady_clock::time_point start_t = chrono::steady_clock::now();
    srand(1);
    string imagePath = argv[1];
    Configuration config(argv[2]);

    system("rm debugFiles/detect/*");   //DEBUG
    system("rm debugFiles/read/*"); //DEBUG
    system("rm -r result/*");   //DEBUG

    Detector detector(config.detectModelPath, config.detectWeightsPath, config.detectMeanPath, config.detectRegionWidth, config.detectRegionHeight);
    Reader reader(config.readModelPath, config.readWeightsPath, config.readMeanPath, config.readRegionWidth, config.readRegionHeight);
    Mat image = imread(imagePath);

    chrono::steady_clock::time_point init_t = chrono::steady_clock::now();
    vector<Mat> numPlateImgs;
    vector<RotatedRect> numPlateBoxes;
    detector.detectNumPlates(image, numPlateImgs, numPlateBoxes);
    chrono::steady_clock::time_point detected_t = chrono::steady_clock::now();

    vector<string> numPlateStrs;
    for(Mat numPlateImg : numPlateImgs){    //TODO: use iterator for performance
        string numPlateStr = reader.readNumPlate( numPlateImg);
        numPlateStrs.push_back(numPlateStr);
    }
    chrono::steady_clock::time_point read_t = chrono::steady_clock::now();

    for(int boxNo=0; boxNo < numPlateBoxes.size(); boxNo++){
        RotatedRect numPlateBox = numPlateBoxes[boxNo];
        string numPlateStr = numPlateStrs[boxNo];
        drawResult(image, numPlateBox, numPlateStr);
    }
    chrono::steady_clock::time_point drawn_t = chrono::steady_clock::now();

    imdisplay(image, imagePath);
    // imwrite("result.jpg", image);

    cout << numPlateBoxes.size() << " number plates found in " << imagePath << endl;
    cout << "Initialisation: " << chrono::duration_cast<std::chrono::milliseconds> (init_t - start_t).count() << " ms" << endl;
    cout << "Detection: " << chrono::duration_cast<std::chrono::milliseconds> (detected_t - init_t).count() << " ms" << endl;
    cout << "Recognition: " << chrono::duration_cast<std::chrono::milliseconds> (read_t - detected_t).count() << " ms" << endl;
    cout << "Drawing results: " << chrono::duration_cast<std::chrono::milliseconds> (drawn_t - read_t).count() << " ms" << endl;
}
