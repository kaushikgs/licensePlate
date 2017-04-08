#include <opencv2/core/core.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include "mser.h"
#include "Convnet.h"
#include "Reader.h"

using namespace std;
using namespace cv;
using namespace caffe;
using namespace extrema;

char imdisplay3(Mat dispImg, string windowName = "Display"){
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, dispImg);
    char k = waitKey(0);
    destroyWindow(windowName);
    return k;
}

struct Candidate{
    Rect boundBox;
    char label;

    // Candidate(Rect &box, char labelChar){
    //     // this->label = labelChar;
    //     this -> label = 'A';
    //     boundBox = box;
    // }    

    Candidate(Rect &box, int labelCode){
        boundBox = box;
        if(labelCode < 10){
            label = '0' + labelCode;
        }
        else if(labelCode < 36){
            label = 'A' + labelCode - 10;
        }
        else{
            label = '_';    //should never happen
        }
    }

    bool operator<(const Candidate &b) const {
        Point c1 = (boundBox.tl() + boundBox.br()) * 0.5;
        Point c2 = (b.boundBox.tl() + b.boundBox.br()) * 0.5;

        // if(c1.y < c2.y) return true;
        // else if (c1.y > c2.y)   return false;

        if (c1.x < c2.x)   return true;
        else if (c1.x > c2.x)   return false;
        
        return false;
    }
};

Reader::Reader(string modelPath, string weightsPath, string meanPath, int regionWidth, int regionHeight)
 : batchSize(64), regionSize(regionWidth, regionHeight), convnet(modelPath, weightsPath)
{
	this->meanPath = meanPath;
    setMean(meanPath);
    numClasses = 37;
}

/* Load the mean file in binaryproto format. */
void Reader::setMean(const string& meanPath) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(meanPath.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    // CHECK_EQ(mean_blob.channels(), num_channels_)
    //   << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    // for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    // }

    /* Merge the separate channels into a single image. */
    // cv::Mat mean;
    cv::merge(channels, mean);
    // mean_ = mean;

    // cout << "Printing the mean -----------------" << endl;
    // for(int i=0; i<regionSize.height; i++){
    //     for(int j=0; j<regionSize.width; j++){
    //         cout << mean.at<float>(i,j) << " ";
    //     }
    //     cout << endl;
    // }
    // cout << "-----------------------------------" << endl;
}

vector<RLERegion> importRLEVector(string tempRLEPath){
    vector<RLERegion> rle_vector;
    ifstream rleFile(tempRLEPath);

    for(int k = 0; k<2; k++){
        int numRegns;
        rleFile >> numRegns;
        for(int i=0; i < numRegns; i++)
        { 
            int regnLines;
            rleFile >> regnLines;
            RLERegion newRLE;
            for (int j=0; j < regnLines; j++){
                RLEItem newLine;
                rleFile >> newLine.line >> newLine.col1 >> newLine.col2;
                newRLE.rle.push_back(newLine);
            }
            rle_vector.push_back(newRLE);
        }
    }
    rleFile.close();
    return rle_vector;
}

void nms(const std::vector<cv::Rect>& srcRects, std::vector<int>& resIdxs, float thresh)
{
    resIdxs.clear();

    const size_t size = srcRects.size();
    if (!size)
    {
        return;
    }

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    std::multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.insert(std::pair<int, size_t>(srcRects[i].area(), i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];

        resIdxs.push_back(lastElem->second);

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

            float intArea = (rect1 & rect2).area();
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                pos = idxs.erase(pos);
            }
            else
            {
                ++pos;
            }
        }
    }
}

void genMSERRLEs(Mat &image, vector<RLERegion> &mserRLEs, vector<Rect> &mserBoxes){
    // extrema::ExtremaParams p;
    // p.preprocess = 0;
    // p.max_area = 0.1;
    // p.min_size = 30;
    // p.min_margin = 10;
    // p.relative = 0;
    // p.verbose = 0;
    // p.debug = 0;
    // double scale_factor = 1.0;
    // scale_factor = scale_factor * 2; /* compensate covariance matrix */

    // computeMSERRLEs(image, mserRLEs, mserBoxes, p, scale_factor);
    string tempImgPath = "temp/img.jpg";
    string tempRLEPath = "temp/rle.txt";
    imwrite(tempImgPath, image);
    // imdisplay3(image);

    string cmd = "/home/kaushik/arun/code/MSER/Untitled\\ Folder/extrema-edu/extrema/extrema-bin";
    cmd = cmd + " -per 0.1 -i " + tempImgPath + " -o " + tempRLEPath + " -t 0";
    system(cmd.c_str());
    
    vector<RLERegion> allRles = importRLEVector(tempRLEPath);
    vector<Rect> allRects;
    convRleToRect(allRles, allRects);

    vector<int> fewIdxs;
    nms(allRects, fewIdxs, 0.5);
    for(int i : fewIdxs){
        mserRLEs.push_back(allRles[i]);
        mserBoxes.push_back(allRects[i]);
    }
}

Mat Reader::makeMatFrmRLE(RLERegion &region, Rect &boundBox){
    Mat outImg(boundBox.height, boundBox.width, CV_8UC1, cv::Scalar(255));  //white bg
    vector<RLEItem> regionItems = region.rle;

    for(RLEItem item : region.rle){
        for(int col = item.col1; col <= item.col2; col++){
            int x = col - boundBox.x;
            int y = item.line - boundBox.y;
            outImg.at<uchar>(y, x) = 0;
        }
    }

    resize(outImg, outImg, regionSize);
    int pad = 5;
    copyMakeBorder( outImg, outImg, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(255));
    return outImg;
}

string makeNumPlateStr(Mat &img, vector<Candidate> &selectedCandidates){
    // sort(selectedCandidates.begin(), selectedCandidates.end()); //TODO: this will have NLogN object copies. Use pointers instead.
    // string numPlateStr = "";
    // char lastChar = ' ';
    // for(vector<Candidate>::iterator iter = selectedCandidates.begin(); iter != selectedCandidates.end(); iter++){
    //     if(iter->label != lastChar)
    //         numPlateStr = numPlateStr + iter->label;
    //     lastChar = iter->label;
    // }

    float avgWidth;
    int wsum = 0;
    for(Candidate c : selectedCandidates){
        wsum = wsum + c.boundBox.width;
    }
    avgWidth = (float) wsum / selectedCandidates.size();

    vector<Candidate> filtered;
    for(Candidate c : selectedCandidates){
        if(c.boundBox.width < (1.5 * avgWidth))
            filtered.push_back(c);
    }

    vector<Candidate> top, bot;
    int mid = img.rows / 2;
    for(Candidate c : filtered){
        Point center = (c.boundBox.tl() + c.boundBox.br()) * 0.5;
        if(center.y < mid)  top.push_back(c);
        else                bot.push_back(c);
    }

    float topAvg, botAvg;
    int sumTop = 0, sumBot = 0;
    for(Candidate c : top){
        Point center = (c.boundBox.tl() + c.boundBox.br()) * 0.5;
        sumTop = sumTop + c.boundBox.y;
    }
    for(Candidate c : bot){
        Point center = (c.boundBox.tl() + c.boundBox.br()) * 0.5;
        sumBot = sumBot + c.boundBox.y;
    }

    topAvg = (float) sumTop / top.size();
    botAvg = (float) sumBot / bot.size();
    string str = "";
    char prevChar = ' ';
    if((botAvg - topAvg) < (img.rows / 10)){
        sort(filtered.begin(), filtered.end());
        for(Candidate c : filtered){
            // if(c.label != prevChar)
                str = str + c.label;
            prevChar = c.label;
        }
    }
    else{
        sort(top.begin(), top.end());
        sort(bot.begin(), bot.end());
        for(Candidate c : top){
            // if(c.label != prevChar)
                str = str + c.label;
            prevChar = c.label;
        }
        for(Candidate c : bot){
            // if(c.label != prevChar)
                str = str + c.label;
            prevChar = c.label;
        }
    }

    return str;
}

void printMat1(Mat &mat){
    cout << "Printing the CV_8U mat -----------------" << endl;
    for(int i=0; i<mat.rows; i++){
        for(int j=0; j<mat.cols; j++){
            cout << (int) mat.at<uchar>(i,j) << " ";
        }
        cout << endl;
    }
    cout << "-----------------------------------" << endl;
}

void printMat2(Mat &mat){
    cout << "Printing the CV_32F mat -----------------" << endl;
    for(int i=0; i<mat.rows; i++){
        for(int j=0; j<mat.cols; j++){
            cout << mat.at<float>(i,j) << " ";
        }
        cout << endl;
    }
    cout << "-----------------------------------" << endl;
}

string Reader::readNumPlate(Mat &numPlateImg){
	vector<RLERegion> mserRLEs;
    vector<Rect> mserBoxes;
    genMSERRLEs(numPlateImg, mserRLEs, mserBoxes);
    
    mkdir("result", 0777);
    for(int i=0; i<37; i++){
        mkdir((string("result/") + to_string(i) + "/").c_str(), 0777);
    }
    // for(int i = 0; i < 26; i++){
    //     mkdir((string("result/") + to_string('A' + i) + "/").c_str(), 0777);
    // }
    // mkdir((string("result/") + "," + "/").c_str(), 0777);

    int numBatches = ceil( ((float) mserRLEs.size()) / batchSize);
    vector<Mat> batchMats;
    vector<Candidate> selectedCandidates;
    
    for(int batchNo = 0, readNo = 0, writeNo = 0; batchNo < numBatches; batchNo++){
        int curBatchSize;
        if(batchNo == (numBatches - 1) && ((mserRLEs.size() % batchSize) != 0))
            curBatchSize = mserRLEs.size() % batchSize;
        else
            curBatchSize = batchSize;

        batchMats.clear();
        for(int i = 0; i < curBatchSize; i++){
            Mat candidateMat = makeMatFrmRLE(mserRLEs[readNo], mserBoxes[readNo]);
            imwrite(string("debugFiles/read/candidate_") + to_string(readNo) + ".jpg", candidateMat);   //DEBUG
            // printMat1(candidateMat);
            candidateMat.convertTo(candidateMat, CV_32FC1);
            // printMat2(candidateMat);
            subtract(candidateMat, mean, candidateMat);
            // cout << "after mean subtraction type " << candidateMat.type() << endl;
            // printMat2(candidateMat);
            // cout << candidateMat.channels() << " channels vs. " << mean.channels() << endl;
            // cout << candidateMat.size() << " size vs. " << mean.size() << endl;
            // cout << candidateMat.type() << " type vs. " << mean.type() << endl;
            batchMats.push_back(candidateMat);
            readNo++;
        }

        vector<float> batchScores = convnet.scoreBatch(batchMats);
        for(int i=0; i < curBatchSize; i++){
            vector<float>::iterator begin = batchScores.begin() + (i * numClasses);
            vector<float>::iterator end = batchScores.begin() + (i * numClasses) + numClasses;
            
            auto ptr = max_element(begin, end);
            int labelCode;
            float threshold = 0.8;
            if(*ptr < threshold)
                labelCode = numClasses-1;
            else
                labelCode = ptr - begin;
            // char label = 'A'; //imdisplay2(Mat(numPlateImg, mserBoxes[batchNo * batchSize + i]), batchMats[i]);
            imwrite(string("result/") + to_string(labelCode) + "/" + to_string(writeNo) + ".jpg", makeMatFrmRLE(mserRLEs[writeNo], mserBoxes[writeNo]));    //DEBUG

            // if(label != ',')
            //     selectedCandidates.push_back(Candidate(mserBoxes[batchNo * batchSize + i], label));
            // labelCode = 10;
            
            if(labelCode != (numClasses-1)) // not none class
                selectedCandidates.push_back( Candidate(mserBoxes[writeNo], labelCode));

            writeNo++;
        }
    }

    string numPlateStr = makeNumPlateStr(numPlateImg, selectedCandidates);

    for(Candidate c : selectedCandidates){
        Scalar color(rand()%255, rand()%255, rand()%255);
        rectangle(numPlateImg, c.boundBox, color, 2);
        putText(numPlateImg, string(1, c.label), c.boundBox.tl(), FONT_HERSHEY_SIMPLEX, 2, color, 2);
    }
    imwrite(string("debugFiles/read/numPlate_") + to_string(rand()%255) + ".jpg", numPlateImg);
    
    return numPlateStr;
}

// string Reader::readNumPlate(Mat &numPlateImg){
//     return "dummy";
// }