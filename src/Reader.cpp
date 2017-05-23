#include <opencv2/core/core.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <cmath>
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

Reader::Reader(string configPath) {
	string modelPath, weightsPath, meanPath;
    int regionWidth, regionHeight;
    ifstream configFile(configPath);
    configFile >> modelPath >> weightsPath >> meanPath;
    configFile >> regionWidth >> regionHeight >> batchSize >> threshold;
    configFile.close();

    regionSize = Size(regionWidth, regionHeight);
    convnet = Convnet(modelPath, weightsPath);
    setMean(meanPath);

    numDetections = 0;  //DEBUG
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
    cv::Mat mean_;
    cv::merge(channels, mean_);
    mean = mean_;

    // cv::Scalar channel_mean = cv::mean(mean_);
    // Size meanSize(40, 110);
    // mean = cv::Mat(meanSize, mean_.type(), channel_mean);

    // cout << "Printing the mean -----------------" << endl;
    // for(int i=0; i<regionSize.height; i++){
    //     for(int j=0; j<regionSize.width; j++){
    //         cout << mean.at<float>(i,j) << " ";
    //     }
    //     cout << endl;
    // }
    // cout << "-----------------------------------" << endl;
}

void filterMSERs(vector<RLERegion> &allRles, vector<Rect> &allRects){
    auto iterRle = allRles.begin();
    for(auto iterRect = allRects.begin(); iterRect != allRects.end(); ){
        if(iterRect->width > 2*(iterRect->height)){
            iterRect = allRects.erase(iterRect);
            iterRle = allRles.erase(iterRle);
        }
        else{
            iterRect++;
            iterRle++;
        }
    }
}

void nms(const std::vector<cv::Rect>& srcRects, std::vector<int>& resIdxs, float thresh)
{
    resIdxs.clear();

    const size_t size = srcRects.size();
    if (!size)
    {
        return;
    }

    // Sort the bounding boxes by the area of the bounding box
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
    extrema::ExtremaParams p;
    p.preprocess = 0;
    p.max_area = 0.1;
    p.min_size = 30;
    p.min_margin = 10;
    p.relative = 0;
    p.verbose = 0;
    p.debug = 0;
    double scale_factor = 1.0;
    scale_factor = scale_factor * 2; /* compensate covariance matrix */

    // string tempImgPath = "temp/img.jpg";
    // string tempRLEPath = "temp/rle.txt";
    // imwrite(tempImgPath, image);
    
    // string cmd = "/home/kaushik/arun/code/MSER/Untitled\\ Folder/extrema-edu/extrema/extrema-bin";
    // cmd = cmd + " -per 0.1 -ms 10 -i " + tempImgPath + " -o " + tempRLEPath + " -t 0";
    // system(cmd.c_str());
    
    // vector<RLERegion> allRles = importRLEVector(tempRLEPath);
    vector<RLERegion> allRles;
    vector<Rect> allRects;
    // convRleToRect(allRles, allRects);
    computeMSERRLEs(image, allRles, allRects, p, scale_factor);

    filterMSERs(allRles, allRects);
  
    vector<int> fewIdxs;
    nms(allRects, fewIdxs, 0.6);
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

float getModeWidth(vector<Candidate> &selectedCandidates, int imgWidth){
    // float avgWidth;
    // int wsum = 0;
    // for(Candidate c : selectedCandidates){
    //     wsum = wsum + c.boundBox.width;
    // }
    // avgWidth = (float) wsum / selectedCandidates.size();
    // return avgWidth;

    for(int window = 5; window < imgWidth/4; window++){
        vector<int> histogram( imgWidth/window + 1, 0);
        for(Candidate c : selectedCandidates){
            int binNo = c.boundBox.width / window;
            histogram[binNo] = histogram[binNo] + 1;
        }
        auto maxIter = max_element(histogram.begin(), histogram.end());
        int maxNo = *maxIter;
        int sum=0;
        if((maxNo / (float) selectedCandidates.size()) > 0.5){
            int maxBinNo = maxIter - histogram.begin();
            for(Candidate c : selectedCandidates){
                int binNo = floor(c.boundBox.width / window);
                if(binNo == maxBinNo){
                    sum = sum + c.boundBox.width;
                }
            }
            return (float) sum / maxNo;
        }
    }

    return -1;
}

bool areaCompare(Candidate *a, Candidate *b){
    return (a->boundBox.area() > b->boundBox.area());
}

void filterCandidates(vector<Candidate> &candidates, vector<Candidate> &filtered, int imgWidth){
    float modeWidth = getModeWidth(candidates, imgWidth);
    if (modeWidth == -1){
        return;
    }

    // for(Candidate c : candidates){
    //     if(c.boundBox.width < (1.5 * modeWidth))
    //         filtered.push_back(c);
    // }

    vector<Candidate*> all, chosen, notchosen;
    for(int i=0; i<candidates.size(); i++){
        all.push_back(&candidates[i]);
    }

    sort(all.begin(), all.end(), areaCompare);

    for(int i=0; i<all.size(); i++){
        Rect small = all[i]->boundBox;
        bool add = true;
        for(auto iter = chosen.begin(); iter!=chosen.end();){
            Rect big = (*iter)->boundBox;
            if((big & small) == small){
                if(abs(big.width - modeWidth) > abs(small.width - modeWidth)){  //small is better
                    chosen.erase(iter);
                }
                else{
                    add = false;
                    break;
                }
            }
            else{
                iter++;
            }
        }
        if(add){
            chosen.push_back(all[i]);
        }
    }

    for(Candidate *c : chosen){
        if (c->boundBox.width < (2*modeWidth))
            filtered.push_back(*c);
    }
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

    vector<Candidate> filtered;
    filterCandidates(selectedCandidates, filtered, img.cols);

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

void printProbs(vector<float> &probs){
    for(int i=0; i<10; i++){
        cout << i << ": " << probs[i] << endl;
    }
    int cint = 10;
    for(char c = 'A'; c <= 'Z'; c++){
        cout << c << ": " << probs[cint] << endl;
        cint++;
    }
    cout << "None" << ": " << probs[cint] << endl;
}

void drawRegions(Mat &img, string imageName, int numDetections, vector<Rect> &boxes){
    Mat drawImg = img.clone();
    for(Rect r : boxes){
        int thickness = ceil(img.cols / 1000.0);
        Scalar color(rand()%200, rand()%200, rand()%200);   //avoid whitey colors
        rectangle(drawImg, r, color, thickness);
    }
    imwrite(string("debugFiles/read/") + imageName + "_candidates_" + to_string(numDetections) + ".jpg", drawImg);
}

void drawResult(Mat &img, string imageName, int numDetections, vector<Candidate> &candidates){
    Mat drawImg = img.clone();
    for(Candidate c : candidates){
        int thickness = ceil(img.cols / 1000.0);
        Scalar color(rand()%200, rand()%200, rand()%200);   //avoid whitey colors
        rectangle(drawImg, c.boundBox, color, thickness);
        putText(drawImg, string(1, c.label), c.boundBox.tl(), FONT_HERSHEY_SIMPLEX, thickness, color, thickness);
    }
    imwrite(string("debugFiles/read/") + imageName + "_numplate_" + to_string(numDetections) + ".jpg", drawImg);
}

string Reader::readNumPlate(Mat &numPlateImg, string imageName){
    int maxCols = 500;
    if(numPlateImg.cols > maxCols){
        float scale = (float) maxCols / numPlateImg.cols;
        Size newSize = Size(numPlateImg.cols * scale, numPlateImg.rows * scale);
        resize(numPlateImg, numPlateImg, newSize);
    }

    vector<RLERegion> mserRLEs;
    vector<Rect> mserBoxes;
    genMSERRLEs(numPlateImg, mserRLEs, mserBoxes);
    
#ifdef DEBUG
    drawRegions(numPlateImg, imageName, numDetections, mserBoxes);
    for(int i=0; i<37; i++){
        mkdir((string("debugFiles/read/") + to_string(i) + "/").c_str(), 0777);
    }
    mkdir("debugFiles/read/candidates/", 0777);
#endif /* DEBUG */

    int numBatches = ceil( ((float) mserRLEs.size()) / batchSize);
    vector<Mat> batchMats;
    vector<Candidate> selectedCandidates;
    // cout << "Num candidates " << mserRLEs.size() << " Num batches " << numBatches << endl;
    
    for(int batchNo = 0, readNo = 0, writeNo = 0; batchNo < numBatches; batchNo++){
        int curBatchSize;
        if(batchNo == (numBatches - 1) && ((mserRLEs.size() % batchSize) != 0))
            curBatchSize = mserRLEs.size() % batchSize;
        else
            curBatchSize = batchSize;

        batchMats.clear();
        for(int i = 0; i < curBatchSize; i++){
            Mat candidateMat = makeMatFrmRLE(mserRLEs[readNo], mserBoxes[readNo]);
#ifdef DEBUG
            imwrite(string("debugFiles/read/candidates/") + imageName + "_candidate_" + to_string(numDetections) + "_" + to_string(readNo) + ".jpg", candidateMat);   //DEBUG
#endif /* DEBUG */
            candidateMat.convertTo(candidateMat, CV_32FC1);
            subtract(candidateMat, mean, candidateMat);
            batchMats.push_back(candidateMat);
            readNo++;
        }

        vector<float> batchScores = convnet.scoreBatch(batchMats);
        for(int i=0; i < curBatchSize; i++){
            vector<float>::iterator begin = batchScores.begin() + (i * numClasses);
            vector<float>::iterator end = batchScores.begin() + (i * numClasses) + numClasses;
            
            auto ptr = max_element(begin, end);
            int labelCode;
            if(*ptr < threshold)
                labelCode = numClasses-1;
            else
                labelCode = ptr - begin;
#ifdef DEBUG
            imwrite(string("debugFiles/read/") + to_string(labelCode) + "/" + imageName + "_" + to_string(numDetections) + "_" + to_string(writeNo) + ".jpg", makeMatFrmRLE(mserRLEs[writeNo], mserBoxes[writeNo]));
#endif /* DEBUG */

            if(labelCode != (numClasses-1)) // not none class
                selectedCandidates.push_back( Candidate(mserBoxes[writeNo], labelCode));

            writeNo++;
        }
    }

    string numPlateStr = makeNumPlateStr(numPlateImg, selectedCandidates);

#ifdef DEBUG
    drawResult(numPlateImg, imageName, numDetections, selectedCandidates);
#endif /* DEBUG */
    
    numDetections++;
    return numPlateStr;
}

// string Reader::readNumPlate(Mat &numPlateImg){
//     return "dummy";
// }