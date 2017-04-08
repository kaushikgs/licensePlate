#ifndef CONVNET_H
#define CONVNET_H

#include <memory>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>

class Convnet{
	std::string weightsPath;
	std::string modelPath;
	int maxBatchSize;

	std::shared_ptr<caffe::Net<float>> net_;	//TODO: does boost::shared_ptr work better
    cv::Size input_geometry_;
    int num_channels_;

public:
	Convnet(std::string modelPath, std::string weightsPath, int maxBatchSize = 64);
	std::vector<float> scoreBatch(const std::vector<cv::Mat> imgs);

private:
    // void SetMean(const string& mean_file);
    void WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch);
    void PreprocessBatch(const std::vector<cv::Mat> imgs,
                         std::vector< std::vector<cv::Mat> >* input_batch);

// private:
//     cv::Mat mean_;
//     int max_batchSize_;
};

#endif