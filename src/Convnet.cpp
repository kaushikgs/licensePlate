#include <cstdlib>
#include <vector>
#include <string>
#include "Convnet.h"

using namespace std;
using namespace cv;
using namespace caffe;

Convnet::Convnet(){
    
}

Convnet::Convnet(string modelPath, std::string weightsPath, int maxBatchSize){
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif
    this->maxBatchSize = maxBatchSize;

    /* Load the network. */
    net_.reset(new Net<float>(modelPath, TEST));
    net_->CopyTrainedLayersFrom(weightsPath);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Convnet::WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch){
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float* input_data = input_layer->mutable_cpu_data();
    for ( int j = 0; j < num; j++){
        vector<cv::Mat> input_channels;
        for (int i = 0; i < input_layer->channels(); ++i){
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += width * height;
        }
        input_batch -> push_back(vector<cv::Mat>(input_channels));
    }
}

void Convnet::PreprocessBatch(const vector<cv::Mat> imgs,
                                 std::vector< std::vector<cv::Mat> >* input_batch){
    for (int i = 0 ; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        std::vector<cv::Mat> *input_channels = &(input_batch->at(i));

        /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
        cv::split(img, *input_channels);

//      CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
//      == net_->input_blobs()[0]->cpu_data())
//      << "Input channels are not wrapping the input layer of the network.";
    }
}

vector<float> Convnet::scoreBatch(const std::vector<cv::Mat> imgs){
    Blob<float>* input_layer = net_->input_blobs()[0];

    input_layer->Reshape(imgs.size(), num_channels_,
                         input_geometry_.height,
                         input_geometry_.width);

    /* Forward dimension change to all layers. */
    net_->Reshape();

    vector<vector<Mat>> input_batch;
    WrapBatchInputLayer(&input_batch);

    PreprocessBatch(imgs, &input_batch);

    net_->ForwardPrefilled();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels()*imgs.size();
    return vector<float>(begin, end);
}

// vector<float> Convnet::scoreBatch(const std::vector<cv::Mat> imgs, int numClasses){
//     vector<float> randomVec(numClasses*imgs.size());
//     for(int i=0; i<randomVec.size(); i++){
//         randomVec[i] = (rand()%100) / 100.0;
//     }
//     return randomVec;
// }