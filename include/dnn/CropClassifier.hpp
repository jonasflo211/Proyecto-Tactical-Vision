#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct CropPrediction {
    int classId = -1;
    float confidence = 0.0f;
    std::vector<float> probabilities;
};

class CropClassifier {
public:
    bool load(const std::string& onnxPath,
              const cv::Size& inputSize = cv::Size(64, 128),
              bool useOpenCL = false);
    bool empty() const { return net.empty(); }
    CropPrediction predict(const cv::Mat& frame, const cv::Rect& bbox) const;

private:
    mutable cv::dnn::Net net;
    cv::Size inputSize = cv::Size(64, 128);

    static std::vector<float> softmax(const cv::Mat& logits);
};
