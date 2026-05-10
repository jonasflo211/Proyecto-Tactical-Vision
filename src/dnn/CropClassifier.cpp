#include "../../include/dnn/CropClassifier.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

bool CropClassifier::load(const std::string& onnxPath, const cv::Size& size, bool useOpenCL) {
    try {
        net = cv::dnn::readNetFromONNX(onnxPath);
    } catch (const cv::Exception&) {
        net = cv::dnn::Net();
        return false;
    }
    inputSize = size;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(useOpenCL ? cv::dnn::DNN_TARGET_OPENCL : cv::dnn::DNN_TARGET_CPU);
    return !net.empty();
}

CropPrediction CropClassifier::predict(const cv::Mat& frame, const cv::Rect& bbox) const {
    CropPrediction pred;
    if (net.empty() || frame.empty()) return pred;

    cv::Rect clip = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (clip.area() <= 0) return pred;

    cv::Mat crop = frame(clip);
    cv::Mat blob = cv::dnn::blobFromImage(
        crop,
        1.0 / 255.0,
        inputSize,
        cv::Scalar(),
        true,
        false,
        CV_32F);

    net.setInput(blob);
    cv::Mat logits = net.forward();
    pred.probabilities = softmax(logits);
    if (pred.probabilities.empty()) return pred;

    auto it = std::max_element(pred.probabilities.begin(), pred.probabilities.end());
    pred.classId = (int)std::distance(pred.probabilities.begin(), it);
    pred.confidence = *it;
    return pred;
}

std::vector<float> CropClassifier::softmax(const cv::Mat& logits) {
    cv::Mat flat = logits.reshape(1, 1);
    if (flat.empty()) return {};

    std::vector<float> values(flat.cols);
    float maxLogit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < flat.cols; ++i) {
        values[i] = flat.at<float>(0, i);
        maxLogit = std::max(maxLogit, values[i]);
    }

    float sum = 0.0f;
    for (float& v : values) {
        v = std::exp(v - maxLogit);
        sum += v;
    }
    if (sum <= 0.0f) return values;
    for (float& v : values) v /= sum;
    return values;
}
