#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "common/Detection.hpp"

struct DeepSORTTrack {
    int id = -1;
    cv::Rect bbox;
    cv::Point2f centroid;
    cv::Scalar color;
    cv::Mat embedding;
    int ageFrames = 0;
    int missedFrames = 0;
};

class DeepSORTTracker {
public:
    DeepSORTTracker();
    void update(const std::vector<Detection>& detections, const std::vector<cv::Mat>& embeddings);
    std::vector<DeepSORTTrack> getTracks() const;
private:
    float iou(const cv::Rect& a, const cv::Rect& b) const;
    float cosineSim(const cv::Mat& a, const cv::Mat& b) const;

    std::vector<DeepSORTTrack> tracks;
    int nextId = 0;
    int maxMissed = 30;
};
