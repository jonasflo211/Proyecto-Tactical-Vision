#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "common/Detection.hpp"

struct ByteTrackTrack {
    int id = -1;
    cv::Rect bbox;
    cv::Point2f centroid;
    cv::Scalar color;
    float score = 0.0f;
    int ageFrames = 0;
    int missedFrames = 0;
    cv::KalmanFilter kf;
};

class ByteTrackTracker {
public:
    ByteTrackTracker();
    void update(const std::vector<Detection>& detections);
    std::vector<ByteTrackTrack> getTracks() const;
private:
    float iou(const cv::Rect& a, const cv::Rect& b) const;
    void initKF(ByteTrackTrack& t) const;
    void suppressDuplicateTracks();

    std::vector<ByteTrackTrack> tracks;
    int nextId = 0;
    float highThr = 0.5f;
    float lowThr = 0.1f;
    float matchIou = 0.3f;
    float duplicateIou = 0.40f;
    int maxMissed = 30;
};
