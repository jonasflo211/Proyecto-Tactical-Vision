#pragma once
#include <opencv2/opencv.hpp>

struct ReIDResult {
    cv::Mat embedding;
    cv::Scalar teamColor;
    cv::Vec3f dominantHsv;
    bool dominantValid = false;
};

class ReIDEmbedder {
public:
    ReIDEmbedder() = default;
    ReIDResult infer(const cv::Mat& frame, const cv::Rect& bbox) const;
private:
    cv::Mat makeEmbedding(const cv::Mat& hsvRoi, const cv::Mat& mask) const;
    cv::Scalar classifyTeamColor(const cv::Mat& hsvRoi, const cv::Mat& mask) const;
};
