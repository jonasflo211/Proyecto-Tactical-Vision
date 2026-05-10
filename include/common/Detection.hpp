#pragma once
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect bbox;
    float score;
    cv::Point2f centroid;
    cv::Mat hist;
    cv::Scalar color;
};