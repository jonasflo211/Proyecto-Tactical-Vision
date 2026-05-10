#pragma once

#include <opencv2/opencv.hpp>

#include <map>
#include <string>
#include <vector>

#include "common/Detection.hpp"

struct GroundTruthBox {
    int frame = 0;
    int id = -1;
    cv::Rect bbox;
};

struct FrameEvaluation {
    int frame = 0;
    int truthCount = 0;
    int detectionCount = 0;
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    double meanMatchedIou = 0.0;
};

struct EvaluationSummary {
    int framesEvaluated = 0;
    int truthCount = 0;
    int detectionCount = 0;
    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    double meanMatchedIou = 0.0;

    double precision() const;
    double recall() const;
    double f1() const;
};

float boxIou(const cv::Rect& a, const cv::Rect& b);
std::map<int, std::vector<GroundTruthBox>> loadGroundTruthCsv(const std::string& path,
                                                              std::string& error);
FrameEvaluation evaluateFrame(int frame,
                              const std::vector<GroundTruthBox>& truth,
                              const std::vector<Detection>& detections,
                              float iouThreshold);
EvaluationSummary summarizeEvaluations(const std::vector<FrameEvaluation>& frames);
