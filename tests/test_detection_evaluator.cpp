#include "evaluation/DetectionEvaluator.hpp"

#include <cassert>
#include <iostream>
#include <vector>

namespace {

Detection detection(const cv::Rect& bbox) {
    Detection d;
    d.bbox = bbox;
    d.score = 0.9f;
    return d;
}

GroundTruthBox truth(const cv::Rect& bbox) {
    GroundTruthBox gt;
    gt.frame = 1;
    gt.bbox = bbox;
    return gt;
}

} // namespace

int main() {
    std::vector<GroundTruthBox> labels = {
        truth(cv::Rect(10, 10, 20, 40)),
        truth(cv::Rect(100, 100, 20, 40))
    };
    std::vector<Detection> detections = {
        detection(cv::Rect(12, 12, 20, 40)),
        detection(cv::Rect(250, 250, 20, 40))
    };

    FrameEvaluation frame = evaluateFrame(1, labels, detections, 0.5f);
    assert(frame.truePositives == 1);
    assert(frame.falsePositives == 1);
    assert(frame.falseNegatives == 1);
    assert(frame.meanMatchedIou > 0.7);

    EvaluationSummary summary = summarizeEvaluations({frame});
    assert(summary.framesEvaluated == 1);
    assert(summary.precision() == 0.5);
    assert(summary.recall() == 0.5);
    assert(summary.f1() == 0.5);

    std::cout << "test_detection_evaluator OK" << std::endl;
    return 0;
}
