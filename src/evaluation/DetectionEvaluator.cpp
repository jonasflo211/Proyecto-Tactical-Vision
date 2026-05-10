#include "evaluation/DetectionEvaluator.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace {

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
}

bool parseInt(const std::string& value, int& out) {
    try {
        size_t consumed = 0;
        out = std::stoi(value, &consumed);
        return consumed == value.size();
    } catch (...) {
        return false;
    }
}

} // namespace

double EvaluationSummary::precision() const {
    int denom = truePositives + falsePositives;
    return denom > 0 ? (double)truePositives / (double)denom : 0.0;
}

double EvaluationSummary::recall() const {
    int denom = truePositives + falseNegatives;
    return denom > 0 ? (double)truePositives / (double)denom : 0.0;
}

double EvaluationSummary::f1() const {
    double p = precision();
    double r = recall();
    return (p + r) > 0.0 ? 2.0 * p * r / (p + r) : 0.0;
}

float boxIou(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int w = std::max(0, x2 - x1);
    int h = std::max(0, y2 - y1);
    int inter = w * h;
    int uni = a.area() + b.area() - inter;
    return uni > 0 ? (float)inter / (float)uni : 0.0f;
}

std::map<int, std::vector<GroundTruthBox>> loadGroundTruthCsv(const std::string& path,
                                                              std::string& error) {
    std::map<int, std::vector<GroundTruthBox>> byFrame;
    std::ifstream file(path);
    if (!file.is_open()) {
        error = "No se pudo abrir labels CSV: " + path;
        return byFrame;
    }

    std::string line;
    int lineNo = 0;
    while (std::getline(file, line)) {
        lineNo++;
        if (line.empty()) continue;
        if (lineNo == 1 && line.find("frame") != std::string::npos) continue;

        auto cols = splitCsvLine(line);
        if (cols.size() != 5 && cols.size() != 6) {
            error = "Linea CSV invalida " + std::to_string(lineNo) +
                    ": usa frame,x,y,w,h o frame,id,x,y,w,h";
            return {};
        }

        GroundTruthBox gt;
        int x = 0, y = 0, w = 0, h = 0;
        bool ok = false;
        if (cols.size() == 5) {
            ok = parseInt(cols[0], gt.frame) &&
                 parseInt(cols[1], x) &&
                 parseInt(cols[2], y) &&
                 parseInt(cols[3], w) &&
                 parseInt(cols[4], h);
        } else {
            ok = parseInt(cols[0], gt.frame) &&
                 parseInt(cols[1], gt.id) &&
                 parseInt(cols[2], x) &&
                 parseInt(cols[3], y) &&
                 parseInt(cols[4], w) &&
                 parseInt(cols[5], h);
        }
        if (!ok || gt.frame < 1 || w <= 0 || h <= 0) {
            error = "Valores invalidos en labels CSV linea " + std::to_string(lineNo);
            return {};
        }

        gt.bbox = cv::Rect(x, y, w, h);
        byFrame[gt.frame].push_back(gt);
    }
    return byFrame;
}

FrameEvaluation evaluateFrame(int frame,
                              const std::vector<GroundTruthBox>& truth,
                              const std::vector<Detection>& detections,
                              float iouThreshold) {
    FrameEvaluation result;
    result.frame = frame;
    result.truthCount = (int)truth.size();
    result.detectionCount = (int)detections.size();

    struct Pair {
        int truthIdx = -1;
        int detIdx = -1;
        float iou = 0.0f;
    };

    std::vector<Pair> pairs;
    for (int t = 0; t < (int)truth.size(); ++t) {
        for (int d = 0; d < (int)detections.size(); ++d) {
            float iou = boxIou(truth[t].bbox, detections[d].bbox);
            if (iou >= iouThreshold) pairs.push_back({t, d, iou});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
        return a.iou > b.iou;
    });

    std::vector<bool> truthUsed(truth.size(), false);
    std::vector<bool> detUsed(detections.size(), false);
    double iouSum = 0.0;
    for (const auto& pair : pairs) {
        if (truthUsed[pair.truthIdx] || detUsed[pair.detIdx]) continue;
        truthUsed[pair.truthIdx] = true;
        detUsed[pair.detIdx] = true;
        result.truePositives++;
        iouSum += pair.iou;
    }

    result.falsePositives = result.detectionCount - result.truePositives;
    result.falseNegatives = result.truthCount - result.truePositives;
    result.meanMatchedIou = result.truePositives > 0 ? iouSum / result.truePositives : 0.0;
    return result;
}

EvaluationSummary summarizeEvaluations(const std::vector<FrameEvaluation>& frames) {
    EvaluationSummary summary;
    summary.framesEvaluated = (int)frames.size();
    double weightedIou = 0.0;
    for (const auto& f : frames) {
        summary.truthCount += f.truthCount;
        summary.detectionCount += f.detectionCount;
        summary.truePositives += f.truePositives;
        summary.falsePositives += f.falsePositives;
        summary.falseNegatives += f.falseNegatives;
        weightedIou += f.meanMatchedIou * f.truePositives;
    }
    summary.meanMatchedIou = summary.truePositives > 0
        ? weightedIou / summary.truePositives
        : 0.0;
    return summary;
}
