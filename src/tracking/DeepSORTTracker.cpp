#include "../../include/tracking/DeepSORTTracker.hpp"
#include <algorithm>

DeepSORTTracker::DeepSORTTracker() = default;

float DeepSORTTracker::iou(const cv::Rect& a, const cv::Rect& b) const {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int w = std::max(0, x2 - x1);
    int h = std::max(0, y2 - y1);
    int inter = w * h;
    int uni = a.area() + b.area() - inter;
    return (uni > 0) ? (float)inter / (float)uni : 0.0f;
}

float DeepSORTTracker::cosineSim(const cv::Mat& a, const cv::Mat& b) const {
    if (a.empty() || b.empty()) return 0.0f;
    float dot = (float)a.dot(b);
    float na = std::max(1e-6f, (float)cv::norm(a));
    float nb = std::max(1e-6f, (float)cv::norm(b));
    return dot / (na * nb);
}

void DeepSORTTracker::update(const std::vector<Detection>& detections, const std::vector<cv::Mat>& embeddings) {
    for (auto& t : tracks) {
        t.ageFrames++;
        t.missedFrames++;
    }

    std::vector<bool> detUsed(detections.size(), false);

    for (auto& t : tracks) {
        float bestScore = -1.0f;
        int bestIdx = -1;
        for (size_t i = 0; i < detections.size(); ++i) {
            if (detUsed[i]) continue;
            float iouScore = iou(t.bbox, detections[i].bbox);
            float appScore = cosineSim(t.embedding, embeddings[i]);
            float score = 0.6f * iouScore + 0.4f * appScore;
            if (score > bestScore) {
                bestScore = score;
                bestIdx = (int)i;
            }
        }
        if (bestIdx != -1 && bestScore > 0.2f) {
            const auto& d = detections[bestIdx];
            t.bbox = d.bbox;
            t.centroid = d.centroid;
            t.color = d.color;
            t.embedding = embeddings[bestIdx].clone();
            t.missedFrames = 0;
            detUsed[bestIdx] = true;
        }
    }

    for (size_t i = 0; i < detections.size(); ++i) {
        if (detUsed[i]) continue;
        DeepSORTTrack t;
        t.id = nextId++;
        t.bbox = detections[i].bbox;
        t.centroid = detections[i].centroid;
        t.color = detections[i].color;
        t.embedding = embeddings[i].clone();
        t.ageFrames = 1;
        t.missedFrames = 0;
        tracks.push_back(t);
    }

    for (auto it = tracks.begin(); it != tracks.end();) {
        if (it->missedFrames > maxMissed) it = tracks.erase(it);
        else ++it;
    }
}

std::vector<DeepSORTTrack> DeepSORTTracker::getTracks() const {
    return tracks;
}
