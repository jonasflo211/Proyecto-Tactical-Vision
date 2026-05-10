#include "../../include/tracking/ByteTrackTracker.hpp"
#include <algorithm>

ByteTrackTracker::ByteTrackTracker() = default;

float ByteTrackTracker::iou(const cv::Rect& a, const cv::Rect& b) const {
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

void ByteTrackTracker::initKF(ByteTrackTrack& t) const {
    t.kf = cv::KalmanFilter(4, 2);
    t.kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1,0,1,0, 0,1,0,1, 0,0,1,0, 0,0,0,1);
    cv::setIdentity(t.kf.measurementMatrix);
    cv::setIdentity(t.kf.processNoiseCov, cv::Scalar::all(1e-3));
    cv::setIdentity(t.kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(t.kf.errorCovPost, cv::Scalar::all(1e-1));
    t.kf.statePost.at<float>(0) = t.centroid.x;
    t.kf.statePost.at<float>(1) = t.centroid.y;
    t.kf.statePost.at<float>(2) = 0;
    t.kf.statePost.at<float>(3) = 0;
}

void ByteTrackTracker::suppressDuplicateTracks() {
    if (tracks.size() < 2) return;

    std::vector<bool> remove(tracks.size(), false);
    for (size_t i = 0; i < tracks.size(); ++i) {
        if (remove[i] || tracks[i].missedFrames > 0) continue;
        for (size_t j = i + 1; j < tracks.size(); ++j) {
            if (remove[j] || tracks[j].missedFrames > 0) continue;
            if (iou(tracks[i].bbox, tracks[j].bbox) < duplicateIou) continue;

            bool keepI = tracks[i].score > tracks[j].score ||
                         (tracks[i].score == tracks[j].score && tracks[i].ageFrames >= tracks[j].ageFrames);
            remove[keepI ? j : i] = true;
            if (!keepI) break;
        }
    }

    size_t write = 0;
    for (size_t read = 0; read < tracks.size(); ++read) {
        if (!remove[read]) {
            if (write != read) tracks[write] = std::move(tracks[read]);
            write++;
        }
    }
    tracks.resize(write);
}

void ByteTrackTracker::update(const std::vector<Detection>& detections) {
    std::vector<Detection> high, low;
    high.reserve(detections.size());
    low.reserve(detections.size());
    for (const auto& d : detections) {
        if (d.score >= highThr) high.push_back(d);
        else if (d.score >= lowThr) low.push_back(d);
    }

    // Predict
    for (auto& t : tracks) {
        cv::Mat p = t.kf.predict();
        t.centroid.x = p.at<float>(0);
        t.centroid.y = p.at<float>(1);
        t.bbox.x = (int)std::round(t.centroid.x - t.bbox.width / 2.0f);
        t.bbox.y = (int)std::round(t.centroid.y - t.bbox.height / 2.0f);
        t.ageFrames++;
        t.missedFrames++;
    }

    std::vector<bool> detUsedHigh(high.size(), false);
    std::vector<bool> detUsedLow(low.size(), false);

    // Match with high confidence
    for (auto& t : tracks) {
        float bestIou = 0.0f;
        int bestIdx = -1;
        for (size_t i = 0; i < high.size(); ++i) {
            if (detUsedHigh[i]) continue;
            float v = iou(t.bbox, high[i].bbox);
            if (v > bestIou) {
                bestIou = v;
                bestIdx = (int)i;
            }
        }
        if (bestIdx != -1 && bestIou >= matchIou) {
            const auto& d = high[bestIdx];
            t.bbox = d.bbox;
            t.centroid = d.centroid;
            t.score = d.score;
            t.color = d.color;
            t.missedFrames = 0;
            cv::Mat m(2, 1, CV_32F);
            m.at<float>(0) = t.centroid.x;
            m.at<float>(1) = t.centroid.y;
            t.kf.correct(m);
            detUsedHigh[bestIdx] = true;
        }
    }

    // Match remaining tracks with low confidence detections
    for (auto& t : tracks) {
        if (t.missedFrames == 0) continue;
        float bestIou = 0.0f;
        int bestIdx = -1;
        for (size_t i = 0; i < low.size(); ++i) {
            if (detUsedLow[i]) continue;
            float v = iou(t.bbox, low[i].bbox);
            if (v > bestIou) {
                bestIou = v;
                bestIdx = (int)i;
            }
        }
        if (bestIdx != -1 && bestIou >= matchIou) {
            const auto& d = low[bestIdx];
            t.bbox = d.bbox;
            t.centroid = d.centroid;
            t.score = d.score;
            t.color = d.color;
            t.missedFrames = 0;
            cv::Mat m(2, 1, CV_32F);
            m.at<float>(0) = t.centroid.x;
            m.at<float>(1) = t.centroid.y;
            t.kf.correct(m);
            detUsedLow[bestIdx] = true;
        }
    }

    // New tracks from unmatched high detections
    for (size_t i = 0; i < high.size(); ++i) {
        if (detUsedHigh[i]) continue;
        ByteTrackTrack t;
        t.id = nextId++;
        t.bbox = high[i].bbox;
        t.centroid = high[i].centroid;
        t.score = high[i].score;
        t.color = high[i].color;
        t.ageFrames = 1;
        t.missedFrames = 0;
        initKF(t);
        tracks.push_back(t);
    }

    suppressDuplicateTracks();

    // Prune
    for (auto it = tracks.begin(); it != tracks.end();) {
        if (it->missedFrames > maxMissed) it = tracks.erase(it);
        else ++it;
    }
}

std::vector<ByteTrackTrack> ByteTrackTracker::getTracks() const {
    return tracks;
}
