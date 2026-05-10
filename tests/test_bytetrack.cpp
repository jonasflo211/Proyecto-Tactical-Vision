#include "tracking/ByteTrackTracker.hpp"

#include <cassert>
#include <iostream>
#include <vector>

namespace {

Detection makeDetection(const cv::Rect& bbox, float score) {
    Detection d;
    d.bbox = bbox;
    d.score = score;
    d.centroid = cv::Point2f(bbox.x + bbox.width * 0.5f, bbox.y + bbox.height * 0.5f);
    d.color = cv::Scalar(0, 0, 255);
    return d;
}

} // namespace

int main() {
    ByteTrackTracker tracker;
    tracker.update({
        makeDetection(cv::Rect(10, 10, 20, 40), 0.95f),
        makeDetection(cv::Rect(100, 100, 20, 40), 0.90f)
    });

    auto tracks = tracker.getTracks();
    assert(tracks.size() == 2);
    int firstId = tracks[0].id;

    tracker.update({
        makeDetection(cv::Rect(12, 12, 20, 40), 0.92f),
        makeDetection(cv::Rect(102, 102, 20, 40), 0.88f)
    });
    tracks = tracker.getTracks();
    assert(tracks.size() == 2);
    assert(tracks[0].missedFrames == 0);
    assert(tracks[0].id == firstId);

    tracker.update({});
    tracks = tracker.getTracks();
    assert(tracks.size() == 2);
    assert(tracks[0].missedFrames == 1);

    std::cout << "test_bytetrack OK" << std::endl;
    return 0;
}
