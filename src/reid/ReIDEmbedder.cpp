#include "../../include/reid/ReIDEmbedder.hpp"
#include <opencv2/imgproc.hpp>

ReIDResult ReIDEmbedder::infer(const cv::Mat& frame, const cv::Rect& bbox) const {
    cv::Rect bb = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    // Focus on upper body to avoid shorts/socks bias
    int upperH = std::max(1, (int)(bb.height * 0.60f));
    cv::Rect upper(bb.x, bb.y, bb.width, upperH);
    upper = upper & cv::Rect(0, 0, frame.cols, frame.rows);
    cv::Mat roi = frame(upper);

    cv::Mat hsvRoi;
    cv::cvtColor(roi, hsvRoi, cv::COLOR_BGR2HSV);

    // Mask out green field pixels to focus on jerseys
    cv::Mat greenMask;
    cv::inRange(hsvRoi, cv::Scalar(35, 40, 40), cv::Scalar(90, 255, 255), greenMask);
    cv::Mat mask;
    cv::bitwise_not(greenMask, mask);

    if (cv::countNonZero(mask) < 50) {
        // Fallback: avoid unstable color when ROI is mostly field
        mask.setTo(255);
    }
    ReIDResult result;
    result.embedding = makeEmbedding(hsvRoi, mask);
    result.teamColor = classifyTeamColor(hsvRoi, mask);
    int validCount = cv::countNonZero(mask);
    if (validCount >= 50) {
        cv::Scalar meanHsv = cv::mean(hsvRoi, mask);
        result.dominantHsv = cv::Vec3f((float)meanHsv[0], (float)meanHsv[1], (float)meanHsv[2]);
        result.dominantValid = true;
    }
    return result;
}

cv::Mat ReIDEmbedder::makeEmbedding(const cv::Mat& hsvRoi, const cv::Mat& mask) const {
    int hbins = 8, sbins = 8;
    int histSize[] = {hbins, sbins};
    float hranges[] = {0, 180};
    float sranges[] = {0, 256};
    const float* ranges[] = {hranges, sranges};
    int channels[] = {0, 1};

    cv::Mat hist;
    cv::calcHist(&hsvRoi, 1, channels, mask, hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 1.0, 0, cv::NORM_L2);
    return hist.reshape(1, 1);
}

cv::Scalar ReIDEmbedder::classifyTeamColor(const cv::Mat& hsvRoi, const cv::Mat& mask) const {
    cv::Mat redMask1, redMask2, redMask;
    cv::inRange(hsvRoi, cv::Scalar(0, 60, 40), cv::Scalar(12, 255, 255), redMask1);
    cv::inRange(hsvRoi, cv::Scalar(168, 60, 40), cv::Scalar(180, 255, 255), redMask2);
    cv::bitwise_or(redMask1, redMask2, redMask);
    cv::bitwise_and(redMask, mask, redMask);

    cv::Mat whiteMask;
    cv::inRange(hsvRoi, cv::Scalar(0, 0, 160), cv::Scalar(180, 50, 255), whiteMask);
    cv::bitwise_and(whiteMask, mask, whiteMask);

    cv::Mat darkMask;
    cv::inRange(hsvRoi, cv::Scalar(0, 0, 0), cv::Scalar(180, 95, 95), darkMask);
    cv::bitwise_and(darkMask, mask, darkMask);

    int redCount = cv::countNonZero(redMask);
    int whiteCount = cv::countNonZero(whiteMask);
    int darkCount = cv::countNonZero(darkMask);
    int totalPixels = cv::countNonZero(mask);
    if (totalPixels <= 0) return cv::Scalar(0, 255, 255);

    float redRatio = (float)redCount / (float)totalPixels;
    float whiteRatio = (float)whiteCount / (float)totalPixels;
    float darkRatio = (float)darkCount / (float)totalPixels;

    if (redRatio >= 0.12f && redRatio > whiteRatio * 0.9f) {
        return cv::Scalar(0, 0, 255);
    }
    if (whiteRatio >= 0.16f || (whiteRatio >= 0.08f && darkRatio >= 0.16f)) {
        return cv::Scalar(255, 255, 255);
    }

    // Unknown team color
    return cv::Scalar(0, 255, 255);
}
