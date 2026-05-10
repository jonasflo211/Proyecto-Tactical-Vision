#include "../../include/detection/ClassicDetector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iostream>

ClassicDetector::ClassicDetector()
    : hog(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9) {}

void ClassicDetector::configure(const ClassicDetectorParams& p) {
    params = p;
    bg = cv::createBackgroundSubtractorMOG2(params.bgHistory, params.bgVarThreshold, params.detectShadows);
    if (params.useHogSvm && !params.svmModelPath.empty()) {
        loadSvm(params.svmModelPath);
    }
    if (params.useCnnValidator && !params.cnnValidatorModelPath.empty()) {
        if (!cnnValidator.load(params.cnnValidatorModelPath)) {
            std::cerr << "[WARN] No se pudo cargar CNN validator: "
                      << params.cnnValidatorModelPath << std::endl;
        }
    }
    ready = true;
}

void ClassicDetector::reset() {
    bg.release();
    svm.release();
    cnnValidator = CropClassifier();
    ready = false;
    hardNegativeIndex = 0;
    candidates.clear();
    lastFieldArea.release();
    lastFrameSize = cv::Size();
}

cv::Mat ClassicDetector::rawGreenMask(const cv::Mat& frame) const {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat broad;
    cv::Mat strict;
    cv::inRange(hsv,
                cv::Scalar(params.greenHueLow1, params.greenSatLow, params.greenValLow),
                cv::Scalar(params.greenHueHigh1, 255, 255),
                broad);
    cv::inRange(hsv,
                cv::Scalar(params.greenHueLow2, params.greenSatLow + 15, params.greenValLow + 10),
                cv::Scalar(params.greenHueHigh2, 255, 255),
                strict);
    cv::bitwise_or(broad, strict, broad);

    if (params.useExcessGreen) {
        std::vector<cv::Mat> bgr;
        cv::split(frame, bgr);
        cv::Mat b, g, r;
        bgr[0].convertTo(b, CV_16S);
        bgr[1].convertTo(g, CV_16S);
        bgr[2].convertTo(r, CV_16S);

        cv::Mat exg = 2 * g - r - b;
        cv::Mat greenDominant = (g > r) & (g > b);
        cv::Mat exgMask;
        cv::threshold(exg, exgMask, params.excessGreenThreshold, 255, cv::THRESH_BINARY);
        exgMask.convertTo(exgMask, CV_8U);
        cv::bitwise_and(exgMask, greenDominant, exgMask);
        cv::bitwise_or(broad, exgMask, broad);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(broad, broad, cv::MORPH_OPEN, kernel);
    return broad;
}

cv::Mat ClassicDetector::greenMask(const cv::Mat& frame) const {
    cv::Mat broad = rawGreenMask(frame);

    int openK = std::max(1, params.fieldOpen);
    if (openK % 2 == 0) openK += 1;
    int closeK = std::max(1, params.fieldClose);
    if (closeK % 2 == 0) closeK += 1;
    cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(openK, openK));
    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(closeK, closeK));
    cv::morphologyEx(broad, broad, cv::MORPH_OPEN, openKernel);
    cv::morphologyEx(broad, broad, cv::MORPH_CLOSE, closeKernel);
    cv::dilate(broad, broad, openKernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(broad, broad, cv::MORPH_CLOSE, closeKernel);
    return broad;
}

cv::Mat ClassicDetector::fieldMask(const cv::Mat& frame) {
    cv::Mat green = greenMask(frame);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(green, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat fieldArea(frame.rows, frame.cols, CV_8U, cv::Scalar(0));
    if (contours.empty()) {
        if (!lastFieldArea.empty() && lastFieldArea.size() == frame.size()) return lastFieldArea.clone();
        fieldArea.setTo(255);
        lastFieldArea = fieldArea.clone();
        return fieldArea;
    }

    size_t bestIdx = 0;
    double bestArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a > bestArea) {
            bestArea = a;
            bestIdx = i;
        }
    }

    cv::drawContours(fieldArea, contours, (int)bestIdx, cv::Scalar(255), cv::FILLED);

    if (params.fieldDilate > 0) {
        int k = params.fieldDilate;
        if (k % 2 == 0) k += 1;
        cv::Mat dk = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        cv::dilate(fieldArea, fieldArea, dk);
        cv::morphologyEx(fieldArea, fieldArea, cv::MORPH_CLOSE, dk);
    }
    if (params.fieldErode > 0) {
        int k = params.fieldErode;
        if (k % 2 == 0) k += 1;
        cv::Mat ek = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
        cv::erode(fieldArea, fieldArea, ek);
    }

    float fieldRatio = (float)cv::countNonZero(fieldArea) / (float)std::max(1, frame.rows * frame.cols);
    if (fieldRatio < params.minFieldAreaRatio) {
        if (!lastFieldArea.empty() && lastFieldArea.size() == frame.size()) return lastFieldArea.clone();
        fieldArea.setTo(255);
    } else if (!lastFieldArea.empty() && lastFieldArea.size() == frame.size()) {
        double alpha = std::max(0.0f, std::min(1.0f, params.fieldMaskBlend));
        cv::Mat blended;
        cv::addWeighted(fieldArea, 1.0 - alpha, lastFieldArea, alpha, 0.0, blended);
        cv::threshold(blended, fieldArea, 127, 255, cv::THRESH_BINARY);
    }
    lastFieldArea = fieldArea.clone();
    return fieldArea;
}

cv::Mat ClassicDetector::whiteLineMask(const cv::Mat& frame, const cv::Mat& fieldArea) const {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::Mat white;
    cv::inRange(hsv, cv::Scalar(0, 0, 165), cv::Scalar(180, 75, 255), white);
    cv::bitwise_and(white, fieldArea, white);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(white, white, cv::MORPH_OPEN, kernel);

    cv::Mat lineOnly(frame.rows, frame.cols, CV_8U, cv::Scalar(0));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(white.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& c : contours) {
        cv::Rect r = cv::boundingRect(c);
        double area = cv::contourArea(c);
        if (r.area() <= 0 || area < 12.0) continue;

        float wideAspect = r.height > 0 ? (float)r.width / (float)r.height : 0.0f;
        float tallAspect = r.width > 0 ? (float)r.height / (float)r.width : 0.0f;
        float hRel = frame.rows > 0 ? (float)r.height / (float)frame.rows : 0.0f;
        float wRel = frame.cols > 0 ? (float)r.width / (float)frame.cols : 0.0f;
        bool longThin = (wideAspect > 3.0f && hRel < 0.045f) ||
                        (tallAspect > 4.0f && wRel < 0.025f);
        bool compactPlayerLike = r.height > 10 && r.width > 4 && tallAspect > 0.7f && wideAspect < 1.4f;
        if (longThin && !compactPlayerLike) {
            cv::drawContours(lineOnly, std::vector<std::vector<cv::Point>>{c}, -1, cv::Scalar(255), -1);
        }
    }
    return lineOnly;
}

cv::Mat ClassicDetector::playerColorMask(const cv::Mat& frame,
                                         const cv::Mat& fieldArea,
                                         const cv::Mat& green,
                                         const cv::Mat& white,
                                         const cv::Mat& shadow) const {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat rawGrass, rawGrassStrict;
    cv::inRange(hsv,
                cv::Scalar(params.greenHueLow1, params.greenSatLow, params.greenValLow),
                cv::Scalar(params.greenHueHigh1, 255, 255),
                rawGrass);
    cv::inRange(hsv,
                cv::Scalar(params.greenHueLow2, params.greenSatLow + 15, params.greenValLow + 10),
                cv::Scalar(params.greenHueHigh2, 255, 255),
                rawGrassStrict);
    cv::bitwise_or(rawGrass, rawGrassStrict, rawGrass);

    cv::Mat red1, red2, red, whiteKit, darkKit, yellowKit, blueKit, saturated, mask;
    cv::inRange(hsv, cv::Scalar(0, 55, 45), cv::Scalar(12, 255, 255), red1);
    cv::inRange(hsv, cv::Scalar(165, 55, 45), cv::Scalar(180, 255, 255), red2);
    cv::bitwise_or(red1, red2, red);

    // White kits need to stay, but pitch markings are removed with whiteLineMask().
    cv::inRange(hsv, cv::Scalar(0, 0, 135), cv::Scalar(180, 95, 255), whiteKit);
    cv::inRange(hsv, cv::Scalar(0, 0, 18), cv::Scalar(180, 125, 95), darkKit);
    cv::inRange(hsv, cv::Scalar(18, 65, 80), cv::Scalar(42, 255, 255), yellowKit);
    cv::inRange(hsv, cv::Scalar(92, 45, 45), cv::Scalar(132, 255, 255), blueKit);

    cv::inRange(hsv, cv::Scalar(0, 45, 40), cv::Scalar(180, 255, 255), saturated);
    saturated.setTo(0, rawGrass);

    cv::bitwise_or(red, whiteKit, mask);
    cv::bitwise_or(mask, darkKit, mask);
    cv::bitwise_or(mask, yellowKit, mask);
    cv::bitwise_or(mask, blueKit, mask);
    cv::bitwise_or(mask, saturated, mask);

    mask.setTo(0, rawGrass);
    cv::bitwise_and(mask, fieldArea, mask);
    mask.setTo(0, white);

    cv::Mat grassShadow;
    cv::bitwise_and(shadow, green, grassShadow);
    mask.setTo(0, grassShadow);

    cv::Mat smallKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, smallKernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, smallKernel);

    return mask;
}

cv::Mat ClassicDetector::shadowMask(const cv::Mat& frame, const cv::Mat& fieldArea) const {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::Mat shadow;
    cv::inRange(hsv, cv::Scalar(20, 20, 20), cv::Scalar(105, 170, 115), shadow);
    cv::bitwise_and(shadow, fieldArea, shadow);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(shadow, shadow, cv::MORPH_OPEN, kernel);
    return shadow;
}

std::vector<cv::Rect> ClassicDetector::splitWideBlob(const cv::Rect& bbox,
                                                     const cv::Mat& motionMask,
                                                     const cv::Size& bounds) const {
    std::vector<cv::Rect> out;
    cv::Rect clip = bbox & cv::Rect(0, 0, bounds.width, bounds.height);
    float wideAspect = clip.height > 0 ? (float)clip.width / (float)clip.height : 0.0f;
    if (!params.splitWideBlobs || clip.area() <= 0 || wideAspect < params.splitWideAspect) {
        out.push_back(clip);
        return out;
    }

    cv::Mat roi = motionMask(clip).clone();
    cv::Mat ccMask;
    cv::morphologyEx(roi, ccMask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    cv::Mat labels, stats, centroids;
    int components = cv::connectedComponentsWithStats(ccMask, labels, stats, centroids, 8, CV_32S);
    int minComponentArea = std::max(12, (int)std::round(clip.area() * params.splitMinComponentAreaRatio));
    for (int label = 1; label < components; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        if (area < minComponentArea || h < clip.height * 0.32f || w < 5) continue;

        cv::Rect r(stats.at<int>(label, cv::CC_STAT_LEFT),
                   stats.at<int>(label, cv::CC_STAT_TOP), w, h);
        int padX = std::max(2, r.width / 5);
        int padY = std::max(2, r.height / 10);
        r.x = std::max(0, r.x - padX);
        r.y = std::max(0, r.y - padY);
        r.width = std::min(clip.width - r.x, r.width + 2 * padX);
        r.height = std::min(clip.height - r.y, r.height + 2 * padY);
        out.push_back((cv::Rect(clip.x + r.x, clip.y + r.y, r.width, r.height)) &
                      cv::Rect(0, 0, bounds.width, bounds.height));
    }

    if (out.size() >= 2) {
        std::sort(out.begin(), out.end(), [](const cv::Rect& a, const cv::Rect& b) {
            return a.x < b.x;
        });
        if ((int)out.size() > params.splitMaxParts) out.resize(params.splitMaxParts);
        return out;
    }
    out.clear();

    std::vector<int> projection(clip.width, 0);
    for (int x = 0; x < roi.cols; ++x) {
        projection[x] = cv::countNonZero(roi.col(x));
    }
    std::vector<int> smoothProjection = projection;
    for (int x = 0; x < clip.width; ++x) {
        int sum = 0;
        int count = 0;
        for (int dx = -2; dx <= 2; ++dx) {
            int xx = x + dx;
            if (xx < 0 || xx >= clip.width) continue;
            sum += projection[xx];
            count++;
        }
        smoothProjection[x] = count > 0 ? sum / count : projection[x];
    }

    int maxProj = *std::max_element(smoothProjection.begin(), smoothProjection.end());
    int valleyLimit = std::max(1, (int)std::round(maxProj * params.splitValleyRatio));
    std::vector<int> splits;
    int minSpan = std::max(8, clip.width / 5);
    for (int x = minSpan; x < clip.width - minSpan; ++x) {
        bool localMin = smoothProjection[x] <= valleyLimit;
        for (int dx = -2; dx <= 2 && localMin; ++dx) {
            int xx = std::max(0, std::min(clip.width - 1, x + dx));
            if (smoothProjection[xx] > valleyLimit) localMin = false;
        }
        if (localMin) {
            splits.push_back(x);
            x += minSpan / 2;
        }
    }

    if (splits.empty()) {
        if (wideAspect < params.splitForceAspect) {
            out.push_back(clip);
            return out;
        }
        int expectedWidth = std::max(12, (int)std::round(clip.height / 2.2));
        int pieces = std::max(1, std::min(params.splitMaxParts, (int)std::round((double)clip.width / expectedWidth)));
        if (pieces <= 1) {
            out.push_back(clip);
            return out;
        }
        for (int i = 1; i < pieces; ++i) {
            splits.push_back(i * clip.width / pieces);
        }
    }

    int start = 0;
    for (int s : splits) {
        cv::Rect part(clip.x + start, clip.y, std::max(1, s - start), clip.height);
        if (part.width >= 6) out.push_back(part & cv::Rect(0, 0, bounds.width, bounds.height));
        start = s;
    }
    cv::Rect part(clip.x + start, clip.y, std::max(1, clip.width - start), clip.height);
    if (part.width >= 6) out.push_back(part & cv::Rect(0, 0, bounds.width, bounds.height));
    return out.empty() ? std::vector<cv::Rect>{clip} : out;
}

bool ClassicDetector::validateCandidate(const cv::Mat& frame,
                                        const cv::Mat& fieldArea,
                                        const cv::Mat& green,
                                        const cv::Mat& white,
                                        const cv::Mat& shadow,
                                        const cv::Mat& motion,
                                        const cv::Rect& bbox,
                                        Detection& out) const {
    cv::Rect clip = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (clip.area() <= 0) return false;

    int frameArea = frame.cols * frame.rows;
    float bottomRel = frame.rows > 0 ? (float)(clip.y + clip.height) / (float)frame.rows : 0.0f;
    float horizon = std::max(0.05f, std::min(0.95f, params.horizonRel));
    float t = bottomRel > horizon ? (bottomRel - horizon) / (1.0f - horizon) : 0.0f;
    t = std::max(0.0f, std::min(1.0f, t));
    auto lerp = [t](float farValue, float nearValue) {
        return farValue + (nearValue - farValue) * t;
    };

    int dynamicMinArea = (int)std::round(lerp((float)params.minAreaFar, (float)params.minAreaNear));
    dynamicMinArea = std::max(20, std::min(dynamicMinArea, std::max(params.minArea, params.minAreaNear)));

    int nearMaxArea = params.maxAreaNear > 0 ? params.maxAreaNear : params.maxArea;
    if (nearMaxArea <= 0 || nearMaxArea > frameArea) nearMaxArea = (int)(frameArea * 0.16f);
    int dynamicMaxArea = (int)std::round(lerp((float)params.maxAreaFar, (float)nearMaxArea));
    dynamicMaxArea = std::max(dynamicMinArea + 1, std::min(dynamicMaxArea, frameArea));

    int area = clip.area();
    if (area < dynamicMinArea || area > dynamicMaxArea) return false;

    float tallAspect = clip.width > 0 ? (float)clip.height / (float)clip.width : 0.0f;
    float wideAspect = clip.height > 0 ? (float)clip.width / (float)clip.height : 99.0f;
    if (tallAspect < 0.75f || tallAspect > 4.2f) return false;
    float maxWideAspect = std::min(params.maxWideAspect, lerp(params.maxWideAspectFar, params.maxWideAspectNear));
    if (wideAspect > maxWideAspect) return false;

    float wRel = frame.cols > 0 ? (float)clip.width / (float)frame.cols : 0.0f;
    float maxWidthRel = std::min(params.maxWidthRel, lerp(params.maxWidthRelFar, params.maxWidthRelNear));
    if (wRel > maxWidthRel) return false;

    float hRel = frame.rows > 0 ? (float)clip.height / (float)frame.rows : 0.0f;
    float minH = params.minHRelFar + (params.minHRelNear - params.minHRelFar) * t;
    float maxH = params.maxHRelFar + (params.maxHRelNear - params.maxHRelFar) * t;
    minH = std::max(minH, params.minHeightRel * 0.5f);
    maxH = std::min(maxH, params.maxHeightRel);
    if (hRel < minH || hRel > maxH) return false;

    if (params.requireBottomOnField) {
        int bx = clip.x + clip.width / 2;
        int by = clip.y + clip.height - 1;
        if (fieldArea.at<unsigned char>(by, bx) == 0) return false;

        int supportedFeet = 0;
        int footY = std::max(0, std::min(frame.rows - 1, by));
        for (float relX : {0.25f, 0.50f, 0.75f}) {
            int fx = std::max(0, std::min(frame.cols - 1, clip.x + (int)std::round(clip.width * relX)));
            if (fieldArea.at<unsigned char>(footY, fx) > 0) supportedFeet++;
        }
        if (supportedFeet < 2) return false;

        int bandH = std::max(3, (int)std::round(clip.height * 0.18f));
        cv::Rect bottomBand(clip.x, std::max(0, clip.y + clip.height - bandH), clip.width, bandH);
        bottomBand &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (bottomBand.area() <= 0) return false;
        float bottomSupport = (float)cv::countNonZero(fieldArea(bottomBand)) / (float)bottomBand.area();
        if (bottomSupport < params.minBottomFieldSupport) return false;

        int groundY = std::min(frame.rows - 1, clip.y + clip.height);
        int groundH = std::max(4, (int)std::round(clip.height * 0.16f));
        cv::Rect groundBand(clip.x - clip.width / 4, groundY,
                            clip.width + clip.width / 2, groundH);
        groundBand &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (groundBand.area() > 0) {
            float groundGreen = (float)cv::countNonZero(green(groundBand)) / (float)groundBand.area();
            if (groundGreen < params.minGroundGreenSupport) return false;
        }
    }

    if (params.minFieldOverlap > 0.0f) {
        float overlap = (float)cv::countNonZero(fieldArea(clip)) / (float)clip.area();
        if (overlap < params.minFieldOverlap) return false;
    }

    float greenRatio = (float)cv::countNonZero(green(clip)) / (float)clip.area();
    float nonGreenRatio = 1.0f - greenRatio;
    float minNonGreen = std::max(0.0f, std::min(params.minNonGreenRatio, lerp(params.minNonGreenFar, params.minNonGreenNear)));
    if (nonGreenRatio < minNonGreen || greenRatio > params.maxGreenRatio) return false;

    float whiteRatio = (float)cv::countNonZero(white(clip)) / (float)clip.area();
    if (whiteRatio > params.maxWhiteRatio) return false;
    if (tallAspect > 3.35f && whiteRatio > 0.32f) return false;

    float shadowRatio = (float)cv::countNonZero(shadow(clip)) / (float)clip.area();
    if (shadowRatio > params.maxShadowRatio && nonGreenRatio < 0.45f) return false;

    float motionRatio = (float)cv::countNonZero(motion(clip)) / (float)clip.area();
    float minMotion = std::min(params.minMotionRatio, lerp(params.minMotionFar, params.minMotionNear));
    if (motionRatio < minMotion) return false;

    std::vector<std::vector<cv::Point>> localContours;
    cv::Mat local = motion(clip).clone();
    cv::findContours(local, localContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double contourArea = 0.0;
    for (const auto& c : localContours) contourArea += cv::contourArea(c);
    float fill = (float)(contourArea / std::max(1, clip.area()));
    float minFill = std::min(params.minFillRatio, lerp(params.minFillFar, params.minFillNear));
    if (fill < minFill) return false;

    out.bbox = clip;
    out.centroid = cv::Point2f(clip.x + clip.width / 2.0f, clip.y + clip.height / 2.0f);
    float geometryScore = 1.0f - std::min(1.0f, std::abs(tallAspect - 2.0f) / 2.8f);
    out.score = 0.58f + std::min(0.35f, nonGreenRatio * 0.18f + motionRatio * 0.16f + geometryScore * 0.08f);
    out.color = cv::Scalar(0, 255, 255);
    return true;
}

void ClassicDetector::computeHog(const cv::Mat& sample, cv::Mat& row) const {
    cv::Mat bgr;
    if (sample.channels() == 1) cv::cvtColor(sample, bgr, cv::COLOR_GRAY2BGR);
    else bgr = sample;
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(64, 128));
    std::vector<float> descriptors;
    hog.compute(resized, descriptors, cv::Size(8, 8), cv::Size(0, 0));
    row = cv::Mat(1, (int)descriptors.size(), CV_32F);
    for (int i = 0; i < row.cols; ++i) row.at<float>(0, i) = descriptors[i];
}

bool ClassicDetector::validateWithSvm(const cv::Mat& frame, const cv::Rect& bbox, float& score) const {
    score = 0.0f;
    if (!params.useHogSvm || svm.empty()) return true;
    cv::Rect clip = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (clip.area() <= 0) return false;
    cv::Mat row;
    computeHog(frame(clip), row);
    float raw = svm->predict(row, cv::noArray(), cv::ml::StatModel::RAW_OUTPUT);
    float label = svm->predict(row);
    score = (label > 0.0f) ? std::abs(raw) : -std::abs(raw);
    return label > 0.0f && score >= params.svmMinScore;
}

bool ClassicDetector::validateWithCnn(const cv::Mat& frame, const cv::Rect& bbox, float& confidence) const {
    confidence = 0.0f;
    if (!params.useCnnValidator || cnnValidator.empty()) return true;
    CropPrediction pred = cnnValidator.predict(frame, bbox);
    if (pred.classId < 0 || pred.probabilities.size() < 2) return false;
    confidence = pred.probabilities[1];
    return pred.classId == 1 && confidence >= params.cnnValidatorMinConfidence;
}

cv::Rect ClassicDetector::expandPlayerBox(const cv::Rect& bbox, const cv::Size& bounds) const {
    cv::Rect clip = bbox & cv::Rect(0, 0, bounds.width, bounds.height);
    if (clip.area() <= 0) return clip;

    float cx = clip.x + clip.width * 0.5f;
    int bottom = clip.y + clip.height;
    float targetW = std::max((float)clip.width * 1.15f, (float)clip.height / 2.6f);
    float targetH = std::max((float)clip.height * 1.08f, targetW * 1.9f);

    int w = std::max(clip.width, (int)std::round(targetW));
    int h = std::max(clip.height, (int)std::round(targetH));
    cv::Rect expanded((int)std::round(cx - w * 0.5f),
                      bottom - h,
                      w,
                      h);
    return expanded & cv::Rect(0, 0, bounds.width, bounds.height);
}

void ClassicDetector::maybeSaveHardNegative(const cv::Mat& frame, const cv::Rect& bbox, float score) {
    if (!params.saveHardNegatives || params.hardNegativeDir.empty()) return;
    cv::Rect clip = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (clip.area() <= 0) return;
    try {
        std::filesystem::create_directories(params.hardNegativeDir);
        std::string name = params.hardNegativeDir + "/hard_" +
                           std::to_string(hardNegativeIndex++) + "_s" +
                           std::to_string((int)std::round(score * 100.0f)) + ".png";
        cv::imwrite(name, frame(clip));
    } catch (...) {
    }
}

bool ClassicDetector::loadSvm(const std::string& modelPath) {
    try {
        svm = cv::Algorithm::load<cv::ml::SVM>(modelPath);
    } catch (const cv::Exception&) {
        svm.release();
    }
    return !svm.empty();
}

bool ClassicDetector::trainSvmFromDataset(const std::string& positivesDir,
                                          const std::string& negativesDir,
                                          const std::string& outputModelPath) {
    cv::Mat samples;
    cv::Mat labels;
    auto loadDir = [&](const std::string& dir, int label) {
        if (dir.empty() || !std::filesystem::exists(dir)) return;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
            if (img.empty()) continue;
            cv::Mat row;
            computeHog(img, row);
            samples.push_back(row);
            labels.push_back(label);
        }
    };
    loadDir(positivesDir, 1);
    loadDir(negativesDir, -1);
    if (samples.rows < 2 || labels.rows < 2) return false;

    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setC(0.01);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-4));
    bool ok = svm->train(samples, cv::ml::ROW_SAMPLE, labels);
    if (ok && !outputModelPath.empty()) {
        svm->save(outputModelPath);
    }
    return ok;
}

std::vector<Detection> ClassicDetector::nonMaxSuppression(std::vector<Detection> detections) const {
    if (detections.size() < 2 || params.nmsThreshold <= 0.0f) return detections;

    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    std::vector<Detection> kept;
    std::vector<bool> removed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!removed[j] && iou(detections[i].bbox, detections[j].bbox) > params.nmsThreshold) {
                removed[j] = true;
            }
        }
    }
    return kept;
}

float ClassicDetector::iou(const cv::Rect& a, const cv::Rect& b) const {
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

std::vector<Detection> ClassicDetector::temporalConfirm(const std::vector<Detection>& raw) {
    std::vector<Detection> confirmed;
    if (params.confirmFrames <= 1) return raw;

    std::vector<bool> used(raw.size(), false);

    for (auto& c : candidates) {
        float bestIou = 0.0f;
        int bestIdx = -1;
        for (size_t i = 0; i < raw.size(); ++i) {
            if (used[i]) continue;
            float v = iou(c.bbox, raw[i].bbox);
            if (v > bestIou) {
                bestIou = v;
                bestIdx = (int)i;
            }
        }
        if (bestIdx != -1 && bestIou >= params.confirmIou) {
            c.bbox = raw[bestIdx].bbox;
            c.age++;
            c.missed = 0;
            used[bestIdx] = true;
        } else {
            c.missed++;
        }
    }

    for (size_t i = 0; i < raw.size(); ++i) {
        if (used[i]) continue;
        Candidate c;
        c.bbox = raw[i].bbox;
        candidates.push_back(c);
    }

    for (auto it = candidates.begin(); it != candidates.end();) {
        if (it->missed > params.maxCandidateMiss) it = candidates.erase(it);
        else ++it;
    }

    for (const auto& c : candidates) {
        if (c.age >= params.confirmFrames) {
            Detection d;
            d.bbox = c.bbox;
            d.score = 0.7f;
            d.centroid = cv::Point2f(c.bbox.x + c.bbox.width / 2.0f,
                                     c.bbox.y + c.bbox.height / 2.0f);
            confirmed.push_back(d);
        }
    }
    return confirmed;
}

std::vector<Detection> ClassicDetector::detect(const cv::Mat& frame) {
    if (!ready) {
        configure(params);
    }
    std::vector<Detection> detections;
    if (frame.empty()) return detections;
    if (lastFrameSize != frame.size()) {
        candidates.clear();
        lastFrameSize = frame.size();
    }

    // Preprocessing: gray -> blur -> equalize
    cv::Mat gray;
    if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else gray = frame.clone();

    if (params.blur > 1) {
        int k = params.blur;
        if (k % 2 == 0) k += 1;
        cv::GaussianBlur(gray, gray, cv::Size(k, k), 0);
    }

    if (params.useEqualize) {
        cv::equalizeHist(gray, gray);
    }

    cv::Mat fgMask;
    if (params.useCanny) {
        cv::Mat edges;
        cv::Canny(gray, edges, params.canny1, params.canny2);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(edges, fgMask, kernel, cv::Point(-1, -1), 1);
    } else {
        bg->apply(gray, fgMask);
    }

    cv::Mat fieldArea = fieldMask(frame);
    cv::Mat green = rawGreenMask(frame);
    cv::Mat white = whiteLineMask(frame, fieldArea);
    cv::Mat shadow = shadowMask(frame, fieldArea);
    cv::Mat combined;
    cv::bitwise_and(fgMask, fieldArea, combined);
    cv::threshold(combined, combined, params.detectShadows ? 200 : 1, 255, cv::THRESH_BINARY);
    combined.setTo(0, white);
    cv::Mat likelyGrassShadow;
    cv::bitwise_and(shadow, green, likelyGrassShadow);
    combined.setTo(0, likelyGrassShadow);

    if (params.useColorCandidates) {
        cv::Mat playerColors = playerColorMask(frame, fieldArea, green, white, shadow);
        cv::bitwise_or(combined, playerColors, combined);
    }

    if (params.morph > 0) {
        int mk = params.morph;
        if (mk % 2 == 0) mk += 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(mk, mk));
        cv::morphologyEx(combined, combined, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(combined, combined, cv::MORPH_CLOSE, kernel);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(combined, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& c : contours) {
        cv::Rect bb = cv::boundingRect(c);
        std::vector<cv::Rect> parts = splitWideBlob(bb, combined, frame.size());
        for (const auto& part : parts) {
            Detection d;
            if (!validateCandidate(frame, fieldArea, green, white, shadow, combined, part, d)) continue;
            cv::Rect expandedBox = expandPlayerBox(d.bbox, frame.size());
            if (expandedBox.area() > 0) {
                d.bbox = expandedBox;
                d.centroid = cv::Point2f(d.bbox.x + d.bbox.width / 2.0f,
                                         d.bbox.y + d.bbox.height / 2.0f);
            }
            float svmScore = 0.0f;
            if (!validateWithSvm(frame, d.bbox, svmScore)) {
                maybeSaveHardNegative(frame, d.bbox, svmScore);
                continue;
            }
            float cnnConfidence = 0.0f;
            if (!validateWithCnn(frame, d.bbox, cnnConfidence)) {
                maybeSaveHardNegative(frame, d.bbox, cnnConfidence);
                continue;
            }
            if (params.useHogSvm && !svm.empty()) {
                d.score = std::max(0.1f, std::min(0.99f, 0.5f + svmScore * 0.1f));
            }
            if (params.useCnnValidator && !cnnValidator.empty()) {
                d.score = std::max(d.score, std::max(0.1f, std::min(0.99f, cnnConfidence)));
            }
            detections.push_back(d);
        }
    }

    detections = nonMaxSuppression(detections);
    return temporalConfirm(detections);
}
