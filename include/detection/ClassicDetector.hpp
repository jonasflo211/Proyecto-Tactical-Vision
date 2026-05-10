#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>
#include "dnn/CropClassifier.hpp"
#include "common/Detection.hpp"

struct ClassicDetectorParams {
    int minArea = 200;
    int maxArea = 0;
    int minAreaFar = 70;
    int minAreaNear = 280;
    int maxAreaFar = 9000;
    int maxAreaNear = 0;
    int blur = 5;
    int morph = 5;
    bool useEqualize = true;
    bool useCanny = false;
    int canny1 = 60;
    int canny2 = 150;
    int bgHistory = 250;
    double bgVarThreshold = 12.0;
    int fieldErode = 15;
    float minNonGreenRatio = 0.15f;
    float minHeightRel = 0.03f;
    float maxHeightRel = 0.45f;
    float minFieldOverlap = 0.60f;
    float minBottomFieldSupport = 0.55f;
    float minGroundGreenSupport = 0.08f;
    bool requireBottomOnField = true;
    float horizonRel = 0.35f;
    float minHRelNear = 0.05f;
    float maxHRelNear = 0.45f;
    float minHRelFar = 0.015f;
    float maxHRelFar = 0.18f;
    int confirmFrames = 3;
    int maxCandidateMiss = 5;
    float confirmIou = 0.3f;
    bool detectShadows = false;
    int greenHueLow1 = 30;
    int greenHueHigh1 = 95;
    int greenHueLow2 = 35;
    int greenHueHigh2 = 85;
    int greenSatLow = 35;
    int greenValLow = 35;
    int fieldOpen = 5;
    int fieldClose = 21;
    int fieldDilate = 9;
    bool useExcessGreen = true;
    bool useColorCandidates = true;
    int excessGreenThreshold = 18;
    float minFieldAreaRatio = 0.18f;
    float fieldMaskBlend = 0.35f;
    float maxGreenRatio = 0.82f;
    float minNonGreenFar = 0.12f;
    float minNonGreenNear = 0.24f;
    float maxWhiteRatio = 0.58f;
    float maxShadowRatio = 0.58f;
    float minMotionRatio = 0.08f;
    float minMotionFar = 0.035f;
    float minMotionNear = 0.10f;
    float minFillRatio = 0.22f;
    float minFillFar = 0.12f;
    float minFillNear = 0.24f;
    float maxWidthRel = 0.22f;
    float maxWidthRelFar = 0.08f;
    float maxWidthRelNear = 0.22f;
    float maxWideAspect = 1.25f;
    float maxWideAspectFar = 1.45f;
    float maxWideAspectNear = 1.18f;
    float nmsThreshold = 0.45f;
    bool splitWideBlobs = true;
    float splitWideAspect = 0.68f;
    float splitForceAspect = 0.95f;
    float splitValleyRatio = 0.26f;
    float splitMinComponentAreaRatio = 0.055f;
    int splitMaxParts = 4;
    bool useHogSvm = false;
    float svmMinScore = 0.0f;
    std::string svmModelPath;
    bool useCnnValidator = false;
    std::string cnnValidatorModelPath;
    float cnnValidatorMinConfidence = 0.65f;
    bool saveHardNegatives = false;
    std::string hardNegativeDir = "dataset/negatives/hard";
};

class ClassicDetector {
public:
    ClassicDetector();
    void configure(const ClassicDetectorParams& params);
    void reset();
    bool loadSvm(const std::string& modelPath);
    bool trainSvmFromDataset(const std::string& positivesDir,
                             const std::string& negativesDir,
                             const std::string& outputModelPath);
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    struct Candidate {
        cv::Rect bbox;
        int age = 1;
        int missed = 0;
    };

    ClassicDetectorParams params;
    cv::Ptr<cv::BackgroundSubtractor> bg;
    cv::Ptr<cv::ml::SVM> svm;
    cv::HOGDescriptor hog;
    CropClassifier cnnValidator;
    bool ready = false;
    int hardNegativeIndex = 0;
    cv::Size lastFrameSize;
    cv::Mat lastFieldArea;
    std::vector<Candidate> candidates;
    cv::Mat fieldMask(const cv::Mat& frame);
    cv::Mat rawGreenMask(const cv::Mat& frame) const;
    cv::Mat greenMask(const cv::Mat& frame) const;
    cv::Mat playerColorMask(const cv::Mat& frame,
                            const cv::Mat& fieldArea,
                            const cv::Mat& green,
                            const cv::Mat& white,
                            const cv::Mat& shadow) const;
    cv::Mat whiteLineMask(const cv::Mat& frame, const cv::Mat& fieldArea) const;
    cv::Mat shadowMask(const cv::Mat& frame, const cv::Mat& fieldArea) const;
    std::vector<cv::Rect> splitWideBlob(const cv::Rect& bbox,
                                        const cv::Mat& motionMask,
                                        const cv::Size& bounds) const;
    bool validateCandidate(const cv::Mat& frame,
                           const cv::Mat& fieldArea,
                           const cv::Mat& green,
                           const cv::Mat& white,
                           const cv::Mat& shadow,
                           const cv::Mat& motion,
                           const cv::Rect& bbox,
                           Detection& out) const;
    bool validateWithSvm(const cv::Mat& frame, const cv::Rect& bbox, float& score) const;
    bool validateWithCnn(const cv::Mat& frame, const cv::Rect& bbox, float& confidence) const;
    cv::Rect expandPlayerBox(const cv::Rect& bbox, const cv::Size& bounds) const;
    void maybeSaveHardNegative(const cv::Mat& frame, const cv::Rect& bbox, float score);
    void computeHog(const cv::Mat& sample, cv::Mat& row) const;
    std::vector<Detection> nonMaxSuppression(std::vector<Detection> detections) const;
    float iou(const cv::Rect& a, const cv::Rect& b) const;
    std::vector<Detection> temporalConfirm(const std::vector<Detection>& raw);
};
