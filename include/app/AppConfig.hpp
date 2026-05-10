#pragma once

#include <string>
#include <vector>

#include "detection/ClassicDetector.hpp"

struct AppConfig {
    bool showHelp = false;
    bool verbose = false;
    bool noUi = false;
    int maxFrames = 0;

    std::string inputPath = "input.mp4";
    std::string outputPath = "output.avi";
    std::string detectorType = "classic";
    std::string trackerType = "bytetrack";
    std::string profile = "default";

    int minArea = 280;
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
    int bgHistory = 400;
    double bgVar = 16.0;
    int fieldErode = 22;
    bool useColorCandidates = true;
    float minNonGreen = 0.22f;
    float minNonGreenFar = 0.12f;
    float minNonGreenNear = 0.24f;
    float minHRel = 0.03f;
    float maxHRel = 0.45f;
    float minFieldOverlap = 0.78f;
    float minBottomFieldSupport = 0.62f;
    float minGroundGreenSupport = 0.10f;
    bool requireBottomOnField = true;
    float horizonRel = 0.35f;
    float minHRelNear = 0.045f;
    float maxHRelNear = 0.45f;
    float minHRelFar = 0.014f;
    float maxHRelFar = 0.16f;
    int confirmFrames = 3;
    int maxCandidateMiss = 5;
    float confirmIou = 0.35f;
    float minMotionFar = 0.035f;
    float minMotionNear = 0.10f;
    float minFillFar = 0.12f;
    float minFillNear = 0.24f;
    float maxWidthRelFar = 0.08f;
    float maxWidthRelNear = 0.22f;
    float maxWideAspectFar = 1.45f;
    float maxWideAspectNear = 1.18f;
    float nmsThreshold = 0.28f;
    float splitWideAspect = 0.68f;
    float splitForceAspect = 0.95f;
    float splitValleyRatio = 0.26f;
    float splitMinComponentAreaRatio = 0.055f;
    int splitMaxParts = 4;

    bool dropUnknown = false;
    bool teamAuto = false;
    int teamWarmup = 60;
    int teamMinSamples = 40;

    bool useHogSvm = false;
    std::string svmModel = "dataset/hog_svm.yml";
    std::string svmPositives = "dataset/positives";
    std::string svmNegatives = "dataset/negatives";
    bool trainSvm = false;
    bool trainSvmOnly = false;
    bool saveHardNegatives = false;
    std::string hardNegativeDir = "dataset/negatives/hard";

    bool useCnnValidator = false;
    std::string cnnValidatorModel = "dataset/player_validator.onnx";
    float cnnValidatorMinConfidence = 0.65f;
    std::string teamCnnModel;
    float teamCnnMinConfidence = 0.55f;

    bool tacticalEnabled = true;
    bool calibrateHomography = false;
    std::string homographyFile = "dataset/homography.yml";
    bool exportCsv = true;
    std::string csvPath = "dataset/tracking_export.csv";

    ClassicDetectorParams toDetectorParams() const;
};

bool parseAppConfig(int argc, char** argv, AppConfig& config, std::string& error);
std::string appUsage(const char* executableName);
