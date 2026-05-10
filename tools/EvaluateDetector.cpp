#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "app/AppConfig.hpp"
#include "detection/ClassicDetector.hpp"
#include "evaluation/DetectionEvaluator.hpp"

namespace {

struct EvalConfig {
    AppConfig app;
    std::string labelsPath;
    std::string predictionsPath;
    float iouThreshold = 0.5f;
};

bool requiresValue(int index, int argc, const std::string& flag, std::string& error) {
    if (index + 1 < argc) return true;
    error = "Falta valor para " + flag;
    return false;
}

bool parseFloatArg(const std::string& value, float& out) {
    try {
        size_t consumed = 0;
        out = std::stof(value, &consumed);
        return consumed == value.size();
    } catch (...) {
        return false;
    }
}

std::string usage(const char* exe) {
    return std::string("Uso: ") + exe +
        " --input <video> --labels <csv> [opciones]\n\n"
        "CSV labels: frame,x,y,w,h o frame,id,x,y,w,h en coordenadas 1280x720.\n"
        "Opciones utiles:\n"
        "  --profile <default|sensitive>\n"
        "  --iou <float>                 Default: 0.5\n"
        "  --predictions <csv>           Guarda detecciones por frame\n"
        "  --max-frames <int>\n"
        "  --help\n";
}

bool parseEvalConfig(int argc, char** argv, EvalConfig& config, std::string& error) {
    std::vector<std::string> appArgs;
    appArgs.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            config.app.showHelp = true;
            return true;
        }
        if (arg == "--labels") {
            if (!requiresValue(i, argc, arg, error)) return false;
            config.labelsPath = argv[++i];
            continue;
        }
        if (arg == "--predictions") {
            if (!requiresValue(i, argc, arg, error)) return false;
            config.predictionsPath = argv[++i];
            continue;
        }
        if (arg == "--iou") {
            if (!requiresValue(i, argc, arg, error)) return false;
            if (!parseFloatArg(argv[++i], config.iouThreshold)) {
                error = "Valor invalido para --iou";
                return false;
            }
            continue;
        }

        appArgs.push_back(arg);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            appArgs.push_back(argv[++i]);
        }
    }

    std::vector<char*> appArgv;
    appArgv.reserve(appArgs.size());
    for (auto& arg : appArgs) appArgv.push_back(arg.data());
    if (!parseAppConfig((int)appArgv.size(), appArgv.data(), config.app, error)) return false;

    config.app.noUi = true;
    config.app.tacticalEnabled = false;
    config.app.exportCsv = false;

    if (config.labelsPath.empty()) {
        error = "Debes pasar --labels <csv>";
        return false;
    }
    config.iouThreshold = std::max(0.05f, std::min(0.95f, config.iouThreshold));
    return true;
}

void writePredictionsHeader(std::ofstream& file) {
    if (file.is_open()) {
        file << "frame,x,y,w,h,score\n";
    }
}

void writePredictions(std::ofstream& file, int frame, const std::vector<Detection>& detections) {
    if (!file.is_open()) return;
    for (const auto& d : detections) {
        file << frame << ","
             << d.bbox.x << ","
             << d.bbox.y << ","
             << d.bbox.width << ","
             << d.bbox.height << ","
             << d.score << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    EvalConfig config;
    std::string error;
    if (!parseEvalConfig(argc, argv, config, error)) {
        std::cerr << "[ERROR] " << error << "\n\n" << usage(argv[0]);
        return 1;
    }
    if (config.app.showHelp) {
        std::cout << usage(argv[0]);
        return 0;
    }

    std::string labelsError;
    auto labelsByFrame = loadGroundTruthCsv(config.labelsPath, labelsError);
    if (!labelsError.empty()) {
        std::cerr << "[ERROR] " << labelsError << std::endl;
        return 1;
    }
    if (labelsByFrame.empty()) {
        std::cerr << "[ERROR] El CSV de labels no contiene cajas." << std::endl;
        return 1;
    }

    cv::VideoCapture cap(config.app.inputPath);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] No se pudo abrir el video: " << config.app.inputPath << std::endl;
        return 1;
    }

    ClassicDetector detector;
    detector.configure(config.app.toDetectorParams());

    std::ofstream predictions;
    if (!config.predictionsPath.empty()) {
        predictions.open(config.predictionsPath, std::ios::out | std::ios::trunc);
        if (!predictions.is_open()) {
            std::cerr << "[ERROR] No se pudo abrir predictions CSV: "
                      << config.predictionsPath << std::endl;
            return 1;
        }
        writePredictionsHeader(predictions);
    }

    std::vector<FrameEvaluation> frameResults;
    cv::Mat frame;
    int frameIndex = 0;
    int lastLabelFrame = labelsByFrame.rbegin()->first;
    int frameLimit = config.app.maxFrames > 0
        ? std::min(config.app.maxFrames, lastLabelFrame)
        : lastLabelFrame;

    while (frameIndex < frameLimit && cap.read(frame)) {
        frameIndex++;
        cv::resize(frame, frame, cv::Size(1280, 720));
        auto detections = detector.detect(frame);
        writePredictions(predictions, frameIndex, detections);

        auto it = labelsByFrame.find(frameIndex);
        if (it == labelsByFrame.end()) continue;

        FrameEvaluation result = evaluateFrame(frameIndex, it->second, detections,
                                               config.iouThreshold);
        frameResults.push_back(result);
    }

    auto summary = summarizeEvaluations(frameResults);
    std::cout << "frames_evaluated," << summary.framesEvaluated << "\n"
              << "truth," << summary.truthCount << "\n"
              << "detections," << summary.detectionCount << "\n"
              << "tp," << summary.truePositives << "\n"
              << "fp," << summary.falsePositives << "\n"
              << "fn," << summary.falseNegatives << "\n"
              << "precision," << summary.precision() << "\n"
              << "recall," << summary.recall() << "\n"
              << "f1," << summary.f1() << "\n"
              << "mean_iou," << summary.meanMatchedIou << std::endl;

    return 0;
}
