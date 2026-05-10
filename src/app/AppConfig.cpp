#include "app/AppConfig.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

bool needsValue(int index, int argc, const std::string& flag, std::string& error) {
    if (index + 1 < argc) return true;
    error = "Falta valor para " + flag;
    return false;
}

int parseInt(const std::string& value, const std::string&) {
    size_t consumed = 0;
    int out = std::stoi(value, &consumed);
    if (consumed != value.size()) throw std::invalid_argument("valor entero invalido");
    return out;
}

float parseFloat(const std::string& value, const std::string&) {
    size_t consumed = 0;
    float out = std::stof(value, &consumed);
    if (consumed != value.size()) throw std::invalid_argument("valor decimal invalido");
    return out;
}

double parseDouble(const std::string& value, const std::string&) {
    size_t consumed = 0;
    double out = std::stod(value, &consumed);
    if (consumed != value.size()) throw std::invalid_argument("valor decimal invalido");
    return out;
}

bool parseBool(const std::string& value, const std::string& flag) {
    int raw = parseInt(value, flag);
    if (raw != 0 && raw != 1) throw std::invalid_argument("usa 0 o 1");
    return raw != 0;
}

template <typename T>
T clampValue(T value, T low, T high) {
    return std::max(low, std::min(high, value));
}

void applyProfile(AppConfig& config, const std::string& profile) {
    config.profile = profile;
    if (profile == "default") return;
    if (profile == "sensitive") {
        config.useCanny = true;
        config.useColorCandidates = false;
        config.confirmFrames = 1;
        config.minFieldOverlap = 0.55f;
        config.minBottomFieldSupport = 0.35f;
        config.minGroundGreenSupport = 0.04f;
        config.minNonGreen = 0.12f;
        config.minNonGreenFar = 0.06f;
        config.minNonGreenNear = 0.12f;
        return;
    }
    throw std::invalid_argument("perfil desconocido: usa default o sensitive");
}

} // namespace

ClassicDetectorParams AppConfig::toDetectorParams() const {
    ClassicDetectorParams params;
    params.minArea = minArea;
    params.maxArea = maxArea;
    params.minAreaFar = minAreaFar;
    params.minAreaNear = minAreaNear;
    params.maxAreaFar = maxAreaFar;
    params.maxAreaNear = maxAreaNear;
    params.blur = blur;
    params.morph = morph;
    params.useEqualize = useEqualize;
    params.useCanny = useCanny;
    params.canny1 = canny1;
    params.canny2 = canny2;
    params.bgHistory = bgHistory;
    params.bgVarThreshold = bgVar;
    params.fieldErode = fieldErode;
    params.useColorCandidates = useColorCandidates;
    params.minNonGreenRatio = minNonGreen;
    params.minNonGreenFar = minNonGreenFar;
    params.minNonGreenNear = minNonGreenNear;
    params.minHeightRel = minHRel;
    params.maxHeightRel = maxHRel;
    params.minFieldOverlap = minFieldOverlap;
    params.minBottomFieldSupport = minBottomFieldSupport;
    params.minGroundGreenSupport = minGroundGreenSupport;
    params.requireBottomOnField = requireBottomOnField;
    params.horizonRel = horizonRel;
    params.minHRelNear = minHRelNear;
    params.maxHRelNear = maxHRelNear;
    params.minHRelFar = minHRelFar;
    params.maxHRelFar = maxHRelFar;
    params.confirmFrames = confirmFrames;
    params.maxCandidateMiss = maxCandidateMiss;
    params.confirmIou = confirmIou;
    params.minMotionFar = minMotionFar;
    params.minMotionNear = minMotionNear;
    params.minFillFar = minFillFar;
    params.minFillNear = minFillNear;
    params.maxWidthRelFar = maxWidthRelFar;
    params.maxWidthRelNear = maxWidthRelNear;
    params.maxWideAspectFar = maxWideAspectFar;
    params.maxWideAspectNear = maxWideAspectNear;
    params.nmsThreshold = nmsThreshold;
    params.splitWideAspect = splitWideAspect;
    params.splitForceAspect = splitForceAspect;
    params.splitValleyRatio = splitValleyRatio;
    params.splitMinComponentAreaRatio = splitMinComponentAreaRatio;
    params.splitMaxParts = splitMaxParts;
    params.useHogSvm = useHogSvm;
    params.svmModelPath = svmModel;
    params.useCnnValidator = useCnnValidator;
    params.cnnValidatorModelPath = cnnValidatorModel;
    params.cnnValidatorMinConfidence = cnnValidatorMinConfidence;
    params.saveHardNegatives = saveHardNegatives;
    params.hardNegativeDir = hardNegativeDir;
    return params;
}

bool parseAppConfig(int argc, char** argv, AppConfig& config, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            config.showHelp = true;
            return true;
        }

        auto value = [&]() -> std::string {
            if (!needsValue(i, argc, arg, error)) return {};
            return argv[++i];
        };

        try {
            if (arg == "--input") config.inputPath = value();
            else if (arg == "--output") config.outputPath = value();
            else if (arg == "--profile") applyProfile(config, value());
            else if (arg == "--detector") config.detectorType = value();
            else if (arg == "--tracker") config.trackerType = value();
            else if (arg == "--min-area") config.minArea = std::max(50, parseInt(value(), arg));
            else if (arg == "--max-area") config.maxArea = std::max(0, parseInt(value(), arg));
            else if (arg == "--min-area-far") config.minAreaFar = std::max(20, parseInt(value(), arg));
            else if (arg == "--min-area-near") config.minAreaNear = std::max(config.minAreaFar, parseInt(value(), arg));
            else if (arg == "--max-area-far") config.maxAreaFar = std::max(config.minAreaFar + 1, parseInt(value(), arg));
            else if (arg == "--max-area-near") config.maxAreaNear = std::max(0, parseInt(value(), arg));
            else if (arg == "--blur") config.blur = std::max(0, parseInt(value(), arg));
            else if (arg == "--morph") config.morph = std::max(0, parseInt(value(), arg));
            else if (arg == "--equalize") config.useEqualize = parseBool(value(), arg);
            else if (arg == "--use-canny") config.useCanny = parseBool(value(), arg);
            else if (arg == "--canny1") config.canny1 = std::max(10, parseInt(value(), arg));
            else if (arg == "--canny2") config.canny2 = std::max(config.canny1, parseInt(value(), arg));
            else if (arg == "--bg-history") config.bgHistory = std::max(30, parseInt(value(), arg));
            else if (arg == "--bg-var") config.bgVar = std::max(4.0, parseDouble(value(), arg));
            else if (arg == "--field-erode") config.fieldErode = std::max(0, parseInt(value(), arg));
            else if (arg == "--color-candidates") config.useColorCandidates = parseBool(value(), arg);
            else if (arg == "--min-non-green") config.minNonGreen = clampValue(parseFloat(value(), arg), 0.0f, 0.9f);
            else if (arg == "--min-non-green-far") config.minNonGreenFar = clampValue(parseFloat(value(), arg), 0.0f, 0.9f);
            else if (arg == "--min-non-green-near") config.minNonGreenNear = clampValue(parseFloat(value(), arg), 0.0f, 0.9f);
            else if (arg == "--min-h-rel") config.minHRel = std::max(0.005f, parseFloat(value(), arg));
            else if (arg == "--max-h-rel") config.maxHRel = std::max(config.minHRel, parseFloat(value(), arg));
            else if (arg == "--min-field-overlap") config.minFieldOverlap = clampValue(parseFloat(value(), arg), 0.0f, 1.0f);
            else if (arg == "--min-bottom-field-support") config.minBottomFieldSupport = clampValue(parseFloat(value(), arg), 0.0f, 1.0f);
            else if (arg == "--min-ground-green-support") config.minGroundGreenSupport = clampValue(parseFloat(value(), arg), 0.0f, 1.0f);
            else if (arg == "--require-bottom-on-field") config.requireBottomOnField = parseBool(value(), arg);
            else if (arg == "--horizon-rel") config.horizonRel = clampValue(parseFloat(value(), arg), 0.05f, 0.95f);
            else if (arg == "--min-h-rel-near") config.minHRelNear = std::max(0.005f, parseFloat(value(), arg));
            else if (arg == "--max-h-rel-near") config.maxHRelNear = std::max(config.minHRelNear, parseFloat(value(), arg));
            else if (arg == "--min-h-rel-far") config.minHRelFar = std::max(0.001f, parseFloat(value(), arg));
            else if (arg == "--max-h-rel-far") config.maxHRelFar = std::max(config.minHRelFar, parseFloat(value(), arg));
            else if (arg == "--confirm-frames") config.confirmFrames = std::max(1, parseInt(value(), arg));
            else if (arg == "--max-candidate-miss") config.maxCandidateMiss = std::max(0, parseInt(value(), arg));
            else if (arg == "--confirm-iou") config.confirmIou = clampValue(parseFloat(value(), arg), 0.05f, 0.9f);
            else if (arg == "--min-motion-far") config.minMotionFar = clampValue(parseFloat(value(), arg), 0.0f, 1.0f);
            else if (arg == "--min-motion-near") config.minMotionNear = clampValue(parseFloat(value(), arg), 0.0f, 1.0f);
            else if (arg == "--min-fill-far") config.minFillFar = clampValue(parseFloat(value(), arg), 0.0f, 1.0f);
            else if (arg == "--min-fill-near") config.minFillNear = clampValue(parseFloat(value(), arg), 0.0f, 1.0f);
            else if (arg == "--max-width-rel-far") config.maxWidthRelFar = clampValue(parseFloat(value(), arg), 0.01f, 1.0f);
            else if (arg == "--max-width-rel-near") config.maxWidthRelNear = std::max(config.maxWidthRelFar, clampValue(parseFloat(value(), arg), 0.01f, 1.0f));
            else if (arg == "--max-wide-aspect-far") config.maxWideAspectFar = std::max(0.5f, parseFloat(value(), arg));
            else if (arg == "--max-wide-aspect-near") config.maxWideAspectNear = std::max(0.5f, parseFloat(value(), arg));
            else if (arg == "--nms-threshold") config.nmsThreshold = clampValue(parseFloat(value(), arg), 0.05f, 0.95f);
            else if (arg == "--split-wide-aspect") config.splitWideAspect = std::max(0.3f, parseFloat(value(), arg));
            else if (arg == "--split-force-aspect") config.splitForceAspect = std::max(config.splitWideAspect, parseFloat(value(), arg));
            else if (arg == "--split-valley-ratio") config.splitValleyRatio = clampValue(parseFloat(value(), arg), 0.05f, 0.8f);
            else if (arg == "--split-min-component-area-ratio") config.splitMinComponentAreaRatio = clampValue(parseFloat(value(), arg), 0.005f, 0.5f);
            else if (arg == "--split-max-parts") config.splitMaxParts = clampValue(parseInt(value(), arg), 2, 8);
            else if (arg == "--drop-unknown") config.dropUnknown = parseBool(value(), arg);
            else if (arg == "--team-auto") config.teamAuto = parseBool(value(), arg);
            else if (arg == "--team-warmup") config.teamWarmup = std::max(10, parseInt(value(), arg));
            else if (arg == "--team-min-samples") config.teamMinSamples = std::max(10, parseInt(value(), arg));
            else if (arg == "--hog-svm") config.useHogSvm = parseBool(value(), arg);
            else if (arg == "--svm-model") config.svmModel = value();
            else if (arg == "--svm-positives") config.svmPositives = value();
            else if (arg == "--svm-negatives") config.svmNegatives = value();
            else if (arg == "--train-svm") config.trainSvm = parseBool(value(), arg);
            else if (arg == "--train-svm-only") config.trainSvmOnly = parseBool(value(), arg);
            else if (arg == "--save-hard-negatives") config.saveHardNegatives = parseBool(value(), arg);
            else if (arg == "--hard-negative-dir") config.hardNegativeDir = value();
            else if (arg == "--cnn-validator") config.useCnnValidator = parseBool(value(), arg);
            else if (arg == "--cnn-validator-model") config.cnnValidatorModel = value();
            else if (arg == "--cnn-validator-threshold") config.cnnValidatorMinConfidence = clampValue(parseFloat(value(), arg), 0.01f, 0.99f);
            else if (arg == "--team-cnn-model") config.teamCnnModel = value();
            else if (arg == "--team-cnn-threshold") config.teamCnnMinConfidence = clampValue(parseFloat(value(), arg), 0.01f, 0.99f);
            else if (arg == "--tactical") config.tacticalEnabled = parseBool(value(), arg);
            else if (arg == "--calibrate-homography") config.calibrateHomography = parseBool(value(), arg);
            else if (arg == "--homography-file") config.homographyFile = value();
            else if (arg == "--export-csv") config.exportCsv = parseBool(value(), arg);
            else if (arg == "--csv") config.csvPath = value();
            else if (arg == "--verbose") config.verbose = parseBool(value(), arg);
            else if (arg == "--no-ui") config.noUi = parseBool(value(), arg);
            else if (arg == "--max-frames") config.maxFrames = std::max(0, parseInt(value(), arg));
            else {
                error = "Parametro desconocido: " + arg;
                return false;
            }
        } catch (const std::exception& ex) {
            error = "Valor invalido para " + arg + ": " + ex.what();
            return false;
        }
        if (!error.empty()) return false;
    }

    if (config.detectorType != "classic") {
        error = "Detector no soportado: " + config.detectorType + ". Usa classic.";
        return false;
    }
    if (config.trackerType != "bytetrack" && config.trackerType != "deepsort") {
        error = "Tracker no soportado: " + config.trackerType + ". Usa bytetrack o deepsort.";
        return false;
    }
    return true;
}

std::string appUsage(const char* executableName) {
    std::ostringstream out;
    out << "Uso: " << executableName << " [opciones]\n\n"
        << "Opciones principales:\n"
        << "  --input <path>                  Video de entrada. Default: input.mp4\n"
        << "  --output <path>                 Video de salida. Default: output.avi\n"
        << "  --profile <default|sensitive>   Perfil de parametros clasicos\n"
        << "  --tracker <bytetrack|deepsort>  Tracker a usar. Default: bytetrack\n"
        << "  --tactical <0|1>                Activa analisis tactico. Default: 1\n"
        << "  --homography-file <path>        YAML de homografia\n"
        << "  --calibrate-homography <0|1>    Calibra con 4 clicks\n"
        << "  --export-csv <0|1>              Exporta tracking tactico\n"
        << "  --csv <path>                    Ruta CSV\n"
        << "  --team-auto <0|1>               Agrupa equipos por color\n"
        << "  --drop-unknown <0|1>            Omite jugadores sin equipo confiable\n"
        << "  --color-candidates <0|1>        Segmenta candidatos no verdes en cancha\n"
        << "  --hog-svm <0|1>                 Validador HOG+SVM opcional\n"
        << "  --train-svm-only <0|1>          Entrena SVM y termina\n"
        << "  --verbose <0|1>                 Log por frame\n"
        << "  --no-ui <0|1>                   Ejecuta sin ventana interactiva\n"
        << "  --max-frames <int>              Procesa solo N frames; 0 procesa todo\n"
        << "  --help                          Muestra esta ayuda\n";
    return out.str();
}
