#include <opencv2/opencv.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Args {
    std::string imagesDir;
    std::string annotationsPath;
    std::string outputDir = "dataset";
    int frameStep = 1;
    int negativesPerFrame = 4;
    int personClass = -1;
    float minVisibility = 0.0f;
    float padding = 0.12f;
    float negMaxIou = 0.05f;
    int minCropW = 8;
    int minCropH = 16;
};

struct MotBox {
    int frame = 0;
    int id = -1;
    cv::Rect2f box;
    float conf = 1.0f;
    int cls = -1;
    float visibility = 1.0f;
};

static void printUsage() {
    std::cout
        << "Uso:\n"
        << "  dataset_cropper --images <dir_frames> --annotations <gt.txt> --output dataset\n\n"
        << "Formato esperado de anotaciones: MOTChallenge/SportsMOT/SoccerNet Tracking\n"
        << "  frame,id,x,y,w,h,conf,class,visibility\n\n"
        << "Opciones:\n"
        << "  --frame-step <int>                 usar 1 de cada N frames\n"
        << "  --negatives-per-frame <int>        negativos por frame anotado\n"
        << "  --person-class <int>               filtra clase; -1 acepta todas\n"
        << "  --min-visibility <float>           visibilidad minima\n"
        << "  --padding <float>                  margen alrededor del jugador\n"
        << "  --neg-max-iou <float>              maximo solapamiento de negativos\n";
}

static std::vector<std::string> splitLine(std::string line) {
    for (char& ch : line) {
        if (ch == ',') ch = ' ';
    }
    std::stringstream ss(line);
    std::vector<std::string> out;
    std::string token;
    while (ss >> token) out.push_back(token);
    return out;
}

static bool parseArgs(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) return "";
            return argv[++i];
        };
        if (arg == "--images") args.imagesDir = next();
        else if (arg == "--annotations") args.annotationsPath = next();
        else if (arg == "--output") args.outputDir = next();
        else if (arg == "--frame-step") args.frameStep = std::max(1, std::atoi(next().c_str()));
        else if (arg == "--negatives-per-frame") args.negativesPerFrame = std::max(0, std::atoi(next().c_str()));
        else if (arg == "--person-class") args.personClass = std::atoi(next().c_str());
        else if (arg == "--min-visibility") args.minVisibility = std::max(0.0f, std::min(1.0f, (float)std::atof(next().c_str())));
        else if (arg == "--padding") args.padding = std::max(0.0f, std::min(1.0f, (float)std::atof(next().c_str())));
        else if (arg == "--neg-max-iou") args.negMaxIou = std::max(0.0f, std::min(1.0f, (float)std::atof(next().c_str())));
        else if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        }
    }
    if (args.imagesDir.empty() || args.annotationsPath.empty()) {
        printUsage();
        return false;
    }
    return true;
}

static bool isImageFile(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

static std::vector<fs::path> listImages(const std::string& imagesDir) {
    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(imagesDir)) {
        if (entry.is_regular_file() && isImageFile(entry.path())) {
            images.push_back(entry.path());
        }
    }
    std::sort(images.begin(), images.end());
    return images;
}

static std::map<int, fs::path> mapFrames(const std::vector<fs::path>& images) {
    std::map<int, fs::path> byFrame;
    for (size_t i = 0; i < images.size(); ++i) {
        byFrame[(int)i + 1] = images[i];
        try {
            int stemFrame = std::stoi(images[i].stem().string());
            byFrame[stemFrame] = images[i];
        } catch (...) {
        }
    }
    return byFrame;
}

static std::map<int, std::vector<MotBox>> loadAnnotations(const Args& args) {
    std::ifstream in(args.annotationsPath);
    std::map<int, std::vector<MotBox>> boxesByFrame;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> cols = splitLine(line);
        if (cols.size() < 6) continue;

        MotBox b;
        b.frame = std::atoi(cols[0].c_str());
        b.id = std::atoi(cols[1].c_str());
        b.box = cv::Rect2f((float)std::atof(cols[2].c_str()),
                           (float)std::atof(cols[3].c_str()),
                           (float)std::atof(cols[4].c_str()),
                           (float)std::atof(cols[5].c_str()));
        if (cols.size() > 6) b.conf = (float)std::atof(cols[6].c_str());
        if (cols.size() > 7) b.cls = std::atoi(cols[7].c_str());
        if (cols.size() > 8) b.visibility = (float)std::atof(cols[8].c_str());

        if (b.frame <= 0 || b.box.width <= 0 || b.box.height <= 0) continue;
        if (args.personClass >= 0 && b.cls != args.personClass) continue;
        // Some MOT-style exports use negative visibility to mean "unknown".
        // Keep those boxes instead of dropping the entire sequence.
        if (b.visibility >= 0.0f && b.visibility < args.minVisibility) continue;
        boxesByFrame[b.frame].push_back(b);
    }
    return boxesByFrame;
}

static float iou(const cv::Rect& a, const cv::Rect& b) {
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

static cv::Rect paddedRect(const cv::Rect2f& src, float pad, const cv::Size& bounds) {
    float px = src.width * pad;
    float py = src.height * pad;
    cv::Rect r((int)std::round(src.x - px),
               (int)std::round(src.y - py),
               (int)std::round(src.width + 2.0f * px),
               (int)std::round(src.height + 2.0f * py));
    return r & cv::Rect(0, 0, bounds.width, bounds.height);
}

static bool overlapsAny(const cv::Rect& candidate, const std::vector<cv::Rect>& positives, float maxIou) {
    for (const auto& p : positives) {
        if (iou(candidate, p) > maxIou) return true;
    }
    return false;
}

static void saveCrop(const cv::Mat& frame, const cv::Rect& crop, const fs::path& dir,
                     const std::string& prefix, int frameId, int index) {
    if (crop.area() <= 0) return;
    std::ostringstream name;
    name << prefix << "_f" << std::setw(6) << std::setfill('0') << frameId
         << "_" << std::setw(4) << std::setfill('0') << index << ".png";
    cv::imwrite((dir / name.str()).string(), frame(crop));
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        }
    }

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;

    std::vector<fs::path> images = listImages(args.imagesDir);
    if (images.empty()) {
        std::cerr << "[ERROR] No se encontraron imagenes en: " << args.imagesDir << std::endl;
        return 1;
    }

    auto frames = mapFrames(images);
    auto annotations = loadAnnotations(args);
    fs::path positivesDir = fs::path(args.outputDir) / "positives";
    fs::path negativesDir = fs::path(args.outputDir) / "negatives";
    fs::create_directories(positivesDir);
    fs::create_directories(negativesDir);

    int positivesSaved = 0;
    int negativesSaved = 0;
    int framesUsed = 0;

    for (const auto& [frameId, boxes] : annotations) {
        if ((frameId - 1) % args.frameStep != 0) continue;
        auto imgIt = frames.find(frameId);
        if (imgIt == frames.end()) continue;

        cv::Mat frame = cv::imread(imgIt->second.string(), cv::IMREAD_COLOR);
        if (frame.empty()) continue;
        framesUsed++;

        std::vector<cv::Rect> positiveRects;
        int localPositive = 0;
        for (const auto& b : boxes) {
            cv::Rect r = paddedRect(b.box, args.padding, frame.size());
            if (r.width < args.minCropW || r.height < args.minCropH) continue;
            positiveRects.push_back(r);
            saveCrop(frame, r, positivesDir, "player", frameId, localPositive++);
            positivesSaved++;
        }

        if (positiveRects.empty() || args.negativesPerFrame <= 0) continue;

        int avgW = 0;
        int avgH = 0;
        for (const auto& r : positiveRects) {
            avgW += r.width;
            avgH += r.height;
        }
        avgW = std::max(args.minCropW, avgW / (int)positiveRects.size());
        avgH = std::max(args.minCropH, avgH / (int)positiveRects.size());

        cv::RNG rng((uint64)frameId * 2654435761ULL);
        int localNegative = 0;
        int attempts = 0;
        while (localNegative < args.negativesPerFrame && attempts < args.negativesPerFrame * 80) {
            attempts++;
            int w = std::max(args.minCropW, (int)std::round(avgW * rng.uniform(0.65, 1.35)));
            int h = std::max(args.minCropH, (int)std::round(avgH * rng.uniform(0.65, 1.35)));
            if (w >= frame.cols || h >= frame.rows) continue;
            int x = rng.uniform(0, frame.cols - w);
            int y = rng.uniform(0, frame.rows - h);
            cv::Rect candidate(x, y, w, h);
            if (overlapsAny(candidate, positiveRects, args.negMaxIou)) continue;
            saveCrop(frame, candidate, negativesDir, "neg", frameId, localNegative++);
            negativesSaved++;
        }
    }

    std::cout << "[INFO] Frames usados: " << framesUsed << std::endl;
    std::cout << "[INFO] Positivos guardados: " << positivesSaved << std::endl;
    std::cout << "[INFO] Negativos guardados: " << negativesSaved << std::endl;
    std::cout << "[INFO] Dataset local: " << fs::absolute(args.outputDir).string() << std::endl;
    return 0;
}
