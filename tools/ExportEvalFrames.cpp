#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace {

struct Config {
    std::string input = "input.mp4";
    std::string outputDir = "dataset/eval_frames";
    int start = 1;
    int end = 250;
    int step = 25;
};

bool parseInt(const std::string& value, int& out) {
    try {
        size_t consumed = 0;
        out = std::stoi(value, &consumed);
        return consumed == value.size();
    } catch (...) {
        return false;
    }
}

std::string usage(const char* exe) {
    return std::string("Uso: ") + exe +
        " --input input.mp4 --output-dir dataset/eval_frames --start 1 --end 250 --step 25\n";
}

bool parseArgs(int argc, char** argv, Config& config, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") return true;
        if (i + 1 >= argc) {
            error = "Falta valor para " + arg;
            return false;
        }
        std::string value = argv[++i];
        if (arg == "--input") config.input = value;
        else if (arg == "--output-dir") config.outputDir = value;
        else if (arg == "--start" && !parseInt(value, config.start)) error = "Valor invalido para --start";
        else if (arg == "--end" && !parseInt(value, config.end)) error = "Valor invalido para --end";
        else if (arg == "--step" && !parseInt(value, config.step)) error = "Valor invalido para --step";
        else if (arg != "--start" && arg != "--end" && arg != "--step") {
            error = "Parametro desconocido: " + arg;
            return false;
        }
        if (!error.empty()) return false;
    }
    config.start = std::max(1, config.start);
    config.end = std::max(config.start, config.end);
    config.step = std::max(1, config.step);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Config config;
    std::string error;
    if (!parseArgs(argc, argv, config, error)) {
        std::cerr << "[ERROR] " << error << "\n\n" << usage(argv[0]);
        return 1;
    }

    cv::VideoCapture cap(config.input);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] No se pudo abrir video: " << config.input << std::endl;
        return 1;
    }

    std::filesystem::create_directories(config.outputDir);
    int saved = 0;
    cv::Mat frame;
    for (int frameNo = config.start; frameNo <= config.end; frameNo += config.step) {
        cap.set(cv::CAP_PROP_POS_FRAMES, frameNo - 1);
        if (!cap.read(frame)) break;
        cv::resize(frame, frame, cv::Size(1280, 720));
        std::string path = config.outputDir + "/frame_" + std::to_string(frameNo) + ".jpg";
        if (cv::imwrite(path, frame)) saved++;
    }

    std::cout << "[INFO] Frames exportados: " << saved
              << " en " << config.outputDir << std::endl;
    std::cout << "[INFO] Etiqueta con CSV: frame,x,y,w,h o frame,id,x,y,w,h" << std::endl;
    return 0;
}
