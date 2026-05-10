#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <optional>
#include <iostream>
#include <chrono>
#include <sstream>
#include "detection/ClassicDetector.hpp"
#include "dnn/CropClassifier.hpp"
#include "tracking/ByteTrackTracker.hpp"
#include "tracking/DeepSORTTracker.hpp"
#include "reid/ReIDEmbedder.hpp"
#include "analysis/TacticalAnalyzer.hpp"
#include "analysis/AdvancedTactics.hpp"
#include "app/AppConfig.hpp"
#include "common/Detection.hpp"

struct UIState {
    int showBoxes = 1;
    int showIds = 1;
    int showCentroids = 1;
    int showTeams = 1;
    int showPanel = 1;
    int showTrails = 1;
    int lineThickness = 2;     // 1..5
    int fontScale10 = 7;       // 4..12 -> 0.4..1.2
    int panelAlpha = 45;       // 10..80
};

static std::vector<std::string> splitItems(const std::string& text, const std::string& sep = " | ") {
    std::vector<std::string> out;
    if (text.empty()) return out;
    size_t start = 0;
    while (start < text.size()) {
        size_t pos = text.find(sep, start);
        std::string item = text.substr(start, pos == std::string::npos ? std::string::npos : pos - start);
        if (!item.empty()) out.push_back(item);
        if (pos == std::string::npos) break;
        start = pos + sep.size();
    }
    return out;
}

static std::string ellipsize(const std::string& text, size_t maxChars) {
    if (text.size() <= maxChars) return text;
    if (maxChars <= 3) return text.substr(0, maxChars);
    return text.substr(0, maxChars - 3) + "...";
}

static void drawMetricBar(cv::Mat& frame, const cv::Point& pos, int width, float value,
                          const cv::Scalar& color, const std::string& label) {
    value = std::max(0.0f, std::min(1.0f, value));
    cv::putText(frame, label, pos, cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(205, 215, 215), 1);
    cv::Rect bg(pos.x, pos.y + 8, width, 8);
    cv::rectangle(frame, bg, cv::Scalar(55, 65, 65), -1);
    cv::rectangle(frame, cv::Rect(bg.x, bg.y, (int)std::round(width * value), bg.height), color, -1);
}

static void drawPanel(cv::Mat& frame, const UIState& ui, const std::string& detectorType,
                      const std::string& trackerType, int frameIdx, int playerCount,
                      float compactRed, float compactWhite, const std::string& dangerText,
                      const std::string& alertsText) {
    if (!ui.showPanel) return;
    cv::Mat overlay = frame.clone();
    cv::Rect panel(16, 16, 430, 214);
    cv::rectangle(overlay, panel, cv::Scalar(18, 28, 28), -1);
    cv::rectangle(overlay, panel, cv::Scalar(60, 95, 85), 1);
    double alpha = std::max(0.58, ui.panelAlpha / 100.0);
    cv::addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame);

    auto put = [&](int y, const std::string& text, const cv::Scalar& c, double fs = 0.46, int t = 1) {
        cv::putText(frame, text, cv::Point(32, y), cv::FONT_HERSHEY_SIMPLEX, fs, c, t);
    };

    cv::rectangle(frame, cv::Rect(16, 16, 430, 30), cv::Scalar(28, 54, 48), -1);
    put(38, "TACTICAL LIVE ANALYSIS", cv::Scalar(185, 255, 220), 0.52, 1);
    cv::putText(frame, "Frame " + std::to_string(frameIdx), cv::Point(336, 38),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(210, 220, 220), 1);

    put(67, "Detector " + detectorType + "  |  Tracker " + trackerType,
        cv::Scalar(220, 230, 230), 0.43, 1);
    put(91, "Jugadores activos: " + std::to_string(playerCount),
        playerCount >= 8 ? cv::Scalar(80, 255, 150) : cv::Scalar(80, 205, 255), 0.48, 1);

    drawMetricBar(frame, cv::Point(32, 116), 160, compactRed, cv::Scalar(45, 65, 230),
                  "Compactacion Rojo " + std::to_string((int)(compactRed * 100)) + "%");
    drawMetricBar(frame, cv::Point(222, 116), 160, compactWhite, cv::Scalar(235, 235, 235),
                  "Compactacion Blanco " + std::to_string((int)(compactWhite * 100)) + "%");

    int y = 158;
    if (!dangerText.empty()) {
        cv::rectangle(frame, cv::Rect(32, y - 17, 382, 22), cv::Scalar(35, 35, 95), -1);
        put(y, "ALERTA: " + ellipsize(dangerText, 46), cv::Scalar(80, 120, 255), 0.43, 1);
        y += 28;
    } else {
        cv::rectangle(frame, cv::Rect(32, y - 17, 170, 22), cv::Scalar(38, 64, 45), -1);
        put(y, "Riesgo estable", cv::Scalar(95, 245, 155), 0.43, 1);
        y += 28;
    }

    auto alerts = splitItems(alertsText);
    if (alerts.empty()) {
        put(y, "Sin alertas tacticas criticas", cv::Scalar(150, 160, 160), 0.42, 1);
    } else {
        int shown = 0;
        for (const auto& alert : alerts) {
            if (shown >= 2) break;
            put(y + shown * 20, "- " + ellipsize(alert, 48), cv::Scalar(0, 230, 245), 0.4, 1);
            shown++;
        }
    }
}

static void drawTrail(cv::Mat& frame, const std::deque<cv::Point>& pts, const cv::Scalar& color) {
    if (pts.size() < 2) return;
    for (size_t i = 1; i < pts.size(); ++i) {
        cv::line(frame, pts[i - 1], pts[i], color, 2);
    }
}

static int smoothTeamLabel(std::deque<int>& history, int label, size_t maxHist) {
    if (label != 2) {
        history.push_back(label);
        if (history.size() > maxHist) history.pop_front();
    }
    int count0 = 0;
    int count1 = 0;
    for (int v : history) {
        if (v == 0) count0++;
        else if (v == 1) count1++;
    }
    if (count0 == 0 && count1 == 0) return 2;
    return (count0 >= count1) ? 0 : 1;
}

static int smoothThreeClassLabel(std::deque<int>& history, int label, size_t maxHist) {
    if (label >= 0 && label <= 2) {
        history.push_back(label);
        if (history.size() > maxHist) history.pop_front();
    }
    int counts[3] = {0, 0, 0};
    for (int v : history) {
        if (v >= 0 && v <= 2) counts[v]++;
    }
    int best = 2;
    for (int i = 0; i < 3; ++i) {
        if (counts[i] > counts[best]) best = i;
    }
    return best;
}

static cv::Rect expandRect(const cv::Rect& r, float scale, const cv::Size& bounds) {
    float cx = r.x + r.width * 0.5f;
    float cy = r.y + r.height * 0.5f;
    float w = r.width * scale;
    float h = r.height * scale;
    cv::Rect out((int)std::round(cx - w * 0.5f), (int)std::round(cy - h * 0.5f),
                 (int)std::round(w), (int)std::round(h));
    return out & cv::Rect(0, 0, bounds.width, bounds.height);
}

struct FieldOverlayState {
    std::vector<cv::Vec4i> lines;
    std::optional<cv::Point> ballCenter;
    int ballRadius = 0;
    cv::Rect goalLeft;
    cv::Rect goalRight;
    int lastUpdateFrame = -999;
};

static cv::Mat quickFieldMask(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::Mat greenMask;
    cv::inRange(hsv, cv::Scalar(35, 40, 40), cv::Scalar(90, 255, 255), greenMask);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(greenMask, greenMask, cv::MORPH_CLOSE, kernel);
    return greenMask;
}

static void detectLinesGoalsBall(const cv::Mat& frame, FieldOverlayState& st) {
    st.lines.clear();
    st.ballCenter.reset();
    st.ballRadius = 0;
    st.goalLeft = cv::Rect();
    st.goalRight = cv::Rect();

    cv::Mat fieldMask = quickFieldMask(frame);

    // Lines: white-ish on field
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::Mat whiteMask;
    cv::inRange(hsv, cv::Scalar(0, 0, 170), cv::Scalar(180, 60, 255), whiteMask);
    cv::bitwise_and(whiteMask, fieldMask, whiteMask);
    cv::Mat edges;
    cv::Canny(whiteMask, edges, 50, 150);
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180.0, 60, 40, 8);
    st.lines = lines;

    // Goals: look for near-vertical white lines at left/right edges
    std::vector<cv::Vec4i> leftLines, rightLines;
    int w = frame.cols;
    for (const auto& l : lines) {
        int x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
        int dx = std::abs(x2 - x1);
        int dy = std::abs(y2 - y1);
        if (dy < 40) continue;
        if (dx > dy * 0.2) continue;
        int mx = (x1 + x2) / 2;
        if (mx < w * 0.15) leftLines.push_back(l);
        else if (mx > w * 0.85) rightLines.push_back(l);
    }
    auto boxFromLines = [&](const std::vector<cv::Vec4i>& ls) -> cv::Rect {
        if (ls.empty()) return cv::Rect();
        int minx = w, miny = frame.rows, maxx = 0, maxy = 0;
        for (const auto& l : ls) {
            minx = std::min(minx, std::min(l[0], l[2]));
            maxx = std::max(maxx, std::max(l[0], l[2]));
            miny = std::min(miny, std::min(l[1], l[3]));
            maxy = std::max(maxy, std::max(l[1], l[3]));
        }
        return cv::Rect(minx, miny, std::max(1, maxx - minx), std::max(1, maxy - miny));
    };
    st.goalLeft = boxFromLines(leftLines);
    st.goalRight = boxFromLines(rightLines);

    // Ball detection disabled by request
}

static void drawBoxPretty(cv::Mat& frame, const cv::Rect& box, const cv::Scalar& color, int thickness) {
    if (box.area() <= 0) return;
    int t = std::max(1, thickness);
    cv::rectangle(frame, box, cv::Scalar(0, 0, 0), t + 2);
    cv::rectangle(frame, box, color, t);
}

static cv::Rect shrinkRect(const cv::Rect& r, float scale) {
    float cx = r.x + r.width * 0.5f;
    float cy = r.y + r.height * 0.5f;
    float w = r.width * scale;
    float h = r.height * scale;
    cv::Rect out((int)std::round(cx - w * 0.5f), (int)std::round(cy - h * 0.5f),
                 (int)std::round(w), (int)std::round(h));
    return out;
}

int main(int argc, char** argv) {
    AppConfig config;
    std::string configError;
    if (!parseAppConfig(argc, argv, config, configError)) {
        std::cerr << "[ERROR] " << configError << "\n\n" << appUsage(argv[0]);
        return 1;
    }
    if (config.showHelp) {
        std::cout << appUsage(argv[0]);
        return 0;
    }
    std::cout << "[INFO] Inicio del programa" << std::endl;

    std::filesystem::path outputPath = config.outputPath;
    if (!outputPath.has_parent_path()) {
        outputPath = std::filesystem::current_path() / outputPath;
    }
    if (outputPath.has_parent_path()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }

    ClassicDetector detector;
    ClassicDetectorParams detParams = config.toDetectorParams();

    if (config.trainSvm || config.trainSvmOnly) {
        std::cout << "[INFO] Entrenando HOG+SVM desde cero con positivos: " << config.svmPositives
                  << " y negativos: " << config.svmNegatives << std::endl;
        if (!detector.trainSvmFromDataset(config.svmPositives, config.svmNegatives, config.svmModel)) {
            std::cerr << "[ERROR] No se pudo entrenar el SVM. Revisa dataset/positives y dataset/negatives." << std::endl;
            return 1;
        }
        std::cout << "[INFO] Modelo SVM guardado en: " << config.svmModel << std::endl;
        if (config.trainSvmOnly) return 0;
    }

    cv::VideoCapture cap(config.inputPath);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] No se pudo abrir el archivo de entrada. Verifica la ruta." << std::endl;
        return 1;
    }

    FieldHomography fieldHomography;
    if (config.tacticalEnabled) {
        if (fieldHomography.load(config.homographyFile)) {
            std::cout << "[INFO] Homografia cargada: " << config.homographyFile << std::endl;
        } else {
            std::cout << "[INFO] No hay homografia lista. Usa --calibrate-homography 1 para crear "
                      << config.homographyFile << std::endl;
        }
        if (config.calibrateHomography) {
            cv::Mat firstFrame;
            if (cap.read(firstFrame)) {
                cv::resize(firstFrame, firstFrame, cv::Size(1280, 720));
                if (fieldHomography.calibrateFromFrame(firstFrame, "Calibrar homografia")) {
                    if (fieldHomography.save(config.homographyFile)) {
                        std::cout << "[INFO] Homografia guardada en: " << config.homographyFile << std::endl;
                    }
                } else {
                    std::cout << "[WARN] Calibracion cancelada o incompleta." << std::endl;
                }
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            }
        }
    }

    cv::VideoWriter writer;
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0 || !std::isfinite(fps)) fps = 25.0;
    const cv::Size outSize(1280, 720);
    auto openWriter = [&](int codec) {
        writer.open(outputPath.string(), cv::CAP_FFMPEG, codec, fps, outSize);
        if (!writer.isOpened()) {
            writer.open(outputPath.string(), codec, fps, outSize);
        }
    };
    int fourcc = cv::VideoWriter::fourcc('X','V','I','D');
    openWriter(fourcc);
    if (!writer.isOpened()) {
        fourcc = cv::VideoWriter::fourcc('M','J','P','G');
        openWriter(fourcc);
    }
    if (!writer.isOpened()) {
        fourcc = cv::VideoWriter::fourcc('I','Y','U','V');
        openWriter(fourcc);
    }
    std::cout << "[INFO] Output: " << outputPath.string() << std::endl;

    detector.configure(detParams);

    ByteTrackTracker byteTracker;
    DeepSORTTracker deepSort;
    ReIDEmbedder reid;
    TacticalAnalyzer tactical;
    TeamColorClassifier teamClassifier;
    teamClassifier.configure(config.teamAuto, config.teamWarmup, config.teamMinSamples);
    CropClassifier teamCnn;
    if (!config.teamCnnModel.empty()) {
        if (teamCnn.load(config.teamCnnModel)) {
            std::cout << "[INFO] CNN equipos cargada: " << config.teamCnnModel << std::endl;
        } else {
            std::cerr << "[WARN] No se pudo cargar CNN equipos: " << config.teamCnnModel << std::endl;
        }
    }
    TrackingDataExporter exporter;
    if (config.tacticalEnabled && config.exportCsv) {
        if (exporter.open(config.csvPath)) {
            std::cout << "[INFO] CSV tactico: " << config.csvPath << std::endl;
        } else {
            std::cerr << "[WARN] No se pudo abrir CSV tactico: " << config.csvPath << std::endl;
        }
    }
    HeatmapAnalyzer heatmaps;
    FormationAnalyzer formationAnalyzer;
    PassingLaneAnalyzer passingAnalyzer;
    SpaceControlAnalyzer spaceAnalyzer;
    StrategyAdvisor strategyAdvisor;
    TacticalMinimap tacticalMinimap;

    UIState ui;
    const std::string windowName = "Football Analytics - UI";
    if (!config.noUi) {
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(windowName, 1280, 720);
        cv::createTrackbar("LineThick", windowName, nullptr, 5);
        cv::createTrackbar("Font x0.1", windowName, nullptr, 12);
        cv::createTrackbar("PanelAlpha", windowName, nullptr, 80);
        cv::setTrackbarPos("LineThick", windowName, ui.lineThickness);
        cv::setTrackbarPos("Font x0.1", windowName, ui.fontScale10);
        cv::setTrackbarPos("PanelAlpha", windowName, ui.panelAlpha);
    }

    bool paused = false;
    int stepFrames = 0;
    std::unordered_map<int, std::deque<cv::Point>> trails;
    std::unordered_map<int, std::deque<int>> teamHistory;
    std::unordered_map<int, cv::Point2f> lastFieldPos;
    const size_t maxTrail = 20;
    const size_t maxTeamHist = 8;
    FieldOverlayState overlay;

    cv::Mat frame;
    cv::Mat renderFrame;
    int frameCount = 0;
    long long totalDetections = 0;
    long long totalTracks = 0;
    int maxTracks = 0;
    auto processingStart = std::chrono::steady_clock::now();
    while (true) {
        bool processedFrame = false;
        if (!config.noUi && cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1) {
            break;
        }
        if (!config.noUi) {
            ui.lineThickness = std::max(1, cv::getTrackbarPos("LineThick", windowName));
            ui.fontScale10 = std::max(4, cv::getTrackbarPos("Font x0.1", windowName));
            ui.panelAlpha = std::max(10, cv::getTrackbarPos("PanelAlpha", windowName));
        }

        if (!paused) {
            if (!cap.read(frame)) break;
            processedFrame = true;
            frameCount++;
            if (config.verbose) {
                std::cout << "[INFO] Procesando frame " << frameCount << std::endl;
            }
            cv::resize(frame, frame, cv::Size(1280,720));

            std::vector<Detection> detections;
            std::vector<cv::Mat> embeddings;
            std::vector<Detection> filtered;
            std::vector<cv::Mat> filteredEmb;
            detections = detector.detect(frame);
            filtered.reserve(detections.size());
            filteredEmb.reserve(detections.size());
            for (auto& det : detections) {
                cv::Rect bb = det.bbox & cv::Rect(0, 0, frame.cols, frame.rows);
                if (bb.area() <= 0) continue;
                det.centroid = cv::Point2f(bb.x + bb.width / 2.0f, bb.y + bb.height / 2.0f);
                ReIDResult rid = reid.infer(frame, bb);
                det.color = rid.teamColor;
                filtered.push_back(det);
                filteredEmb.push_back(rid.embedding);
            }
            detections.swap(filtered);
            embeddings.swap(filteredEmb);
            totalDetections += (long long)detections.size();

            std::vector<TacticalPlayer> tacticalPlayers;
            std::vector<int> trackIds;
            std::vector<cv::Rect> trackBoxes;
            std::vector<cv::Scalar> trackColors;
            std::vector<cv::Point2f> trackCenters;
            if (config.trackerType == "deepsort") {
                deepSort.update(detections, embeddings);
                auto tracks = deepSort.getTracks();
                for (const auto& t : tracks) {
                    if (t.missedFrames > 0) continue;
                    TacticalPlayer p;
                    p.id = t.id;
                    p.bbox = t.bbox;
                    p.centroid = t.centroid;
                    p.color = t.color;
                    tacticalPlayers.push_back(p);
                    trackIds.push_back(t.id);
                    trackBoxes.push_back(t.bbox);
                    trackColors.push_back(t.color);
                    trackCenters.push_back(t.centroid);
                }
            } else {
                byteTracker.update(detections);
                auto tracks = byteTracker.getTracks();
                for (const auto& t : tracks) {
                    if (t.missedFrames > 0) continue;
                    TacticalPlayer p;
                    p.id = t.id;
                    p.bbox = t.bbox;
                    p.centroid = t.centroid;
                    p.color = t.color;
                    tacticalPlayers.push_back(p);
                    trackIds.push_back(t.id);
                    trackBoxes.push_back(t.bbox);
                    trackColors.push_back(t.color);
                    trackCenters.push_back(t.centroid);
                }
            }

            renderFrame = frame.clone();
            std::vector<TacticalTrackedPlayer> fieldPlayers;
            std::unordered_map<int, cv::Point2f> imagePointById;
            std::vector<TacticalPlayer> visibleTacticalPlayers;
            std::vector<int> visibleTrackIds;
            std::vector<cv::Rect> visibleTrackBoxes;
            std::vector<cv::Scalar> visibleTrackColors;
            std::vector<cv::Point2f> visibleTrackCenters;
            for (size_t i = 0; i < trackIds.size(); ++i) {
                int label = 2;
                if (!teamCnn.empty()) {
                    CropPrediction teamPred = teamCnn.predict(frame, trackBoxes[i]);
                    if (teamPred.classId >= 0 && teamPred.confidence >= config.teamCnnMinConfidence) {
                        label = teamPred.classId;
                    }
                } else {
                    label = teamClassifier.classify(frame, trackBoxes[i], trackIds[i], trackColors[i]);
                }
                int smoothLabel = !teamCnn.empty()
                    ? smoothThreeClassLabel(teamHistory[trackIds[i]], label, maxTeamHist)
                    : smoothTeamLabel(teamHistory[trackIds[i]], label, maxTeamHist);
                trackColors[i] = teamClassifier.colorForTeam(smoothLabel);
                tacticalPlayers[i].color = trackColors[i];
                if (config.dropUnknown && smoothLabel == 2) {
                    continue;
                }
                if (config.dropUnknown) {
                    visibleTacticalPlayers.push_back(tacticalPlayers[i]);
                    visibleTrackIds.push_back(trackIds[i]);
                    visibleTrackBoxes.push_back(trackBoxes[i]);
                    visibleTrackColors.push_back(trackColors[i]);
                    visibleTrackCenters.push_back(trackCenters[i]);
                }

                TacticalTrackedPlayer fp;
                fp.id = trackIds[i];
                fp.teamId = smoothLabel;
                fp.bbox = trackBoxes[i];
                fp.imagePos = cv::Point2f(trackBoxes[i].x + trackBoxes[i].width * 0.5f,
                                          (float)(trackBoxes[i].y + trackBoxes[i].height));
                imagePointById[fp.id] = fp.imagePos;
                if (fieldHomography.isReady()) {
                    fp.fieldPos = fieldHomography.project(fp.imagePos);
                    fp.hasField = fp.fieldPos.x >= -5.0f && fp.fieldPos.x <= 110.0f &&
                                  fp.fieldPos.y >= -5.0f && fp.fieldPos.y <= 73.0f;
                    if (fp.hasField) {
                        auto prev = lastFieldPos.find(fp.id);
                        if (prev != lastFieldPos.end()) {
                            fp.vx = (float)((fp.fieldPos.x - prev->second.x) * fps);
                            fp.vy = (float)((fp.fieldPos.y - prev->second.y) * fps);
                            fp.speed = std::sqrt(fp.vx * fp.vx + fp.vy * fp.vy);
                        }
                        lastFieldPos[fp.id] = fp.fieldPos;
                    }
                }
                fieldPlayers.push_back(fp);
            }
            if (config.dropUnknown) {
                tacticalPlayers.swap(visibleTacticalPlayers);
                trackIds.swap(visibleTrackIds);
                trackBoxes.swap(visibleTrackBoxes);
                trackColors.swap(visibleTrackColors);
                trackCenters.swap(visibleTrackCenters);
            }
            totalTracks += (long long)trackIds.size();
            maxTracks = std::max(maxTracks, (int)trackIds.size());
            if (config.tacticalEnabled) {
                tactical.update(tacticalPlayers, renderFrame.size());
            }
            std::vector<PassLane> passLanes;
            std::vector<std::string> recommendations;
            if (config.tacticalEnabled && fieldHomography.isReady()) {
                heatmaps.update(fieldPlayers);
                auto formations = formationAnalyzer.analyze(fieldPlayers);
                passLanes = passingAnalyzer.analyze(fieldPlayers);
                SpaceControlSummary controlSummary;
                cv::Mat controlMap = spaceAnalyzer.render(fieldPlayers, cv::Size(400, 260), controlSummary);
                recommendations = strategyAdvisor.advise(fieldPlayers, formations, passLanes, controlSummary);

                for (const auto& p : fieldPlayers) {
                    if (exporter.isOpen()) exporter.write(frameCount, frameCount / fps, p);
                }

                for (const auto& lane : passLanes) {
                    auto a = imagePointById.find(lane.fromId);
                    auto b = imagePointById.find(lane.toId);
                    if (a == imagePointById.end() || b == imagePointById.end()) continue;
                    cv::Scalar laneColor = lane.open ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 220);
                    cv::line(renderFrame, a->second, b->second, laneColor, 1);
                }

                cv::Mat mini = tacticalMinimap.draw(fieldPlayers, passLanes,
                                                    heatmaps.renderTeam(0, cv::Size(400, 260)),
                                                    heatmaps.renderTeam(1, cv::Size(400, 260)),
                                                    controlMap, recommendations);
                int miniW = std::min(mini.cols, (int)std::round(renderFrame.cols * 0.45));
                int miniH = mini.rows * miniW / mini.cols;
                cv::resize(mini, mini, cv::Size(miniW, miniH));
                cv::Rect dst(renderFrame.cols - mini.cols - 15, renderFrame.rows - mini.rows - 15,
                             mini.cols, mini.rows);
                cv::Mat roi = renderFrame(dst);
                cv::addWeighted(mini, 0.92, roi, 0.08, 0, roi);
            }
            if (config.tacticalEnabled && ui.showTeams) {
                tactical.draw(renderFrame, tacticalPlayers);
            }

            // Update overlays every few frames to save time
            if (frameCount - overlay.lastUpdateFrame >= 5) {
                detectLinesGoalsBall(renderFrame, overlay);
                overlay.lastUpdateFrame = frameCount;
            }
            // Draw lines
            for (const auto& l : overlay.lines) {
                cv::line(renderFrame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
                         cv::Scalar(255, 255, 0), 1);
            }
            // Draw goals
            if (overlay.goalLeft.area() > 0) {
                cv::rectangle(renderFrame, overlay.goalLeft, cv::Scalar(0, 255, 255), 2);
            }
            if (overlay.goalRight.area() > 0) {
                cv::rectangle(renderFrame, overlay.goalRight, cv::Scalar(0, 255, 255), 2);
            }
            // Draw ball
            if (overlay.ballCenter.has_value()) {
                cv::circle(renderFrame, overlay.ballCenter.value(), std::max(2, overlay.ballRadius),
                           cv::Scalar(0, 255, 255), 2);
                cv::circle(renderFrame, overlay.ballCenter.value(), 2, cv::Scalar(0, 0, 0), -1);
            }

            for (size_t i = 0; i < trackIds.size(); ++i) {
                int id = trackIds[i];
                const auto& bb = trackBoxes[i];
                const auto& color = trackColors[i];
                const auto& c = trackCenters[i];

                if (ui.showTrails) {
                    auto& q = trails[id];
                    q.push_back(cv::Point((int)c.x, (int)c.y));
                    if (q.size() > maxTrail) q.pop_front();
                    drawTrail(renderFrame, q, color);
                }
                if (ui.showBoxes) {
                    cv::Rect drawBox = shrinkRect(bb, 0.78f) & cv::Rect(0, 0, renderFrame.cols, renderFrame.rows);
                    cv::Rect padded = expandRect(drawBox, 1.05f, renderFrame.size());
                    drawBoxPretty(renderFrame, padded, color, ui.lineThickness);
                }
                if (ui.showIds) {
                    double fs = ui.fontScale10 / 10.0;
                    cv::Rect drawBox = shrinkRect(bb, 0.78f) & cv::Rect(0, 0, renderFrame.cols, renderFrame.rows);
                    cv::Point org = drawBox.tl();
                    std::string text = std::to_string(id);
                    int baseline = 0;
                    cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fs, 2, &baseline);
                    cv::Rect bg(org.x, std::max(0, org.y - ts.height - 4), ts.width + 6, ts.height + 6);
                    bg = bg & cv::Rect(0, 0, renderFrame.cols, renderFrame.rows);
                    cv::rectangle(renderFrame, bg, cv::Scalar(0, 0, 0), -1);
                    cv::putText(renderFrame, text, cv::Point(bg.x + 3, bg.y + ts.height + 1),
                                cv::FONT_HERSHEY_SIMPLEX, fs, color, 2);
                }
                if (ui.showCentroids) {
                    cv::circle(renderFrame, c, 3, color, -1);
                }
            }

            drawPanel(renderFrame, ui, config.detectorType, config.trackerType, frameCount, (int)tacticalPlayers.size(),
                      config.tacticalEnabled ? tactical.getCompactnessRed() : 0.0f,
                      config.tacticalEnabled ? tactical.getCompactnessWhite() : 0.0f,
                      config.tacticalEnabled ? tactical.getDangerText() : std::string(),
                      config.tacticalEnabled ? tactical.getAlertsText() : std::string());

            if (!writer.isOpened()) {
                std::cerr << "[ERROR] No se pudo abrir el archivo de salida output.avi." << std::endl;
                break;
            }
            writer.write(renderFrame);
        }

        if (!config.noUi && !renderFrame.empty()) {
            cv::imshow(windowName, renderFrame);
        }
        if (!config.noUi) {
            int key = cv::waitKey(paused ? 0 : 1);
            if (key == 'q' || key == 27) break;
            if (key == ' ') paused = !paused;
            if (key == 'n' && paused) { stepFrames = 1; paused = false; }
            if (key == 'b') ui.showBoxes = !ui.showBoxes;
            if (key == 'i') ui.showIds = !ui.showIds;
            if (key == 'c') ui.showCentroids = !ui.showCentroids;
            if (key == 't') ui.showTeams = !ui.showTeams;
            if (key == 'r') ui.showTrails = !ui.showTrails;
            if (key == 'p') ui.showPanel = !ui.showPanel;
        }
        if (processedFrame && stepFrames > 0) {
            stepFrames--;
            if (stepFrames == 0) paused = true;
        }
        if (config.maxFrames > 0 && frameCount >= config.maxFrames) {
            break;
        }
    }
    auto processingEnd = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(processingEnd - processingStart).count();
    std::cout << "[INFO] Procesamiento finalizado. Frames procesados: " << frameCount << std::endl;
    if (frameCount > 0) {
        std::cout << "[INFO] Detecciones promedio/frame: "
                  << (double)totalDetections / (double)frameCount << std::endl;
        std::cout << "[INFO] Tracks promedio/frame: "
                  << (double)totalTracks / (double)frameCount
                  << "  Max tracks: " << maxTracks << std::endl;
        if (elapsed > 0.0) {
            std::cout << "[INFO] Rendimiento procesamiento: "
                      << (double)frameCount / elapsed << " FPS" << std::endl;
        }
    }
    cap.release();
    writer.release();
    exporter.close();
    std::cout << "[INFO] Archivo output.avi guardado en: " << outputPath.string() << std::endl;
    return 0;
}
