#include "../../include/analysis/AdvancedTactics.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace {
constexpr float kFieldW = 105.0f;
constexpr float kFieldH = 68.0f;

cv::Vec3f hsvFeature(const cv::Vec3f& hsv) {
    float h = hsv[0] * 2.0f * 3.14159265f / 180.0f;
    float s = hsv[1] / 255.0f;
    return cv::Vec3f(std::cos(h), std::sin(h), s);
}

float pointSegmentDistance(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
    cv::Point2f ab = b - a;
    float denom = ab.dot(ab);
    if (denom <= 1e-4f) return (float)cv::norm(p - a);
    float t = std::max(0.0f, std::min(1.0f, (p - a).dot(ab) / denom));
    cv::Point2f proj = a + ab * t;
    return (float)cv::norm(p - proj);
}

std::string teamName(int teamId) {
    if (teamId == 0) return "Rojo";
    if (teamId == 1) return "Blanco";
    return "Equipo";
}

struct CalibrationState {
    cv::Mat base;
    std::vector<cv::Point2f> points;
};

void onHomographyMouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* state = static_cast<CalibrationState*>(userdata);
    if (!state || state->points.size() >= 4) return;
    state->points.emplace_back((float)x, (float)y);
}
}

bool FieldHomography::load(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["homography"] >> H;
    ready = !H.empty() && H.rows == 3 && H.cols == 3;
    return ready;
}

bool FieldHomography::save(const std::string& path) const {
    if (!ready) return false;
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "field_width_m" << kFieldW;
    fs << "field_height_m" << kFieldH;
    fs << "homography" << H;
    return true;
}

bool FieldHomography::calibrateFromFrame(const cv::Mat& frame, const std::string& windowName) {
    if (frame.empty()) return false;
    CalibrationState state;
    state.base = frame.clone();
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::setMouseCallback(windowName, onHomographyMouse, &state);
    while (state.points.size() < 4) {
        cv::Mat view = state.base.clone();
        for (size_t i = 0; i < state.points.size(); ++i) {
            cv::circle(view, state.points[i], 6, cv::Scalar(0, 255, 255), -1);
            cv::putText(view, std::to_string((int)i + 1), state.points[i] + cv::Point2f(8, -8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        }
        cv::putText(view, "Click 4 esquinas: sup-izq, sup-der, inf-der, inf-izq",
                    cv::Point(25, 35), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2);
        cv::imshow(windowName, view);
        int key = cv::waitKey(20);
        if (key == 27 || key == 'q') {
            cv::setMouseCallback(windowName, nullptr, nullptr);
            cv::destroyWindow(windowName);
            return false;
        }
    }
    cv::setMouseCallback(windowName, nullptr, nullptr);
    cv::destroyWindow(windowName);

    std::vector<cv::Point2f> field = {
        {0.0f, 0.0f}, {kFieldW, 0.0f}, {kFieldW, kFieldH}, {0.0f, kFieldH}
    };
    H = cv::findHomography(state.points, field);
    ready = !H.empty();
    return ready;
}

cv::Point2f FieldHomography::project(const cv::Point2f& imagePoint) const {
    if (!ready) return cv::Point2f(-1.0f, -1.0f);
    std::vector<cv::Point2f> src{imagePoint};
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, H);
    return dst.empty() ? cv::Point2f(-1.0f, -1.0f) : dst[0];
}

std::vector<cv::Point2f> FieldHomography::project(const std::vector<cv::Point2f>& imagePoints) const {
    std::vector<cv::Point2f> dst;
    if (!ready || imagePoints.empty()) return dst;
    cv::perspectiveTransform(imagePoints, dst, H);
    return dst;
}

void TeamColorClassifier::configure(bool enabledValue, int warmupValue, int minSamplesValue) {
    enabled = enabledValue;
    warmupFrames = std::max(5, warmupValue);
    minSamples = std::max(10, minSamplesValue);
}

cv::Vec3f TeamColorClassifier::extractTorsoHsv(const cv::Mat& frame, const cv::Rect& bbox, bool& valid) const {
    valid = false;
    cv::Rect box = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (box.width < 4 || box.height < 8) return cv::Vec3f();
    cv::Rect torso(box.x + box.width / 4, box.y + box.height / 5,
                   std::max(2, box.width / 2), std::max(4, box.height / 3));
    torso &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (torso.empty()) return cv::Vec3f();

    cv::Mat hsv;
    cv::cvtColor(frame(torso), hsv, cv::COLOR_BGR2HSV);
    cv::Scalar mean = cv::mean(hsv);
    valid = true;
    return cv::Vec3f((float)mean[0], (float)mean[1], (float)mean[2]);
}

bool TeamColorClassifier::train() {
    if (samples.size() < (size_t)minSamples) return false;
    cv::Mat data((int)samples.size(), 3, CV_32F);
    for (int i = 0; i < data.rows; ++i) {
        cv::Vec3f f = hsvFeature(samples[i]);
        data.at<float>(i, 0) = f[0];
        data.at<float>(i, 1) = f[1];
        data.at<float>(i, 2) = f[2];
    }
    cv::Mat labels, centers;
    cv::kmeans(data, 2, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 60, 0.01),
               3, cv::KMEANS_PP_CENTERS, centers);
    c1 = cv::Vec3f(centers.at<float>(0, 0), centers.at<float>(0, 1), centers.at<float>(0, 2));
    c2 = cv::Vec3f(centers.at<float>(1, 0), centers.at<float>(1, 1), centers.at<float>(1, 2));
    trained = true;
    return true;
}

int TeamColorClassifier::vote(int trackId, int rawTeam) {
    if (rawTeam < 0 || rawTeam > 1) return 2;
    auto& h = votesByTrack[trackId];
    h.push_back(rawTeam);
    if (h.size() > 10) h.pop_front();
    int a = 0;
    int b = 0;
    for (int v : h) {
        if (v == 0) a++;
        else if (v == 1) b++;
    }
    return (a >= b) ? 0 : 1;
}

int TeamColorClassifier::classify(const cv::Mat& frame, const cv::Rect& bbox, int trackId, const cv::Scalar& fallbackColor) {
    int fallback = 2;
    if (fallbackColor == cv::Scalar(0, 0, 255)) fallback = 0;
    else if (fallbackColor == cv::Scalar(255, 255, 255)) fallback = 1;
    if (!enabled) return fallback;

    bool valid = false;
    cv::Vec3f hsv = extractTorsoHsv(frame, bbox, valid);
    if (!valid) return vote(trackId, fallback);
    if (!trained) {
        samples.push_back(hsv);
        framesSeen++;
        if (framesSeen >= warmupFrames) train();
        return vote(trackId, fallback);
    }

    cv::Vec3f f = hsvFeature(hsv);
    auto dist = [](const cv::Vec3f& a, const cv::Vec3f& b) {
        cv::Vec3f d = a - b;
        return d.dot(d);
    };
    int raw = (dist(f, c1) <= dist(f, c2)) ? 0 : 1;
    return vote(trackId, raw);
}

cv::Scalar TeamColorClassifier::colorForTeam(int teamId) const {
    if (teamId == 0) return cv::Scalar(0, 0, 255);
    if (teamId == 1) return cv::Scalar(255, 255, 255);
    return cv::Scalar(0, 255, 255);
}

bool TrackingDataExporter::open(const std::string& path) {
    file.open(path, std::ios::out | std::ios::trunc);
    if (!file.is_open()) return false;
    file << "frame,timestamp,player_id,team_id,x_image,y_image,x_field,y_field,speed,vx,vy\n";
    return true;
}

void TrackingDataExporter::write(int frame, double timestamp, const TacticalTrackedPlayer& p) {
    if (!file.is_open()) return;
    file << frame << "," << timestamp << "," << p.id << "," << p.teamId << ","
         << p.imagePos.x << "," << p.imagePos.y << ",";
    if (p.hasField) file << p.fieldPos.x << "," << p.fieldPos.y;
    else file << ",";
    file << "," << p.speed << "," << p.vx << "," << p.vy << "\n";
}

void TrackingDataExporter::close() {
    if (file.is_open()) file.close();
}

HeatmapAnalyzer::HeatmapAnalyzer(int colsValue, int rowsValue)
    : cols(colsValue), rows(rowsValue),
      team0(cv::Mat::zeros(rowsValue, colsValue, CV_32F)),
      team1(cv::Mat::zeros(rowsValue, colsValue, CV_32F)) {}

void HeatmapAnalyzer::update(const std::vector<TacticalTrackedPlayer>& players) {
    for (const auto& p : players) {
        if (!p.hasField) continue;
        int x = std::max(0, std::min(cols - 1, (int)std::round(p.fieldPos.x / kFieldW * (cols - 1))));
        int y = std::max(0, std::min(rows - 1, (int)std::round(p.fieldPos.y / kFieldH * (rows - 1))));
        if (p.teamId == 0) team0.at<float>(y, x) += 1.0f;
        else if (p.teamId == 1) team1.at<float>(y, x) += 1.0f;
        auto& m = playersHeat[p.id];
        if (m.empty()) m = cv::Mat::zeros(rows, cols, CV_32F);
        m.at<float>(y, x) += 1.0f;
    }
}

cv::Mat HeatmapAnalyzer::renderTeam(int teamId, const cv::Size& size) const {
    cv::Mat src = (teamId == 0) ? team0 : team1;
    cv::Mat blur, norm, color, resized;
    cv::GaussianBlur(src, blur, cv::Size(0, 0), 2.2);
    cv::normalize(blur, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(norm, color, cv::COLORMAP_JET);
    cv::resize(color, resized, size);
    return resized;
}

std::vector<TeamFormationMetrics> FormationAnalyzer::analyze(const std::vector<TacticalTrackedPlayer>& players) const {
    std::vector<TeamFormationMetrics> out;
    for (int team = 0; team <= 1; ++team) {
        std::vector<cv::Point2f> pts;
        for (const auto& p : players) {
            if (p.teamId == team && p.hasField) pts.push_back(p.fieldPos);
        }
        TeamFormationMetrics m;
        m.teamId = team;
        m.count = (int)pts.size();
        if (pts.empty()) {
            out.push_back(m);
            continue;
        }
        float minX = pts[0].x, maxX = pts[0].x, minY = pts[0].y, maxY = pts[0].y;
        for (const auto& p : pts) {
            minX = std::min(minX, p.x);
            maxX = std::max(maxX, p.x);
            minY = std::min(minY, p.y);
            maxY = std::max(maxY, p.y);
        }
        m.depth = maxX - minX;
        m.width = maxY - minY;
        if (pts.size() >= 3) {
            std::vector<cv::Point2f> hull;
            cv::convexHull(pts, hull);
            m.hullArea = (float)cv::contourArea(hull);
        }
        m.compactness = (m.width * m.depth > 1.0f) ? std::min(1.0f, m.hullArea / (m.width * m.depth)) : 0.0f;
        std::sort(pts.begin(), pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
        m.defensiveLine = pts[std::max(0, (int)pts.size() / 3 - 1)].x;
        m.midfieldLine = pts[pts.size() / 2].x;
        m.offensiveLine = pts[std::min((int)pts.size() - 1, (int)(pts.size() * 2 / 3))].x;
        m.tooLong = m.depth > 58.0f;
        m.tooNarrow = m.width < 24.0f && m.count >= 5;
        m.disordered = m.compactness < 0.18f && m.count >= 5;
        out.push_back(m);
    }
    return out;
}

std::vector<PassLane> PassingLaneAnalyzer::analyze(const std::vector<TacticalTrackedPlayer>& players) const {
    std::vector<PassLane> lanes;
    for (const auto& a : players) {
        if (!a.hasField || a.teamId > 1) continue;
        for (const auto& b : players) {
            if (a.id >= b.id || !b.hasField || b.teamId != a.teamId) continue;
            float dist = (float)cv::norm(a.fieldPos - b.fieldPos);
            if (dist < 4.0f || dist > 38.0f) continue;
            float nearest = 999.0f;
            for (const auto& r : players) {
                if (!r.hasField || r.teamId == a.teamId || r.teamId > 1) continue;
                nearest = std::min(nearest, pointSegmentDistance(r.fieldPos, a.fieldPos, b.fieldPos));
            }
            PassLane lane;
            lane.fromId = a.id;
            lane.toId = b.id;
            lane.teamId = a.teamId;
            lane.nearestOpponentMeters = nearest;
            lane.open = nearest > 3.0f;
            lanes.push_back(lane);
        }
    }
    return lanes;
}

cv::Mat SpaceControlAnalyzer::render(const std::vector<TacticalTrackedPlayer>& players, const cv::Size& size,
                                     SpaceControlSummary& summary) const {
    summary = SpaceControlSummary();
    cv::Mat img(size, CV_8UC3, cv::Scalar(35, 95, 45));
    int cols = 24;
    int rows = 16;
    float cw = size.width / (float)cols;
    float ch = size.height / (float)rows;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            cv::Point2f field((x + 0.5f) / cols * kFieldW, (y + 0.5f) / rows * kFieldH);
            float best0 = 1e6f, best1 = 1e6f;
            for (const auto& p : players) {
                if (!p.hasField || p.teamId > 1) continue;
                float projectedSpeed = std::min(8.0f, p.speed);
                float value = (float)cv::norm(field - p.fieldPos) / (1.0f + projectedSpeed * 0.08f);
                if (p.teamId == 0) best0 = std::min(best0, value);
                else if (p.teamId == 1) best1 = std::min(best1, value);
            }
            int owner = (best0 <= best1) ? 0 : 1;
            cv::Scalar color = owner == 0 ? cv::Scalar(35, 55, 170) : cv::Scalar(190, 190, 190);
            cv::Rect cell((int)std::round(x * cw), (int)std::round(y * ch),
                          (int)std::ceil(cw), (int)std::ceil(ch));
            cv::rectangle(img, cell, color, -1);
            bool central = x >= cols / 3 && x < 2 * cols / 3;
            bool left = y < rows / 3;
            bool right = y >= 2 * rows / 3;
            if (central && owner == 0) summary.centralTeam0++;
            if (central && owner == 1) summary.centralTeam1++;
            if (left && owner == 0) summary.leftTeam0++;
            if (left && owner == 1) summary.leftTeam1++;
            if (right && owner == 0) summary.rightTeam0++;
            if (right && owner == 1) summary.rightTeam1++;
        }
    }
    return img;
}

std::vector<std::string> StrategyAdvisor::advise(const std::vector<TacticalTrackedPlayer>& players,
                                                 const std::vector<TeamFormationMetrics>& formations,
                                                 const std::vector<PassLane>& lanes,
                                                 const SpaceControlSummary& control) const {
    std::vector<std::string> out;
    for (const auto& f : formations) {
        if (f.count < 3) continue;
        std::string name = teamName(f.teamId);
        if (f.tooLong) out.push_back(name + ": reducir distancia entre lineas");
        if (f.tooNarrow) out.push_back(name + ": abrir amplitud antes de progresar");
        if (f.width > 48.0f && f.depth < 32.0f) out.push_back(name + ": ataque vertical disponible");
        if (f.disordered) out.push_back(name + ": reorganizar alturas del bloque");
    }

    std::unordered_map<int, int> openByPlayer;
    for (const auto& l : lanes) {
        if (!l.open) continue;
        openByPlayer[l.fromId]++;
        openByPlayer[l.toId]++;
    }
    for (const auto& [id, count] : openByPlayer) {
        if (count >= 3) out.push_back("Jugador " + std::to_string(id) + ": opcion de creacion");
    }

    if (control.centralTeam0 > control.centralTeam1 * 1.25f) out.push_back("Blanco: riesgo defensivo en carril central");
    if (control.centralTeam1 > control.centralTeam0 * 1.25f) out.push_back("Rojo: riesgo defensivo en carril central");
    if (control.leftTeam0 >= control.leftTeam1 + 8) out.push_back("Rojo: superioridad por banda superior");
    if (control.leftTeam1 >= control.leftTeam0 + 8) out.push_back("Blanco: superioridad por banda superior");
    if (control.rightTeam0 >= control.rightTeam1 + 8) out.push_back("Rojo: superioridad por banda inferior");
    if (control.rightTeam1 >= control.rightTeam0 + 8) out.push_back("Blanco: superioridad por banda inferior");

    int open0 = 0, open1 = 0;
    for (const auto& l : lanes) {
        if (l.open && l.teamId == 0) open0++;
        if (l.open && l.teamId == 1) open1++;
    }
    if (open0 >= 4) out.push_back("Rojo: progresar usando linea de pase abierta");
    if (open1 >= 4) out.push_back("Blanco: progresar usando linea de pase abierta");
    if (open0 <= 1) out.push_back("Presionar a Rojo: pocas opciones de pase");
    if (open1 <= 1) out.push_back("Presionar a Blanco: pocas opciones de pase");

    if (out.size() > 6) out.resize(6);
    return out;
}

cv::Point TacticalMinimap::mapPoint(const cv::Point2f& fieldPos, const cv::Rect& fieldRect) const {
    int x = fieldRect.x + (int)std::round(fieldPos.x / kFieldW * fieldRect.width);
    int y = fieldRect.y + (int)std::round(fieldPos.y / kFieldH * fieldRect.height);
    return cv::Point(x, y);
}

void TacticalMinimap::drawPitch(cv::Mat& img, const cv::Rect& r) const {
    cv::rectangle(img, r, cv::Scalar(235, 235, 235), 2);
    cv::line(img, cv::Point(r.x + r.width / 2, r.y), cv::Point(r.x + r.width / 2, r.y + r.height),
             cv::Scalar(210, 210, 210), 1);
    cv::circle(img, cv::Point(r.x + r.width / 2, r.y + r.height / 2), 32, cv::Scalar(210, 210, 210), 1);
    cv::rectangle(img, cv::Rect(r.x, r.y + r.height / 2 - 55, 70, 110), cv::Scalar(210, 210, 210), 1);
    cv::rectangle(img, cv::Rect(r.x + r.width - 70, r.y + r.height / 2 - 55, 70, 110), cv::Scalar(210, 210, 210), 1);
}

cv::Mat TacticalMinimap::draw(const std::vector<TacticalTrackedPlayer>& players,
                              const std::vector<PassLane>& lanes,
                              const cv::Mat& heatTeam0,
                              const cv::Mat& heatTeam1,
                              const cv::Mat& control,
                              const std::vector<std::string>& recommendations,
                              const cv::Point2f* ballFieldPos) const {
    cv::Mat img(360, 620, CV_8UC3, cv::Scalar(12, 28, 26));
    cv::rectangle(img, cv::Rect(0, 0, img.cols, img.rows), cv::Scalar(18, 38, 34), -1);
    cv::Rect fieldRect(18, 46, 410, 266);
    cv::Rect sidePanel(446, 46, 156, 266);
    cv::rectangle(img, sidePanel, cv::Scalar(20, 35, 35), -1);
    cv::rectangle(img, sidePanel, cv::Scalar(75, 115, 105), 1);

    if (!control.empty()) {
        cv::Mat resized;
        cv::resize(control, resized, fieldRect.size());
        resized.copyTo(img(fieldRect));
    }
    if (!heatTeam0.empty()) {
        cv::Mat h0;
        cv::resize(heatTeam0, h0, fieldRect.size());
        cv::addWeighted(h0, 0.22, img(fieldRect), 0.78, 0, img(fieldRect));
    }
    if (!heatTeam1.empty()) {
        cv::Mat h1;
        cv::resize(heatTeam1, h1, fieldRect.size());
        cv::addWeighted(h1, 0.16, img(fieldRect), 0.84, 0, img(fieldRect));
    }
    drawPitch(img, fieldRect);

    for (const auto& p : players) {
        if (!p.hasField) continue;
        auto& trail = fieldTrails[p.id];
        trail.push_back(p.fieldPos);
        if (trail.size() > 24) trail.pop_front();
        cv::Scalar c = p.teamId == 0 ? cv::Scalar(0, 0, 190) : cv::Scalar(220, 220, 220);
        for (size_t i = 1; i < trail.size(); ++i) {
            cv::line(img, mapPoint(trail[i - 1], fieldRect), mapPoint(trail[i], fieldRect), c, 1);
        }
    }

    std::unordered_map<int, TacticalTrackedPlayer> byId;
    for (const auto& p : players) byId[p.id] = p;
    for (const auto& l : lanes) {
        auto a = byId.find(l.fromId);
        auto b = byId.find(l.toId);
        if (a == byId.end() || b == byId.end() || !a->second.hasField || !b->second.hasField) continue;
        cv::Scalar c = l.open ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 0, 220);
        cv::line(img, mapPoint(a->second.fieldPos, fieldRect), mapPoint(b->second.fieldPos, fieldRect), c, l.open ? 2 : 1);
    }

    for (const auto& p : players) {
        if (!p.hasField) continue;
        cv::Point mp = mapPoint(p.fieldPos, fieldRect);
        cv::Scalar c = p.teamId == 0 ? cv::Scalar(0, 0, 255) : (p.teamId == 1 ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 255, 255));
        cv::circle(img, mp, 6, cv::Scalar(0, 0, 0), -1);
        cv::circle(img, mp, 5, c, -1);
        cv::putText(img, std::to_string(p.id), mp + cv::Point(7, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(245, 245, 245), 1);
    }
    if (ballFieldPos) {
        cv::Point bp = mapPoint(*ballFieldPos, fieldRect);
        cv::circle(img, bp, 5, cv::Scalar(0, 255, 255), -1);
        cv::circle(img, bp, 6, cv::Scalar(0, 0, 0), 1);
    }

    cv::putText(img, "MINIMAPA TACTICO", cv::Point(18, 28), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(210, 255, 230), 1);
    cv::putText(img, "STRATEGY ADVISOR", cv::Point(456, 28), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(210, 255, 230), 1);

    int y = 70;
    int idx = 1;
    for (const auto& rec : recommendations) {
        if (idx > 5) break;
        cv::rectangle(img, cv::Rect(456, y - 15, 134, 29), cv::Scalar(26, 50, 46), -1);
        cv::circle(img, cv::Point(466, y - 2), 6, cv::Scalar(0, 210, 255), -1);
        cv::putText(img, std::to_string(idx), cv::Point(463, y + 2), cv::FONT_HERSHEY_SIMPLEX, 0.28,
                    cv::Scalar(0, 0, 0), 1);
        std::string text = rec.size() > 31 ? rec.substr(0, 28) + "..." : rec;
        cv::putText(img, text, cv::Point(478, y + 2), cv::FONT_HERSHEY_SIMPLEX, 0.32,
                    cv::Scalar(225, 245, 240), 1);
        y += 39;
        idx++;
    }
    if (recommendations.empty()) {
        cv::putText(img, "Sin alertas", cv::Point(456, 78), cv::FONT_HERSHEY_SIMPLEX, 0.38,
                    cv::Scalar(150, 170, 165), 1);
    }
    return img;
}
