#include "../../include/analysis/TacticalAnalyzer.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

void TacticalAnalyzer::update(const std::vector<TacticalPlayer>& players, const cv::Size& frameSize) {
    cv::Point2f sumRed(0, 0);
    cv::Point2f sumWhite(0, 0);
    countRed = 0;
    countWhite = 0;
    compactRed = 0.0f;
    compactWhite = 0.0f;
    dangerRed = false;
    dangerWhite = false;
    dangerText.clear();
    alertsText.clear();
    dangerZoneRed = cv::Rect();
    dangerZoneWhite = cv::Rect();
    zoneRed.assign(gridRows * gridCols, 0);
    zoneWhite.assign(gridRows * gridCols, 0);

    std::vector<cv::Point2f> redPts;
    std::vector<cv::Point2f> whitePts;
    redPts.reserve(players.size());
    whitePts.reserve(players.size());

    for (const auto& p : players) {
        if (p.color == cv::Scalar(0, 0, 255)) {
            sumRed += p.centroid;
            countRed++;
            redPts.push_back(p.centroid);
        } else if (p.color == cv::Scalar(255, 255, 255)) {
            sumWhite += p.centroid;
            countWhite++;
            whitePts.push_back(p.centroid);
        }

        int gx = std::min(gridCols - 1, std::max(0, (int)(p.centroid.x / (frameSize.width / (float)gridCols))));
        int gy = std::min(gridRows - 1, std::max(0, (int)(p.centroid.y / (frameSize.height / (float)gridRows))));
        int idx = gy * gridCols + gx;
        if (p.color == cv::Scalar(0, 0, 255)) zoneRed[idx] += 1;
        else if (p.color == cv::Scalar(255, 255, 255)) zoneWhite[idx] += 1;
    }

    teamCenterRed = (countRed > 0) ? (sumRed * (1.0f / countRed)) : cv::Point2f(-1, -1);
    teamCenterWhite = (countWhite > 0) ? (sumWhite * (1.0f / countWhite)) : cv::Point2f(-1, -1);

    if (teamCenterRed.x < 0 || teamCenterRed.y < 0) {
        teamCenterRed = cv::Point2f(frameSize.width * 0.25f, frameSize.height * 0.5f);
    }
    if (teamCenterWhite.x < 0 || teamCenterWhite.y < 0) {
        teamCenterWhite = cv::Point2f(frameSize.width * 0.75f, frameSize.height * 0.5f);
    }

    auto computeCompactness = [&](const std::vector<cv::Point2f>& pts, const cv::Point2f& center) {
        if (pts.empty()) return 0.0f;
        float meanDist = 0.0f;
        for (const auto& p : pts) {
            meanDist += static_cast<float>(cv::norm(p - center));
        }
        meanDist /= (float)pts.size();
        float diag = std::sqrt((float)frameSize.width * frameSize.width + (float)frameSize.height * frameSize.height);
        float norm = std::min(1.0f, meanDist / (0.35f * diag));
        return 1.0f - norm;
    };
    compactRed = computeCompactness(redPts, teamCenterRed);
    compactWhite = computeCompactness(whitePts, teamCenterWhite);

    auto pushHistory = [&](std::deque<cv::Point2f>& h, const cv::Point2f& c) {
        h.push_back(c);
        if ((int)h.size() > historyMax) h.pop_front();
    };
    if (countRed > 0) pushHistory(historyRed, teamCenterRed);
    if (countWhite > 0) pushHistory(historyWhite, teamCenterWhite);

    auto computeDanger = [&](const std::vector<cv::Point2f>& pts, const std::deque<cv::Point2f>& hist,
                             bool& dangerFlag, cv::Rect& dangerZone, const cv::Scalar& color, const std::string& name) {
        if (pts.empty() || hist.size() < 2) return;
        float dx = hist.back().x - hist.front().x;
        float moveThr = frameSize.width * 0.02f;
        float thirdW = frameSize.width / 3.0f;
        int leftCount = 0;
        int rightCount = 0;
        for (const auto& p : pts) {
            if (p.x < thirdW) leftCount++;
            else if (p.x > 2.0f * thirdW) rightCount++;
        }
        if (dx > moveThr && rightCount >= 3) {
            dangerFlag = true;
            dangerZone = cv::Rect((int)(2.0f * thirdW), 0, (int)thirdW, frameSize.height);
        } else if (dx < -moveThr && leftCount >= 3) {
            dangerFlag = true;
            dangerZone = cv::Rect(0, 0, (int)thirdW, frameSize.height);
        }
        if (dangerFlag) {
            if (!dangerText.empty()) dangerText += " | ";
            dangerText += "Peligro " + name + " (densidad + avance)";
        }
    };

    computeDanger(redPts, historyRed, dangerRed, dangerZoneRed, cv::Scalar(0, 0, 255), "Rojo");
    computeDanger(whitePts, historyWhite, dangerWhite, dangerZoneWhite, cv::Scalar(255, 255, 255), "Blanco");

    auto addAlert = [&](const std::string& s) {
        if (alertsText.empty()) alertsText = s;
        else alertsText += " | " + s;
    };

    auto countThirds = [&](const std::vector<cv::Point2f>& pts, int& left, int& mid, int& right) {
        left = mid = right = 0;
        float thirdW = frameSize.width / 3.0f;
        for (const auto& p : pts) {
            if (p.x < thirdW) left++;
            else if (p.x > 2.0f * thirdW) right++;
            else mid++;
        }
    };

    int lR=0, mR=0, rR=0, lW=0, mW=0, rW=0;
    countThirds(redPts, lR, mR, rR);
    countThirds(whitePts, lW, mW, rW);

    if (countRed >= 5 && compactRed > 0.70f) addAlert("Rojo: bloque compacto");
    if (countWhite >= 5 && compactWhite > 0.70f) addAlert("Blanco: bloque compacto");
    if (countRed >= 5 && compactRed < 0.35f) addAlert("Rojo: equipo estirado");
    if (countWhite >= 5 && compactWhite < 0.35f) addAlert("Blanco: equipo estirado");
    if (mR >= 5) addAlert("Rojo: congestion central");
    if (mW >= 5) addAlert("Blanco: congestion central");
    if (lR >= 4) addAlert("Rojo: sobrecarga izquierda");
    if (rR >= 4) addAlert("Rojo: sobrecarga derecha");
    if (lW >= 4) addAlert("Blanco: sobrecarga izquierda");
    if (rW >= 4) addAlert("Blanco: sobrecarga derecha");
}

void TacticalAnalyzer::draw(cv::Mat& frame, const std::vector<TacticalPlayer>& players) const {
    cv::circle(frame, teamCenterRed, 8, cv::Scalar(0, 0, 0), -1);
    cv::circle(frame, teamCenterRed, 6, cv::Scalar(0, 0, 255), -1);
    cv::circle(frame, teamCenterWhite, 8, cv::Scalar(0, 0, 0), -1);
    cv::circle(frame, teamCenterWhite, 6, cv::Scalar(255, 255, 255), -1);
    cv::line(frame, teamCenterRed, teamCenterWhite, cv::Scalar(0, 210, 255), 1);

    auto drawDanger = [&](bool dangerFlag, const cv::Rect& zone, const cv::Scalar& c) {
        if (!dangerFlag || zone.area() <= 0) return;
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, zone, c, -1);
        cv::addWeighted(overlay, 0.15, frame, 0.85, 0, frame);
    };
    drawDanger(dangerRed, dangerZoneRed, cv::Scalar(0, 0, 255));
    drawDanger(dangerWhite, dangerZoneWhite, cv::Scalar(255, 255, 255));

    // Mapa compacto de ocupacion (3x3) en esquina superior derecha.
    int mapW = 220;
    int mapH = 128;
    int x0 = frame.cols - mapW - 20;
    int y0 = 20;
    cv::Rect mapRect(x0, y0, mapW, mapH);
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, mapRect, cv::Scalar(14, 24, 24), -1);
    cv::rectangle(overlay, mapRect, cv::Scalar(60, 95, 85), 1);
    cv::addWeighted(overlay, 0.64, frame, 0.36, 0, frame);

    int cellW = mapW / gridCols;
    int cellH = mapH / gridRows;
    for (int r = 0; r < gridRows; ++r) {
        for (int c = 0; c < gridCols; ++c) {
            int idx = r * gridCols + c;
            int red = (idx < (int)zoneRed.size()) ? zoneRed[idx] : 0;
            int white = (idx < (int)zoneWhite.size()) ? zoneWhite[idx] : 0;
            float rI = std::min(1.0f, red / 3.0f);
            float wI = std::min(1.0f, white / 3.0f);
            cv::Rect cell(x0 + c * cellW, y0 + r * cellH, cellW, cellH);
            if (rI > 0.0f) {
                cv::Mat o = frame.clone();
                cv::rectangle(o, cell, cv::Scalar(0, 0, 255), -1);
                cv::addWeighted(o, 0.15 + 0.25 * rI, frame, 0.85 - 0.25 * rI, 0, frame);
            }
            if (wI > 0.0f) {
                cv::Mat o = frame.clone();
                cv::rectangle(o, cell, cv::Scalar(255, 255, 255), -1);
                cv::addWeighted(o, 0.12 + 0.20 * wI, frame, 0.88 - 0.20 * wI, 0, frame);
            }
            cv::rectangle(frame, cell, cv::Scalar(80, 95, 90), 1);
        }
    }
    cv::putText(frame, "Ocupacion", cv::Point(x0 + 8, y0 + 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(210, 235, 225), 1);

    for (const auto& p : players) {
        cv::circle(frame, p.centroid, 3, p.color, -1);
    }
}
