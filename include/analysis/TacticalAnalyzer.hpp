#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>

struct TacticalPlayer {
    int id = -1;
    cv::Rect bbox;
    cv::Point2f centroid;
    cv::Scalar color;
};

class TacticalAnalyzer {
public:
    TacticalAnalyzer() = default;
    void update(const std::vector<TacticalPlayer>& players, const cv::Size& frameSize);
    void draw(cv::Mat& frame, const std::vector<TacticalPlayer>& players) const;
    float getCompactnessRed() const { return compactRed; }
    float getCompactnessWhite() const { return compactWhite; }
    bool isDangerRed() const { return dangerRed; }
    bool isDangerWhite() const { return dangerWhite; }
    cv::Rect getDangerZoneRed() const { return dangerZoneRed; }
    cv::Rect getDangerZoneWhite() const { return dangerZoneWhite; }
    const std::vector<int>& getZoneRed() const { return zoneRed; }
    const std::vector<int>& getZoneWhite() const { return zoneWhite; }
    std::string getDangerText() const { return dangerText; }
    std::string getAlertsText() const { return alertsText; }
private:
    cv::Point2f teamCenterRed = cv::Point2f(-1, -1);
    cv::Point2f teamCenterWhite = cv::Point2f(-1, -1);
    int countRed = 0;
    int countWhite = 0;
    float compactRed = 0.0f;
    float compactWhite = 0.0f;
    bool dangerRed = false;
    bool dangerWhite = false;
    cv::Rect dangerZoneRed;
    cv::Rect dangerZoneWhite;
    std::string dangerText;
    std::string alertsText;
    std::vector<int> zoneRed;
    std::vector<int> zoneWhite;
    std::deque<cv::Point2f> historyRed;
    std::deque<cv::Point2f> historyWhite;
    int gridRows = 3;
    int gridCols = 3;
    int historyMax = 12;
};
