#pragma once
#include <opencv2/opencv.hpp>
#include <deque>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

struct TacticalTrackedPlayer {
    int id = -1;
    int teamId = 2;
    cv::Rect bbox;
    cv::Point2f imagePos;
    cv::Point2f fieldPos;
    bool hasField = false;
    float speed = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
};

struct PassLane {
    int fromId = -1;
    int toId = -1;
    int teamId = 2;
    float nearestOpponentMeters = 999.0f;
    bool open = false;
};

struct TeamFormationMetrics {
    int teamId = 2;
    int count = 0;
    float width = 0.0f;
    float depth = 0.0f;
    float hullArea = 0.0f;
    float compactness = 0.0f;
    float defensiveLine = 0.0f;
    float midfieldLine = 0.0f;
    float offensiveLine = 0.0f;
    bool tooLong = false;
    bool tooNarrow = false;
    bool disordered = false;
};

struct SpaceControlSummary {
    int centralTeam0 = 0;
    int centralTeam1 = 0;
    int leftTeam0 = 0;
    int leftTeam1 = 0;
    int rightTeam0 = 0;
    int rightTeam1 = 0;
};

class FieldHomography {
public:
    bool load(const std::string& path);
    bool save(const std::string& path) const;
    bool isReady() const { return ready; }
    bool calibrateFromFrame(const cv::Mat& frame, const std::string& windowName);
    cv::Point2f project(const cv::Point2f& imagePoint) const;
    std::vector<cv::Point2f> project(const std::vector<cv::Point2f>& imagePoints) const;

private:
    cv::Mat H;
    bool ready = false;
};

class TeamColorClassifier {
public:
    TeamColorClassifier() = default;
    void configure(bool enabled, int warmupFrames, int minSamples);
    int classify(const cv::Mat& frame, const cv::Rect& bbox, int trackId, const cv::Scalar& fallbackColor);
    cv::Scalar colorForTeam(int teamId) const;
    bool isTrained() const { return trained; }

private:
    cv::Vec3f extractTorsoHsv(const cv::Mat& frame, const cv::Rect& bbox, bool& valid) const;
    bool train();
    int vote(int trackId, int rawTeam);

    bool enabled = true;
    bool trained = false;
    int framesSeen = 0;
    int warmupFrames = 60;
    int minSamples = 40;
    std::vector<cv::Vec3f> samples;
    cv::Vec3f c1;
    cv::Vec3f c2;
    std::unordered_map<int, std::deque<int>> votesByTrack;
};

class TrackingDataExporter {
public:
    bool open(const std::string& path);
    void write(int frame, double timestamp, const TacticalTrackedPlayer& p);
    void close();
    bool isOpen() const { return file.is_open(); }

private:
    std::ofstream file;
};

class HeatmapAnalyzer {
public:
    HeatmapAnalyzer(int cols = 105, int rows = 68);
    void update(const std::vector<TacticalTrackedPlayer>& players);
    cv::Mat renderTeam(int teamId, const cv::Size& size) const;

private:
    int cols;
    int rows;
    cv::Mat team0;
    cv::Mat team1;
    std::unordered_map<int, cv::Mat> playersHeat;
};

class FormationAnalyzer {
public:
    std::vector<TeamFormationMetrics> analyze(const std::vector<TacticalTrackedPlayer>& players) const;
};

class PassingLaneAnalyzer {
public:
    std::vector<PassLane> analyze(const std::vector<TacticalTrackedPlayer>& players) const;
};

class SpaceControlAnalyzer {
public:
    cv::Mat render(const std::vector<TacticalTrackedPlayer>& players, const cv::Size& size,
                   SpaceControlSummary& summary) const;
};

class StrategyAdvisor {
public:
    std::vector<std::string> advise(const std::vector<TacticalTrackedPlayer>& players,
                                    const std::vector<TeamFormationMetrics>& formations,
                                    const std::vector<PassLane>& lanes,
                                    const SpaceControlSummary& control) const;
};

class TacticalMinimap {
public:
    cv::Mat draw(const std::vector<TacticalTrackedPlayer>& players,
                 const std::vector<PassLane>& lanes,
                 const cv::Mat& heatTeam0,
                 const cv::Mat& heatTeam1,
                 const cv::Mat& control,
                 const std::vector<std::string>& recommendations,
                 const cv::Point2f* ballFieldPos = nullptr) const;

private:
    cv::Point mapPoint(const cv::Point2f& fieldPos, const cv::Rect& fieldRect) const;
    void drawPitch(cv::Mat& img, const cv::Rect& fieldRect) const;
    mutable std::unordered_map<int, std::deque<cv::Point2f>> fieldTrails;
};
