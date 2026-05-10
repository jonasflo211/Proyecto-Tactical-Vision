#include "analysis/AdvancedTactics.hpp"

#include <cassert>
#include <iostream>
#include <vector>

namespace {

TacticalTrackedPlayer player(int id, int team, float x, float y) {
    TacticalTrackedPlayer p;
    p.id = id;
    p.teamId = team;
    p.hasField = true;
    p.fieldPos = cv::Point2f(x, y);
    return p;
}

} // namespace

int main() {
    std::vector<TacticalTrackedPlayer> players = {
        player(1, 0, 10.0f, 20.0f),
        player(2, 0, 24.0f, 22.0f),
        player(3, 0, 36.0f, 24.0f),
        player(4, 1, 70.0f, 20.0f),
        player(5, 1, 82.0f, 24.0f),
        player(6, 1, 90.0f, 28.0f)
    };

    FormationAnalyzer formationAnalyzer;
    auto formations = formationAnalyzer.analyze(players);
    assert(formations.size() == 2);
    assert(formations[0].count == 3);
    assert(formations[1].count == 3);

    PassingLaneAnalyzer passingAnalyzer;
    auto lanes = passingAnalyzer.analyze(players);
    assert(!lanes.empty());

    SpaceControlSummary summary;
    SpaceControlAnalyzer spaceAnalyzer;
    cv::Mat control = spaceAnalyzer.render(players, cv::Size(120, 80), summary);
    assert(!control.empty());
    assert(control.cols == 120);
    assert(control.rows == 80);

    std::cout << "test_tactical_analysis OK" << std::endl;
    return 0;
}
