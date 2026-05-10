#include "app/AppConfig.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int main() {
    AppConfig config;
    std::string error;
    std::vector<const char*> argv = {
        "football_analytics",
        "--input", "match.mp4",
        "--output", "out.avi",
        "--tracker", "deepsort",
        "--team-auto", "1",
        "--drop-unknown", "1",
        "--verbose", "1",
        "--profile", "sensitive",
        "--no-ui", "1",
        "--max-frames", "42",
        "--min-area", "120"
    };

    assert(parseAppConfig((int)argv.size(), const_cast<char**>(argv.data()), config, error));
    assert(error.empty());
    assert(config.inputPath == "match.mp4");
    assert(config.outputPath == "out.avi");
    assert(config.trackerType == "deepsort");
    assert(config.teamAuto);
    assert(config.dropUnknown);
    assert(config.verbose);
    assert(config.profile == "sensitive");
    assert(config.useCanny);
    assert(!config.useColorCandidates);
    assert(config.noUi);
    assert(config.maxFrames == 42);
    assert(config.minArea == 120);

    AppConfig bad;
    std::string badError;
    std::vector<const char*> badArgv = {
        "football_analytics",
        "--tracker", "unknown"
    };
    assert(!parseAppConfig((int)badArgv.size(), const_cast<char**>(badArgv.data()), bad, badError));
    assert(!badError.empty());

    std::cout << "test_app_config OK" << std::endl;
    return 0;
}
