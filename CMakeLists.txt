cmake_minimum_required(VERSION 3.16)

project(football_analytics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

function(copy_runtime_dependencies target_name)
    if (WIN32)
        if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:${target_name}>
                    $<TARGET_FILE_DIR:${target_name}>
                COMMAND_EXPAND_LISTS
            )
        endif()

        if (OpenCV_DIR)
            get_filename_component(_opencv_dir_abs "${OpenCV_DIR}" ABSOLUTE)
            set(_opencv_bin_candidates
                "${_opencv_dir_abs}/../bin"
                "${_opencv_dir_abs}/../../bin"
                "${_opencv_dir_abs}/../../../x64/vc16/bin"
                "${_opencv_dir_abs}/../../../bin"
            )
            foreach(_opencv_bin ${_opencv_bin_candidates})
                if (EXISTS "${_opencv_bin}")
                    file(GLOB _opencv_runtime_dlls "${_opencv_bin}/*.dll")
                    if (_opencv_runtime_dlls)
                        add_custom_command(TARGET ${target_name} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                ${_opencv_runtime_dlls}
                                $<TARGET_FILE_DIR:${target_name}>
                        )
                        break()
                    endif()
                endif()
            endforeach()
        endif()
    endif()
endfunction()

set(FOOTBALL_CORE_SOURCES
    src/app/AppConfig.cpp
    src/detection/ClassicDetector.cpp
    src/dnn/CropClassifier.cpp
    src/reid/ReIDEmbedder.cpp
    src/analysis/TacticalAnalyzer.cpp
    src/analysis/AdvancedTactics.cpp
    src/evaluation/DetectionEvaluator.cpp
    src/tracking/ByteTrackTracker.cpp
    src/tracking/DeepSORTTracker.cpp
)

add_library(football_core ${FOOTBALL_CORE_SOURCES})
target_include_directories(football_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(football_core PUBLIC ${OpenCV_LIBS})

if (MSVC)
    target_compile_options(football_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(football_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(football_analytics src/main.cpp)
target_link_libraries(football_analytics PRIVATE football_core)
copy_runtime_dependencies(football_analytics)

add_executable(dataset_cropper tools/DatasetCropper.cpp)
target_include_directories(dataset_cropper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dataset_cropper PRIVATE ${OpenCV_LIBS})
copy_runtime_dependencies(dataset_cropper)

add_executable(evaluate_detector tools/EvaluateDetector.cpp)
target_link_libraries(evaluate_detector PRIVATE football_core)
copy_runtime_dependencies(evaluate_detector)

add_executable(export_eval_frames tools/ExportEvalFrames.cpp)
target_include_directories(export_eval_frames PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(export_eval_frames PRIVATE ${OpenCV_LIBS})
copy_runtime_dependencies(export_eval_frames)

enable_testing()

add_executable(test_app_config tests/test_app_config.cpp)
target_link_libraries(test_app_config PRIVATE football_core)
add_test(NAME app_config COMMAND test_app_config)
copy_runtime_dependencies(test_app_config)

add_executable(test_bytetrack tests/test_bytetrack.cpp)
target_link_libraries(test_bytetrack PRIVATE football_core)
add_test(NAME bytetrack COMMAND test_bytetrack)
copy_runtime_dependencies(test_bytetrack)

add_executable(test_tactical_analysis tests/test_tactical_analysis.cpp)
target_link_libraries(test_tactical_analysis PRIVATE football_core)
add_test(NAME tactical_analysis COMMAND test_tactical_analysis)
copy_runtime_dependencies(test_tactical_analysis)

add_executable(test_detection_evaluator tests/test_detection_evaluator.cpp)
target_link_libraries(test_detection_evaluator PRIVATE football_core)
add_test(NAME detection_evaluator COMMAND test_detection_evaluator)
copy_runtime_dependencies(test_detection_evaluator)
