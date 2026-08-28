cmake_minimum_required(VERSION 3.13)

project(VideoSubFinderHeadless LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)

add_library(videosubfinder_headless STATIC
    VideoSubFinderHeadless.cpp
    ../IPAlgorithms/IPAlgorithms.cpp
    ../IPAlgorithms/MyClosedFigure.cpp
    ../IPAlgorithms/SSAlgorithms.cpp
)

target_compile_features(videosubfinder_headless PUBLIC cxx_std_17)
target_include_directories(videosubfinder_headless
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ../Include
        ../IPAlgorithms
        ${OpenCV_INCLUDE_DIRS}
)
target_compile_definitions(videosubfinder_headless PRIVATE
    VSF_HEADLESS
)
target_link_libraries(videosubfinder_headless PUBLIC
    ${OpenCV_LIBS}
    TBB::tbb
)

install(TARGETS videosubfinder_headless ARCHIVE DESTINATION lib)
install(FILES VideoSubFinderHeadless.h DESTINATION include)
