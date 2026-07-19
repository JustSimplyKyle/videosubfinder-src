cmake_minimum_required(VERSION 3.13)

project(VideoSubFinderHeadless LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(wxWidgets REQUIRED COMPONENTS core base)
find_package(TBB REQUIRED)

include(${wxWidgets_USE_FILE})

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
        ${wxWidgets_INCLUDE_DIRS}
)
target_compile_definitions(videosubfinder_headless PRIVATE
    VSF_HEADLESS
    ${wxWidgets_DEFINITIONS}
    $<$<CONFIG:Debug>:${wxWidgets_DEFINITIONS_DEBUG}>
)
target_compile_options(videosubfinder_headless PRIVATE ${wxWidgets_CXX_FLAGS})
target_link_libraries(videosubfinder_headless PUBLIC
    ${OpenCV_LIBS}
    ${wxWidgets_LIBRARIES}
    TBB::tbb
)

install(TARGETS videosubfinder_headless ARCHIVE DESTINATION lib)
install(FILES VideoSubFinderHeadless.h DESTINATION include)
