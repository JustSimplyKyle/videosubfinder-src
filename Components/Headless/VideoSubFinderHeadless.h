#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Stable C ABI for embedding VideoSubFinder's native C++ image processing.
int32_t vsf_headless_api_version(void);

// Runs the original VideoSubFinder frame transform on packed BGR24 pixels.
// Returns 1 when the transform finds a candidate, 0 when it does not, and a
// negative value for invalid input or an internal C++ error.
int32_t vsf_headless_transform_bgr(
    const uint8_t* bgr,
    size_t bgr_len,
    int32_t width,
    int32_t height,
    uint8_t* transformed,
    size_t transformed_len);

typedef int32_t (*vsf_headless_next_frame_fn)(
    void* context,
    uint8_t* bgr,
    size_t bgr_len,
    int64_t* timestamp_ms);

typedef int32_t (*vsf_headless_segment_fn)(
    void* context,
    int64_t start_ms,
    int64_t end_ms,
    const uint8_t* ocr_image,
    size_t ocr_image_len,
    int32_t width,
    int32_t height,
    int32_t channels);

// Parameters consumed by FastSearchSubtitles. Integer booleans must be 0 or 1.
// This is passed by value semantics: the search copies every field before it
// begins pulling frames.
typedef struct vsf_headless_search_params
{
    int32_t threads;
    int32_t min_subtitle_frames;
    double text_percent;
    double min_text_length;
    double vertical_edges_line_error;
    double ila_points_line_error;
    int32_t max_frame_gap_down;
    int32_t max_frame_gap_up;
    int32_t use_isa_images;
    int32_t use_ila_images;
    int32_t replace_isa_with_filtered;
    int32_t apply_ocr_image_cleanup;
} vsf_headless_search_params;

// Runs VideoSubFinder's complete temporal subtitle search. Frames are pulled
// from `next_frame`; detections are either cleaned by FindTextLines or returned
// as temporal BGR frames according to `params`. No GUI or image files are used.
int32_t vsf_headless_search(
    int32_t width,
    int32_t height,
    const vsf_headless_search_params* params,
    void* context,
    vsf_headless_next_frame_fn next_frame,
    vsf_headless_segment_fn segment);

#ifdef __cplusplus
}
#endif
