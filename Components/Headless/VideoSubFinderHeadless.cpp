#include "VideoSubFinderHeadless.h"

#include "IPAlgorithms.h"
#include "SSAlgorithms.h"
#include "Video.h"

#include <cstring>
#include <cmath>
#include <limits>
#include <mutex>
#include <vector>

// The legacy algorithm's logging hooks normally live in the wxWidgets GUI.
// Headless consumers do not write those GUI report files.
wxString g_ReportFileName;
wxString g_ErrorFileName;

void PlaySound(wxString) {}

namespace
{
std::mutex search_mutex;
void* active_context = nullptr;
vsf_headless_segment_fn active_segment = nullptr;
bool active_apply_ocr_image_cleanup = true;

bool valid_ratio(double value)
{
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

bool valid_search_params(const vsf_headless_search_params& params)
{
    return params.threads >= 1 && params.threads <= 256 &&
        params.min_subtitle_frames >= 1 && params.min_subtitle_frames <= 1000 &&
        valid_ratio(params.text_percent) && valid_ratio(params.min_text_length) &&
        valid_ratio(params.vertical_edges_line_error) &&
        valid_ratio(params.ila_points_line_error) &&
        params.max_frame_gap_down >= 0 && params.max_frame_gap_down <= 1000 &&
        params.max_frame_gap_up >= 0 && params.max_frame_gap_up <= 1000 &&
        (params.use_isa_images == 0 || params.use_isa_images == 1) &&
        (params.use_ila_images == 0 || params.use_ila_images == 1) &&
        (params.replace_isa_with_filtered == 0 || params.replace_isa_with_filtered == 1) &&
        (params.apply_ocr_image_cleanup == 0 || params.apply_ocr_image_cleanup == 1);
}

class CallbackVideo final : public CVideo
{
public:
    CallbackVideo(
        int width,
        int height,
        void* context,
        vsf_headless_next_frame_fn next_frame)
        : context_(context), next_frame_(next_frame), frame_(static_cast<size_t>(width) * height * 3)
    {
        m_Width = width;
        m_Height = height;
        m_Duration = -1;
        m_Pos = 0;
        SetVideoWindowSettins(0.0, 1.0, 0.0, 1.0);
        m_Inited = read_next();
    }

    bool SetNullRender() override { return true; }
    void SetPos(s64) override {}
    s64 GetPos() override { return m_Pos; }

    s64 OneStepWithTimeout() override
    {
        read_next();
        return m_Pos;
    }

    int GetFrameDataSize() override
    {
        return static_cast<int>(frame_.size());
    }

    void GetFrameData(simple_buffer<u8>& output) override
    {
        if (!frame_.empty())
        {
            std::memcpy(output.m_pData, frame_.data(), frame_.size());
        }
    }

    int ConvertToBGR(
        u8* frame_data,
        simple_buffer<u8>& output,
        int xmin,
        int xmax,
        int ymin,
        int ymax) override
    {
        const int output_width = xmax - xmin + 1;
        const int output_height = ymax - ymin + 1;
        if (frame_data == nullptr || output_width <= 0 || output_height <= 0)
        {
            return -1;
        }

        for (int y = 0; y < output_height; ++y)
        {
            const size_t source_offset =
                (static_cast<size_t>(ymin + y) * m_Width + xmin) * 3;
            const size_t destination_offset = static_cast<size_t>(y) * output_width * 3;
            std::memcpy(
                output.m_pData + destination_offset,
                frame_data + source_offset,
                static_cast<size_t>(output_width) * 3);
        }
        return 0;
    }

    bool source_error() const { return source_error_; }

private:
    bool read_next()
    {
        int64_t timestamp_ms = m_Pos;
        const int32_t result = next_frame_(
            context_, frame_.data(), frame_.size(), &timestamp_ms);
        if (result > 0)
        {
            m_Pos = timestamp_ms;
            return true;
        }
        if (result < 0)
        {
            source_error_ = true;
        }
        return false;
    }

    void* context_;
    vsf_headless_next_frame_fn next_frame_;
    std::vector<u8> frame_;
    bool source_error_ = false;
};
}

int32_t vsf_headless_api_version(void)
{
    return 2;
}

int32_t vsf_headless_transform_bgr(
    const uint8_t* bgr,
    size_t bgr_len,
    int32_t width,
    int32_t height,
    uint8_t* transformed,
    size_t transformed_len)
{
    if (bgr == nullptr || transformed == nullptr || width <= 0 || height <= 0)
    {
        return -1;
    }

    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (pixels > static_cast<size_t>(std::numeric_limits<int>::max()) / 3 ||
        pixels > std::numeric_limits<size_t>::max() / 3 ||
        bgr_len != pixels * 3 || transformed_len != pixels)
    {
        return -2;
    }

    try
    {
        // The legacy implementation uses process-global configuration.
        const std::lock_guard<std::mutex> lock(search_mutex);

        // This target deliberately has no CUDA dependency.
        g_use_cuda_gpu = false;
        g_use_ocl = false;
        g_disable_save_images = true;
        g_show_results = false;
        g_show_sf_results = false;

        simple_buffer<u8> input(static_cast<int>(pixels * 3));
        simple_buffer<u8> first_filtered(static_cast<int>(pixels), static_cast<u8>(0));
        simple_buffer<u8> second_filtered(static_cast<int>(pixels), static_cast<u8>(0));
        simple_buffer<u8> output(static_cast<int>(pixels), static_cast<u8>(0));
        simple_buffer<u8> edges(static_cast<int>(pixels), static_cast<u8>(0));
        simple_buffer<u8> luminance(static_cast<int>(pixels), static_cast<u8>(0));

        std::memcpy(input.m_pData, bgr, pixels * 3);
        const int result = GetTransformedImage(
            input,
            first_filtered,
            second_filtered,
            output,
            edges,
            luminance,
            width,
            height,
            width,
            height,
            0,
            width - 1);
        std::memcpy(transformed, output.m_pData, pixels);
        return result;
    }
    catch (...)
    {
        return -3;
    }
}

void VsfHeadlessProcessResult(
    s64 start,
    s64 end,
    simple_buffer<u8>& bgr,
    simple_buffer<u8>& isa,
    simple_buffer<u16>& ila,
    int w,
    int h,
    int W,
    int H,
    int xmin,
    int ymin)
{
    if (active_segment == nullptr)
    {
        return;
    }

    try
    {
        const int pixels = w * h;
        if (!active_apply_ocr_image_cleanup)
        {
            if (active_segment(
                active_context,
                start,
                end,
                bgr.m_pData,
                static_cast<size_t>(pixels) * 3,
                w,
                h,
                3) < 0)
            {
                g_RunSubSearch = 0;
            }
            return;
        }

        simple_buffer<u8> first_filtered(pixels, static_cast<u8>(0));
        simple_buffer<u8> second_filtered(pixels, static_cast<u8>(0));
        simple_buffer<u8> transformed(pixels, static_cast<u8>(0));
        simple_buffer<u8> edges(pixels, static_cast<u8>(0));
        simple_buffer<u8> luminance(pixels, static_cast<u8>(0));

        GetTransformedImage(
            bgr,
            first_filtered,
            second_filtered,
            transformed,
            edges,
            luminance,
            w,
            h,
            W,
            H,
            0,
            w - 1);

        // This mirrors OCRPanel's in-memory reconstruction from the temporal
        // ISA/ILA images produced by FastSearchSubtitles.
        simple_buffer<u8> detected_isa(isa);
        simple_buffer<u8> combined(transformed);
        CombineTwoImages(combined, detected_isa, w, h);
        RestoreStillExistLines(detected_isa, combined, w, h, W, H);
        CombineTwoImages(first_filtered, detected_isa, w, h);
        ExtendImFWithDataFromImNF(detected_isa, first_filtered, w, h);

        simple_buffer<u8> ila_mask(pixels, static_cast<u8>(0));
        for (int i = 0; i < pixels; ++i)
        {
            ila_mask[i] = ila[i] == 0 ? 0 : 255;
        }
        IntersectTwoImages(detected_isa, ila_mask, w, h);
        IntersectTwoImages(first_filtered, ila_mask, w, h);

        const int cleaned_width = w * g_scale;
        const int cleaned_height = h * g_scale;
        simple_buffer<u8> cleaned(cleaned_width * cleaned_height, static_cast<u8>(255));
        const int cleanup_result = FindTextLines(
            bgr,
            cleaned,
            detected_isa,
            first_filtered,
            edges,
            ila_mask,
            wxT(""),
            wxT(""),
            w,
            h,
            W,
            H,
            xmin,
            ymin);

        if (cleanup_result > 0)
        {
            if (active_segment(
                active_context,
                start,
                end,
                cleaned.m_pData,
                static_cast<size_t>(cleaned_width) * cleaned_height,
                cleaned_width,
                cleaned_height,
                1) < 0)
            {
                g_RunSubSearch = 0;
            }
            return;
        }

        // Keep a detected subtitle even if the final line cleanup rejects it.
        // Convert the ISA mask to the black-on-white convention expected by OCR.
        simple_buffer<u8> fallback(pixels, static_cast<u8>(255));
        for (int i = 0; i < pixels; ++i)
        {
            fallback[i] = isa[i] == 0 ? 255 : 0;
        }
        if (active_segment(
            active_context,
            start,
            end,
            fallback.m_pData,
            static_cast<size_t>(pixels),
            w,
            h,
            1) < 0)
        {
            g_RunSubSearch = 0;
        }
    }
    catch (...)
    {
        // Detection continues; a failed segment cleanup is omitted.
    }
}

int32_t vsf_headless_search(
    int32_t width,
    int32_t height,
    const vsf_headless_search_params* params,
    void* context,
    vsf_headless_next_frame_fn next_frame,
    vsf_headless_segment_fn segment)
{
    if (width <= 0 || height <= 0 || params == nullptr || context == nullptr ||
        next_frame == nullptr || segment == nullptr)
    {
        return -1;
    }
    const size_t pixels = static_cast<size_t>(width) * height;
    if (pixels > static_cast<size_t>(std::numeric_limits<int>::max()) / 3)
    {
        return -2;
    }
    if (!valid_search_params(*params))
    {
        return -5;
    }

    const std::lock_guard<std::mutex> lock(search_mutex);
    try
    {
        g_use_cuda_gpu = false;
        g_use_ocl = false;
        g_disable_save_images = true;
        g_show_results = false;
        g_show_sf_results = false;
        g_save_each_substring_separately = false;
        g_save_scaled_images = true;
        g_threads = params->threads;
        g_DL = params->min_subtitle_frames;
        g_tp = params->text_percent;
        g_mtpl = params->min_text_length;
        g_veple = params->vertical_edges_line_error;
        g_ilaple = params->ila_points_line_error;
        g_max_dl_down = params->max_frame_gap_down;
        g_max_dl_up = params->max_frame_gap_up;
        g_use_ISA_images_for_search_subtitles = params->use_isa_images != 0;
        g_use_ILA_images_for_search_subtitles = params->use_ila_images != 0;
        g_replace_ISA_by_filtered_version = params->replace_isa_with_filtered != 0;
        g_RunSubSearch = 1;

        active_context = context;
        active_segment = segment;
        active_apply_ocr_image_cleanup = params->apply_ocr_image_cleanup != 0;

        CallbackVideo video(width, height, context, next_frame);
        if (!video.m_Inited)
        {
            active_context = nullptr;
            active_segment = nullptr;
            active_apply_ocr_image_cleanup = true;
            g_RunSubSearch = 0;
            return video.source_error() ? -4 : 0;
        }

        FastSearchSubtitles(&video, 0, -1);
        active_context = nullptr;
        active_segment = nullptr;
        active_apply_ocr_image_cleanup = true;
        g_RunSubSearch = 0;
        return video.source_error() ? -4 : 0;
    }
    catch (...)
    {
        active_context = nullptr;
        active_segment = nullptr;
        active_apply_ocr_image_cleanup = true;
        g_RunSubSearch = 0;
        return -3;
    }
}
