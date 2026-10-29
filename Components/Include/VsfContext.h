#pragma once

// Per-search configuration and state for the image processing algorithms.
//
// These used to be process-wide `g_*` globals. Each `g_*` name is now a macro
// that resolves to a field of the context installed on the current thread, so
// the algorithm code still reads and writes `g_DL`, `g_scale`, ... unchanged.
//
// Threads started through `shared_custom_task` or `run_in_parallel` inherit
// the context of the thread that started them (see DataTypes.h). Threads that
// never had a context installed share one process-wide default instance, which
// preserves the original single-search behaviour.
//
// Included from DataTypes.h after `color_range` and `TextAlignment`.

#include <cassert>
#include <chrono>
#include <vector>

class CVideo;

struct VsfContext
{
	// SSAlgorithms
	std::chrono::time_point<std::chrono::high_resolution_clock> StartTimeRunSubSearch = std::chrono::high_resolution_clock::now();
	int		RunSubSearch = 0;

	int		threads = -1; // number of threads

	int		DL = 6;	 //sub frame length
	double	tp = 0.3;	 //text percent
	double	mtpl = 0.022;  //min text len (in percent)
	double	veple = 0.30; //vedges points line error
	double	ilaple = 0.30; //ILA points line error

	bool	use_ISA_images_for_search_subtitles = true;
	bool	use_ILA_images_for_search_subtitles = true;
	bool	replace_ISA_by_filtered_version = true;
	int		max_dl_down = 20;
	int		max_dl_up = 40;

	double	video_contrast = 1.0; //1.0 default without change
	double	video_gamma = 1.0; //1.0 default without change

	CVideo*	pV = nullptr;

	// IPAlgorithms
	double	mthr = 0.4;  //moderate threshold
	double	smthr = 0.25;  //moderate threshold for scaled image
	double	mnthr = 0.3; //moderate threshold for NEdges
	int		segw = 8;  //segment width
	int		segh = 3;  //segment height
	int		msegc = 2; //minimum segments count
	int		scd = 800;   //min sum color diff
	double	btd = 0.05;   //between text distace
	double	to = 0.1; // Max Text Offset

	int		mpn = 50;	 //min points number
	double	mpd = 0.3;   //min points density
	double	msh = 0.01;	 //min symbol height in percents to full image height
	double	msd = 0.2;   //min symbol density (percent of its pixels / w*h of symbol)
	double	mpned = 0.3; //min NEdges points density

	int		min_alpha_color = 1;

	int		dmaxy = 8;

	/// min-max point size for resolution ~ 480p=640×480 scaled to x4 == 4-18p (1-4.5p in original size)
	double	minpw = 4.0 / 640.0;
	double	maxpw = 18.0 / 640.0;
	double	minph = 4.0 / 480.0;
	double	maxph = 18.0 / 480.0;
	double	minpwh = 2.0 / 3.0;

	int		min_dI = 9;
	int		min_dQ = 9;
	int		min_ddI = 14;
	int		min_ddQ = 14;

	int		scale = 4;

	bool	use_ILA_images_for_getting_txt_symbols_areas = false;
	bool	use_ILA_images_before_clear_txt_images_from_borders = false;
	bool	use_ILA_images_for_clear_txt_images = true;
	bool	use_gradient_images_for_clear_txt_images = true;
	bool	clear_txt_images_by_main_color = true;

	bool	clear_image_logical = false;

	bool	generate_cleared_text_images_on_test = false;
	bool	show_results = false;
	bool	show_sf_results = false;
	bool	clear_test_images_folder = true;
	bool	show_transformed_images_only = false;

	bool	use_ocl = true;
	bool	use_cuda_gpu = true;

	wxArrayString use_filter_color;
	wxArrayString use_outline_filter_color;
	std::vector<color_range> color_ranges;
	std::vector<color_range> outline_color_ranges;

	int		dL_color = 40;
	int		dA_color = 30;
	int		dB_color = 30;

	bool	combine_to_single_cluster = false;

	int		cuda_kmeans_initial_loop_iterations = 10;
	int		cuda_kmeans_loop_iterations = 30;
	int		cpu_kmeans_initial_loop_iterations = 10;
	int		cpu_kmeans_loop_iterations = 10;

	double	min_h = 12.0 / 720.0; // ~ min sub height in percents to image height

	// Settings for logical filtering
	bool	remove_wide_symbols = true;

	bool	disable_save_images = false;

	bool	save_each_substring_separately = true;
	bool	save_scaled_images = true;

	bool	border_is_darker = true;

	bool	extend_by_grey_color = false;
	int		allow_min_luminance = 100;

	TextAlignment text_alignment = TextAlignment::Center;

	// Opaque per-search data owned by the embedding host (e.g. the headless
	// API's result callback). The algorithms never touch it.
	void*	host = nullptr;
};

inline thread_local VsfContext* t_vsf_ctx = nullptr;

inline VsfContext& vsf_default_ctx()
{
	static VsfContext ctx;
	return ctx;
}

inline VsfContext& vsf_ctx()
{
#ifdef VSF_HEADLESS
	// Every headless entry point installs a context, and worker threads
	// inherit it. A null here means a thread was started without going
	// through DataTypes.h and would silently race on the shared default.
	assert(t_vsf_ctx != nullptr && "VsfContext missing on this thread");
#endif
	return t_vsf_ctx != nullptr ? *t_vsf_ctx : vsf_default_ctx();
}

// Installs `ctx` on the current thread for the lifetime of the scope.
class VsfContextScope
{
public:
	explicit VsfContextScope(VsfContext* ctx) : m_prev(t_vsf_ctx) { t_vsf_ctx = ctx; }
	~VsfContextScope() { t_vsf_ctx = m_prev; }
	VsfContextScope(const VsfContextScope&) = delete;
	VsfContextScope& operator=(const VsfContextScope&) = delete;

private:
	VsfContext* m_prev;
};

// Wraps `func` so that it runs with the calling thread's current context,
// whichever thread ends up executing it.
template <typename _Function>
inline auto vsf_bind_ctx(const _Function& func)
{
	return [ctx = t_vsf_ctx, func = func]() mutable {
		VsfContextScope scope(ctx);
		func();
	};
}

#define g_StartTimeRunSubSearch (vsf_ctx().StartTimeRunSubSearch)
#define g_RunSubSearch (vsf_ctx().RunSubSearch)
#define g_threads (vsf_ctx().threads)
#define g_DL (vsf_ctx().DL)
#define g_tp (vsf_ctx().tp)
#define g_mtpl (vsf_ctx().mtpl)
#define g_veple (vsf_ctx().veple)
#define g_ilaple (vsf_ctx().ilaple)
#define g_use_ISA_images_for_search_subtitles (vsf_ctx().use_ISA_images_for_search_subtitles)
#define g_use_ILA_images_for_search_subtitles (vsf_ctx().use_ILA_images_for_search_subtitles)
#define g_replace_ISA_by_filtered_version (vsf_ctx().replace_ISA_by_filtered_version)
#define g_max_dl_down (vsf_ctx().max_dl_down)
#define g_max_dl_up (vsf_ctx().max_dl_up)
#define g_video_contrast (vsf_ctx().video_contrast)
#define g_video_gamma (vsf_ctx().video_gamma)
#define g_pV (vsf_ctx().pV)

#define g_mthr (vsf_ctx().mthr)
#define g_smthr (vsf_ctx().smthr)
#define g_mnthr (vsf_ctx().mnthr)
#define g_segw (vsf_ctx().segw)
#define g_segh (vsf_ctx().segh)
#define g_msegc (vsf_ctx().msegc)
#define g_scd (vsf_ctx().scd)
#define g_btd (vsf_ctx().btd)
#define g_to (vsf_ctx().to)
#define g_mpn (vsf_ctx().mpn)
#define g_mpd (vsf_ctx().mpd)
#define g_msh (vsf_ctx().msh)
#define g_msd (vsf_ctx().msd)
#define g_mpned (vsf_ctx().mpned)
#define g_min_alpha_color (vsf_ctx().min_alpha_color)
#define g_dmaxy (vsf_ctx().dmaxy)
#define g_minpw (vsf_ctx().minpw)
#define g_maxpw (vsf_ctx().maxpw)
#define g_minph (vsf_ctx().minph)
#define g_maxph (vsf_ctx().maxph)
#define g_minpwh (vsf_ctx().minpwh)
#define g_min_dI (vsf_ctx().min_dI)
#define g_min_dQ (vsf_ctx().min_dQ)
#define g_min_ddI (vsf_ctx().min_ddI)
#define g_min_ddQ (vsf_ctx().min_ddQ)
#define g_scale (vsf_ctx().scale)
#define g_use_ILA_images_for_getting_txt_symbols_areas (vsf_ctx().use_ILA_images_for_getting_txt_symbols_areas)
#define g_use_ILA_images_before_clear_txt_images_from_borders (vsf_ctx().use_ILA_images_before_clear_txt_images_from_borders)
#define g_use_ILA_images_for_clear_txt_images (vsf_ctx().use_ILA_images_for_clear_txt_images)
#define g_use_gradient_images_for_clear_txt_images (vsf_ctx().use_gradient_images_for_clear_txt_images)
#define g_clear_txt_images_by_main_color (vsf_ctx().clear_txt_images_by_main_color)
#define g_clear_image_logical (vsf_ctx().clear_image_logical)
#define g_generate_cleared_text_images_on_test (vsf_ctx().generate_cleared_text_images_on_test)
#define g_show_results (vsf_ctx().show_results)
#define g_show_sf_results (vsf_ctx().show_sf_results)
#define g_clear_test_images_folder (vsf_ctx().clear_test_images_folder)
#define g_show_transformed_images_only (vsf_ctx().show_transformed_images_only)
#define g_use_ocl (vsf_ctx().use_ocl)
#define g_use_cuda_gpu (vsf_ctx().use_cuda_gpu)
#define g_use_filter_color (vsf_ctx().use_filter_color)
#define g_use_outline_filter_color (vsf_ctx().use_outline_filter_color)
#define g_color_ranges (vsf_ctx().color_ranges)
#define g_outline_color_ranges (vsf_ctx().outline_color_ranges)
#define g_dL_color (vsf_ctx().dL_color)
#define g_dA_color (vsf_ctx().dA_color)
#define g_dB_color (vsf_ctx().dB_color)
#define g_combine_to_single_cluster (vsf_ctx().combine_to_single_cluster)
#define g_cuda_kmeans_initial_loop_iterations (vsf_ctx().cuda_kmeans_initial_loop_iterations)
#define g_cuda_kmeans_loop_iterations (vsf_ctx().cuda_kmeans_loop_iterations)
#define g_cpu_kmeans_initial_loop_iterations (vsf_ctx().cpu_kmeans_initial_loop_iterations)
#define g_cpu_kmeans_loop_iterations (vsf_ctx().cpu_kmeans_loop_iterations)
#define g_min_h (vsf_ctx().min_h)
#define g_remove_wide_symbols (vsf_ctx().remove_wide_symbols)
#define g_disable_save_images (vsf_ctx().disable_save_images)
#define g_save_each_substring_separately (vsf_ctx().save_each_substring_separately)
#define g_save_scaled_images (vsf_ctx().save_scaled_images)
#define g_border_is_darker (vsf_ctx().border_is_darker)
#define g_extend_by_grey_color (vsf_ctx().extend_by_grey_color)
#define g_allow_min_luminance (vsf_ctx().allow_min_luminance)
#define g_text_alignment (vsf_ctx().text_alignment)
