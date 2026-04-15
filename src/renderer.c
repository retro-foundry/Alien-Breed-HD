/*
 * Alien Breed 3D I - PC Port
 * renderer.c - Software 3D renderer (chunky buffer)
 *
 * Translated from: AB3DI.s DrawDisplay (lines 3395-3693),
 *                  WallRoutine3.ChipMem.s, ObjDraw3.ChipRam.s
 *
 * The renderer draws into a flat 8-bit indexed pixel buffer.
 * The buffer is RENDER_STRIDE bytes wide and RENDER_HEIGHT lines tall.
 * Each byte is a palette index (0-31 for game colors).
 *
 * The rendering pipeline:
 *   1. Clear framebuffer
 *   2. Compute view transform (sin/cos from angpos)
 *   3. Sky on open ceilings is drawn during zone rendering
 *      (backdrop zones + synthesized hole fill), not as a full-screen pre-pass
 *   4. RotateLevelPts: transform level vertices to view space
 *   5. RotateObjectPts: transform object positions to view space
 *   6. For each zone (back-to-front from OrderZones):
 *      a. Set left/right clip from LEVELCLIPSZ
 *      b. Determine split (upper/lower room)
 *      c. DoThisRoom: iterate zone graph data, dispatch:
 *         - Walls  -> column-by-column textured drawing
 *         - Floors -> span-based textured drawing
 *         - Objects -> scaled sprite drawing
 *   7. Resolve underwater tint post-pass
 *   8. Draw gun overlay
 *   9. Swap buffers
 */

#include "renderer.h"
#include "renderer_alloc.h"
#include "renderer_3dobj.h"
#include "level.h"
#include "math_tables.h"
#include "game_data.h"
#include "game_types.h"
#include "visibility.h"
#include "audio.h"
#include <SDL.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

#if defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)))
#include <emmintrin.h>
#define AB3D_HAVE_SSE2 1
#else
#define AB3D_HAVE_SSE2 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define AB3D_PREFETCH_READ(p)  __builtin_prefetch((p), 0, 0)
#define AB3D_PREFETCH_WRITE(p) __builtin_prefetch((p), 1, 0)
#elif AB3D_HAVE_SSE2
#include <xmmintrin.h>
#define AB3D_PREFETCH_READ(p)  _mm_prefetch((const char *)(p), _MM_HINT_T0)
#define AB3D_PREFETCH_WRITE(p) _mm_prefetch((const char *)(p), _MM_HINT_T1)
#else
#define AB3D_PREFETCH_READ(p)  ((void)0)
#define AB3D_PREFETCH_WRITE(p) ((void)0)
#endif

/* Non-temporal uint16_t store: bypasses L1/L2 cache on write to avoid polluting
 * the cache hierarchy with strided column-major cw_buffer writes (960KB, exceeds L2).
 * Caller must issue _mm_sfence() before the buffer is read back.
 * GCC/Clang: __builtin_nontemporal_store generates per-target optimal code.
 * MSVC: no 16-bit NT store exists in the ISA; falls back to plain assign
 * (removing the cw prefetch hints still reduces fetch pressure). */
#if defined(__GNUC__) || defined(__clang__)
#define AB3D_NT_STORE_U16(ptr, val) __builtin_nontemporal_store((uint16_t)(val), (uint16_t *)(ptr))
#else
#define AB3D_NT_STORE_U16(ptr, val) (*(ptr) = (uint16_t)(val))
#endif

#if AB3D_HAVE_SSE2
#define AB3D_NT_STORE_FENCE() _mm_sfence()
#else
#define AB3D_NT_STORE_FENCE() ((void)0)
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define AB3D_RESTRICT __restrict
#else
#define AB3D_RESTRICT
#endif

#ifndef AB3D_CW_COL_MAJOR
#define AB3D_CW_COL_MAJOR 0
#endif

#ifndef AB3D_ENABLE_FLOOR_COL_FAST
#define AB3D_ENABLE_FLOOR_COL_FAST 0
#endif

/* Experimental edge-derived column bounds can under-cover some floor polys.
 * Keep this off by default until bounds generation is fully robust. */
#ifndef AB3D_FLOOR_FAST_USE_EDGE_BOUNDS
#define AB3D_FLOOR_FAST_USE_EDGE_BOUNDS 0
#endif

/* Ceiling in floor-column fast path is currently experimental; keep disabled by
 * default because broad scenes regress despite higher fast coverage. */
#ifndef AB3D_FLOOR_COL_FAST_INCLUDE_CEILING
#define AB3D_FLOOR_COL_FAST_INCLUDE_CEILING 0
#endif

static inline size_t renderer_cw_index_xy(int x, int y, int width, int height)
{
#if AB3D_CW_COL_MAJOR
    (void)width;
    return (size_t)x * (size_t)height + (size_t)y;
#else
    (void)height;
    return (size_t)y * (size_t)width + (size_t)x;
#endif
}

static inline size_t renderer_cw_step_y(int width)
{
#if AB3D_CW_COL_MAJOR
    (void)width;
    return 1u;
#else
    return (size_t)width;
#endif
}

static inline size_t renderer_cw_step_x(int height)
{
#if AB3D_CW_COL_MAJOR
    return (size_t)height;
#else
    (void)height;
    return 1u;
#endif
}

static inline uint16_t renderer_cw_load_xy(const uint16_t *cw, int x, int y, int width, int height)
{
    return cw[renderer_cw_index_xy(x, y, width, height)];
}

static inline void renderer_cw_store_xy(uint16_t *cw, int x, int y, int width, int height, uint16_t v)
{
    cw[renderer_cw_index_xy(x, y, width, height)] = v;
}

#if defined(__clang__) || defined(__GNUC__)
#define AB3D_ATTR_UNUSED __attribute__((unused))
#else
#define AB3D_ATTR_UNUSED
#endif

#if defined(_MSC_VER)
#define AB3D_CACHELINE_ALIGN __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
#define AB3D_CACHELINE_ALIGN __attribute__((aligned(64)))
#else
#define AB3D_CACHELINE_ALIGN
#endif

#if defined(_MSC_VER)
#define AB3D_THREAD_LOCAL __declspec(thread)
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define AB3D_THREAD_LOCAL _Thread_local
#elif defined(__GNUC__) || defined(__clang__)
#define AB3D_THREAD_LOCAL __thread
#else
#define AB3D_THREAD_LOCAL
#endif

/* Sprite row caches for renderer_draw_sprite_ctx hot path (per-thread). */
static AB3D_THREAD_LOCAL int g_sprite_src_row_lut[RENDER_INTERNAL_MAX_DIM];
static AB3D_THREAD_LOCAL uint8_t g_sprite_col_texel_lut[256];

/* Floor/ceiling UV step per pixel: d1>>FLOOR_STEP_SHIFT (same at any width so texture scale is correct). */
#define FLOOR_STEP_SHIFT  (6 + RENDER_SCALE_LOG2)  /* d1>>9 at RENDER_SCALE=8 */
/* Camera UV scale in fixed-point, matching Amiga sxoff/szoff setup in pastsides:
 * xoff/zoff words are promoted to 16.16 before entering pastfloorbright. */
#define FLOOR_CAM_UV_SCALE  65536  /* 1<<16 */
/* Floor distance uses one reciprocal lookup per screen-row distance instead of
 * redoing the same divide in every floor span setup. */
#define FLOOR_ROW_RECIP_SHIFT 24
/* Strict Amiga parity: no extra edge expansion beyond geometry rasterization. */
#define FLOOR_EDGE_EXTRA  0
#define CEILING_EDGE_EXTRA 0
#define PORTAL_EDGE_EXTRA 0
/* AB3DI itsachunkyfloor does `sub.w #12,topclip` before itsafloordraw.
 * Scale to current render resolution. */
#define CHUNKY_TOPCLIP_BIAS  (12 * RENDER_SCALE)
/* Minimum z in view space; vertices behind this are clipped. Used for walls and floor polygons. */
#define RENDERER_NEAR_PLANE 3
/* Amiga BitMapObj path uses cmp.w #50,d1 ; ble objbehind.
 * Keep this near cutoff for billboards so close sprites don't over-scale. */
#define SPRITE_NEAR_CLIP_Z 50
/* Reference Z used to project two-level zone split height to screen Y. */
#define TWO_LEVEL_SPLIT_REF_Z 400

/* -----------------------------------------------------------------------
 * Global renderer state
 * ----------------------------------------------------------------------- */
RendererState g_renderer;
/* When zero (default), raster skips redundant ARGB writes; display unpacks from cw_buffer. */
static int g_renderer_rgb_raster_expand = 0;
/* Set by display.c when the GL weapon/tint post-pass will handle gun+tint this frame. */
static int s_weapon_post_gl_active = 0;
static int s_gl_water_tint_post_active = 0;
/* Offscreen picking buffers (front/back swap with render buffers each frame). */
#define RENDERER_PICK_ZONE_NONE 0xFFFFu
static uint16_t *g_pick_zone_buffer = NULL;
static uint16_t *g_pick_zone_back_buffer = NULL;
static uint8_t *g_pick_player_buffer = NULL;
static uint8_t *g_pick_player_back_buffer = NULL;
/* Picking is one-shot on demand (F2): armed -> captured on next frame only. */
static int g_pick_capture_armed = 0;
static int g_pick_capture_active = 0;
static int g_pick_last_frame_valid = 0;
static int g_renderfix_l6_zone120_seen = 0;
static int g_renderfix_l6_zone123_seen = 0;
/* Set per-frame in renderer_draw_display; hot paths read it via RenderSliceContext. */
static int g_renderer_profile_collect_stats = 0;

typedef struct {
    int16_t left;
    int16_t top;
    int16_t right;
    int16_t bot;
    int16_t center_x;
    uint8_t is_spill;
} RendererDebugSpillSpriteRect;

#define RENDERER_DEBUG_SPILL_SPRITE_RECTS_MAX 8192
static RendererDebugSpillSpriteRect g_debug_spill_sprite_rects[RENDERER_DEBUG_SPILL_SPRITE_RECTS_MAX];
static int g_debug_spill_sprite_rect_count = 0;
static uint64_t g_debug_spill_zone_log_sig = 0;
static int g_debug_spill_zone_log_valid = 0;

void renderer_set_weapon_post_gl_active(int active) { s_weapon_post_gl_active = active; }
void renderer_set_gl_water_tint_post_active(int active) { s_gl_water_tint_post_active = active; }
int8_t renderer_get_last_fill_screen_water(void) { return g_renderer.last_fill_screen_water; }

#define RASTER_PUT_PP(pp, val) do { \
    if (g_renderer_rgb_raster_expand) \
        *(*(pp))++ = (val); \
    else \
        (*(pp))++; \
} while (0)

/* Horizontal projection reference width (effective logical width used for X projection). */
static int g_proj_base_width = RENDER_DEFAULT_WIDTH;
static uint32_t g_floor_row_abs_recip_fp[RENDER_INTERNAL_MAX_DIM + 1];
static int g_floor_row_abs_recip_limit = 0;

static inline int renderer_clamp_base_width(int w)
{
    if (w < 96) w = 96;
    if (w > RENDER_INTERNAL_MAX_DIM) w = RENDER_INTERNAL_MAX_DIM;
    return w;
}

/* Keep projection stable when users change render size at the same aspect
 * (e.g. 1280x720 <-> 1920x1080 <-> 3840x2160). Normalize logical width to a
 * fixed 1080-high baseline so same-aspect resolutions share one X-FOV. */
static inline int renderer_proj_effective_base_width_from_state(const GameState *state)
{
    int w = RENDER_DEFAULT_WIDTH;
    int h = RENDER_DEFAULT_HEIGHT;
    if (state) {
        w = (int)state->cfg_render_width;
        h = (int)state->cfg_render_height;
    }
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    {
        int64_t scaled = ((int64_t)w * 1080 + (int64_t)h / 2) / (int64_t)h;
        if (scaled < 1) scaled = 1;
        if (scaled > INT_MAX) scaled = INT_MAX;
        w = (int)scaled;
    }
    return renderer_clamp_base_width(w);
}

static inline int renderer_proj_x_scale_px_for_state(const RendererState *r, const GameState *state)
{
    int base_w = RENDER_DEFAULT_WIDTH;
    if (state) base_w = renderer_proj_effective_base_width_from_state(state);
    else base_w = renderer_clamp_base_width(g_proj_base_width);
    int cur_w = (r && r->width > 0) ? r->width : base_w;
    int64_t s = ((int64_t)RENDER_SCALE * (int64_t)cur_w + (int64_t)base_w / 2) / (int64_t)base_w;
    if (s < 1) s = 1;
    if (s > INT_MAX) s = INT_MAX;
    return (int)s;
}

/* Horizontal focal scale in internal-render pixels.
 * Keeps output FOV stable when supersampling changes internal width. */
static inline int renderer_proj_x_scale_px(void)
{
    return renderer_proj_x_scale_px_for_state(&g_renderer, NULL);
}

static inline int renderer_sprite_scale_x_for_state(const RendererState *r, const GameState *state)
{
    return 128 * renderer_proj_x_scale_px_for_state(r, state);
}

static inline int renderer_sprite_scale_x(void)
{
    /* Width path equivalent of SPRITE_SIZE_SCALE, but tied to runtime horizontal projection. */
    return renderer_sprite_scale_x_for_state(&g_renderer, NULL);
}

#define RENDERER_MAX_ZONE_ORDER 256

typedef struct {
    int16_t top_clip;
    int16_t bot_clip;
    int16_t wall_top_clip;
    int16_t wall_bot_clip;
    uint8_t valid;
} RendererZoneSectionClip;

typedef struct {
    int count;
    int16_t zone_ids[RENDERER_MAX_ZONE_ORDER];
    int16_t left_px[RENDERER_MAX_ZONE_ORDER];
    int16_t right_px[RENDERER_MAX_ZONE_ORDER];
    uint8_t valid[RENDERER_MAX_ZONE_ORDER];
    uint8_t draw_upper_first[RENDERER_MAX_ZONE_ORDER];
    RendererZoneSectionClip lower_clip[RENDERER_MAX_ZONE_ORDER];
    RendererZoneSectionClip upper_clip[RENDERER_MAX_ZONE_ORDER];
} RendererWorldZonePrepass;

typedef struct {
    uint64_t wall_segments;
    uint64_t wall_columns;
    uint64_t wall_pixels_core;
    uint64_t wall_pixels_side_ext;
    uint64_t wall_pixels_cap_ext;
    uint64_t ticks_wall;
    uint64_t floor_spans;
    uint64_t floor_pixels;
    uint64_t floor_fast_spans;
    uint64_t floor_fast_pixels;
    uint64_t water_pixels;
    uint64_t ticks_floor;
    uint64_t ticks_water;
    uint64_t sprite_calls;
    uint64_t sprite_columns;
    uint64_t sprite_pixels_visible;
    uint64_t sprite_pixels_wall_occluded;
    uint64_t sprite_pixels_spill_occluded;
    uint64_t sprite_pixels_tested;
    uint64_t sprite_pixels_drawn;
    uint64_t ticks_sprite;
} RendererWorkloadStats;

#define RENDERER_F2_MAX_SPRITE_LOG_ENTRIES 768
#define RENDERER_F2_MAX_ZONE_LIST          64

enum {
    RENDERER_F2_SPRITE_SOURCE_OBJECT = 0,
    RENDERER_F2_SPRITE_SOURCE_SHOT = 1,
    RENDERER_F2_SPRITE_SOURCE_EXPLOSION = 2
};

typedef struct {
    uint8_t source_type;
    int16_t source_slot;
    int16_t zone_id;
    uint16_t draw_count;
    uint16_t spill_draw_count;
    uint16_t local_suppressed_count;
    uint8_t adjacent_count;
    uint8_t spill_zone_count;
    int16_t adjacent_zones[RENDERER_F2_MAX_ZONE_LIST];
    int16_t spill_zones[RENDERER_F2_MAX_ZONE_LIST];
} RendererF2SpriteSpillEntry;

typedef struct {
    int valid;
    uint32_t frame_idx;
    int width;
    int height;
    double frame_ms;
    int prepass_total_zones;
    int prepass_valid_zones;
    int zones_drawn;
    int zones_drawn_lower;
    int zones_drawn_upper;
    int zone_draw_order_count;
    int16_t zone_draw_order[RENDERER_MAX_ZONE_ORDER];
    int sprite_spill_count;
    uint8_t sprite_spill_overflow;
    RendererF2SpriteSpillEntry sprite_spill_entries[RENDERER_F2_MAX_SPRITE_LOG_ENTRIES];
    RendererWorkloadStats workload;
} RendererF2PickSnapshot;

static RendererF2PickSnapshot g_renderer_f2_pick_snapshot;

typedef struct {
    int active;
    int zone_id;
    int entry_index;
    int entry_type;
    int is_water;
    int row_min;
    int row_max;
    int col_min;
    int col_max;
    uint64_t submitted_pixels;
    uint64_t first_claim_pixels;
    uint64_t prefilled_pixels;
    uint64_t wall_clip_cover_pixels;
    uint64_t same_output_pixels;
    uint64_t changed_output_pixels;
    uint64_t water_fast_path_pixels;
    uint64_t water_fallback_pixels;
    uint64_t water_blend_pixels;
    uint64_t water_single_pixels;
    uint64_t water_backbuf0_pixels;
    uint64_t water_backbuf1_pixels;
    uint8_t *before_tags;
    uint16_t *before_cw;
    size_t before_count;
    int before_ready;
} RendererZoneTraceFloorStats;

static void renderer_workload_stats_reset(RendererWorkloadStats *stats);
static void renderer_workload_stats_add(RendererWorkloadStats *dst, const RendererWorkloadStats *src);
static uint64_t renderer_workload_estimated_writes(const RendererWorkloadStats *stats);

static void renderer_draw_world_slice(GameState *state,
                                      const RendererWorldZonePrepass *zone_prepass,
                                      int16_t col_start, int16_t col_end,
                                      uint32_t frame_idx, int trace_clip,
                                      int8_t *out_fill_screen_water,
                                      RendererWorkloadStats *out_workload_stats);
static void renderer_build_zone_section_clips(GameState *state, int16_t zone_id,
                                              RendererZoneSectionClip *out_lower,
                                              RendererZoneSectionClip *out_upper,
                                              uint8_t *out_draw_upper_first);
static void renderer_draw_gun_columns(GameState *state, int col_start, int col_end);
static int renderer_compute_zone_clip_span(GameState *state, int16_t zone_id,
                                           uint32_t frame_idx, int trace_clip,
                                           int16_t *out_left_px, int16_t *out_right_px);
static void renderer_apply_underwater_tint_slice(int8_t fill_screen_water,
                                                  int16_t row_start, int16_t row_end,
                                                  int16_t col_start, int16_t col_end,
                                                  const uint32_t *src_rgb, const uint16_t *src_cw,
                                                  uint32_t *dst_rgb, uint16_t *dst_cw);
static void renderer_build_world_zone_prepass(GameState *state, uint32_t frame_idx,
                                              int trace_clip, RendererWorldZonePrepass *out);

#ifndef AB3D_NO_THREADS
#define RENDERER_MAX_THREADS 64
/* Keep column strips wide enough to reduce overlap/synchronization overhead. */
#define RENDERER_MIN_COLS_PER_WORKER 32

typedef enum {
    RENDERER_THREAD_JOB_WORLD = 0,
    RENDERER_THREAD_JOB_WATER_TINT = 1
} RendererThreadJobType;

struct RendererThreadWorker {
    SDL_Thread *thread;
    int index;
    int16_t col_start;
    int16_t col_end;
    int8_t fill_screen_water;
#if UINTPTR_MAX > 0xFFFFFFFFu
    char _pad[104];
#else
    char _pad[112];
#endif
};
typedef struct RendererThreadWorker RendererThreadWorker;

_Static_assert(sizeof(RendererThreadWorker) == 128, "RendererThreadWorker must be 128 bytes for cache-line isolation");

typedef struct {
    int initialized;
    int cpu_count;
    int worker_count;
    SDL_atomic_t stop;
    SDL_atomic_t job_generation;
    SDL_atomic_t pending_workers;
    int active_workers;
    RendererThreadJobType job_type;
    GameState *job_state;
    const RendererWorldZonePrepass *world_zone_prepass;
    uint32_t frame_idx;
    int trace_clip;
    int8_t tint_fill_screen_water;
    const uint32_t *post_src_rgb;
    const uint16_t *post_src_cw;
    uint32_t *post_dst_rgb;
    uint16_t *post_dst_cw;
    SDL_sem *done_sem;
    SDL_sem *worker_sems[RENDERER_MAX_THREADS];
    RendererThreadWorker workers[RENDERER_MAX_THREADS];
    RendererWorkloadStats worker_world_stats[RENDERER_MAX_THREADS];
    int last_logged_strip_width;
    int last_logged_workers;
} RendererThreadPool;

static RendererThreadPool g_renderer_thread_pool;
static int g_prof_last_world_workers = 0;
static int g_prof_last_tint_workers = 0;
static int g_renderer_thread_max_workers = 0; /* 0 = use max available workers */
#endif

#if defined(__EMSCRIPTEN__) && !defined(AB3D_NO_THREADS)
#include <emscripten.h>
/* SDL_GetCPUCount() is often 1 in browsers; use hardwareConcurrency for worker pool sizing. */
EM_JS(int, renderer_emscripten_hw_concurrency, (void), {
    if (typeof navigator === 'undefined' || !navigator.hardwareConcurrency)
        return 4;
    var n = navigator.hardwareConcurrency | 0;
    if (n < 2) n = 2;
    if (n > 64) n = 64;
    return n;
});
#endif

/* -----------------------------------------------------------------------
 * Big-endian read helpers (level data is Amiga big-endian)
 * ----------------------------------------------------------------------- */
static inline int16_t rd16(const uint8_t *p) {
    return (int16_t)((p[0] << 8) | p[1]);
}
static inline int32_t rd32(const uint8_t *p) {
    return (int32_t)((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
}

/* Resolve ListOfGraphRooms entry to a concrete zone id.
 * Amiga level data stores a graph index when zone_graph_adds/graphics are present. */
static int renderer_resolve_lgr_entry_zone_id(const LevelState *level, int16_t entry_word, int16_t *out_zone_id)
{
    if (!out_zone_id || entry_word < 0) return 0;
    if (level && level->zone_graph_adds && level->graphics) {
        if (level->num_zone_graph_entries > 0 && entry_word >= level->num_zone_graph_entries)
            return 0;
        {
            uint32_t gfx_off = (uint32_t)rd32(level->zone_graph_adds + (unsigned)entry_word * 8u);
            if (level->graphics_byte_count > 0 &&
                ((size_t)gfx_off + 2u > level->graphics_byte_count))
                return 0;
            *out_zone_id = rd16(level->graphics + gfx_off);
            return 1;
        }
    }
    *out_zone_id = entry_word;
    return 1;
}

/* Forward decls used by early helpers (defined later). */
static uint32_t amiga12_to_argb(uint16_t c12);
static int renderer_zone_contains_point(const LevelState *level, int16_t zone_id, int32_t x, int32_t z);
static int renderer_zone_clip_segment_to_dx_ranges(const LevelState *level,
                                                   int16_t zone_id,
                                                   int32_t seg_x0,
                                                   int32_t seg_z0,
                                                   int32_t seg_x1,
                                                   int32_t seg_z1,
                                                   int width,
                                                   int clip_dx_start,
                                                   int clip_dx_end,
                                                   int *out_dx_start,
                                                   int *out_dx_end,
                                                   int max_ranges);

#define RENDERER_GEOM_CLIP_MAX_T_VALUES 1024
#define RENDERER_GEOM_CLIP_MAX_SPANS    256
#define RENDERER_SPILL_SEARCH_MAX_VISITED   16
#define RENDERER_SPILL_SEARCH_MAX_NEIGHBORS 32
#define RENDERER_SPILL_SEARCH_MAX_GEOM_SPANS 16

/* -----------------------------------------------------------------------
 * Automap (seen wall list + raster)
 * ----------------------------------------------------------------------- */
#define AUTOMAP_HASH_EMPTY 0u

/* Automap zoom scalar: world units per pixel (bigger = zoom out). */
static int32_t g_automap_units_per_px = 8; /* was 4: zoomed out 2x */
#define AUTOMAP_UNITS_PER_PX_MIN 2
#define AUTOMAP_UNITS_PER_PX_MAX 128

/* Cached key-bit -> automap color mapping (derived from key sprites). */
static uintptr_t g_automap_key_cache_tag = 0;
static uint16_t g_automap_key_bit_to_c12[8]; /* bits 0..7 -> color; 0 = unknown */
static uint8_t  g_automap_key_bit_frame_idx[8]; /* bits 0..7 -> key sprite frame 0..3 */
static uint8_t  g_automap_key_bit_valid_mask = 0;
/* Canonical key colors (match HUD fallback palette): yellow, red, green, blue. */
static const uint16_t k_automap_key_frame_default_c12[4] = {
    0x0FC2u, 0x0F44u, 0x04D6u, 0x048Fu
};

/* Overlay used to refresh door colors every frame (O(seen * doors)); throttle to ~4 Hz. */
static uint32_t g_automap_last_door_refresh_ms = 0;

/* Workers run renderer_draw_world_slice in parallel; automap updates must be serialized. */
static SDL_mutex *g_automap_mutex;

/* Synthesized backdrop sky-hole polygons, built once per level load. */
typedef struct {
    int16_t sides;
    int16_t pt_indices[100];
} RendererSkyCachePoly;

typedef struct {
    uint32_t start;
    uint32_t count;
} RendererSkyCacheBucket;

typedef struct {
    int floor_polys_seen;
    int floor_polys_sky_added;
    int sky_added_missing_matching_roof;
    int floor_polys_with_matching_roof;
    int sky_push_failed;
} RendererSkyBuildDebug;

static RendererSkyCachePoly *g_level_sky_cache_polys;
static uint32_t g_level_sky_cache_poly_count;
static uint32_t g_level_sky_cache_poly_cap;
static RendererSkyCacheBucket *g_level_sky_cache_buckets;
static int g_level_sky_cache_zone_slots;

static void renderer_reset_level_sky_cache_internal(void);
static void renderer_floor_fast_release_scratch(void);

static void automap_lock(void)
{
    if (g_automap_mutex) SDL_LockMutex(g_automap_mutex);
}

static void automap_unlock(void)
{
    if (g_automap_mutex) SDL_UnlockMutex(g_automap_mutex);
}

static uint32_t automap_hash_u32(uint32_t x)
{
    /* Simple 32-bit mix (good enough for gfx offsets). */
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

uint32_t renderer_automap_seen_key_plus1(uint32_t gfx_off,
                                         int16_t x1, int16_t z1,
                                         int16_t x2, int16_t z2)
{
    /* Canonicalize endpoint order so drawing direction does not affect dedupe. */
    if (x2 < x1 || (x2 == x1 && z2 < z1)) {
        int16_t tx = x1, tz = z1;
        x1 = x2; z1 = z2;
        x2 = tx; z2 = tz;
    }

    uint32_t ep0 = (uint32_t)(uint16_t)x1 | ((uint32_t)(uint16_t)z1 << 16);
    uint32_t ep1 = (uint32_t)(uint16_t)x2 | ((uint32_t)(uint16_t)z2 << 16);

    uint32_t k = automap_hash_u32(gfx_off ^ 0x9E3779B9u);
    k = automap_hash_u32(k ^ ep0);
    k = automap_hash_u32(k ^ (ep1 * 0x85EBCA6Bu));

    /* Hash table stores key+1; 0 means empty slot. */
    k += 1u;
    if (k == 0u) k = 1u;
    return k;
}

static AB3D_ATTR_UNUSED void automap_reset(LevelState *level)
{
    if (!level) return;
    level->automap_seen_count = 0;
    if (level->automap_seen_hash && level->automap_seen_hash_cap) {
        memset(level->automap_seen_hash, 0, (size_t)level->automap_seen_hash_cap * sizeof(uint32_t));
    }
}

static int automap_ensure_hash(LevelState *level, uint32_t want_cap_pow2)
{
    if (!level) return 0;
    if (want_cap_pow2 < 1024u) want_cap_pow2 = 1024u;
    /* ensure power-of-two */
    if ((want_cap_pow2 & (want_cap_pow2 - 1u)) != 0u) {
        uint32_t p = 1u;
        while (p < want_cap_pow2) p <<= 1u;
        want_cap_pow2 = p;
    }

    if (level->automap_seen_hash && level->automap_seen_hash_cap >= want_cap_pow2)
        return 1;

    uint32_t *new_tab = (uint32_t *)calloc((size_t)want_cap_pow2, sizeof(uint32_t));
    if (!new_tab) return 0;

    /* Rehash existing entries if any. Table stores seen-wall key+1, 0 = empty. */
    if (level->automap_seen_hash && level->automap_seen_hash_cap) {
        for (uint32_t i = 0; i < level->automap_seen_hash_cap; i++) {
            uint32_t v = level->automap_seen_hash[i];
            if (v == AUTOMAP_HASH_EMPTY) continue;
            uint32_t key_plus1 = v;
            uint32_t cap = want_cap_pow2;
            uint32_t mask = cap - 1u;
            uint32_t h = automap_hash_u32(key_plus1) & mask;
            while (new_tab[h] != AUTOMAP_HASH_EMPTY) h = (h + 1u) & mask;
            new_tab[h] = key_plus1;
        }
        free(level->automap_seen_hash);
    }

    level->automap_seen_hash = new_tab;
    level->automap_seen_hash_cap = want_cap_pow2;
    return 1;
}

static int automap_ensure_walls(LevelState *level, uint32_t want_cap)
{
    if (!level) return 0;
    if (want_cap < 256u) want_cap = 256u;
    if (level->automap_seen_walls && level->automap_seen_cap >= want_cap)
        return 1;
    uint32_t new_cap = (level->automap_seen_cap > 0) ? level->automap_seen_cap : 256u;
    while (new_cap < want_cap) new_cap *= 2u;
    AutomapSeenWall *nw = (AutomapSeenWall *)realloc(level->automap_seen_walls, (size_t)new_cap * sizeof(AutomapSeenWall));
    if (!nw) return 0;
    level->automap_seen_walls = nw;
    level->automap_seen_cap = new_cap;
    return 1;
}

static uint8_t automap_level_key_mask(const LevelState *level)
{
    /* Derive which condition bits correspond to keys by scanning key pickup objects.
     * Keys store their condition bit(s) in obj.can_see (int8). */
    if (!level || !level->object_data) return 0;
    uint8_t mask = 0;
    for (int i = 0; i < 250; i++) {
        const GameObject *obj = (const GameObject *)(level->object_data + (size_t)i * OBJECT_SIZE);
        if (OBJ_CID(obj) < 0) break;
        if ((int8_t)obj->obj.number != (int8_t)OBJ_NBR_KEY) continue;
        mask |= (uint8_t)obj->obj.can_see;
    }
    return mask;
}

/* Key sprite (type 5) layout matches frames_keys[] below. */
static const uint16_t k_automap_key_ptr_off[4] = { 0u, 0u, 32u * 4u, 32u * 4u };
static const uint16_t k_automap_key_down_strip[4] = { 0u, 32u, 0u, 32u };

static uint32_t automap_c12_saturation(uint16_t c12)
{
    uint32_t r = (uint32_t)((c12 >> 8) & 0xFu);
    uint32_t g = (uint32_t)((c12 >> 4) & 0xFu);
    uint32_t b = (uint32_t)(c12 & 0xFu);
    uint32_t mx = r > g ? r : g;
    mx = mx > b ? mx : b;
    uint32_t mn = r < g ? r : g;
    mn = mn < b ? mn : b;
    return mx - mn;
}

/* Sample key art like in-game sprites: brightness-graded .pal + pick saturated texels (not grey metal average). */
static AB3D_ATTR_UNUSED uint16_t automap_key_frame_representative_c12(const RendererState *r, int frame_idx)
{
    if (!r) return 0;
    if (frame_idx < 0 || frame_idx >= 4) return 0;
    if (!r->sprite_wad[5] || !r->sprite_ptr[5] || !r->sprite_pal_data[5]) return 0;

    uint32_t ptr_off = (uint32_t)k_automap_key_ptr_off[frame_idx];
    uint16_t down_strip = k_automap_key_down_strip[frame_idx];

    const uint8_t *wad = r->sprite_wad[5];
    size_t wad_size = r->sprite_wad_size[5];
    const uint8_t *ptr = r->sprite_ptr[5];
    size_t ptr_size = r->sprite_ptr_size[5];
    const uint8_t *pal = r->sprite_pal_data[5];
    size_t pal_size = r->sprite_pal_size[5];

    if (ptr_size < ptr_off + 4u) return 0;
    /* Multi-level palette: 15 x 32 colors x 2 bytes (see renderer_draw_sprite_ctx). */
    uint32_t pal_level_off = 0;
    if (pal_size >= 960u)
        pal_level_off = 64u * 7u; /* mid brightness (~typical mid-distance key) */
    if (pal_level_off + 64u > pal_size)
        pal_level_off = 0;

    const int cols = 32;
    const int rows = 32;
    uint16_t best_c12 = 0;
    uint32_t best_sat = 0;
    uint32_t best_luma = 0;

    for (int c = 0; c < cols; c++) {
        uint32_t entry_off = ptr_off + (uint32_t)c * 4u;
        if (entry_off + 4u > ptr_size) continue;
        const uint8_t *entry = ptr + entry_off;
        uint8_t mode = entry[0];
        uint32_t wad_off = ((uint32_t)entry[1] << 16) | ((uint32_t)entry[2] << 8) | (uint32_t)entry[3];
        if (mode == 0 && wad_off == 0) continue;
        if (wad_off >= wad_size) continue;
        const uint8_t *src = wad + wad_off;

        for (int r0 = 0; r0 < rows; r0++) {
            int row_idx = (int)down_strip + r0;
            if (wad_off + (size_t)(row_idx + 1) * 2u > wad_size) continue;
            uint16_t w = (uint16_t)((src[row_idx * 2u] << 8) | src[row_idx * 2u + 1u]);
            uint8_t texel;
            if (mode == 0) texel = (uint8_t)(w & 0x1Fu);
            else if (mode == 1) texel = (uint8_t)((w >> 5) & 0x1Fu);
            else texel = (uint8_t)((w >> 10) & 0x1Fu);
            if (texel == 0) continue;
            uint32_t ci = pal_level_off + (uint32_t)texel * 2u;
            if (ci + 1u >= pal_size) continue;
            uint16_t c12 = (uint16_t)((pal[ci] << 8) | pal[ci + 1u]);
            uint32_t sat = automap_c12_saturation(c12);
            uint32_t luma = (uint32_t)((c12 >> 8) & 0xFu) + (uint32_t)((c12 >> 4) & 0xFu) + (uint32_t)(c12 & 0xFu);
            if (sat > best_sat || (sat == best_sat && luma > best_luma)) {
                best_sat = sat;
                best_luma = luma;
                best_c12 = c12;
            }
        }
    }
    return best_c12;
}

static uintptr_t automap_key_cache_tag_for_level(const LevelState *level)
{
    uintptr_t tag = ((uintptr_t)g_renderer.sprite_wad[5] << 1) ^
                    ((uintptr_t)g_renderer.sprite_ptr[5] << 3) ^
                    ((uintptr_t)g_renderer.sprite_pal_data[5] << 5);
    if (!level || !level->object_data) return tag;

    uint32_t h = 0xA3612B29u;
    for (int i = 0; i < 250; i++) {
        const GameObject *obj = (const GameObject *)(level->object_data + (size_t)i * OBJECT_SIZE);
        if (OBJ_CID(obj) < 0) break;
        if ((int8_t)obj->obj.number != (int8_t)OBJ_NBR_KEY) continue;

        uint32_t v = (uint32_t)(uint8_t)obj->obj.can_see;
        v |= (uint32_t)(uint16_t)rd16(obj->raw + 10) << 8;
        v ^= (uint32_t)i * 0x9E3779B9u;
        h = automap_hash_u32(h ^ v);
    }

    tag ^= (uintptr_t)h;
#if UINTPTR_MAX > 0xFFFFFFFFu
    tag ^= (uintptr_t)automap_hash_u32(h ^ 0xD2511F53u) << 32;
#endif
    return tag;
}

static void automap_refresh_key_bit_colors(const LevelState *level)
{
    /* Recompute when level or key sprite assets change. */
    uintptr_t tag = automap_key_cache_tag_for_level(level);
    if (tag == g_automap_key_cache_tag) return;
    g_automap_key_cache_tag = tag;
    memset(g_automap_key_bit_to_c12, 0, sizeof(g_automap_key_bit_to_c12));
    memset(g_automap_key_bit_frame_idx, 0, sizeof(g_automap_key_bit_frame_idx));
    g_automap_key_bit_valid_mask = 0;
    if (!level || !level->object_data) return;

    /* Per-frame key color: use canonical stable colors to avoid palette-sampling
     * highlights tinting blue/green keys toward yellow on the automap. */
    uint16_t frame_c12[4];
    for (int fi = 0; fi < 4; fi++)
        frame_c12[fi] = k_automap_key_frame_default_c12[fi];

    /* Map condition bits -> key art: each key object ties can_see bits to objVectFrameNumber. */
    for (int i = 0; i < 250; i++) {
        const GameObject *obj = (const GameObject *)(level->object_data + (size_t)i * OBJECT_SIZE);
        if (OBJ_CID(obj) < 0) break;
        if ((int8_t)obj->obj.number != (int8_t)OBJ_NBR_KEY) continue;
        uint8_t bitmask = (uint8_t)obj->obj.can_see;
        int16_t frame_num = rd16(obj->raw + 10);
        int fi = (int)((frame_num % 4 + 4) % 4);

        for (int b = 0; b < 8; b++) {
            uint8_t bit = (uint8_t)(1u << b);
            if ((bitmask & bit) == 0) continue;
            if ((g_automap_key_bit_valid_mask & bit) != 0) continue;
            g_automap_key_bit_to_c12[b] = frame_c12[fi];
            g_automap_key_bit_frame_idx[b] = (uint8_t)fi;
            g_automap_key_bit_valid_mask |= bit;
        }
    }

    /* Fallback when a door bit has no key object in the map: 1->frame0, 2->frame1, 4->frame2, 8->frame3. */
    static const uint8_t k_nibble_to_frame[4] = { 0, 1, 2, 3 };
    for (int n = 0; n < 4; n++) {
        uint8_t bit = (uint8_t)(1u << n);
        if ((g_automap_key_bit_valid_mask & bit) != 0) continue;
        uint16_t c = frame_c12[k_nibble_to_frame[n]];
        if (c != 0) g_automap_key_bit_to_c12[n] = c;
        g_automap_key_bit_frame_idx[n] = k_nibble_to_frame[n];
    }
}

static uint8_t automap_key_id_from_door_flags_masked(uint16_t door_flags, uint8_t key_mask)
{
    /* Return:
     * - 0x00: no key requirement
     * - one key bit (low nibble always treated as key; plus discovered key_mask bits): keyed door
     * - 0x80: switch/condition door (non-zero flags but no key bits) */
    uint8_t f = (uint8_t)(door_flags & 0x00FFu);
    /* Original levels use key bits in low nibble; keep that as baseline even if
     * key objects are currently inactive, then expand with any discovered key bits. */
    uint8_t keys = (uint8_t)(f & (uint8_t)(key_mask | 0x0Fu));
    if (keys != 0) {
        /* Prefer single bit if present, else pick lowest set key bit. */
        if ((keys & (uint8_t)(keys - 1u)) == 0u) return keys;
        return (uint8_t)(keys & (uint8_t)(-(int8_t)keys));
    }
    if (f != 0) return 0x80u;
    return 0u;
}

static int automap_segments_match_unordered_i16(int16_t ax1, int16_t az1, int16_t ax2, int16_t az2,
                                                int16_t bx1, int16_t bz1, int16_t bx2, int16_t bz2)
{
    if (ax1 == bx1 && az1 == bz1 && ax2 == bx2 && az2 == bz2) return 1;
    if (ax1 == bx2 && az1 == bz2 && ax2 == bx1 && az2 == bz1) return 1;
    return 0;
}

static int automap_entry_fline_matches_segment(const LevelState *level, const uint8_t *ent,
                                               int16_t x1, int16_t z1, int16_t x2, int16_t z2)
{
    if (!level || !level->floor_lines || !ent) return 0;
    int16_t fline = rd16(ent + 0);
    if (fline < 0 || (int32_t)fline >= level->num_floor_lines) return 0;

    const uint8_t *fl = level->floor_lines + (size_t)(uint16_t)fline * 16u;
    int16_t fx1 = rd16(fl + 0);
    int16_t fz1 = rd16(fl + 2);
    int16_t fx2 = (int16_t)(fx1 + rd16(fl + 4));
    int16_t fz2 = (int16_t)(fz1 + rd16(fl + 6));
    return automap_segments_match_unordered_i16(x1, z1, x2, z2, fx1, fz1, fx2, fz2);
}

#define AUTOMAP_META_ZONE_MASK      0x3FFFu
#define AUTOMAP_META_INTERNAL_KNOWN 0x4000u
#define AUTOMAP_META_INTERNAL       0x8000u
#define AUTOMAP_META_NOT_INTERNAL   0x7FFFu

static uint16_t automap_pack_zone_hint(int16_t zone_id)
{
    if (zone_id < 0) return 0u;
    if (zone_id > (int16_t)(AUTOMAP_META_ZONE_MASK - 1u))
        zone_id = (int16_t)(AUTOMAP_META_ZONE_MASK - 1u);
    return (uint16_t)((uint16_t)zone_id + 1u);
}

static int16_t automap_unpack_zone_hint(uint16_t packed)
{
    uint16_t zone = (uint16_t)(packed & AUTOMAP_META_ZONE_MASK);
    if (zone == 0u) return -1;
    return (int16_t)(zone - 1u);
}

static uint16_t automap_meta_set_zone_hint(uint16_t meta, int16_t zone_id)
{
    uint16_t flags = (uint16_t)(meta & (AUTOMAP_META_INTERNAL_KNOWN | AUTOMAP_META_INTERNAL));
    uint16_t zone = automap_pack_zone_hint(zone_id);
    return (uint16_t)(flags | zone);
}

static int automap_meta_internal_known(uint16_t meta)
{
    return (meta & AUTOMAP_META_INTERNAL_KNOWN) != 0u;
}

static int automap_meta_internal(uint16_t meta)
{
    return (meta & AUTOMAP_META_INTERNAL) != 0u;
}

static uint16_t automap_meta_set_internal(uint16_t meta, int is_internal)
{
    meta |= AUTOMAP_META_INTERNAL_KNOWN;
    if (is_internal)
        meta |= AUTOMAP_META_INTERNAL;
    else
        meta = (uint16_t)(meta & AUTOMAP_META_NOT_INTERNAL);
    return meta;
}

/* Internal lines are floor lines that connect to another zone (connect >= 0). */
static int automap_segment_is_internal_connected(const LevelState *level,
                                                 int16_t x1, int16_t z1,
                                                 int16_t x2, int16_t z2)
{
    if (!level || !level->floor_lines || level->num_floor_lines <= 0) return 0;

    for (int32_t i = 0; i < level->num_floor_lines; i++) {
        const uint8_t *fl = level->floor_lines + (size_t)i * 16u;
        int16_t fx1 = rd16(fl + 0);
        int16_t fz1 = rd16(fl + 2);
        int16_t fx2 = (int16_t)(fx1 + rd16(fl + 4));
        int16_t fz2 = (int16_t)(fz1 + rd16(fl + 6));

        if (!automap_segments_match_unordered_i16(x1, z1, x2, z2, fx1, fz1, fx2, fz2))
            continue;

        if (rd16(fl + 8) >= 0) return 1;
    }
    return 0;
}

static uint8_t automap_door_key_for_wall_gfx_off(const LevelState *level, uint32_t gfx_off,
                                                  int16_t x1, int16_t z1, int16_t x2, int16_t z2,
                                                  int16_t zone_hint,
                                                  int *out_is_door)
{
    if (out_is_door) *out_is_door = 0;
    if (!level || !level->door_wall_list || !level->door_wall_list_offsets || level->num_doors <= 0)
        return 0;

    uint8_t key_mask = automap_level_key_mask(level);
    int matched_door = -1;
    int matched_score = -1;

    /* door_wall_list entries are 10 bytes: fline(be16) + gfx_off(be32) + gfx_base(be32) */
    const uint8_t *lst = level->door_wall_list;
    for (int di = 0; di < level->num_doors; di++) {
        uint32_t start = level->door_wall_list_offsets[di];
        uint32_t end = level->door_wall_list_offsets[di + 1];
        int door_zone = -1;
        if (level->door_data)
            door_zone = (int)rd16(level->door_data + (size_t)di * 22u + 0u);
        for (uint32_t j = start; j < end; j++) {
            const uint8_t *ent = lst + (size_t)j * 10u;
            uint32_t ent_gfx = (uint32_t)rd32(ent + 2);
            int seg_match = automap_entry_fline_matches_segment(level, ent, x1, z1, x2, z2);
            int gfx_match = (ent_gfx == gfx_off);
            int zone_match = (zone_hint >= 0 && door_zone >= 0 && door_zone == zone_hint);
            int score;

            if (!gfx_match && !seg_match) continue;

            /* Prefer zone+segment+gfx, then zone+segment, then segment+gfx, then segment, then gfx. */
            score = (zone_match ? 4 : 0) + (seg_match ? 2 : 0) + (gfx_match ? 1 : 0);
            if (score > matched_score) {
                matched_score = score;
                matched_door = di;
                if (score >= 7) break; /* best possible */
            }
        }
        if (matched_score >= 7) break;
    }

    {
        if (matched_door < 0) return 0;

        if (out_is_door) *out_is_door = 1;
        if (!level->door_data) return 0;
        const uint8_t *door = level->door_data + (size_t)matched_door * 22u;
        uint16_t door_flags = (uint16_t)rd16(door + 20);
        return automap_key_id_from_door_flags_masked(door_flags, key_mask);
    }
}

static void automap_mark_seen(LevelState *level,
                              uint32_t gfx_off,
                              int16_t zone_id,
                              int16_t x1, int16_t z1, int16_t x2, int16_t z2,
                              uint8_t is_door, uint8_t door_key_id)
{
    if (!level) return;
    if (!level->graphics || level->graphics_byte_count == 0) return;
    if (gfx_off == 0 || gfx_off >= level->graphics_byte_count) return;

    automap_lock();

    /* Lazy init structures. */
    if (!automap_ensure_hash(level, 2048u)) {
        automap_unlock();
        return;
    }

    /* Maintain load factor <= ~0.6 */
    if (level->automap_seen_hash_cap && level->automap_seen_count * 10u >= level->automap_seen_hash_cap * 6u) {
        if (!automap_ensure_hash(level, level->automap_seen_hash_cap * 2u)) {
            automap_unlock();
            return;
        }
    }

    uint32_t key_plus1 = renderer_automap_seen_key_plus1(gfx_off, x1, z1, x2, z2);
    uint32_t cap = level->automap_seen_hash_cap;
    uint32_t mask = cap - 1u;
    uint32_t h = automap_hash_u32(key_plus1) & mask;
    while (1) {
        uint32_t v = level->automap_seen_hash[h];
        if (v == AUTOMAP_HASH_EMPTY) break;
        if (v == key_plus1) {
            /* Existing entry: refresh metadata so stale color assignments self-heal. */
            for (uint32_t i = 0; i < level->automap_seen_count; i++) {
                AutomapSeenWall *ew = &level->automap_seen_walls[i];
                uint32_t ek = renderer_automap_seen_key_plus1(ew->gfx_off, ew->x1, ew->z1, ew->x2, ew->z2);
                if (ek != key_plus1) continue;
                if (is_door) {
                    ew->is_door = 1;
                    ew->door_key_id = door_key_id;
                }
                {
                    uint16_t meta = ew->reserved;
                    if (zone_id >= 0) meta = automap_meta_set_zone_hint(meta, zone_id);
                    if (!automap_meta_internal_known(meta)) {
                        int is_internal = automap_segment_is_internal_connected(level,
                                                                               ew->x1, ew->z1,
                                                                               ew->x2, ew->z2);
                        meta = automap_meta_set_internal(meta, is_internal);
                    }
                    ew->reserved = meta;
                }
                break;
            }
            automap_unlock();
            return; /* already seen */
        }
        h = (h + 1u) & mask;
    }

    if (!automap_ensure_walls(level, level->automap_seen_count + 1u)) {
        automap_unlock();
        return;
    }

    AutomapSeenWall *w = &level->automap_seen_walls[level->automap_seen_count++];
    w->gfx_off = gfx_off;
    w->x1 = x1; w->z1 = z1;
    w->x2 = x2; w->z2 = z2;
    w->is_door = is_door;
    w->door_key_id = door_key_id;
    {
        uint16_t meta = automap_pack_zone_hint(zone_id);
        int is_internal = automap_segment_is_internal_connected(level, x1, z1, x2, z2);
        w->reserved = automap_meta_set_internal(meta, is_internal);
    }

    level->automap_seen_hash[h] = key_plus1;
    automap_unlock();
}

static inline uint16_t automap_color_for(uint8_t is_door, uint8_t key_id)
{
    if (!is_door) return 0x0FFFu; /* white */
    if (key_id == 0) return 0x0888u; /* grey: door without key */
    if (key_id == 0x80u) return 0x0444u; /* dark grey: switch/condition door */
    return 0x0F0Fu; /* hot pink: any key-locked door */
}

uint16_t renderer_key_condition_bit_color_c12(const GameState *state, int bit_index)
{
    if (!state || bit_index < 0 || bit_index > 3) return 0;
    automap_refresh_key_bit_colors(&state->level);
    return g_automap_key_bit_to_c12[bit_index];
}

int renderer_key_sprite_frame_for_condition_bit(const GameState *state, int bit_index)
{
    if (!state || bit_index < 0 || bit_index > 3) return 0;
    automap_refresh_key_bit_colors(&state->level);
    return (int)g_automap_key_bit_frame_idx[bit_index];
}

uintptr_t renderer_key_sprite_hud_cache_tag(const GameState *state)
{
    if (!state) return 0;
    return automap_key_cache_tag_for_level(&state->level);
}

/* Rasterize key sprite type 5 frame (0..3) to ARGB8888; texel 0 = transparent. Matches automap sampling. */
int renderer_key_sprite_rasterize_frame_argb(int frame_index, uint32_t *out, int stride_pixels)
{
    if (frame_index < 0 || frame_index >= 4 || !out || stride_pixels < 32) return 0;
    if (!g_renderer.sprite_wad[5] || !g_renderer.sprite_ptr[5] || !g_renderer.sprite_pal_data[5]) return 0;

    uint32_t ptr_off = (uint32_t)k_automap_key_ptr_off[frame_index];
    uint16_t down_strip = k_automap_key_down_strip[frame_index];

    const uint8_t *wad = g_renderer.sprite_wad[5];
    size_t wad_size = g_renderer.sprite_wad_size[5];
    const uint8_t *ptr = g_renderer.sprite_ptr[5];
    size_t ptr_size = g_renderer.sprite_ptr_size[5];
    const uint8_t *pal = g_renderer.sprite_pal_data[5];
    size_t pal_size = g_renderer.sprite_pal_size[5];

    uint32_t pal_level_off = 0;
    if (pal_size >= 960u)
        pal_level_off = 64u * 7u;
    if (pal_level_off + 64u > pal_size)
        pal_level_off = 0;

    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            uint32_t *dst = out + x + y * stride_pixels;
            uint32_t entry_off = ptr_off + (uint32_t)x * 4u;
            if (entry_off + 4u > ptr_size) {
                *dst = 0;
                continue;
            }
            const uint8_t *entry = ptr + entry_off;
            uint8_t mode = entry[0];
            uint32_t wad_off = ((uint32_t)entry[1] << 16) | ((uint32_t)entry[2] << 8) | (uint32_t)entry[3];
            if (mode == 0 && wad_off == 0) {
                *dst = 0;
                continue;
            }
            if (wad_off >= wad_size) {
                *dst = 0;
                continue;
            }
            const uint8_t *src = wad + wad_off;
            int row_idx = (int)down_strip + y;
            if (wad_off + (size_t)(row_idx + 1) * 2u > wad_size) {
                *dst = 0;
                continue;
            }
            uint16_t w = (uint16_t)((src[row_idx * 2u] << 8) | src[row_idx * 2u + 1u]);
            uint8_t texel;
            if (mode == 0) texel = (uint8_t)(w & 0x1Fu);
            else if (mode == 1) texel = (uint8_t)((w >> 5) & 0x1Fu);
            else texel = (uint8_t)((w >> 10) & 0x1Fu);
            if (texel == 0) {
                *dst = 0;
                continue;
            }
            uint32_t ci = pal_level_off + (uint32_t)texel * 2u;
            if (ci + 1u >= pal_size) {
                *dst = 0;
                continue;
            }
            uint16_t c12 = (uint16_t)((pal[ci] << 8) | pal[ci + 1u]);
            uint32_t argb = amiga12_to_argb(c12);
            *dst = (argb & 0x00FFFFFFu) | 0xFF000000u;
        }
    }
    return 1;
}

void renderer_automap_lock(void)
{
    automap_lock();
}

void renderer_automap_unlock(void)
{
    automap_unlock();
}

int renderer_automap_collect_line_segments(GameState *state,
                                           int *x0, int *y0, int *x1, int *y1,
                                           uint16_t *c12, int max_lines)
{
    if (!state || max_lines < 1 || !x0 || !y0 || !x1 || !y1 || !c12) return 0;
    LevelState *level = &state->level;

    PlayerState *plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
    int16_t px = (int16_t)(plr->xoff >> 16);
    int16_t pz = (int16_t)(plr->zoff >> 16);

    int w = g_renderer.width;
    int h = g_renderer.height;
    int cx = w / 2;
    int cy = h / 2;
    const int mx = w - 1; /* horizontal flip: x' = mx - x */

    int32_t map_scale = g_automap_units_per_px;
    if (map_scale < 1) map_scale = 1;

    int n = 0;
    /* Reserve last 3 slots for the player arrow so overlay draws walls first, player on top. */
    int wall_cap = (max_lines >= 3) ? (max_lines - 3) : max_lines;

    uint32_t now_ms = SDL_GetTicks();
    int refresh_doors = (g_automap_last_door_refresh_ms == 0u) ||
        (now_ms - g_automap_last_door_refresh_ms >= 250u);
    if (refresh_doors)
        g_automap_last_door_refresh_ms = now_ms;

    automap_lock();
    if (level->automap_seen_walls && level->automap_seen_count != 0) {
        automap_refresh_key_bit_colors(level);

        for (uint32_t i = 0; i < level->automap_seen_count && n < wall_cap; i++) {
        AutomapSeenWall *sw = &level->automap_seen_walls[i];
        uint8_t is_door = sw->is_door;
        uint8_t key_id = sw->door_key_id;
        /* Door state / key pickup: expensive scan — not every frame (was freezing the game). */
        if (is_door && refresh_doors) {
            int now_is_door = 0;
            int16_t zone_hint = automap_unpack_zone_hint(sw->reserved);
            uint8_t now_key_id = automap_door_key_for_wall_gfx_off(level, sw->gfx_off,
                                                                    sw->x1, sw->z1, sw->x2, sw->z2,
                                                                    zone_hint,
                                                                    &now_is_door);
            if (now_is_door && now_key_id != key_id) {
                key_id = now_key_id;
                sw->door_key_id = now_key_id;
            }
        }
        int32_t ax0 = (int32_t)sw->x1 - (int32_t)px;
        int32_t az0 = (int32_t)sw->z1 - (int32_t)pz;
        int32_t ax1 = (int32_t)sw->x2 - (int32_t)px;
        int32_t az1 = (int32_t)sw->z2 - (int32_t)pz;
        int wx0 = cx + (int)(ax0 / map_scale);
        int wy0 = cy + (int)(az0 / map_scale);
        int wx1 = cx + (int)(ax1 / map_scale);
        int wy1 = cy + (int)(az1 / map_scale);
        x0[n] = mx - wx0;
        y0[n] = wy0;
        x1[n] = mx - wx1;
        y1[n] = wy1;
        {
            /* Internal edge dimming uses metadata from automap_mark_seen only. Per-frame
             * automap_segment_is_internal_connected here was O(seen * floor_lines) and froze. */
            uint16_t meta = sw->reserved;
            c12[n] = automap_color_for(is_door, key_id);
            if (automap_meta_internal_known(meta) && automap_meta_internal(meta))
                c12[n] = (uint16_t)(c12[n] | RENDERER_AUTOMAP_SEGFLAG_INTERNAL);
        }
        n++;
        }
    }
    automap_unlock();

    /* Player arrow (after walls so the SDL/GL overlay draws it last). */
    if (max_lines >= 3 && n + 3 <= max_lines) {
        uint16_t cw = (uint16_t)(0x0FFFu | RENDERER_AUTOMAP_SEGFLAG_PLAYER);
        int ang = (int)(plr->angpos & 0x1FFF);
        int32_t fx = (int32_t)sin_lookup(ang);
        int32_t fz = (int32_t)cos_lookup(ang);
        int32_t len = 40;
        int tipx = cx + (int)((fx * len) / 32768);
        int tipy = cy + (int)((fz * len) / 32768);
        int aL = (ang + (ANGLE_FULL * 3) / 8) & ANGLE_MASK;
        int aR = (ang + (ANGLE_FULL * 5) / 8) & ANGLE_MASK;
        int32_t lx = (int32_t)sin_lookup(aL);
        int32_t lz = (int32_t)cos_lookup(aL);
        int32_t rx = (int32_t)sin_lookup(aR);
        int32_t rz = (int32_t)cos_lookup(aR);
        int wing = 20;
        int lox = tipx + (int)((lx * wing) / 32768);
        int loy = tipy + (int)((lz * wing) / 32768);
        int rox = tipx + (int)((rx * wing) / 32768);
        int roy = tipy + (int)((rz * wing) / 32768);

        x0[n] = mx - cx; y0[n] = cy; x1[n] = mx - tipx; y1[n] = tipy; c12[n] = cw; n++;
        x0[n] = mx - tipx; y0[n] = tipy; x1[n] = mx - lox; y1[n] = loy; c12[n] = cw; n++;
        x0[n] = mx - tipx; y0[n] = tipy; x1[n] = mx - rox; y1[n] = roy; c12[n] = cw; n++;
    }
    return n;
}

/* -----------------------------------------------------------------------
 * SCALE table (from Macros.i)
 *
 * Amiga SCALE macro (Macros.i): d6 0..64 -> LUT block offset. 32 blocks of 64 bytes.
 *   d6=0  = closest (brightest), d6=64 = farthest (dimmest).
 * ----------------------------------------------------------------------- */
static const uint16_t wall_scale_table[65] = {
    64*0,  64*1,  64*1,  64*2,  64*2,  64*3,  64*3,  64*4,
    64*4,  64*5,  64*5,  64*6,  64*6,  64*7,  64*7,  64*8,
    64*8,  64*9,  64*9,  64*10, 64*10, 64*11, 64*11, 64*12,
    64*12, 64*13, 64*13, 64*14, 64*14, 64*15, 64*15,        /* d6 0..30 */
    64*16, 64*16, 64*17, 64*17, 64*18, 64*18, 64*19, 64*19,
    64*20, 64*20, 64*21, 64*21, 64*22, 64*22, 64*23, 64*23,
    64*24, 64*24, 64*25, 64*25, 64*26, 64*26, 64*27, 64*27,
    64*28, 64*28, 64*29, 64*29, 64*30, 64*30, 64*31, 64*31, 64*31, 64*31  /* d6 31..64 */
};

/* AB3DI.s floorbright table (index 0..28 -> FloorPalScaled level 0..14). */
static const uint8_t floor_bright_level_table[29] = {
    0, 1, 1, 2, 2, 3, 3, 4, 4, 5,
    5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
    10, 11, 11, 12, 12, 13, 13, 14, 14
};

#define FLOOR_PAL_LEVEL_COUNT 15

/*
 * RenderSliceContext layout is tuned for cache efficiency:
 *   - wall_cache_cw[32] (64 B) starts at offset 0 = exactly 1 cache line
 *   - wall_cache_rgb[32] (128 B) follows at offset 64 = 2 cache lines
 *   - Per-column metadata and clip fields pack into cache line 3
 *   - Cold floor span caches (~23 KB, lazily filled) go last
 * The struct itself is cache-line-aligned so the hot arrays land on
 * cache-line boundaries for every stack / heap instance.
 */
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4324) /* structure was padded due to alignment specifier */
#endif
typedef AB3D_CACHELINE_ALIGN struct {
    /* ---- Hot: per-pixel wall shade LUT (inner loop reads) ---- */
    uint16_t wall_cache_cw[32];          /* 64 B — cache line 0 */
    uint32_t wall_cache_rgb[32];         /* 128 B — cache lines 1-2 */

    /* ---- Warm: per-column wall state ---- */
    const uint8_t *wall_cache_pal;
    const uint8_t *cur_wall_pal;
    uint16_t wall_cache_block_off;
    int16_t  wall_cache_valid;

    /* ---- Warm: clip bounds (few reads per column) ---- */
    int16_t left_clip;
    int16_t right_clip;
    int16_t top_clip;
    int16_t bot_clip;
    int16_t strip_left;
    int16_t strip_right;
    int16_t wall_top_clip;
    int16_t wall_bot_clip;
    const int16_t *foreground_floor_occlude_top;
    RendererZoneTraceFloorStats *active_floor_trace_stats;
    int16_t slice_left;
    int16_t slice_right;
    int16_t pick_zone_id;
    uint8_t pick_player_id;
    int8_t  fill_screen_water;
    int8_t  update_column_clip;
    uint8_t profile_collect_stats;
    uint8_t _profile_pad[3];
    const RendererWorldZonePrepass *zone_prepass;
    RendererWorkloadStats workload_stats;

    /* ---- Cold: floor palette span cache (lazily populated) ---- */
    const uint8_t *floor_pal_cache_src;
    uint8_t  floor_pal_cache_all_levels_ready;
    uint8_t  floor_pal_cache_valid[FLOOR_PAL_LEVEL_COUNT];
    const uint16_t *floor_pal_cw_levels[FLOOR_PAL_LEVEL_COUNT];
    const uint32_t *floor_pal_rgb_levels[FLOOR_PAL_LEVEL_COUNT];
    uint16_t floor_span_cw_cache[FLOOR_PAL_LEVEL_COUNT][256];
    uint32_t floor_span_rgb_cache[FLOOR_PAL_LEVEL_COUNT][256];
} RenderSliceContext;
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

static void render_slice_context_reset(RenderSliceContext *ctx,
                                       int16_t left, int16_t right,
                                       int16_t top, int16_t bot)
{
    ctx->left_clip = left;
    ctx->right_clip = right;
    ctx->top_clip = top;
    ctx->bot_clip = bot;
    ctx->wall_top_clip = -1;
    ctx->wall_bot_clip = -1;
    ctx->foreground_floor_occlude_top = NULL;
    ctx->active_floor_trace_stats = NULL;
    ctx->slice_left = left;
    ctx->slice_right = right;
    ctx->pick_zone_id = -1;
    ctx->pick_player_id = 0;
    ctx->cur_wall_pal = NULL;
    ctx->fill_screen_water = 0;
}

static void render_slice_context_init(RenderSliceContext *ctx,
                                      int16_t left, int16_t right,
                                      int16_t top, int16_t bot)
{
    /* Zero only the hot/warm portion (wall caches + clip fields, ~240 B).
     * The cold floor span caches (~23 KB) are lazily populated and guarded
     * by floor_pal_cache_valid[], so we just zero the validity flags and
     * the source pointer instead of memset'ing the entire struct. */
    memset(ctx->wall_cache_cw, 0, sizeof(ctx->wall_cache_cw));
    memset(ctx->wall_cache_rgb, 0, sizeof(ctx->wall_cache_rgb));
    ctx->wall_cache_pal = NULL;
    ctx->wall_cache_valid = 0;
    renderer_workload_stats_reset(&ctx->workload_stats);
    ctx->profile_collect_stats = g_renderer_profile_collect_stats ? 1u : 0u;
    ctx->zone_prepass = NULL;
    ctx->floor_pal_cache_src = NULL;
    ctx->floor_pal_cache_all_levels_ready = 0u;
    memset(ctx->floor_pal_cache_valid, 0, sizeof(ctx->floor_pal_cache_valid));
    memset(ctx->floor_pal_cw_levels, 0, sizeof(ctx->floor_pal_cw_levels));
    memset(ctx->floor_pal_rgb_levels, 0, sizeof(ctx->floor_pal_rgb_levels));
    ctx->strip_left = left;
    ctx->strip_right = right;
    render_slice_context_reset(ctx, left, right, top, bot);
    ctx->wall_cache_block_off = 0xFFFFu;
    ctx->update_column_clip = 1;
}

static inline uint8_t *renderer_active_buf(void)
{
    return g_renderer.buffer;
}

static inline uint32_t *renderer_active_rgb(void)
{
    return g_renderer.rgb_buffer;
}

static inline uint16_t *renderer_active_cw(void)
{
    return g_renderer.cw_buffer;
}

static inline uint16_t *renderer_active_pick_zone(void)
{
    if (!g_pick_capture_active) return NULL;
    return g_pick_zone_buffer;
}

static inline uint8_t *renderer_active_pick_player(void)
{
    if (!g_pick_capture_active) return NULL;
    return g_pick_player_buffer;
}

static inline uint16_t renderer_pick_zone_encode(int16_t zone_id)
{
    if (zone_id < 0) return RENDERER_PICK_ZONE_NONE;
    return (uint16_t)zone_id;
}

static void renderer_pick_clear_active_buffers(void)
{
    int w = g_renderer.width;
    int h = g_renderer.height;
    size_t count;
    if (!g_pick_zone_buffer || !g_pick_player_buffer) return;
    if (w <= 0 || h <= 0) return;
    count = (size_t)w * (size_t)h;
    memset(g_pick_zone_buffer, 0xFF, count * sizeof(uint16_t));
    memset(g_pick_player_buffer, 0, count * sizeof(uint8_t));
}

static inline void renderer_pick_mark_row_span(const RenderSliceContext *ctx,
                                               int y, int xl, int xr,
                                               uint16_t zone_value)
{
    uint16_t *pick_zone = renderer_active_pick_zone();
    uint8_t *pick_player = renderer_active_pick_player();
    int w = g_renderer.width;
    size_t row;
    if ((!pick_zone && !pick_player) || !ctx) return;
    if (w <= 0 || y < 0 || y >= g_renderer.height) return;
    if (xl < 0) xl = 0;
    if (xr >= w) xr = w - 1;
    if (xl > xr) return;
    row = (size_t)y * (size_t)w + (size_t)xl;
    if (pick_zone) {
        for (int x = xl; x <= xr; x++) pick_zone[row + (size_t)(x - xl)] = zone_value;
    }
    if (pick_player) {
        memset(pick_player + row, 0, (size_t)(xr - xl + 1));
    }
}

static inline void renderer_pick_mark_wall_column(const RenderSliceContext *ctx,
                                                  int x, int ct, int cb,
                                                  int do_ext_l, int do_ext_r,
                                                  int top_ext, int bot_ext)
{
    uint16_t *pick_zone = renderer_active_pick_zone();
    uint8_t *pick_player = renderer_active_pick_player();
    uint16_t zone_value;
    int w = g_renderer.width;
    size_t pix;
    if ((!pick_zone && !pick_player) || !ctx) return;
    if (w <= 0 || x < 0 || x >= w || ct > cb) return;

    zone_value = renderer_pick_zone_encode(ctx->pick_zone_id);
    pix = (size_t)ct * (size_t)w + (size_t)x;
    for (int y = ct; y <= cb; y++) {
        if (pick_zone) {
            pick_zone[pix] = zone_value;
            if (do_ext_l && x > 0) pick_zone[pix - 1] = zone_value;
            if (do_ext_r && x + 1 < w) pick_zone[pix + 1] = zone_value;
        }
        if (pick_player) {
            pick_player[pix] = 0;
            if (do_ext_l && x > 0) pick_player[pix - 1] = 0;
            if (do_ext_r && x + 1 < w) pick_player[pix + 1] = 0;
        }
        pix += (size_t)w;
    }

    if (top_ext && ct > 0) {
        size_t up = ((size_t)(ct - 1) * (size_t)w) + (size_t)x;
        if (pick_zone) pick_zone[up] = zone_value;
        if (pick_player) pick_player[up] = 0;
    }
    if (bot_ext && cb + 1 < g_renderer.height) {
        size_t dn = ((size_t)(cb + 1) * (size_t)w) + (size_t)x;
        if (pick_zone) pick_zone[dn] = zone_value;
        if (pick_player) pick_player[dn] = 0;
    }
}

uint32_t *renderer_get_active_rgb_target(void)
{
    return renderer_active_rgb();
}

uint16_t *renderer_get_active_cw_target(void)
{
    return renderer_active_cw();
}

void renderer_set_rgb_raster_expand(int enabled)
{
    g_renderer_rgb_raster_expand = enabled ? 1 : 0;
}

int renderer_get_rgb_raster_expand(void)
{
    return g_renderer_rgb_raster_expand;
}

#ifndef AB3D_NO_THREADS
static int8_t merge_fill_screen_water(int8_t a, int8_t b)
{
    if (a > 0 || b > 0) return 0x0F;
    if (a < 0 || b < 0) return (int8_t)-1;
    return 0;
}

static int renderer_thread_worker_main(void *userdata)
{
    RendererThreadWorker *worker = (RendererThreadWorker*)userdata;
    RendererThreadPool *pool = &g_renderer_thread_pool;
    if (!worker) return 0;

    if (worker->index < 0 || worker->index >= pool->worker_count ||
        !pool->worker_sems[worker->index] || !pool->done_sem) {
        return 0;
    }

    for (;;) {
        SDL_SemWait(pool->worker_sems[worker->index]);

        if (SDL_AtomicGet(&pool->stop)) {
            renderer_floor_fast_release_scratch();
            return 0;
        }

        const int active = (worker->index < pool->active_workers);
        const int16_t col_start = worker->col_start;
        const int16_t col_end = worker->col_end;
        const RendererThreadJobType job_type = pool->job_type;
        GameState *state = pool->job_state;
        const RendererWorldZonePrepass *world_zone_prepass = pool->world_zone_prepass;
        uint32_t frame_idx = pool->frame_idx;
        int trace_clip = pool->trace_clip;
        int8_t tint_fill_screen_water = pool->tint_fill_screen_water;
        const uint32_t *post_src_rgb = pool->post_src_rgb;
        const uint16_t *post_src_cw = pool->post_src_cw;
        uint32_t *post_dst_rgb = pool->post_dst_rgb;
        uint16_t *post_dst_cw = pool->post_dst_cw;

        int8_t fill_screen_water = 0;
        RendererWorkloadStats world_stats;
        renderer_workload_stats_reset(&world_stats);
        if (active && col_start < col_end) {
            if (job_type == RENDERER_THREAD_JOB_WORLD && state) {
                renderer_draw_world_slice(state, world_zone_prepass,
                                          col_start, col_end,
                                          frame_idx, trace_clip,
                                          &fill_screen_water, &world_stats);
                if (state->cfg_weapon_draw)
                    renderer_draw_gun_columns(state, col_start, col_end);
            } else if (job_type == RENDERER_THREAD_JOB_WATER_TINT) {
                renderer_apply_underwater_tint_slice(tint_fill_screen_water,
                                                     0, (int16_t)g_renderer.height,
                                                     col_start, col_end,
                                                     post_src_rgb, post_src_cw,
                                                     post_dst_rgb, post_dst_cw);
            }
            AB3D_NT_STORE_FENCE();
        }

        worker->fill_screen_water = fill_screen_water;
        if (worker->index >= 0 && worker->index < RENDERER_MAX_THREADS) {
            pool->worker_world_stats[worker->index] = world_stats;
        }
        if (active) {
            /* SDL_AtomicAdd returns the value before the add; last worker sees previous==1. */
            int prev = SDL_AtomicAdd(&pool->pending_workers, -1);
            if (prev == 1) {
                SDL_SemPost(pool->done_sem);
            }
        }
    }
}

static void renderer_threads_shutdown(void)
{
    RendererThreadPool *pool = &g_renderer_thread_pool;

    SDL_AtomicSet(&pool->stop, 1);
    for (int i = 0; i < pool->worker_count; i++) {
        if (pool->worker_sems[i]) {
            SDL_SemPost(pool->worker_sems[i]);
        }
    }

    for (int i = 0; i < pool->worker_count; i++) {
        if (pool->workers[i].thread) {
            SDL_WaitThread(pool->workers[i].thread, NULL);
            pool->workers[i].thread = NULL;
        }
    }

    if (pool->done_sem) {
        SDL_DestroySemaphore(pool->done_sem);
        pool->done_sem = NULL;
    }
    for (int i = 0; i < RENDERER_MAX_THREADS; i++) {
        if (pool->worker_sems[i]) {
            SDL_DestroySemaphore(pool->worker_sems[i]);
            pool->worker_sems[i] = NULL;
        }
    }

    pool->initialized = 0;
    pool->worker_count = 0;
    pool->cpu_count = 0;
    SDL_AtomicSet(&pool->pending_workers, 0);
    pool->active_workers = 0;
    SDL_AtomicSet(&pool->stop, 0);
}

static void renderer_threads_init(void)
{
    RendererThreadPool *pool = &g_renderer_thread_pool;
    memset(pool, 0, sizeof(*pool));
    pool->last_logged_strip_width = -1;
    pool->last_logged_workers = -1;

    int cpu_count = SDL_GetCPUCount();
    if (cpu_count < 1) cpu_count = 1;
#if defined(__EMSCRIPTEN__)
    {
        int hw = renderer_emscripten_hw_concurrency();
        if (hw > cpu_count) {
            printf("[RENDERER] threading: SDL_GetCPUCount()=%d, using %d from navigator.hardwareConcurrency (wasm)\n",
                   cpu_count, hw);
            cpu_count = hw;
        }
    }
#endif
    if (cpu_count > RENDERER_MAX_THREADS) cpu_count = RENDERER_MAX_THREADS;
    pool->cpu_count = cpu_count;

    if (cpu_count <= 1) {
        printf("[RENDERER] threading: CPU count=%d (single-threaded runtime)\n", cpu_count);
        pool->initialized = 1;
        return;
    }

    pool->done_sem = SDL_CreateSemaphore(0);
    if (!pool->done_sem) {
        printf("[RENDERER] threading: disabled (failed to create done semaphore)\n");
        renderer_threads_shutdown();
        return;
    }

    pool->worker_count = cpu_count;
    for (int i = 0; i < pool->worker_count; i++) {
        pool->worker_sems[i] = SDL_CreateSemaphore(0);
        if (!pool->worker_sems[i]) {
            printf("[RENDERER] threading: disabled (failed to create worker semaphore %d)\n", i);
            pool->worker_count = i;
            renderer_threads_shutdown();
            pool->initialized = 1;
            return;
        }
    }

    for (int i = 0; i < pool->worker_count; i++) {
        pool->workers[i].index = i;
        pool->workers[i].col_start = 0;
        pool->workers[i].col_end = 0;
        pool->workers[i].fill_screen_water = 0;
        renderer_workload_stats_reset(&pool->worker_world_stats[i]);
        pool->workers[i].thread = SDL_CreateThread(renderer_thread_worker_main,
                                                   "ab3d_render_worker",
                                                   &pool->workers[i]);
        if (!pool->workers[i].thread) {
            printf("[RENDERER] threading: failed to create worker %d; falling back to single-threaded\n", i);
            pool->worker_count = i;
            renderer_threads_shutdown();
            pool->initialized = 1;
            return;
        }
    }

    pool->initialized = 1;
    printf("[RENDERER] threading: initialized %d worker thread(s), cpu_count=%d\n",
           pool->worker_count, pool->cpu_count);
}

static int renderer_prepare_worker_columns(RendererThreadPool *pool, int width,
                                           int log_columns)
{
    int active_workers = pool->worker_count;
    int worker_cap = g_renderer_thread_max_workers;
    if (worker_cap <= 0) worker_cap = pool->worker_count;
    if (worker_cap > RENDERER_MAX_THREADS) worker_cap = RENDERER_MAX_THREADS;
    if (active_workers > width) active_workers = width;
    if (active_workers > worker_cap) {
        active_workers = worker_cap;
    }
    {
        int max_workers_by_cols = width / RENDERER_MIN_COLS_PER_WORKER;
        if (max_workers_by_cols < 1) max_workers_by_cols = 1;
        if (active_workers > max_workers_by_cols) active_workers = max_workers_by_cols;
    }
    if (active_workers <= 0) return 0;

    int32_t bounds[RENDERER_MAX_THREADS + 1];
    for (int i = 0; i <= active_workers; i++) {
        bounds[i] = (int32_t)(((int64_t)i * (int64_t)width) / (int64_t)active_workers);
    }

    for (int i = 0; i < active_workers; i++) {
        pool->workers[i].col_start = (int16_t)bounds[i];
        pool->workers[i].col_end = (int16_t)bounds[i + 1];
        pool->workers[i].fill_screen_water = 0;
        renderer_workload_stats_reset(&pool->worker_world_stats[i]);
    }
    for (int i = active_workers; i < pool->worker_count; i++) {
        pool->workers[i].col_start = 0;
        pool->workers[i].col_end = 0;
        pool->workers[i].fill_screen_water = 0;
        renderer_workload_stats_reset(&pool->worker_world_stats[i]);
    }

    if (log_columns && (pool->last_logged_strip_width != width || pool->last_logged_workers != active_workers)) {
        printf("[RENDERER] threading: dispatch %d column strip(s) over width=%d\n",
               active_workers, width);
        for (int i = 0; i < active_workers; i++) {
            printf("[RENDERER] threading: cols %d = [%d,%d)\n",
                   i, (int)pool->workers[i].col_start, (int)pool->workers[i].col_end);
        }
        pool->last_logged_strip_width = width;
        pool->last_logged_workers = active_workers;
    }

    return active_workers;
}

static int renderer_dispatch_threaded_world(GameState *state,
                                            const RendererWorldZonePrepass *zone_prepass,
                                            uint32_t frame_idx,
                                            int trace_clip, int8_t *out_fill_screen_water,
                                            RendererWorkloadStats *out_workload_stats)
{
    RendererThreadPool *pool = &g_renderer_thread_pool;
    g_prof_last_world_workers = 0;
    if (out_workload_stats) renderer_workload_stats_reset(out_workload_stats);
    if (!pool->initialized || pool->worker_count <= 0 || pool->cpu_count <= 1) return 0;
    if (!state) return 0;
    if (!zone_prepass) return 0;

    int width = g_renderer.width;
    if (width <= 0) return 0;

    int active_workers = renderer_prepare_worker_columns(pool, width, 1);
    if (active_workers <= 0) {
        return 0;
    }

    pool->job_state = state;
    pool->world_zone_prepass = zone_prepass;
    pool->frame_idx = frame_idx;
    pool->trace_clip = trace_clip;
    pool->job_type = RENDERER_THREAD_JOB_WORLD;
    pool->tint_fill_screen_water = 0;
    pool->post_src_rgb = NULL;
    pool->post_src_cw = NULL;
    pool->post_dst_rgb = NULL;
    pool->post_dst_cw = NULL;
    pool->active_workers = active_workers;
    SDL_AtomicSet(&pool->pending_workers, active_workers);
    SDL_AtomicAdd(&pool->job_generation, 1);
    SDL_MemoryBarrierRelease();
    for (int i = 0; i < pool->worker_count; i++) {
        SDL_SemPost(pool->worker_sems[i]);
    }

    SDL_SemWait(pool->done_sem);

    int8_t fill_screen_water = 0;
    RendererWorkloadStats merged_stats;
    renderer_workload_stats_reset(&merged_stats);
    for (int i = 0; i < active_workers; i++) {
        fill_screen_water = merge_fill_screen_water(fill_screen_water, pool->workers[i].fill_screen_water);
        renderer_workload_stats_add(&merged_stats, &pool->worker_world_stats[i]);
    }
    g_prof_last_world_workers = active_workers;

    if (out_fill_screen_water) *out_fill_screen_water = fill_screen_water;
    if (out_workload_stats) *out_workload_stats = merged_stats;
    return 1;
}

static int renderer_dispatch_threaded_underwater_tint(int8_t fill_screen_water)
{
    RendererThreadPool *pool = &g_renderer_thread_pool;
    g_prof_last_tint_workers = 0;
    if (!pool->initialized || pool->worker_count <= 0 || pool->cpu_count <= 1) return 0;
    if (fill_screen_water == 0) return 0;

    RendererState *r = &g_renderer;
    if (!r->rgb_buffer || !r->cw_buffer) return 0;

    int width = r->width;
    int height = r->height;
    if (width <= 0 || height <= 0) return 0;

    int active_workers = renderer_prepare_worker_columns(pool, width, 0);
    if (active_workers <= 0) {
        return 0;
    }

    pool->job_state = NULL;
    pool->world_zone_prepass = NULL;
    pool->frame_idx = 0;
    pool->trace_clip = 0;
    pool->job_type = RENDERER_THREAD_JOB_WATER_TINT;
    pool->tint_fill_screen_water = fill_screen_water;
    pool->post_src_rgb = r->rgb_buffer;
    pool->post_src_cw = r->cw_buffer;
    pool->post_dst_rgb = r->rgb_buffer;
    pool->post_dst_cw = r->cw_buffer;
    pool->active_workers = active_workers;
    SDL_AtomicSet(&pool->pending_workers, active_workers);
    SDL_AtomicAdd(&pool->job_generation, 1);
    SDL_MemoryBarrierRelease();
    for (int i = 0; i < pool->worker_count; i++) {
        SDL_SemPost(pool->worker_sems[i]);
    }

    SDL_SemWait(pool->done_sem);
    g_prof_last_tint_workers = active_workers;

    return 1;
}

#else
static void renderer_threads_shutdown(void) { }
static void renderer_threads_init(void) { }
#endif

/* Water animation / assets (Amiga: watertouse, wtan, wateroff, fillscrnwater). */
static uint16_t g_water_wtan = 0;
static uint8_t g_water_off = 0;
static uint8_t g_water_anim_cursor = 0;
static uint8_t g_water_src_phase = 0;
static uint16_t g_water_src_off = 0;
#define WATER_BRIGHTEN_ROW_BYTES 512u
#define WATER_BRIGHTEN_ROW_COUNT 22u
/* Interpolated per-display-frame refraction phase (smooths 50Hz step on 60Hz+ displays). */
static uint16_t g_water_wtan_draw = 0;
/* Millisecond remainder used to keep average animation speed at exactly 50Hz. */
static uint32_t g_water_ms_remainder = 0;
/* Gameplay tweak: run water animation at half speed. */
static uint32_t g_water_speed_ms_remainder = 0;
static const uint8_t *g_water_file = NULL;
static size_t g_water_file_size = 0;
static uint8_t g_water_file_level_max = 0;
static uint8_t g_water_level_phase_lut[8][64 * 64];
static int g_water_level_phase_lut_ready = 0;
static const uint8_t *g_water_brighten = NULL;
static size_t g_water_brighten_size = 0;
static uint16_t g_water_brighten_cw_lut[WATER_BRIGHTEN_ROW_COUNT][256];
static int g_water_brighten_cw_lut_ready = 0;
static const uint16_t g_water_src_offsets[8] = { 0, 2, 256, 258, 512, 514, 768, 770 };
/* Debug view: show floor/ceiling Gouraud term without distance attenuation. */
static int g_debug_floor_gouraud_only = 0;
/* Debug view: tint spill-rendered billboard pixels pink during sprite draw. */
static int g_debug_spill_visualize = 0;

static uint16_t renderer_debug_zone_color_cw(int16_t zone_id)
{
    uint32_t h = ((uint32_t)(uint16_t)zone_id * 2654435761u) ^ 0x9E3779B9u;
    uint16_t r = (uint16_t)(8u + ((h >> 0) & 7u));
    uint16_t g = (uint16_t)(8u + ((h >> 5) & 7u));
    uint16_t b = (uint16_t)(8u + ((h >> 10) & 7u));
    if (r == g && g == b) {
        b = (uint16_t)((b + 5u) & 15u);
        if (b < 4u) b = (uint16_t)(b + 4u);
    }
    return (uint16_t)((r << 8) | (g << 4) | b);
}

static void renderer_debug_spill_reset_sprite_rects(void)
{
    g_debug_spill_sprite_rect_count = 0;
}

static void renderer_debug_spill_note_sprite_rect(int left,
                                                  int top,
                                                  int right,
                                                  int bot,
                                                  int center_x,
                                                  int is_spill)
{
    int w = g_renderer.width;
    int h = g_renderer.height;
    if (!g_debug_spill_visualize) return;
    if (g_debug_spill_sprite_rect_count >= RENDERER_DEBUG_SPILL_SPRITE_RECTS_MAX) return;
    if (w <= 0 || h <= 0) return;

    if (left < 0) left = 0;
    if (top < 0) top = 0;
    if (right >= w) right = w - 1;
    if (bot >= h) bot = h - 1;
    if (center_x < 0) center_x = 0;
    if (center_x >= w) center_x = w - 1;
    if (left > right || top > bot) return;

    {
        RendererDebugSpillSpriteRect *r =
            &g_debug_spill_sprite_rects[g_debug_spill_sprite_rect_count++];
        r->left = (int16_t)left;
        r->top = (int16_t)top;
        r->right = (int16_t)right;
        r->bot = (int16_t)bot;
        r->center_x = (int16_t)center_x;
        r->is_spill = (uint8_t)(is_spill ? 1 : 0);
    }
}

static void renderer_debug_draw_rect_outline(uint16_t *cw,
                                             uint32_t *rgb,
                                             int left,
                                             int top,
                                             int right,
                                             int bot,
                                             uint16_t c12,
                                             uint32_t c32)
{
    int w = g_renderer.width;
    int h = g_renderer.height;
    if (!cw || w <= 0 || h <= 0) return;

    if (left < 0) left = 0;
    if (top < 0) top = 0;
    if (right >= w) right = w - 1;
    if (bot >= h) bot = h - 1;
    if (left > right || top > bot) return;

    for (int x = left; x <= right; x++) {
        renderer_cw_store_xy(cw, x, top, w, h, c12);
        renderer_cw_store_xy(cw, x, bot, w, h, c12);
        if (g_renderer_rgb_raster_expand && rgb) {
            rgb[(size_t)top * (size_t)w + (size_t)x] = c32;
            rgb[(size_t)bot * (size_t)w + (size_t)x] = c32;
        }
    }
    for (int y = top; y <= bot; y++) {
        renderer_cw_store_xy(cw, left, y, w, h, c12);
        renderer_cw_store_xy(cw, right, y, w, h, c12);
        if (g_renderer_rgb_raster_expand && rgb) {
            rgb[(size_t)y * (size_t)w + (size_t)left] = c32;
            rgb[(size_t)y * (size_t)w + (size_t)right] = c32;
        }
    }
}

static void renderer_debug_spill_log_zone_colors(const RendererWorldZonePrepass *zone_prepass,
                                                 uint32_t frame_idx)
{
    if (!g_debug_spill_visualize || !zone_prepass) return;

    int zone_count = zone_prepass->count;
    if (zone_count < 0) zone_count = 0;
    if (zone_count > RENDERER_MAX_ZONE_ORDER) zone_count = RENDERER_MAX_ZONE_ORDER;

    uint64_t sig = 1469598103934665603ull;
    int listed = 0;
    for (int i = zone_count - 1; i >= 0; i--) {
        if (!zone_prepass->valid[i]) continue;
        int16_t zone_id = zone_prepass->zone_ids[i];
        uint16_t c12 = renderer_debug_zone_color_cw(zone_id);
        sig ^= (uint64_t)(uint16_t)zone_id;
        sig *= 1099511628211ull;
        sig ^= (uint64_t)c12;
        sig *= 1099511628211ull;
        listed++;
    }

    if (g_debug_spill_zone_log_valid && sig == g_debug_spill_zone_log_sig)
        return;

    g_debug_spill_zone_log_sig = sig;
    g_debug_spill_zone_log_valid = 1;

    printf("[RENDER][F7] frame=%u zone_colors[%d]:",
           (unsigned)frame_idx,
           listed);
    for (int i = zone_count - 1; i >= 0; i--) {
        if (!zone_prepass->valid[i]) continue;
        int16_t zone_id = zone_prepass->zone_ids[i];
        uint16_t c12 = renderer_debug_zone_color_cw(zone_id);
        printf(" %d=#%03X", (int)zone_id, (unsigned)c12);
    }
    printf("\n");
}

static void renderer_apply_spill_visualize_debug_overlay(const RendererWorldZonePrepass *zone_prepass,
                                                         uint32_t frame_idx)
{
    if (!g_debug_spill_visualize) return;
    if (!g_pick_capture_active || !g_pick_zone_buffer) return;

    uint16_t *cw = renderer_active_cw();
    uint32_t *rgb = renderer_active_rgb();
    int w = g_renderer.width;
    int h = g_renderer.height;
    if (!cw || w <= 0 || h <= 0) return;

    renderer_debug_spill_log_zone_colors(zone_prepass, frame_idx);

    for (int y = 0; y < h; y++) {
        size_t row = (size_t)y * (size_t)w;
        for (int x = 0; x < w; x++) {
            size_t idx = row + (size_t)x;
            uint16_t zone = g_pick_zone_buffer[idx];
            if (zone == RENDERER_PICK_ZONE_NONE) continue;
            uint16_t c12 = renderer_debug_zone_color_cw((int16_t)zone);
#if AB3D_CW_COL_MAJOR
            renderer_cw_store_xy(cw, x, y, w, h, c12);
#else
            cw[idx] = c12;
#endif
            if (g_renderer_rgb_raster_expand && rgb) {
                rgb[idx] = amiga12_to_argb(c12);
            }
        }
    }

    {
        const uint16_t local_c12 = 0x0FFFu;
        const uint16_t spill_c12 = 0x0F0Fu;
        const uint32_t local_c32 = amiga12_to_argb(local_c12);
        const uint32_t spill_c32 = amiga12_to_argb(spill_c12);
        for (int i = 0; i < g_debug_spill_sprite_rect_count; i++) {
            const RendererDebugSpillSpriteRect *r = &g_debug_spill_sprite_rects[i];
            renderer_debug_draw_rect_outline(cw,
                                             rgb,
                                             (int)r->left,
                                             (int)r->top,
                                             (int)r->right,
                                             (int)r->bot,
                                             r->is_spill ? spill_c12 : local_c12,
                                             r->is_spill ? spill_c32 : local_c32);
            if (r->is_spill) {
                int cx = (int)r->center_x;
                int top = (int)r->top;
                int bot = (int)r->bot;
                const uint16_t center_c12 = 0x00FFu;
                const uint32_t center_c32 = amiga12_to_argb(center_c12);
                if (cx < (int)r->left) cx = (int)r->left;
                if (cx > (int)r->right) cx = (int)r->right;
                for (int y = top; y <= bot; y++) {
                    renderer_cw_store_xy(cw, cx, y, w, h, center_c12);
                    if (g_renderer_rgb_raster_expand && rgb) {
                        rgb[(size_t)y * (size_t)w + (size_t)cx] = center_c32;
                    }
                }
            }
        }
    }
}

/* AB3DI.s DrawDisplay:
 *   watertouse = *waterpt++;
 *   if (waterpt == end) waterpt = start;
 *   wtan += 640 (mod 8192), wateroff++ (mod 64). */
static void renderer_build_water_level_phase_lut(const uint8_t *water_file, size_t water_file_size)
{
    g_water_level_phase_lut_ready = 0;
    if (!water_file || water_file_size < 65536u) return;

    for (int phase = 0; phase < 8; phase++) {
        size_t phase_off = (size_t)g_water_src_offsets[phase];
        for (int v = 0; v < 64; v++) {
            for (int u = 0; u < 64; u++) {
                uint16_t d5 = (uint16_t)(((uint16_t)v << 8) | (uint16_t)u);
                size_t wi = ((size_t)d5 << 2) + phase_off;
                g_water_level_phase_lut[phase][(v << 6) | u] = (wi < water_file_size) ? water_file[wi] : 0;
            }
        }
    }

    g_water_level_phase_lut_ready = 1;
}

static void renderer_build_water_brighten_cw_lut(const uint8_t *water_brighten, size_t water_brighten_size)
{
    g_water_brighten_cw_lut_ready = 0;
    if (!water_brighten || water_brighten_size < (WATER_BRIGHTEN_ROW_COUNT * WATER_BRIGHTEN_ROW_BYTES)) return;

    for (size_t row = 0; row < WATER_BRIGHTEN_ROW_COUNT; row++) {
        const uint8_t *src = water_brighten + row * WATER_BRIGHTEN_ROW_BYTES;
        for (int sample = 0; sample < 256; sample++) {
            size_t idx = (size_t)sample * 2u;
            g_water_brighten_cw_lut[row][sample] = (uint16_t)((src[idx] << 8) | src[idx + 1u]);
        }
    }

    g_water_brighten_cw_lut_ready = 1;
}

static void renderer_advance_water_anim(void)
{
    g_water_src_phase = (uint8_t)(g_water_anim_cursor & 7u);
    g_water_src_off = g_water_src_offsets[g_water_src_phase];
    g_water_anim_cursor = (uint8_t)((g_water_src_phase + 1u) & 7u);
    g_water_wtan = (uint16_t)((g_water_wtan + 640u) & 8191u);
    g_water_off = (uint8_t)((g_water_off + 1u) & 63u);
}

void renderer_step_water_anim(int steps)
{
    if (steps <= 0) return;
    if (steps > 64) steps = 64;
    while (steps-- > 0) {
        renderer_advance_water_anim();
    }
    g_water_wtan_draw = g_water_wtan;
}

void renderer_step_water_anim_ms(uint32_t elapsed_ms)
{
    if (elapsed_ms == 0u) return;

    /* Slow water movement to 50%: convert real elapsed ms to simulation ms / 2. */
    g_water_speed_ms_remainder += elapsed_ms;
    elapsed_ms = g_water_speed_ms_remainder / 2u;
    g_water_speed_ms_remainder %= 2u;
    if (elapsed_ms == 0u) return;

    g_water_ms_remainder += elapsed_ms;
    if (g_water_ms_remainder > 2000u) g_water_ms_remainder = 2000u;

    /* Amiga cadence: one water step per 20 ms (50Hz). */
    int steps = (int)(g_water_ms_remainder / 20u);
    g_water_ms_remainder %= 20u;
    if (steps > 0) {
        renderer_step_water_anim(steps);
    }

    /* Interpolate within the next 20 ms step for smoother temporal motion. */
    {
        uint32_t frac_num = g_water_ms_remainder; /* 0..19 */
        uint32_t interp = ((uint32_t)640u * frac_num) / 20u;
        g_water_wtan_draw = (uint16_t)((g_water_wtan + interp) & 8191u);
    }
}

int renderer_toggle_floor_gouraud_debug_view(void)
{
    g_debug_floor_gouraud_only = g_debug_floor_gouraud_only ? 0 : 1;
    printf("[RENDER] Floor Gouraud-only visualization: %s (F6)\n",
           g_debug_floor_gouraud_only ? "ON" : "OFF");
    return g_debug_floor_gouraud_only;
}

int renderer_get_floor_gouraud_debug_view(void)
{
    return g_debug_floor_gouraud_only;
}

int renderer_toggle_spill_visualize_debug_view(void)
{
    g_debug_spill_visualize = g_debug_spill_visualize ? 0 : 1;
    renderer_debug_spill_reset_sprite_rects();
    g_debug_spill_zone_log_valid = 0;
    g_debug_spill_zone_log_sig = 0;
    printf("[RENDER] Spill visualization: %s (F7, zones solid + sprite outlines)\n",
           g_debug_spill_visualize ? "ON" : "OFF");
    return g_debug_spill_visualize;
}

int renderer_get_spill_visualize_debug_view(void)
{
    return g_debug_spill_visualize;
}

void renderer_automap_adjust_scale(int delta_steps)
{
    int32_t v = g_automap_units_per_px + (int32_t)delta_steps;
    if (v < (int32_t)AUTOMAP_UNITS_PER_PX_MIN) v = (int32_t)AUTOMAP_UNITS_PER_PX_MIN;
    if (v > (int32_t)AUTOMAP_UNITS_PER_PX_MAX) v = (int32_t)AUTOMAP_UNITS_PER_PX_MAX;
    g_automap_units_per_px = v;
}

/* -----------------------------------------------------------------------
 * Convert a 12-bit Amiga color word (0x0RGB) to ARGB8888.
 *
 * Only 4096 distinct input values exist, so we pre-build a lookup table
 * in renderer_init() and reduce every call site to a single array read.
 * ----------------------------------------------------------------------- */
static uint32_t amiga12_lut[4096];
/* Rounded 8-bit channel -> 4-bit nibble: exactly matches (v+8)/17 clamp. */
static uint8_t byte_to_nibble_lut[256];
/* Exact 4-bit channel blend for alpha 0..255 and packed (bg<<4)|fg nibble pairs. */
static uint8_t amiga12_blend_nibble_lut[256][256];
static int amiga12_blend_nibble_lut_ready = 0;
/* Full RGB24 -> Amiga12 lookup table (16,777,216 entries, ~32 MiB). */
static uint16_t *argb24_to_amiga12_lut = NULL;
static int argb24_to_amiga12_lut_ready = 0;
/* Exact floor(x/255) for x in [0, 130050] used by blend_argb. */
#define DIV255_LUT_MAX 130050u
static uint16_t div255_lut[DIV255_LUT_MAX + 1u];
static uint32_t g_render_frame_counter = 0;
static int g_renderer_zone_trace_active = 0;
static uint32_t g_renderer_zone_trace_frame_idx = 0;
static int g_renderer_zone_trace_requested = 0;
static int g_clip_trace_initialized = 0;
static int g_clip_trace_frames_left = 0;
static int g_zone_trace_initialized = 0;
static int g_zone_trace_frames_left = 0;
static int g_trace_filter_initialized = 0;
static int g_trace_filter_have_level = 0;
static int g_trace_filter_level = 0;
static int g_trace_filter_have_player_zone = 0;
static int g_trace_filter_player_zone = 0;

static void renderer_workload_stats_reset(RendererWorkloadStats *stats)
{
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
}

static void renderer_workload_stats_add(RendererWorkloadStats *dst, const RendererWorkloadStats *src)
{
    if (!dst || !src) return;
    dst->wall_segments += src->wall_segments;
    dst->wall_columns += src->wall_columns;
    dst->wall_pixels_core += src->wall_pixels_core;
    dst->wall_pixels_side_ext += src->wall_pixels_side_ext;
    dst->wall_pixels_cap_ext += src->wall_pixels_cap_ext;
    dst->ticks_wall += src->ticks_wall;
    dst->floor_spans += src->floor_spans;
    dst->floor_pixels += src->floor_pixels;
    dst->floor_fast_spans += src->floor_fast_spans;
    dst->floor_fast_pixels += src->floor_fast_pixels;
    dst->water_pixels += src->water_pixels;
    dst->ticks_floor += src->ticks_floor;
    dst->ticks_water += src->ticks_water;
    dst->sprite_calls += src->sprite_calls;
    dst->sprite_columns += src->sprite_columns;
    dst->sprite_pixels_visible += src->sprite_pixels_visible;
    dst->sprite_pixels_wall_occluded += src->sprite_pixels_wall_occluded;
    dst->sprite_pixels_spill_occluded += src->sprite_pixels_spill_occluded;
    dst->sprite_pixels_tested += src->sprite_pixels_tested;
    dst->sprite_pixels_drawn += src->sprite_pixels_drawn;
    dst->ticks_sprite += src->ticks_sprite;
}

static void renderer_workload_stats_diff(RendererWorkloadStats *dst,
                                         const RendererWorkloadStats *after,
                                         const RendererWorkloadStats *before)
{
    if (!dst) return;
    renderer_workload_stats_reset(dst);
    if (!after || !before) return;
    dst->wall_segments = after->wall_segments - before->wall_segments;
    dst->wall_columns = after->wall_columns - before->wall_columns;
    dst->wall_pixels_core = after->wall_pixels_core - before->wall_pixels_core;
    dst->wall_pixels_side_ext = after->wall_pixels_side_ext - before->wall_pixels_side_ext;
    dst->wall_pixels_cap_ext = after->wall_pixels_cap_ext - before->wall_pixels_cap_ext;
    dst->ticks_wall = after->ticks_wall - before->ticks_wall;
    dst->floor_spans = after->floor_spans - before->floor_spans;
    dst->floor_pixels = after->floor_pixels - before->floor_pixels;
    dst->floor_fast_spans = after->floor_fast_spans - before->floor_fast_spans;
    dst->floor_fast_pixels = after->floor_fast_pixels - before->floor_fast_pixels;
    dst->water_pixels = after->water_pixels - before->water_pixels;
    dst->ticks_floor = after->ticks_floor - before->ticks_floor;
    dst->ticks_water = after->ticks_water - before->ticks_water;
    dst->sprite_calls = after->sprite_calls - before->sprite_calls;
    dst->sprite_columns = after->sprite_columns - before->sprite_columns;
    dst->sprite_pixels_visible = after->sprite_pixels_visible - before->sprite_pixels_visible;
    dst->sprite_pixels_wall_occluded = after->sprite_pixels_wall_occluded - before->sprite_pixels_wall_occluded;
    dst->sprite_pixels_spill_occluded = after->sprite_pixels_spill_occluded - before->sprite_pixels_spill_occluded;
    dst->sprite_pixels_tested = after->sprite_pixels_tested - before->sprite_pixels_tested;
    dst->sprite_pixels_drawn = after->sprite_pixels_drawn - before->sprite_pixels_drawn;
    dst->ticks_sprite = after->ticks_sprite - before->ticks_sprite;
}

static uint64_t renderer_workload_estimated_writes(const RendererWorkloadStats *stats)
{
    if (!stats) return 0;
    return stats->wall_pixels_core +
           stats->wall_pixels_side_ext +
           stats->wall_pixels_cap_ext +
           stats->floor_pixels +
           stats->sprite_pixels_drawn;
}

typedef struct {
    int initialized;
    int enabled;
    uint64_t perf_freq;
    uint64_t report_interval_ticks;
    uint64_t report_window_start;
    uint64_t next_report_at;
    uint64_t frames;
    uint64_t threaded_world_frames;
    uint64_t threaded_tint_frames;
    uint64_t world_workers_total;
    uint64_t world_workers_samples;
    uint64_t tint_workers_total;
    uint64_t tint_workers_samples;
    uint64_t ticks_total;
    uint64_t ticks_setup_prepass;
    uint64_t ticks_world;
    uint64_t ticks_tint;
    uint64_t ticks_gun;
    uint64_t ticks_swap;
    uint64_t screen_pixels_total;
    uint64_t prepass_zone_total;
    uint64_t prepass_zone_valid_total;
    uint64_t prepass_clip_pixels_total;
    uint64_t fill_water_strong_frames;
    uint64_t fill_water_weak_frames;
    uint64_t slow_frames;
    double slow_threshold_ms;
    RendererWorkloadStats workload_totals;
    uint64_t worst_frame_idx;
    double worst_total_ms;
    double worst_setup_ms;
    double worst_world_ms;
    double worst_tint_ms;
    double worst_gun_ms;
    double worst_swap_ms;
    int worst_prepass_total;
    int worst_prepass_valid;
    int worst_prepass_clip_pixels;
    int worst_world_workers;
    int worst_tint_workers;
    int8_t worst_fill_screen_water;
    uint64_t worst_screen_pixels;
    RendererWorkloadStats worst_workload;
} RendererProfileState;

static RendererProfileState g_renderer_profile;

static int renderer_profile_enabled(void)
{
    if (!g_renderer_profile.initialized) {
        uint64_t now = SDL_GetPerformanceCounter();
        uint64_t freq = SDL_GetPerformanceFrequency();
        if (freq == 0) freq = 1;
        g_renderer_profile.initialized = 1;
        g_renderer_profile.perf_freq = freq;
        g_renderer_profile.report_window_start = now;

        {
            const char *env = SDL_getenv("AB3D_PROFILE_RENDER");
            /* Default off; profiling is opt-in with a positive env value. */
            if (!env || !*env) {
                g_renderer_profile.enabled = 0;
            } else {
                g_renderer_profile.enabled = (atoi(env) > 0) ? 1 : 0;
            }
        }

        {
            int report_ms = 1000;
            const char *env = SDL_getenv("AB3D_PROFILE_RENDER_EVERY_MS");
            if (env && *env) {
                int v = atoi(env);
                if (v >= 100 && v <= 30000) report_ms = v;
            }
            g_renderer_profile.report_interval_ticks = (uint64_t)(((double)freq * (double)report_ms) / 1000.0);
            if (g_renderer_profile.report_interval_ticks == 0) g_renderer_profile.report_interval_ticks = 1;
        }
        {
            double slow_ms = 12.5;
            const char *env = SDL_getenv("AB3D_PROFILE_RENDER_SLOW_MS");
            if (env && *env) {
                double v = atof(env);
                if (v >= 1.0 && v <= 1000.0) slow_ms = v;
                else if (v <= 0.0) slow_ms = 0.0;
            }
            g_renderer_profile.slow_threshold_ms = slow_ms;
        }
        g_renderer_profile.next_report_at = now + g_renderer_profile.report_interval_ticks;

        if (g_renderer_profile.enabled) {
            printf("[RPROF] enabled (interval=%llu ms, slow>=%.2f ms)\n",
                   (unsigned long long)((g_renderer_profile.report_interval_ticks * 1000ULL) / g_renderer_profile.perf_freq),
                   g_renderer_profile.slow_threshold_ms);
            fflush(stdout);
        }
    }
    return g_renderer_profile.enabled;
}

static void renderer_profile_maybe_report(uint64_t now)
{
    if (!renderer_profile_enabled()) return;
    if (now < g_renderer_profile.next_report_at) return;
    if (g_renderer_profile.frames == 0) {
        g_renderer_profile.report_window_start = now;
        g_renderer_profile.next_report_at = now + g_renderer_profile.report_interval_ticks;
        return;
    }

    {
        double freq = (double)g_renderer_profile.perf_freq;
        double frames = (double)g_renderer_profile.frames;
        double elapsed_ms = ((double)(now - g_renderer_profile.report_window_start) * 1000.0) / freq;
        double avg_total = ((double)g_renderer_profile.ticks_total * 1000.0) / (freq * frames);
        double avg_setup = ((double)g_renderer_profile.ticks_setup_prepass * 1000.0) / (freq * frames);
        double avg_world = ((double)g_renderer_profile.ticks_world * 1000.0) / (freq * frames);
        double avg_tint = ((double)g_renderer_profile.ticks_tint * 1000.0) / (freq * frames);
        double avg_gun = ((double)g_renderer_profile.ticks_gun * 1000.0) / (freq * frames);
        double avg_swap = ((double)g_renderer_profile.ticks_swap * 1000.0) / (freq * frames);
        double fps = (elapsed_ms > 0.0) ? (frames * 1000.0 / elapsed_ms) : 0.0;
        double threaded_world_pct = (100.0 * (double)g_renderer_profile.threaded_world_frames) / frames;
        double threaded_tint_pct = (100.0 * (double)g_renderer_profile.threaded_tint_frames) / frames;
        double avg_world_workers = (g_renderer_profile.world_workers_samples > 0)
            ? ((double)g_renderer_profile.world_workers_total / (double)g_renderer_profile.world_workers_samples)
            : 0.0;
        double avg_tint_workers = (g_renderer_profile.tint_workers_samples > 0)
            ? ((double)g_renderer_profile.tint_workers_total / (double)g_renderer_profile.tint_workers_samples)
            : 0.0;
        double avg_prepass_total = (double)g_renderer_profile.prepass_zone_total / frames;
        double avg_prepass_valid = (double)g_renderer_profile.prepass_zone_valid_total / frames;
        double avg_prepass_clip_px = (double)g_renderer_profile.prepass_clip_pixels_total / frames;
        uint64_t writes_total = renderer_workload_estimated_writes(&g_renderer_profile.workload_totals);
        uint64_t sprite_alpha_reject = 0;
        if (g_renderer_profile.workload_totals.sprite_pixels_tested >
            g_renderer_profile.workload_totals.sprite_pixels_drawn) {
            sprite_alpha_reject = g_renderer_profile.workload_totals.sprite_pixels_tested -
                                  g_renderer_profile.workload_totals.sprite_pixels_drawn;
        }
        double avg_writes = (double)writes_total / frames;
        double avg_world_wall = ((double)g_renderer_profile.workload_totals.ticks_wall * 1000.0) / (freq * frames);
        double avg_world_floor = ((double)g_renderer_profile.workload_totals.ticks_floor * 1000.0) / (freq * frames);
        double avg_world_sprite = ((double)g_renderer_profile.workload_totals.ticks_sprite * 1000.0) / (freq * frames);
        double avg_floor_px = (double)g_renderer_profile.workload_totals.floor_pixels / frames;
        double avg_floor_fast_px = (double)g_renderer_profile.workload_totals.floor_fast_pixels / frames;
        double avg_floor_fast_spans = (double)g_renderer_profile.workload_totals.floor_fast_spans / frames;
        double avg_wall_px = (double)(g_renderer_profile.workload_totals.wall_pixels_core +
                                      g_renderer_profile.workload_totals.wall_pixels_side_ext +
                                      g_renderer_profile.workload_totals.wall_pixels_cap_ext) / frames;
        double avg_wall_side_px = (double)g_renderer_profile.workload_totals.wall_pixels_side_ext / frames;
        double avg_wall_cap_px = (double)g_renderer_profile.workload_totals.wall_pixels_cap_ext / frames;
        double avg_wall_cols = (double)g_renderer_profile.workload_totals.wall_columns / frames;
        double avg_wall_segs = (double)g_renderer_profile.workload_totals.wall_segments / frames;
        double avg_wall_px_per_col = (g_renderer_profile.workload_totals.wall_columns > 0)
            ? ((double)(g_renderer_profile.workload_totals.wall_pixels_core +
                        g_renderer_profile.workload_totals.wall_pixels_side_ext +
                        g_renderer_profile.workload_totals.wall_pixels_cap_ext) /
               (double)g_renderer_profile.workload_totals.wall_columns)
            : 0.0;
        double avg_wall_cols_per_seg = (g_renderer_profile.workload_totals.wall_segments > 0)
            ? ((double)g_renderer_profile.workload_totals.wall_columns /
               (double)g_renderer_profile.workload_totals.wall_segments)
            : 0.0;
        double avg_wall_ext_pct = (avg_wall_px > 0.0)
            ? (100.0 * (avg_wall_side_px + avg_wall_cap_px) / avg_wall_px)
            : 0.0;
        double avg_wall_ms_per_mpx = (avg_wall_px > 0.0)
            ? (avg_world_wall * 1000000.0 / avg_wall_px)
            : 0.0;
        double avg_floor_ms_per_mpx = (avg_floor_px > 0.0)
            ? (avg_world_floor * 1000000.0 / avg_floor_px)
            : 0.0;
        double avg_floor_fast_pct = (avg_floor_px > 0.0)
            ? (100.0 * avg_floor_fast_px / avg_floor_px)
            : 0.0;
        double avg_world_ms_per_mwrite = (avg_writes > 0.0)
            ? (avg_world * 1000000.0 / avg_writes)
            : 0.0;
        double avg_overdraw = (g_renderer_profile.screen_pixels_total > 0)
            ? ((double)writes_total / (double)g_renderer_profile.screen_pixels_total)
            : 0.0;
        uint64_t worst_writes = renderer_workload_estimated_writes(&g_renderer_profile.worst_workload);
        double worst_world_wall = ((double)g_renderer_profile.worst_workload.ticks_wall * 1000.0) / freq;
        double worst_world_floor = ((double)g_renderer_profile.worst_workload.ticks_floor * 1000.0) / freq;
        double worst_world_sprite = ((double)g_renderer_profile.worst_workload.ticks_sprite * 1000.0) / freq;
        double worst_floor_px = (double)g_renderer_profile.worst_workload.floor_pixels;
        double worst_floor_fast_px = (double)g_renderer_profile.worst_workload.floor_fast_pixels;
        double worst_wall_px = (double)(g_renderer_profile.worst_workload.wall_pixels_core +
                                        g_renderer_profile.worst_workload.wall_pixels_side_ext +
                                        g_renderer_profile.worst_workload.wall_pixels_cap_ext);
        double worst_wall_px_per_col = (g_renderer_profile.worst_workload.wall_columns > 0)
            ? (worst_wall_px / (double)g_renderer_profile.worst_workload.wall_columns)
            : 0.0;
        double worst_wall_cols_per_seg = (g_renderer_profile.worst_workload.wall_segments > 0)
            ? ((double)g_renderer_profile.worst_workload.wall_columns /
               (double)g_renderer_profile.worst_workload.wall_segments)
            : 0.0;
        double worst_wall_ext_pct = (worst_wall_px > 0.0)
            ? (100.0 * (double)(g_renderer_profile.worst_workload.wall_pixels_side_ext +
                                g_renderer_profile.worst_workload.wall_pixels_cap_ext) / worst_wall_px)
            : 0.0;
        double worst_wall_ms_per_mpx = (worst_wall_px > 0.0)
            ? (worst_world_wall * 1000000.0 / worst_wall_px)
            : 0.0;
        double worst_floor_ms_per_mpx = (worst_floor_px > 0.0)
            ? (worst_world_floor * 1000000.0 / worst_floor_px)
            : 0.0;
        double worst_floor_fast_pct = (worst_floor_px > 0.0)
            ? (100.0 * worst_floor_fast_px / worst_floor_px)
            : 0.0;
        double worst_world_ms_per_mwrite = (worst_writes > 0)
            ? (g_renderer_profile.worst_world_ms * 1000000.0 / (double)worst_writes)
            : 0.0;
        double worst_overdraw = (g_renderer_profile.worst_screen_pixels > 0)
            ? ((double)worst_writes / (double)g_renderer_profile.worst_screen_pixels)
            : 0.0;
        const char *worst_water = "none";
        if (g_renderer_profile.worst_fill_screen_water > 0) worst_water = "strong";
        else if (g_renderer_profile.worst_fill_screen_water < 0) worst_water = "weak";

        printf("[RPROF] frames=%llu fps=%.1f ms(avg): total=%.3f setup=%.3f world=%.3f tint=%.3f gun=%.3f swap=%.3f threaded: world=%.0f%% tint=%.0f%% workers(avg): world=%.1f tint=%.1f\n",
               (unsigned long long)g_renderer_profile.frames,
               fps,
               avg_total, avg_setup, avg_world, avg_tint, avg_gun, avg_swap,
               threaded_world_pct, threaded_tint_pct,
               avg_world_workers, avg_tint_workers);
        printf("[RPROF] workload(avg/frame): prepass=%.1f/%.1f clip_px=%.0f writes=%.0f overdraw=%.2fx wall_px=%.0f floor_px=%.0f water_px=%.0f sprite_draw=%.0f sprite_test=%.0f sprite_wall_occ=%.0f sprite_spill_occ=%.0f sprite_alpha_drop=%.0f water_frames: strong=%llu weak=%llu slow>=%.2fms=%llu\n",
               avg_prepass_valid, avg_prepass_total, avg_prepass_clip_px,
               avg_writes, avg_overdraw,
               (double)(g_renderer_profile.workload_totals.wall_pixels_core +
                        g_renderer_profile.workload_totals.wall_pixels_side_ext +
                        g_renderer_profile.workload_totals.wall_pixels_cap_ext) / frames,
               (double)g_renderer_profile.workload_totals.floor_pixels / frames,
               (double)g_renderer_profile.workload_totals.water_pixels / frames,
               (double)g_renderer_profile.workload_totals.sprite_pixels_drawn / frames,
               (double)g_renderer_profile.workload_totals.sprite_pixels_tested / frames,
               (double)g_renderer_profile.workload_totals.sprite_pixels_wall_occluded / frames,
               (double)g_renderer_profile.workload_totals.sprite_pixels_spill_occluded / frames,
               (double)sprite_alpha_reject / frames,
               (unsigned long long)g_renderer_profile.fill_water_strong_frames,
               (unsigned long long)g_renderer_profile.fill_water_weak_frames,
               g_renderer_profile.slow_threshold_ms,
               (unsigned long long)g_renderer_profile.slow_frames);
         printf("[RPROF] world(avg): wall=%.3fms(%.0f%%) floor=%.3fms(%.0f%%) sprite=%.3fms(%.0f%%) efficiency=%.3f ms/Mwrite\n",
             avg_world_wall, (avg_world > 0.0) ? (100.0 * avg_world_wall / avg_world) : 0.0,
             avg_world_floor, (avg_world > 0.0) ? (100.0 * avg_world_floor / avg_world) : 0.0,
             avg_world_sprite, (avg_world > 0.0) ? (100.0 * avg_world_sprite / avg_world) : 0.0,
             avg_world_ms_per_mwrite);
         printf("[RPROF] wall(avg): segs=%.0f cols=%.0f px/col=%.1f cols/seg=%.1f side_px=%.0f cap_px=%.0f ext=%.1f%% efficiency=%.3f ms/Mwallpx\n",
             avg_wall_segs, avg_wall_cols, avg_wall_px_per_col, avg_wall_cols_per_seg,
             avg_wall_side_px, avg_wall_cap_px, avg_wall_ext_pct, avg_wall_ms_per_mpx);
         printf("[RPROF] floor(avg): px=%.0f fast_px=%.0f fast_spans=%.0f fast=%.0f%% efficiency=%.3f ms/Mfloorpx\n",
             avg_floor_px, avg_floor_fast_px, avg_floor_fast_spans, avg_floor_fast_pct, avg_floor_ms_per_mpx);
        if (g_renderer_profile.worst_total_ms > 0.0) {
            printf("[RPROF] worst frame=%llu total=%.3fms setup=%.3f world=%.3f tint=%.3f gun=%.3f swap=%.3f prepass=%d/%d clip_px=%d workers(world/tint)=%d/%d writes=%llu overdraw=%.2fx wall_px=%llu floor_px=%llu water_px=%llu sprite_draw=%llu sprite_test=%llu sprite_wall_occ=%llu sprite_spill_occ=%llu water=%s\n",
                   (unsigned long long)g_renderer_profile.worst_frame_idx,
                   g_renderer_profile.worst_total_ms,
                   g_renderer_profile.worst_setup_ms,
                   g_renderer_profile.worst_world_ms,
                   g_renderer_profile.worst_tint_ms,
                   g_renderer_profile.worst_gun_ms,
                   g_renderer_profile.worst_swap_ms,
                   g_renderer_profile.worst_prepass_valid,
                   g_renderer_profile.worst_prepass_total,
                   g_renderer_profile.worst_prepass_clip_pixels,
                   g_renderer_profile.worst_world_workers,
                   g_renderer_profile.worst_tint_workers,
                   (unsigned long long)worst_writes,
                   worst_overdraw,
                   (unsigned long long)(g_renderer_profile.worst_workload.wall_pixels_core +
                                        g_renderer_profile.worst_workload.wall_pixels_side_ext +
                                        g_renderer_profile.worst_workload.wall_pixels_cap_ext),
                   (unsigned long long)g_renderer_profile.worst_workload.floor_pixels,
                   (unsigned long long)g_renderer_profile.worst_workload.water_pixels,
                   (unsigned long long)g_renderer_profile.worst_workload.sprite_pixels_drawn,
                   (unsigned long long)g_renderer_profile.worst_workload.sprite_pixels_tested,
                   (unsigned long long)g_renderer_profile.worst_workload.sprite_pixels_wall_occluded,
                   (unsigned long long)g_renderer_profile.worst_workload.sprite_pixels_spill_occluded,
                   worst_water);
                 printf("[RPROF] worst world breakdown: wall=%.3fms floor=%.3fms sprite=%.3fms efficiency=%.3f ms/Mwrite\n",
                     worst_world_wall, worst_world_floor, worst_world_sprite, worst_world_ms_per_mwrite);
                     printf("[RPROF] worst wall breakdown: cols=%llu segs=%llu px/col=%.1f cols/seg=%.1f ext=%.1f%% efficiency=%.3f ms/Mwallpx\n",
                         (unsigned long long)g_renderer_profile.worst_workload.wall_columns,
                         (unsigned long long)g_renderer_profile.worst_workload.wall_segments,
                         worst_wall_px_per_col, worst_wall_cols_per_seg,
                         worst_wall_ext_pct, worst_wall_ms_per_mpx);
                    printf("[RPROF] worst floor breakdown: px=%llu fast_px=%llu fast=%.0f%% efficiency=%.3f ms/Mfloorpx\n",
                        (unsigned long long)g_renderer_profile.worst_workload.floor_pixels,
                        (unsigned long long)g_renderer_profile.worst_workload.floor_fast_pixels,
                        worst_floor_fast_pct,
                        worst_floor_ms_per_mpx);
        }
        fflush(stdout);
    }

    g_renderer_profile.report_window_start = now;
    g_renderer_profile.next_report_at = now + g_renderer_profile.report_interval_ticks;
    g_renderer_profile.frames = 0;
    g_renderer_profile.threaded_world_frames = 0;
    g_renderer_profile.threaded_tint_frames = 0;
    g_renderer_profile.world_workers_total = 0;
    g_renderer_profile.world_workers_samples = 0;
    g_renderer_profile.tint_workers_total = 0;
    g_renderer_profile.tint_workers_samples = 0;
    g_renderer_profile.ticks_total = 0;
    g_renderer_profile.ticks_setup_prepass = 0;
    g_renderer_profile.ticks_world = 0;
    g_renderer_profile.ticks_tint = 0;
    g_renderer_profile.ticks_gun = 0;
    g_renderer_profile.ticks_swap = 0;
    g_renderer_profile.screen_pixels_total = 0;
    g_renderer_profile.prepass_zone_total = 0;
    g_renderer_profile.prepass_zone_valid_total = 0;
    g_renderer_profile.prepass_clip_pixels_total = 0;
    g_renderer_profile.fill_water_strong_frames = 0;
    g_renderer_profile.fill_water_weak_frames = 0;
    g_renderer_profile.slow_frames = 0;
    renderer_workload_stats_reset(&g_renderer_profile.workload_totals);
    g_renderer_profile.worst_frame_idx = 0;
    g_renderer_profile.worst_total_ms = 0.0;
    g_renderer_profile.worst_setup_ms = 0.0;
    g_renderer_profile.worst_world_ms = 0.0;
    g_renderer_profile.worst_tint_ms = 0.0;
    g_renderer_profile.worst_gun_ms = 0.0;
    g_renderer_profile.worst_swap_ms = 0.0;
    g_renderer_profile.worst_prepass_total = 0;
    g_renderer_profile.worst_prepass_valid = 0;
    g_renderer_profile.worst_prepass_clip_pixels = 0;
    g_renderer_profile.worst_world_workers = 0;
    g_renderer_profile.worst_tint_workers = 0;
    g_renderer_profile.worst_fill_screen_water = 0;
    g_renderer_profile.worst_screen_pixels = 0;
    renderer_workload_stats_reset(&g_renderer_profile.worst_workload);
}

/* Debug helper:
 * Set AB3D_CLIP_TRACE_FRAMES=N to print portal clip windows for first N frames. */
static AB3D_ATTR_UNUSED int renderer_take_clip_trace_slot(void)
{
    if (!g_clip_trace_initialized) {
        const char *env = getenv("AB3D_CLIP_TRACE_FRAMES");
        if (env) g_clip_trace_frames_left = atoi(env);
        g_clip_trace_initialized = 1;
    }
    if (g_clip_trace_frames_left > 0) {
        g_clip_trace_frames_left--;
        return 1;
    }
    return 0;
}

static int renderer_peek_clip_trace_slot(void)
{
    if (!g_clip_trace_initialized) {
        const char *env = getenv("AB3D_CLIP_TRACE_FRAMES");
        if (env) g_clip_trace_frames_left = atoi(env);
        g_clip_trace_initialized = 1;
    }
    return (g_clip_trace_frames_left > 0) ? 1 : 0;
}

static void renderer_consume_clip_trace_slot(void)
{
    if (!g_clip_trace_initialized) {
        (void)renderer_peek_clip_trace_slot();
    }
    if (g_clip_trace_frames_left > 0) g_clip_trace_frames_left--;
}

/* Debug helper:
 * Set AB3D_ZONE_TRACE_FRAMES=N to print prepass visibility and per-zone draw deltas
 * for the first N frames. Intended for diagnosing unexpected zone visibility. */
static AB3D_ATTR_UNUSED int renderer_take_zone_trace_slot(void)
{
    if (!g_zone_trace_initialized) {
        const char *env = getenv("AB3D_ZONE_TRACE_FRAMES");
        if (env) g_zone_trace_frames_left = atoi(env);
        g_zone_trace_initialized = 1;
    }
    if (g_zone_trace_frames_left > 0) {
        g_zone_trace_frames_left--;
        return 1;
    }
    return 0;
}

static int renderer_peek_zone_trace_slot(void)
{
    if (!g_zone_trace_initialized) {
        const char *env = getenv("AB3D_ZONE_TRACE_FRAMES");
        if (env) g_zone_trace_frames_left = atoi(env);
        g_zone_trace_initialized = 1;
    }
    return (g_zone_trace_frames_left > 0) ? 1 : 0;
}

static void renderer_consume_zone_trace_slot(void)
{
    if (!g_zone_trace_initialized) {
        (void)renderer_peek_zone_trace_slot();
    }
    if (g_zone_trace_frames_left > 0) g_zone_trace_frames_left--;
}

static int renderer_profile_env_enabled(void)
{
    const char *env = SDL_getenv("AB3D_PROFILE_RENDER");
    if (!env || !*env) return 0;
    return (atoi(env) > 0) ? 1 : 0;
}

/* Optional trace-area filters shared by CLIP and ZONEVIS traces.
 * If unset, traces fire on the next eligible frame.
 * If set, traces only consume a slot when the current frame matches.
 * Values use the engine's internal indexing (same as logs / save data). */
static void renderer_trace_filter_init(void)
{
    if (!g_trace_filter_initialized) {
        const char *env = SDL_getenv("AB3D_TRACE_LEVEL");
        if (env && *env) {
            g_trace_filter_have_level = 1;
            g_trace_filter_level = atoi(env);
        }
        env = SDL_getenv("AB3D_TRACE_PLAYER_ZONE");
        if (env && *env) {
            g_trace_filter_have_player_zone = 1;
            g_trace_filter_player_zone = atoi(env);
        }
        g_trace_filter_initialized = 1;
    }
}

static int renderer_trace_filters_match(const GameState *state)
{
    const PlayerState *plr;
    renderer_trace_filter_init();
    if (!state) return 0;
    plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
    if (g_trace_filter_have_level && state->current_level != g_trace_filter_level) {
        return 0;
    }
    if (g_trace_filter_have_player_zone && plr->zone != g_trace_filter_player_zone) {
        return 0;
    }
    return 1;
}

void renderer_request_zone_trace(void)
{
    g_renderer_zone_trace_requested = 1;
}

static const uint8_t *renderer_find_lgr_entry_for_zone(const GameState *state,
                                                       int16_t zone_id,
                                                       int *out_lgr_index,
                                                       int16_t *out_clip_off)
{
    const uint8_t *lgr = state ? state->view_list_of_graph_rooms : NULL;
    int lgr_index = 0;

    if (out_lgr_index) *out_lgr_index = -1;
    if (out_clip_off) *out_clip_off = -1;
    if (!state || !lgr || zone_id < 0) return NULL;

    while (rd16(lgr) >= 0) {
        int16_t entry_zone = -1;
        if (renderer_resolve_lgr_entry_zone_id(&state->level, rd16(lgr), &entry_zone) &&
            entry_zone == zone_id) {
            if (out_lgr_index) *out_lgr_index = lgr_index;
            if (out_clip_off) *out_clip_off = rd16(lgr + 2);
            return lgr;
        }
        lgr += 8;
        lgr_index++;
    }

    return NULL;
}

static int renderer_count_lgr_entries(const GameState *state)
{
    const uint8_t *lgr = state ? state->view_list_of_graph_rooms : NULL;
    int count = 0;
    if (!lgr) return 0;
    while (rd16(lgr) >= 0) {
        count++;
        lgr += 8;
    }
    return count;
}

static void renderer_log_world_zone_prepass(const GameState *state,
                                            const RendererWorldZonePrepass *prepass,
                                            uint32_t frame_idx)
{
    const PlayerState *plr = state ? ((state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1) : NULL;
    int count = 0;
    int valid = 0;
    int lgr_count = renderer_count_lgr_entries(state);

    if (!prepass) return;
    count = prepass->count;
    if (count < 0) count = 0;
    if (count > RENDERER_MAX_ZONE_ORDER) count = RENDERER_MAX_ZONE_ORDER;
    for (int i = 0; i < count; i++) {
        if (prepass->valid[i]) valid++;
    }

        printf("[ZONEVIS][frame %u] begin level=%d player_zone=%d zone_order=%d lgr=%d prepass_valid=%d width=%d threads=%d\n",
            (unsigned)frame_idx,
            state ? (int)state->current_level : -1,
            plr ? (int)plr->zone : -1,
            count, lgr_count, valid, g_renderer.width,
            (state && state->cfg_render_threads && !g_renderer_zone_trace_active) ? 1 : 0);
    for (int i = 0; i < count; i++) {
        int16_t zone_id = prepass->zone_ids[i];
        int lgr_index = -1;
        int16_t clip_off = -1;
        int dup_of = -1;
        int clip_left = (int)prepass->left_px[i];
        int clip_right = (int)prepass->right_px[i];
        int clip_w = clip_right - clip_left;

        (void)renderer_find_lgr_entry_for_zone(state, zone_id, &lgr_index, &clip_off);
        for (int j = 0; j < i; j++) {
            if (prepass->zone_ids[j] == zone_id) {
                dup_of = j;
                break;
            }
        }

        if (!prepass->valid[i]) {
            clip_left = 0;
            clip_right = 0;
            clip_w = 0;
        }

         printf("[ZONEVIS][frame %u] prepass[%02d] zone=%d dup_of=%d lgr_idx=%d clip_off=%d valid=%d clip=[%d,%d) w=%d lower=%d[%d,%d] upper=%d[%d,%d] order=%s\n",
               (unsigned)frame_idx, i, (int)zone_id, dup_of, lgr_index, (int)clip_off,
             prepass->valid[i] ? 1 : 0, clip_left, clip_right, clip_w,
             prepass->lower_clip[i].valid ? 1 : 0,
             (int)prepass->lower_clip[i].top_clip,
             (int)prepass->lower_clip[i].bot_clip,
             prepass->upper_clip[i].valid ? 1 : 0,
             (int)prepass->upper_clip[i].top_clip,
             (int)prepass->upper_clip[i].bot_clip,
             prepass->draw_upper_first[i] ? "upper-first" : "lower-first");
    }
    fflush(stdout);
}

static void renderer_log_world_zone_draw(uint32_t frame_idx,
                                         int order_idx,
                                         int16_t zone_id,
                                         int left_clip_px,
                                         int right_clip_px,
                                         int drew_lower,
                                         int drew_upper,
                                         int upper_first,
                                         uint64_t zone_ticks,
                                         const RendererWorkloadStats *delta)
{
    double zone_ms = 0.0;
    double wall_ms = 0.0;
    double floor_ms = 0.0;
    double water_ms = 0.0;
    double dry_floor_ms = 0.0;
    double sprite_ms = 0.0;
    uint64_t writes = renderer_workload_estimated_writes(delta);
    uint64_t wall_px = 0;
    uint64_t dry_floor_px = 0;
    uint64_t freq = 0;

    if (!delta) return;
    wall_px = delta->wall_pixels_core + delta->wall_pixels_side_ext + delta->wall_pixels_cap_ext;
    freq = SDL_GetPerformanceFrequency();
    if (freq == 0) freq = 1;
    zone_ms = ((double)zone_ticks * 1000.0) / (double)freq;
    wall_ms = ((double)delta->ticks_wall * 1000.0) / (double)freq;
    floor_ms = ((double)delta->ticks_floor * 1000.0) / (double)freq;
    water_ms = ((double)delta->ticks_water * 1000.0) / (double)freq;
    dry_floor_ms = floor_ms - water_ms;
    if (dry_floor_ms < 0.0) dry_floor_ms = 0.0;
    sprite_ms = ((double)delta->ticks_sprite * 1000.0) / (double)freq;
    dry_floor_px = (delta->floor_pixels >= delta->water_pixels)
        ? (delta->floor_pixels - delta->water_pixels)
        : 0;

    const char *passes = "none";
    const char *order = "none";
    if (drew_lower && drew_upper) {
        passes = "upper+lower";
        order = upper_first ? "upper-first" : "lower-first";
    } else if (drew_upper) {
        passes = "upper";
        order = "upper-only";
    } else if (drew_lower) {
        passes = "lower";
        order = "lower-only";
    }

    printf("[ZONEVIS][frame %u] draw[%02d] zone=%d clip=[%d,%d) passes=%s order=%s ms=%.3f wall_ms=%.3f floor_ms=%.3f dry_floor_ms=%.3f water_ms=%.3f sprite_ms=%.3f writes=%llu wall_px=%llu floor_px=%llu dry_floor_px=%llu water_px=%llu sprite_draw=%llu sprite_test=%llu\n",
           (unsigned)frame_idx, order_idx, (int)zone_id,
           left_clip_px, right_clip_px,
           passes,
           order,
           zone_ms,
           wall_ms,
           floor_ms,
           dry_floor_ms,
           water_ms,
           sprite_ms,
           (unsigned long long)writes,
           (unsigned long long)wall_px,
           (unsigned long long)delta->floor_pixels,
           (unsigned long long)dry_floor_px,
           (unsigned long long)delta->water_pixels,
           (unsigned long long)delta->sprite_pixels_drawn,
           (unsigned long long)delta->sprite_pixels_tested);
}

/* Project rotated X/Z to current render pixel-space X, matching wall path math.
 * vz is in 24.8 fixed-point (ROT_Z_FRAC_BITS fractional bits). */
static inline int project_x_to_pixels(int32_t vx, int32_t vz)
{
    if (vz <= 0) return (vx >= 0) ? g_renderer.width : -g_renderer.width;
    int center_x = (g_renderer.width * 47) / 96;
    int proj_x = renderer_proj_x_scale_px();
    return (int)(((int64_t)vx * (int64_t)proj_x << ROT_Z_FRAC_BITS) / (int64_t)vz) + center_x;
}

/* Project world Y to current render pixel-space Y using nearest-pixel rounding.
 * Matching wall/floor rounding avoids 1px seams along shared edges.
 * vz is in 24.8 fixed-point (ROT_Z_FRAC_BITS fractional bits). */
static inline int project_y_to_pixels_round(int32_t vy, int32_t vz, int32_t proj_y_scale, int center_y)
{
    int64_t den = (vz > 0) ? (int64_t)vz : 1;
    int64_t num = ((int64_t)vy * (int64_t)proj_y_scale * (int64_t)RENDER_SCALE) << ROT_Z_FRAC_BITS;
    int64_t q = (num >= 0) ? ((num + den / 2) / den) : ((num - den / 2) / den);
    return (int)q + center_y;
}

typedef struct {
    int width;
    int base_w;
    int center_x;
    int16_t scaleval;
    int32_t cam_scale;
    int32_t cos_v;
    int32_t sin_v;
    uint64_t dist_num_abs;
} FloorDrawCommon;

typedef struct {
    int32_t dist;
    int32_t u_step;
    int32_t v_step;
    uint32_t u_base;
    uint32_t v_base;
    int16_t bright_term;
} FloorRowMath;

static void renderer_floor_prepare_row_recip_table(int height)
{
    int max_dist = height;
    if (max_dist < 1) max_dist = 1;
    if (max_dist > RENDER_INTERNAL_MAX_DIM) max_dist = RENDER_INTERNAL_MAX_DIM;
    if (g_floor_row_abs_recip_limit >= max_dist) return;

    for (int dist = g_floor_row_abs_recip_limit + 1; dist <= max_dist; dist++) {
        g_floor_row_abs_recip_fp[dist] = (uint32_t)((((uint64_t)1u << FLOOR_ROW_RECIP_SHIFT) + (uint64_t)dist / 2u) / (uint64_t)dist);
    }
    g_floor_row_abs_recip_limit = max_dist;
}

static inline int renderer_floor_row_dist_from_screen_y(int y, int center_y)
{
    int row_dist = y - center_y;
    if (row_dist == 0) row_dist = (y < center_y) ? -1 : 1;
    return row_dist;
}

static inline void renderer_floor_apply_scale_pair(int32_t *d1, int32_t *d2, int16_t scaleval)
{
    if (scaleval == 0) return;

    int s = (int)scaleval;
    if (s > 0) {
        if (s > 15) s = 15;
        {
            int64_t t1 = (int64_t)(*d1) << s;
            int64_t t2 = (int64_t)(*d2) << s;
            if (t1 > INT32_MAX) t1 = INT32_MAX;
            if (t1 < INT32_MIN) t1 = INT32_MIN;
            if (t2 > INT32_MAX) t2 = INT32_MAX;
            if (t2 < INT32_MIN) t2 = INT32_MIN;
            *d1 = (int32_t)t1;
            *d2 = (int32_t)t2;
        }
    } else {
        s = -s;
        if (s > 15) s = 15;
        *d1 >>= s;
        *d2 >>= s;
    }
}

static inline int32_t renderer_floor_compute_cam_scale(int16_t scaleval)
{
    int32_t cam_scale = FLOOR_CAM_UV_SCALE;
    if (scaleval != 0) {
        int s = (int)scaleval;
        if (s > 0) {
            if (s > 15) s = 15;
            {
                int64_t t = (int64_t)cam_scale << s;
                if (t > INT32_MAX) t = INT32_MAX;
                cam_scale = (int32_t)t;
            }
        } else {
            s = -s;
            if (s > 15) s = 15;
            cam_scale >>= s;
            if (cam_scale < 1) cam_scale = 1;
        }
    }
    return cam_scale;
}

static inline void renderer_floor_prepare_common(FloorDrawCommon *common,
                                                 const RendererState *rs,
                                                 int32_t floor_height,
                                                 int16_t scaleval)
{
    int32_t fh_8 = floor_height >> (WORLD_Y_FRAC_BITS - 1);
    uint64_t dist_num_abs;
    int64_t dist_num = (int64_t)fh_8 * (int64_t)rs->proj_y_scale * (int64_t)RENDER_SCALE;

    if (dist_num < 0) dist_num_abs = (uint64_t)(-dist_num);
    else dist_num_abs = (uint64_t)dist_num;

    common->width = (rs->width > 0) ? rs->width : 1;
    common->base_w = renderer_clamp_base_width(g_proj_base_width);
    common->center_x = (common->width * 47) / 96;
    common->scaleval = scaleval;
    common->cam_scale = renderer_floor_compute_cam_scale(scaleval);
    common->cos_v = ((int32_t)rs->cosval) << 1;
    common->sin_v = ((int32_t)rs->sinval) << 1;
    common->dist_num_abs = dist_num_abs;
}

static inline int32_t renderer_floor_compute_dist(const RendererState *rs,
                                                  const FloorDrawCommon *common,
                                                  int row_dist)
{
    int abs_row_dist = (row_dist < 0) ? -row_dist : row_dist;
    uint64_t dist_u64;

    if (abs_row_dist <= 3) {
        return rs->floor_uv_dist_near;
    }

    if (abs_row_dist <= g_floor_row_abs_recip_limit && g_floor_row_abs_recip_fp[abs_row_dist] != 0u) {
        uint64_t scaled = common->dist_num_abs * (uint64_t)g_floor_row_abs_recip_fp[abs_row_dist];
        dist_u64 = (scaled + ((uint64_t)1u << (FLOOR_ROW_RECIP_SHIFT - 1))) >> FLOOR_ROW_RECIP_SHIFT;
    } else {
        dist_u64 = (common->dist_num_abs + (uint64_t)abs_row_dist / 2u) / (uint64_t)abs_row_dist;
    }

    if (dist_u64 < 16u) dist_u64 = 16u;
    if (dist_u64 > (uint64_t)rs->floor_uv_dist_max) dist_u64 = (uint64_t)rs->floor_uv_dist_max;
    return (int32_t)dist_u64;
}

static inline void renderer_floor_prepare_row_math(FloorRowMath *row_math,
                                                   const RendererState *rs,
                                                   const FloorDrawCommon *common,
                                                   int row_dist)
{
    int32_t dist = renderer_floor_compute_dist(rs, common, row_dist);
    int32_t d1 = (int32_t)((int64_t)dist * (int64_t)common->cos_v);
    int32_t d2 = (int32_t)(-((int64_t)dist * (int64_t)common->sin_v));
    int32_t u_step;
    int32_t v_step;

    renderer_floor_apply_scale_pair(&d1, &d2, common->scaleval);

    if (common->width == common->base_w) {
        u_step = (int32_t)(d1 >> FLOOR_STEP_SHIFT);
        v_step = (int32_t)(d2 >> FLOOR_STEP_SHIFT);
    } else {
        int64_t den = ((int64_t)common->width << FLOOR_STEP_SHIFT);
        int64_t num_u = (int64_t)d1 * (int64_t)common->base_w;
        int64_t num_v = (int64_t)d2 * (int64_t)common->base_w;
        if (num_u >= 0) u_step = (int32_t)((num_u + den / 2) / den);
        else            u_step = (int32_t)((num_u - den / 2) / den);
        if (num_v >= 0) v_step = (int32_t)((num_v + den / 2) / den);
        else            v_step = (int32_t)((num_v - den / 2) / den);
    }

    {
        int64_t start_u64 = -(int64_t)d2 - (int64_t)common->center_x * (int64_t)u_step;
        int64_t start_v64 = (int64_t)d1 - (int64_t)common->center_x * (int64_t)v_step;
        start_u64 += (int64_t)rs->xoff * (int64_t)common->cam_scale;
        start_v64 += (int64_t)rs->zoff * (int64_t)common->cam_scale;

        row_math->dist = dist;
        row_math->u_step = u_step;
        row_math->v_step = v_step;
        row_math->u_base = (uint32_t)(int32_t)start_u64;
        row_math->v_base = (uint32_t)(int32_t)start_v64;
        row_math->bright_term = (int16_t)(5 + (dist >> 8));
    }
}

static RendererZoneSectionClip renderer_make_zone_section_clip_invalid(void)
{
    RendererZoneSectionClip clip;
    clip.top_clip = 0;
    clip.bot_clip = -1;
    clip.wall_top_clip = -1;
    clip.wall_bot_clip = -1;
    clip.valid = 0;
    return clip;
}

static RendererZoneSectionClip renderer_make_zone_section_clip_full(int height)
{
    RendererZoneSectionClip clip = renderer_make_zone_section_clip_invalid();
    if (height <= 0) return clip;
    clip.bot_clip = (int16_t)(height - 1);
    clip.valid = 1;
    return clip;
}

static void renderer_apply_zone_section_clip(RenderSliceContext *ctx,
                                             const RendererZoneSectionClip *clip)
{
    if (!ctx || !clip) return;
    ctx->top_clip = clip->top_clip;
    ctx->bot_clip = clip->bot_clip;
    ctx->wall_top_clip = clip->wall_top_clip;
    ctx->wall_bot_clip = clip->wall_bot_clip;
}

static void renderer_build_zone_section_clips(GameState *state, int16_t zone_id,
                                              RendererZoneSectionClip *out_lower,
                                              RendererZoneSectionClip *out_upper,
                                              uint8_t *out_draw_upper_first)
{
    RendererZoneSectionClip invalid = renderer_make_zone_section_clip_invalid();
    RendererZoneSectionClip full = renderer_make_zone_section_clip_full(g_renderer.height);
    int h = g_renderer.height;

    if (out_lower) *out_lower = invalid;
    if (out_upper) *out_upper = invalid;
    if (out_draw_upper_first) *out_draw_upper_first = 0;

    if (!out_lower || !out_upper || h <= 0) return;
    if (!state || !level_zone_has_upper_layer(&state->level, zone_id)) {
        *out_lower = full;
        return;
    }

    if (!state->level.zone_adds || !state->level.data) {
        *out_lower = full;
        return;
    }

    {
        int zone_slots = level_zone_slot_count(&state->level);
        if (zone_id < 0 || zone_id >= zone_slots) {
            *out_lower = full;
            return;
        }
        if (state->level.num_zone_graph_entries > 0 && zone_id >= state->level.num_zone_graph_entries) {
            *out_lower = full;
            return;
        }
    }

    {
        PlayerState *plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
        int32_t zone_off = rd32(state->level.zone_adds + zone_id * 4);
        const uint8_t *zd;
        int32_t split_height;

        if (zone_off < 0) {
            *out_lower = full;
            return;
        }
        if (state->level.data_byte_count > 0 && ((size_t)zone_off + 10u > state->level.data_byte_count)) {
            *out_lower = full;
            return;
        }

        zd = state->level.data + zone_off;
        split_height = rd32(zd + ZONE_OFF_ROOF);
        if (out_draw_upper_first) *out_draw_upper_first = (plr->yoff >= split_height) ? 1u : 0u;

        /* A single flat split line in screen space is not perspective-correct
         * for two-level rooms viewed through angled/distant portals and can
         * drop valid upper geometry. Keep both passes full-room and let the
         * existing in-zone clipping paths handle final visibility. */
        *out_upper = full;
        *out_lower = full;
    }
}

/* 16.16 fixed-point screen X to pixel column: floor for left span bound, ceil for right.
 * Polygon DDA used truncation for both; trunc underestimates the right edge by up to 1px. */
static inline int renderer_fp16_x_floor_px(int64_t x_fp)
{
    return (int)(x_fp >> 16);
}

static inline int renderer_fp16_x_ceil_px(int64_t x_fp)
{
    if (x_fp <= 0)
        return (int)(x_fp >> 16);
    return (int)((x_fp + 65535LL) >> 16);
}

static inline int renderer_fp16_y_floor_px(int64_t y_fp)
{
    return (int)(y_fp >> 16);
}

static inline int renderer_fp16_y_ceil_px(int64_t y_fp)
{
    if (y_fp <= 0)
        return (int)(y_fp >> 16);
    return (int)((y_fp + 65535LL) >> 16);
}

/* Floor/ceiling edge tables are int16_t; clamp projected X before storing so
 * near-plane projection spikes cannot wrap and corrupt span bounds. */
static inline int16_t renderer_clamp_edge_x_i16(int x)
{
    if (x < INT16_MIN) return INT16_MIN;
    if (x > INT16_MAX) return INT16_MAX;
    return (int16_t)x;
}

static inline int16_t renderer_clamp_edge_y_i16(int y)
{
    if (y < INT16_MIN) return INT16_MIN;
    if (y > INT16_MAX) return INT16_MAX;
    return (int16_t)y;
}

static inline void renderer_floor_column_bounds_add_run(int16_t *col_top_tab,
                                                        int16_t *col_bot_tab,
                                                        int x,
                                                        int top_y,
                                                        int bot_y,
                                                        int *x_min_io,
                                                        int *x_max_io)
{
    if (!col_top_tab || !col_bot_tab || x < 0) return;
    if (top_y > bot_y) return;

    if (top_y < col_top_tab[x]) col_top_tab[x] = renderer_clamp_edge_y_i16(top_y);
    if (bot_y > col_bot_tab[x]) col_bot_tab[x] = renderer_clamp_edge_y_i16(bot_y);
    if (x_min_io && x < *x_min_io) *x_min_io = x;
    if (x_max_io && x > *x_max_io) *x_max_io = x;
}

static inline int renderer_floor_col_bounds_find_next(int *next_idx, int idx)
{
    while (next_idx[idx] != idx) {
        next_idx[idx] = next_idx[next_idx[idx]];
        idx = next_idx[idx];
    }
    return idx;
}

static void renderer_floor_column_bounds_add_edge(int16_t *col_top_tab,
                                                  int16_t *col_bot_tab,
                                                  int x_min_clamp,
                                                  int x_max_clamp,
                                                  int y_min_clamp,
                                                  int y_max_clamp,
                                                  int sx1,
                                                  int sy1_raw,
                                                  int sx2,
                                                  int sy2_raw,
                                                  int *x_min_io,
                                                  int *x_max_io)
{
    if (!col_top_tab || !col_bot_tab) return;

    if (sx1 == sx2) {
        int x = sx1;
        int top_y = sy1_raw;
        int bot_y = sy2_raw;
        if (x < x_min_clamp || x > x_max_clamp) return;
        if (top_y > bot_y) {
            int t = top_y;
            top_y = bot_y;
            bot_y = t;
        }
        if (top_y < y_min_clamp) top_y = y_min_clamp;
        if (bot_y > y_max_clamp) bot_y = y_max_clamp;
        renderer_floor_column_bounds_add_run(col_top_tab, col_bot_tab, x,
                                             top_y, bot_y,
                                             x_min_io, x_max_io);
        return;
    }

    {
        int col_start = (sx1 < sx2) ? sx1 : sx2;
        int col_end = (sx1 > sx2) ? sx1 : sx2;
        int64_t y_fp;
        int64_t dy_fp;

        if (col_start < x_min_clamp) col_start = x_min_clamp;
        if (col_end > x_max_clamp) col_end = x_max_clamp;
        if (col_start > col_end) return;

        if (sx1 < sx2) {
            y_fp = (int64_t)sy1_raw << 16;
            dy_fp = ((int64_t)(sy2_raw - sy1_raw) << 16) / (int64_t)(sx2 - sx1);
            y_fp += dy_fp * (int64_t)(col_start - sx1);
        } else {
            y_fp = (int64_t)sy2_raw << 16;
            dy_fp = ((int64_t)(sy1_raw - sy2_raw) << 16) / (int64_t)(sx1 - sx2);
            y_fp += dy_fp * (int64_t)(col_start - sx2);
        }

        for (int x = col_start; x <= col_end; x++) {
            int top_y = renderer_fp16_y_floor_px(y_fp);
            int bot_y = renderer_fp16_y_ceil_px(y_fp);
            if (top_y < y_min_clamp) top_y = y_min_clamp;
            if (bot_y > y_max_clamp) bot_y = y_max_clamp;
            renderer_floor_column_bounds_add_run(col_top_tab, col_bot_tab, x,
                                                 top_y, bot_y,
                                                 x_min_io, x_max_io);
            y_fp += dy_fp;
        }
    }
}

static void build_argb24_to_amiga12_lut(void)
{
    if (argb24_to_amiga12_lut_ready) return;
    if (!argb24_to_amiga12_lut) {
        size_t lut_size = (size_t)(1u << 24) * sizeof(uint16_t);
        argb24_to_amiga12_lut = (uint16_t*)malloc(lut_size);
    }
    if (!argb24_to_amiga12_lut) {
        argb24_to_amiga12_lut_ready = 0;
        return;
    }

    for (uint32_t r = 0; r < 256u; r++) {
        uint16_t r4 = (uint16_t)((uint16_t)byte_to_nibble_lut[r] << 8);
        for (uint32_t g = 0; g < 256u; g++) {
            uint16_t rg = (uint16_t)(r4 | ((uint16_t)byte_to_nibble_lut[g] << 4));
            uint32_t base = (r << 16) | (g << 8);
            for (uint32_t b = 0; b < 256u; b++) {
                argb24_to_amiga12_lut[base | b] = (uint16_t)(rg | byte_to_nibble_lut[b]);
            }
        }
    }
    argb24_to_amiga12_lut_ready = 1;
}

static void build_amiga12_lut(void)
{
    for (int i = 0; i < 4096; i++) {
        uint32_t r4 = ((unsigned)i >> 8) & 0xFu;
        uint32_t g4 = ((unsigned)i >> 4) & 0xFu;
        uint32_t b4 =  (unsigned)i       & 0xFu;
        amiga12_lut[i] = RENDER_RGB_RASTER_PIXEL((r4 * 0x11u << 16) | (g4 * 0x11u << 8) | (b4 * 0x11u));
    }
    for (int i = 0; i < 256; i++) {
        unsigned n = (unsigned)(i + 8) / 17u;
        if (n > 15u) n = 15u;
        byte_to_nibble_lut[i] = (uint8_t)n;
    }
    for (unsigned i = 0; i <= DIV255_LUT_MAX; i++) {
        div255_lut[i] = (uint16_t)(i / 255u);
    }
    amiga12_blend_nibble_lut_ready = 0;
    for (unsigned alpha = 0; alpha < 256u; alpha++) {
        unsigned inv = 255u - alpha;
        uint8_t *alpha_lut = amiga12_blend_nibble_lut[alpha];
        for (unsigned bg4 = 0; bg4 < 16u; bg4++) {
            unsigned bg8 = bg4 * 0x11u;
            for (unsigned fg4 = 0; fg4 < 16u; fg4++) {
                unsigned fg8 = fg4 * 0x11u;
                unsigned pair = (bg4 << 4) | fg4;
                alpha_lut[pair] = byte_to_nibble_lut[div255_lut[bg8 * inv + fg8 * alpha]];
            }
        }
    }
    amiga12_blend_nibble_lut_ready = 1;
    build_argb24_to_amiga12_lut();
}

static inline uint32_t amiga12_to_argb(uint16_t w)
{
    return amiga12_lut[w & 0xFFFu];
}

static inline uint16_t argb_to_amiga12(uint32_t c)
{
    if (argb24_to_amiga12_lut_ready) {
        return argb24_to_amiga12_lut[c & 0x00FFFFFFu];
    }
    uint32_t r4 = (uint32_t)byte_to_nibble_lut[(c >> 16) & 0xFFu];
    uint32_t g4 = (uint32_t)byte_to_nibble_lut[(c >> 8) & 0xFFu];
    uint32_t b4 = (uint32_t)byte_to_nibble_lut[c & 0xFFu];
    return (uint16_t)((r4 << 8) | (g4 << 4) | b4);
}

uint16_t renderer_argb_to_amiga12(uint32_t argb)
{
    return argb_to_amiga12(argb);
}

static inline uint32_t blend_argb(uint32_t bg, uint32_t fg, uint32_t alpha_fg)
{
    if (alpha_fg >= 255u) return fg;
    if (alpha_fg == 0u) return bg;
    uint32_t inv = 255u - alpha_fg;
    uint32_t br = (bg >> 16) & 0xFFu;
    uint32_t bg_g = (bg >> 8) & 0xFFu;
    uint32_t bb = bg & 0xFFu;
    uint32_t fr = (fg >> 16) & 0xFFu;
    uint32_t fg_g = (fg >> 8) & 0xFFu;
    uint32_t fb = fg & 0xFFu;
    uint32_t r = (uint32_t)div255_lut[br * inv + fr * alpha_fg];
    uint32_t g = (uint32_t)div255_lut[bg_g * inv + fg_g * alpha_fg];
    uint32_t b = (uint32_t)div255_lut[bb * inv + fb * alpha_fg];
    return RENDER_RGB_RASTER_PIXEL((r << 16) | (g << 8) | b);
}

static inline uint16_t blend_amiga12(uint16_t bg, uint16_t fg, uint32_t alpha_fg)
{
    if (alpha_fg >= 255u) return (uint16_t)(fg & 0x0FFFu);
    if (alpha_fg == 0u) return (uint16_t)(bg & 0x0FFFu);
    if (amiga12_blend_nibble_lut_ready) {
        const uint8_t *blend_lut = amiga12_blend_nibble_lut[alpha_fg & 0xFFu];
        uint16_t r4 = blend_lut[((((uint32_t)bg >> 8) & 0xFu) << 4) | (((uint32_t)fg >> 8) & 0xFu)];
        uint16_t g4 = blend_lut[((((uint32_t)bg >> 4) & 0xFu) << 4) | (((uint32_t)fg >> 4) & 0xFu)];
        uint16_t b4 = blend_lut[(((uint32_t)bg & 0xFu) << 4) | ((uint32_t)fg & 0xFu)];
        return (uint16_t)((r4 << 8) | (g4 << 4) | b4);
    }
    uint32_t inv = 255u - alpha_fg;

    uint32_t br = (((uint32_t)bg >> 8) & 0xFu) * 0x11u;
    uint32_t bg_g = (((uint32_t)bg >> 4) & 0xFu) * 0x11u;
    uint32_t bb = ((uint32_t)bg & 0xFu) * 0x11u;

    uint32_t fr = (((uint32_t)fg >> 8) & 0xFu) * 0x11u;
    uint32_t fg_g = (((uint32_t)fg >> 4) & 0xFu) * 0x11u;
    uint32_t fb = ((uint32_t)fg & 0xFu) * 0x11u;

    uint32_t r = (uint32_t)div255_lut[br * inv + fr * alpha_fg];
    uint32_t g = (uint32_t)div255_lut[bg_g * inv + fg_g * alpha_fg];
    uint32_t b = (uint32_t)div255_lut[bb * inv + fb * alpha_fg];

    uint16_t r4 = (uint16_t)byte_to_nibble_lut[r];
    uint16_t g4 = (uint16_t)byte_to_nibble_lut[g];
    uint16_t b4 = (uint16_t)byte_to_nibble_lut[b];
    return (uint16_t)((r4 << 8) | (g4 << 4) | b4);
}

static void floor_span_prepare_pal_cache(RenderSliceContext *ctx,
                                         const uint8_t *pal_lut_src, int level,
                                         const uint16_t **cw_out)
{
    if (!pal_lut_src || level < 0 || level >= FLOOR_PAL_LEVEL_COUNT) {
        *cw_out = NULL;
        return;
    }

    if (ctx->floor_pal_cache_src != pal_lut_src) {
        ctx->floor_pal_cache_src = pal_lut_src;
        ctx->floor_pal_cache_all_levels_ready = 0u;
        memset(ctx->floor_pal_cache_valid, 0, sizeof(ctx->floor_pal_cache_valid));
        memset(ctx->floor_pal_cw_levels, 0, sizeof(ctx->floor_pal_cw_levels));
        memset(ctx->floor_pal_rgb_levels, 0, sizeof(ctx->floor_pal_rgb_levels));
    }

    if (!ctx->floor_pal_cache_valid[level]) {
        const uint8_t *lut = pal_lut_src + level * 512;
        for (int ti = 0; ti < 256; ti++) {
            uint16_t cw = (uint16_t)((lut[ti * 2] << 8) | lut[ti * 2 + 1]);
            ctx->floor_span_cw_cache[level][ti] = cw;
            ctx->floor_span_rgb_cache[level][ti] = amiga12_to_argb(cw);
        }
        ctx->floor_pal_cache_valid[level] = 1;
        ctx->floor_pal_cw_levels[level] = ctx->floor_span_cw_cache[level];
        ctx->floor_pal_rgb_levels[level] = ctx->floor_span_rgb_cache[level];
    }

    *cw_out = ctx->floor_span_cw_cache[level];
}

static void floor_span_prepare_pal_cache_all_levels(RenderSliceContext *ctx,
                                                    const uint8_t *pal_lut_src)
{
    if (!ctx || !pal_lut_src) return;
    if (ctx->floor_pal_cache_src != pal_lut_src) {
        ctx->floor_pal_cache_src = pal_lut_src;
        ctx->floor_pal_cache_all_levels_ready = 0u;
        memset(ctx->floor_pal_cache_valid, 0, sizeof(ctx->floor_pal_cache_valid));
        memset(ctx->floor_pal_cw_levels, 0, sizeof(ctx->floor_pal_cw_levels));
        memset(ctx->floor_pal_rgb_levels, 0, sizeof(ctx->floor_pal_rgb_levels));
    }
    if (ctx->floor_pal_cache_all_levels_ready) return;

    for (int level = 0; level < FLOOR_PAL_LEVEL_COUNT; level++) {
        const uint16_t *cw_level = NULL;
        floor_span_prepare_pal_cache(ctx, pal_lut_src, level, &cw_level);
    }
    ctx->floor_pal_cache_all_levels_ready = 1u;
}

void renderer_set_water_assets(const uint8_t *water_file, size_t water_file_size,
                               const uint8_t *water_brighten, size_t water_brighten_size)
{
    g_water_file = water_file;
    g_water_file_size = water_file_size;
    g_water_file_level_max = 0;
    g_water_level_phase_lut_ready = 0;
    g_water_brighten_cw_lut_ready = 0;
    if (water_file && water_file_size > 0) {
        uint8_t level_max = 0;
        for (size_t i = 0; i < water_file_size; i++) {
            if (water_file[i] > level_max) level_max = water_file[i];
        }
        g_water_file_level_max = level_max;
        renderer_build_water_level_phase_lut(water_file, water_file_size);
    }
    g_water_brighten = water_brighten;
    g_water_brighten_size = water_brighten_size;
    if (g_water_file_level_max <= 9u) {
        renderer_build_water_brighten_cw_lut(water_brighten, water_brighten_size);
    }
    g_water_anim_cursor = 0;
    g_water_src_phase = 0;
    g_water_src_off = 0;
    g_water_ms_remainder = 0;
    g_water_speed_ms_remainder = 0;
    /* First DrawDisplay on Amiga immediately sets watertouse/wtan/wateroff. */
    renderer_advance_water_anim();
    g_water_wtan_draw = g_water_wtan;
}

/* Sprite brightness -> palette level mapping (from ObjDraw3.ChipRam.s objscalecols).
 * Raw brightness d6 (0..61+) maps to palette byte offset (64*level).
 * dcb.w 2,64*0 / dcb.w 4,64*1 / ... / dcb.w 20,64*14
 * Index clamped to 0..61; returns byte offset into .pal data. */
static const uint16_t obj_scale_cols[] = {
    64*0,  64*0,                                                   /* d6=0..1 */
    64*1,  64*1,  64*1,  64*1,                                     /* d6=2..5 */
    64*2,  64*2,  64*2,  64*2,                                     /* d6=6..9 */
    64*3,  64*3,  64*3,  64*3,                                     /* d6=10..13 */
    64*4,  64*4,  64*4,  64*4,                                     /* d6=14..17 */
    64*5,  64*5,  64*5,  64*5,                                     /* d6=18..21 */
    64*6,  64*6,  64*6,  64*6,                                     /* d6=22..25 */
    64*7,  64*7,  64*7,  64*7,                                     /* d6=26..29 */
    64*8,  64*8,  64*8,  64*8,                                     /* d6=30..33 */
    64*9,  64*9,  64*9,  64*9,                                     /* d6=34..37 */
    64*10, 64*10, 64*10, 64*10,                                    /* d6=38..41 */
    64*11, 64*11, 64*11, 64*11,                                    /* d6=42..45 */
    64*12, 64*12, 64*12, 64*12,                                    /* d6=46..49 */
    64*13, 64*13, 64*13, 64*13,                                    /* d6=50..53 */
    64*14, 64*14, 64*14, 64*14, 64*14, 64*14, 64*14, 64*14,       /* d6=54..61 */
    64*14, 64*14, 64*14, 64*14, 64*14, 64*14, 64*14, 64*14,
    64*14, 64*14, 64*14, 64*14
};
#define OBJ_SCALE_COLS_SIZE (sizeof(obj_scale_cols) / sizeof(obj_scale_cols[0]))
/* Gun ptr frame offsets (GUNS_FRAMES): 8 guns x 4 frames = 32 entries.
 * Each entry is byte offset into gun_ptr for that (gun, frame) column list. */
#define GUN_COLS 96
#define GUN_STRIDE (GUN_COLS * 4)
#define GUN_LINES 58
#define ROCKET_LAUNCHER_GUN_IDX 2
static const uint32_t gun_ptr_frame_offsets[32] = {
    GUN_STRIDE * 20, GUN_STRIDE * 21, GUN_STRIDE * 22, GUN_STRIDE * 23, /* gun 0 */
    GUN_STRIDE * 4,  GUN_STRIDE * 5,  GUN_STRIDE * 6,  GUN_STRIDE * 7,  /* gun 1 */
    GUN_STRIDE * 16, GUN_STRIDE * 17, GUN_STRIDE * 18, GUN_STRIDE * 19, /* gun 2 */
    GUN_STRIDE * 12, GUN_STRIDE * 13, GUN_STRIDE * 14, GUN_STRIDE * 15, /* gun 3 */
    GUN_STRIDE * 24, GUN_STRIDE * 25, GUN_STRIDE * 26, GUN_STRIDE * 27, /* gun 4 */
    0, 0, 0, 0, 0, 0, 0, 0,                                             /* guns 5,6 */
    GUN_STRIDE * 0,  GUN_STRIDE * 1,  GUN_STRIDE * 2,  GUN_STRIDE * 3,  /* gun 7 */
};

/* -----------------------------------------------------------------------
 * Initialization / Shutdown
 * ----------------------------------------------------------------------- */
static void free_buffers(void)
{
    ab3d_aligned_free(g_renderer.buffer);
    g_renderer.buffer = NULL;
    ab3d_aligned_free(g_renderer.back_buffer);
    g_renderer.back_buffer = NULL;
    ab3d_aligned_free(g_renderer.rgb_buffer);
    g_renderer.rgb_buffer = NULL;
    ab3d_aligned_free(g_renderer.rgb_back_buffer);
    g_renderer.rgb_back_buffer = NULL;
    ab3d_aligned_free(g_renderer.cw_buffer);
    g_renderer.cw_buffer = NULL;
    ab3d_aligned_free(g_renderer.cw_back_buffer);
    g_renderer.cw_back_buffer = NULL;
    ab3d_aligned_free(g_pick_zone_buffer);
    g_pick_zone_buffer = NULL;
    ab3d_aligned_free(g_pick_zone_back_buffer);
    g_pick_zone_back_buffer = NULL;
    ab3d_aligned_free(g_pick_player_buffer);
    g_pick_player_buffer = NULL;
    ab3d_aligned_free(g_pick_player_back_buffer);
    g_pick_player_back_buffer = NULL;
    ab3d_aligned_free(g_renderer.clip.top);
    g_renderer.clip.top = NULL;
    ab3d_aligned_free(g_renderer.clip.bot);
    g_renderer.clip.bot = NULL;
    ab3d_aligned_free(g_renderer.clip.z);
    g_renderer.clip.z = NULL;
    ab3d_aligned_free(g_renderer.clip.top2);
    g_renderer.clip.top2 = NULL;
    ab3d_aligned_free(g_renderer.clip.bot2);
    g_renderer.clip.bot2 = NULL;
    ab3d_aligned_free(g_renderer.clip.z2);
    g_renderer.clip.z2 = NULL;
}

static void allocate_buffers(int w, int h)
{
    g_renderer.width = w;
    g_renderer.height = h;
    g_renderer.present_width = w;
    g_renderer.present_height = h;

    size_t buf_size = (size_t)w * h;
    g_renderer.buffer = (uint8_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, buf_size, 1);
    g_renderer.back_buffer = (uint8_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, buf_size, 1);

    size_t rgb_size = buf_size * sizeof(uint32_t);
    g_renderer.rgb_buffer = (uint32_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, 1, rgb_size);
    g_renderer.rgb_back_buffer = (uint32_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, 1, rgb_size);

    size_t cw_size = buf_size * sizeof(uint16_t);
    g_renderer.cw_buffer = (uint16_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, 1, cw_size);
    g_renderer.cw_back_buffer = (uint16_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, 1, cw_size);

    g_pick_zone_buffer = (uint16_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, 1, cw_size);
    g_pick_zone_back_buffer = (uint16_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, 1, cw_size);
    g_pick_player_buffer = (uint8_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, 1, buf_size);
    g_pick_player_back_buffer = (uint8_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, 1, buf_size);
    if (g_pick_zone_buffer) memset(g_pick_zone_buffer, 0xFF, cw_size);
    if (g_pick_zone_back_buffer) memset(g_pick_zone_back_buffer, 0xFF, cw_size);

    size_t clip_size = (size_t)w * sizeof(int16_t);
    g_renderer.clip.top = (int16_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, 1, clip_size);
    g_renderer.clip.bot = (int16_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, 1, clip_size);
    g_renderer.clip.z = (int32_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, (size_t)w, sizeof(int32_t));
    g_renderer.clip.top2 = (int16_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, 1, clip_size);
    g_renderer.clip.bot2 = (int16_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, 1, clip_size);
    g_renderer.clip.z2 = (int32_t *)ab3d_aligned_calloc(AB3D_CACHE_LINE_SIZE, (size_t)w, sizeof(int32_t));

    g_renderer.top_clip = 0;
    g_renderer.bot_clip = (int16_t)(h - 1);
    g_renderer.wall_top_clip = -1;
    g_renderer.wall_bot_clip = -1;
    g_renderer.left_clip = 0;
    g_renderer.right_clip = (int16_t)w;

    /* Safe defaults before first renderer_draw_display sets y_proj_scale-based caps. */
    g_renderer.floor_uv_dist_max = 30000;
    g_renderer.floor_uv_dist_near = 32000;
}

void renderer_init(void)
{
    build_amiga12_lut();
    memset(&g_renderer, 0, sizeof(g_renderer));
    allocate_buffers(RENDER_WIDTH, RENDER_HEIGHT);
    g_automap_mutex = SDL_CreateMutex();
    if (!g_automap_mutex) {
        printf("[RENDERER] Warning: automap mutex unavailable (possible data races with threaded draw)\n");
    }
    renderer_threads_init();
    printf("[RENDERER] Initialized: %dx%d\n", g_renderer.width, g_renderer.height);
}

void renderer_resize(int w, int h)
{
    if (w < 96) w = 96;
    if (h < 80) h = 80;
    if (w > RENDER_INTERNAL_MAX_DIM) w = RENDER_INTERNAL_MAX_DIM;
    if (h > RENDER_INTERNAL_MAX_DIM) h = RENDER_INTERNAL_MAX_DIM;
    free_buffers();
    allocate_buffers(w, h);
}

void renderer_shutdown(void)
{
    renderer_threads_shutdown();
    renderer_reset_level_sky_cache_internal();
    renderer_floor_fast_release_scratch();
    if (g_automap_mutex) {
        SDL_DestroyMutex(g_automap_mutex);
        g_automap_mutex = NULL;
    }
    free_buffers();
    free(argb24_to_amiga12_lut);
    argb24_to_amiga12_lut = NULL;
    argb24_to_amiga12_lut_ready = 0;
    printf("[RENDERER] Shutdown\n");
}

/* Row templates for fast RGB clear (avoid per-pixel loops). Max width from renderer_resize. */
#define CLEAR_ROW_MAX RENDER_INTERNAL_MAX_DIM
static uint32_t s_clear_sky_row[CLEAR_ROW_MAX];
static uint16_t s_clear_sky_cw_row[CLEAR_ROW_MAX];
static int s_clear_rows_inited = 0;

static AB3D_ATTR_UNUSED void init_clear_rows(void)
{
    if (s_clear_rows_inited) return;
    for (int i = 0; i < CLEAR_ROW_MAX; i++) {
        s_clear_sky_row[i] = RENDER_RGB_CLEAR_SKY_PIXEL;
        s_clear_sky_cw_row[i] = 0x0EEEu;
    }
    s_clear_rows_inited = 1;
}

void renderer_clear(uint8_t color)
{
#if RENDER_CLEAR
    int w = g_renderer.width, h = g_renderer.height;
    if (g_renderer.buffer) {
        memset(g_renderer.buffer, color, (size_t)w * h);
    }
    if (g_renderer.rgb_buffer) {
        init_clear_rows();
        uint32_t *p = g_renderer.rgb_buffer;
        size_t row_bytes = (size_t)w * sizeof(uint32_t);
        if (w <= CLEAR_ROW_MAX) {
            for (int y = 0; y < h; y++)
                memcpy(p + (size_t)y * w, s_clear_sky_row, row_bytes);
        } else {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) p[y * w + x] = RENDER_RGB_CLEAR_SKY_PIXEL;
            }
        }
    }
    if (g_renderer.cw_buffer) {
        init_clear_rows();
        uint16_t *p = g_renderer.cw_buffer;
        size_t row_bytes = (size_t)w * sizeof(uint16_t);
        if (w <= CLEAR_ROW_MAX) {
            for (int y = 0; y < h; y++)
                memcpy(p + (size_t)y * w, s_clear_sky_cw_row, row_bytes);
        } else {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) p[y * w + x] = 0x0EEEu;
            }
        }
    }
#else
    (void)color;
#endif
}

/* Amiga Anims.s putinbackdrop: pan u0 = (ang & 8191) * 432 / 8192 into cylindrical sky map (432 px wrap). */
#define SKY_PAN_WIDTH 432
/* Standard backfile: 76 bytes per column = 38 x 16-bit Amiga color words (column-major). */
#define SKY_AMIGA_BYTES_PER_COL 76

static const uint8_t *s_sky_pixels;
static int s_sky_tex_w = 1;
static int s_sky_tex_h = 1;
static size_t s_sky_data_bytes;
static int s_sky_bytes_per_col = SKY_AMIGA_BYTES_PER_COL;
/* 1 = row-major (stride = tex_w*2 per texel row); 0 = column-major (Amiga fromback, 76 bytes/col). */
static int s_sky_row_major = 0;
/* 0 = Amiga 16-bit BE column-major (no separate palette), 1 = 8-bit row-major + LUT, 2 = procedural */
static int s_sky_mode = 2;
/* Cached sky column count: depends on width + projection (recomputed when width changes). */
static int s_cached_sky_view_cols = 0;
static int s_cached_sky_view_cols_w = -1;
static uint32_t s_sky_argb[256];
static uint16_t s_sky_cw[256];

static void sky_build_lut_default(void)
{
    for (int i = 0; i < 256; i++) {
        int r = (i * 45) / 255;
        int g = (i * 70) / 255;
        int b = 30 + (i * 225) / 255;
        uint32_t px = RENDER_RGB_RASTER_PIXEL(((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b);
        s_sky_argb[i] = px;
        s_sky_cw[i] = argb_to_amiga12(px);
    }
}

static void sky_build_lut_from_rgb768(const uint8_t *rgb_palette_768)
{
    for (int i = 0; i < 256; i++) {
        uint32_t r = rgb_palette_768[i * 3 + 0];
        uint32_t g = rgb_palette_768[i * 3 + 1];
        uint32_t b = rgb_palette_768[i * 3 + 2];
        uint32_t px = RENDER_RGB_RASTER_PIXEL((r << 16) | (g << 8) | b);
        s_sky_argb[i] = px;
        s_sky_cw[i] = argb_to_amiga12(px);
    }
}

static inline uint16_t sky_fetch_cw_mode0(int tx, int ty)
{
    if (tx < 0) tx = 0;
    if (ty < 0) ty = 0;
    if (tx >= s_sky_tex_w) tx = s_sky_tex_w - 1;
    if (ty >= s_sky_tex_h) ty = s_sky_tex_h - 1;
    size_t off;
    if (s_sky_row_major) {
        off = (size_t)ty * (size_t)s_sky_tex_w * 2u + (size_t)tx * 2u;
    } else {
        off = (size_t)tx * (size_t)s_sky_bytes_per_col + (size_t)ty * 2u;
    }
    if (off + 1u >= s_sky_data_bytes) return 0x0FFFu;
    return (uint16_t)((s_sky_pixels[off] << 8) | s_sky_pixels[off + 1u]);
}

/* Unchecked: tx, ty must be in [0,tw-1] x [0,th-1] and off in bounds (bilinear inner loop). */
static inline uint16_t sky_fetch_cw_mode0_uc(int tx, int ty)
{
    size_t off;
    if (s_sky_row_major) {
        off = (size_t)ty * (size_t)s_sky_tex_w * 2u + (size_t)tx * 2u;
    } else {
        off = (size_t)tx * (size_t)s_sky_bytes_per_col + (size_t)ty * 2u;
    }
    return (uint16_t)((s_sky_pixels[off] << 8) | s_sky_pixels[off + 1u]);
}

static AB3D_ATTR_UNUSED inline uint32_t sky_fetch_argb_mode0(int tx, int ty)
{
    return amiga12_to_argb(sky_fetch_cw_mode0(tx, ty));
}

static inline uint32_t sky_fetch_argb_mode1(int tx, int ty)
{
    if (tx < 0) tx = 0;
    if (ty < 0) ty = 0;
    if (tx >= s_sky_tex_w) tx = s_sky_tex_w - 1;
    if (ty >= s_sky_tex_h) ty = s_sky_tex_h - 1;
    size_t off = (size_t)ty * (size_t)s_sky_tex_w + (size_t)tx;
    size_t lim = (size_t)s_sky_tex_w * (size_t)s_sky_tex_h;
    if (off >= lim) return s_sky_argb[0];
    return s_sky_argb[s_sky_pixels[off]];
}

static inline uint32_t sky_lerp_chan(uint32_t a, uint32_t b, uint32_t t16)
{
    return (uint32_t)((int32_t)a + ((((int32_t)b - (int32_t)a) * (int32_t)t16 + 32768) >> 16));
}

static inline uint32_t sky_bilinear_argb(uint32_t c00, uint32_t c10, uint32_t c01, uint32_t c11,
                                         uint32_t fx, uint32_t fy)
{
    uint32_t r00 = (c00 >> 16) & 0xFFu, g00 = (c00 >> 8) & 0xFFu, b00 = c00 & 0xFFu;
    uint32_t r10 = (c10 >> 16) & 0xFFu, g10 = (c10 >> 8) & 0xFFu, b10 = c10 & 0xFFu;
    uint32_t r01 = (c01 >> 16) & 0xFFu, g01 = (c01 >> 8) & 0xFFu, b01 = c01 & 0xFFu;
    uint32_t r11 = (c11 >> 16) & 0xFFu, g11 = (c11 >> 8) & 0xFFu, b11 = c11 & 0xFFu;

    uint32_t r0 = sky_lerp_chan(r00, r10, fx);
    uint32_t g0 = sky_lerp_chan(g00, g10, fx);
    uint32_t b0 = sky_lerp_chan(b00, b10, fx);
    uint32_t r1 = sky_lerp_chan(r01, r11, fx);
    uint32_t g1 = sky_lerp_chan(g01, g11, fx);
    uint32_t b1 = sky_lerp_chan(b01, b11, fx);

    uint32_t r = sky_lerp_chan(r0, r1, fy);
    uint32_t g = sky_lerp_chan(g0, g1, fy);
    uint32_t b = sky_lerp_chan(b0, b1, fy);
    return RENDER_RGB_RASTER_PIXEL((r << 16) | (g << 8) | b);
}

void renderer_set_sky_assets(const uint8_t *chunky_pixels, int tex_w, int tex_h, size_t data_bytes,
                              const uint8_t *rgb_palette_768)
{
    s_sky_pixels = chunky_pixels;
    s_sky_tex_w = tex_w > 0 ? tex_w : 1;
    s_sky_tex_h = tex_h > 0 ? tex_h : 1;
    s_sky_data_bytes = data_bytes;
    s_sky_mode = 2;
    s_sky_bytes_per_col = SKY_AMIGA_BYTES_PER_COL;

    if (!chunky_pixels) {
        s_sky_row_major = 0;
        sky_build_lut_default();
        return;
    }

    /* Amiga backfile (Anims.s): 32832 B = 432 cols x 38 rows x 2 bytes, column-major BE color words. */
    if (data_bytes == 32832u && tex_w == 432 && tex_h == 38) {
        s_sky_mode = 0;
        s_sky_bytes_per_col = 76;
        s_sky_row_major = 0; /* strict Amiga parity for canonical backfile */
        return;
    }

    /* Same byte count can be 432x76 8-bit row-major — only when height says so. */
    if (data_bytes == 32832u && tex_w == 432 && tex_h == 76) {
        s_sky_mode = 1;
        if (rgb_palette_768)
            sky_build_lut_from_rgb768(rgb_palette_768);
        else
            sky_build_lut_default();
        return;
    }

    /* 8-bit indexed row-major */
    if ((size_t)tex_w * (size_t)tex_h == data_bytes) {
        s_sky_mode = 1;
        if (rgb_palette_768)
            sky_build_lut_from_rgb768(rgb_palette_768);
        else
            sky_build_lut_default();
        return;
    }

    /* Non-standard Amiga column-major: width 432, even bytes per column */
    if (tex_w == 432 && data_bytes % 432u == 0u) {
        int bpc = (int)(data_bytes / 432u);
        if (bpc >= 2 && (bpc % 2) == 0) {
            int rows = bpc / 2;
            if (rows >= 1) {
                s_sky_tex_h = rows;
                s_sky_bytes_per_col = bpc;
                s_sky_mode = 0;
                return;
            }
        }
    }

    /* Unknown — procedural only (avoid wrong striping). */
    s_sky_pixels = NULL;
    s_sky_mode = 2;
    s_sky_row_major = 1;
    sky_build_lut_default();
}

static void renderer_draw_sky_pass_rows(int16_t angpos, int16_t row_start, int16_t row_end,
                                        int16_t col_start, int16_t col_end)
{
    uint8_t *buf = renderer_active_buf();
    uint32_t *rgb = renderer_active_rgb();
    uint16_t *cw = renderer_active_cw();
    if (!buf || !rgb || !cw) return;
    int w = g_renderer.width;
    int h = g_renderer.height;
    if (w < 1 || h < 1) return;
    int y0 = row_start;
    int y1 = row_end;
    if (y0 < 0) y0 = 0;
    if (y1 > h) y1 = h;
    if (y0 >= y1) return;
    int x0 = (int)col_start;
    int x1 = (int)col_end;
    if (x0 < 0) x0 = 0;
    if (x1 > w) x1 = w;
    if (x0 >= x1) return;

#if !RENDER_SKY
    (void)angpos;
#if RENDER_CLEAR
    {
        init_clear_rows();
        size_t row_bytes_rgb = (size_t)(x1 - x0) * sizeof(uint32_t);
        size_t row_bytes_cw = (size_t)(x1 - x0) * sizeof(uint16_t);
        if (w <= CLEAR_ROW_MAX) {
            for (int y = y0; y < y1; y++) {
                size_t row = (size_t)y * (size_t)w + (size_t)x0;
                memset(buf + row, 0, (size_t)(x1 - x0));
                if (g_renderer_rgb_raster_expand)
                    memcpy(rgb + row, s_clear_sky_row + (size_t)x0, row_bytes_rgb);
                memcpy(cw + row, s_clear_sky_cw_row + (size_t)x0, row_bytes_cw);
            }
        } else {
            const uint32_t below_px = RENDER_RGB_CLEAR_SKY_PIXEL;
            const uint16_t below_cw = 0x0EEEu;
            for (int y = y0; y < y1; y++) {
                size_t row = (size_t)y * (size_t)w;
                for (int x = x0; x < x1; x++) {
                    size_t p = row + (size_t)x;
                    buf[p] = 0;
                    if (g_renderer_rgb_raster_expand)
                        rgb[p] = below_px;
                    renderer_cw_store_xy(cw, x, y, w, h, below_cw);
                }
            }
        }
    }
#endif
#else

    /* Use sub-column pan precision so sky rotation does not quantize/judder at small turns. */
    const int32_t sky_pan_period_fp = SKY_PAN_WIDTH << 16;
    int32_t u0_fp = (int32_t)(((int64_t)(angpos & 8191) * ((int64_t)SKY_PAN_WIDTH << 16)) / 8192);
    /* Sky + below-band clear: each threaded worker fills only its column strip. */
    int sky_h = h;
    /* Match sky horizontal span to the renderer's actual projection FOV (cached: same for all strips). */
    int sky_view_cols;
    if (w != s_cached_sky_view_cols_w) {
        int center_x = (w * 47) / 96;
        int left_px = center_x;
        int right_px = (w - 1) - center_x;
        const double focal_px = (double)(64 * renderer_proj_x_scale_px());
        const double hfov =
            atan((double)left_px / focal_px) +
            atan((double)right_px / focal_px);
        double sky_cols_f = ((double)SKY_PAN_WIDTH * hfov) / (2.0 * 3.14159265358979323846);
        sky_view_cols = (int)(sky_cols_f + 0.5);
        if (sky_view_cols < 1) sky_view_cols = 1;
        if (sky_view_cols > SKY_PAN_WIDTH) sky_view_cols = SKY_PAN_WIDTH;
        s_cached_sky_view_cols = sky_view_cols;
        s_cached_sky_view_cols_w = w;
    } else {
        sky_view_cols = s_cached_sky_view_cols;
    }

    if (s_sky_pixels && s_sky_mode == 0) {
        int th = s_sky_tex_h;
        int tw = s_sky_tex_w;
        const int32_t tw_scale_fp = ((int32_t)tw << 16) / SKY_PAN_WIDTH;
#if SKY_BILINEAR_FILTER
        if (th > 1 && tw > 1 && w > 1 && sky_h > 1) {
            const int32_t sx_step_fp = (int32_t)(((int64_t)(sky_view_cols - 1) << 16) / (int64_t)(w - 1));
            const int64_t v_den = (int64_t)(sky_h > 1 ? sky_h - 1 : 1);
            for (int y = y0; y < y1 && y < sky_h; y++) {
                int64_t v_fp = (((int64_t)y * (th - 1)) << 16) / v_den;
                int v0 = (int)(v_fp >> 16);
                int v1 = (v0 < th - 1) ? (v0 + 1) : v0;
                uint32_t fy = (uint32_t)(v_fp & 0xFFFF);
                size_t row = (size_t)y * (size_t)w;
                int32_t sx_fp = 0;
                for (int xi = 0; xi < x0; xi++) sx_fp += sx_step_fp;
                for (int x = x0; x < x1; x++, sx_fp += sx_step_fp) {
                    int32_t pan_fp = u0_fp + sx_fp;
                    if (pan_fp >= sky_pan_period_fp) pan_fp -= sky_pan_period_fp;
                    int32_t tx_fp = (int32_t)(((int64_t)pan_fp * (int64_t)tw_scale_fp) >> 16);
                    int tx0 = (int)(tx_fp >> 16);
                    uint32_t fx = (uint32_t)(tx_fp & 0xFFFF);
                    int tx1 = tx0 + 1;
                    if (tx1 >= tw) tx1 = 0;
                    uint32_t c00 = amiga12_to_argb(sky_fetch_cw_mode0_uc(tx0, v0));
                    uint32_t c10 = amiga12_to_argb(sky_fetch_cw_mode0_uc(tx1, v0));
                    uint32_t c01 = amiga12_to_argb(sky_fetch_cw_mode0_uc(tx0, v1));
                    uint32_t c11 = amiga12_to_argb(sky_fetch_cw_mode0_uc(tx1, v1));
                    uint32_t px = sky_bilinear_argb(c00, c10, c01, c11, fx, fy);
                    size_t p = row + (size_t)x;
                    buf[p] = 0;
                    if (g_renderer_rgb_raster_expand)
                        rgb[p] = px;
                    renderer_cw_store_xy(cw, x, y, w, h, argb_to_amiga12(px));
                }
            }
        } else
#endif
        {
            /* Nearest-neighbour path (or filtering disabled). */
            int bpc = s_sky_bytes_per_col;
            size_t row_stride = (size_t)tw * 2u;
            const int64_t r_den = (int64_t)(sky_h > 1 ? sky_h - 1 : 1);
            const int32_t sx_step_fp = (int32_t)(((int64_t)sky_view_cols << 16) / (int64_t)w);
            for (int y = y0; y < y1 && y < sky_h; y++) {
                int rpix = (int)((int64_t)y * (th - 1) / r_den);
                if (rpix < 0) rpix = 0;
                if (rpix >= th) rpix = th - 1;
                size_t row = (size_t)y * (size_t)w;
                int32_t sx_fp = 0;
                for (int xi = 0; xi < x0; xi++) sx_fp += sx_step_fp;
                for (int x = x0; x < x1; x++) {
                    int32_t pan_fp = u0_fp + sx_fp;
                    if (pan_fp >= sky_pan_period_fp) pan_fp -= sky_pan_period_fp;
                    int c = (int)(pan_fp >> 16);
                    c = (int)(((int64_t)c * (int64_t)tw_scale_fp) >> 16);
                    if (c >= tw) c = tw - 1;
                    size_t off;
                    if (s_sky_row_major) off = (size_t)rpix * row_stride + (size_t)c * 2u;
                    else                 off = (size_t)c * (size_t)bpc + (size_t)rpix * 2u;
                    if (off + 2u > s_sky_data_bytes) continue;
                    uint16_t cw12 = (uint16_t)((s_sky_pixels[off] << 8) | s_sky_pixels[off + 1]);
                    size_t p = row + (size_t)x;
                    buf[p] = 0;
                    if (g_renderer_rgb_raster_expand)
                        rgb[p] = amiga12_to_argb(cw12);
                    renderer_cw_store_xy(cw, x, y, w, h, cw12);
                    sx_fp += sx_step_fp;
                }
            }
        }
    } else if (s_sky_pixels && s_sky_mode == 1) {
        int th = s_sky_tex_h;
        int tw = s_sky_tex_w;
        const int32_t tw_scale_fp = ((int32_t)tw << 16) / SKY_PAN_WIDTH;
#if SKY_BILINEAR_FILTER
        if (th > 1 && tw > 1 && w > 1 && sky_h > 1) {
            const int32_t sx_step_fp = (int32_t)(((int64_t)(sky_view_cols - 1) << 16) / (int64_t)(w - 1));
            const int64_t v_den = (int64_t)(sky_h > 1 ? sky_h - 1 : 1);
            for (int y = y0; y < y1 && y < sky_h; y++) {
                int64_t v_fp = (((int64_t)y * (th - 1)) << 16) / v_den;
                int v0 = (int)(v_fp >> 16);
                int v1 = (v0 < th - 1) ? (v0 + 1) : v0;
                uint32_t fy = (uint32_t)(v_fp & 0xFFFF);
                size_t row = (size_t)y * (size_t)w;
                int32_t sx_fp = 0;
                for (int xi = 0; xi < x0; xi++) sx_fp += sx_step_fp;
                for (int x = x0; x < x1; x++, sx_fp += sx_step_fp) {
                    int32_t pan_fp = u0_fp + sx_fp;
                    if (pan_fp >= sky_pan_period_fp) pan_fp -= sky_pan_period_fp;
                    int32_t tx_fp = (int32_t)(((int64_t)pan_fp * (int64_t)tw_scale_fp) >> 16);
                    int tx0 = (int)(tx_fp >> 16);
                    uint32_t fx = (uint32_t)(tx_fp & 0xFFFF);
                    int tx1 = tx0 + 1;
                    if (tx1 >= tw) tx1 = 0;
                    uint32_t c00 = sky_fetch_argb_mode1(tx0, v0);
                    uint32_t c10 = sky_fetch_argb_mode1(tx1, v0);
                    uint32_t c01 = sky_fetch_argb_mode1(tx0, v1);
                    uint32_t c11 = sky_fetch_argb_mode1(tx1, v1);
                    uint32_t px = sky_bilinear_argb(c00, c10, c01, c11, fx, fy);
                    size_t p = row + (size_t)x;
                    buf[p] = 0;
                    if (g_renderer_rgb_raster_expand)
                        rgb[p] = px;
                    renderer_cw_store_xy(cw, x, y, w, h, argb_to_amiga12(px));
                }
            }
        } else
#endif
        {
            /* Nearest-neighbour path (or filtering disabled). */
            const int64_t v_den = (int64_t)(sky_h > 1 ? sky_h - 1 : 1);
            const int32_t sx_step_fp = (int32_t)(((int64_t)sky_view_cols << 16) / (int64_t)w);
            for (int y = y0; y < y1 && y < sky_h; y++) {
                int v = (int)((int64_t)y * (th - 1) / v_den);
                if (v < 0) v = 0;
                if (v >= th) v = th - 1;
                size_t row = (size_t)y * (size_t)w;
                int32_t sx_fp = 0;
                for (int xi = 0; xi < x0; xi++) sx_fp += sx_step_fp;
                for (int x = x0; x < x1; x++) {
                    int32_t pan_fp = u0_fp + sx_fp;
                    if (pan_fp >= sky_pan_period_fp) pan_fp -= sky_pan_period_fp;
                    int tu = (int)(pan_fp >> 16);
                    tu = (int)(((int64_t)tu * (int64_t)tw_scale_fp) >> 16);
                    if (tu >= tw) tu = tw - 1;
                    size_t toff = (size_t)v * (size_t)tw + (size_t)tu;
                    if (toff >= (size_t)tw * (size_t)th) continue;
                    uint8_t idx = s_sky_pixels[toff];
                    size_t p = row + (size_t)x;
                    buf[p] = 0;
                    if (g_renderer_rgb_raster_expand)
                        rgb[p] = s_sky_argb[idx];
                    renderer_cw_store_xy(cw, x, y, w, h, s_sky_cw[idx]);
                    sx_fp += sx_step_fp;
                }
            }
        }
    } else {
        const int32_t sx_step_fp = (int32_t)(((int64_t)sky_view_cols << 16) / (int64_t)w);
        const int32_t shade_u_scale_fp = ((int32_t)40 << 16) / SKY_PAN_WIDTH;
        for (int y = y0; y < y1 && y < sky_h; y++) {
            int t = (y * 255) / (sky_h > 1 ? sky_h - 1 : 1);
            size_t row = (size_t)y * (size_t)w;
            int32_t sx_fp = 0;
            for (int xi = 0; xi < x0; xi++) sx_fp += sx_step_fp;
            for (int x = x0; x < x1; x++) {
                int32_t pan_fp = u0_fp + sx_fp;
                if (pan_fp >= sky_pan_period_fp) pan_fp -= sky_pan_period_fp;
                int u = (int)(pan_fp >> 16);
                int shade = t + (int)(((int64_t)u * (int64_t)shade_u_scale_fp) >> 16);
                if (shade > 255) shade = 255;
                int r = (shade * 40) / 255;
                int g = (shade * 90) / 255;
                int b = 30 + (shade * 200) / 255;
                uint32_t px = RENDER_RGB_RASTER_PIXEL(((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b);
                size_t p = row + (size_t)x;
                buf[p] = 0;
                if (g_renderer_rgb_raster_expand)
                    rgb[p] = px;
                renderer_cw_store_xy(cw, x, y, w, h, argb_to_amiga12(px));
                sx_fp += sx_step_fp;
            }
        }
    }

#if RENDER_CLEAR
    /* Below sky band: same clear as renderer_clear (empty canvas for world draw). */
    {
        init_clear_rows();
        const uint32_t below_px = RENDER_RGB_CLEAR_SKY_PIXEL;
        const uint16_t below_cw = 0x0EEEu;
        size_t row_bytes_rgb = (size_t)(x1 - x0) * sizeof(uint32_t);
        size_t row_bytes_cw = (size_t)(x1 - x0) * sizeof(uint16_t);
        if (w <= CLEAR_ROW_MAX) {
            for (int y = sky_h; y < h; y++) {
                size_t row = (size_t)y * (size_t)w + (size_t)x0;
                memset(buf + row, 0, (size_t)(x1 - x0));
                if (g_renderer_rgb_raster_expand)
                    memcpy(rgb + row, s_clear_sky_row + (size_t)x0, row_bytes_rgb);
                memcpy(cw + row, s_clear_sky_cw_row + (size_t)x0, row_bytes_cw);
            }
        } else {
            for (int y = sky_h; y < h; y++) {
                size_t row = (size_t)y * (size_t)w;
                for (int x = x0; x < x1; x++) {
                    size_t p = row + (size_t)x;
                    buf[p] = 0;
                    if (g_renderer_rgb_raster_expand)
                        rgb[p] = below_px;
                    cw[p] = below_cw;
                }
            }
        }
    }
#endif
#endif
}

void renderer_draw_sky_pass(int16_t angpos)
{
    renderer_draw_sky_pass_rows(angpos, 0, (int16_t)g_renderer.height,
                                0, (int16_t)g_renderer.width);
}

void renderer_swap(void)
{
    uint8_t *tmp = g_renderer.buffer;
    g_renderer.buffer = g_renderer.back_buffer;
    g_renderer.back_buffer = tmp;

    uint32_t *tmp2 = g_renderer.rgb_buffer;
    g_renderer.rgb_buffer = g_renderer.rgb_back_buffer;
    g_renderer.rgb_back_buffer = tmp2;

    uint16_t *tmp3 = g_renderer.cw_buffer;
    g_renderer.cw_buffer = g_renderer.cw_back_buffer;
    g_renderer.cw_back_buffer = tmp3;

    {
        uint16_t *tmp_pick_zone = g_pick_zone_buffer;
        g_pick_zone_buffer = g_pick_zone_back_buffer;
        g_pick_zone_back_buffer = tmp_pick_zone;
    }
    {
        uint8_t *tmp_pick_player = g_pick_player_buffer;
        g_pick_player_buffer = g_pick_player_back_buffer;
        g_pick_player_back_buffer = tmp_pick_player;
    }
}

const uint8_t *renderer_get_buffer(void)
{
    return g_renderer.back_buffer; /* The just-drawn frame */
}

const uint32_t *renderer_get_rgb_buffer(void)
{
    return g_renderer.rgb_back_buffer; /* The just-drawn RGB frame */
}

const uint16_t *renderer_get_cw_buffer(void)
{
    return g_renderer.cw_back_buffer; /* The just-drawn 12-bit color-word frame */
}

void renderer_request_center_pick_capture(void)
{
    g_pick_capture_armed = 1;
}

void renderer_get_center_pick(int16_t *out_zone_id, int *out_player_id)
{
    int w = g_renderer.width;
    int h = g_renderer.height;
    if (out_zone_id) *out_zone_id = -1;
    if (out_player_id) *out_player_id = 0;
    if (!g_pick_last_frame_valid) return;
    if (w <= 0 || h <= 0 || !g_pick_zone_back_buffer || !g_pick_player_back_buffer) return;

    {
        int cx = w / 2;
        int cy = h / 2;
        size_t idx = (size_t)cy * (size_t)w + (size_t)cx;
        uint16_t zone = g_pick_zone_back_buffer[idx];
        if (out_zone_id) {
            *out_zone_id = (zone == RENDERER_PICK_ZONE_NONE) ? -1 : (int16_t)zone;
        }
        if (out_player_id) {
            *out_player_id = (int)g_pick_player_back_buffer[idx];
        }
    }
}

static double renderer_percent_u64(uint64_t part, uint64_t whole)
{
    if (whole == 0) return 0.0;
    return ((double)part * 100.0) / (double)whole;
}

static void renderer_f2_pick_snapshot_clear(RendererF2PickSnapshot *snap)
{
    if (!snap) return;
    memset(snap, 0, sizeof(*snap));
}

static const char *renderer_f2_sprite_source_name(uint8_t source_type)
{
    switch (source_type) {
    case RENDERER_F2_SPRITE_SOURCE_OBJECT:
        return "obj";
    case RENDERER_F2_SPRITE_SOURCE_SHOT:
        return "shot";
    case RENDERER_F2_SPRITE_SOURCE_EXPLOSION:
        return "expl";
    default:
        return "unknown";
    }
}

static void renderer_f2_log_zone_list(const int16_t *zones, uint8_t count)
{
    if (!zones || count == 0) {
        printf("none");
        return;
    }
    for (int i = 0; i < (int)count; i++) {
        printf((i == 0) ? "%d" : ",%d", (int)zones[i]);
    }
}

void renderer_log_f2_pick_debug(const GameState *state,
                                int16_t standing_zone,
                                int16_t looking_zone)
{
    const RendererF2PickSnapshot *snap = &g_renderer_f2_pick_snapshot;
    const PlayerState *view_plr = (state && state->mode == MODE_SLAVE) ? &state->plr2 : (state ? &state->plr1 : NULL);
    int level_1_indexed = state ? ((state->current_level >= 0) ? ((int)state->current_level + 1) : -1) : -1;
    int player_x = view_plr ? (int)(view_plr->xoff >> 16) : 0;
    int player_y = view_plr ? (int)(view_plr->yoff >> 8) : 0;
    int player_z = view_plr ? (int)(view_plr->zoff >> 16) : 0;
    int player_x_raw = view_plr ? (int)view_plr->xoff : 0;
    int player_y_raw = view_plr ? (int)view_plr->yoff : 0;
    int player_z_raw = view_plr ? (int)view_plr->zoff : 0;
    int fps_hud = state ? (int)state->fps_display : 0;
    double frame_ms = snap->valid ? snap->frame_ms : 0.0;
    uint64_t wall_px = 0;
    uint64_t dry_floor_px = 0;
    uint64_t sprite_alpha_drop = 0;
    uint64_t writes = 0;
    uint64_t screen_px = 0;
    double overdraw = 0.0;
    double fps_frame = (frame_ms > 0.0) ? (1000.0 / frame_ms) : 0.0;

            printf("[DEBUG][F2] level=%d standing_zone=%d looking_zone=%d player_xyz=(%d,%d,%d) player_raw=(%d,%d,%d) fps_hud=%d fps_frame=%.1f frame_ms=%.3f\n",
            level_1_indexed,
           (int)standing_zone,
           (int)looking_zone,
            player_x,
            player_y,
            player_z,
                player_x_raw,
                player_y_raw,
                player_z_raw,
           fps_hud,
           fps_frame,
           frame_ms);

    if (!snap->valid) {
        printf("[DEBUG][F2] renderer snapshot unavailable\n");
        fflush(stdout);
        return;
    }

    wall_px = snap->workload.wall_pixels_core +
              snap->workload.wall_pixels_side_ext +
              snap->workload.wall_pixels_cap_ext;
    dry_floor_px = (snap->workload.floor_pixels >= snap->workload.water_pixels)
        ? (snap->workload.floor_pixels - snap->workload.water_pixels)
        : 0;
    sprite_alpha_drop = (snap->workload.sprite_pixels_tested >= snap->workload.sprite_pixels_drawn)
        ? (snap->workload.sprite_pixels_tested - snap->workload.sprite_pixels_drawn)
        : 0;
    writes = renderer_workload_estimated_writes(&snap->workload);
    screen_px = (uint64_t)(snap->width > 0 ? snap->width : 0) *
                (uint64_t)(snap->height > 0 ? snap->height : 0);
    overdraw = (screen_px > 0) ? ((double)writes / (double)screen_px) : 0.0;

    printf("[DEBUG][F2] frame=%u size=%dx%d zones_drawn=%d (lower_passes=%d upper_passes=%d) prepass_valid=%d prepass_total=%d\n",
           (unsigned)snap->frame_idx,
           snap->width,
           snap->height,
           snap->zones_drawn,
           snap->zones_drawn_lower,
           snap->zones_drawn_upper,
           snap->prepass_valid_zones,
           snap->prepass_total_zones);

    if (snap->zone_draw_order_count > 0) {
        printf("[DEBUG][F2] zones_draw_order[%d]:", snap->zone_draw_order_count);
        for (int i = 0; i < snap->zone_draw_order_count; i++) {
            printf((i == 0) ? " %d" : ",%d", (int)snap->zone_draw_order[i]);
        }
        printf("\n");
    } else {
        printf("[DEBUG][F2] zones_draw_order[0]: none\n");
    }

    printf("[DEBUG][F2] walls segments=%llu columns=%llu pixels(core=%llu side_ext=%llu cap_ext=%llu total=%llu)\n",
           (unsigned long long)snap->workload.wall_segments,
           (unsigned long long)snap->workload.wall_columns,
           (unsigned long long)snap->workload.wall_pixels_core,
           (unsigned long long)snap->workload.wall_pixels_side_ext,
           (unsigned long long)snap->workload.wall_pixels_cap_ext,
           (unsigned long long)wall_px);

    printf("[DEBUG][F2] billboards calls=%llu columns=%llu visible_rows=%llu tested_rows=%llu drawn_px=%llu wall_occluded_rows=%llu spill_occluded_rows=%llu alpha_drop=%llu\n",
           (unsigned long long)snap->workload.sprite_calls,
           (unsigned long long)snap->workload.sprite_columns,
           (unsigned long long)snap->workload.sprite_pixels_visible,
           (unsigned long long)snap->workload.sprite_pixels_tested,
           (unsigned long long)snap->workload.sprite_pixels_drawn,
           (unsigned long long)snap->workload.sprite_pixels_wall_occluded,
           (unsigned long long)snap->workload.sprite_pixels_spill_occluded,
           (unsigned long long)sprite_alpha_drop);

    printf("[DEBUG][F2] spill breakdown tested=%.1f%% of visible spill_occ=%.1f%% of visible wall_occ=%.1f%% of visible alpha_drop=%.1f%% of tested drawn=%.1f%% of tested (%.1f%% of visible)\n",
           renderer_percent_u64(snap->workload.sprite_pixels_tested,
                                snap->workload.sprite_pixels_visible),
           renderer_percent_u64(snap->workload.sprite_pixels_spill_occluded,
                                snap->workload.sprite_pixels_visible),
           renderer_percent_u64(snap->workload.sprite_pixels_wall_occluded,
                                snap->workload.sprite_pixels_visible),
           renderer_percent_u64(sprite_alpha_drop,
                                snap->workload.sprite_pixels_tested),
           renderer_percent_u64(snap->workload.sprite_pixels_drawn,
                                snap->workload.sprite_pixels_tested),
           renderer_percent_u64(snap->workload.sprite_pixels_drawn,
                                snap->workload.sprite_pixels_visible));

    printf("[DEBUG][F2] overdraw writes=%llu screen_px=%llu overdraw=%.2fx write_mix(wall=%.1f%% floor=%.1f%% sprite=%.1f%%) floor(dry=%llu water=%llu)\n",
           (unsigned long long)writes,
           (unsigned long long)screen_px,
           overdraw,
           renderer_percent_u64(wall_px, writes),
           renderer_percent_u64(snap->workload.floor_pixels, writes),
           renderer_percent_u64(snap->workload.sprite_pixels_drawn, writes),
           (unsigned long long)dry_floor_px,
           (unsigned long long)snap->workload.water_pixels);

    if (snap->sprite_spill_count > 0) {
        printf("[DEBUG][F2] sprite_spill_entries=%d%s\n",
               snap->sprite_spill_count,
               snap->sprite_spill_overflow ? " (truncated)" : "");
        for (int i = 0; i < snap->sprite_spill_count; i++) {
            const RendererF2SpriteSpillEntry *entry = &snap->sprite_spill_entries[i];
            printf("[DEBUG][F2] sprite[%s#%d] zone=%d adjacent[%u]={",
                   renderer_f2_sprite_source_name(entry->source_type),
                   (int)entry->source_slot,
                   (int)entry->zone_id,
                   (unsigned)entry->adjacent_count);
            renderer_f2_log_zone_list(entry->adjacent_zones, entry->adjacent_count);
            printf("} spill_zones[%u]={", (unsigned)entry->spill_zone_count);
            renderer_f2_log_zone_list(entry->spill_zones, entry->spill_zone_count);
            printf("} draws=%u spill_draws=%u local_suppressed=%u\n",
                   (unsigned)entry->draw_count,
                   (unsigned)entry->spill_draw_count,
                   (unsigned)entry->local_suppressed_count);
        }
    } else {
        printf("[DEBUG][F2] sprite_spill_entries=0\n");
    }
    fflush(stdout);
}

int renderer_get_width(void)  { return g_renderer.width; }
int renderer_get_height(void) { return g_renderer.height; }
int renderer_get_stride(void) { return g_renderer.width; }
void renderer_set_present_size(int w, int h)
{
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    g_renderer.present_width = w;
    g_renderer.present_height = h;
}

/* -----------------------------------------------------------------------
 * Pixel writing helpers
 * ----------------------------------------------------------------------- */
static AB3D_ATTR_UNUSED inline void put_pixel(uint8_t *buf, int x, int y, uint8_t color)
{
    int w = g_renderer.width, h = g_renderer.height;
    if (x >= 0 && x < w && y >= 0 && y < h) {
        buf[y * w + x] = color;
    }
}

static AB3D_ATTR_UNUSED inline void draw_vline(uint8_t *buf, int x, int y_top, int y_bot,
                               uint8_t color)
{
    int w = g_renderer.width, h = g_renderer.height;
    if (x < 0 || x >= w) return;
    if (y_top < 0) y_top = 0;
    if (y_bot >= h) y_bot = h - 1;
    for (int y = y_top; y <= y_bot; y++) {
        buf[y * w + x] = color;
    }
}

/* -----------------------------------------------------------------------
 * Rotate a single level point
 *
 * Translated from AB3DI.s RotateLevelPts (lines 4087-4166).
 *
 * For each point:
 *   dx = point.x - xoff
 *   dz = point.z - zoff
 *   rotated.x = dx * cos - dz * sin  (scaled, with xwobble added)
 *   rotated.z = dx * sin + dz * cos  (depth)
 *   on_screen.x = rotated.x / rotated.z  (perspective divide)
 * ----------------------------------------------------------------------- */
static void rotate_one_point(RendererState *r, const uint8_t *pts, int idx)
{
    int16_t sin_v = r->sinval;
    int16_t cos_v = r->cosval;
    int16_t cam_x = r->xoff;
    int16_t cam_z = r->zoff;

    int16_t px = rd16(pts + idx * 4);
    int16_t pz = rd16(pts + idx * 4 + 2);

    int16_t dx = (int16_t)(px - cam_x);
    int16_t dz = (int16_t)(pz - cam_z);

    /* Rotation (from ASM):
     * view_x = dx * cos - dz * sin    (d2 = d0*d6 - d1*d6_swapped)
     * view_z = dx * sin + dz * cos    (d1 = d0*d6_swapped + d1*d6)
     *
     * Amiga used swap (>>16) then <<7 for X, losing 7 fractional bits.
     * We keep full precision: (vx*2) >> 9 == (vx>>16)<<7 but without
     * the intermediate int16 truncation. Z is stored in 24.8 fixed-point
     * instead of integer to eliminate single-step jitter in projection. */
    int32_t vx = (int32_t)dx * cos_v - (int32_t)dz * sin_v;
    vx <<= 1;              /* add.l d2,d2 in ASM */
    int32_t vx_fine = (vx >> 9) + r->xwobble;

    int32_t vz = (int32_t)dx * sin_v + (int32_t)dz * cos_v;
    vz <<= 2;              /* asl.l #2 in ASM */

    r->rotated[idx].x = vx_fine;
    r->rotated[idx].z = vz >> (16 - ROT_Z_FRAC_BITS);

    /* Legacy on_screen: Amiga 96-column projection (kept for portal clip code). */
    int16_t vz16 = (int16_t)(vz >> 16);
    if (vz16 > 0) {
        int32_t screen_x = (vx_fine / vz16) + 47;
        r->on_screen[idx].screen_x = (int16_t)screen_x;
        r->on_screen[idx].flags = 0;
    } else {
        r->on_screen[idx].screen_x = (int16_t)(((int16_t)vx_fine > 0) ? 96 : 0);
        r->on_screen[idx].flags = 1;
    }
    r->rotated_stamp[idx] = r->rotate_stamp;
}

static void renderer_begin_rotate_frame(RendererState *r)
{
    if (!r) return;
    r->rotate_stamp++;
    if (r->rotate_stamp == 0) {
        memset(r->rotated_stamp, 0, sizeof(r->rotated_stamp));
        r->rotate_stamp = 1;
    }
}

static int renderer_ensure_level_point_rotated(GameState *state, int16_t idx)
{
    RendererState *r = &g_renderer;
    if (!state || !state->level.points) return 0;
    if (idx < 0 || idx >= MAX_POINTS) return 0;
    if (r->rotate_stamp != 0 && r->rotated_stamp[idx] == r->rotate_stamp) return 1;
    rotate_one_point(r, state->level.points, idx);
    return 1;
}

/* -----------------------------------------------------------------------
 * RotateLevelPts
 *
 * Translated from AB3DI.s RotateLevelPts (lines 4087-4166).
 *
 * Uses PointsToRotatePtr: a list of 16-bit indices into the Points array,
 * terminated by a negative value. Only the listed points are transformed.
 * This list comes from the current zone data at offset ToZonePts (34).
 * ----------------------------------------------------------------------- */
#define TO_ZONE_PTS_OFFSET 34

void renderer_rotate_level_pts(GameState *state)
{
    RendererState *r = &g_renderer;

    if (!state->level.points) return;
    const uint8_t *pts = state->level.points;
    renderer_begin_rotate_frame(r);

    /* Get the PointsToRotatePtr from the current player's zone data.
     * It's stored as an offset from the room pointer into level data. */
    PlayerState *plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;

    if (plr->roompt > 0 && state->level.data) {
        /* The zone data at roompt+34 (ToZonePts) contains a word offset.
         * Adding that to roompt gives the points-to-rotate list location. */
        const uint8_t *room = state->level.data + plr->roompt;
        int16_t ptr_off = rd16(room + TO_ZONE_PTS_OFFSET);
        const uint8_t *pt_list = room + ptr_off;

        /* Iterate the list of point indices */
        int safety = MAX_POINTS;
        while (safety-- > 0) {
            int16_t idx = rd16(pt_list);
            if (idx < 0) break;
            pt_list += 2;

            if (idx < MAX_POINTS) {
                rotate_one_point(r, pts, idx);
            }
        }
    } else {
        /* Fallback: no room data loaded yet - rotate all points */
        int num_pts = state->level.num_object_points;
        if (num_pts <= 0) return;
        if (num_pts > MAX_POINTS) num_pts = MAX_POINTS;

        for (int i = 0; i < num_pts; i++) {
            rotate_one_point(r, pts, i);
        }
    }
}

/* -----------------------------------------------------------------------
 * RotateObjectPts
 *
 * Translated from AB3DI.s RotateObjectPts (lines 4308-4362).
 *
 * Rotates object (enemy/pickup/bullet) positions from world space
 * to view space for sprite rendering.
 * ----------------------------------------------------------------------- */
void renderer_rotate_object_pts(GameState *state)
{
    RendererState *r = &g_renderer;

    if (!state->level.object_points || !state->level.object_data) return;

    int16_t sin_v = r->sinval;
    int16_t cos_v = r->cosval;
    int16_t cam_x = r->xoff;
    int16_t cam_z = r->zoff;

    int num_pts = state->level.num_object_points;
    if (num_pts <= 0) return;
    if (num_pts > MAX_OBJ_POINTS) num_pts = MAX_OBJ_POINTS;

    const uint8_t *pts      = state->level.object_points;
    const uint8_t *prev_pts = state->level.prev_object_points;
    float alpha = state->obj_interp_alpha;

    /* Amiga ObjDraw: ObjRotated is indexed by POINT number, not object index.
     * Rotate every point; when drawing, object uses (object_data[0]) as pt num
     * to look up ObjRotated[pt_num]. So keys (and others) scale correctly.
     *
     * When a prev snapshot is available, linearly interpolate between the
     * previous tick's positions and the current tick's positions using alpha
     * (0 = just ticked, 1 = about to tick next).  This produces sub-tick
     * smooth motion at display frame rate without changing the 50 Hz simulation. */
    for (int pt = 0; pt < num_pts; pt++) {
        int16_t px, pz;
        if (prev_pts && alpha > 0.0f) {
            int16_t px0 = rd16(prev_pts + pt * 8);
            int16_t pz0 = rd16(prev_pts + pt * 8 + 4);
            int16_t px1 = rd16(pts      + pt * 8);
            int16_t pz1 = rd16(pts      + pt * 8 + 4);
            px = (int16_t)(px0 + (int16_t)((float)(px1 - px0) * alpha));
            pz = (int16_t)(pz0 + (int16_t)((float)(pz1 - pz0) * alpha));
        } else {
            px = rd16(pts + pt * 8);
            pz = rd16(pts + pt * 8 + 4);
        }

        int16_t dx = (int16_t)(px - cam_x);
        int16_t dz = (int16_t)(pz - cam_z);

        /* Same rotation as level points — full-precision X and 24.8 Z */
        int32_t vx = (int32_t)dx * cos_v - (int32_t)dz * sin_v;
        vx <<= 1;

        int32_t vz = (int32_t)dx * sin_v + (int32_t)dz * cos_v;
        vz <<= 2;

        int32_t vx_fine = (vx >> 9) + r->xwobble;

        r->obj_rotated[pt].x = (int16_t)(vx >> 16);
        r->obj_rotated[pt].z = vz >> (16 - ROT_Z_FRAC_BITS);
        r->obj_rotated[pt].x_fine = vx_fine;
    }
}

/* Wall texture index for switches (io.c wall_texture_table). Must be before wall raster helpers. */
#define SWITCHES_WALL_TEX_ID  11

static void renderer_column_clip_add_span(int col, int top, int bot, int32_t z)
{
    ColumnClip *clip = &g_renderer.clip;
    if (col < 0 || col >= g_renderer.width) return;
    if (!clip->top || !clip->bot || !clip->z ||
        !clip->top2 || !clip->bot2 || !clip->z2) {
        return;
    }
    if (z <= 0) return;
    if (top > bot) {
        int t = top;
        top = bot;
        bot = t;
    }

    int16_t tops[2] = { clip->top[col], clip->top2[col] };
    int16_t bots[2] = { clip->bot[col], clip->bot2[col] };
    int32_t zs[2] = { clip->z[col], clip->z2[col] };

    for (int i = 0; i < 2; i++) {
        int valid = (zs[i] > 0 && tops[i] <= bots[i]);
        if (!valid) continue;
        /* Touching spans are merged to keep compact upper/lower coverage. */
        if (!(bot < (int)tops[i] - 1 || top > (int)bots[i] + 1)) {
            if (top < tops[i]) tops[i] = (int16_t)top;
            if (bot > bots[i]) bots[i] = (int16_t)bot;
            if (z < zs[i]) zs[i] = z;
            goto write_back;
        }
    }

    for (int i = 0; i < 2; i++) {
        int valid = (zs[i] > 0 && tops[i] <= bots[i]);
        if (!valid) {
            tops[i] = (int16_t)top;
            bots[i] = (int16_t)bot;
            zs[i] = z;
            goto write_back;
        }
    }

    /* Two spans already present: keep the two nearest spans by depth. */
    {
        int replace = (zs[0] >= zs[1]) ? 0 : 1;
        if (z < zs[replace]) {
            tops[replace] = (int16_t)top;
            bots[replace] = (int16_t)bot;
            zs[replace] = z;
        } else {
            return;
        }
    }

write_back:
    if (zs[0] > 0 && zs[1] > 0 && tops[0] > tops[1]) {
        int16_t tt = tops[0];
        int16_t tb = bots[0];
        int32_t tz = zs[0];
        tops[0] = tops[1];
        bots[0] = bots[1];
        zs[0] = zs[1];
        tops[1] = tt;
        bots[1] = tb;
        zs[1] = tz;
    }
    clip->top[col] = tops[0];
    clip->bot[col] = bots[0];
    clip->z[col] = zs[0];
    clip->top2[col] = tops[1];
    clip->bot2[col] = bots[1];
    clip->z2[col] = zs[1];
}

int32_t renderer_column_clip_nearest_z_at(int col, int row)
{
    const ColumnClip *clip = &g_renderer.clip;
    int32_t nearest_z = 0;
    if (col < 0 || col >= g_renderer.width) return 0;
    if (row < 0 || row >= g_renderer.height) return 0;

    if (!clip->top || !clip->bot || !clip->z ||
        !clip->top2 || !clip->bot2 || !clip->z2) {
        return 0;
    }

    if (clip->z[col] > 0 && clip->top[col] <= clip->bot[col] &&
        row >= clip->top[col] && row <= clip->bot[col]) {
        nearest_z = clip->z[col];
    }
    if (clip->z2[col] > 0 && clip->top2[col] <= clip->bot2[col] &&
        row >= clip->top2[col] && row <= clip->bot2[col]) {
        if (nearest_z <= 0 || clip->z2[col] < nearest_z) {
            nearest_z = clip->z2[col];
        }
    }
    return nearest_z;
}

/* Nearest wall Z in a column that overlaps [top, bot]. Used to decide whether
 * boundary extension pixels (x+/-1) have matching real wall geometry nearby. */
static int32_t renderer_column_clip_nearest_z_overlap(const ColumnClip *clip, int col, int top, int bot)
{
    int32_t nearest_z = 0;
    if (!clip) return 0;
    if (col < 0 || col >= g_renderer.width) return 0;
    if (top > bot) {
        int t = top;
        top = bot;
        bot = t;
    }

    if (!clip->top || !clip->bot || !clip->z ||
        !clip->top2 || !clip->bot2 || !clip->z2) {
        return 0;
    }

    if (clip->z[col] > 0 && clip->top[col] <= clip->bot[col]) {
        if (!(bot < clip->top[col] || top > clip->bot[col])) {
            nearest_z = clip->z[col];
        }
    }
    if (clip->z2[col] > 0 && clip->top2[col] <= clip->bot2[col]) {
        if (!(bot < clip->top2[col] || top > clip->bot2[col])) {
            if (nearest_z <= 0 || clip->z2[col] < nearest_z) {
                nearest_z = clip->z2[col];
            }
        }
    }
    return nearest_z;
}

static void renderer_zone_trace_floor_stats_init(RendererZoneTraceFloorStats *stats,
                                                 int active,
                                                 int zone_id,
                                                 int entry_index,
                                                 int entry_type,
                                                 int is_water)
{
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
    stats->active = active ? 1 : 0;
    stats->zone_id = zone_id;
    stats->entry_index = entry_index;
    stats->entry_type = entry_type;
    stats->is_water = is_water ? 1 : 0;
    stats->row_min = INT_MAX;
    stats->row_max = INT_MIN;
    stats->col_min = INT_MAX;
    stats->col_max = INT_MIN;
    stats->before_tags = NULL;
    stats->before_cw = NULL;
}

static inline void renderer_zone_trace_floor_stats_note_water_span(RendererZoneTraceFloorStats *stats,
                                                                   uint64_t pixel_count,
                                                                   int used_fast_path,
                                                                   int used_blend)
{
    if (!stats || !stats->active || !stats->is_water || pixel_count == 0) return;
    if (used_fast_path) stats->water_fast_path_pixels += pixel_count;
    else stats->water_fallback_pixels += pixel_count;
    if (used_blend) stats->water_blend_pixels += pixel_count;
    else stats->water_single_pixels += pixel_count;
}

static inline void renderer_zone_trace_floor_stats_note_water_sample0(RendererZoneTraceFloorStats *stats,
                                                                      int used_back_buffer)
{
    if (!stats || !stats->active || !stats->is_water || !used_back_buffer) return;
    stats->water_backbuf0_pixels++;
}

static inline void renderer_zone_trace_floor_stats_note_water_sample1(RendererZoneTraceFloorStats *stats,
                                                                      int used_back_buffer)
{
    if (!stats || !stats->active || !stats->is_water || !used_back_buffer) return;
    stats->water_backbuf1_pixels++;
}

static inline int renderer_water_span_needs_back_buffer(const uint8_t *tags, int span_len)
{
    if (!tags || span_len <= 0) return 0;
    for (int i = 0; i < span_len; i++) {
        uint8_t tag = tags[i];
        if (tag == 0 || tag == 4) return 1;
    }
    return 0;
}

static void renderer_zone_trace_floor_stats_accumulate_edges(RendererZoneTraceFloorStats *stats,
                                                             const RenderSliceContext *ctx,
                                                             const int16_t *left_edge,
                                                             const int16_t *right_edge,
                                                             int poly_top,
                                                             int poly_bot)
{
    uint8_t *buf = renderer_active_buf();
    uint16_t *cw = renderer_active_cw();
    int w = g_renderer.width;
    int h = g_renderer.height;

    if (!stats || !stats->active || !ctx || !left_edge || !right_edge || !buf) return;
    if (poly_top > poly_bot) return;

    for (int row = poly_top; row <= poly_bot; row++) {
        if (row < 0 || row >= h) continue;
        if (row < ctx->top_clip || row > ctx->bot_clip) continue;

        int16_t le = left_edge[row];
        int16_t re = right_edge[row];
        if (le >= w || re < 0) continue;

        int cle = (le < ctx->left_clip) ? ctx->left_clip : le;
        int cre = (re >= ctx->right_clip) ? (ctx->right_clip - 1) : re;
        if (cle > cre) continue;

        stats->submitted_pixels += (uint64_t)(cre - cle + 1);
        if (row < stats->row_min) stats->row_min = row;
        if (row > stats->row_max) stats->row_max = row;
        if (cle < stats->col_min) stats->col_min = cle;
        if (cre > stats->col_max) stats->col_max = cre;

        {
            size_t row_off = (size_t)row * (size_t)w;
            for (int x = cle; x <= cre; x++) {
                if (buf[row_off + (size_t)x] != 0) stats->prefilled_pixels++;
                else stats->first_claim_pixels++;
                if (renderer_column_clip_nearest_z_at(x, row) > 0) {
                    stats->wall_clip_cover_pixels++;
                }
            }
        }
    }

    if (!cw || stats->submitted_pixels == 0) return;

    stats->before_tags = (uint8_t *)malloc((size_t)stats->submitted_pixels * sizeof(*stats->before_tags));
    stats->before_cw = (uint16_t *)malloc((size_t)stats->submitted_pixels * sizeof(*stats->before_cw));
    if (!stats->before_tags || !stats->before_cw) {
        free(stats->before_tags);
        free(stats->before_cw);
        stats->before_tags = NULL;
        stats->before_cw = NULL;
        return;
    }

    {
        size_t idx = 0;
        for (int row = poly_top; row <= poly_bot; row++) {
            if (row < 0 || row >= h) continue;
            if (row < ctx->top_clip || row > ctx->bot_clip) continue;

            int16_t le = left_edge[row];
            int16_t re = right_edge[row];
            if (le >= w || re < 0) continue;

            int cle = (le < ctx->left_clip) ? ctx->left_clip : le;
            int cre = (re >= ctx->right_clip) ? (ctx->right_clip - 1) : re;
            if (cle > cre) continue;

            {
                size_t row_off = (size_t)row * (size_t)w;
                for (int x = cle; x <= cre; x++) {
                    stats->before_tags[idx] = buf[row_off + (size_t)x];
                    stats->before_cw[idx] = renderer_cw_load_xy(cw, x, row, w, h);
                    idx++;
                }
            }
        }
        stats->before_count = idx;
        stats->before_ready = (idx == (size_t)stats->submitted_pixels) ? 1 : 0;
        if (!stats->before_ready) {
            free(stats->before_tags);
            free(stats->before_cw);
            stats->before_tags = NULL;
            stats->before_cw = NULL;
            stats->before_count = 0;
        }
    }
}

static void renderer_zone_trace_floor_stats_finalize(RendererZoneTraceFloorStats *stats,
                                                     const RenderSliceContext *ctx,
                                                     const int16_t *left_edge,
                                                     const int16_t *right_edge,
                                                     int poly_top,
                                                     int poly_bot)
{
    uint8_t *buf = renderer_active_buf();
    uint16_t *cw = renderer_active_cw();
    int w = g_renderer.width;
    int h = g_renderer.height;

    if (!stats || !stats->active || !stats->before_ready || !ctx || !left_edge || !right_edge || !buf || !cw) return;
    if (poly_top > poly_bot) return;

    {
        size_t idx = 0;
        for (int row = poly_top; row <= poly_bot; row++) {
            if (row < 0 || row >= h) continue;
            if (row < ctx->top_clip || row > ctx->bot_clip) continue;

            int16_t le = left_edge[row];
            int16_t re = right_edge[row];
            if (le >= w || re < 0) continue;

            int cle = (le < ctx->left_clip) ? ctx->left_clip : le;
            int cre = (re >= ctx->right_clip) ? (ctx->right_clip - 1) : re;
            if (cle > cre) continue;

            {
                size_t row_off = (size_t)row * (size_t)w;
                for (int x = cle; x <= cre && idx < stats->before_count; x++) {
                    uint8_t after_tag = buf[row_off + (size_t)x];
                    uint16_t after_cw = renderer_cw_load_xy(cw, x, row, w, h);
                    if (after_tag == stats->before_tags[idx] && after_cw == stats->before_cw[idx])
                        stats->same_output_pixels++;
                    else
                        stats->changed_output_pixels++;
                    idx++;
                }
            }
        }
    }

    free(stats->before_tags);
    free(stats->before_cw);
    stats->before_tags = NULL;
    stats->before_cw = NULL;
    stats->before_count = 0;
    stats->before_ready = 0;
}

static void renderer_zone_trace_floor_stats_log(const RendererZoneTraceFloorStats *stats)
{
    double first_claim_pct;
    double prefilled_pct;
    double wall_clip_pct;
    double same_output_pct;
    double changed_output_pct;
    uint64_t water_drawn_pixels;
    double water_fast_path_pct;
    double water_fallback_pct;
    double water_blend_pct;
    double water_single_pct;
    double water_backbuf0_pct;
    double water_backbuf1_pct;

    if (!stats || !stats->active) return;

    first_claim_pct = (stats->submitted_pixels > 0)
        ? (100.0 * (double)stats->first_claim_pixels / (double)stats->submitted_pixels)
        : 0.0;
    prefilled_pct = (stats->submitted_pixels > 0)
        ? (100.0 * (double)stats->prefilled_pixels / (double)stats->submitted_pixels)
        : 0.0;
    wall_clip_pct = (stats->submitted_pixels > 0)
        ? (100.0 * (double)stats->wall_clip_cover_pixels / (double)stats->submitted_pixels)
        : 0.0;
    same_output_pct = (stats->submitted_pixels > 0)
        ? (100.0 * (double)stats->same_output_pixels / (double)stats->submitted_pixels)
        : 0.0;
    changed_output_pct = (stats->submitted_pixels > 0)
        ? (100.0 * (double)stats->changed_output_pixels / (double)stats->submitted_pixels)
        : 0.0;
    water_drawn_pixels = stats->water_fast_path_pixels + stats->water_fallback_pixels;
    water_fast_path_pct = (water_drawn_pixels > 0)
        ? (100.0 * (double)stats->water_fast_path_pixels / (double)water_drawn_pixels)
        : 0.0;
    water_fallback_pct = (water_drawn_pixels > 0)
        ? (100.0 * (double)stats->water_fallback_pixels / (double)water_drawn_pixels)
        : 0.0;
    water_blend_pct = (water_drawn_pixels > 0)
        ? (100.0 * (double)stats->water_blend_pixels / (double)water_drawn_pixels)
        : 0.0;
    water_single_pct = (water_drawn_pixels > 0)
        ? (100.0 * (double)stats->water_single_pixels / (double)water_drawn_pixels)
        : 0.0;
    water_backbuf0_pct = (water_drawn_pixels > 0)
        ? (100.0 * (double)stats->water_backbuf0_pixels / (double)water_drawn_pixels)
        : 0.0;
    water_backbuf1_pct = (stats->water_blend_pixels > 0)
        ? (100.0 * (double)stats->water_backbuf1_pixels / (double)stats->water_blend_pixels)
        : 0.0;

    if (stats->is_water) {
        printf("[ZONEFLOOR][frame %u] zone=%d entry[%02d] type=%d water=%d rows=[%d,%d] cols=[%d,%d] submitted=%llu first_claim=%llu(%.1f%%) prefilled=%llu(%.1f%%) same_out=%llu(%.1f%%) changed_out=%llu(%.1f%%) wall_clip_cover=%llu(%.1f%%) water_drawn=%llu fast=%llu(%.1f%%) fallback=%llu(%.1f%%) blend=%llu(%.1f%%) single=%llu(%.1f%%) backbuf0=%llu(%.1f%%) backbuf1=%llu(%.1f%% of blend)\n",
               (unsigned)g_renderer_zone_trace_frame_idx,
               stats->zone_id,
               stats->entry_index,
               stats->entry_type,
               stats->is_water,
               (stats->row_min == INT_MAX) ? -1 : stats->row_min,
               (stats->row_max == INT_MIN) ? -1 : stats->row_max,
               (stats->col_min == INT_MAX) ? -1 : stats->col_min,
               (stats->col_max == INT_MIN) ? -1 : stats->col_max,
               (unsigned long long)stats->submitted_pixels,
               (unsigned long long)stats->first_claim_pixels,
               first_claim_pct,
               (unsigned long long)stats->prefilled_pixels,
               prefilled_pct,
               (unsigned long long)stats->same_output_pixels,
               same_output_pct,
               (unsigned long long)stats->changed_output_pixels,
               changed_output_pct,
               (unsigned long long)stats->wall_clip_cover_pixels,
               wall_clip_pct,
               (unsigned long long)water_drawn_pixels,
               (unsigned long long)stats->water_fast_path_pixels,
               water_fast_path_pct,
               (unsigned long long)stats->water_fallback_pixels,
               water_fallback_pct,
               (unsigned long long)stats->water_blend_pixels,
               water_blend_pct,
               (unsigned long long)stats->water_single_pixels,
               water_single_pct,
               (unsigned long long)stats->water_backbuf0_pixels,
               water_backbuf0_pct,
               (unsigned long long)stats->water_backbuf1_pixels,
               water_backbuf1_pct);
    } else {
        printf("[ZONEFLOOR][frame %u] zone=%d entry[%02d] type=%d water=%d rows=[%d,%d] cols=[%d,%d] submitted=%llu first_claim=%llu(%.1f%%) prefilled=%llu(%.1f%%) same_out=%llu(%.1f%%) changed_out=%llu(%.1f%%) wall_clip_cover=%llu(%.1f%%)\n",
               (unsigned)g_renderer_zone_trace_frame_idx,
               stats->zone_id,
               stats->entry_index,
               stats->entry_type,
               stats->is_water,
               (stats->row_min == INT_MAX) ? -1 : stats->row_min,
               (stats->row_max == INT_MIN) ? -1 : stats->row_max,
               (stats->col_min == INT_MAX) ? -1 : stats->col_min,
               (stats->col_max == INT_MIN) ? -1 : stats->col_max,
               (unsigned long long)stats->submitted_pixels,
               (unsigned long long)stats->first_claim_pixels,
               first_claim_pct,
               (unsigned long long)stats->prefilled_pixels,
               prefilled_pct,
               (unsigned long long)stats->same_output_pixels,
               same_output_pct,
               (unsigned long long)stats->changed_output_pixels,
               changed_output_pct,
               (unsigned long long)stats->wall_clip_cover_pixels,
               wall_clip_pct);
    }
}

/* Wall column loop uses inverse-Z in 8.24 (INVZ_ONE). Projecting Y as world_y*K/z with an
 * integer z from z = INVZ_ONE/inv_z truncates twice vs. world_y*K*inv_z/INVZ_ONE (one divide).
 * The latter matches true perspective along the span and reduces stair-steps at floor/ceiling. */
static AB3D_ATTR_UNUSED int wall_proj_y_screen_invz(int16_t world_y, int64_t inv_z_fp, int32_t proj_y_scale, int height)
{
    const int64_t INVZ_ONE = (1LL << 24);
    if (inv_z_fp <= 0) inv_z_fp = 1;
    int64_t num = (int64_t)world_y * (int64_t)proj_y_scale * (int64_t)RENDER_SCALE * inv_z_fp;
    int64_t q = (num >= 0)
        ? ((num + INVZ_ONE / 2) / INVZ_ONE)
        : ((num - INVZ_ONE / 2) / INVZ_ONE);
    return (int)q + height / 2;
}

/* -----------------------------------------------------------------------
 * Wall rendering — one entry per wall segment (not per screen column).
 *
 * Translated from WallRoutine3.ChipMem.s ScreenWallstripdraw / walldraw.
 * Perspective-correct 1/z and tex/z interpolation runs in the outer loop;
 * each column runs the vertical strip raster (formerly draw_wall_column).
 * ----------------------------------------------------------------------- */
static void draw_wall_rasterize_segment(
    RenderSliceContext *ctx,
    int draw_start, int draw_end, int scr_x1, int span,
    int64_t inv_z1_fp, int64_t inv_z_delta_fp,
    int64_t tex_over_z1_fp, int64_t tex_over_z_delta_fp,
    int32_t left_brightness, int32_t bright_delta,
    int16_t top, int16_t bot,
    const uint8_t *texture,
    uint8_t valand, uint8_t valshift, int16_t horand,
    int16_t totalyoff, int16_t fromtile,
    int16_t tex_id, int16_t wall_height_for_tex,
    int16_t d6_max)
{
    const int64_t INVZ_ONE = (1LL << 24);
    const int profile_collect_stats = (ctx && ctx->profile_collect_stats);

    uint8_t * AB3D_RESTRICT buf = renderer_active_buf();
    uint32_t * AB3D_RESTRICT rgb = renderer_active_rgb();
    uint16_t * AB3D_RESTRICT cw = renderer_active_cw();
    if (!ctx) return;
    if (!buf || !rgb || !cw) return;

    const int expand = g_renderer_rgb_raster_expand;
    const int width = g_renderer.width;
    const int height = g_renderer.height;
    const size_t wstride = (size_t)width;
    const size_t cw_step_y = renderer_cw_step_y(width);
    const size_t cw_step_x = renderer_cw_step_x(height);
    const int32_t proj_y_scale = g_renderer.proj_y_scale;
    const int half_h = height / 2;

    if (d6_max < 0) d6_max = 0;
    if (d6_max > 64) d6_max = 64;

    /* --- Hoisted constants (invariant across all columns) --- */
    const int effective_top = (ctx->wall_top_clip >= 0) ? (int)ctx->wall_top_clip : (int)ctx->top_clip;
    const int effective_bot = (ctx->wall_bot_clip >= 0) ? (int)ctx->wall_bot_clip : (int)ctx->bot_clip;
    const int top_clip_val = (int)ctx->top_clip;
    const int bot_clip_val = (int)ctx->bot_clip;
    const int update_column_clip = (ctx->update_column_clip != 0);
    const int16_t *foreground_floor_occlude_top = ctx->foreground_floor_occlude_top;
    const int pick_capture_active = (g_pick_capture_active != 0);

    /* Precompute projection factors: top/bot * proj_y_scale * RENDER_SCALE */
    const int64_t top_proj = (int64_t)top * (int64_t)proj_y_scale * (int64_t)RENDER_SCALE;
    const int64_t bot_proj = (int64_t)bot * (int64_t)proj_y_scale * (int64_t)RENDER_SCALE;

    /* Precompute texture height constants */
    const int rows = 1 << valshift;
    int tex_h = (int)wall_height_for_tex;
    if (tex_h < 1) tex_h = 1;
    if (tex_id == SWITCHES_WALL_TEX_ID) tex_h = rows;
    else if (rows < 64 && tex_h < 64) tex_h = 64;
    const int64_t h_shifted = (int64_t)tex_h << 16;
    const int32_t yoff_base = (int32_t)((unsigned)totalyoff & (unsigned)valand);

    const uint8_t *pal = ctx->cur_wall_pal;
    const int has_tex_pal = (texture != NULL && pal != NULL);

    /* --- Endpoint-anchored interpolation ---
     * Compute start/end from the full line equation, then walk with extra fractional precision.
     * This keeps motion stable while preserving accurate segment endpoints when clipping shifts. */
    enum { INTERP_SUB_BITS = 16 };
    const int64_t start_col = (int64_t)(draw_start - scr_x1);
    const int64_t end_col = (int64_t)(draw_end - scr_x1);
    const int64_t draw_cols = end_col - start_col;

    const int64_t inv_z_start = inv_z1_fp + (inv_z_delta_fp * start_col) / span;
    const int64_t inv_z_end = inv_z1_fp + (inv_z_delta_fp * end_col) / span;
    const int64_t tex_z_start = tex_over_z1_fp + (tex_over_z_delta_fp * start_col) / span;
    const int64_t tex_z_end = tex_over_z1_fp + (tex_over_z_delta_fp * end_col) / span;
    const int64_t bright_start_fp = ((int64_t)left_brightness << 16) +
                                    (((int64_t)bright_delta << 16) * start_col) / span;
    const int64_t bright_end_fp = ((int64_t)left_brightness << 16) +
                                  (((int64_t)bright_delta << 16) * end_col) / span;

    int64_t inv_z_acc = inv_z_start << INTERP_SUB_BITS;
    int64_t tex_z_acc = tex_z_start << INTERP_SUB_BITS;
    int64_t bright_acc = bright_start_fp << INTERP_SUB_BITS;
    int64_t inv_z_step_acc = 0;
    int64_t tex_z_step_acc = 0;
    int64_t bright_step_acc = 0;
    if (draw_cols > 0) {
        inv_z_step_acc = ((inv_z_end - inv_z_start) << INTERP_SUB_BITS) / draw_cols;
        tex_z_step_acc = ((tex_z_end - tex_z_start) << INTERP_SUB_BITS) / draw_cols;
        bright_step_acc = ((bright_end_fp - bright_start_fp) << INTERP_SUB_BITS) / draw_cols;
    }

    for (int screen_x = draw_start; screen_x <= draw_end; screen_x++) {
        int64_t inv_z_fp = inv_z_acc >> INTERP_SUB_BITS;
        if (inv_z_fp <= 0) inv_z_fp = 1;

        /* 32-bit reciprocal: both INVZ_ONE and inv_z_fp fit in 32 bits for typical Z values */
        const uint32_t inv_z_u = (uint32_t)inv_z_fp;
        int32_t col_z = (int32_t)((16777216u + inv_z_u / 2u) / inv_z_u);
        if (col_z < 1) col_z = 1;

        int32_t wall_bright = (int32_t)(bright_acc >> (16 + INTERP_SUB_BITS));
        int amiga_d6 = (col_z >> 7) + (wall_bright * 2);
        if (amiga_d6 < 0) amiga_d6 = 0;
        if (amiga_d6 > d6_max) amiga_d6 = d6_max;

        /* Inline projection with precomputed factors */
        int64_t tnum = top_proj * inv_z_fp;
        int y_top_scr = (int)((tnum >= 0 ? (tnum + (INVZ_ONE / 2)) : (tnum - (INVZ_ONE / 2))) / INVZ_ONE) + half_h;
        int64_t bnum = bot_proj * inv_z_fp;
        int y_bot_scr = (int)((bnum >= 0 ? (bnum + (INVZ_ONE / 2)) : (bnum - (INVZ_ONE / 2))) / INVZ_ONE) + half_h;

        /* Texture column via perspective divide */
        int64_t tex_t_fp64 = ((tex_z_acc >> INTERP_SUB_BITS) * INVZ_ONE) / inv_z_fp;
        int tex_col = ((int32_t)(tex_t_fp64 >> 24) & horand) + fromtile;

        /* Step interpolators */
        inv_z_acc += inv_z_step_acc;
        tex_z_acc += tex_z_step_acc;
        bright_acc += bright_step_acc;

        const int x = screen_x;

        int yt = y_top_scr, yb = y_bot_scr;
        if (yb <= yt) yb = yt + 1;
        int ct = (yt < effective_top) ? effective_top : yt;
        int cb = (yb > effective_bot) ? effective_bot : yb;
        int floor_occluded_bot = 0;
        if (foreground_floor_occlude_top) {
            int occ_top = (int)foreground_floor_occlude_top[x];
            if (occ_top <= cb) {
                if (occ_top <= ct) continue;
                cb = occ_top - 1;
                floor_occluded_bot = 1;
            }
        }
        if (ct > cb) continue;

        /* Palette cache */
        uint16_t lut_block_off = wall_scale_table[amiga_d6];
        uint32_t fallback_rgb = 0;
        uint16_t fallback_cw = 0;
        if (has_tex_pal) {
            if (!ctx->wall_cache_valid || ctx->wall_cache_pal != pal || ctx->wall_cache_block_off != lut_block_off) {
                for (int ti = 0; ti < 32; ti++) {
                    int lut_off = lut_block_off + ti * 2;
                    uint16_t c12 = ((uint16_t)pal[lut_off] << 8) | pal[lut_off + 1];
                    ctx->wall_cache_cw[ti] = c12;
                    ctx->wall_cache_rgb[ti] = amiga12_to_argb(c12);
                }
                ctx->wall_cache_pal = pal;
                ctx->wall_cache_block_off = lut_block_off;
                ctx->wall_cache_valid = 1;
            }
        } else {
            int gray = (64 - amiga_d6) * 255 / 64;
            if (gray < 0)   gray = 0;
            if (gray > 255) gray = 255;
            fallback_rgb = RENDER_RGB_RASTER_PIXEL(((uint32_t)gray << 16) | ((uint32_t)gray << 8) | (uint32_t)gray);
            fallback_cw = argb_to_amiga12(fallback_rgb);
        }

        /* Texture addressing */
        int strip_index = tex_col / 3;
        int pack_mode   = tex_col % 3;
        int strip_offset = strip_index << (valshift + 1);
        const uint8_t pack_shift = (uint8_t)(pack_mode * 5);

        int wall_pixels = yb - y_top_scr;
        if (wall_pixels < 1) wall_pixels = 1;
        int32_t tex_step = (int32_t)(h_shifted / wall_pixels);
        int32_t tex_y = (ct - y_top_scr) * tex_step + ((int32_t)yoff_base << 16);

        /* Edge extension only on segment boundary columns.  Interior columns
         * of the same segment are consecutive, so the ±1 extension from one
         * column is immediately overwritten by the adjacent column's center
         * pixel — pure redundant overdraw (~67% of wall pixel writes). */
        const int do_ext_l = (screen_x == draw_start) && (x > ctx->slice_left);
        const int do_ext_r = (screen_x == draw_end) && (x + 1 < ctx->slice_right);

        if (profile_collect_stats) {
            uint64_t run = (uint64_t)(cb - ct + 1);
            ctx->workload_stats.wall_columns++;
            ctx->workload_stats.wall_pixels_core += run;
            if (do_ext_l) ctx->workload_stats.wall_pixels_side_ext += run;
            if (do_ext_r) ctx->workload_stats.wall_pixels_side_ext += run;
            if (ct > top_clip_val) ctx->workload_stats.wall_pixels_cap_ext++;
            if (!floor_occluded_bot && cb + 1 <= bot_clip_val) ctx->workload_stats.wall_pixels_cap_ext++;
        }

        size_t pix = (size_t)ct * wstride + (size_t)x;
        size_t pix_cw = renderer_cw_index_xy(x, ct, width, height);

        /* ---- Main vertical raster with merged horizontal edge extension ----
         * Textured loops are 4-way specialized on the column-constant
         * (do_ext_l, do_ext_r) pair so the compiler eliminates dead extension
         * writes instead of branching every pixel.  Write prefetches hide the
         * vertical-stride cache misses inherent to column-major rasterization. */
#define WALL_PF_DIST 8
        if (has_tex_pal && strip_offset >= 0) {
            const uint8_t *tex_strip = texture + strip_offset;
            const uint16_t *cache_cw  = ctx->wall_cache_cw;
            const uint32_t *cache_rgb = ctx->wall_cache_rgb;

            /* Dominant wall path: no RGB expand, no side extension. Keep loop body minimal. */
            if (!expand && !do_ext_l && !do_ext_r && (unsigned)pack_mode < 3u) {
                size_t pix_hot = pix;
                size_t pix_hot_cw = pix_cw;
                size_t pf_hot = pix_hot + (size_t)wstride * WALL_PF_DIST;
                int32_t tex_y_hot = tex_y;
                const int hot_count = cb - ct + 1;
                const int do_pf_hot = (hot_count > WALL_PF_DIST);

#define WALL_TEX_HOT_ROW(SHIFT_) \
    do { \
        int ra_ = ((int)(tex_y_hot >> 16) & valand) << 1; \
        uint16_t w_ = ((uint16_t)tex_strip[ra_] << 8) | tex_strip[ra_ + 1]; \
        uint8_t t_ = (uint8_t)((w_ >> (SHIFT_)) & 31u); \
        buf[pix_hot] = 2; \
        AB3D_NT_STORE_U16(&cw[pix_hot_cw], cache_cw[t_]); \
        pix_hot += wstride; \
        pix_hot_cw += cw_step_y; \
        pf_hot += wstride; \
        tex_y_hot += tex_step; \
    } while (0)

                switch (pack_mode) {
                    case 0:
                        if (do_pf_hot) {
                            int i = 0;
                            for (; i + 3 < hot_count; i += 4) {
                                AB3D_PREFETCH_WRITE(&buf[pf_hot]);
                                WALL_TEX_HOT_ROW(0);
                                WALL_TEX_HOT_ROW(0);
                                WALL_TEX_HOT_ROW(0);
                                WALL_TEX_HOT_ROW(0);
                            }
                            for (; i < hot_count; i++) {
                                WALL_TEX_HOT_ROW(0);
                            }
                        } else {
                            for (int i = 0; i < hot_count; i++) {
                                WALL_TEX_HOT_ROW(0);
                            }
                        }
                        break;
                    case 1:
                        if (do_pf_hot) {
                            int i = 0;
                            for (; i + 3 < hot_count; i += 4) {
                                AB3D_PREFETCH_WRITE(&buf[pf_hot]);
                                WALL_TEX_HOT_ROW(5);
                                WALL_TEX_HOT_ROW(5);
                                WALL_TEX_HOT_ROW(5);
                                WALL_TEX_HOT_ROW(5);
                            }
                            for (; i < hot_count; i++) {
                                WALL_TEX_HOT_ROW(5);
                            }
                        } else {
                            for (int i = 0; i < hot_count; i++) {
                                WALL_TEX_HOT_ROW(5);
                            }
                        }
                        break;
                    default: /* pack_mode == 2 */
                        if (do_pf_hot) {
                            int i = 0;
                            for (; i + 3 < hot_count; i += 4) {
                                AB3D_PREFETCH_WRITE(&buf[pf_hot]);
                                WALL_TEX_HOT_ROW(10);
                                WALL_TEX_HOT_ROW(10);
                                WALL_TEX_HOT_ROW(10);
                                WALL_TEX_HOT_ROW(10);
                            }
                            for (; i < hot_count; i++) {
                                WALL_TEX_HOT_ROW(10);
                            }
                        } else {
                            for (int i = 0; i < hot_count; i++) {
                                WALL_TEX_HOT_ROW(10);
                            }
                        }
                        break;
                }
#undef WALL_TEX_HOT_ROW

                pix = pix_hot;
                pix_cw = pix_hot_cw;
                tex_y = tex_y_hot;
            } else if (expand) {
#define WALL_TEX_E(EL, ER) \
    do { for (int y_ = ct; y_ <= cb; y_++) { \
        AB3D_PREFETCH_WRITE(&buf[pix + (size_t)wstride * WALL_PF_DIST]); \
        AB3D_PREFETCH_WRITE(&rgb[pix + (size_t)wstride * WALL_PF_DIST]); \
        int ra_ = ((int)(tex_y >> 16) & valand) << 1; \
        uint16_t w_ = ((uint16_t)tex_strip[ra_] << 8) | tex_strip[ra_ + 1]; \
        uint8_t t_ = (uint8_t)((w_ >> pack_shift) & 31u); \
        uint16_t cv_ = cache_cw[t_]; uint32_t rv_ = cache_rgb[t_]; \
        buf[pix] = 2; AB3D_NT_STORE_U16(&cw[pix_cw], cv_); rgb[pix] = rv_; \
        if (EL) { buf[pix - 1] = 2; cw[pix_cw - cw_step_x] = cv_; rgb[pix - 1] = rv_; } \
        if (ER) { buf[pix + 1] = 2; cw[pix_cw + cw_step_x] = cv_; rgb[pix + 1] = rv_; } \
        pix += wstride; pix_cw += cw_step_y; tex_y += tex_step; \
    } } while (0)
                if      ( do_ext_l &&  do_ext_r) WALL_TEX_E(1, 1);
                else if ( do_ext_l             ) WALL_TEX_E(1, 0);
                else if (              do_ext_r) WALL_TEX_E(0, 1);
                else                             WALL_TEX_E(0, 0);
#undef WALL_TEX_E
            } else {
#define WALL_TEX_C(EL, ER) \
    do { for (int y_ = ct; y_ <= cb; y_++) { \
        AB3D_PREFETCH_WRITE(&buf[pix + (size_t)wstride * WALL_PF_DIST]); \
        int ra_ = ((int)(tex_y >> 16) & valand) << 1; \
        uint16_t w_ = ((uint16_t)tex_strip[ra_] << 8) | tex_strip[ra_ + 1]; \
        uint8_t t_ = (uint8_t)((w_ >> pack_shift) & 31u); \
        uint16_t cv_ = cache_cw[t_]; \
        buf[pix] = 2; AB3D_NT_STORE_U16(&cw[pix_cw], cv_); \
        if (EL) { buf[pix - 1] = 2; cw[pix_cw - cw_step_x] = cv_; } \
        if (ER) { buf[pix + 1] = 2; cw[pix_cw + cw_step_x] = cv_; } \
        pix += wstride; pix_cw += cw_step_y; tex_y += tex_step; \
    } } while (0)
                if      ( do_ext_l &&  do_ext_r) WALL_TEX_C(1, 1);
                else if ( do_ext_l             ) WALL_TEX_C(1, 0);
                else if (              do_ext_r) WALL_TEX_C(0, 1);
                else                             WALL_TEX_C(0, 0);
#undef WALL_TEX_C
            }
        } else if (has_tex_pal) {
            uint16_t cw0 = ctx->wall_cache_cw[0];
            uint32_t rgb0 = ctx->wall_cache_rgb[0];
            if (expand) {
                for (int y = ct; y <= cb; y++) {
                    AB3D_PREFETCH_WRITE(&buf[pix + (size_t)wstride * WALL_PF_DIST]);
                    AB3D_PREFETCH_WRITE(&rgb[pix + (size_t)wstride * WALL_PF_DIST]);
                    buf[pix] = 2; AB3D_NT_STORE_U16(&cw[pix_cw], cw0); rgb[pix] = rgb0;
                    if (do_ext_l) { buf[pix - 1] = 2; cw[pix_cw - cw_step_x] = cw0; rgb[pix - 1] = rgb0; }
                    if (do_ext_r) { buf[pix + 1] = 2; cw[pix_cw + cw_step_x] = cw0; rgb[pix + 1] = rgb0; }
                    pix += wstride;
                    pix_cw += cw_step_y;
                }
            } else {
                for (int y = ct; y <= cb; y++) {
                    AB3D_PREFETCH_WRITE(&buf[pix + (size_t)wstride * WALL_PF_DIST]);
                    buf[pix] = 2; AB3D_NT_STORE_U16(&cw[pix_cw], cw0);
                    if (do_ext_l) { buf[pix - 1] = 2; cw[pix_cw - cw_step_x] = cw0; }
                    if (do_ext_r) { buf[pix + 1] = 2; cw[pix_cw + cw_step_x] = cw0; }
                    pix += wstride;
                    pix_cw += cw_step_y;
                }
            }
        } else {
            if (expand) {
                for (int y = ct; y <= cb; y++) {
                    AB3D_PREFETCH_WRITE(&buf[pix + (size_t)wstride * WALL_PF_DIST]);
                    AB3D_PREFETCH_WRITE(&rgb[pix + (size_t)wstride * WALL_PF_DIST]);
                    buf[pix] = 2; rgb[pix] = fallback_rgb; AB3D_NT_STORE_U16(&cw[pix_cw], fallback_cw);
                    if (do_ext_l) { buf[pix - 1] = 2; rgb[pix - 1] = fallback_rgb; cw[pix_cw - cw_step_x] = fallback_cw; }
                    if (do_ext_r) { buf[pix + 1] = 2; rgb[pix + 1] = fallback_rgb; cw[pix_cw + cw_step_x] = fallback_cw; }
                    pix += wstride;
                    pix_cw += cw_step_y;
                }
            } else {
                for (int y = ct; y <= cb; y++) {
                    AB3D_PREFETCH_WRITE(&buf[pix + (size_t)wstride * WALL_PF_DIST]);
                    buf[pix] = 2; AB3D_NT_STORE_U16(&cw[pix_cw], fallback_cw);
                    if (do_ext_l) { buf[pix - 1] = 2; cw[pix_cw - cw_step_x] = fallback_cw; }
                    if (do_ext_r) { buf[pix + 1] = 2; cw[pix_cw + cw_step_x] = fallback_cw; }
                    pix += wstride;
                    pix_cw += cw_step_y;
                }
            }
        }
#undef WALL_PF_DIST

        /* Vertical edge extension (one row above/below) */
        {
            size_t first_pix = (size_t)ct * wstride + (size_t)x;
            size_t last_pix  = (size_t)cb * wstride + (size_t)x;
            size_t first_pix_cw = renderer_cw_index_xy(x, ct, width, height);
            size_t last_pix_cw  = renderer_cw_index_xy(x, cb, width, height);
            if (ct > top_clip_val) {
                size_t up = first_pix - wstride;
                buf[up] = 2;
                cw[first_pix_cw - cw_step_y] = cw[first_pix_cw];
                if (expand) rgb[up] = rgb[first_pix];
            }
            if (!floor_occluded_bot && cb + 1 <= bot_clip_val) {
                size_t dn = last_pix + wstride;
                buf[dn] = 2;
                cw[last_pix_cw + cw_step_y] = cw[last_pix_cw];
                if (expand) rgb[dn] = rgb[last_pix];
            }
        }
        if (pick_capture_active) {
            renderer_pick_mark_wall_column(ctx, x, ct, cb,
                                           do_ext_l, do_ext_r,
                                           (ct > top_clip_val),
                                           (!floor_occluded_bot && (cb + 1 <= bot_clip_val)));
        }

        /* Column clip update.
         * Register real wall geometry at x unconditionally.
         *
         * For extension pixels at x+/-1, add clip spans only when neighboring
         * columns already contain overlapping wall spans. This preserves seam
         * occlusion while avoiding phantom occluders at stepped boundaries. */
        if (update_column_clip) {
            int occ_top = ct, occ_bot = cb;
            if (ct > top_clip_val) occ_top = ct - 1;
            if (cb + 1 <= bot_clip_val) occ_bot = cb + 1;
            renderer_column_clip_add_span(x, occ_top, occ_bot, col_z);

            if (do_ext_l) {
                int32_t nz = renderer_column_clip_nearest_z_overlap(&g_renderer.clip, x - 1, ct, cb);
                if (nz > 0) {
                    int32_t dz = nz - col_z;
                    if (dz < 0) dz = -dz;
                    if (dz <= 3) {
                        int32_t ext_z = (nz < col_z) ? nz : col_z;
                        renderer_column_clip_add_span(x - 1, ct, cb, ext_z);
                    }
                }
            }
            if (do_ext_r) {
                int32_t nz = renderer_column_clip_nearest_z_overlap(&g_renderer.clip, x + 1, ct, cb);
                if (nz > 0) {
                    int32_t dz = nz - col_z;
                    if (dz < 0) dz = -dz;
                    if (dz <= 3) {
                        int32_t ext_z = (nz < col_z) ? nz : col_z;
                        renderer_column_clip_add_span(x + 1, ct, cb, ext_z);
                    }
                }
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Draw a wall segment between two rotated endpoints
 *
 * Translated from WallRoutine3.ChipMem.s walldraw/screendivide.
 *
 * Takes two endpoints in view space, projects them, and draws
 * columns from left to right with perspective-correct texturing.
 * ----------------------------------------------------------------------- */
static void renderer_draw_wall_ctx(RenderSliceContext *ctx,
                                   int32_t x1, int32_t z1, int32_t x2, int32_t z2,
                                   int16_t top, int16_t bot,
                                   const uint8_t *texture, int16_t tex_start,
                                   int16_t tex_end, int16_t left_brightness, int16_t right_brightness,
                                   uint8_t valand, uint8_t valshift, int16_t horand,
                                   int16_t totalyoff, int16_t fromtile,
                                   int16_t tex_id, int16_t wall_height_for_tex,
                                   int16_t d6_max)
{
    /* Both behind camera - skip */
    if (z1 <= 0 && z2 <= 0) return;

    /* Clip to near plane */
    int32_t cx1 = x1, cz1 = z1;
    int32_t cx2 = x2, cz2 = z2;
    int16_t ct1 = tex_start, ct2 = tex_end;

    /* Z values are 24.8 fixed-point; near plane = 1 in world units = ROT_Z_ONE in fp. */
    const int32_t NEAR_PLANE = ROT_Z_FROM_INT(1);
    
    if (cz1 < NEAR_PLANE) {
        int32_t dz = cz2 - cz1;
        if (dz == 0) { cz1 = NEAR_PLANE; cx1 = (int16_t)((cx1 + cx2) / 2); ct1 = (int16_t)((ct1 + ct2) / 2); }
        else {
            int32_t t = (int32_t)((int64_t)(NEAR_PLANE - cz1) * 65536 / dz);
            cx1 = (int32_t)(cx1 + (int64_t)(cx2 - cx1) * t / 65536);
            cz1 = NEAR_PLANE;
            ct1 = (int16_t)(ct1 + (int32_t)(ct2 - ct1) * t / 65536);
        }
    }
    if (cz2 < NEAR_PLANE) {
        int32_t dz = cz1 - cz2;
        if (dz == 0) { cz2 = NEAR_PLANE; cx2 = cx1; ct2 = ct1; }
        else {
            int32_t t = (int32_t)((int64_t)(NEAR_PLANE - cz2) * 65536 / dz);
            cx2 = (int32_t)(cx2 + (int64_t)(cx1 - cx2) * t / 65536);
            cz2 = NEAR_PLANE;
            ct2 = (int16_t)(ct2 + (int32_t)(ct1 - ct2) * t / 65536);
        }
    }

    /* Back-face cull (Amiga itsawalldraw/pastclip behavior):
     * with z>0, front-facing walls project left->right and satisfy x1/z1 < x2/z2.
     * Equivalent signed test in view space: (x1*z2 - z1*x2) < 0. */
    {
        int64_t face = (int64_t)cx1 * (int64_t)cz2 - (int64_t)cz1 * (int64_t)cx2;
        if (face > 0) return;
    }

    /* Project in fine pixel space for smooth motion. */
    int scr_x1 = project_x_to_pixels(cx1, cz1);
    int scr_x2 = project_x_to_pixels(cx2, cz2);

    if (scr_x2 < ctx->left_clip || scr_x1 >= ctx->right_clip) return;

    /* Clamp drawn range to clip region so we only iterate over visible columns.
     * This avoids int32 overflow in interpolation when the wall extends far off the left
     * (col = left_clip - scr_x1 can be huge, and col * 65536 overflows). */
    int draw_start = scr_x1;
    if (draw_start < ctx->left_clip) draw_start = ctx->left_clip;
    int draw_end = scr_x2;
    if (draw_end >= ctx->right_clip) draw_end = ctx->right_clip - 1;
    if (draw_start > draw_end) return;

    /* Wall span for interpolation (use at least 1 to avoid division by zero) */
    int span = scr_x2 - scr_x1;
    if (span <= 0) span = 1;

    /* Perspective-correct: interpolate 1/z and tex/z with higher fixed-point precision.
     * cz1/cz2 are 24.8 fixed-point Z; scale INVZ numerator by ROT_Z_ONE so the
     * resulting inv_z is in the same scale as the old integer-Z path. */
    const int64_t INVZ_ONE = (1LL << 24); /* 8.24 — used in the rasterizer denominator */
    const int64_t INVZ_FP_SCALE = ((int64_t)INVZ_ONE << ROT_Z_FRAC_BITS);
    int64_t inv_z1_fp = INVZ_FP_SCALE / cz1;
    int64_t inv_z2_fp = INVZ_FP_SCALE / cz2;
    int64_t tex_over_z1_fp = (int64_t)ct1 * INVZ_FP_SCALE / cz1;
    int64_t tex_over_z2_fp = (int64_t)ct2 * INVZ_FP_SCALE / cz2;
    int64_t inv_z_delta_fp = inv_z2_fp - inv_z1_fp;
    int64_t tex_over_z_delta_fp = tex_over_z2_fp - tex_over_z1_fp;
    int32_t bright_delta = (int32_t)right_brightness - (int32_t)left_brightness;

    int profile_collect_stats = (ctx && ctx->profile_collect_stats);
    uint64_t wall_t0 = 0;
    if (profile_collect_stats) {
        ctx->workload_stats.wall_segments++;
        wall_t0 = SDL_GetPerformanceCounter();
    }

    draw_wall_rasterize_segment(ctx,
                                draw_start, draw_end, scr_x1, span,
                                inv_z1_fp, inv_z_delta_fp,
                                tex_over_z1_fp, tex_over_z_delta_fp,
                                (int32_t)left_brightness, bright_delta,
                                top, bot, texture,
                                valand, valshift, horand, totalyoff, fromtile,
                                tex_id, wall_height_for_tex, d6_max);
    if (profile_collect_stats) {
        ctx->workload_stats.ticks_wall += SDL_GetPerformanceCounter() - wall_t0;
    }
}

void renderer_draw_wall(int32_t x1, int32_t z1, int32_t x2, int32_t z2,
                        int16_t top, int16_t bot,
                        const uint8_t *texture, int16_t tex_start,
                        int16_t tex_end, int16_t left_brightness, int16_t right_brightness,
                        uint8_t valand, uint8_t valshift, int16_t horand,
                        int16_t totalyoff, int16_t fromtile,
                        int16_t tex_id, int16_t wall_height_for_tex,
                        int16_t d6_max)
{
    RenderSliceContext ctx;
    render_slice_context_init(&ctx, g_renderer.left_clip, g_renderer.right_clip,
                              g_renderer.top_clip, g_renderer.bot_clip);
    ctx.wall_top_clip = g_renderer.wall_top_clip;
    ctx.wall_bot_clip = g_renderer.wall_bot_clip;
    ctx.cur_wall_pal = g_renderer.cur_wall_pal;
    renderer_draw_wall_ctx(&ctx, x1, z1, x2, z2, top, bot, texture, tex_start, tex_end,
                           left_brightness, right_brightness, valand, valshift, horand,
                           totalyoff, fromtile, tex_id, wall_height_for_tex, d6_max);
}

/* -----------------------------------------------------------------------
 * Sky ceiling span (screen-space)
 *
 * Used for synthesized backdrop hole-fill spans. Explicit roof polygons
 * are still rendered by the normal floor/roof span path.
 * ----------------------------------------------------------------------- */
static void renderer_draw_sky_ceiling_span_ctx(RenderSliceContext *ctx,
                                               int16_t y, int16_t x_left, int16_t x_right)
{
    RendererState *rs = &g_renderer;
    uint8_t *buf = renderer_active_buf();
    uint32_t *rgb = renderer_active_rgb();
    uint16_t *cwbuf = renderer_active_cw();
    if (!buf || !rgb || !cwbuf) return;
    if (y < 0 || y >= rs->height) return;
    if (y < (int)ctx->top_clip || y > (int)ctx->bot_clip) return;

    int xl = (x_left < ctx->left_clip) ? ctx->left_clip : x_left;
    int xr = (x_right >= ctx->right_clip) ? ctx->right_clip - 1 : x_right;
    if (xl > xr) return;
    renderer_pick_mark_row_span(ctx, y, xl, xr, RENDERER_PICK_ZONE_NONE);

#if !RENDER_SKY
    {
        const uint32_t sky_px = RENDER_RGB_CLEAR_SKY_PIXEL;
        const uint16_t sky_cw = 0x0EEEu;
        int w = rs->width;
        size_t row = (size_t)y * (size_t)w;
        for (int x = xl; x <= xr; x++) {
            size_t p = row + (size_t)x;
            buf[p] = 0;
            if (g_renderer_rgb_raster_expand)
                rgb[p] = sky_px;
            renderer_cw_store_xy(cwbuf, x, y, w, rs->height, sky_cw);
        }
    }
#else
    {
        int w = rs->width;
        int h = rs->height;
        int16_t angpos = rs->sky_frame_angpos;
        const int32_t sky_pan_period_fp = SKY_PAN_WIDTH << 16;
        int32_t u0_fp = (int32_t)(((int64_t)(angpos & 8191) * ((int64_t)SKY_PAN_WIDTH << 16)) / 8192);
        int sky_h = h;
        int sky_view_cols;
        if (w != s_cached_sky_view_cols_w) {
            int center_x = (w * 47) / 96;
            int left_px = center_x;
            int right_px = (w - 1) - center_x;
            const double focal_px = (double)(64 * renderer_proj_x_scale_px());
            const double hfov =
                atan((double)left_px / focal_px) +
                atan((double)right_px / focal_px);
            double sky_cols_f = ((double)SKY_PAN_WIDTH * hfov) / (2.0 * 3.14159265358979323846);
            sky_view_cols = (int)(sky_cols_f + 0.5);
            if (sky_view_cols < 1) sky_view_cols = 1;
            if (sky_view_cols > SKY_PAN_WIDTH) sky_view_cols = SKY_PAN_WIDTH;
            s_cached_sky_view_cols = sky_view_cols;
            s_cached_sky_view_cols_w = w;
        } else {
            sky_view_cols = s_cached_sky_view_cols;
        }

        const int32_t sx_step_fp = (int32_t)(((int64_t)sky_view_cols << 16) / (int64_t)w);
        int32_t sx_fp = (int32_t)((int64_t)xl * (int64_t)sx_step_fp);

        size_t row = (size_t)y * (size_t)w;

        if (s_sky_pixels && s_sky_mode == 0) {
            int th = s_sky_tex_h;
            int tw = s_sky_tex_w;
            const int32_t tw_scale_fp = ((int32_t)tw << 16) / SKY_PAN_WIDTH;
            const int64_t r_den = (int64_t)(sky_h > 1 ? sky_h - 1 : 1);
            int bpc = s_sky_bytes_per_col;
            size_t row_stride = (size_t)tw * 2u;
            int rpix = (int)((int64_t)y * (th - 1) / r_den);
            if (rpix < 0) rpix = 0;
            if (rpix >= th) rpix = th - 1;
            const size_t sky_lim = s_sky_data_bytes;
            const uint8_t *sky_px = s_sky_pixels;
            if (s_sky_row_major) {
                const size_t rm_base = (size_t)rpix * row_stride;
                for (int x = xl; x <= xr; x++) {
                    int32_t pan_fp = u0_fp + sx_fp;
                    if (pan_fp >= sky_pan_period_fp) pan_fp -= sky_pan_period_fp;
                    int c = (int)(((int64_t)(pan_fp >> 16) * (int64_t)tw_scale_fp) >> 16);
                    if (c >= tw) c = tw - 1;
                    size_t off = rm_base + (size_t)c * 2u;
                    if (off + 2u > sky_lim) { sx_fp += sx_step_fp; continue; }
                    uint16_t cw12 = (uint16_t)((sky_px[off] << 8) | sky_px[off + 1u]);
                    size_t p = row + (size_t)x;
                    buf[p] = 0;
                    if (g_renderer_rgb_raster_expand)
                        rgb[p] = amiga12_to_argb(cw12);
                    renderer_cw_store_xy(cwbuf, x, y, w, h, cw12);
                    sx_fp += sx_step_fp;
                }
            } else {
                const size_t cm_base = (size_t)rpix * 2u;
                for (int x = xl; x <= xr; x++) {
                    int32_t pan_fp = u0_fp + sx_fp;
                    if (pan_fp >= sky_pan_period_fp) pan_fp -= sky_pan_period_fp;
                    int c = (int)(((int64_t)(pan_fp >> 16) * (int64_t)tw_scale_fp) >> 16);
                    if (c >= tw) c = tw - 1;
                    size_t off = (size_t)c * (size_t)bpc + cm_base;
                    if (off + 2u > sky_lim) { sx_fp += sx_step_fp; continue; }
                    uint16_t cw12 = (uint16_t)((sky_px[off] << 8) | sky_px[off + 1u]);
                    size_t p = row + (size_t)x;
                    buf[p] = 0;
                    if (g_renderer_rgb_raster_expand)
                        rgb[p] = amiga12_to_argb(cw12);
                    renderer_cw_store_xy(cwbuf, x, y, w, h, cw12);
                    sx_fp += sx_step_fp;
                }
            }
        } else if (s_sky_pixels && s_sky_mode == 1) {
            int th = s_sky_tex_h;
            int tw = s_sky_tex_w;
            const int32_t tw_scale_fp = ((int32_t)tw << 16) / SKY_PAN_WIDTH;
            const int64_t v_den = (int64_t)(sky_h > 1 ? sky_h - 1 : 1);
            int v = (int)((int64_t)y * (th - 1) / v_den);
            if (v < 0) v = 0;
            if (v >= th) v = th - 1;
            const size_t v_base = (size_t)v * (size_t)tw;
            const size_t sky1_lim = (size_t)tw * (size_t)th;
            const uint8_t *sky_px1 = s_sky_pixels;
            for (int x = xl; x <= xr; x++) {
                int32_t pan_fp = u0_fp + sx_fp;
                if (pan_fp >= sky_pan_period_fp) pan_fp -= sky_pan_period_fp;
                int tu = (int)(((int64_t)(pan_fp >> 16) * (int64_t)tw_scale_fp) >> 16);
                if (tu >= tw) tu = tw - 1;
                size_t toff = v_base + (size_t)tu;
                if (toff >= sky1_lim) {
                    sx_fp += sx_step_fp;
                    continue;
                }
                uint8_t idx = sky_px1[toff];
                size_t p = row + (size_t)x;
                buf[p] = 0;
                if (g_renderer_rgb_raster_expand)
                    rgb[p] = s_sky_argb[idx];
                renderer_cw_store_xy(cwbuf, x, y, w, h, s_sky_cw[idx]);
                sx_fp += sx_step_fp;
            }
        } else {
            const int32_t shade_u_scale_fp = ((int32_t)40 << 16) / SKY_PAN_WIDTH;
            int t = (y * 255) / (sky_h > 1 ? sky_h - 1 : 1);
            for (int x = xl; x <= xr; x++) {
                int32_t pan_fp = u0_fp + sx_fp;
                if (pan_fp >= sky_pan_period_fp) pan_fp -= sky_pan_period_fp;
                int u = (int)(pan_fp >> 16);
                int shade = t + (int)(((int64_t)u * (int64_t)shade_u_scale_fp) >> 16);
                if (shade > 255) shade = 255;
                int r = (shade * 40) / 255;
                int g = (shade * 90) / 255;
                int b = 30 + (shade * 200) / 255;
                uint32_t px = RENDER_RGB_RASTER_PIXEL(((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b);
                size_t p = row + (size_t)x;
                buf[p] = 0;
                if (g_renderer_rgb_raster_expand)
                    rgb[p] = px;
                renderer_cw_store_xy(cwbuf, x, y, w, h, argb_to_amiga12(px));
                sx_fp += sx_step_fp;
            }
        }
    }
#endif
}

/* -----------------------------------------------------------------------
 * Floor/ceiling span rendering
 *
 * Translated from BumpMap.s / AB3DI.s itsafloordraw.
 *
 * Draws a horizontal span of floor or ceiling at a given height.
 * ----------------------------------------------------------------------- */
static void renderer_draw_floor_span_ctx(RenderSliceContext *ctx,
                                         int16_t y, int16_t x_left, int16_t x_right,
                                         int16_t shade_x_left, int16_t shade_x_right,
                                         int32_t floor_height, const uint8_t *texture, const uint8_t *floor_pal,
                                         int16_t brightness, int16_t left_brightness, int16_t right_brightness,
                                         int16_t use_gouraud,
                                         int16_t scaleval, int is_water,
                                         int16_t water_rows_left)
{
    (void)water_rows_left;
    /* Translated from AB3DI.s pastfloorbright (line 6657).
     *
     * Key insight: The Amiga packs UV into d5.w = ((V & 63) << 8) | (U & 63)
     * and samples texture at index d5.w * 4 = V*1024 + U*4.
     * This means the 256-byte-wide texture is sampled at every 4th texel.
     *
     * Step computation (per-pixel):
     *   d1 = dist * cosval (U step across screen)
     *   d2 = -dist * sinval (V step across screen)
     *
     * Starting position involves centering with 3/4 factors plus camera pos.
     */
    RendererState *rs = &g_renderer;
    FloorDrawCommon floor_common;
    FloorRowMath row_math;
    uint8_t *buf = renderer_active_buf();
    uint32_t *rgb = renderer_active_rgb();
    uint16_t *cwbuf = renderer_active_cw();
    if (!buf || !rgb || !cwbuf) return;
    if (y < 0 || y >= rs->height) return;
    /* Clip to active slice vertical band (full height for column-threaded workers). */
    if (y < (int)ctx->top_clip || y > (int)ctx->bot_clip) return;

    int xl = (x_left < ctx->left_clip) ? ctx->left_clip : x_left;
    int xr = (x_right >= ctx->right_clip) ? ctx->right_clip - 1 : x_right;
    if (xl > xr) return;

    if (ctx->foreground_floor_occlude_top) {
        const int16_t *foreground_floor_occlude_top = ctx->foreground_floor_occlude_top;
        int seg_start = -1;

        for (int x = xl; x <= xr + 1; x++) {
            int visible = 0;
            if (x <= xr) {
                int occ_top = (int)foreground_floor_occlude_top[x];
                visible = (y < occ_top);
            }

            if (visible) {
                if (seg_start < 0) seg_start = x;
            } else if (seg_start >= 0) {
                const int16_t *saved_occlude = ctx->foreground_floor_occlude_top;
                ctx->foreground_floor_occlude_top = NULL;
                renderer_draw_floor_span_ctx(ctx, y, (int16_t)seg_start, (int16_t)(x - 1),
                                             shade_x_left, shade_x_right,
                                             floor_height, texture, floor_pal,
                                             brightness, left_brightness, right_brightness,
                                             use_gouraud, scaleval, is_water, water_rows_left);
                ctx->foreground_floor_occlude_top = saved_occlude;
                seg_start = -1;
            }
        }
        return;
    }

    renderer_pick_mark_row_span(ctx, y, xl, xr, renderer_pick_zone_encode(ctx->pick_zone_id));

    const int expand = g_renderer_rgb_raster_expand;

    renderer_floor_prepare_common(&floor_common, rs, floor_height, scaleval);

    int center = rs->height / 2;  /* Match wall/floor projection center */
    int row_dist = renderer_floor_row_dist_from_screen_y(y, center);
    int abs_row_dist = (row_dist < 0) ? -row_dist : row_dist;
    const int use_gour = (use_gouraud != 0 && !is_water);
    /* Amiga floor distance attenuation uses 80-line screen-space row offsets (d0 around center=40). */
    int row80 = (int)(((int64_t)y * 80) / ((rs->height > 0) ? rs->height : 1));
    int row_dist80 = row80 - 40;
    if (row_dist80 == 0) row_dist80 = (y < center) ? -1 : 1;

    /* Fallback grayscale only (when no floor palette is loaded). */
    int zone_d6 = brightness * 2;

    /* UV distance path (kept separate from brightness falloff):
     * Use a half-step conversion on the world-space height so floor/ceiling texel
     * size lands between the old over-zoomed and the recent under-zoomed result. */
    int32_t dist = renderer_floor_compute_dist(rs, &floor_common, row_dist);

    /* Amiga formula: d6 = (dist >> 7) + zone_bright. Higher d6 = darker. */
    int amiga_d6 = (dist >> 7) + zone_d6;
    if (amiga_d6 < 0) amiga_d6 = 0;
    if (amiga_d6 > 64) amiga_d6 = 64;
    int gray = (64 - amiga_d6) * 255 / 64;

    renderer_floor_prepare_row_math(&row_math, rs, &floor_common, row_dist);

    /* Step per pixel:
     * - At base render width, keep the original mapping.
     * - When supersampling increases internal width, scale step down so output FOV stays unchanged. */
    int w = floor_common.width;
    int32_t u_step = row_math.u_step;
    int32_t v_step = row_math.v_step;

    const uint8_t *pal_lut_src = floor_pal ? floor_pal : rs->floor_pal;
    int floor_pal_level = 0;
    if (pal_lut_src && !use_gour) {
        int bright_idx = brightness + 5 + (dist >> 8);
        if (bright_idx < 0) bright_idx = 0;
        if (bright_idx > 28) bright_idx = 28;
        floor_pal_level = floor_bright_level_table[bright_idx];
    }

    const uint16_t *span_cw = NULL;
    const int use_span_lut = (texture != NULL && pal_lut_src != NULL && !is_water && !use_gour);
    if (use_span_lut) {
        floor_span_prepare_pal_cache(ctx, pal_lut_src, floor_pal_level, &span_cw);
    }

    /* UV accumulators: 32-bit wrapping is sufficient - we only need (fp>>16)&63 for tile coords.
     * The 64-bit start computation above already handles the large intermediate values;
     * truncating to 32 bits here halves the per-pixel addition cost. */
    uint32_t u_fp = row_math.u_base;
    uint32_t v_fp = row_math.v_base;
    if (xl > 0) {
        u_fp += (uint32_t)((int64_t)xl * (int64_t)u_step);
        v_fp += (uint32_t)((int64_t)xl * (int64_t)v_step);
    }

    int32_t water_refr_y_off_fp = 0;
    if (is_water) {
        /* AB3DI texturedwater computes vertical refraction once per scanline. */
        int sin_idx = (((dist << 7) + (int32_t)g_water_wtan_draw) & 8191);
        /* AB3DI texturedwater reads directly from bigsine (range ~[-32767,32767]).
         * Our shared sin_lookup table is half-scale (~[-16384,16384]), so double
         * here to match Amiga water wobble without affecting non-water systems. */
        int32_t sine = (int32_t)sin_lookup((int16_t)sin_idx) << 1;
        int32_t den = dist + 300;
        /* AB3DI texturedwater:
         *   d0 = ((sin(dst*128 + wtan) / (dst+300)) >> 6) + 2 */
        int32_t refr_y_off_fp = 0;
        if (den != 0) {
            refr_y_off_fp = (int32_t)(((int64_t)sine << 8) / ((int64_t)den << 6));
        }
        refr_y_off_fp += (2 << 8);

        /* AB3DI disttobot clamp is in 80-line view space:
         *   if (d0 >= disttobot) d0 = disttobot-1 */
        {
            int amiga_y = (int)(((int64_t)y * 80) / ((rs->height > 0) ? rs->height : 1));
            int disttobot = 79 - amiga_y;
            if (disttobot < 1) disttobot = 1;
            if (refr_y_off_fp >= (disttobot << 8))
                refr_y_off_fp = (disttobot << 8) - 1;
        }
        if (row_dist < 0) refr_y_off_fp = -refr_y_off_fp; /* AB3DI: if (above) d0 = -d0 */

        /* Convert Amiga 80-line offset to current render height. */
        if (rs->height != 80) {
            refr_y_off_fp = (int32_t)(((int64_t)refr_y_off_fp * rs->height) / 80);
        }

        water_refr_y_off_fp = refr_y_off_fp;
    }

    const int span_len = xr - xl + 1;
    if (ctx && ctx->profile_collect_stats) {
        ctx->workload_stats.floor_spans++;
        ctx->workload_stats.floor_pixels += (uint64_t)span_len;
        if (is_water) {
            ctx->workload_stats.water_pixels += (uint64_t)span_len;
        }
    }
    uint8_t *row8 = buf + (size_t)y * w + (size_t)xl;
    uint32_t *row32 = rgb + (size_t)y * w + (size_t)xl;
    uint16_t *row16 = cwbuf + (size_t)y * w + (size_t)xl;
    const int cw_col_major = AB3D_CW_COL_MAJOR;
    const int floor_cw_nt = (cw_col_major && !is_water);
    const size_t cw_step_x = renderer_cw_step_x(rs->height);
    size_t row16_idx = renderer_cw_index_xy(xl, y, w, rs->height);
#define FLOOR_CW_STORE_ROW(V_) \
    do { \
        if (!cw_col_major) { \
            *row16++ = (V_); \
        } else { \
            if (floor_cw_nt) AB3D_NT_STORE_U16(&cwbuf[row16_idx], (V_)); \
            else cwbuf[row16_idx] = (V_); \
            row16_idx += cw_step_x; \
            row16++; \
        } \
    } while (0)
#define FLOOR_CW_STORE_P16(V_) \
    do { \
        if (!cw_col_major) { \
            *p16++ = (V_); \
        } else { \
            if (floor_cw_nt) AB3D_NT_STORE_U16(&cwbuf[p16_idx], (V_)); \
            else cwbuf[p16_idx] = (V_); \
            p16_idx += cw_step_x; \
            p16++; \
        } \
    } while (0)
    size_t water_refr_base0 = 0;
    size_t water_refr_base1 = 0;
    size_t water_refr_cw_base0 = 0;
    size_t water_refr_cw_base1 = 0;
    int water_refr_frac = 0;
    int water_has_next_refr = 0;
    const int water_has_back_buffers = (rs->rgb_back_buffer && rs->cw_back_buffer);
    if (is_water) {
        int32_t refr_y_fp = ((int32_t)y << 8) + water_refr_y_off_fp;
        int refr_y = (int)(refr_y_fp >> 8);
        water_refr_frac = (int)(refr_y_fp & 0xFF);
        int refr_y_next = refr_y + 1;
        if (refr_y < 0) refr_y = 0;
        if (refr_y >= rs->height) refr_y = rs->height - 1;
        if (refr_y_next < 0) refr_y_next = 0;
        if (refr_y_next >= rs->height) refr_y_next = rs->height - 1;
        water_has_next_refr = (water_refr_frac > 0 && refr_y_next != refr_y);
        water_refr_base0 = (size_t)refr_y * (size_t)rs->width;
        water_refr_base1 = (size_t)refr_y_next * (size_t)rs->width;
        water_refr_cw_base0 = renderer_cw_index_xy(xl, refr_y, rs->width, rs->height);
        water_refr_cw_base1 = renderer_cw_index_xy(xl, refr_y_next, rs->width, rs->height);
    }

    /* Fast non-water textured path: no per-pixel branching. */
    if (use_span_lut && span_cw) {
        if (expand) {
            uint8_t *p8 = row8;
            uint32_t *p32 = row32;
            uint16_t *p16 = row16;
            size_t p16_idx = row16_idx;
            const uint32_t *span_rgb = ctx->floor_span_rgb_cache[floor_pal_level];
            uint32_t u_pf = u_fp + (uint32_t)u_step * 8u;
            uint32_t v_pf = v_fp + (uint32_t)v_step * 8u;
            int i = 0;
#if AB3D_HAVE_SSE2
            if (!cw_col_major) for (; i <= span_len - 4; i += 4) {
                if (i + 8 < span_len) {
                    AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                }
                uint8_t t0 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;
                uint8_t t1 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;
                uint8_t t2 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;
                uint8_t t3 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;
                u_pf += (uint32_t)u_step * 4u;
                v_pf += (uint32_t)v_step * 4u;

                uint32_t px4[4] = {
                    span_rgb[t0], span_rgb[t1], span_rgb[t2], span_rgb[t3]
                };
                _mm_storeu_si128((__m128i *)(void *)p32,
                                   _mm_loadu_si128((const __m128i *)(const void *)px4));
                uint16_t c0 = span_cw[t0];
                uint16_t c1 = span_cw[t1];
                uint16_t c2 = span_cw[t2];
                uint16_t c3 = span_cw[t3];
                __m128i qcw = _mm_set_epi16(0, 0, 0, 0,
                    (int16_t)c3, (int16_t)c2, (int16_t)c1, (int16_t)c0);
                _mm_storel_epi64((__m128i *)(void *)p16, qcw);
                *(uint32_t *)(void *)p8 = 0x01010101u;
                p8 += 4;
                p32 += 4;
                p16 += 4;
            }
#endif
            if (cw_col_major) {
                for (; i <= span_len - 4; i += 4) {
                    if (i + 8 < span_len) {
                        AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                    }
                    uint8_t t0 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step;
                    v_fp += (uint32_t)v_step;
                    uint8_t t1 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step;
                    v_fp += (uint32_t)v_step;
                    uint8_t t2 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step;
                    v_fp += (uint32_t)v_step;
                    uint8_t t3 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step;
                    v_fp += (uint32_t)v_step;
                    u_pf += (uint32_t)u_step * 4u;
                    v_pf += (uint32_t)v_step * 4u;

                    *(uint32_t *)(void *)p8 = 0x01010101u;
                    p8 += 4;
                    p32[0] = span_rgb[t0];
                    p32[1] = span_rgb[t1];
                    p32[2] = span_rgb[t2];
                    p32[3] = span_rgb[t3];
                    p32 += 4;

                    AB3D_NT_STORE_U16(&cwbuf[p16_idx], span_cw[t0]); p16_idx += cw_step_x;
                    AB3D_NT_STORE_U16(&cwbuf[p16_idx], span_cw[t1]); p16_idx += cw_step_x;
                    AB3D_NT_STORE_U16(&cwbuf[p16_idx], span_cw[t2]); p16_idx += cw_step_x;
                    AB3D_NT_STORE_U16(&cwbuf[p16_idx], span_cw[t3]); p16_idx += cw_step_x;
                    p16 += 4;
                }
            }
            for (; i < span_len; i++) {
                if (i + 8 < span_len) {
                    AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                }
                uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;
                u_pf += (uint32_t)u_step;
                v_pf += (uint32_t)v_step;

                *p8++ = 1;
                *p32++ = span_rgb[texel];
                FLOOR_CW_STORE_P16(span_cw[texel]);
            }
        } else {
            uint8_t *p8 = row8;
            uint16_t *p16 = row16;
            size_t p16_idx = row16_idx;
            uint32_t u_pf = u_fp + (uint32_t)u_step * 8u;
            uint32_t v_pf = v_fp + (uint32_t)v_step * 8u;
            int i = 0;
#if AB3D_HAVE_SSE2
            if (!cw_col_major) for (; i <= span_len - 4; i += 4) {
                if (i + 8 < span_len) {
                    AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                }
                uint8_t t0 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;
                uint8_t t1 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;
                uint8_t t2 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;
                uint8_t t3 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;
                u_pf += (uint32_t)u_step * 4u;
                v_pf += (uint32_t)v_step * 4u;

                uint16_t c0 = span_cw[t0];
                uint16_t c1 = span_cw[t1];
                uint16_t c2 = span_cw[t2];
                uint16_t c3 = span_cw[t3];
                __m128i qcw = _mm_set_epi16(0, 0, 0, 0,
                    (int16_t)c3, (int16_t)c2, (int16_t)c1, (int16_t)c0);
                _mm_storel_epi64((__m128i *)(void *)p16, qcw);
                *(uint32_t *)(void *)p8 = 0x01010101u;
                p8 += 4;
                p16 += 4;
            }
#endif
            if (cw_col_major) {
                for (; i <= span_len - 4; i += 4) {
                    if (i + 8 < span_len) {
                        AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                    }
                    uint8_t t0 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step;
                    v_fp += (uint32_t)v_step;
                    uint8_t t1 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step;
                    v_fp += (uint32_t)v_step;
                    uint8_t t2 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step;
                    v_fp += (uint32_t)v_step;
                    uint8_t t3 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step;
                    v_fp += (uint32_t)v_step;
                    u_pf += (uint32_t)u_step * 4u;
                    v_pf += (uint32_t)v_step * 4u;

                    *(uint32_t *)(void *)p8 = 0x01010101u;
                    p8 += 4;
                    AB3D_NT_STORE_U16(&cwbuf[p16_idx], span_cw[t0]); p16_idx += cw_step_x;
                    AB3D_NT_STORE_U16(&cwbuf[p16_idx], span_cw[t1]); p16_idx += cw_step_x;
                    AB3D_NT_STORE_U16(&cwbuf[p16_idx], span_cw[t2]); p16_idx += cw_step_x;
                    AB3D_NT_STORE_U16(&cwbuf[p16_idx], span_cw[t3]); p16_idx += cw_step_x;
                    p16 += 4;
                }
            }
            for (; i < span_len; i++) {
                if (i + 8 < span_len) {
                    AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                }
                uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;
                u_pf += (uint32_t)u_step;
                v_pf += (uint32_t)v_step;

                *p8++ = 1;
                FLOOR_CW_STORE_P16(span_cw[texel]);
            }
        }
        return;
    }

    if (is_water) {
        RendererZoneTraceFloorStats *water_trace_stats = (ctx && ctx->active_floor_trace_stats &&
                                  ctx->active_floor_trace_stats->active &&
                                  ctx->active_floor_trace_stats->is_water)
            ? ctx->active_floor_trace_stats
            : NULL;
        uint8_t *p8 = row8;
        uint32_t *p32 = row32;
        uint16_t *p16 = row16;
        size_t p16_idx = row16_idx;
        size_t bg_i0 = water_refr_base0 + (size_t)xl;
        size_t bg_i1 = water_refr_base1 + (size_t)xl;
        size_t bg_cw_i0 = water_refr_cw_base0;
        size_t bg_cw_i1 = water_refr_cw_base1;
        const int water_has_file = (g_water_file && g_water_file_size >= 65536u);
        const int water_has_brighten = (g_water_brighten && g_water_brighten_size >= 512u);
        const int water_has_brighten_cw_lut = (g_water_brighten_cw_lut_ready && g_water_file_level_max <= 9u);
        const uint8_t *water_phase_lut = g_water_level_phase_lut_ready
            ? g_water_level_phase_lut[g_water_src_phase & 7u]
            : NULL;
        const uint8_t *water_blend_nibble_lut = amiga12_blend_nibble_lut_ready
            ? amiga12_blend_nibble_lut[water_refr_frac & 0xFFu]
            : NULL;
        uint32_t dist_off = (((uint32_t)dist) & 0xFF00u) << 1;
        if (dist_off > (uint32_t)(12 * 512)) dist_off = (uint32_t)(12 * 512);

        if (!expand && water_has_file && water_has_brighten) {
            size_t water_file_lookup_span = (size_t)g_water_src_off + 65536u;
            size_t water_brighten_lookup_span = (size_t)dist_off + ((size_t)g_water_file_level_max << 9) + 512u;
            if ((water_phase_lut || g_water_file_size >= water_file_lookup_span) &&
                (water_has_brighten_cw_lut || g_water_brighten_size >= water_brighten_lookup_span)) {
                const uint8_t *water_file = water_phase_lut ? NULL : (g_water_file + (size_t)g_water_src_off);
                const uint8_t *water_brighten = g_water_brighten + (size_t)dist_off;
                const int water_brighten_row_base = (int)(dist_off >> 9);
                const int sample0_needs_back_buffer = water_has_back_buffers
                    ? renderer_water_span_needs_back_buffer(buf + bg_i0, span_len)
                    : 0;
                const int sample1_needs_back_buffer = (water_has_back_buffers && water_has_next_refr)
                    ? renderer_water_span_needs_back_buffer(buf + bg_i1, span_len)
                    : 0;
                renderer_zone_trace_floor_stats_note_water_span(water_trace_stats,
                                                                (uint64_t)span_len,
                                                                1,
                                                                water_has_next_refr);

                /* Common stock-data path: every WaterFile/Brighten lookup is in-range,
                 * so avoid the per-pixel bounds checks and RGB-side bookkeeping. */
                if (!water_has_next_refr) {
                    if (!sample0_needs_back_buffer) {
                        if (water_phase_lut && water_has_brighten_cw_lut) {
                            for (int i = 0; i < span_len; i++, bg_i0++, bg_i1++, bg_cw_i0 += cw_step_x, bg_cw_i1 += cw_step_x) {
                                uint8_t u8 = (uint8_t)((u_fp >> 16) & 0xFFu);
                                uint8_t v8 = (uint8_t)((v_fp >> 16) & 0xFFu);
                                u_fp += (uint32_t)u_step;
                                v_fp += (uint32_t)v_step;

                                uint16_t d5_word = (uint16_t)(((uint16_t)v8 << 8) | (uint16_t)u8);
                                uint16_t water_d5 = (uint16_t)(d5_word + (uint16_t)g_water_off);
                                water_d5 &= 0x3F3Fu;
                                uint16_t water_idx = (uint16_t)((((water_d5 >> 8) & 0x3Fu) << 6) | (water_d5 & 0x3Fu));
                                uint8_t water_level = water_phase_lut[water_idx];
                                uint16_t out_cw = g_water_brighten_cw_lut[water_brighten_row_base + water_level][cwbuf[bg_cw_i0] & 0xFFu];

                                *p8++ = 4;
                                p32++;
                                FLOOR_CW_STORE_P16(out_cw);
                            }
                        } else {
                            for (int i = 0; i < span_len; i++, bg_i0++, bg_i1++, bg_cw_i0 += cw_step_x, bg_cw_i1 += cw_step_x) {
                                uint8_t u8 = (uint8_t)((u_fp >> 16) & 0xFFu);
                                uint8_t v8 = (uint8_t)((v_fp >> 16) & 0xFFu);
                                u_fp += (uint32_t)u_step;
                                v_fp += (uint32_t)v_step;

                                uint16_t d5_word = (uint16_t)(((uint16_t)v8 << 8) | (uint16_t)u8);
                                uint16_t water_d5 = (uint16_t)(d5_word + (uint16_t)g_water_off);
                                water_d5 &= 0x3F3Fu;
                                uint16_t water_idx = (uint16_t)((((water_d5 >> 8) & 0x3Fu) << 6) | (water_d5 & 0x3Fu));
                                uint8_t water_level = water_phase_lut
                                    ? water_phase_lut[water_idx]
                                    : water_file[(size_t)water_d5 << 2];

                                uint16_t bg_cw0 = cwbuf[bg_cw_i0];
                                uint16_t out_cw;
                                if (water_has_brighten_cw_lut) {
                                    out_cw = g_water_brighten_cw_lut[water_brighten_row_base + water_level][bg_cw0 & 0xFFu];
                                } else {
                                    size_t bi0 = ((size_t)water_level << 9) + (size_t)(bg_cw0 & 0xFFu) * 2u;
                                    out_cw = (uint16_t)((water_brighten[bi0] << 8) | water_brighten[bi0 + 1u]);
                                }

                                *p8++ = 4;
                                p32++;
                                FLOOR_CW_STORE_P16(out_cw);
                            }
                        }
                    } else {
                        for (int i = 0; i < span_len; i++, bg_i0++, bg_i1++, bg_cw_i0 += cw_step_x, bg_cw_i1 += cw_step_x) {
                            uint8_t u8 = (uint8_t)((u_fp >> 16) & 0xFFu);
                            uint8_t v8 = (uint8_t)((v_fp >> 16) & 0xFFu);
                            u_fp += (uint32_t)u_step;
                            v_fp += (uint32_t)v_step;

                            uint16_t d5_word = (uint16_t)(((uint16_t)v8 << 8) | (uint16_t)u8);
                            uint16_t water_d5 = (uint16_t)(d5_word + (uint16_t)g_water_off);
                            water_d5 &= 0x3F3Fu;
                            uint16_t water_idx = (uint16_t)((((water_d5 >> 8) & 0x3Fu) << 6) | (water_d5 & 0x3Fu));
                            uint8_t water_level = water_phase_lut
                                ? water_phase_lut[water_idx]
                                : water_file[(size_t)water_d5 << 2];

                            uint16_t bg_cw0 = cwbuf[bg_cw_i0];
                            int sample0_back_buffer = ((buf[bg_i0] == 0 || buf[bg_i0] == 4) && water_has_back_buffers);
                            if (sample0_back_buffer) {
                                bg_cw0 = rs->cw_back_buffer[bg_cw_i0];
                            }
                            renderer_zone_trace_floor_stats_note_water_sample0(water_trace_stats, sample0_back_buffer);

                            uint16_t out_cw;
                            if (water_has_brighten_cw_lut) {
                                out_cw = g_water_brighten_cw_lut[water_brighten_row_base + water_level][bg_cw0 & 0xFFu];
                            } else {
                                size_t bi0 = ((size_t)water_level << 9) + (size_t)(bg_cw0 & 0xFFu) * 2u;
                                out_cw = (uint16_t)((water_brighten[bi0] << 8) | water_brighten[bi0 + 1u]);
                            }

                            *p8++ = 4;
                            p32++;
                            FLOOR_CW_STORE_P16(out_cw);
                        }
                    }
                } else {
                    if (!sample0_needs_back_buffer && !sample1_needs_back_buffer) {
                        if (water_phase_lut && water_has_brighten_cw_lut && water_blend_nibble_lut) {
                            for (int i = 0; i < span_len; i++, bg_i0++, bg_i1++, bg_cw_i0 += cw_step_x, bg_cw_i1 += cw_step_x) {
                                uint8_t u8 = (uint8_t)((u_fp >> 16) & 0xFFu);
                                uint8_t v8 = (uint8_t)((v_fp >> 16) & 0xFFu);
                                u_fp += (uint32_t)u_step;
                                v_fp += (uint32_t)v_step;

                                uint16_t d5_word = (uint16_t)(((uint16_t)v8 << 8) | (uint16_t)u8);
                                uint16_t water_d5 = (uint16_t)(d5_word + (uint16_t)g_water_off);
                                water_d5 &= 0x3F3Fu;
                                uint16_t water_idx = (uint16_t)((((water_d5 >> 8) & 0x3Fu) << 6) | (water_d5 & 0x3Fu));
                                uint8_t water_level = water_phase_lut[water_idx];
                                const uint16_t *brighten_row = g_water_brighten_cw_lut[water_brighten_row_base + water_level];
                                uint16_t out_cw0 = brighten_row[cwbuf[bg_cw_i0] & 0xFFu];
                                uint16_t out_cw1 = brighten_row[cwbuf[bg_cw_i1] & 0xFFu];
                                uint16_t r4 = water_blend_nibble_lut[((((uint32_t)out_cw0 >> 8) & 0xFu) << 4) | (((uint32_t)out_cw1 >> 8) & 0xFu)];
                                uint16_t g4 = water_blend_nibble_lut[((((uint32_t)out_cw0 >> 4) & 0xFu) << 4) | (((uint32_t)out_cw1 >> 4) & 0xFu)];
                                uint16_t b4 = water_blend_nibble_lut[(((uint32_t)out_cw0 & 0xFu) << 4) | ((uint32_t)out_cw1 & 0xFu)];

                                *p8++ = 4;
                                p32++;
                                FLOOR_CW_STORE_P16((uint16_t)((r4 << 8) | (g4 << 4) | b4));
                            }
                        } else {
                            for (int i = 0; i < span_len; i++, bg_i0++, bg_i1++, bg_cw_i0 += cw_step_x, bg_cw_i1 += cw_step_x) {
                                uint8_t u8 = (uint8_t)((u_fp >> 16) & 0xFFu);
                                uint8_t v8 = (uint8_t)((v_fp >> 16) & 0xFFu);
                                u_fp += (uint32_t)u_step;
                                v_fp += (uint32_t)v_step;

                                uint16_t d5_word = (uint16_t)(((uint16_t)v8 << 8) | (uint16_t)u8);
                                uint16_t water_d5 = (uint16_t)(d5_word + (uint16_t)g_water_off);
                                water_d5 &= 0x3F3Fu;
                                uint16_t water_idx = (uint16_t)((((water_d5 >> 8) & 0x3Fu) << 6) | (water_d5 & 0x3Fu));
                                uint8_t water_level = water_phase_lut
                                    ? water_phase_lut[water_idx]
                                    : water_file[(size_t)water_d5 << 2];

                                uint16_t bg_cw0 = cwbuf[bg_cw_i0];
                                uint16_t bg_cw1 = cwbuf[bg_cw_i1];
                                uint16_t out_cw0;
                                uint16_t out_cw1;
                                if (water_has_brighten_cw_lut) {
                                    const uint16_t *brighten_row = g_water_brighten_cw_lut[water_brighten_row_base + water_level];
                                    out_cw0 = brighten_row[bg_cw0 & 0xFFu];
                                    out_cw1 = brighten_row[bg_cw1 & 0xFFu];
                                } else {
                                    size_t bi0 = ((size_t)water_level << 9) + (size_t)(bg_cw0 & 0xFFu) * 2u;
                                    size_t bi1 = ((size_t)water_level << 9) + (size_t)(bg_cw1 & 0xFFu) * 2u;
                                    out_cw0 = (uint16_t)((water_brighten[bi0] << 8) | water_brighten[bi0 + 1u]);
                                    out_cw1 = (uint16_t)((water_brighten[bi1] << 8) | water_brighten[bi1 + 1u]);
                                }

                                *p8++ = 4;
                                p32++;
                                if (water_blend_nibble_lut) {
                                    uint16_t r4 = water_blend_nibble_lut[((((uint32_t)out_cw0 >> 8) & 0xFu) << 4) | (((uint32_t)out_cw1 >> 8) & 0xFu)];
                                    uint16_t g4 = water_blend_nibble_lut[((((uint32_t)out_cw0 >> 4) & 0xFu) << 4) | (((uint32_t)out_cw1 >> 4) & 0xFu)];
                                    uint16_t b4 = water_blend_nibble_lut[(((uint32_t)out_cw0 & 0xFu) << 4) | ((uint32_t)out_cw1 & 0xFu)];
                                    FLOOR_CW_STORE_P16((uint16_t)((r4 << 8) | (g4 << 4) | b4));
                                } else {
                                    FLOOR_CW_STORE_P16(blend_amiga12(out_cw0, out_cw1, (uint32_t)water_refr_frac));
                                }
                            }
                        }
                    } else {
                        for (int i = 0; i < span_len; i++, bg_i0++, bg_i1++, bg_cw_i0 += cw_step_x, bg_cw_i1 += cw_step_x) {
                            uint8_t u8 = (uint8_t)((u_fp >> 16) & 0xFFu);
                            uint8_t v8 = (uint8_t)((v_fp >> 16) & 0xFFu);
                            u_fp += (uint32_t)u_step;
                            v_fp += (uint32_t)v_step;

                            uint16_t d5_word = (uint16_t)(((uint16_t)v8 << 8) | (uint16_t)u8);
                            uint16_t water_d5 = (uint16_t)(d5_word + (uint16_t)g_water_off);
                            water_d5 &= 0x3F3Fu;
                            uint16_t water_idx = (uint16_t)((((water_d5 >> 8) & 0x3Fu) << 6) | (water_d5 & 0x3Fu));
                            uint8_t water_level = water_phase_lut
                                ? water_phase_lut[water_idx]
                                : water_file[(size_t)water_d5 << 2];

                            uint16_t bg_cw0 = cwbuf[bg_cw_i0];
                            int sample0_back_buffer = ((buf[bg_i0] == 0 || buf[bg_i0] == 4) && water_has_back_buffers);
                            if (sample0_back_buffer) {
                                bg_cw0 = rs->cw_back_buffer[bg_cw_i0];
                            }
                            renderer_zone_trace_floor_stats_note_water_sample0(water_trace_stats, sample0_back_buffer);
                            uint16_t bg_cw1 = cwbuf[bg_cw_i1];
                            int sample1_back_buffer = ((buf[bg_i1] == 0 || buf[bg_i1] == 4) && water_has_back_buffers);
                            if (sample1_back_buffer) {
                                bg_cw1 = rs->cw_back_buffer[bg_cw_i1];
                            }
                            renderer_zone_trace_floor_stats_note_water_sample1(water_trace_stats, sample1_back_buffer);

                            uint16_t out_cw0;
                            uint16_t out_cw1;
                            if (water_has_brighten_cw_lut) {
                                const uint16_t *brighten_row = g_water_brighten_cw_lut[water_brighten_row_base + water_level];
                                out_cw0 = brighten_row[bg_cw0 & 0xFFu];
                                out_cw1 = brighten_row[bg_cw1 & 0xFFu];
                            } else {
                                size_t bi0 = ((size_t)water_level << 9) + (size_t)(bg_cw0 & 0xFFu) * 2u;
                                size_t bi1 = ((size_t)water_level << 9) + (size_t)(bg_cw1 & 0xFFu) * 2u;
                                out_cw0 = (uint16_t)((water_brighten[bi0] << 8) | water_brighten[bi0 + 1u]);
                                out_cw1 = (uint16_t)((water_brighten[bi1] << 8) | water_brighten[bi1 + 1u]);
                            }

                            *p8++ = 4;
                            p32++;
                            if (water_blend_nibble_lut) {
                                uint16_t r4 = water_blend_nibble_lut[((((uint32_t)out_cw0 >> 8) & 0xFu) << 4) | (((uint32_t)out_cw1 >> 8) & 0xFu)];
                                uint16_t g4 = water_blend_nibble_lut[((((uint32_t)out_cw0 >> 4) & 0xFu) << 4) | (((uint32_t)out_cw1 >> 4) & 0xFu)];
                                uint16_t b4 = water_blend_nibble_lut[(((uint32_t)out_cw0 & 0xFu) << 4) | ((uint32_t)out_cw1 & 0xFu)];
                                FLOOR_CW_STORE_P16((uint16_t)((r4 << 8) | (g4 << 4) | b4));
                            } else {
                                FLOOR_CW_STORE_P16(blend_amiga12(out_cw0, out_cw1, (uint32_t)water_refr_frac));
                            }
                        }
                    }
                }
                return;
            }
        }

        renderer_zone_trace_floor_stats_note_water_span(water_trace_stats,
                                                        (uint64_t)span_len,
                                                        0,
                                                        water_has_next_refr);

        for (int i = 0; i < span_len; i++, bg_i0++, bg_i1++, bg_cw_i0 += cw_step_x, bg_cw_i1 += cw_step_x) {
            uint8_t u8 = (uint8_t)((u_fp >> 16) & 0xFFu);
            uint8_t v8 = (uint8_t)((v_fp >> 16) & 0xFFu);
            u_fp += (uint32_t)u_step;
            v_fp += (uint32_t)v_step;

            /* Amiga-style textured water: refract existing pixels instead of drawing a solid color. */
            uint16_t d5_word = (uint16_t)(((uint16_t)v8 << 8) | (uint16_t)u8);
            uint16_t water_d5 = (uint16_t)(d5_word + (uint16_t)g_water_off);
            water_d5 &= 0x3F3Fu;
            uint16_t water_idx = (uint16_t)((((water_d5 >> 8) & 0x3Fu) << 6) | (water_d5 & 0x3Fu));
            uint8_t water_level = 0;

            if (water_has_file) {
                if (water_phase_lut) {
                    water_level = water_phase_lut[water_idx];
                } else {
                    size_t wi = ((size_t)water_d5 << 2) + (size_t)g_water_src_off;
                    if (wi < g_water_file_size) {
                        water_level = g_water_file[wi];
                    }
                }
            } else if (texture) {
                uint32_t tex_idx = (((uint32_t)v8 & 63u) << 10) | (((uint32_t)water_d5 & 63u) << 2);
                water_level = (uint8_t)(texture[tex_idx] >> 4);
            }

            uint16_t bg_cw0 = cwbuf[bg_cw_i0];
            uint32_t bg0 = 0;
            if (expand)
                bg0 = rgb[bg_i0];
            if ((buf[bg_i0] == 0 || buf[bg_i0] == 4) && water_has_back_buffers) {
                /* AB3DI texturedwater samples from display memory while floor lines are streamed.
                 * When refraction points at rows not written yet this frame, those pixels still
                 * contain prior-frame values; mirror that by sampling back-buffer content.
                 * Also avoid water-over-water feedback between adjacent zones in this port's
                 * per-zone streaming path by treating existing water-tagged pixels the same way. */
                bg_cw0 = rs->cw_back_buffer[bg_cw_i0];
                if (expand)
                    bg0 = rs->rgb_back_buffer[bg_i0];
                renderer_zone_trace_floor_stats_note_water_sample0(water_trace_stats, 1);
            }
            uint8_t bg_sample0 = (uint8_t)(bg_cw0 & 0xFFu);

            uint32_t bg1 = bg0;
            uint16_t bg_cw1 = bg_cw0;
            uint8_t bg_sample1 = bg_sample0;
            if (water_has_next_refr) {
                bg_cw1 = cwbuf[bg_cw_i1];
                if (expand)
                    bg1 = rgb[bg_i1];
                if ((buf[bg_i1] == 0 || buf[bg_i1] == 4) && water_has_back_buffers) {
                    bg_cw1 = rs->cw_back_buffer[bg_cw_i1];
                    if (expand)
                        bg1 = rs->rgb_back_buffer[bg_i1];
                    renderer_zone_trace_floor_stats_note_water_sample1(water_trace_stats, 1);
                }
                bg_sample1 = (uint8_t)(bg_cw1 & 0xFFu);
            }

            uint32_t out = 0;
            uint16_t out_cw;
            if (water_has_brighten) {
                /* Amiga texturedwater:
                 *   d0 = WaterFile word, then move.b sampled_pixel_lowbyte,d0,
                 *   output = brightentab[d0]. */
                size_t bi0 = (size_t)dist_off + ((size_t)water_level << 9) + (size_t)bg_sample0 * 2u;
                uint32_t out0 = 0;
                uint16_t out_cw0;
                if (bi0 + 1u < g_water_brighten_size) {
                    out_cw0 = (uint16_t)((g_water_brighten[bi0] << 8) | g_water_brighten[bi0 + 1u]);
                    if (expand)
                        out0 = amiga12_to_argb(out_cw0);
                } else {
                    out_cw0 = bg_cw0;
                    if (expand)
                        out0 = bg0;
                }

                if (water_has_next_refr) {
                    size_t bi1 = (size_t)dist_off + ((size_t)water_level << 9) + (size_t)bg_sample1 * 2u;
                    uint32_t out1 = 0;
                    uint16_t out_cw1;
                    if (bi1 + 1u < g_water_brighten_size) {
                        out_cw1 = (uint16_t)((g_water_brighten[bi1] << 8) | g_water_brighten[bi1 + 1u]);
                        if (expand)
                            out1 = amiga12_to_argb(out_cw1);
                    } else {
                        out_cw1 = bg_cw1;
                        if (expand)
                            out1 = bg1;
                    }
                    if (expand) {
                        out = blend_argb(out0, out1, (uint32_t)water_refr_frac);
                        out_cw = argb_to_amiga12(out);
                    } else {
                        out_cw = blend_amiga12(out_cw0, out_cw1, (uint32_t)water_refr_frac);
                    }
                } else {
                    if (expand)
                        out = out0;
                    out_cw = out_cw0;
                }
            } else {
                /* Fallback when brighten table is missing. */
                if (expand) {
                    uint32_t bg = water_has_next_refr ? blend_argb(bg0, bg1, (uint32_t)water_refr_frac) : bg0;
                    uint32_t br = (bg >> 16) & 0xFFu;
                    uint32_t bg_g = (bg >> 8) & 0xFFu;
                    uint32_t bb = bg & 0xFFu;
                    uint32_t shade = 160u + ((uint32_t)water_level * 6u);
                    uint32_t r = (br * ((shade > 96u) ? (shade - 96u) : 0u)) >> 8;
                    uint32_t g = (bg_g * shade) >> 8;
                    uint32_t b = (bb * (shade + 28u)) >> 8;
                    b += 10u;
                    if (r > 255u) r = 255u;
                    if (g > 255u) g = 255u;
                    if (b > 255u) b = 255u;
                    out = blend_argb(bg, RENDER_RGB_RASTER_PIXEL((r << 16) | (g << 8) | b), 120u);
                    out_cw = argb_to_amiga12(out);
                } else {
                    uint16_t bg_cw = water_has_next_refr
                        ? blend_amiga12(bg_cw0, bg_cw1, (uint32_t)water_refr_frac)
                        : bg_cw0;
                    uint32_t br = (((uint32_t)bg_cw >> 8) & 0xFu) * 0x11u;
                    uint32_t bg_g = (((uint32_t)bg_cw >> 4) & 0xFu) * 0x11u;
                    uint32_t bb = ((uint32_t)bg_cw & 0xFu) * 0x11u;
                    uint32_t shade = 160u + ((uint32_t)water_level * 6u);
                    uint32_t r = (br * ((shade > 96u) ? (shade - 96u) : 0u)) >> 8;
                    uint32_t g = (bg_g * shade) >> 8;
                    uint32_t b = (bb * (shade + 28u)) >> 8;
                    b += 10u;
                    if (r > 255u) r = 255u;
                    if (g > 255u) g = 255u;
                    if (b > 255u) b = 255u;
                    {
                        uint16_t tint_cw = (uint16_t)(((uint16_t)byte_to_nibble_lut[r] << 8)
                                                    | ((uint16_t)byte_to_nibble_lut[g] << 4)
                                                    |  (uint16_t)byte_to_nibble_lut[b]);
                        out_cw = blend_amiga12(bg_cw, tint_cw, 120u);
                    }
                }
            }

            *p8++ = 4; /* water tag: avoid wall-join fill smearing */
            if (expand)
                *p32++ = out;
            else
                p32++;
            FLOOR_CW_STORE_P16(out_cw);
        }
        return;
    }

    /* Gouraud floor path (Amiga GOURSEL floor/roof): interpolate brightness levels across the span. */
    int32_t gour_level_fp = 0;
    int32_t gour_level_step = 0;
    int32_t gour_bright_fp = 0;
    int32_t gour_bright_step = 0;
    const int dist_add = (dist >> 7);
    if (use_gour) {
        const int need_gour_bright = (g_debug_floor_gouraud_only || !(texture && pal_lut_src));
        /* Use the full original polygon span (x_left..x_right) for the gradient, then
         * advance to the clipped draw start xl.  Using the clipped width (xr-xl) here
         * was the column-strip shading artifact: strips with a clipped left edge would
         * start at the wrong brightness and step at the wrong rate. */
        int full_span_w = (int)shade_x_right - (int)shade_x_left;
        int left_level = left_brightness + dist_add;
        int right_level = right_brightness + dist_add;
        if (left_level < 0) left_level = 0;
        if (right_level < 0) right_level = 0;
        left_level >>= 1;
        right_level >>= 1;
        if (left_level > 14) left_level = 14;
        if (right_level > 14) right_level = 14;
        gour_level_step  = (full_span_w > 0) ? (int32_t)(((int64_t)(right_level      - left_level)      << 16) / full_span_w) : 0;
        /* Advance from the polygon's left edge to the clipped draw start. */
        int clip_advance = xl - (int)shade_x_left;
        gour_level_fp  = ((int32_t)left_level      << 16) + (int32_t)((int64_t)gour_level_step  * clip_advance);
        if (need_gour_bright) {
            gour_bright_step = (full_span_w > 0) ? (int32_t)(((int64_t)(right_brightness - left_brightness) << 16) / full_span_w) : 0;
            gour_bright_fp = ((int32_t)left_brightness << 16) + (int32_t)((int64_t)gour_bright_step * clip_advance);
        }
    }

    if (g_debug_floor_gouraud_only && !is_water) {
        for (int i = 0; i < span_len; i++) {
            int g;
            if (use_gour) {
                int gour_bright = (int)(gour_bright_fp >> 16);
                gour_bright_fp += gour_bright_step;
                /* Raw Gouraud term only: map signed brightness around mid-gray. */
                g = 128 + gour_bright * 4;
                if (g < 0) g = 0;
                if (g > 255) g = 255;
            } else {
                g = 96;
            }
            uint32_t argb = RENDER_RGB_RASTER_PIXEL(((uint32_t)g << 16) | ((uint32_t)g << 8) | (uint32_t)g);
            *row8++ = 1;
            RASTER_PUT_PP(&row32, argb);
            FLOOR_CW_STORE_ROW(argb_to_amiga12(argb));
        }
        return;
    }

    if (texture && pal_lut_src) {
        uint8_t *p8 = row8;
        uint32_t *p32 = row32;
        uint16_t *p16 = row16;
        size_t p16_idx = row16_idx;

        if (use_gour) {
            floor_span_prepare_pal_cache_all_levels(ctx, pal_lut_src);
            const uint16_t *const *gour_cw_levels = ctx->floor_pal_cw_levels;
            const uint32_t *const *gour_rgb_levels = ctx->floor_pal_rgb_levels;
            uint32_t u_pf = u_fp + (uint32_t)u_step * 8u;
            uint32_t v_pf = v_fp + (uint32_t)v_step * 8u;

            /* Exact fast lane: if Gouraud level does not change across the span,
             * run the same constant-level texture loop as non-Gouraud. */
            if (gour_level_step == 0) {
                int level = (int)(gour_level_fp >> 16);
                const uint16_t *span_cw;
                const uint32_t *span_rgb;

                if (level < 0) level = 0;
                if (level >= FLOOR_PAL_LEVEL_COUNT) level = FLOOR_PAL_LEVEL_COUNT - 1;
                span_cw = gour_cw_levels[level];
                span_rgb = gour_rgb_levels[level];

                if (expand) {
                    int i = 0;
                    for (; i <= span_len - 4; i += 4) {
                        if (i + 8 < span_len) {
                            AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                        }

                        uint8_t t0 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                        u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                        uint8_t t1 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                        u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                        uint8_t t2 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                        u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                        uint8_t t3 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                        u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                        u_pf += (uint32_t)u_step * 4u;
                        v_pf += (uint32_t)v_step * 4u;

                        *p8++ = 1; *p32++ = span_rgb[t0]; FLOOR_CW_STORE_P16(span_cw[t0]);
                        *p8++ = 1; *p32++ = span_rgb[t1]; FLOOR_CW_STORE_P16(span_cw[t1]);
                        *p8++ = 1; *p32++ = span_rgb[t2]; FLOOR_CW_STORE_P16(span_cw[t2]);
                        *p8++ = 1; *p32++ = span_rgb[t3]; FLOOR_CW_STORE_P16(span_cw[t3]);
                    }

                    for (; i < span_len; i++) {
                        if (i + 8 < span_len) {
                            AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                        }

                        uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                        u_fp += (uint32_t)u_step;
                        v_fp += (uint32_t)v_step;
                        u_pf += (uint32_t)u_step;
                        v_pf += (uint32_t)v_step;

                        *p8++ = 1;
                        *p32++ = span_rgb[texel];
                        FLOOR_CW_STORE_P16(span_cw[texel]);
                    }
                } else {
                    int i = 0;
                    for (; i <= span_len - 4; i += 4) {
                        if (i + 8 < span_len) {
                            AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                        }

                        uint8_t t0 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                        u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                        uint8_t t1 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                        u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                        uint8_t t2 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                        u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                        uint8_t t3 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                        u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                        u_pf += (uint32_t)u_step * 4u;
                        v_pf += (uint32_t)v_step * 4u;

                        *p8++ = 1; FLOOR_CW_STORE_P16(span_cw[t0]);
                        *p8++ = 1; FLOOR_CW_STORE_P16(span_cw[t1]);
                        *p8++ = 1; FLOOR_CW_STORE_P16(span_cw[t2]);
                        *p8++ = 1; FLOOR_CW_STORE_P16(span_cw[t3]);
                    }

                    for (; i < span_len; i++) {
                        if (i + 8 < span_len) {
                            AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                        }

                        uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                        u_fp += (uint32_t)u_step;
                        v_fp += (uint32_t)v_step;
                        u_pf += (uint32_t)u_step;
                        v_pf += (uint32_t)v_step;

                        *p8++ = 1;
                        FLOOR_CW_STORE_P16(span_cw[texel]);
                    }
                }
                return;
            }

            if (expand) {
                int i = 0;
                for (; i <= span_len - 4; i += 4) {
                    if (i + 8 < span_len) {
                        AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                    }

                    int g0 = (int)(gour_level_fp >> 16); gour_level_fp += gour_level_step;
                    int g1 = (int)(gour_level_fp >> 16); gour_level_fp += gour_level_step;
                    int g2 = (int)(gour_level_fp >> 16); gour_level_fp += gour_level_step;
                    int g3 = (int)(gour_level_fp >> 16); gour_level_fp += gour_level_step;
                    if (g0 < 0) g0 = 0; if (g0 >= FLOOR_PAL_LEVEL_COUNT) g0 = FLOOR_PAL_LEVEL_COUNT - 1;
                    if (g1 < 0) g1 = 0; if (g1 >= FLOOR_PAL_LEVEL_COUNT) g1 = FLOOR_PAL_LEVEL_COUNT - 1;
                    if (g2 < 0) g2 = 0; if (g2 >= FLOOR_PAL_LEVEL_COUNT) g2 = FLOOR_PAL_LEVEL_COUNT - 1;
                    if (g3 < 0) g3 = 0; if (g3 >= FLOOR_PAL_LEVEL_COUNT) g3 = FLOOR_PAL_LEVEL_COUNT - 1;

                    uint8_t t0 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                    uint8_t t1 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                    uint8_t t2 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                    uint8_t t3 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                    u_pf += (uint32_t)u_step * 4u;
                    v_pf += (uint32_t)v_step * 4u;

                    *p8++ = 1; *p32++ = gour_rgb_levels[g0][t0]; FLOOR_CW_STORE_P16(gour_cw_levels[g0][t0]);
                    *p8++ = 1; *p32++ = gour_rgb_levels[g1][t1]; FLOOR_CW_STORE_P16(gour_cw_levels[g1][t1]);
                    *p8++ = 1; *p32++ = gour_rgb_levels[g2][t2]; FLOOR_CW_STORE_P16(gour_cw_levels[g2][t2]);
                    *p8++ = 1; *p32++ = gour_rgb_levels[g3][t3]; FLOOR_CW_STORE_P16(gour_cw_levels[g3][t3]);
                }

                for (; i < span_len; i++) {
                    if (i + 8 < span_len) {
                        AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                    }

                    int gour_level = (int)(gour_level_fp >> 16);
                    gour_level_fp += gour_level_step;
                    if (gour_level < 0) gour_level = 0;
                    if (gour_level >= FLOOR_PAL_LEVEL_COUNT) gour_level = FLOOR_PAL_LEVEL_COUNT - 1;

                    uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step;
                    v_fp += (uint32_t)v_step;
                    u_pf += (uint32_t)u_step;
                    v_pf += (uint32_t)v_step;

                    *p8++ = 1;
                    *p32++ = gour_rgb_levels[gour_level][texel];
                    FLOOR_CW_STORE_P16(gour_cw_levels[gour_level][texel]);
                }
            } else {
                int i = 0;
                for (; i <= span_len - 4; i += 4) {
                    if (i + 8 < span_len) {
                        AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                    }

                    int g0 = (int)(gour_level_fp >> 16); gour_level_fp += gour_level_step;
                    int g1 = (int)(gour_level_fp >> 16); gour_level_fp += gour_level_step;
                    int g2 = (int)(gour_level_fp >> 16); gour_level_fp += gour_level_step;
                    int g3 = (int)(gour_level_fp >> 16); gour_level_fp += gour_level_step;
                    if (g0 < 0) g0 = 0; if (g0 >= FLOOR_PAL_LEVEL_COUNT) g0 = FLOOR_PAL_LEVEL_COUNT - 1;
                    if (g1 < 0) g1 = 0; if (g1 >= FLOOR_PAL_LEVEL_COUNT) g1 = FLOOR_PAL_LEVEL_COUNT - 1;
                    if (g2 < 0) g2 = 0; if (g2 >= FLOOR_PAL_LEVEL_COUNT) g2 = FLOOR_PAL_LEVEL_COUNT - 1;
                    if (g3 < 0) g3 = 0; if (g3 >= FLOOR_PAL_LEVEL_COUNT) g3 = FLOOR_PAL_LEVEL_COUNT - 1;

                    uint8_t t0 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                    uint8_t t1 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                    uint8_t t2 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                    uint8_t t3 = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step; v_fp += (uint32_t)v_step;
                    u_pf += (uint32_t)u_step * 4u;
                    v_pf += (uint32_t)v_step * 4u;

                    *p8++ = 1; FLOOR_CW_STORE_P16(gour_cw_levels[g0][t0]);
                    *p8++ = 1; FLOOR_CW_STORE_P16(gour_cw_levels[g1][t1]);
                    *p8++ = 1; FLOOR_CW_STORE_P16(gour_cw_levels[g2][t2]);
                    *p8++ = 1; FLOOR_CW_STORE_P16(gour_cw_levels[g3][t3]);
                }

                for (; i < span_len; i++) {
                    if (i + 8 < span_len) {
                        AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                    }

                    int gour_level = (int)(gour_level_fp >> 16);
                    gour_level_fp += gour_level_step;
                    if (gour_level < 0) gour_level = 0;
                    if (gour_level >= FLOOR_PAL_LEVEL_COUNT) gour_level = FLOOR_PAL_LEVEL_COUNT - 1;

                    uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step;
                    v_fp += (uint32_t)v_step;
                    u_pf += (uint32_t)u_step;
                    v_pf += (uint32_t)v_step;

                    *p8++ = 1;
                    FLOOR_CW_STORE_P16(gour_cw_levels[gour_level][texel]);
                }
            }
            return;
        }

        {
            const uint8_t *lut = pal_lut_src + floor_pal_level * 512;
            if (expand) {
                uint32_t u_pf = u_fp + (uint32_t)u_step * 8u;
                uint32_t v_pf = v_fp + (uint32_t)v_step * 8u;
                for (int i = 0; i < span_len; i++) {
                    if (i + 8 < span_len) {
                        AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                    }
                    uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step;
                    v_fp += (uint32_t)v_step;
                    u_pf += (uint32_t)u_step;
                    v_pf += (uint32_t)v_step;
                    uint16_t out_cw = (uint16_t)((lut[texel * 2] << 8) | lut[texel * 2 + 1]);

                    *p8++ = 1;
                    *p32++ = amiga12_to_argb(out_cw);
                    FLOOR_CW_STORE_P16(out_cw);
                }
            } else {
                uint32_t u_pf = u_fp + (uint32_t)u_step * 8u;
                uint32_t v_pf = v_fp + (uint32_t)v_step * 8u;
                for (int i = 0; i < span_len; i++) {
                    if (i + 8 < span_len) {
                        AB3D_PREFETCH_READ(&texture[((u_pf >> 14) & 0xFCu) | ((v_pf >> 6) & 0xFC00u)]);
                    }
                    uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                    u_fp += (uint32_t)u_step;
                    v_fp += (uint32_t)v_step;
                    u_pf += (uint32_t)u_step;
                    v_pf += (uint32_t)v_step;
                    uint16_t out_cw = (uint16_t)((lut[texel * 2] << 8) | lut[texel * 2 + 1]);

                    *p8++ = 1;
                    FLOOR_CW_STORE_P16(out_cw);
                }
            }
            return;
        }
    }

    if (texture) {
        uint8_t *p8 = row8;
        uint32_t *p32 = row32;
        uint16_t *p16 = row16;
        size_t p16_idx = row16_idx;

        if (use_gour) {
            for (int i = 0; i < span_len; i++) {
                int gour_bright = (int)(gour_bright_fp >> 16);
                gour_bright_fp += gour_bright_step;
                int d6 = dist_add + (gour_bright * 2);
                if (d6 < 0) d6 = 0;
                if (d6 > 64) d6 = 64;
                int gour_gray = (64 - d6) * 255 / 64;

                uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;

                int lit = ((int)texel * gour_gray) >> 8;
                uint32_t argb = RENDER_RGB_RASTER_PIXEL(((uint32_t)lit << 16) | ((uint32_t)lit << 8) | (uint32_t)lit);
                *p8++ = 1;
                RASTER_PUT_PP(&p32, argb);
                FLOOR_CW_STORE_P16(argb_to_amiga12(argb));
            }
            return;
        }

        for (int i = 0; i < span_len; i++) {
            uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
            u_fp += (uint32_t)u_step;
            v_fp += (uint32_t)v_step;
            int lit = ((int)texel * gray) >> 8;
            uint32_t argb = RENDER_RGB_RASTER_PIXEL(((uint32_t)lit << 16) | ((uint32_t)lit << 8) | (uint32_t)lit);
            *row8++ = 1;
            RASTER_PUT_PP(&row32, argb);
            FLOOR_CW_STORE_ROW(argb_to_amiga12(argb));
        }
        return;
    }

    if (use_gour) {
        for (int i = 0; i < span_len; i++) {
            int gour_bright = (int)(gour_bright_fp >> 16);
            gour_bright_fp += gour_bright_step;
            int d6 = dist_add + (gour_bright * 2);
            if (d6 < 0) d6 = 0;
            if (d6 > 64) d6 = 64;
            int g = (64 - d6) * 255 / 64;
            uint32_t argb = RENDER_RGB_RASTER_PIXEL(((uint32_t)g << 16) | ((uint32_t)g << 8) | (uint32_t)g);
            *row8++ = 1;
            RASTER_PUT_PP(&row32, argb);
            FLOOR_CW_STORE_ROW(argb_to_amiga12(argb));
        }
    } else {
        uint32_t argb = RENDER_RGB_RASTER_PIXEL(((uint32_t)gray << 16) | ((uint32_t)gray << 8) | (uint32_t)gray);
        uint16_t out_cw = argb_to_amiga12(argb);
        for (int i = 0; i < span_len; i++) {
            *row8++ = 1;
            RASTER_PUT_PP(&row32, argb);
            FLOOR_CW_STORE_ROW(out_cw);
        }
    }

#undef FLOOR_CW_STORE_ROW
#undef FLOOR_CW_STORE_P16
}

void renderer_draw_floor_span(int16_t y, int16_t x_left, int16_t x_right,
                              int32_t floor_height, const uint8_t *texture, const uint8_t *floor_pal,
                              int16_t brightness, int16_t left_brightness, int16_t right_brightness,
                              int16_t use_gouraud,
                              int16_t scaleval, int is_water,
                              int16_t water_rows_left)
{
    RenderSliceContext ctx;
    render_slice_context_init(&ctx, g_renderer.left_clip, g_renderer.right_clip,
                              g_renderer.top_clip, g_renderer.bot_clip);
    renderer_draw_floor_span_ctx(&ctx, y, x_left, x_right, x_left, x_right, floor_height, texture, floor_pal,
                                 brightness, left_brightness, right_brightness, use_gouraud,
                                 scaleval, is_water, water_rows_left);
}

typedef struct {
    int16_t le;
    int16_t re;
    int32_t u_step;
    int32_t v_step;
    uint32_t u_base;
    uint32_t v_base;
    int32_t bright_base_fp;
    int32_t bright_step_fp;
    int32_t bright_col_fp;
    int16_t bright_term;
    const uint16_t *span_cw;
    const uint32_t *span_rgb;
    uint32_t u_col;
    uint32_t v_col;
    int16_t next_x;
} FloorRowFast;

static AB3D_THREAD_LOCAL FloorRowFast *g_floor_fast_rows_scratch = NULL;
static AB3D_THREAD_LOCAL int g_floor_fast_rows_capacity = 0;
static AB3D_THREAD_LOCAL int16_t *g_floor_fast_col_top_scratch = NULL;
static AB3D_THREAD_LOCAL int16_t *g_floor_fast_col_bot_scratch = NULL;
static AB3D_THREAD_LOCAL int *g_floor_fast_next_top_scratch = NULL;
static AB3D_THREAD_LOCAL int *g_floor_fast_next_bot_scratch = NULL;
static AB3D_THREAD_LOCAL int *g_floor_fast_cov_diff_scratch = NULL;
static AB3D_THREAD_LOCAL int g_floor_fast_col_capacity = 0;

static int renderer_floor_fast_ensure_rows_capacity(int row_count)
{
    int new_cap;
    FloorRowFast *new_rows;

    if (row_count <= 0) return 0;
    if (g_floor_fast_rows_scratch && row_count <= g_floor_fast_rows_capacity) return 1;

    new_cap = (g_floor_fast_rows_capacity > 0) ? g_floor_fast_rows_capacity : 64;
    while (new_cap < row_count) {
        if (new_cap > INT_MAX / 2) {
            new_cap = row_count;
            break;
        }
        new_cap *= 2;
    }

    new_rows = (FloorRowFast *)realloc(g_floor_fast_rows_scratch, (size_t)new_cap * sizeof(*new_rows));
    if (!new_rows) return 0;

    g_floor_fast_rows_scratch = new_rows;
    g_floor_fast_rows_capacity = new_cap;
    return 1;
}

static int renderer_floor_fast_ensure_cols_capacity(int col_count)
{
    int new_cap;

    if (col_count <= 0) return 0;
    if (g_floor_fast_col_top_scratch &&
        g_floor_fast_col_bot_scratch &&
        g_floor_fast_next_top_scratch &&
        g_floor_fast_next_bot_scratch &&
        g_floor_fast_cov_diff_scratch &&
        col_count <= g_floor_fast_col_capacity) {
        return 1;
    }

    new_cap = (g_floor_fast_col_capacity > 0) ? g_floor_fast_col_capacity : 64;
    while (new_cap < col_count) {
        if (new_cap > INT_MAX / 2) {
            new_cap = col_count;
            break;
        }
        new_cap *= 2;
    }

    {
        int16_t *new_top = (int16_t *)realloc(g_floor_fast_col_top_scratch, (size_t)new_cap * sizeof(*new_top));
        if (!new_top) return 0;
        g_floor_fast_col_top_scratch = new_top;
    }
    {
        int16_t *new_bot = (int16_t *)realloc(g_floor_fast_col_bot_scratch, (size_t)new_cap * sizeof(*new_bot));
        if (!new_bot) return 0;
        g_floor_fast_col_bot_scratch = new_bot;
    }
    {
        int *new_next_top = (int *)realloc(g_floor_fast_next_top_scratch, (size_t)(new_cap + 1) * sizeof(*new_next_top));
        if (!new_next_top) return 0;
        g_floor_fast_next_top_scratch = new_next_top;
    }
    {
        int *new_next_bot = (int *)realloc(g_floor_fast_next_bot_scratch, (size_t)(new_cap + 1) * sizeof(*new_next_bot));
        if (!new_next_bot) return 0;
        g_floor_fast_next_bot_scratch = new_next_bot;
    }
    {
        int *new_cov_diff = (int *)realloc(g_floor_fast_cov_diff_scratch, (size_t)(new_cap + 1) * sizeof(*new_cov_diff));
        if (!new_cov_diff) return 0;
        g_floor_fast_cov_diff_scratch = new_cov_diff;
    }

    g_floor_fast_col_capacity = new_cap;
    return 1;
}

static void renderer_floor_fast_release_scratch(void)
{
    free(g_floor_fast_rows_scratch);
    g_floor_fast_rows_scratch = NULL;
    g_floor_fast_rows_capacity = 0;

    free(g_floor_fast_col_top_scratch);
    g_floor_fast_col_top_scratch = NULL;
    free(g_floor_fast_col_bot_scratch);
    g_floor_fast_col_bot_scratch = NULL;
    free(g_floor_fast_next_top_scratch);
    g_floor_fast_next_top_scratch = NULL;
    free(g_floor_fast_next_bot_scratch);
    g_floor_fast_next_bot_scratch = NULL;
    free(g_floor_fast_cov_diff_scratch);
    g_floor_fast_cov_diff_scratch = NULL;
    g_floor_fast_col_capacity = 0;
}

static void renderer_floor_fast_seed_rows(FloorRowFast *rows,
                                          int row_count,
                                          int use_gour_floor)
{
    if (!rows || row_count <= 0) return;

    for (int row_idx = 0; row_idx < row_count; row_idx++) {
        FloorRowFast *row = &rows[row_idx];
        int x_seed;
        uint32_t x_u32;
        if (row->le > row->re) continue;
        x_seed = (int)row->le;
        x_u32 = (uint32_t)x_seed;
        row->u_col = row->u_base + x_u32 * (uint32_t)row->u_step;
        row->v_col = row->v_base + x_u32 * (uint32_t)row->v_step;
        if (use_gour_floor) {
            row->bright_col_fp = row->bright_base_fp;
        } else {
            row->bright_col_fp = 0;
        }
        row->next_x = (int16_t)x_seed;
    }
}

static uint64_t renderer_draw_floor_fast_column(const RenderSliceContext *ctx,
                                                FloorRowFast *rows,
                                                const uint16_t *const *gour_cw_levels,
                                                const uint32_t *const *gour_rgb_levels,
                                                const uint8_t *texture,
                                                uint8_t *buf,
                                                uint32_t *rgb,
                                                uint16_t *cwbuf,
                                                int w,
                                                int h,
                                                int y_base,
                                                int pick_capture_active,
                                                int assume_contiguous_coverage,
                                                int use_gour_floor,
                                                int expand,
                                                int x,
                                                int top,
                                                int bot)
{
    const int16_t *foreground_floor_occlude_top = ctx ? ctx->foreground_floor_occlude_top : NULL;
    uint64_t drawn_pixels = 0;
    int run_start = (pick_capture_active && !assume_contiguous_coverage) ? -1 : INT_MIN;
    const int x_next = x + 1;
    const int floor_pf_dist = 8;
    const size_t floor_pf_stride = (size_t)w * (size_t)floor_pf_dist;

    if (!ctx || !rows || !texture || !buf || !cwbuf) return 0;
    if (x < 0 || x >= w || top > bot) return 0;

    if (foreground_floor_occlude_top) {
        int occ_top = (int)foreground_floor_occlude_top[x];
        if (occ_top <= bot) {
            if (occ_top <= top) return 0;
            bot = occ_top - 1;
        }
    }
    if (top > bot) return 0;
    if (top < y_base) top = y_base;
    if (top > bot) return 0;

    {
        size_t cw_idx = renderer_cw_index_xy(x, top, w, h);
        size_t pix = (size_t)top * (size_t)w + (size_t)x;

        if (!use_gour_floor) {
            if (expand) {
                FloorRowFast *row = rows + (top - y_base);
                if (assume_contiguous_coverage) {
                    for (int y = top; y <= bot; y++, row++) {
                        if (row->next_x != x) {
                            int dx = x - row->next_x;
                            row->u_col += (uint32_t)dx * (uint32_t)row->u_step;
                            row->v_col += (uint32_t)dx * (uint32_t)row->v_step;
                        }
                        {
                            uint32_t u_fp = row->u_col;
                            uint32_t v_fp = row->v_col;
                            uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                            uint16_t out_cw = row->span_cw[texel];

                            buf[pix] = 1;
                            AB3D_NT_STORE_U16(&cwbuf[cw_idx], out_cw);
                            rgb[pix] = row->span_rgb[texel];
                            drawn_pixels++;
                        }
                        row->u_col += (uint32_t)row->u_step;
                        row->v_col += (uint32_t)row->v_step;
                        row->next_x = (int16_t)x_next;

                        cw_idx += 1u;
                        pix += (size_t)w;
                    }
                } else {
                    for (int y = top; y <= bot; y++, row++) {
                        int draw_here = (x >= row->le && x <= row->re);

                        if (y + floor_pf_dist <= bot) {
                            AB3D_PREFETCH_WRITE(&buf[pix + floor_pf_stride]);
                            AB3D_PREFETCH_WRITE(&rgb[pix + floor_pf_stride]);
                        }

                        if (draw_here) {
                            if (row->next_x != x) {
                                int dx = x - row->next_x;
                                row->u_col += (uint32_t)dx * (uint32_t)row->u_step;
                                row->v_col += (uint32_t)dx * (uint32_t)row->v_step;
                            }
                            uint32_t u_fp = row->u_col;
                            uint32_t v_fp = row->v_col;
                            uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                            uint16_t out_cw = row->span_cw[texel];

                            if (run_start == -1) run_start = y;
                            buf[pix] = 1;
                            AB3D_NT_STORE_U16(&cwbuf[cw_idx], out_cw);
                            rgb[pix] = row->span_rgb[texel];
                            drawn_pixels++;
                            row->u_col += (uint32_t)row->u_step;
                            row->v_col += (uint32_t)row->v_step;
                            row->next_x = (int16_t)x_next;
                        } else if (run_start >= 0) {
                            renderer_pick_mark_wall_column(ctx, x, run_start, y - 1, 0, 0, 0, 0);
                            run_start = -1;
                        }

                        cw_idx += 1u;
                        pix += (size_t)w;
                    }
                }
            } else {
                FloorRowFast *row = rows + (top - y_base);
                if (assume_contiguous_coverage) {
                    for (int y = top; y <= bot; y++, row++) {
                        if (row->next_x != x) {
                            int dx = x - row->next_x;
                            row->u_col += (uint32_t)dx * (uint32_t)row->u_step;
                            row->v_col += (uint32_t)dx * (uint32_t)row->v_step;
                        }
                        {
                            uint32_t u_fp = row->u_col;
                            uint32_t v_fp = row->v_col;
                            uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                            uint16_t out_cw = row->span_cw[texel];

                            buf[pix] = 1;
                            AB3D_NT_STORE_U16(&cwbuf[cw_idx], out_cw);
                            drawn_pixels++;
                        }
                        row->u_col += (uint32_t)row->u_step;
                        row->v_col += (uint32_t)row->v_step;
                        row->next_x = (int16_t)x_next;

                        cw_idx += 1u;
                        pix += (size_t)w;
                    }
                } else {
                    for (int y = top; y <= bot; y++, row++) {
                        int draw_here = (x >= row->le && x <= row->re);

                        if (y + floor_pf_dist <= bot) {
                            AB3D_PREFETCH_WRITE(&buf[pix + floor_pf_stride]);
                        }

                        if (draw_here) {
                            if (row->next_x != x) {
                                int dx = x - row->next_x;
                                row->u_col += (uint32_t)dx * (uint32_t)row->u_step;
                                row->v_col += (uint32_t)dx * (uint32_t)row->v_step;
                            }
                            uint32_t u_fp = row->u_col;
                            uint32_t v_fp = row->v_col;
                            uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                            uint16_t out_cw = row->span_cw[texel];

                            if (run_start == -1) run_start = y;
                            buf[pix] = 1;
                            AB3D_NT_STORE_U16(&cwbuf[cw_idx], out_cw);
                            drawn_pixels++;
                            row->u_col += (uint32_t)row->u_step;
                            row->v_col += (uint32_t)row->v_step;
                            row->next_x = (int16_t)x_next;
                        } else if (run_start >= 0) {
                            renderer_pick_mark_wall_column(ctx, x, run_start, y - 1, 0, 0, 0, 0);
                            run_start = -1;
                        }

                        cw_idx += 1u;
                        pix += (size_t)w;
                    }
                }
            }
        } else {
            if (expand) {
                FloorRowFast *row = rows + (top - y_base);
                if (assume_contiguous_coverage) {
                    for (int y = top; y <= bot; y++, row++) {
                        if (row->next_x != x) {
                            int dx = x - row->next_x;
                            row->u_col += (uint32_t)dx * (uint32_t)row->u_step;
                            row->v_col += (uint32_t)dx * (uint32_t)row->v_step;
                            row->bright_col_fp += (int32_t)(((int64_t)dx) * (int64_t)row->bright_step_fp);
                        }
                        {
                            uint32_t u_fp = row->u_col;
                            uint32_t v_fp = row->v_col;
                            uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                            int level = (int)(row->bright_col_fp >> 16);
                            uint16_t out_cw;
                            uint32_t out_rgb;

                            if (level < 0) level = 0;
                            if (level >= FLOOR_PAL_LEVEL_COUNT) level = FLOOR_PAL_LEVEL_COUNT - 1;
                            {
                                const uint16_t *cw_level = gour_cw_levels[level];
                                const uint32_t *rgb_level = gour_rgb_levels[level];
                                out_cw = cw_level ? cw_level[texel] : row->span_cw ? row->span_cw[texel] : 0;
                                out_rgb = rgb_level ? rgb_level[texel] : row->span_rgb ? row->span_rgb[texel] : amiga12_to_argb(out_cw);
                            }

                            buf[pix] = 1;
                            AB3D_NT_STORE_U16(&cwbuf[cw_idx], out_cw);
                            rgb[pix] = out_rgb;
                            drawn_pixels++;
                        }
                        row->u_col += (uint32_t)row->u_step;
                        row->v_col += (uint32_t)row->v_step;
                        row->bright_col_fp += row->bright_step_fp;
                        row->next_x = (int16_t)x_next;

                        cw_idx += 1u;
                        pix += (size_t)w;
                    }
                } else {
                    for (int y = top; y <= bot; y++, row++) {
                        int draw_here = (x >= row->le && x <= row->re);

                        if (y + floor_pf_dist <= bot) {
                            AB3D_PREFETCH_WRITE(&buf[pix + floor_pf_stride]);
                            AB3D_PREFETCH_WRITE(&rgb[pix + floor_pf_stride]);
                        }

                        if (draw_here) {
                            if (row->next_x != x) {
                                int dx = x - row->next_x;
                                row->u_col += (uint32_t)dx * (uint32_t)row->u_step;
                                row->v_col += (uint32_t)dx * (uint32_t)row->v_step;
                                row->bright_col_fp += (int32_t)(((int64_t)dx) * (int64_t)row->bright_step_fp);
                            }
                            uint32_t u_fp = row->u_col;
                            uint32_t v_fp = row->v_col;
                            uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                            int level = (int)(row->bright_col_fp >> 16);
                            uint16_t out_cw;
                            uint32_t out_rgb;

                            if (level < 0) level = 0;
                            if (level >= FLOOR_PAL_LEVEL_COUNT) level = FLOOR_PAL_LEVEL_COUNT - 1;
                            {
                                const uint16_t *cw_level = gour_cw_levels[level];
                                const uint32_t *rgb_level = gour_rgb_levels[level];
                                out_cw = cw_level ? cw_level[texel] : row->span_cw ? row->span_cw[texel] : 0;
                                out_rgb = rgb_level ? rgb_level[texel] : row->span_rgb ? row->span_rgb[texel] : amiga12_to_argb(out_cw);
                            }

                            if (run_start == -1) run_start = y;
                            buf[pix] = 1;
                            AB3D_NT_STORE_U16(&cwbuf[cw_idx], out_cw);
                            rgb[pix] = out_rgb;
                            drawn_pixels++;
                            row->u_col += (uint32_t)row->u_step;
                            row->v_col += (uint32_t)row->v_step;
                            row->bright_col_fp += row->bright_step_fp;
                            row->next_x = (int16_t)x_next;
                        } else if (run_start >= 0) {
                            renderer_pick_mark_wall_column(ctx, x, run_start, y - 1, 0, 0, 0, 0);
                            run_start = -1;
                        }

                        cw_idx += 1u;
                        pix += (size_t)w;
                    }
                }
            } else {
                FloorRowFast *row = rows + (top - y_base);
                if (assume_contiguous_coverage) {
                    for (int y = top; y <= bot; y++, row++) {
                        if (row->next_x != x) {
                            int dx = x - row->next_x;
                            row->u_col += (uint32_t)dx * (uint32_t)row->u_step;
                            row->v_col += (uint32_t)dx * (uint32_t)row->v_step;
                            row->bright_col_fp += (int32_t)(((int64_t)dx) * (int64_t)row->bright_step_fp);
                        }
                        {
                            uint32_t u_fp = row->u_col;
                            uint32_t v_fp = row->v_col;
                            uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                            int level = (int)(row->bright_col_fp >> 16);
                            uint16_t out_cw;

                            if (level < 0) level = 0;
                            if (level >= FLOOR_PAL_LEVEL_COUNT) level = FLOOR_PAL_LEVEL_COUNT - 1;
                            {
                                const uint16_t *cw_level = gour_cw_levels[level];
                                out_cw = cw_level ? cw_level[texel] : row->span_cw ? row->span_cw[texel] : 0;
                            }

                            buf[pix] = 1;
                            AB3D_NT_STORE_U16(&cwbuf[cw_idx], out_cw);
                            drawn_pixels++;
                        }
                        row->u_col += (uint32_t)row->u_step;
                        row->v_col += (uint32_t)row->v_step;
                        row->bright_col_fp += row->bright_step_fp;
                        row->next_x = (int16_t)x_next;

                        cw_idx += 1u;
                        pix += (size_t)w;
                    }
                } else {
                    for (int y = top; y <= bot; y++, row++) {
                        int draw_here = (x >= row->le && x <= row->re);

                        if (y + floor_pf_dist <= bot) {
                            AB3D_PREFETCH_WRITE(&buf[pix + floor_pf_stride]);
                        }

                        if (draw_here) {
                            if (row->next_x != x) {
                                int dx = x - row->next_x;
                                row->u_col += (uint32_t)dx * (uint32_t)row->u_step;
                                row->v_col += (uint32_t)dx * (uint32_t)row->v_step;
                                row->bright_col_fp += (int32_t)(((int64_t)dx) * (int64_t)row->bright_step_fp);
                            }
                            uint32_t u_fp = row->u_col;
                            uint32_t v_fp = row->v_col;
                            uint8_t texel = texture[((u_fp >> 14) & 0xFCu) | ((v_fp >> 6) & 0xFC00u)];
                            int level = (int)(row->bright_col_fp >> 16);
                            uint16_t out_cw;

                            if (level < 0) level = 0;
                            if (level >= FLOOR_PAL_LEVEL_COUNT) level = FLOOR_PAL_LEVEL_COUNT - 1;
                            {
                                const uint16_t *cw_level = gour_cw_levels[level];
                                out_cw = cw_level ? cw_level[texel] : row->span_cw ? row->span_cw[texel] : 0;
                            }

                            if (run_start == -1) run_start = y;
                            buf[pix] = 1;
                            AB3D_NT_STORE_U16(&cwbuf[cw_idx], out_cw);
                            drawn_pixels++;
                            row->u_col += (uint32_t)row->u_step;
                            row->v_col += (uint32_t)row->v_step;
                            row->bright_col_fp += row->bright_step_fp;
                            row->next_x = (int16_t)x_next;
                        } else if (run_start >= 0) {
                            renderer_pick_mark_wall_column(ctx, x, run_start, y - 1, 0, 0, 0, 0);
                            run_start = -1;
                        }

                        cw_idx += 1u;
                        pix += (size_t)w;
                    }
                }
            }
        }
    }

    if (run_start >= 0) {
        renderer_pick_mark_wall_column(ctx, x, run_start, bot, 0, 0, 0, 0);
    } else if (pick_capture_active && assume_contiguous_coverage && drawn_pixels > 0) {
        renderer_pick_mark_wall_column(ctx, x, top, bot, 0, 0, 0, 0);
    }

    return drawn_pixels;
}

static int renderer_draw_floor_columns_ctx_fast(RenderSliceContext *ctx,
                                                const int16_t *left_edge,
                                                const int16_t *right_edge_tab,
                                                const int16_t *col_top_tab,
                                                const int16_t *col_bot_tab,
                                                int col_x_min,
                                                int col_x_max,
                                                const int16_t *left_bright_tab,
                                                const int16_t *right_bright_tab,
                                                int poly_top, int poly_bot,
                                                int32_t floor_height,
                                                const uint8_t *texture,
                                                const uint8_t *floor_pal,
                                                int16_t brightness,
                                                int16_t use_gour_floor,
                                                int16_t scaleval)
{
#if !AB3D_CW_COL_MAJOR
    (void)ctx; (void)left_edge; (void)right_edge_tab; (void)col_top_tab; (void)col_bot_tab;
    (void)col_x_min; (void)col_x_max; (void)poly_top; (void)poly_bot;
    (void)left_bright_tab; (void)right_bright_tab;
    (void)floor_height; (void)texture; (void)floor_pal; (void)brightness;
    (void)use_gour_floor; (void)scaleval;
    return 0;
#else
    RendererState *rs = &g_renderer;
    FloorDrawCommon floor_common;
    uint8_t *buf = renderer_active_buf();
    uint32_t *rgb = renderer_active_rgb();
    uint16_t *cwbuf = renderer_active_cw();
    if (!ctx || !left_edge || !right_edge_tab || !texture || !floor_pal) return 0;
    if (!buf || !rgb || !cwbuf) return 0;

    const int w = rs->width;
    const int h = rs->height;
    if (w < 1 || h < 1) return 0;

    int y0 = poly_top;
    int y1 = poly_bot;
    if (y0 < ctx->top_clip) y0 = ctx->top_clip;
    if (y1 > ctx->bot_clip) y1 = ctx->bot_clip;
    if (y0 < 0) y0 = 0;
    if (y1 >= h) y1 = h - 1;
    if (y0 > y1) return 0;
    {
        const int row_count = y1 - y0 + 1;
        FloorRowFast *rows;
        if (!renderer_floor_fast_ensure_rows_capacity(row_count)) return 0;
        rows = g_floor_fast_rows_scratch;
        for (int row_idx = 0; row_idx < row_count; row_idx++) {
            rows[row_idx].le = 1;
            rows[row_idx].re = 0;
        }

        const int expand = g_renderer_rgb_raster_expand;
        const int center = h / 2;
        const int pick_capture_active = (g_pick_capture_active != 0);
        uint64_t drawn_px_count = 0;
        uint64_t drawn_col_count = 0;
        const uint16_t *gour_cw_levels[FLOOR_PAL_LEVEL_COUNT];
        const uint32_t *gour_rgb_levels[FLOOR_PAL_LEVEL_COUNT];

        renderer_floor_prepare_common(&floor_common, rs, floor_height, scaleval);
        floor_span_prepare_pal_cache_all_levels(ctx, floor_pal);
        for (int level = 0; level < FLOOR_PAL_LEVEL_COUNT; level++) {
            gour_cw_levels[level] = ctx->floor_pal_cw_levels[level];
            gour_rgb_levels[level] = expand ? ctx->floor_pal_rgb_levels[level] : NULL;
        }

        if (use_gour_floor) {
            for (int y = y0; y <= y1; y++) {
                FloorRowFast *row = &rows[y - y0];
                int16_t raw_le = left_edge[y];
                int16_t raw_re = right_edge_tab[y];
                int16_t le = raw_le;
                int16_t re = raw_re;
                FloorRowMath row_math;
                int32_t bl_edge;
                int32_t br_edge;
                int32_t full_span_w;
                int32_t clip_advance;
                int32_t gour_level_step;
                int32_t gour_level_fp;
                int left_level;
                int right_level;
                int dist_add;

                if (le >= w || re < 0) continue;
                if (le < ctx->left_clip) le = ctx->left_clip;
                if (re >= ctx->right_clip) re = (int16_t)(ctx->right_clip - 1);
                if (le > re) continue;

                renderer_floor_prepare_row_math(&row_math,
                                                rs,
                                                &floor_common,
                                                renderer_floor_row_dist_from_screen_y(y, center));

                bl_edge = left_bright_tab ? left_bright_tab[y] : brightness;
                br_edge = right_bright_tab ? right_bright_tab[y] : brightness;
                full_span_w = (int32_t)raw_re - (int32_t)raw_le;
                clip_advance = (int32_t)le - (int32_t)raw_le;
                dist_add = row_math.dist >> 7;

                left_level = bl_edge + dist_add;
                right_level = br_edge + dist_add;
                if (left_level < 0) left_level = 0;
                if (right_level < 0) right_level = 0;
                left_level >>= 1;
                right_level >>= 1;
                if (left_level >= FLOOR_PAL_LEVEL_COUNT) left_level = FLOOR_PAL_LEVEL_COUNT - 1;
                if (right_level >= FLOOR_PAL_LEVEL_COUNT) right_level = FLOOR_PAL_LEVEL_COUNT - 1;

                gour_level_step = (full_span_w > 0)
                    ? (int32_t)(((int64_t)(right_level - left_level) << 16) / (int64_t)full_span_w)
                    : 0;
                gour_level_fp = ((int32_t)left_level << 16)
                    + (int32_t)((int64_t)gour_level_step * (int64_t)clip_advance);

                row->le = le;
                row->re = re;
                row->u_step = row_math.u_step;
                row->v_step = row_math.v_step;
                row->u_base = row_math.u_base;
                row->v_base = row_math.v_base;
                row->bright_term = 0;
                row->span_cw = NULL;
                row->span_rgb = NULL;
                row->bright_base_fp = gour_level_fp;
                row->bright_step_fp = gour_level_step;
            }
        } else {
            for (int y = y0; y <= y1; y++) {
                FloorRowFast *row = &rows[y - y0];
                int16_t le = left_edge[y];
                int16_t re = right_edge_tab[y];
                FloorRowMath row_math;
                int bright_idx;
                int floor_pal_level;

                if (le >= w || re < 0) continue;
                if (le < ctx->left_clip) le = ctx->left_clip;
                if (re >= ctx->right_clip) re = (int16_t)(ctx->right_clip - 1);
                if (le > re) continue;

                renderer_floor_prepare_row_math(&row_math,
                                                rs,
                                                &floor_common,
                                                renderer_floor_row_dist_from_screen_y(y, center));

                bright_idx = brightness + row_math.bright_term;
                if (bright_idx < 0) bright_idx = 0;
                if (bright_idx > 28) bright_idx = 28;
                floor_pal_level = floor_bright_level_table[bright_idx];

                row->le = le;
                row->re = re;
                row->u_step = row_math.u_step;
                row->v_step = row_math.v_step;
                row->u_base = row_math.u_base;
                row->v_base = row_math.v_base;
                row->span_cw = gour_cw_levels[floor_pal_level];
                row->span_rgb = expand ? gour_rgb_levels[floor_pal_level] : NULL;
            }
        }

        if (col_top_tab && col_bot_tab && col_x_min <= col_x_max) {
            renderer_floor_fast_seed_rows(rows, row_count, use_gour_floor);
            for (int x = col_x_min; x <= col_x_max; x++) {
                int top = col_top_tab[x];
                int bot = col_bot_tab[x];
                if (top > bot) continue;
                if (top < y0) top = y0;
                if (bot > y1) bot = y1;
                if (top > bot) continue;
                {
                    uint64_t col_pixels = renderer_draw_floor_fast_column(ctx,
                                                                          rows,
                                                                          gour_cw_levels,
                                                                          gour_rgb_levels,
                                                                          texture,
                                                                          buf,
                                                                          rgb,
                                                                          cwbuf,
                                                                          w,
                                                                          h,
                                                                          y0,
                                                                          pick_capture_active,
                                                                          0,
                                                                          use_gour_floor,
                                                                          expand,
                                                                          x,
                                                                          top,
                                                                          bot);
                    if (col_pixels > 0) {
                        drawn_px_count += col_pixels;
                        drawn_col_count++;
                    }
                }
            }
        } else {
            int x_min = w;
            int x_max = -1;
            for (int y = y0; y <= y1; y++) {
                const FloorRowFast *row = &rows[y - y0];
                if (row->le > row->re) continue;
                if (row->le < x_min) x_min = row->le;
                if (row->re > x_max) x_max = row->re;
            }
            if (x_min <= x_max) {
                const int col_count = x_max - x_min + 1;
                if (renderer_floor_fast_ensure_cols_capacity(col_count)) {
                    int16_t *col_top_exact = g_floor_fast_col_top_scratch;
                    int16_t *col_bot_exact = g_floor_fast_col_bot_scratch;
                    int *next_top = g_floor_fast_next_top_scratch;
                    int *next_bot = g_floor_fast_next_bot_scratch;
                    int *col_cov_diff = g_floor_fast_cov_diff_scratch;
                    for (int i = 0; i < col_count; i++) {
                        col_top_exact[i] = renderer_clamp_edge_y_i16(h);
                        col_bot_exact[i] = -1;
                    }
                    for (int i = 0; i <= col_count; i++) {
                        next_top[i] = i;
                        next_bot[i] = i;
                        col_cov_diff[i] = 0;
                    }

                    for (int y = y0; y <= y1; y++) {
                        const FloorRowFast *row = &rows[y - y0];
                        int xl = row->le;
                        int xr = row->re;
                        if (xl > xr) continue;
                        if (xl < x_min) xl = x_min;
                        if (xr > x_max) xr = x_max;
                        if (xl > xr) continue;

                        {
                            const int l_idx = xl - x_min;
                            const int r_idx = xr - x_min;
                            col_cov_diff[l_idx] += 1;
                            col_cov_diff[r_idx + 1] -= 1;

                            int idx = renderer_floor_col_bounds_find_next(next_top, xl - x_min);
                            const int end_idx = xr - x_min;
                            while (idx <= end_idx) {
                                col_top_exact[idx] = (int16_t)y;
                                next_top[idx] = renderer_floor_col_bounds_find_next(next_top, idx + 1);
                                idx = next_top[idx];
                            }
                        }
                    }

                    for (int y = y1; y >= y0; y--) {
                        const FloorRowFast *row = &rows[y - y0];
                        int xl = row->le;
                        int xr = row->re;
                        if (xl > xr) continue;
                        if (xl < x_min) xl = x_min;
                        if (xr > x_max) xr = x_max;
                        if (xl > xr) continue;

                        {
                            int idx = renderer_floor_col_bounds_find_next(next_bot, xl - x_min);
                            const int end_idx = xr - x_min;
                            while (idx <= end_idx) {
                                col_bot_exact[idx] = (int16_t)y;
                                next_bot[idx] = renderer_floor_col_bounds_find_next(next_bot, idx + 1);
                                idx = next_bot[idx];
                            }
                        }
                    }

                    renderer_floor_fast_seed_rows(rows, row_count, use_gour_floor);
                    {
                        int cov_count = 0;
                        for (int x = x_min; x <= x_max; x++) {
                            int idx = x - x_min;
                            int top;
                            int bot;
                            int assume_contiguous;

                            cov_count += col_cov_diff[idx];
                            top = (int)col_top_exact[idx];
                            bot = (int)col_bot_exact[idx];
                            if (top > bot) continue;
                            if (top < y0) top = y0;
                            if (bot > y1) bot = y1;
                            if (top > bot) continue;

                            assume_contiguous = (cov_count == (bot - top + 1)) ? 1 : 0;
                            {
                                uint64_t col_pixels = renderer_draw_floor_fast_column(ctx,
                                                                                      rows,
                                                                                      gour_cw_levels,
                                                                                      gour_rgb_levels,
                                                                                      texture,
                                                                                      buf,
                                                                                      rgb,
                                                                                      cwbuf,
                                                                                      w,
                                                                                      h,
                                                                                      y0,
                                                                                      pick_capture_active,
                                                                                      assume_contiguous,
                                                                                      use_gour_floor,
                                                                                      expand,
                                                                                      x,
                                                                                      top,
                                                                                      bot);
                                if (col_pixels > 0) {
                                    drawn_px_count += col_pixels;
                                    drawn_col_count++;
                                }
                            }
                        }
                    }
                } else {
                    renderer_floor_fast_seed_rows(rows, row_count, use_gour_floor);
                    for (int x = x_min; x <= x_max; x++) {
                        int top = y1 + 1;
                        int bot = y0 - 1;
                        for (int y = y0; y <= y1; y++) {
                            const FloorRowFast *row = &rows[y - y0];
                            if (row->le > row->re) continue;
                            if (x < row->le || x > row->re) continue;
                            if (y < top) top = y;
                            bot = y;
                        }
                        if (top > bot) continue;
                        {
                            uint64_t col_pixels = renderer_draw_floor_fast_column(ctx,
                                                                                  rows,
                                                                                  gour_cw_levels,
                                                                                  gour_rgb_levels,
                                                                                  texture,
                                                                                  buf,
                                                                                  rgb,
                                                                                  cwbuf,
                                                                                  w,
                                                                                  h,
                                                                                  y0,
                                                                                  pick_capture_active,
                                                                                  0,
                                                                                  use_gour_floor,
                                                                                  expand,
                                                                                  x,
                                                                                  top,
                                                                                  bot);
                            if (col_pixels > 0) {
                                drawn_px_count += col_pixels;
                                drawn_col_count++;
                            }
                        }
                    }
                }
            }
        }

        if (ctx->profile_collect_stats) {
            ctx->workload_stats.floor_spans += drawn_col_count;
            ctx->workload_stats.floor_pixels += drawn_px_count;
            ctx->workload_stats.floor_fast_spans += drawn_col_count;
            ctx->workload_stats.floor_fast_pixels += drawn_px_count;
        }
        return 1;
    }
#endif
}

/* -----------------------------------------------------------------------
 * Sprite rendering
 *
 * Translated from ObjDraw3.ChipRam.s BitMapObj.
 *
 * Amiga sprite WAD format: packed pixel data where each 16-bit word
 * encodes 3 five-bit pixels (bits 0-4, 5-9, 10-14).
 *
 * The .ptr file is a column pointer table: each column is 4 bytes:
 *   byte 0 = "third" (0, 1, or 2) = which 5-bit slice of each word
 *   bytes 1-3 = 24-bit offset into .wad data for this column
 *
 * The .pal file contains a brightness-graded palette:
 *   15 levels x 32 colors x 2 bytes (big-endian 12-bit Amiga color words).
 *   Level N starts at offset N * 64; color C at offset N * 64 + C * 2.
 *
 * Per-pixel decode (same as gun renderer):
 *   third 0: texel = low_byte & 0x1F
 *   third 1: texel = (word >> 5) & 0x1F
 *   third 2: texel = (high_byte >> 2) & 0x1F
 *
 * Uses painter's algorithm (drawn back-to-front by zone order).
 * ----------------------------------------------------------------------- */
static void renderer_draw_sprite_ctx(RenderSliceContext *ctx,
                                     int16_t screen_x, int16_t screen_y,
                                     int16_t width, int16_t height, int16_t z,
                                     const uint8_t *wad, size_t wad_size,
                                     const uint8_t *ptr_data, size_t ptr_size,
                                     const uint8_t *pal, size_t pal_size,
                                     uint32_t ptr_offset, uint16_t down_strip,
                                     int src_cols, int src_rows,
                                     int16_t brightness, int sprite_type,
                                     int16_t clip_left_sx, int16_t clip_right_sx,
                                     int32_t clip_top_world, int32_t clip_bot_world,
                                     int32_t clip_top_sy, int32_t clip_bot_sy,
                                     int is_spill,
                                     const LevelState *geom_clip_level,
                                     int16_t geom_clip_zone,
                                     int32_t billboard_world_x,
                                     int32_t billboard_world_z,
                                     int32_t billboard_world_w,
                                     int16_t billboard_view_right_x,
                                     int16_t billboard_view_right_z)
{
    (void)clip_top_world;
    (void)clip_bot_world;
    (void)sprite_type;
    uint32_t *rgb = renderer_active_rgb();
    uint16_t *cw = renderer_active_cw();
    if (!ctx) return;
    if (!rgb || !cw) return;
    if (z <= SPRITE_NEAR_CLIP_Z) return;
    if (!wad || !ptr_data) return;
    const int rw = g_renderer.width, rh = g_renderer.height;
    if (src_cols < 1) src_cols = 32;
    if (src_rows < 1) src_rows = 32;

    const int eff_cols = src_cols * 2;
    const int eff_rows = src_rows * 2;
    const uint32_t max_ptr_col = (uint32_t)((ptr_size > ptr_offset) ? (ptr_size - ptr_offset) / 4u : 0);
    if (max_ptr_col == 0) return;

    const int sx = screen_x - width / 2;
    const int sy = screen_y - height / 2;

    /* --- Horizontal span (thread slice + screen bounds) --- */
    int clip_left = (int)ctx->left_clip;
    int clip_right = (int)ctx->right_clip;
    if ((int)clip_left_sx > clip_left) clip_left = (int)clip_left_sx;
    if ((int)clip_right_sx < clip_right) clip_right = (int)clip_right_sx;
    if (clip_left < 0) clip_left = 0;
    if (clip_right > rw) clip_right = rw;

    int col_start = (sx > clip_left) ? sx : clip_left;
    if (col_start < 0) col_start = 0;
    const int dx_start = col_start - sx;
    int col_end = sx + width;
    if (col_end > clip_right) col_end = clip_right;
    if (col_end > rw) col_end = rw;
    int dx_end = col_end - sx;
    if (dx_end > width) dx_end = width;
    if (dx_end <= dx_start) return;

    /* --- Vertical clipping (invariant across columns) --- */
    int draw_top = sy;
    int draw_bot = sy + height - 1;
    const int have_plane_clip = (clip_top_sy < clip_bot_sy);
    if (draw_top < 0) draw_top = 0;
    if (draw_bot >= rh) draw_bot = rh - 1;
    if (!have_plane_clip) {
        if (draw_top < (int)ctx->top_clip) draw_top = (int)ctx->top_clip;
        if (draw_bot > (int)ctx->bot_clip) draw_bot = (int)ctx->bot_clip;
    }
    if (have_plane_clip) {
        /* Split-zone object passes already have an exact per-sprite room-plane clip.
         * Reapplying the coarse pass band here can wrongly cull distant billboards
         * when the fixed split line does not match the sprite's true projection. */
        if (draw_top < (int)clip_top_sy) draw_top = (int)clip_top_sy;
        if (draw_bot > (int)clip_bot_sy) draw_bot = (int)clip_bot_sy;
    }
    if (draw_top > draw_bot) return;
    const int draw_row_count = draw_bot - draw_top + 1;
    if (draw_row_count <= 0 || draw_row_count > RENDER_INTERNAL_MAX_DIM) return;
    if (g_debug_spill_visualize) {
        int dbg_left = sx + dx_start;
        int dbg_right = sx + dx_end - 1;
        renderer_debug_spill_note_sprite_rect(dbg_left,
                                              draw_top,
                                              dbg_right,
                                              draw_bot,
                                              screen_x,
                                              is_spill);
    }
    const int profile_collect_stats = (ctx && ctx->profile_collect_stats);
    uint64_t sprite_visible_rows_total = 0;
    uint64_t sprite_wall_occluded_rows_total = 0;
    uint64_t sprite_spill_occluded_rows_total = 0;
    uint64_t sprite_opaque_writes_total = 0;
    if (profile_collect_stats) {
        ctx->workload_stats.sprite_calls++;
        ctx->workload_stats.sprite_columns += (uint64_t)(dx_end - dx_start);
    }

    /* --- Pre-build 32-entry palette LUT for this brightness level ---
     * Replaces per-pixel palette byte reads, c12 construction, and
     * amiga12_to_argb conversion with a single indexed load. */
    int bright_idx = brightness;
    if (bright_idx < 0) bright_idx = 0;
    if (bright_idx > 61) bright_idx = 61;
    uint32_t pal_level_off = obj_scale_cols[bright_idx];
    if (pal && pal_size < 960) pal_level_off = 0;

    uint16_t spr_cw[32];
    uint32_t spr_rgb[32];
    if (pal && pal_size >= 64) {
        uint32_t level_off = (pal_level_off + 64 <= pal_size) ? pal_level_off : 0;
        for (int ti = 1; ti < 32; ti++) {
            uint32_t ci = level_off + (uint32_t)ti * 2;
            if (ci + 1 < pal_size) {
                uint16_t c12 = (uint16_t)((pal[ci] << 8) | pal[ci + 1]);
                spr_cw[ti] = c12;
                spr_rgb[ti] = amiga12_to_argb(c12);
            } else {
                spr_cw[ti] = 0;
                spr_rgb[ti] = 0;
            }
        }
    } else {
        int gray = (bright_idx * 255) / 62;
        if (gray < 90) gray = 90;
        for (int ti = 1; ti < 32; ti++) {
            int shade = (gray * ti) / 31;
            uint32_t c = RENDER_RGB_RASTER_PIXEL(((uint32_t)shade << 16) | ((uint32_t)shade << 8) | (uint32_t)shade);
            spr_rgb[ti] = c;
            spr_cw[ti] = argb_to_amiga12(c);
        }
    }

    /* --- Hoisted constants for the column/row loops --- */
    const int expand = g_renderer_rgb_raster_expand;
    const size_t rw_stride = (size_t)rw;
    const size_t cw_step_y = renderer_cw_step_y(rw);
    const int down_strip_i = (int)down_strip;
    const int spill_visualize = (is_spill && g_debug_spill_visualize) ? 1 : 0;
    const uint16_t spill_vis_cw = 0x0F0Fu;
    const uint32_t spill_vis_rgb = amiga12_to_argb(spill_vis_cw);
    uint16_t *pick_zone = renderer_active_pick_zone();
    uint8_t *pick_player = renderer_active_pick_player();
    const uint8_t pick_player_id = ctx->pick_player_id;
    const int have_pick = (pick_player && pick_player_id) ? 1 : 0;
    const int mark_sprite_pick_zone_none = (g_debug_spill_visualize && pick_zone) ? 1 : 0;
    const int fast_common_no_spill =
        (!have_pick) ? 1 : 0;
    const int geometry_clip_enabled = 0;
    const int have_geometric_zone_clip =
        (geometry_clip_enabled && geom_clip_level && geom_clip_zone >= 0 && billboard_world_w > 0 && width > 0) ? 1 : 0;
    int geom_dx_starts[RENDERER_GEOM_CLIP_MAX_SPANS];
    int geom_dx_ends[RENDERER_GEOM_CLIP_MAX_SPANS];
    int geom_dx_count = 1;
    int geom_dx_idx = 0;

    geom_dx_starts[0] = dx_start;
    geom_dx_ends[0] = dx_end;
    if (have_geometric_zone_clip) {
        int32_t half_span_world = billboard_world_w / 2;
        int64_t off_x64;
        int64_t off_z64;
        int32_t seg_x0;
        int32_t seg_z0;
        int32_t seg_x1;
        int32_t seg_z1;

        if (half_span_world < 1) half_span_world = 1;
        off_x64 = ((int64_t)billboard_view_right_x * (int64_t)half_span_world) / 16384;
        off_z64 = ((int64_t)billboard_view_right_z * (int64_t)half_span_world) / 16384;
        if (off_x64 == 0 && off_z64 == 0) {
            if (billboard_view_right_x != 0) {
                off_x64 = (billboard_view_right_x > 0) ? 1 : -1;
            } else if (billboard_view_right_z != 0) {
                off_z64 = (billboard_view_right_z > 0) ? 1 : -1;
            } else {
                off_x64 = 1;
            }
        }

        seg_x0 = (int32_t)((int64_t)billboard_world_x - off_x64);
        seg_z0 = (int32_t)((int64_t)billboard_world_z - off_z64);
        seg_x1 = (int32_t)((int64_t)billboard_world_x + off_x64);
        seg_z1 = (int32_t)((int64_t)billboard_world_z + off_z64);

        geom_dx_count = renderer_zone_clip_segment_to_dx_ranges(geom_clip_level,
                                                                geom_clip_zone,
                                                                seg_x0, seg_z0,
                                                                seg_x1, seg_z1,
                                                                width,
                                                                0,
                                                                width,
                                                                geom_dx_starts,
                                                                geom_dx_ends,
                                                                RENDERER_GEOM_CLIP_MAX_SPANS);
        if (geom_dx_count <= 0) {
            /* Fail-open: when geometric clip finds no spans, keep the normal
             * screen-span draw so billboards do not disappear entirely. */
            geom_dx_count = 1;
            geom_dx_starts[0] = dx_start;
            geom_dx_ends[0] = dx_end;
        }

        /* Post-clip margin expansion: apply after geometric clipping so
         * ownership is geometry-derived while keeping texcoords unchanged
         * (src_col mapping still uses original dx range). */
        if (geom_dx_count > 0) {
            const int spill_margin_percent = 15;
            int merged_count = 0;

            for (int gi = 0; gi < geom_dx_count; gi++) {
                int s = geom_dx_starts[gi];
                int e = geom_dx_ends[gi];
                int run = e - s;
                int extra;

                if (run <= 0) continue;

                extra = (run * spill_margin_percent + 99) / 100;
                if (extra < 1) extra = 1;

                s -= extra;
                e += extra;
                if (s < dx_start) s = dx_start;
                if (e > dx_end) e = dx_end;
                if (e <= s) continue;

                if (merged_count > 0 && s <= geom_dx_ends[merged_count - 1]) {
                    if (e > geom_dx_ends[merged_count - 1]) {
                        geom_dx_ends[merged_count - 1] = e;
                    }
                } else if (merged_count < RENDERER_GEOM_CLIP_MAX_SPANS) {
                    geom_dx_starts[merged_count] = s;
                    geom_dx_ends[merged_count] = e;
                    merged_count++;
                }
            }

            if (merged_count > 0) {
                geom_dx_count = merged_count;
            }
        }

        /* Intersect geometric ranges with this worker slice after all geometric
         * adjustments so threaded and single-thread paths produce identical
         * ownership at slice boundaries. */
        {
            int sliced_count = 0;
            for (int gi = 0; gi < geom_dx_count; gi++) {
                int s = geom_dx_starts[gi];
                int e = geom_dx_ends[gi];
                if (s < dx_start) s = dx_start;
                if (e > dx_end) e = dx_end;
                if (e <= s) continue;

                if (sliced_count > 0 && s <= geom_dx_ends[sliced_count - 1]) {
                    if (e > geom_dx_ends[sliced_count - 1]) {
                        geom_dx_ends[sliced_count - 1] = e;
                    }
                } else {
                    geom_dx_starts[sliced_count] = s;
                    geom_dx_ends[sliced_count] = e;
                    sliced_count++;
                }
            }
            geom_dx_count = sliced_count;
            if (geom_dx_count <= 0) {
                /* Fail-open after slice intersection as well: if geometric
                 * ownership ranges collapse, keep drawing the normal screen
                 * span instead of culling the sprite. */
                geom_dx_count = 1;
                geom_dx_starts[0] = dx_start;
                geom_dx_ends[0] = dx_end;
            }
        }
    }

    /* Fixed-point 16.16 DDA stepping (replaces per-pixel integer division). */
    const int32_t src_col_step = (width > 1)
        ? (int32_t)(((uint32_t)eff_cols << 16) / (uint32_t)width) : 0;
    const int32_t src_row_step = (height > 1)
        ? (int32_t)(((uint32_t)eff_rows << 16) / (uint32_t)height) : 0;
    const int prefer_col_texel_lut =
        (height >= (eff_rows << 1) && eff_rows <= (int)sizeof(g_sprite_col_texel_lut)) ? 1 : 0;

    int *row_src_lut = g_sprite_src_row_lut;
    {
        int32_t row_fp_lut = (int32_t)(draw_top - sy) * src_row_step;
        for (int row_i = 0; row_i < draw_row_count; row_i++) {
            int src_row = row_fp_lut >> 16;
            row_fp_lut += src_row_step;
            if (src_row >= eff_rows) src_row = eff_rows - 1;
            row_src_lut[row_i] = src_row;
        }
    }

    int32_t src_col_fp = (int32_t)dx_start * src_col_step;

    /* --- Column loop --- */
    for (int dx = dx_start; dx < dx_end; dx++) {
        const int screen_col = sx + dx;

        /* Advance source-column DDA regardless of geometric clip so texture
         * mapping remains stable across rejected columns. */
        int src_col = src_col_fp >> 16;
        src_col_fp += src_col_step;
        if (src_col >= eff_cols) src_col = eff_cols - 1;
        if ((uint32_t)src_col >= max_ptr_col) src_col = (int)(max_ptr_col - 1);

        if (have_geometric_zone_clip) {
            while (geom_dx_idx < geom_dx_count && dx >= geom_dx_ends[geom_dx_idx]) {
                geom_dx_idx++;
            }
            if (geom_dx_idx >= geom_dx_count || dx < geom_dx_starts[geom_dx_idx]) {
                if (profile_collect_stats) {
                    int total_rows = draw_bot - draw_top + 1;
                    if (total_rows > 0)
                        sprite_wall_occluded_rows_total += (uint64_t)total_rows;
                }
                continue;
            }
        }

        uint32_t entry_off = ptr_offset + (uint32_t)src_col * 4;
        if (entry_off + 4 > ptr_size) continue;
        const uint8_t *entry = ptr_data + entry_off;
        const uint8_t mode = entry[0];
        const uint32_t wad_off = ((uint32_t)entry[1] << 16)
                               | ((uint32_t)entry[2] << 8)
                               | (uint32_t)entry[3];
        if (mode == 0 && wad_off == 0) continue;
        if (wad_off >= wad_size) continue;

        const uint8_t *src = wad + wad_off;

        /* Precompute texel shift from PTR mode (replaces per-pixel branch). */
        const int texel_shift = (mode == 0) ? 0 : (mode == 1) ? 5 : 10;

        /* Max valid row_idx for this column (replaces per-pixel bounds check).
         * Each row is 2 bytes; need (row_idx + 1) * 2 <= remaining WAD. */
        const int max_row_idx = (int)((wad_size - wad_off) / 2) - 1;
        if (max_row_idx < 0) continue;
        if (down_strip_i > max_row_idx) continue;

        const uint8_t *src_base = src + (size_t)down_strip_i * 2u;
        const int max_src_row_col = max_row_idx - down_strip_i;
        const int need_src_row_bounds = (max_src_row_col < (eff_rows - 1));
        uint8_t *col_texel_lut = g_sprite_col_texel_lut;
        int use_col_texel_lut = 0;
        if (prefer_col_texel_lut) {
            int decode_rows = max_src_row_col + 1;
            if (decode_rows > eff_rows) decode_rows = eff_rows;
            if (decode_rows > 0) {
                for (int rr = 0; rr < decode_rows; rr++) {
                    const int src_byte = rr << 1;
                    const uint16_t w = (uint16_t)((src_base[src_byte] << 8) | src_base[src_byte + 1]);
                    col_texel_lut[rr] = (uint8_t)((w >> texel_shift) & 0x1F);
                }
                use_col_texel_lut = 1;
            }
        }

        /* Billboards are rendered with geometric zone clipping only (no per-column depth occluders). */
        int seg_top[1], seg_bot[1], seg_count = 1;
        seg_top[0] = draw_top;
        seg_bot[0] = draw_bot;
        if (profile_collect_stats) {
            int total_rows = draw_bot - draw_top + 1;
            int visible_rows = 0;
            if (total_rows < 0) total_rows = 0;
            for (int si = 0; si < seg_count; si++) {
                int run = seg_bot[si] - seg_top[si] + 1;
                if (run > 0) visible_rows += run;
            }
            if (visible_rows < 0) visible_rows = 0;
            if (visible_rows > total_rows) visible_rows = total_rows;
            sprite_visible_rows_total += (uint64_t)visible_rows;
            sprite_wall_occluded_rows_total += (uint64_t)(total_rows - visible_rows);
        }
        if (seg_count <= 0) continue;

        /* Rasterize all visible intervals for this column. */
        if (expand) {
            for (int si = 0; si < seg_count; si++) {
                int col_top = seg_top[si];
                int col_bot = seg_bot[si];
                if (col_top > col_bot) continue;
                int row_lut_idx = col_top - draw_top;
                size_t pix = (size_t)col_top * rw_stride + (size_t)screen_col;
                size_t pix_cw = renderer_cw_index_xy(screen_col, col_top, rw, rh);
                if (fast_common_no_spill) {
                    const int *row_src_ptr = row_src_lut + row_lut_idx;
                    if (!need_src_row_bounds) {
                        if (use_col_texel_lut) {
                            for (int screen_row = col_top; screen_row <= col_bot; screen_row++) {
                                const int src_row = *row_src_ptr++;
                                const uint8_t texel = col_texel_lut[src_row];
                                if (texel != 0) {
                                    if (profile_collect_stats) sprite_opaque_writes_total++;
                                    rgb[pix] = spr_rgb[texel];
                                    cw[pix_cw] = spr_cw[texel];
                                    if (mark_sprite_pick_zone_none) pick_zone[pix] = RENDERER_PICK_ZONE_NONE;
                                }
                                pix += rw_stride;
                                pix_cw += cw_step_y;
                            }
                        } else {
                            for (int screen_row = col_top; screen_row <= col_bot; screen_row++) {
                                const int src_row = *row_src_ptr++;
                                const int src_byte = src_row << 1;
                                const uint16_t w = (uint16_t)((src_base[src_byte] << 8) | src_base[src_byte + 1]);
                                const uint8_t texel = (uint8_t)((w >> texel_shift) & 0x1F);
                                if (texel != 0) {
                                    if (profile_collect_stats) sprite_opaque_writes_total++;
                                    rgb[pix] = spr_rgb[texel];
                                    cw[pix_cw] = spr_cw[texel];
                                    if (mark_sprite_pick_zone_none) pick_zone[pix] = RENDERER_PICK_ZONE_NONE;
                                }
                                pix += rw_stride;
                                pix_cw += cw_step_y;
                            }
                        }
                    } else {
                        if (use_col_texel_lut) {
                            for (int screen_row = col_top; screen_row <= col_bot; screen_row++) {
                                const int src_row = *row_src_ptr++;
                                if (src_row <= max_src_row_col) {
                                    const uint8_t texel = col_texel_lut[src_row];
                                    if (texel != 0) {
                                        if (profile_collect_stats) sprite_opaque_writes_total++;
                                        rgb[pix] = spr_rgb[texel];
                                        cw[pix_cw] = spr_cw[texel];
                                        if (mark_sprite_pick_zone_none) pick_zone[pix] = RENDERER_PICK_ZONE_NONE;
                                    }
                                }
                                pix += rw_stride;
                                pix_cw += cw_step_y;
                            }
                        } else {
                            for (int screen_row = col_top; screen_row <= col_bot; screen_row++) {
                                const int src_row = *row_src_ptr++;
                                if (src_row <= max_src_row_col) {
                                    const int src_byte = src_row << 1;
                                    const uint16_t w = (uint16_t)((src_base[src_byte] << 8) | src_base[src_byte + 1]);
                                    const uint8_t texel = (uint8_t)((w >> texel_shift) & 0x1F);
                                    if (texel != 0) {
                                        if (profile_collect_stats) sprite_opaque_writes_total++;
                                        rgb[pix] = spr_rgb[texel];
                                        cw[pix_cw] = spr_cw[texel];
                                        if (mark_sprite_pick_zone_none) pick_zone[pix] = RENDERER_PICK_ZONE_NONE;
                                    }
                                }
                                pix += rw_stride;
                                pix_cw += cw_step_y;
                            }
                        }
                    }
                } else {
                    for (int screen_row = col_top; screen_row <= col_bot; screen_row++) {
                        const int src_row = row_src_lut[row_lut_idx];
                        if (have_pick) pick_player[pix] = pick_player_id;
                        if (!need_src_row_bounds || src_row <= max_src_row_col) {
                            const int src_byte = src_row << 1;
                            const uint16_t w = (uint16_t)((src_base[src_byte] << 8) | src_base[src_byte + 1]);
                            const uint8_t texel = (uint8_t)((w >> texel_shift) & 0x1F);
                            if (texel != 0) {
                                if (profile_collect_stats) sprite_opaque_writes_total++;
                                if (spill_visualize) {
                                    rgb[pix] = spill_vis_rgb;
                                    cw[pix_cw] = spill_vis_cw;
                                } else {
                                    rgb[pix] = spr_rgb[texel];
                                    cw[pix_cw] = spr_cw[texel];
                                }
                                if (mark_sprite_pick_zone_none) pick_zone[pix] = RENDERER_PICK_ZONE_NONE;
                            }
                        }
                        pix += rw_stride;
                        pix_cw += cw_step_y;
                        row_lut_idx++;
                    }
                }
            }
        } else {
            for (int si = 0; si < seg_count; si++) {
                int col_top = seg_top[si];
                int col_bot = seg_bot[si];
                if (col_top > col_bot) continue;
                int row_lut_idx = col_top - draw_top;
                size_t pix = (size_t)col_top * rw_stride + (size_t)screen_col;
                size_t pix_cw = renderer_cw_index_xy(screen_col, col_top, rw, rh);
                if (fast_common_no_spill) {
                    const int *row_src_ptr = row_src_lut + row_lut_idx;
                    if (!need_src_row_bounds) {
                        if (use_col_texel_lut) {
                            for (int screen_row = col_top; screen_row <= col_bot; screen_row++) {
                                const int src_row = *row_src_ptr++;
                                const uint8_t texel = col_texel_lut[src_row];
                                if (texel != 0) {
                                    if (profile_collect_stats) sprite_opaque_writes_total++;
                                    cw[pix_cw] = spr_cw[texel];
                                    if (mark_sprite_pick_zone_none) pick_zone[pix] = RENDERER_PICK_ZONE_NONE;
                                }
                                pix += rw_stride;
                                pix_cw += cw_step_y;
                            }
                        } else {
                            for (int screen_row = col_top; screen_row <= col_bot; screen_row++) {
                                const int src_row = *row_src_ptr++;
                                const int src_byte = src_row << 1;
                                const uint16_t w = (uint16_t)((src_base[src_byte] << 8) | src_base[src_byte + 1]);
                                const uint8_t texel = (uint8_t)((w >> texel_shift) & 0x1F);
                                if (texel != 0) {
                                    if (profile_collect_stats) sprite_opaque_writes_total++;
                                    cw[pix_cw] = spr_cw[texel];
                                    if (mark_sprite_pick_zone_none) pick_zone[pix] = RENDERER_PICK_ZONE_NONE;
                                }
                                pix += rw_stride;
                                pix_cw += cw_step_y;
                            }
                        }
                    } else {
                        if (use_col_texel_lut) {
                            for (int screen_row = col_top; screen_row <= col_bot; screen_row++) {
                                const int src_row = *row_src_ptr++;
                                if (src_row <= max_src_row_col) {
                                    const uint8_t texel = col_texel_lut[src_row];
                                    if (texel != 0) {
                                        if (profile_collect_stats) sprite_opaque_writes_total++;
                                        cw[pix_cw] = spr_cw[texel];
                                        if (mark_sprite_pick_zone_none) pick_zone[pix] = RENDERER_PICK_ZONE_NONE;
                                    }
                                }
                                pix += rw_stride;
                                pix_cw += cw_step_y;
                            }
                        } else {
                            for (int screen_row = col_top; screen_row <= col_bot; screen_row++) {
                                const int src_row = *row_src_ptr++;
                                if (src_row <= max_src_row_col) {
                                    const int src_byte = src_row << 1;
                                    const uint16_t w = (uint16_t)((src_base[src_byte] << 8) | src_base[src_byte + 1]);
                                    const uint8_t texel = (uint8_t)((w >> texel_shift) & 0x1F);
                                    if (texel != 0) {
                                        if (profile_collect_stats) sprite_opaque_writes_total++;
                                        cw[pix_cw] = spr_cw[texel];
                                        if (mark_sprite_pick_zone_none) pick_zone[pix] = RENDERER_PICK_ZONE_NONE;
                                    }
                                }
                                pix += rw_stride;
                                pix_cw += cw_step_y;
                            }
                        }
                    }
                } else {
                    for (int screen_row = col_top; screen_row <= col_bot; screen_row++) {
                        const int src_row = row_src_lut[row_lut_idx];
                        if (have_pick) pick_player[pix] = pick_player_id;
                        if (!need_src_row_bounds || src_row <= max_src_row_col) {
                            const int src_byte = src_row << 1;
                            const uint16_t w = (uint16_t)((src_base[src_byte] << 8) | src_base[src_byte + 1]);
                            const uint8_t texel = (uint8_t)((w >> texel_shift) & 0x1F);
                            if (texel != 0) {
                                if (profile_collect_stats) sprite_opaque_writes_total++;
                                cw[pix_cw] = spill_visualize ? spill_vis_cw : spr_cw[texel];
                                if (mark_sprite_pick_zone_none) pick_zone[pix] = RENDERER_PICK_ZONE_NONE;
                            }
                        }
                        pix += rw_stride;
                        pix_cw += cw_step_y;
                        row_lut_idx++;
                    }
                }
            }
        }
    }

    if (profile_collect_stats) {
        uint64_t tested_pixels = sprite_visible_rows_total;
        ctx->workload_stats.sprite_pixels_visible += sprite_visible_rows_total;
        ctx->workload_stats.sprite_pixels_wall_occluded += sprite_wall_occluded_rows_total;
        ctx->workload_stats.sprite_pixels_spill_occluded += sprite_spill_occluded_rows_total;
        ctx->workload_stats.sprite_pixels_tested += tested_pixels;
        ctx->workload_stats.sprite_pixels_drawn += sprite_opaque_writes_total;
    }
}

void renderer_draw_sprite(int16_t screen_x, int16_t screen_y,
                          int16_t width, int16_t height, int16_t z,
                          const uint8_t *wad, size_t wad_size,
                          const uint8_t *ptr_data, size_t ptr_size,
                          const uint8_t *pal, size_t pal_size,
                          uint32_t ptr_offset, uint16_t down_strip,
                          int src_cols, int src_rows,
                          int16_t brightness, int sprite_type,
                          int32_t clip_top_sy, int32_t clip_bot_sy)
{
    RenderSliceContext ctx;
    render_slice_context_init(&ctx, g_renderer.left_clip, g_renderer.right_clip,
                              g_renderer.top_clip, g_renderer.bot_clip);
    ctx.pick_player_id = 0;
    renderer_draw_sprite_ctx(&ctx, screen_x, screen_y, width, height, z,
                             wad, wad_size, ptr_data, ptr_size, pal, pal_size,
                             ptr_offset, down_strip, src_cols, src_rows, brightness,
                             sprite_type, ctx.left_clip, ctx.right_clip,
                             INT32_MIN, INT32_MIN,
                             clip_top_sy, clip_bot_sy, 0,
                             NULL, -1,
                             0, 0, 0,
                             0, 0);
}

static void renderer_draw_sprite_ctx_timed(RenderSliceContext *ctx,
                                           int16_t screen_x, int16_t screen_y,
                                           int16_t width, int16_t height, int16_t z,
                                           const uint8_t *wad, size_t wad_size,
                                           const uint8_t *ptr_data, size_t ptr_size,
                                           const uint8_t *pal, size_t pal_size,
                                           uint32_t ptr_offset, uint16_t down_strip,
                                           int src_cols, int src_rows,
                                           int16_t brightness, int sprite_type,
                                           int16_t clip_left_sx, int16_t clip_right_sx,
                                           int32_t clip_top_world, int32_t clip_bot_world,
                                           int32_t clip_top_sy, int32_t clip_bot_sy,
                                           int is_spill,
                                           const LevelState *geom_clip_level,
                                           int16_t geom_clip_zone,
                                           int32_t billboard_world_x,
                                           int32_t billboard_world_z,
                                           int32_t billboard_world_w,
                                           int16_t billboard_view_right_x,
                                           int16_t billboard_view_right_z)
{
    if (!ctx) return;

    if (ctx->profile_collect_stats) {
        uint64_t sprite_t0 = SDL_GetPerformanceCounter();
        renderer_draw_sprite_ctx(ctx,
                                 screen_x, screen_y,
                                 width, height, z,
                                 wad, wad_size,
                                 ptr_data, ptr_size,
                                 pal, pal_size,
                                 ptr_offset, down_strip,
                                 src_cols, src_rows,
                                 brightness, sprite_type,
                                 clip_left_sx, clip_right_sx,
                                 clip_top_world, clip_bot_world,
                                 clip_top_sy, clip_bot_sy,
                                 is_spill,
                                 geom_clip_level,
                                 geom_clip_zone,
                                 billboard_world_x,
                                 billboard_world_z,
                                 billboard_world_w,
                                 billboard_view_right_x,
                                 billboard_view_right_z);
        ctx->workload_stats.ticks_sprite += SDL_GetPerformanceCounter() - sprite_t0;
    } else {
        renderer_draw_sprite_ctx(ctx,
                                 screen_x, screen_y,
                                 width, height, z,
                                 wad, wad_size,
                                 ptr_data, ptr_size,
                                 pal, pal_size,
                                 ptr_offset, down_strip,
                                 src_cols, src_rows,
                                 brightness, sprite_type,
                                 clip_left_sx, clip_right_sx,
                                 clip_top_world, clip_bot_world,
                                 clip_top_sy, clip_bot_sy,
                                 is_spill,
                                 geom_clip_level,
                                 geom_clip_zone,
                                 billboard_world_x,
                                 billboard_world_z,
                                 billboard_world_w,
                                 billboard_view_right_x,
                                 billboard_view_right_z);
    }
}

/* -----------------------------------------------------------------------
 * Draw gun overlay
 *
 * Translated from AB3DI.s DrawInGun (lines 2426-2535).
 * Amiga: gun graphic from Objects+9, GUNYOFFS=20, 3 chunks x 32 = 96 wide,
 * 78-GUNYOFFS = 58 lines tall. If gun graphics are not loaded, nothing is drawn.
 * ----------------------------------------------------------------------- */
static void renderer_draw_gun_columns(GameState *state, int col_start, int col_end)
{
    uint8_t *buf = renderer_active_buf();
    uint32_t *rgb = renderer_active_rgb();
    uint16_t *cw = renderer_active_cw();
    if (!buf || !rgb || !cw) return;
    if (!state) return;

    int rw = g_renderer.width, rh = g_renderer.height;
    if (col_start < 0) col_start = 0;
    if (col_end > rw) col_end = rw;
    if (col_start >= col_end) return;

    if (!state->cfg_weapon_draw) return;

    PlayerState *plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
    if (plr->gun_selected < 0) return;

    int gun_type = plr->gun_selected;
    if (gun_type < 0 || gun_type >= 8) gun_type = 0;

    /* Amiga source gun frame: 96x58. Keep one uniform scale so overlay aspect is stable. */
    const int gun_w_src = GUN_COLS;
    const int gun_h_src = GUN_LINES;

    /* Use fractional scale (not integer steps) so size changes smoothly with resize.
     * Base follows height like the original intent, then fit uniformly to screen bounds. */
    int64_t scale_fp = (((int64_t)RENDER_SCALE * (int64_t)rh) << 16) / RENDER_DEFAULT_HEIGHT;
    int64_t fit_fp_w = ((int64_t)rw << 16) / gun_w_src;
    int64_t fit_fp_h = ((int64_t)rh << 16) / gun_h_src;
    if (scale_fp > fit_fp_w) scale_fp = fit_fp_w;
    if (scale_fp > fit_fp_h) scale_fp = fit_fp_h;
    if (scale_fp < 1) scale_fp = 1;

    int gun_w_draw = (int)(((int64_t)gun_w_src * scale_fp + 0x7FFF) >> 16);
    int gun_h_draw = (int)(((int64_t)gun_h_src * scale_fp + 0x7FFF) >> 16);

    /* If display output is wider/taller than the internal render target (e.g.
     * width clamped at 2048), SDL stretches the final image. Pre-compensate
     * gun width so only the weapon overlay preserves aspect after that stretch. */
    int present_w = (g_renderer.present_width > 0) ? g_renderer.present_width : rw;
    int present_h = (g_renderer.present_height > 0) ? g_renderer.present_height : rh;
    int64_t post_aspect_fp = ((int64_t)present_h * (int64_t)rw << 16)
                           / ((int64_t)present_w * (int64_t)rh);
    if (post_aspect_fp > 0) {
        gun_w_draw = (int)(((int64_t)gun_w_draw * post_aspect_fp + 0x7FFF) >> 16);
    }

    if (gun_w_draw < 1) gun_w_draw = 1;
    if (gun_h_draw < 1) gun_h_draw = 1;
    if (gun_w_draw > rw) gun_w_draw = rw;
    if (gun_h_draw > rh) gun_h_draw = rh;

    /* Keep the weapon at the bottom of the view. */
    int gy = rh - gun_h_draw;
    if (gy < 0) gy = 0;
    int gx = (rw - gun_w_draw) / 2;
    if (gun_type == ROCKET_LAUNCHER_GUN_IDX) {
        /* Port quirk: rocket launcher overlay is anchored to the right edge. */
        gx = rw - gun_w_draw;
    }

    int gun_x1 = gx + gun_w_draw;
    int draw_x0 = col_start;
    if (draw_x0 < gx) draw_x0 = gx;
    int draw_x1 = col_end;
    if (draw_x1 > gun_x1) draw_x1 = gun_x1;
    if (draw_x0 >= draw_x1) return;

    /* Draw from loaded gun data (newgunsinhand.wad + .ptr + .pal) if present */
    const uint8_t *gun_wad = g_renderer.gun_wad;
    const uint8_t *gun_ptr = g_renderer.gun_ptr;
    const uint8_t *gun_pal = g_renderer.gun_pal;
    size_t gun_wad_size = g_renderer.gun_wad_size;

    if (gun_wad && gun_ptr && gun_pal && gun_wad_size > 0) {
        const GunAnim *anim = &gun_anims[gun_type];
        int anim_frame = plr->gun_frame;
        if (anim_frame > anim->num_frames) anim_frame = 0;
        int graphic_frame = anim->frames[anim_frame];
        if (graphic_frame > 3) graphic_frame = 0;
        uint32_t frame_slot = (uint32_t)(gun_type * 4 + graphic_frame);
        if (frame_slot >= 32) frame_slot = 0;
        uint32_t ptr_off = gun_ptr_frame_offsets[frame_slot];

        if (ptr_off != 0 || (gun_type != 5 && gun_type != 6)) {
            /* Draw scaled: map screen pixel to source by (draw size / source size) ratio */
            int draw_sy0 = gy;
            int draw_sy1 = gy + gun_h_draw;
            if (draw_sy0 < 0) draw_sy0 = 0;
            if (draw_sy1 > rh) draw_sy1 = rh;
            for (int sy = draw_sy0; sy < draw_sy1; sy++) {
                if (sy < 0) continue;
                int src_row = (int)((int64_t)(sy - gy) * (int64_t)gun_h_src / gun_h_draw);
                if (src_row >= gun_h_src) continue;
                for (int sx = draw_x0; sx < draw_x1 && sx < rw; sx++) {
                    if (sx < 0) continue;
                    int src_col = (int)((int64_t)(sx - gx) * (int64_t)gun_w_src / gun_w_draw);
                    if (src_col >= gun_w_src) continue;

                    const uint8_t *col_ptr = gun_ptr + ptr_off + (uint32_t)src_col * 4;
                    uint8_t mode = col_ptr[0];
                    uint32_t wad_off = ((uint32_t)col_ptr[1] << 16) | ((uint32_t)col_ptr[2] << 8) | (uint32_t)col_ptr[3];
                    /* Match sprite PTR semantics: skip only when whole entry is zero.
                     * mode!=0 with wad_off==0 is valid (column data at WAD start). */
                    if (mode == 0 && wad_off == 0) continue;
                    if (wad_off >= gun_wad_size) continue;

                    const uint8_t *src = gun_wad + wad_off;
                    if (wad_off + (size_t)(src_row + 1) * 2 > gun_wad_size) continue;
                    uint16_t w = (uint16_t)((src[src_row * 2u] << 8) | src[src_row * 2u + 1]);
                    uint32_t idx;
                    if (mode == 0) {
                        idx = (uint32_t)(w & 31u);
                    } else if (mode == 1) {
                        idx = (uint32_t)((w >> 5) & 31u);
                    } else {
                        idx = (uint32_t)((w >> 10) & 31u);
                    }
                    if (idx == 0) continue;

                    uint16_t c12 = (uint16_t)((gun_pal[idx * 2u] << 8) | gun_pal[idx * 2u + 1]);
                    buf[sy * rw + sx] = 15;
                    if (g_renderer_rgb_raster_expand)
                        rgb[sy * rw + sx] = amiga12_to_argb(c12);
                    renderer_cw_store_xy(cw, sx, sy, rw, rh, c12);
                }
            }
            return;
        }
    }

    /* Do not draw placeholder gun when real gun data is missing or slot unused */
}

void renderer_draw_gun(GameState *state)
{
    renderer_draw_gun_columns(state, 0, g_renderer.width);
}

int renderer_get_gun_draw_info(const GameState *state, int *out_frame_slot,
                               int *out_ix, int *out_iy, int *out_iw, int *out_ih)
{
    if (!state) return 0;
    const PlayerState *plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
    if (plr->gun_selected < 0) return 0;

    int gun_type = plr->gun_selected;
    if (gun_type < 0 || gun_type >= 8) gun_type = 0;

    const GunAnim *anim = &gun_anims[gun_type];
    int anim_frame = plr->gun_frame;
    if (anim_frame > anim->num_frames) anim_frame = 0;
    int graphic_frame = anim->frames[anim_frame];
    if (graphic_frame > 3) graphic_frame = 0;
    int frame_slot = gun_type * 4 + graphic_frame;
    if (frame_slot >= 32) frame_slot = 0;

    /* Guns 5 and 6 have no data */
    if (gun_ptr_frame_offsets[frame_slot] == 0 && (gun_type == 5 || gun_type == 6)) return 0;

    /* Require gun assets to be loaded */
    if (!g_renderer.gun_wad || !g_renderer.gun_ptr || !g_renderer.gun_pal) return 0;

    int rw = g_renderer.width, rh = g_renderer.height;
    if (rw < 1 || rh < 1) return 0;

    const int gun_w_src = GUN_COLS;
    const int gun_h_src = GUN_LINES;

    int64_t scale_fp = (((int64_t)RENDER_SCALE * (int64_t)rh) << 16) / RENDER_DEFAULT_HEIGHT;
    int64_t fit_fp_w = ((int64_t)rw << 16) / gun_w_src;
    int64_t fit_fp_h = ((int64_t)rh << 16) / gun_h_src;
    if (scale_fp > fit_fp_w) scale_fp = fit_fp_w;
    if (scale_fp > fit_fp_h) scale_fp = fit_fp_h;
    if (scale_fp < 1) scale_fp = 1;

    int gun_w_draw = (int)(((int64_t)gun_w_src * scale_fp + 0x7FFF) >> 16);
    int gun_h_draw = (int)(((int64_t)gun_h_src * scale_fp + 0x7FFF) >> 16);
    if (gun_w_draw < 1) gun_w_draw = 1;
    if (gun_h_draw < 1) gun_h_draw = 1;
    if (gun_w_draw > rw) gun_w_draw = rw;
    if (gun_h_draw > rh) gun_h_draw = rh;

    int gy = rh - gun_h_draw;
    if (gy < 0) gy = 0;
    int gx = (rw - gun_w_draw) / 2;
    if (gun_type == ROCKET_LAUNCHER_GUN_IDX)
        gx = rw - gun_w_draw;

    if (out_frame_slot) *out_frame_slot = frame_slot;
    if (out_ix) *out_ix = gx;
    if (out_iy) *out_iy = gy;
    if (out_iw) *out_iw = gun_w_draw;
    if (out_ih) *out_ih = gun_h_draw;
    return 1;
}

int renderer_gun_src_width(void)  { return GUN_COLS;  }
int renderer_gun_src_height(void) { return GUN_LINES; }

int renderer_decode_gun_frame_rgba(int frame_slot, uint32_t *out_rgba)
{
    if (frame_slot < 0 || frame_slot >= 32) return 0;
    if (!g_renderer.gun_wad || !g_renderer.gun_ptr || !g_renderer.gun_pal) return 0;

    int gun_type = frame_slot / 4;
    uint32_t ptr_off = gun_ptr_frame_offsets[frame_slot];
    if (ptr_off == 0 && (gun_type == 5 || gun_type == 6)) return 0;

    const uint8_t *gun_wad = g_renderer.gun_wad;
    const uint8_t *gun_ptr = g_renderer.gun_ptr;
    const uint8_t *gun_pal = g_renderer.gun_pal;
    size_t wad_size = g_renderer.gun_wad_size;

    /* Start with all pixels transparent */
    for (int i = 0; i < GUN_COLS * GUN_LINES; i++)
        out_rgba[i] = 0u;

    for (int src_col = 0; src_col < GUN_COLS; src_col++) {
        const uint8_t *col_ptr = gun_ptr + ptr_off + (uint32_t)src_col * 4;
        uint8_t mode = col_ptr[0];
        uint32_t wad_off = ((uint32_t)col_ptr[1] << 16) | ((uint32_t)col_ptr[2] << 8) | (uint32_t)col_ptr[3];
        if (mode == 0 && wad_off == 0) continue;
        if (wad_off + (size_t)GUN_LINES * 2 > wad_size) continue;

        const uint8_t *src = gun_wad + wad_off;
        for (int src_row = 0; src_row < GUN_LINES; src_row++) {
            uint16_t w = (uint16_t)((src[src_row * 2u] << 8) | src[src_row * 2u + 1]);
            uint32_t idx;
            if (mode == 0)      idx = (uint32_t)(w & 31u);
            else if (mode == 1) idx = (uint32_t)((w >> 5) & 31u);
            else                idx = (uint32_t)((w >> 10) & 31u);
            if (idx == 0) continue;

            uint16_t c12 = (uint16_t)((gun_pal[idx * 2u] << 8) | gun_pal[idx * 2u + 1]);
            uint32_t r8 = ((uint32_t)(c12 >> 8) & 0xFu) * 0x11u;
            uint32_t g8 = ((uint32_t)(c12 >> 4) & 0xFu) * 0x11u;
            uint32_t b8 = ((uint32_t) c12        & 0xFu) * 0x11u;
            /* GL_RGBA, GL_UNSIGNED_BYTE: byte order R,G,B,A */
            out_rgba[src_row * GUN_COLS + src_col] = r8 | (g8 << 8) | (b8 << 16) | 0xFF000000u;
        }
    }
    return 1;
}

/* -----------------------------------------------------------------------
 * Sprite FRAMES tables (from ObjDraw3.ChipRam.s).
 *
 * Each frame entry is {ptr_offset, down_strip}:
 *   ptr_offset = byte offset into .ptr table for the starting column
 *   down_strip = vertical row offset within each column's data
 *
 * The Amiga stores these as dc.w pairs; ptr_offset values are in bytes
 * (column_index * 4 since each PTR entry is 4 bytes).
 * ----------------------------------------------------------------------- */
typedef struct { uint16_t ptr_off; uint16_t down_strip; } SpriteFrame;

/* Type 0 - ALIEN2: 64 cols/frame, walking 0-15, exploding 16-31, dying 32-33 */
static const SpriteFrame frames_alien[] = {
    {0,0},{64*4,0},{64*4*2,0},{64*4*3,0},{64*4*4,0},{64*4*5,0},{64*4*6,0},{64*4*7,0},
    {64*4*8,0},{64*4*9,0},{64*4*10,0},{64*4*11,0},{64*4*12,0},{64*4*13,0},{64*4*14,0},{64*4*15,0},
    /* frames 16-19: gib set 1 (Explode1Anim: vect=0 frames 16-19) */
    {4*(64*16),0},{4*(64*16+16),0},{4*(64*16+32),0},{4*(64*16+48),0},
    /* frames 20-23: gib set 2 (Explode2Anim: vect=0 frames 20-23) */
    {4*(64*16),16},{4*(64*16+16),16},{4*(64*16+32),16},{4*(64*16+48),16},
    /* frames 24-27: gib set 3 (Explode3Anim: vect=0 frames 24-27) */
    {4*(64*16),32},{4*(64*16+16),32},{4*(64*16+32),32},{4*(64*16+48),32},
    /* frames 28-31: gib set 4 (Explode4Anim: vect=0 frames 28-31) */
    {4*(64*16),48},{4*(64*16+16),48},{4*(64*16+32),48},{4*(64*16+48),48},
    /* frames 32-33: death animation (alien body lying on floor) */
    {64*4*17,0},{64*4*18,0}
};

/* Type 1 - PICKUPS: variable layout */
static const SpriteFrame frames_pickups[] = {
    {0,0},                /* 0: medikit */
    {0,32},               /* 1: big gun */
    {64*4,32},            /* 2: bullet */
    {32*4,0},             /* 3: ammo */
    {64*4,0},             /* 4: battery */
    {192*4,0},            /* 5: rockets */
    {128*4,0},{(128+16)*4,0},{(128+32)*4,0},{(128+48)*4,0},   /* 6-9: gunpop */
    {128*4,16},{(128+16)*4,16},{(128+32)*4,16},{(128+48)*4,16}, /* 10-13 */
    {128*4,32},{(128+16)*4,32},{(128+32)*4,32},{(64+16)*4,32}, /* 14-17 */
    {64*4,48},{(64+16)*4,48},                                   /* 18-19 */
    {(64+32)*4,0},        /* 20: rocket launcher */
    {64*4,32},{(64+16)*4,32},{(64+16)*4,48},{64*4,48},          /* 21-24: grenade */
    {128*4,32},           /* 25: shotgun */
    {256*4,0},            /* 26: grenade launcher */
    {64*3*4,32},          /* 27: shotgun shells*4 */
    {(64*3+32)*4,0},      /* 28: shotgun shells*20 */
    {(64*3+32)*4,32}      /* 29: grenade clip */
};

/* Type 2 - BIGBULLET */
static const SpriteFrame frames_bigbullet[] = {
    {0,0},{0,32},{32*4,0},{32*4,32},{64*4,0},{64*4,32},{96*4,0},{96*4,32},
    {128*4,0},{128*4,32},{32*5*4,0},{32*5*4,32},{32*6*4,0},{32*6*4,32},
    {32*7*4,0},{32*7*4,32},{32*8*4,0},{32*8*4,32},{32*9*4,0},{32*9*4,32}
};

/* Type 4 - FLYINGMONSTER: 64 cols/frame, 21 frames */
static const SpriteFrame frames_flying[] = {
    {0,0},{64*4,0},{64*4*2,0},{64*4*3,0},{64*4*4,0},{64*4*5,0},{64*4*6,0},{64*4*7,0},
    {64*4*8,0},{64*4*9,0},{64*4*10,0},{64*4*11,0},{64*4*12,0},{64*4*13,0},{64*4*14,0},{64*4*15,0},
    {64*4*16,0},{64*4*17,0},{64*4*18,0},{64*4*19,0},{64*4*20,0}
};

/* Type 5 - KEYS: 4 frames */
static const SpriteFrame frames_keys[] = {
    {0,0},{0,32},{32*4,0},{32*4,32}
};

/* Type 6 - ROCKETS: 12 frames */
static const SpriteFrame frames_rockets[] = {
    {0,0},{32*4,0},{0,32},{32*4,32},
    {64*4,0},{(64+32)*4,0},{64*4,32},{(64+32)*4,32},
    {128*4,0},{(128+32)*4,0},{128*4,32},{(128+32)*4,32}
};

/* Type 7 - BARREL: 1 frame */
static const SpriteFrame frames_barrel[] = { {0,0} };

/* Type 8 - EXPLOSION: 9 frames (uses explosion.wad/ptr, not bigbullet) */
static const SpriteFrame frames_explosion[] = {
    {0,0},{64*4,0},{64*4*2,0},{64*4*3,0},{64*4*4,0},{64*4*5,0},{64*4*6,0},{64*4*7,0},{64*4*8,0}
};

/* Type 9 - GUNS (in-hand): handled by gun renderer, but provide frames */
static const SpriteFrame frames_guns[] = {
    {96*4*20,0},{96*4*21,0},{96*4*22,0},{96*4*23,0},
    {96*4*4,0},{96*4*5,0},{96*4*6,0},{96*4*7,0},
    {96*4*16,0},{96*4*17,0},{96*4*18,0},{96*4*19,0},
    {96*4*12,0},{96*4*13,0},{96*4*14,0},{96*4*15,0},
    {96*4*24,0},{96*4*25,0},{96*4*26,0},{96*4*27,0},
    {0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
    {96*4*0,0},{96*4*1,0},{96*4*2,0},{96*4*3,0}
};

/* Type 10 - MARINE: 19 frames, 64 cols each */
static const SpriteFrame frames_marine[] = {
    {0,0},{64*4,0},{(64*2)*4,0},{(64*3)*4,0},{(64*4)*4,0},{(64*5)*4,0},
    {(64*6)*4,0},{(64*7)*4,0},{(64*8)*4,0},{(64*9)*4,0},{(64*10)*4,0},
    {(64*11)*4,0},{(64*12)*4,0},{(64*13)*4,0},{(64*14)*4,0},{(64*15)*4,0},
    {(64*16)*4,0},{(64*17)*4,0},{(64*18)*4,0}
};

/* Type 11 - BIGALIEN: 4 frames, 128 cols each */
static const SpriteFrame frames_bigalien[] = {
    {0,0},{128*4,0},{128*4*2,0},{128*4*3,0}
};

/* Type 12 - LAMPS: 1 frame */
static const SpriteFrame frames_lamps[] = { {0,0} };

/* Type 13 - WORM: 21 frames, 90 cols each */
static const SpriteFrame frames_worm[] = {
    {0,0},{90*4,0},{90*4*2,0},{90*4*3,0},{90*4*4,0},{90*4*5,0},{90*4*6,0},{90*4*7,0},
    {90*4*8,0},{90*4*9,0},{90*4*10,0},{90*4*11,0},{90*4*12,0},{90*4*13,0},{90*4*14,0},{90*4*15,0},
    {90*4*16,0},{90*4*17,0},{90*4*18,0},{90*4*19,0},{90*4*20,0}
};

/* Type 14 - BIGCLAWS: 18 frames, 128 cols each */
static const SpriteFrame frames_bigclaws[] = {
    {0,0},{128*4,0},{128*4*2,0},{128*4*3,0},{128*4*4,0},{128*4*5,0},{128*4*6,0},{128*4*7,0},
    {128*4*8,0},{128*4*9,0},{128*4*10,0},{128*4*11,0},{128*4*12,0},{128*4*13,0},{128*4*14,0},{128*4*15,0},
    {128*4*16,0},{128*4*17,0}
};

/* Type 15 - TREE: 22 frames, 64 cols each (with repeats) */
static const SpriteFrame frames_tree[] = {
    {0,0},{64*4,0},{64*2*4,0},{64*3*4,0},
    {0,0},{64*4,0},{64*2*4,0},{64*3*4,0},
    {0,0},{64*4,0},{64*2*4,0},{64*3*4,0},
    {0,0},{64*4,0},{64*2*4,0},{64*3*4,0},
    {0,0},{0,0},
    {32*8*4,0},{32*9*4,0},{32*10*4,0},{32*11*4,0}
};

/* Master lookup: frames table + count per sprite type */
static const struct {
    const SpriteFrame *frames;
    int count;
} sprite_frames_table[MAX_SPRITE_TYPES] = {
    { frames_alien,     sizeof(frames_alien)/sizeof(SpriteFrame) },       /* 0 */
    { frames_pickups,   sizeof(frames_pickups)/sizeof(SpriteFrame) },     /* 1 */
    { frames_bigbullet, sizeof(frames_bigbullet)/sizeof(SpriteFrame) },   /* 2 */
    { NULL, 0 },                                                          /* 3 ugly monster */
    { frames_flying,    sizeof(frames_flying)/sizeof(SpriteFrame) },      /* 4 */
    { frames_keys,      sizeof(frames_keys)/sizeof(SpriteFrame) },        /* 5 */
    { frames_rockets,   sizeof(frames_rockets)/sizeof(SpriteFrame) },     /* 6 */
    { frames_barrel,    sizeof(frames_barrel)/sizeof(SpriteFrame) },      /* 7 */
    { frames_explosion, sizeof(frames_explosion)/sizeof(SpriteFrame) },   /* 8 */
    { frames_guns,      sizeof(frames_guns)/sizeof(SpriteFrame) },        /* 9 */
    { frames_marine,    sizeof(frames_marine)/sizeof(SpriteFrame) },      /* 10 */
    { frames_bigalien,  sizeof(frames_bigalien)/sizeof(SpriteFrame) },    /* 11 */
    { frames_lamps,     sizeof(frames_lamps)/sizeof(SpriteFrame) },       /* 12 */
    { frames_worm,      sizeof(frames_worm)/sizeof(SpriteFrame) },        /* 13 */
    { frames_bigclaws,  sizeof(frames_bigclaws)/sizeof(SpriteFrame) },    /* 14 */
    { frames_tree,      sizeof(frames_tree)/sizeof(SpriteFrame) },        /* 15 */
    { frames_marine,    sizeof(frames_marine)/sizeof(SpriteFrame) },      /* 16 tough (same as 10) */
    { frames_marine,    sizeof(frames_marine)/sizeof(SpriteFrame) },      /* 17 flame (same as 10) */
    { NULL, 0 }, { NULL, 0 }
};

/* Map ObjNumber -> enemy_params index (shared with objects.c death handling). */
static int renderer_obj_type_to_enemy_index(int8_t obj_type)
{
    switch (obj_type) {
    case OBJ_NBR_ALIEN:           return 0;
    case OBJ_NBR_ROBOT:           return 1;
    case OBJ_NBR_HUGE_RED_THING:  return 2;
    case OBJ_NBR_WORM:            return 3;
    case OBJ_NBR_FLAME_MARINE:    return 4;
    case OBJ_NBR_TOUGH_MARINE:    return 5;
    case OBJ_NBR_MARINE:          return 6;  /* Mutant Marine */
    case OBJ_NBR_BIG_NASTY:       return 7;
    case OBJ_NBR_SMALL_RED_THING: return 8;  /* BigClaws */
    case OBJ_NBR_FLYING_NASTY:    return 9;
    case OBJ_NBR_EYEBALL:         return 9;  /* same as FlyingNasty */
    case OBJ_NBR_TREE:            return 10;
    default:                      return -1;
    }
}

/* Distinguish runtime corpses (ported dead-state format) from level-authored
 * decorative OBJ_NBR_DEAD entries. Decorative props must keep authored
 * obj[8]/obj[10] vect/frame, matching Amiga ObjDraw3 BitMapObj. */
static int renderer_dead_object_is_runtime_corpse(const uint8_t *obj)
{
    if (!obj) return 0;

    int8_t original_type = (int8_t)obj[19];
    int param_idx = renderer_obj_type_to_enemy_index(original_type);
    if (param_idx < 0 || param_idx >= num_enemy_types) return 0;

    int death_index = (int)(uint8_t)obj[18];
    if (death_index < 0 || death_index >= 30) return 0;

    int16_t expected_frame = enemy_params[param_idx].death_frames[death_index];
    if (expected_frame < 0) return 0;

    return rd16(obj + 8) == expected_frame;
}

static int16_t renderer_fallback_vect_for_obj_number(int8_t obj_number)
{
    switch ((ObjNumber)obj_number) {
    case OBJ_NBR_ALIEN:           return 0;
    case OBJ_NBR_MEDIKIT:         return 1;
    case OBJ_NBR_BULLET:          return 2;
    case OBJ_NBR_BIG_GUN:         return 9;
    case OBJ_NBR_KEY:             return 5;
    case OBJ_NBR_PLR1:
    case OBJ_NBR_PLR2:
    case OBJ_NBR_MARINE:          return 10;
    case OBJ_NBR_ROBOT:           return 10;
    case OBJ_NBR_BIG_NASTY:       return 11;
    case OBJ_NBR_FLYING_NASTY:    return 4;
    case OBJ_NBR_AMMO:            return 1;
    case OBJ_NBR_BARREL:          return 7;
    case OBJ_NBR_WORM:            return 13;
    case OBJ_NBR_HUGE_RED_THING:  return 14;
    case OBJ_NBR_SMALL_RED_THING: return 14;
    case OBJ_NBR_TREE:            return 15;
    case OBJ_NBR_EYEBALL:         return 0;
    case OBJ_NBR_TOUGH_MARINE:    return 16;
    case OBJ_NBR_FLAME_MARINE:    return 17;
    default:                      return 0;
    }
}

/* Keep spill sizing aligned with final billboard draw sizing so spill gating
 * uses the true on-screen footprint (worm override, fallback vect, pop/barrel
 * explosion billboard handling). */
static void renderer_resolve_billboard_world_size_for_spill(const uint8_t *obj,
                                                            int is_shot_entry,
                                                            int *out_world_w,
                                                            int *out_world_h)
{
    int world_w = 32;
    int world_h = 32;
    if (obj) {
        world_w = (int)obj[6];
        world_h = (int)obj[7];
    }
    if (world_w <= 0) world_w = 32;
    if (world_h <= 0) world_h = 32;

    if (obj) {
        int8_t obj_number = (int8_t)obj[16];
        int16_t vect_num = rd16(obj + 8);
        if (obj_number == OBJ_NBR_DEAD && renderer_dead_object_is_runtime_corpse(obj)) {
            int8_t original_type = (int8_t)obj[19];
            if (original_type >= 0 && original_type <= OBJ_NBR_GAS_PIPE) {
                obj_number = original_type;
            }
        }

        {
            int use_fallback = (vect_num < 0 || vect_num >= MAX_SPRITE_TYPES ||
                                !g_renderer.sprite_wad[vect_num] || !g_renderer.sprite_ptr[vect_num]);
            if (!use_fallback && vect_num == 0 && obj_number != OBJ_NBR_ALIEN &&
                obj_number >= OBJ_NBR_ROBOT && obj_number <= OBJ_NBR_FLAME_MARINE) {
                use_fallback = 1;
            }
            if (use_fallback) {
                vect_num = renderer_fallback_vect_for_obj_number(obj_number);
            }
        }

        if ((ObjNumber)obj_number == OBJ_NBR_WORM) {
            world_w = 90;
            world_h = 100;
        }

        if (vect_num == 8) {
            int explosion_billboard = 0;
            if ((ObjNumber)obj_number == OBJ_NBR_BARREL) {
                explosion_billboard = 1;
            } else if (is_shot_entry && (int8_t)obj[30] != 0) {
                explosion_billboard = 1;
            }
            if (explosion_billboard) {
                world_w *= EXPLOSION_SIZE_CORRECTION;
                world_h *= EXPLOSION_SIZE_CORRECTION;
                if (world_w < 1) world_w = 1;
                if (world_h < 1) world_h = 1;
            }
        }
    }

    if (out_world_w) *out_world_w = world_w;
    if (out_world_h) *out_world_h = world_h;
}

/* RockPop/Explosion world sizes are authored from Amiga BitMapObj tables where
 * both axes use the same <<7 scale.
 *
 * The port projects billboard width via runtime horizontal projection scaling,
 * but height goes through proj_y_scale. Convert Amiga-authored explosion height
 * into that Y domain using the current X sprite scale so explosions preserve the
 * original width:height relationship across aspect/render size changes. */
static inline int explosion_world_h_to_port(const RendererState *r, int world_h_amiga, int sprite_scale_x)
{
    int32_t py = r->proj_y_scale;
    if (py < 1) py = 1;
    if (sprite_scale_x < 1) sprite_scale_x = 1;
    {
        int64_t denom = (int64_t)py * (int64_t)RENDER_SCALE;
        int64_t corrected = ((int64_t)world_h_amiga * (int64_t)sprite_scale_x + (denom / 2)) / denom;
        if (corrected < 1) corrected = 1;
        if (corrected > 32767) corrected = 32767;
        return (int)corrected;
    }
}

#define RENDERER_ZONE_EXIT_LIST_OFF 32
#define RENDERER_FLINE_SIZE         16
#define RENDERER_FLINE_X_OFF        0
#define RENDERER_FLINE_Z_OFF        2
#define RENDERER_FLINE_XLEN_OFF     4
#define RENDERER_FLINE_ZLEN_OFF     6
#define RENDERER_FLINE_CONNECT_OFF  8
#define RENDERER_MAX_ADJ_ZONES      64
#define RENDERER_MAX_ADJ_LINES      512

typedef struct {
    int16_t zone_id;
    int16_t line_start;
    int16_t line_count;
    uint8_t ambiguous;
} RendererAdjZone;

static int renderer_zone_order_index(const GameState *state, int16_t zone_id)
{
    if (!state || zone_id < 0) return -1;
    int count = state->zone_order_count;
    if (count < 0) count = 0;
    if (count > RENDERER_MAX_ZONE_ORDER) count = RENDERER_MAX_ZONE_ORDER;
    for (int i = 0; i < count; i++) {
        if (state->zone_order_zones[i] == zone_id) return i;
    }
    return -1;
}

static int renderer_lookup_zone_prepass_clip_span(const RenderSliceContext *ctx,
                                                  int16_t zone_id,
                                                  int16_t *out_left,
                                                  int16_t *out_right)
{
    if (!ctx || !ctx->zone_prepass || !out_left || !out_right || zone_id < 0)
        return 0;

    const RendererWorldZonePrepass *prepass = ctx->zone_prepass;
    int count = prepass->count;
    if (count < 0) count = 0;
    if (count > RENDERER_MAX_ZONE_ORDER) count = RENDERER_MAX_ZONE_ORDER;

    for (int i = 0; i < count; i++) {
        if (prepass->zone_ids[i] != zone_id) continue;
        if (!prepass->valid[i]) continue;
        *out_left = prepass->left_px[i];
        *out_right = prepass->right_px[i];
        return 1;
    }

    return 0;
}

static int renderer_find_adj_zone_slot(const RendererAdjZone *adj, int adj_count, int16_t zone_id)
{
    for (int i = 0; i < adj_count; i++) {
        if (adj[i].zone_id == zone_id) return i;
    }
    return -1;
}

static int64_t renderer_orient2d_i64(int32_t ax, int32_t az,
                                     int32_t bx, int32_t bz,
                                     int32_t cx, int32_t cz)
{
    return ((int64_t)bx - (int64_t)ax) * ((int64_t)cz - (int64_t)az)
         - ((int64_t)bz - (int64_t)az) * ((int64_t)cx - (int64_t)ax);
}

static int renderer_on_segment_i64(int32_t ax, int32_t az,
                                   int32_t bx, int32_t bz,
                                   int32_t px, int32_t pz)
{
    int32_t minx = (ax < bx) ? ax : bx;
    int32_t maxx = (ax > bx) ? ax : bx;
    int32_t minz = (az < bz) ? az : bz;
    int32_t maxz = (az > bz) ? az : bz;
    return (px >= minx && px <= maxx && pz >= minz && pz <= maxz);
}

static int renderer_segments_intersect_i32(int32_t a1x, int32_t a1z,
                                           int32_t a2x, int32_t a2z,
                                           int32_t b1x, int32_t b1z,
                                           int32_t b2x, int32_t b2z)
{
    int64_t o1 = renderer_orient2d_i64(a1x, a1z, a2x, a2z, b1x, b1z);
    int64_t o2 = renderer_orient2d_i64(a1x, a1z, a2x, a2z, b2x, b2z);
    int64_t o3 = renderer_orient2d_i64(b1x, b1z, b2x, b2z, a1x, a1z);
    int64_t o4 = renderer_orient2d_i64(b1x, b1z, b2x, b2z, a2x, a2z);

    if (((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) &&
        ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))) {
        return 1;
    }

    if (o1 == 0 && renderer_on_segment_i64(a1x, a1z, a2x, a2z, b1x, b1z)) return 1;
    if (o2 == 0 && renderer_on_segment_i64(a1x, a1z, a2x, a2z, b2x, b2z)) return 1;
    if (o3 == 0 && renderer_on_segment_i64(b1x, b1z, b2x, b2z, a1x, a1z)) return 1;
    if (o4 == 0 && renderer_on_segment_i64(b1x, b1z, b2x, b2z, a2x, a2z)) return 1;

    return 0;
}

/* Spill criterion: use the billboard's world-space lateral segment (sprite plane
 * facing the player) and require that it intersects a boundary line shared with
 * the destination zone. */
static int renderer_billboard_lateral_hits_adj_lines(const LevelState *level,
                                                     int32_t px, int32_t pz,
                                                     int32_t half_span_world,
                                                     int16_t view_right_x,
                                                     int16_t view_right_z,
                                                     const int16_t *line_ids,
                                                     int line_count)
{
    if (!level || !level->floor_lines || !line_ids || line_count <= 0) return 0;
    if (half_span_world < 1) half_span_world = 1;

    {
        int64_t off_x64 = ((int64_t)view_right_x * (int64_t)half_span_world) / 16384;
        int64_t off_z64 = ((int64_t)view_right_z * (int64_t)half_span_world) / 16384;
        if (off_x64 == 0 && off_z64 == 0) {
            if (view_right_x != 0) {
                off_x64 = (view_right_x > 0) ? 1 : -1;
            } else if (view_right_z != 0) {
                off_z64 = (view_right_z > 0) ? 1 : -1;
            } else {
                off_x64 = 1;
            }
        }

        int32_t sx1 = (int32_t)((int64_t)px - off_x64);
        int32_t sz1 = (int32_t)((int64_t)pz - off_z64);
        int32_t sx2 = (int32_t)((int64_t)px + off_x64);
        int32_t sz2 = (int32_t)((int64_t)pz + off_z64);

        for (int i = 0; i < line_count; i++) {
            int16_t li = line_ids[i];
            if (li < 0 || (int32_t)li >= level->num_floor_lines) continue;
            {
                const uint8_t *fl = level->floor_lines + (size_t)(uint16_t)li * RENDERER_FLINE_SIZE;
                int32_t x1 = rd16(fl + RENDERER_FLINE_X_OFF);
                int32_t z1 = rd16(fl + RENDERER_FLINE_Z_OFF);
                int32_t x2 = x1 + rd16(fl + RENDERER_FLINE_XLEN_OFF);
                int32_t z2 = z1 + rd16(fl + RENDERER_FLINE_ZLEN_OFF);
                if (renderer_segments_intersect_i32(sx1, sz1, sx2, sz2, x1, z1, x2, z2)) {
                    return 1;
                }
            }
        }
    }

    return 0;
}

static const uint8_t *renderer_zone_exit_list_ptr(const LevelState *level, int16_t zone_id)
{
    int zone_slots;
    int32_t zone_off;
    int16_t list_off;
    int64_t list_abs;
    if (!level || !level->data || !level->zone_adds) return NULL;
    zone_slots = level_zone_slot_count(level);
    if (zone_slots <= 0 || zone_id < 0 || zone_id >= zone_slots) return NULL;

    zone_off = rd32(level->zone_adds + (size_t)(uint16_t)zone_id * 4u);
    if (zone_off < 0) return NULL;
    if (level->data_byte_count > 0 && (size_t)zone_off + 34u > level->data_byte_count) return NULL;

    list_off = rd16(level->data + zone_off + RENDERER_ZONE_EXIT_LIST_OFF);
    list_abs = (int64_t)zone_off + (int64_t)list_off;
    if (list_abs < 0) return NULL;
    if (level->data_byte_count > 0 && (size_t)list_abs + 2u > level->data_byte_count) return NULL;
    return level->data + (size_t)list_abs;
}

static int renderer_point_on_segment_i32(int32_t px, int32_t pz,
                                         int32_t x1, int32_t z1,
                                         int32_t x2, int32_t z2)
{
    int64_t cross = (int64_t)(px - x1) * (int64_t)(z2 - z1) -
                    (int64_t)(pz - z1) * (int64_t)(x2 - x1);
    if (cross != 0) return 0;
    if (px < ((x1 < x2) ? x1 : x2) || px > ((x1 > x2) ? x1 : x2)) return 0;
    if (pz < ((z1 < z2) ? z1 : z2) || pz > ((z1 > z2) ? z1 : z2)) return 0;
    return 1;
}

static int renderer_zone_contains_point(const LevelState *level, int16_t zone_id, int32_t x, int32_t z)
{
    if (!level || !level->data || !level->zone_adds || !level->floor_lines)
        return 0;

    {
        int zone_slots = level_zone_slot_count(level);
        if (zone_slots <= 0 || zone_id < 0 || zone_id >= zone_slots)
            return 0;
    }

    {
        const uint8_t *list = renderer_zone_exit_list_ptr(level, zone_id);
        size_t data_len = level->data_byte_count;
        int inside = 0;
        int edges = 0;

        if (!list) return 0;

        for (int i = 0; i < 256; i++) {
            if (data_len != 0) {
                size_t list_off = (size_t)(list - level->data) + (size_t)i * 2u;
                if (list_off + 2u > data_len)
                    break;
            }

            {
                int16_t entry = rd16(list + (size_t)i * 2u);
                if (entry == -2)
                    break;
                if (entry < 0)
                    continue; /* -1 separates exits and wall list */
                if ((int32_t)entry >= level->num_floor_lines)
                    continue;

                {
                    const uint8_t *fl = level->floor_lines + (size_t)(uint16_t)entry * RENDERER_FLINE_SIZE;
                    int32_t x1 = rd16(fl + RENDERER_FLINE_X_OFF);
                    int32_t z1 = rd16(fl + RENDERER_FLINE_Z_OFF);
                    int32_t x2 = x1 + rd16(fl + RENDERER_FLINE_XLEN_OFF);
                    int32_t z2 = z1 + rd16(fl + RENDERER_FLINE_ZLEN_OFF);
                    edges++;

                    if (renderer_point_on_segment_i32(x, z, x1, z1, x2, z2))
                        return 1;

                    if ((z1 > z) != (z2 > z)) {
                        int64_t dz = (int64_t)z2 - (int64_t)z1;
                        int64_t lhs = (int64_t)(x - x1) * dz;
                        int64_t rhs = (int64_t)(x2 - x1) * (int64_t)(z - z1);
                        if ((dz > 0 && lhs < rhs) || (dz < 0 && lhs > rhs))
                            inside ^= 1;
                    }
                }
            }
        }

        if (edges == 0) return 0;
        return inside;
    }
}

static void renderer_geom_clip_add_t_unique(double *vals, int *io_count, int max_vals, double t)
{
    const double eps = 1e-9;
    int count;
    if (!vals || !io_count || max_vals <= 0) return;

    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;

    count = *io_count;
    if (count < 0) count = 0;
    if (count > max_vals) count = max_vals;

    for (int i = 0; i < count; i++) {
        if (fabs(vals[i] - t) <= eps) return;
    }

    if (count >= max_vals) return;
    vals[count] = t;
    *io_count = count + 1;
}

static int renderer_zone_clip_segment_to_dx_ranges(const LevelState *level,
                                                   int16_t zone_id,
                                                   int32_t seg_x0,
                                                   int32_t seg_z0,
                                                   int32_t seg_x1,
                                                   int32_t seg_z1,
                                                   int width,
                                                   int clip_dx_start,
                                                   int clip_dx_end,
                                                   int *out_dx_start,
                                                   int *out_dx_end,
                                                   int max_ranges)
{
    const double eps = 1e-9;
    const uint8_t *list;
    size_t data_len;
    double t_vals[RENDERER_GEOM_CLIP_MAX_T_VALUES];
    int t_count = 0;
    int out_count = 0;

    if (!level || !level->data || !level->zone_adds || !level->floor_lines) return 0;
    if (!out_dx_start || !out_dx_end || max_ranges <= 0) return 0;
    if (width <= 0 || clip_dx_end <= clip_dx_start) return 0;

    list = renderer_zone_exit_list_ptr(level, zone_id);
    if (!list) return 0;
    data_len = level->data_byte_count;

    renderer_geom_clip_add_t_unique(t_vals, &t_count, RENDERER_GEOM_CLIP_MAX_T_VALUES, 0.0);
    renderer_geom_clip_add_t_unique(t_vals, &t_count, RENDERER_GEOM_CLIP_MAX_T_VALUES, 1.0);

    for (int i = 0; i < 256; i++) {
        if (data_len != 0) {
            size_t list_off = (size_t)(list - level->data) + (size_t)i * 2u;
            if (list_off + 2u > data_len)
                break;
        }

        {
            int16_t entry = rd16(list + (size_t)i * 2u);
            if (entry == -2)
                break;
            if (entry < 0)
                continue;
            if ((int32_t)entry >= level->num_floor_lines)
                continue;

            {
                const uint8_t *fl = level->floor_lines + (size_t)(uint16_t)entry * RENDERER_FLINE_SIZE;
                int32_t ex0 = rd16(fl + RENDERER_FLINE_X_OFF);
                int32_t ez0 = rd16(fl + RENDERER_FLINE_Z_OFF);
                int32_t ex1 = ex0 + rd16(fl + RENDERER_FLINE_XLEN_OFF);
                int32_t ez1 = ez0 + rd16(fl + RENDERER_FLINE_ZLEN_OFF);

                if (!renderer_segments_intersect_i32(seg_x0, seg_z0, seg_x1, seg_z1,
                                                     ex0, ez0, ex1, ez1)) {
                    continue;
                }

                {
                    double sdx = (double)seg_x1 - (double)seg_x0;
                    double sdz = (double)seg_z1 - (double)seg_z0;
                    double edx = (double)ex1 - (double)ex0;
                    double edz = (double)ez1 - (double)ez0;
                    double denom = sdx * edz - sdz * edx;

                    if (fabs(denom) > eps) {
                        double nx = (double)ex0 - (double)seg_x0;
                        double nz = (double)ez0 - (double)seg_z0;
                        double t = (nx * edz - nz * edx) / denom;
                        renderer_geom_clip_add_t_unique(t_vals, &t_count,
                                                        RENDERER_GEOM_CLIP_MAX_T_VALUES,
                                                        t);
                    } else {
                        int64_t o0 = renderer_orient2d_i64(seg_x0, seg_z0, seg_x1, seg_z1, ex0, ez0);
                        int64_t o1 = renderer_orient2d_i64(seg_x0, seg_z0, seg_x1, seg_z1, ex1, ez1);
                        if (o0 == 0 && o1 == 0) {
                            if (llabs((long long)seg_x1 - (long long)seg_x0) >=
                                llabs((long long)seg_z1 - (long long)seg_z0)) {
                                int64_t denom_i = (int64_t)seg_x1 - (int64_t)seg_x0;
                                if (denom_i != 0) {
                                    double t0 = ((double)ex0 - (double)seg_x0) / (double)denom_i;
                                    double t1 = ((double)ex1 - (double)seg_x0) / (double)denom_i;
                                    renderer_geom_clip_add_t_unique(t_vals, &t_count,
                                                                    RENDERER_GEOM_CLIP_MAX_T_VALUES,
                                                                    t0);
                                    renderer_geom_clip_add_t_unique(t_vals, &t_count,
                                                                    RENDERER_GEOM_CLIP_MAX_T_VALUES,
                                                                    t1);
                                }
                            } else {
                                int64_t denom_i = (int64_t)seg_z1 - (int64_t)seg_z0;
                                if (denom_i != 0) {
                                    double t0 = ((double)ez0 - (double)seg_z0) / (double)denom_i;
                                    double t1 = ((double)ez1 - (double)seg_z0) / (double)denom_i;
                                    renderer_geom_clip_add_t_unique(t_vals, &t_count,
                                                                    RENDERER_GEOM_CLIP_MAX_T_VALUES,
                                                                    t0);
                                    renderer_geom_clip_add_t_unique(t_vals, &t_count,
                                                                    RENDERER_GEOM_CLIP_MAX_T_VALUES,
                                                                    t1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i = 1; i < t_count; i++) {
        double v = t_vals[i];
        int j = i - 1;
        while (j >= 0 && t_vals[j] > v) {
            t_vals[j + 1] = t_vals[j];
            j--;
        }
        t_vals[j + 1] = v;
    }

    for (int i = 0; i + 1 < t_count; i++) {
        double a = t_vals[i];
        double b = t_vals[i + 1];
        double mid;
        int32_t mx;
        int32_t mz;
        int sx;
        int ex;

        if (b - a <= eps) continue;

        mid = (a + b) * 0.5;
        mx = (int32_t)llround((double)seg_x0 + ((double)seg_x1 - (double)seg_x0) * mid);
        mz = (int32_t)llround((double)seg_z0 + ((double)seg_z1 - (double)seg_z0) * mid);
        if (!renderer_zone_contains_point(level, zone_id, mx, mz)) continue;

        sx = (int)floor(a * (double)width);
        ex = (int)ceil(b * (double)width);

        if (sx < 0) sx = 0;
        if (ex > width) ex = width;
        if (sx < clip_dx_start) sx = clip_dx_start;
        if (ex > clip_dx_end) ex = clip_dx_end;
        if (ex <= sx) continue;

        if (out_count > 0 && sx <= out_dx_end[out_count - 1]) {
            if (ex > out_dx_end[out_count - 1])
                out_dx_end[out_count - 1] = ex;
            continue;
        }

        if (out_count >= max_ranges) {
            break;
        }
        out_dx_start[out_count] = sx;
        out_dx_end[out_count] = ex;
        out_count++;
    }

    return out_count;
}

static int renderer_zone_list_contains(const int16_t *zones, int count, int16_t zone_id)
{
    if (!zones || count <= 0) return 0;
    for (int i = 0; i < count; i++) {
        if (zones[i] == zone_id) return 1;
    }
    return 0;
}

static int renderer_collect_connected_zones_fast(const LevelState *level,
                                                 int16_t zone_id,
                                                 int16_t *out_zones,
                                                 int max_zones)
{
    const uint8_t *exit_list;
    int zone_slots;
    int out_count = 0;

    if (!level || !out_zones || max_zones <= 0 || !level->floor_lines) return 0;

    zone_slots = level_zone_slot_count(level);
    if (zone_slots <= 0 || zone_id < 0 || zone_id >= zone_slots) return 0;

    exit_list = renderer_zone_exit_list_ptr(level, zone_id);
    if (!exit_list) return 0;

    for (int i = 0; i < 128 && out_count < max_zones; i++) {
        int16_t entry = rd16(exit_list + (size_t)i * 2u);
        if (entry < 0) break;
        if ((int32_t)entry >= level->num_floor_lines) continue;

        {
            const uint8_t *fl = level->floor_lines + (size_t)(uint16_t)entry * RENDERER_FLINE_SIZE;
            int16_t connect = rd16(fl + RENDERER_FLINE_CONNECT_OFF);
            int connect_zone = level_connect_to_zone_index(level, connect);
            if (connect_zone < 0 || connect_zone >= zone_slots || connect_zone == zone_id) continue;
            if (renderer_zone_list_contains(out_zones, out_count, (int16_t)connect_zone)) continue;
            out_zones[out_count++] = (int16_t)connect_zone;
        }
    }

    /* Include reverse links as well so one-way or asymmetrical connect data
     * does not block spill propagation. */
    for (int src_zone = 0; src_zone < zone_slots && out_count < max_zones; src_zone++) {
        const uint8_t *src_list;
        int points_to_zone = 0;

        if (src_zone == zone_id) continue;
        if (renderer_zone_list_contains(out_zones, out_count, (int16_t)src_zone)) continue;

        src_list = renderer_zone_exit_list_ptr(level, (int16_t)src_zone);
        if (!src_list) continue;

        for (int e = 0; e < 128; e++) {
            int16_t entry = rd16(src_list + (size_t)e * 2u);
            if (entry < 0) break;
            if ((int32_t)entry >= level->num_floor_lines) continue;

            {
                const uint8_t *fl = level->floor_lines + (size_t)(uint16_t)entry * RENDERER_FLINE_SIZE;
                int16_t connect = rd16(fl + RENDERER_FLINE_CONNECT_OFF);
                int connect_zone = level_connect_to_zone_index(level, connect);
                if (connect_zone == zone_id) {
                    points_to_zone = 1;
                    break;
                }
            }
        }

        if (!points_to_zone) continue;
        out_zones[out_count++] = (int16_t)src_zone;
    }

    return out_count;
}

static int renderer_compute_billboard_world_segment(int32_t world_x,
                                                    int32_t world_z,
                                                    int32_t world_w,
                                                    int16_t view_right_x,
                                                    int16_t view_right_z,
                                                    int32_t *out_x0,
                                                    int32_t *out_z0,
                                                    int32_t *out_x1,
                                                    int32_t *out_z1)
{
    int32_t half_span_world;
    int64_t off_x64;
    int64_t off_z64;

    if (!out_x0 || !out_z0 || !out_x1 || !out_z1) return 0;
    if (world_w <= 0) return 0;

    half_span_world = world_w / 2;
    if (half_span_world < 1) half_span_world = 1;

    off_x64 = ((int64_t)view_right_x * (int64_t)half_span_world) / 16384;
    off_z64 = ((int64_t)view_right_z * (int64_t)half_span_world) / 16384;
    if (off_x64 == 0 && off_z64 == 0) {
        if (view_right_x != 0) {
            off_x64 = (view_right_x > 0) ? 1 : -1;
        } else if (view_right_z != 0) {
            off_z64 = (view_right_z > 0) ? 1 : -1;
        } else {
            off_x64 = 1;
        }
    }

    *out_x0 = (int32_t)((int64_t)world_x - off_x64);
    *out_z0 = (int32_t)((int64_t)world_z - off_z64);
    *out_x1 = (int32_t)((int64_t)world_x + off_x64);
    *out_z1 = (int32_t)((int64_t)world_z + off_z64);
    return 1;
}

static int renderer_zone_span_overlaps_sprite(const RenderSliceContext *ctx,
                                              GameState *state,
                                              int16_t zone_id,
                                              int sprite_left,
                                              int sprite_right,
                                              int *out_clip_dx_start,
                                              int *out_clip_dx_end)
{
    int16_t zone_left = 0;
    int16_t zone_right = 0;
    int have_zone_clip;
    int l;
    int r;

    if (!ctx || !state || sprite_right <= sprite_left) return 0;

    have_zone_clip = renderer_lookup_zone_prepass_clip_span(ctx, zone_id, &zone_left, &zone_right);
    if (!have_zone_clip) {
        have_zone_clip = renderer_compute_zone_clip_span(state, zone_id, 0u, 0,
                                                         &zone_left, &zone_right);
    }
    if (!have_zone_clip) {
        /* Fail-open for spill discovery: if we cannot resolve a reliable
         * destination clip span, fall back to the current worker slice.
         * Hard rejection here suppresses basic spill across valid neighbors. */
        zone_left = ctx->left_clip;
        zone_right = ctx->right_clip;
    }

    l = sprite_left;
    r = sprite_right;
    if (l < ctx->left_clip) l = ctx->left_clip;
    if (r > ctx->right_clip) r = ctx->right_clip;
    if (l < zone_left) l = zone_left;
    if (r > zone_right) r = zone_right;
    if (r <= l) return 0;

    if (out_clip_dx_start) *out_clip_dx_start = l - sprite_left;
    if (out_clip_dx_end) *out_clip_dx_end = r - sprite_left;
    return 1;
}

static int renderer_collect_spill_zones_recursive(const RenderSliceContext *ctx,
                                                  GameState *state,
                                                  int16_t source_zone,
                                                  int32_t billboard_world_x,
                                                  int32_t billboard_world_z,
                                                  int32_t billboard_world_w,
                                                  int16_t billboard_view_right_x,
                                                  int16_t billboard_view_right_z,
                                                  int sprite_left,
                                                  int sprite_right,
                                                  int sprite_width,
                                                  int16_t *out_zones,
                                                  int max_zones)
{
    const LevelState *level;
    int32_t seg_x0;
    int32_t seg_z0;
    int32_t seg_x1;
    int32_t seg_z1;
    int16_t visited[RENDERER_SPILL_SEARCH_MAX_VISITED];
    int16_t queue[RENDERER_SPILL_SEARCH_MAX_VISITED];
    int visited_count = 0;
    int q_head = 0;
    int q_tail = 0;
    int out_count = 0;
    const int geometry_clip_enabled = 0;

    if (!ctx || !state || !out_zones || max_zones <= 0 || source_zone < 0) return 0;
    if (sprite_width <= 0 || sprite_right <= sprite_left) return 0;

    level = &state->level;
    if (geometry_clip_enabled) {
        if (!renderer_compute_billboard_world_segment(billboard_world_x,
                                                      billboard_world_z,
                                                      billboard_world_w,
                                                      billboard_view_right_x,
                                                      billboard_view_right_z,
                                                      &seg_x0, &seg_z0,
                                                      &seg_x1, &seg_z1)) {
            return 0;
        }
    } else {
        seg_x0 = 0;
        seg_z0 = 0;
        seg_x1 = 0;
        seg_z1 = 0;
    }

    visited[visited_count++] = source_zone;
    queue[q_tail++] = source_zone;

    while (q_head < q_tail) {
        int16_t cur_zone = queue[q_head++];
        int16_t neighbors[RENDERER_SPILL_SEARCH_MAX_NEIGHBORS];
        int neighbor_count = renderer_collect_connected_zones_fast(level,
                                                                   cur_zone,
                                                                   neighbors,
                                                                   RENDERER_SPILL_SEARCH_MAX_NEIGHBORS);

        for (int ni = 0; ni < neighbor_count; ni++) {
            int16_t next_zone = neighbors[ni];
            int clip_dx_start = 0;
            int clip_dx_end = 0;
            int geom_dx_start[RENDERER_SPILL_SEARCH_MAX_GEOM_SPANS];
            int geom_dx_end[RENDERER_SPILL_SEARCH_MAX_GEOM_SPANS];
            int geom_spans;

            if (renderer_zone_list_contains(visited, visited_count, next_zone)) continue;

            if (!geometry_clip_enabled) {
                /* Geometry-off salvage mode: do not require span-overlap
                 * pruning for spill discovery. Conservative overlap tests can
                 * miss basic edge cases and suppress spill almost entirely. */
                clip_dx_start = 0;
                clip_dx_end = sprite_width;
            } else {
                if (!renderer_zone_span_overlaps_sprite(ctx,
                                                        state,
                                                        next_zone,
                                                        sprite_left,
                                                        sprite_right,
                                                        &clip_dx_start,
                                                        &clip_dx_end)) {
                    continue;
                }

                if (clip_dx_start < 0) clip_dx_start = 0;
                if (clip_dx_end > sprite_width) clip_dx_end = sprite_width;
                if (clip_dx_end <= clip_dx_start) continue;
            }

            if (geometry_clip_enabled) {
                geom_spans = renderer_zone_clip_segment_to_dx_ranges(level,
                                                                     next_zone,
                                                                     seg_x0, seg_z0,
                                                                     seg_x1, seg_z1,
                                                                     sprite_width,
                                                                     clip_dx_start,
                                                                     clip_dx_end,
                                                                     geom_dx_start,
                                                                     geom_dx_end,
                                                                     RENDERER_SPILL_SEARCH_MAX_GEOM_SPANS);
                if (geom_spans <= 0) {
                    /* Fail-open spill discovery: keep overlap-qualified adjacent
                     * zones even when geometric clip misses. Geometry can be
                     * incomplete/ambiguous at some boundaries, and hard rejection
                     * here causes spill to disappear almost entirely. */
                    geom_spans = 0;
                }
            } else {
                /* Geometry clipping disabled: keep this neighbor based on
                 * clip-span overlap alone. */
                geom_spans = 0;
            }

            if (visited_count >= RENDERER_SPILL_SEARCH_MAX_VISITED ||
                q_tail >= RENDERER_SPILL_SEARCH_MAX_VISITED) {
                return out_count;
            }

            visited[visited_count++] = next_zone;
            queue[q_tail++] = next_zone;

            if (next_zone != source_zone && !renderer_zone_list_contains(out_zones, out_count, next_zone)) {
                if (out_count < max_zones) {
                    out_zones[out_count++] = next_zone;
                }
            }
        }
    }

    return out_count;
}

static int renderer_resolve_zone_section_world_bounds(const LevelState *level,
                                                      int16_t zone_id,
                                                      int level_filter,
                                                      int32_t *out_top_world,
                                                      int32_t *out_bot_world)
{
    const uint8_t *zone_data;
    int zone_slots;
    int32_t zone_off;
    int32_t zone_top;
    int32_t zone_bot;

    if (!level || !level->data || !level->zone_adds || !out_top_world || !out_bot_world)
        return 0;

    zone_slots = level_zone_slot_count(level);
    if (zone_slots <= 0 || zone_id < 0 || zone_id >= zone_slots) return 0;

    zone_off = rd32(level->zone_adds + (size_t)(uint16_t)zone_id * 4u);
    if (zone_off < 0) return 0;
    if (level->data_byte_count > 0 && (size_t)zone_off + 48u > level->data_byte_count)
        return 0;

    zone_data = level->data + (size_t)zone_off;

    if (level_filter == 1) {
        int32_t upper_gfx_off;
        int has_upper_stream = 0;

        if (!level->zone_graph_adds) return 0;

        upper_gfx_off = rd32(level->zone_graph_adds + (size_t)(uint16_t)zone_id * 8u + 4u);
        has_upper_stream = (upper_gfx_off > 0) &&
                           (level->graphics_byte_count == 0 ||
                            ((size_t)upper_gfx_off + 2u <= level->graphics_byte_count));
        if (has_upper_stream) {
            zone_top = rd32(zone_data + ZONE_OFF_UPPER_ROOF);
            zone_bot = rd32(zone_data + ZONE_OFF_UPPER_FLOOR);
        } else {
            /* Spill from an upper pass into a single-level destination should
             * still draw; fall back to lower section bounds instead of dropping
             * the spill candidate entirely. */
            zone_top = rd32(zone_data + ZONE_OFF_ROOF);
            zone_bot = rd32(zone_data + ZONE_OFF_FLOOR);
        }
    } else {
        zone_top = rd32(zone_data + ZONE_OFF_ROOF);
        zone_bot = rd32(zone_data + ZONE_OFF_FLOOR);
    }

    if (zone_top > zone_bot) {
        int32_t t = zone_top;
        zone_top = zone_bot;
        zone_bot = t;
    }

    *out_top_world = zone_top;
    *out_bot_world = zone_bot;
    return 1;
}

static void renderer_append_boundary_lines_between_zones(const LevelState *level,
                                                         const uint8_t *from_exit_list,
                                                         int zone_slots,
                                                         int16_t target_zone,
                                                         int16_t *out_lines,
                                                         int max_lines,
                                                         int line_start,
                                                         int *io_line_count)
{
    if (!level || !from_exit_list || !out_lines || !io_line_count || max_lines <= 0) return;
    if (target_zone < 0 || target_zone >= zone_slots) return;

    for (int i = 0; i < 128; i++) {
        int16_t entry = rd16(from_exit_list + (size_t)i * 2u);
        if (entry < 0) break; /* -1 ends exits, -2 ends list */
        if (entry < 0 || (int32_t)entry >= level->num_floor_lines) continue;

        {
            const uint8_t *fl = level->floor_lines + (size_t)(uint16_t)entry * RENDERER_FLINE_SIZE;
            int16_t connect = rd16(fl + RENDERER_FLINE_CONNECT_OFF);
            int connect_zone = level_connect_to_zone_index(level, connect);
            if (connect_zone != target_zone) continue;
        }

        {
            int exists = 0;
            for (int j = line_start; j < *io_line_count; j++) {
                if (out_lines[j] == entry) {
                    exists = 1;
                    break;
                }
            }
            if (exists) continue;
        }

        if (*io_line_count < max_lines) {
            out_lines[*io_line_count] = entry;
            (*io_line_count)++;
        }
    }
}

static int renderer_collect_adjacent_zone_sources(const RenderSliceContext *ctx, GameState *state, int16_t zone_id,
                                                  RendererAdjZone *out_adj, int max_adj,
                                                  int16_t *out_lines, int max_lines)
{
    if (!ctx || !state || !out_adj || max_adj <= 0 || !out_lines || max_lines <= 0) return 0;
    LevelState *level = &state->level;
    if (!level->data || !level->zone_adds || !level->floor_lines) return 0;

    int zone_slots = level_zone_slot_count(level);
    if (zone_slots <= 0 || zone_id < 0 || zone_id >= zone_slots) return 0;
    if (zone_slots > 256) zone_slots = 256;

    int adj_count = 0;
    int line_count = 0;
    int cur_order = renderer_zone_order_index(state, zone_id);
    const uint8_t *cur_list = renderer_zone_exit_list_ptr(level, zone_id);
    uint8_t candidate_zones[256];

    if (!cur_list) return 0;

    memset(candidate_zones, 0, sizeof(candidate_zones));

    /* Forward links: current zone exit list points to adjacent source zone.
     * Do not require the source zone to already be in zone_order: that makes
     * spill distance-sensitive, because a nearby billboard can legitimately
     * cross into the current zone before the whole neighboring zone is visible. */
    for (int i = 0; i < 128; i++) {
        int16_t entry = rd16(cur_list + (size_t)i * 2u);
        if (entry < 0) break; /* -1 ends exits, -2 ends list */
        if (entry < 0 || (int32_t)entry >= level->num_floor_lines) continue;

        const uint8_t *fl = level->floor_lines + (size_t)(uint16_t)entry * RENDERER_FLINE_SIZE;
        int16_t connect = rd16(fl + RENDERER_FLINE_CONNECT_OFF);
        int connect_zone = level_connect_to_zone_index(level, connect);
        if (connect_zone < 0 || connect_zone >= zone_slots || connect_zone == zone_id) continue;

        candidate_zones[(uint8_t)connect_zone] = 1;
    }

    /* Reverse links: any zone that points to current zone (covers one-way/broken links). */
    for (int src_zone = 0; src_zone < zone_slots; src_zone++) {
        if (src_zone < 0 || src_zone >= zone_slots || src_zone == zone_id) continue;
        const uint8_t *src_list = renderer_zone_exit_list_ptr(level, src_zone);
        if (!src_list) continue;
        for (int e = 0; e < 128; e++) {
            int16_t entry = rd16(src_list + (size_t)e * 2u);
            if (entry < 0) break;
            if (entry < 0 || (int32_t)entry >= level->num_floor_lines) continue;
            {
                const uint8_t *fl = level->floor_lines + (size_t)(uint16_t)entry * RENDERER_FLINE_SIZE;
                int16_t connect = rd16(fl + RENDERER_FLINE_CONNECT_OFF);
                int connect_zone = level_connect_to_zone_index(level, connect);
                if (connect_zone == zone_id) {
                    if (src_zone < 256) candidate_zones[(uint8_t)src_zone] = 1;
                    break;
                }
            }
        }
    }

    /* Build contiguous per-zone line spans to avoid mixed-zone line lists. */
    for (int src_zone = 0; src_zone < zone_slots; src_zone++) {
        int line_start;
        const uint8_t *src_list;
        if (src_zone < 0 || src_zone >= zone_slots || src_zone == zone_id) continue;
        if (src_zone >= 256 || !candidate_zones[(uint8_t)src_zone]) continue;
        if (adj_count >= max_adj) continue;

        line_start = line_count;
        src_list = renderer_zone_exit_list_ptr(level, src_zone);
        renderer_append_boundary_lines_between_zones(level, cur_list, zone_slots, src_zone,
                                                     out_lines, max_lines, line_start, &line_count);
        renderer_append_boundary_lines_between_zones(level, src_list, zone_slots, zone_id,
                                                     out_lines, max_lines, line_start, &line_count);
        int extracted_lines = line_count - line_start;
        if (extracted_lines < 0) extracted_lines = 0;

        out_adj[adj_count].zone_id = src_zone;
        out_adj[adj_count].line_start = (int16_t)line_start;
        out_adj[adj_count].line_count = (int16_t)extracted_lines;
        out_adj[adj_count].ambiguous = (uint8_t)(extracted_lines <= 0 ? 1 : 0);
        adj_count++;
    }

    for (int i = 0; i < adj_count; i++) {
        int ambiguous = (out_adj[i].ambiguous != 0) ? 1 : 0;
        int16_t adj_left = 0, adj_right = 0;
        int have_adj_clip = renderer_lookup_zone_prepass_clip_span(ctx, out_adj[i].zone_id,
                                                                   &adj_left, &adj_right);
        if (!have_adj_clip) {
            have_adj_clip = renderer_compute_zone_clip_span(state, out_adj[i].zone_id,
                                                            0u, 0, &adj_left, &adj_right);
        }
        if (have_adj_clip) {
            int l = (adj_left > ctx->left_clip) ? adj_left : ctx->left_clip;
            int r = (adj_right < ctx->right_clip) ? adj_right : ctx->right_clip;
            if (l < r) ambiguous = 1;
        }
        if (!ambiguous) {
            int src_order = renderer_zone_order_index(state, out_adj[i].zone_id);
            if (src_order >= 0 && cur_order >= 0) {
                int d = src_order - cur_order;
                if (d < 0) d = -d;
                if (d <= 2) ambiguous = 1;
            }
        }
        out_adj[i].ambiguous = (uint8_t)(ambiguous ? 1 : 0);
    }

    return adj_count;
}

static int renderer_f2_zone_list_add_unique(int16_t *zones,
                                            int max_zones,
                                            uint8_t *io_count,
                                            int16_t zone_id)
{
    if (!zones || !io_count || max_zones <= 0 || zone_id < 0) return 0;

    int count = (int)(*io_count);
    if (count < 0) count = 0;
    if (count > max_zones) count = max_zones;
    for (int i = 0; i < count; i++) {
        if (zones[i] == zone_id) {
            return 0;
        }
    }
    if (count >= max_zones) {
        return 0;
    }

    zones[count] = zone_id;
    *io_count = (uint8_t)(count + 1);
    return 1;
}

static int renderer_collect_adjacent_zones_for_f2(const LevelState *level,
                                                  int16_t zone_id,
                                                  int16_t *out_zones,
                                                  int max_zones)
{
    if (!level || !out_zones || max_zones <= 0 || !level->floor_lines)
        return 0;

    int zone_slots = level_zone_slot_count(level);
    if (zone_slots <= 0 || zone_id < 0 || zone_id >= zone_slots)
        return 0;
    if (zone_slots > 256)
        zone_slots = 256;

    int out_count = 0;
    uint8_t seen[256];
    memset(seen, 0, sizeof(seen));

    {
        const uint8_t *cur_list = renderer_zone_exit_list_ptr(level, zone_id);
        if (cur_list) {
            for (int i = 0; i < 128; i++) {
                int16_t entry = rd16(cur_list + (size_t)i * 2u);
                if (entry < 0) break;
                if (entry < 0 || (int32_t)entry >= level->num_floor_lines) continue;

                const uint8_t *fl = level->floor_lines + (size_t)(uint16_t)entry * RENDERER_FLINE_SIZE;
                int16_t connect = rd16(fl + RENDERER_FLINE_CONNECT_OFF);
                int connect_zone = level_connect_to_zone_index(level, connect);
                if (connect_zone < 0 || connect_zone >= zone_slots || connect_zone == zone_id)
                    continue;
                if (connect_zone >= 256)
                    continue;
                if (seen[(uint8_t)connect_zone])
                    continue;

                seen[(uint8_t)connect_zone] = 1;
                if (out_count < max_zones) {
                    out_zones[out_count] = (int16_t)connect_zone;
                }
                out_count++;
            }
        }
    }

    for (int src_zone = 0; src_zone < zone_slots; src_zone++) {
        if (src_zone == zone_id) continue;
        if (seen[(uint8_t)src_zone]) continue;

        const uint8_t *src_list = renderer_zone_exit_list_ptr(level, (int16_t)src_zone);
        if (!src_list) continue;

        int points_to_zone = 0;
        for (int e = 0; e < 128; e++) {
            int16_t entry = rd16(src_list + (size_t)e * 2u);
            if (entry < 0) break;
            if (entry < 0 || (int32_t)entry >= level->num_floor_lines) continue;

            const uint8_t *fl = level->floor_lines + (size_t)(uint16_t)entry * RENDERER_FLINE_SIZE;
            int16_t connect = rd16(fl + RENDERER_FLINE_CONNECT_OFF);
            int connect_zone = level_connect_to_zone_index(level, connect);
            if (connect_zone == zone_id) {
                points_to_zone = 1;
                break;
            }
        }
        if (!points_to_zone)
            continue;

        seen[(uint8_t)src_zone] = 1;
        if (out_count < max_zones) {
            out_zones[out_count] = (int16_t)src_zone;
        }
        out_count++;
    }

    if (out_count > max_zones)
        out_count = max_zones;
    return out_count;
}

static RendererF2SpriteSpillEntry *renderer_f2_pick_find_or_add_sprite_entry(const LevelState *level,
                                                                              int source_type,
                                                                              int source_slot,
                                                                              int16_t source_zone)
{
    RendererF2PickSnapshot *snap = &g_renderer_f2_pick_snapshot;
    int16_t source_slot16;

    if (source_slot < -32768) source_slot = -32768;
    if (source_slot > 32767) source_slot = 32767;
    source_slot16 = (int16_t)source_slot;

    for (int i = 0; i < snap->sprite_spill_count; i++) {
        RendererF2SpriteSpillEntry *entry = &snap->sprite_spill_entries[i];
        if (entry->source_type == (uint8_t)source_type &&
            entry->source_slot == source_slot16) {
            return entry;
        }
    }

    if (snap->sprite_spill_count >= RENDERER_F2_MAX_SPRITE_LOG_ENTRIES) {
        snap->sprite_spill_overflow = 1;
        return NULL;
    }

    RendererF2SpriteSpillEntry *entry = &snap->sprite_spill_entries[snap->sprite_spill_count++];
    memset(entry, 0, sizeof(*entry));
    entry->source_type = (uint8_t)source_type;
    entry->source_slot = source_slot16;
    entry->zone_id = source_zone;
    if (level && source_zone >= 0) {
        int adj_count = renderer_collect_adjacent_zones_for_f2(level,
                                                               source_zone,
                                                               entry->adjacent_zones,
                                                               RENDERER_F2_MAX_ZONE_LIST);
        if (adj_count < 0) adj_count = 0;
        if (adj_count > RENDERER_F2_MAX_ZONE_LIST) adj_count = RENDERER_F2_MAX_ZONE_LIST;
        entry->adjacent_count = (uint8_t)adj_count;
    }
    return entry;
}

static void renderer_f2_pick_note_sprite_draw(const LevelState *level,
                                              int source_type,
                                              int source_slot,
                                              int16_t source_zone,
                                              int16_t draw_zone,
                                              int is_spill)
{
    if (!g_pick_capture_active) return;

    RendererF2SpriteSpillEntry *entry =
        renderer_f2_pick_find_or_add_sprite_entry(level, source_type, source_slot, source_zone);
    if (!entry) return;

    if (entry->draw_count < 65535u)
        entry->draw_count++;

    if (is_spill) {
        if (entry->spill_draw_count < 65535u)
            entry->spill_draw_count++;
        renderer_f2_zone_list_add_unique(entry->spill_zones,
                                         RENDERER_F2_MAX_ZONE_LIST,
                                         &entry->spill_zone_count,
                                         draw_zone);
    }
}

static void renderer_f2_pick_note_sprite_local_suppressed(const LevelState *level,
                                                          int source_type,
                                                          int source_slot,
                                                          int16_t source_zone)
{
    if (!g_pick_capture_active) return;

    RendererF2SpriteSpillEntry *entry =
        renderer_f2_pick_find_or_add_sprite_entry(level, source_type, source_slot, source_zone);
    if (!entry) return;

    if (entry->local_suppressed_count < 65535u)
        entry->local_suppressed_count++;
}

static int renderer_resolve_sprite_zone_draw_clip(const RenderSliceContext *ctx,
                                                  GameState *state,
                                                  int16_t zone_id,
                                                  int32_t section_top_world,
                                                  int32_t section_bot_world,
                                                  int ignore_sky_top_clip,
                                                  int16_t *out_left,
                                                  int16_t *out_right,
                                                  int32_t *out_top_world,
                                                  int32_t *out_bot_world,
                                                  int *out_ignore_top)
{
    if (!ctx || !state || !out_left || !out_right || !out_top_world || !out_bot_world || !out_ignore_top)
        return 0;

    LevelState *level = &state->level;
    int zone_slots = level_zone_slot_count(level);
    if (!level->data || !level->zone_adds || zone_slots <= 0 || zone_id < 0 || zone_id >= zone_slots)
        return 0;

    int16_t draw_left = ctx->left_clip;
    int16_t draw_right = ctx->right_clip;
    if (draw_left >= draw_right) return 0;

    int32_t zone_roof = section_top_world;
    int32_t zone_floor = section_bot_world;
    if (zone_roof > zone_floor) {
        int32_t t = zone_roof;
        zone_roof = zone_floor;
        zone_floor = t;
    }

    *out_left = draw_left;
    *out_right = draw_right;
    *out_top_world = zone_roof;
    *out_bot_world = zone_floor;
    *out_ignore_top = ignore_sky_top_clip ? 1 : 0;
    return 1;
}

static void renderer_get_explosion_frame_and_world_size(const GameState *state,
                                                        int explosion_index,
                                                        int *out_frame_num,
                                                        int *out_world_w,
                                                        int *out_world_h)
{
    int frame_num = 0;
    int expl_w = 100;
    int expl_h = 100;
    int scale = 100;
    int ft_count = sprite_frames_table[8].count;
    if (ft_count < 1) ft_count = 1;

    if (state && explosion_index >= 0 && explosion_index < state->num_explosions) {
        scale = (int)state->explosions[explosion_index].size_scale;
        if (scale <= 0) scale = 100;

        frame_num = (int)state->explosions[explosion_index].frame;
        if (frame_num < 0) frame_num = 0;
        if (frame_num >= ft_count) frame_num = ft_count - 1;

        if (bullet_pop_tables[2]) {
            const BulletAnimFrame *pf = &bullet_pop_tables[2][frame_num];
            if (pf->width > 0 && pf->height > 0) {
                expl_w = pf->width;
                expl_h = pf->height;
            }
        }
    }

    expl_w = (expl_w * scale) / 100;
    expl_h = (expl_h * scale) / 100;
    if (expl_w < 1) expl_w = 1;
    if (expl_h < 1) expl_h = 1;

    if (out_frame_num) *out_frame_num = frame_num;
    if (out_world_w) *out_world_w = expl_w;
    if (out_world_h) *out_world_h = expl_h;
}

static void renderer_project_zone_world_clip_y(const RendererState *r,
                                               int32_t zone_top_world,
                                               int32_t zone_bot_world,
                                               int32_t y_off,
                                               int32_t z_fp,
                                               int ignore_top_clip,
                                               int32_t *out_top_y,
                                               int32_t *out_bot_y)
{
    int32_t z = (z_fp > 0) ? z_fp : 1;
    int center_y = r->height / 2;
    if (ignore_top_clip) {
        *out_top_y = INT32_MIN;
    } else {
        *out_top_y = (int)(((int64_t)((zone_top_world - y_off) >> WORLD_Y_FRAC_BITS) *
                            (int64_t)r->proj_y_scale * (int64_t)RENDER_SCALE << ROT_Z_FRAC_BITS) / z) + center_y;
    }
    *out_bot_y = (int)(((int64_t)((zone_bot_world - y_off) >> WORLD_Y_FRAC_BITS) *
                        (int64_t)r->proj_y_scale * (int64_t)RENDER_SCALE << ROT_Z_FRAC_BITS) / z) + center_y;
}

/* -----------------------------------------------------------------------
 * Draw objects in the current zone
 *
 * Translated from ObjDraw3.ChipRam.s ObjDraw (lines 38-137).
 *
 * Iterates all objects in ObjectData, finds those in the current zone,
 * sorts by depth, and draws them back-to-front.
 * ----------------------------------------------------------------------- */
/* level_filter: -1 = draw all (single-level zone), 0 = lower floor only, 1 = upper floor only (multi-floor). */
static void draw_zone_objects_ctx(RenderSliceContext *ctx, GameState *state, int16_t zone_id,
                                  int32_t top_of_room, int32_t bot_of_room,
                                  int level_filter,
                                  int ignore_sky_top_clip)
{
    RendererState *r = &g_renderer;
    LevelState *level = &state->level;
    if (!level->object_data || !level->object_points) return;

    int32_t y_off = r->yoff;
    const int sprite_scale_x_for_estimate = renderer_sprite_scale_x_for_state(r, state);
    const int explosion_sprite_scale_x_for_estimate = renderer_sprite_scale_x_for_state(r, state);
    const int enhancement_enabled = state->cfg_billboard_sprite_rendering_enhancement ? 1 : 0;
    const int16_t billboard_view_right_x = (int16_t)(r->cosval / 2);
    const int16_t billboard_view_right_z = (int16_t)(-r->sinval / 2);

    /* Build one depth-sorted list for sprites and particles so painter order is consistent. */
    enum {
        DRAW_SRC_OBJECT = RENDERER_F2_SPRITE_SOURCE_OBJECT,
        DRAW_SRC_SHOT = RENDERER_F2_SPRITE_SOURCE_SHOT,
        DRAW_SRC_EXPLOSION = RENDERER_F2_SPRITE_SOURCE_EXPLOSION
    };
    typedef struct {
        int src;
        int idx;
        int16_t source_zone;
        int32_t z;
        int16_t clip_left;
        int16_t clip_right;
        int32_t zone_top_world;
        int32_t zone_bot_world;
        uint8_t ignore_top_clip;
    } ObjEntry;
    enum {
        /* Level object table is CID-terminated but physically 256 slots in level data.
         * Keep draw-list capacity aligned with all potential contributors. */
        RENDERER_OBJECT_SLOT_SCAN_CAP = 256
    };
    ObjEntry objs[RENDERER_OBJECT_SLOT_SCAN_CAP +
                  NASTY_SHOT_SLOT_COUNT +
                  PLAYER_SHOT_SLOT_COUNT +
                  MAX_EXPLOSIONS];
    const int max_draw_entries = (int)(sizeof(objs) / sizeof(objs[0]));
    int obj_count = 0;

    int num_pts = level->num_object_points;
    if (num_pts > MAX_OBJ_POINTS) num_pts = MAX_OBJ_POINTS;

    /* Object/shot layout keeps the logical upper/lower section flag at byte 63.
     * For sprites in their own zone, prefer that stable gameplay-side section state
     * over a renderer-estimated vertical overlap test; overlap remains useful for
     * adjacent spill across stair/split boundaries. */
    const int obj_off_in_top = 63;

    /* Iterate by object index; each object has point number at offset 0 (ObjDraw: move.w (a0)+,d0).
     * Use that to look up ObjRotated[pt_num] so keys and other pickups use correct position/z. */
    for (int obj_idx = 0; obj_idx < RENDERER_OBJECT_SLOT_SCAN_CAP && obj_count < max_draw_entries; obj_idx++) {
        const uint8_t *obj = level->object_data + obj_idx * OBJECT_SIZE;
        int16_t pt_num = rd16(obj);
        if (pt_num < 0) break; /* End of object list */

        if ((unsigned)pt_num >= (unsigned)num_pts) continue; /* invalid point number */
        int spill_world_w = 32, spill_world_h = 32;
        renderer_resolve_billboard_world_size_for_spill(obj, 0, &spill_world_w, &spill_world_h);

        /* Only draw objects that are currently in this zone (obj_zone is updated by movement). */
        int16_t obj_zone = rd16(obj + 12);
        int obj_on_upper = (obj[obj_off_in_top] != 0);
        if (obj_zone < 0 || obj_zone != (int16_t)zone_id) continue;

        if (level_filter >= 0) {
            if ((level_filter == 1 && !obj_on_upper) ||
                (level_filter == 0 && obj_on_upper))
                continue;
        }

        int16_t draw_clip_left = 0, draw_clip_right = 0;
        int32_t draw_zone_top = 0, draw_zone_bot = 0;
        int draw_ignore_top = 0;
        if (!renderer_resolve_sprite_zone_draw_clip(ctx, state, zone_id,
                                                    top_of_room, bot_of_room,
                                                    ignore_sky_top_clip,
                                                    &draw_clip_left, &draw_clip_right,
                                                    &draw_zone_top, &draw_zone_bot,
                                                    &draw_ignore_top)) {
            continue;
        }
        ObjRotatedPoint *orp = &r->obj_rotated[pt_num];
        int is_poly_object = ((uint8_t)obj[6] == (uint8_t)OBJ_3D_SPRITE);
        {
            /* Amiga parity:
             * - BitMapObj: near reject at z <= 50
             * - PolygonObj: near reject at z <= 0 (see PolygonObj ble polybehind) */
            int32_t near_clip_z = is_poly_object ? 0 : ROT_Z_FROM_INT(SPRITE_NEAR_CLIP_Z);
            if (orp->z <= near_clip_z) continue;
        }

        /* Skip early when this object cannot affect this column strip. */
        {
            int world_w = spill_world_w;
            int z_for_size = ROT_Z_INT(orp->z);
            if (z_for_size < 1) z_for_size = 1;
            int sprite_w_est = (int)((int32_t)world_w * sprite_scale_x_for_estimate / z_for_size) * SPRITE_SIZE_MULTIPLIER;
            if (sprite_w_est < 1) sprite_w_est = 1;
            int scr_x_est = project_x_to_pixels(orp->x_fine, orp->z);
            int spr_l = scr_x_est - sprite_w_est / 2;
            int spr_r = spr_l + sprite_w_est;
            if (spr_r <= (int)draw_clip_left || spr_l >= (int)draw_clip_right) continue;
        }

        objs[obj_count].src = DRAW_SRC_OBJECT;
        objs[obj_count].idx = obj_idx;
        objs[obj_count].source_zone = obj_zone;
        objs[obj_count].z = is_poly_object
            ? poly_object_front_z_for_sort(obj, orp, state)
            : orp->z;
        objs[obj_count].clip_left = draw_clip_left;
        objs[obj_count].clip_right = draw_clip_right;
        objs[obj_count].zone_top_world = draw_zone_top;
        objs[obj_count].zone_bot_world = draw_zone_bot;
        objs[obj_count].ignore_top_clip = (uint8_t)(draw_ignore_top ? 1 : 0);
        obj_count++;
    }

    /* Add bullets/gibs from both shot pools (same zone, depth-sorted with level objects). */
    int shot_pool_slots[2] = { NASTY_SHOT_SLOT_COUNT, PLAYER_SHOT_SLOT_COUNT };
    for (int pool = 0; pool < 2 && obj_count < max_draw_entries; pool++) {
        const uint8_t *shots = (pool == 0) ? level->nasty_shot_data : level->player_shot_data;
        int slots = shot_pool_slots[pool];
        if (!shots) continue;
        for (int slot = 0; slot < slots && obj_count < max_draw_entries; slot++) {
            const uint8_t *obj = shots + slot * OBJECT_SIZE;
            int16_t shot_zone = rd16(obj + 12);
            if (shot_zone < 0) continue; /* OBJ_ZONE */
            if ((int8_t)obj[16] != OBJ_NBR_BULLET) continue;
            int16_t pt_num = rd16(obj);
            if (pt_num < 0 || (unsigned)pt_num >= (unsigned)num_pts) continue;
            int spill_world_w = 32, spill_world_h = 32;
            renderer_resolve_billboard_world_size_for_spill(obj, 1, &spill_world_w, &spill_world_h);
            if (shot_zone != (int16_t)zone_id) continue;
            int shot_on_upper = (obj[obj_off_in_top] != 0);
            if (level_filter >= 0) {
                if ((level_filter == 1 && !shot_on_upper) ||
                    (level_filter == 0 && shot_on_upper))
                    continue;
            }

            int16_t draw_clip_left = 0, draw_clip_right = 0;
            int32_t draw_zone_top = 0, draw_zone_bot = 0;
            int draw_ignore_top = 0;
            if (!renderer_resolve_sprite_zone_draw_clip(ctx, state, zone_id,
                                                        top_of_room, bot_of_room,
                                                        ignore_sky_top_clip,
                                                        &draw_clip_left, &draw_clip_right,
                                                        &draw_zone_top, &draw_zone_bot,
                                                        &draw_ignore_top)) {
                continue;
            }
            ObjRotatedPoint *orp = &r->obj_rotated[pt_num];
            if (orp->z <= ROT_Z_FROM_INT(SPRITE_NEAR_CLIP_Z)) continue;

            {
                int world_w = spill_world_w;
                int z_for_size = ROT_Z_INT(orp->z);
                if (z_for_size < 1) z_for_size = 1;
                int sprite_w_est = (int)((int32_t)world_w * sprite_scale_x_for_estimate / z_for_size) * SPRITE_SIZE_MULTIPLIER;
                if (sprite_w_est < 1) sprite_w_est = 1;
                int scr_x_est = project_x_to_pixels(orp->x_fine, orp->z);
                int spr_l = scr_x_est - sprite_w_est / 2;
                int spr_r = spr_l + sprite_w_est;
                if (spr_r <= (int)draw_clip_left || spr_l >= (int)draw_clip_right) continue;
            }
            objs[obj_count].src = DRAW_SRC_SHOT;
            objs[obj_count].idx = slot + ((pool == 0) ? 0 : NASTY_SHOT_SLOT_COUNT);
            objs[obj_count].source_zone = shot_zone;
            objs[obj_count].z = orp->z;
            objs[obj_count].clip_left = draw_clip_left;
            objs[obj_count].clip_right = draw_clip_right;
            objs[obj_count].zone_top_world = draw_zone_top;
            objs[obj_count].zone_bot_world = draw_zone_bot;
            objs[obj_count].ignore_top_clip = (uint8_t)(draw_ignore_top ? 1 : 0);
            obj_count++;
        }
    }

    /* Add explosion sprites into the same sorted list so particles and billboards interleave by depth. */
    if (state->num_explosions > 0 && obj_count < max_draw_entries &&
        r->sprite_wad[8] && r->sprite_ptr[8]) {
        int16_t sin_v = r->sinval;
        int16_t cos_v = r->cosval;
        int16_t cam_x = r->xoff;
        int16_t cam_z = r->zoff;

        for (int ei = 0; ei < state->num_explosions && obj_count < max_draw_entries; ei++) {
            int16_t expl_zone = state->explosions[ei].zone;
            if (expl_zone != (int16_t)zone_id) continue;
            int expl_on_upper = (state->explosions[ei].in_top != 0);
            if (state->explosions[ei].start_delay > 0) continue;
            if ((int)state->explosions[ei].frame >= 9) continue;
            if (level_filter >= 0) {
                if ((level_filter == 1 && !expl_on_upper) ||
                    (level_filter == 0 && expl_on_upper))
                    continue;
            }

            int16_t draw_clip_left = 0, draw_clip_right = 0;
            int32_t draw_zone_top = 0, draw_zone_bot = 0;
            int draw_ignore_top = 0;
            if (!renderer_resolve_sprite_zone_draw_clip(ctx, state, zone_id,
                                                        top_of_room, bot_of_room,
                                                        ignore_sky_top_clip,
                                                        &draw_clip_left, &draw_clip_right,
                                                        &draw_zone_top, &draw_zone_bot,
                                                        &draw_ignore_top)) {
                continue;
            }
            int16_t dx = (int16_t)(state->explosions[ei].x - cam_x);
            int16_t dz = (int16_t)(state->explosions[ei].z - cam_z);
            int32_t vz = (int32_t)dx * sin_v + (int32_t)dz * cos_v;
            vz <<= 2;
            int32_t orp_z = vz >> (16 - ROT_Z_FRAC_BITS);
            if (orp_z <= ROT_Z_FROM_INT(SPRITE_NEAR_CLIP_Z)) continue;

            {
                int z_for_size = ROT_Z_INT(orp_z);
                if (z_for_size < 1) z_for_size = 1;
                int expl_w_est = 100;
                renderer_get_explosion_frame_and_world_size(state, ei, NULL, &expl_w_est, NULL);
                int world_w = expl_w_est * EXPLOSION_SIZE_CORRECTION;
                if (world_w < 1) world_w = 1;
                int sprite_w_est = (int)((int32_t)world_w * explosion_sprite_scale_x_for_estimate / z_for_size) * SPRITE_SIZE_MULTIPLIER;
                if (sprite_w_est < 1) sprite_w_est = 1;
                int32_t vx = (int32_t)dx * cos_v - (int32_t)dz * sin_v;
                vx <<= 1;
                int32_t vx_fine = (vx >> 9) + r->xwobble;
                int scr_x_est = project_x_to_pixels(vx_fine, orp_z);
                int spr_l = scr_x_est - sprite_w_est / 2;
                int spr_r = spr_l + sprite_w_est;
                if (spr_r <= (int)draw_clip_left || spr_l >= (int)draw_clip_right) continue;
            }

            objs[obj_count].src = DRAW_SRC_EXPLOSION;
            objs[obj_count].idx = ei;
            objs[obj_count].source_zone = expl_zone;
            objs[obj_count].z = orp_z;
            objs[obj_count].clip_left = draw_clip_left;
            objs[obj_count].clip_right = draw_clip_right;
            objs[obj_count].zone_top_world = draw_zone_top;
            objs[obj_count].zone_bot_world = draw_zone_bot;
            objs[obj_count].ignore_top_clip = (uint8_t)(draw_ignore_top ? 1 : 0);
            obj_count++;
        }
    }

    /* Insertion sort by Z descending (farthest first - painter's algorithm). */
    for (int i = 1; i < obj_count; i++) {
        ObjEntry key = objs[i];
        int j = i - 1;
        while (j >= 0 && objs[j].z < key.z) {
            objs[j + 1] = objs[j];
            j--;
        }
        objs[j + 1] = key;
    }

    /* Draw each object (back-to-front, painter's algorithm)
     * Translated from ObjDraw3.ChipRam.s BitMapObj (lines 527-626).
     *
     * For each object, the data layout (from Defs.i):
     *   Offset 0:  object type (word)
     *   Offset 2:  brightness (objVectBright, word)
     *   Offset 4:  Y position (word, in ObjectPoints offset 2)
     *   Offset 6:  world width (byte), offset 7: world height (byte)
     *   Offset 8:  vector number (objVectNumber, word)
     *   Offset 10: frame number (objVectFrameNumber, word)
     *   Offset 14: source columns (byte), offset 15: source rows (byte)
     *   Offset 26: GraphicRoom (word)
     */
    const int expl_vect = 8;
    const SpriteFrame *expl_ft = sprite_frames_table[expl_vect].frames;
    const int expl_ft_count = sprite_frames_table[expl_vect].count;
    const int explosion_sprite_scale_x = renderer_sprite_scale_x_for_state(r, state);
    for (int oi = 0; oi < obj_count; oi++) {
        const ObjEntry *entry = &objs[oi];
        int entry_src = entry->src;
        int16_t draw_clip_left = entry->clip_left;
        int16_t draw_clip_right = entry->clip_right;
        if (draw_clip_left >= draw_clip_right) continue;

        if (entry_src == DRAW_SRC_EXPLOSION) {
            int ei = entry->idx;
            int16_t sin_v = r->sinval;
            int16_t cos_v = r->cosval;
            int16_t cam_x = r->xoff;
            int16_t cam_z = r->zoff;
            int frame_num = 0;
            int expl_w = 100;
            int expl_h = 100;
            renderer_get_explosion_frame_and_world_size(state, ei, &frame_num, &expl_w, &expl_h);
            if (frame_num >= expl_ft_count) frame_num = expl_ft_count - 1;
            if (frame_num < 0) continue;

            int16_t dx = (int16_t)(state->explosions[ei].x - cam_x);
            int16_t dz = (int16_t)(state->explosions[ei].z - cam_z);
            int32_t vx = (int32_t)dx * cos_v - (int32_t)dz * sin_v;
            vx <<= 1;
            int32_t vz = (int32_t)dx * sin_v + (int32_t)dz * cos_v;
            vz <<= 2;
            int32_t vx_fine = (vx >> 9) + r->xwobble;
            int32_t orp_z = vz >> (16 - ROT_Z_FRAC_BITS);
            if (orp_z <= ROT_Z_FROM_INT(SPRITE_NEAR_CLIP_Z)) continue;

            int scr_x = project_x_to_pixels(vx_fine, orp_z);
            int32_t rel_y_8 = (state->explosions[ei].y_floor - y_off) >> WORLD_Y_FRAC_BITS;
            int center_y = r->height / 2;
            int scr_y = (int)(((int64_t)rel_y_8 * (int64_t)r->proj_y_scale * (int64_t)RENDER_SCALE << ROT_Z_FRAC_BITS) / (int64_t)orp_z) + center_y;
            int z_for_size = ROT_Z_INT(orp_z);
            if (z_for_size < 1) z_for_size = 1;
            int expl_h_port = explosion_world_h_to_port(r, expl_h, explosion_sprite_scale_x);
            int sprite_w = (int)((int32_t)expl_w * explosion_sprite_scale_x / z_for_size) * SPRITE_SIZE_MULTIPLIER;
            int sprite_h = (int)((int64_t)expl_h_port * (int64_t)r->proj_y_scale * (int64_t)RENDER_SCALE / z_for_size) * SPRITE_SIZE_MULTIPLIER;
            sprite_w *= EXPLOSION_SIZE_CORRECTION;
            sprite_h *= EXPLOSION_SIZE_CORRECTION;
            if (sprite_w < 1) sprite_w = 1;
            if (sprite_h < 1) sprite_h = 1;

            uint32_t ptr_off = 0;
            uint16_t down_strip = 0;
            if (expl_ft && frame_num < expl_ft_count) {
                ptr_off = expl_ft[frame_num].ptr_off;
                down_strip = expl_ft[frame_num].down_strip;
            }

            int32_t clip_top_y = 0, clip_bot_y = 0;
            renderer_project_zone_world_clip_y(r,
                                               entry->zone_top_world,
                                               entry->zone_bot_world,
                                               y_off,
                                               orp_z,
                                               (int)entry->ignore_top_clip,
                                               &clip_top_y,
                                               &clip_bot_y);
            if (clip_top_y >= clip_bot_y) continue;
            int bright = (ROT_Z_INT(orp_z) >> 7);
            const uint8_t *obj_pal = r->sprite_pal_data[expl_vect];
            size_t obj_pal_size = r->sprite_pal_size[expl_vect];
            int32_t expl_world_w = expl_w * EXPLOSION_SIZE_CORRECTION;
            int16_t spill_zones[RENDERER_F2_MAX_ZONE_LIST];
            int spill_zone_count = 0;
            int sprite_left = scr_x - sprite_w / 2;
            int sprite_right = sprite_left + sprite_w;
            if (expl_world_w < 1) expl_world_w = 1;
            {
                int32_t orp_z_i16 = ROT_Z_INT(orp_z);
                if (orp_z_i16 > 32767) orp_z_i16 = 32767;
                ctx->pick_player_id = 0;

                if (enhancement_enabled) {
                    spill_zone_count = renderer_collect_spill_zones_recursive(ctx,
                                                                              state,
                                                                              entry->source_zone,
                                                                              (int32_t)state->explosions[ei].x,
                                                                              (int32_t)state->explosions[ei].z,
                                                                              expl_world_w,
                                                                              billboard_view_right_x,
                                                                              billboard_view_right_z,
                                                                              sprite_left,
                                                                              sprite_right,
                                                                              sprite_w,
                                                                              spill_zones,
                                                                              RENDERER_F2_MAX_ZONE_LIST);
                }

                renderer_f2_pick_note_sprite_draw(level,
                                                  DRAW_SRC_EXPLOSION,
                                                  ei,
                                                  entry->source_zone,
                                                  entry->source_zone,
                                                  0);

                renderer_draw_sprite_ctx_timed(ctx,
                                               (int16_t)scr_x, (int16_t)scr_y,
                                               (int16_t)sprite_w, (int16_t)sprite_h,
                                               (int16_t)orp_z_i16,
                                               r->sprite_wad[expl_vect], r->sprite_wad_size[expl_vect],
                                               r->sprite_ptr[expl_vect], r->sprite_ptr_size[expl_vect],
                                               obj_pal, obj_pal_size,
                                               ptr_off, down_strip,
                                               32, 32,
                                               (int16_t)bright, expl_vect,
                                               draw_clip_left, draw_clip_right,
                                               entry->zone_top_world, entry->zone_bot_world,
                                               clip_top_y, clip_bot_y,
                                               0,
                                               enhancement_enabled ? level : NULL,
                                               enhancement_enabled ? entry->source_zone : -1,
                                               enhancement_enabled ? (int32_t)state->explosions[ei].x : 0,
                                               enhancement_enabled ? (int32_t)state->explosions[ei].z : 0,
                                               enhancement_enabled ? expl_world_w : 0,
                                               enhancement_enabled ? billboard_view_right_x : 0,
                                               enhancement_enabled ? billboard_view_right_z : 0);

                if (enhancement_enabled && spill_zone_count > 0) {
                    for (int si = 0; si < spill_zone_count; si++) {
                        int16_t spill_zone = spill_zones[si];
                        int32_t spill_zone_top;
                        int32_t spill_zone_bot;
                        int16_t spill_clip_left;
                        int16_t spill_clip_right;
                        int32_t spill_clip_top_world;
                        int32_t spill_clip_bot_world;
                        int spill_ignore_top;
                        int32_t spill_clip_top_y;
                        int32_t spill_clip_bot_y;

                        if (spill_zone < 0 || spill_zone == entry->source_zone) continue;
                        if (!renderer_resolve_zone_section_world_bounds(level,
                                                                        spill_zone,
                                                                        level_filter,
                                                                        &spill_zone_top,
                                                                        &spill_zone_bot)) {
                            continue;
                        }
                        if (!renderer_resolve_sprite_zone_draw_clip(ctx,
                                                                    state,
                                                                    spill_zone,
                                                                    spill_zone_top,
                                                                    spill_zone_bot,
                                                                    0,
                                                                    &spill_clip_left,
                                                                    &spill_clip_right,
                                                                    &spill_clip_top_world,
                                                                    &spill_clip_bot_world,
                                                                    &spill_ignore_top)) {
                            continue;
                        }
                        (void)spill_ignore_top;
                        if (spill_clip_left >= spill_clip_right) continue;

                        renderer_project_zone_world_clip_y(r,
                                                           spill_clip_top_world,
                                                           spill_clip_bot_world,
                                                           y_off,
                                                           orp_z,
                                                           0,
                                                           &spill_clip_top_y,
                                                           &spill_clip_bot_y);
                        if (spill_clip_top_y >= spill_clip_bot_y) continue;

                        renderer_f2_pick_note_sprite_draw(level,
                                                          DRAW_SRC_EXPLOSION,
                                                          ei,
                                                          entry->source_zone,
                                                          spill_zone,
                                                          1);

                        renderer_draw_sprite_ctx_timed(ctx,
                                                       (int16_t)scr_x, (int16_t)scr_y,
                                                       (int16_t)sprite_w, (int16_t)sprite_h,
                                                       (int16_t)orp_z_i16,
                                                       r->sprite_wad[expl_vect], r->sprite_wad_size[expl_vect],
                                                       r->sprite_ptr[expl_vect], r->sprite_ptr_size[expl_vect],
                                                       obj_pal, obj_pal_size,
                                                       ptr_off, down_strip,
                                                       32, 32,
                                                       (int16_t)bright, expl_vect,
                                                       spill_clip_left, spill_clip_right,
                                                       spill_clip_top_world, spill_clip_bot_world,
                                                       spill_clip_top_y, spill_clip_bot_y,
                                                       1,
                                                       NULL,
                                                       -1,
                                                       0,
                                                       0,
                                                       0,
                                                       0,
                                                       0);
                    }
                }
            }
            continue;
        }

        int obj_idx = entry->idx;
        const uint8_t *obj = NULL;
        if (entry_src == DRAW_SRC_OBJECT) {
            obj = level->object_data + obj_idx * OBJECT_SIZE;
        } else {
            if (obj_idx < NASTY_SHOT_SLOT_COUNT) {
                if (!level->nasty_shot_data) continue;
                obj = level->nasty_shot_data + obj_idx * OBJECT_SIZE;
            } else {
                if (!level->player_shot_data) continue;
                obj = level->player_shot_data + (obj_idx - NASTY_SHOT_SLOT_COUNT) * OBJECT_SIZE;
            }
        }
        /* ObjDraw3: cmp.b #$ff,6(a0); bne BitMapObj; bsr PolygonObj.
         * When obj[6]==OBJ_3D_SPRITE the object is a 3D polygon mesh drawn via
         * the PolygonObj pipeline (renderer_3dobj.c). */
        if ((uint8_t)obj[6] == (uint8_t)OBJ_3D_SPRITE) {
            ObjRotatedPoint *orp3d = &r->obj_rotated[rd16(obj)];
            draw_3d_vector_object(obj, orp3d, state,
                                  (int)draw_clip_left, (int)draw_clip_right - 1,
                                  (int)ctx->top_clip, (int)ctx->bot_clip);
            continue;
        }

        int16_t pt_num = rd16(obj);
        if ((unsigned)pt_num >= (unsigned)num_pts) continue;
        ObjRotatedPoint *orp = &r->obj_rotated[pt_num];

        /* Use actual view Z for size so sprites scale at all distances. Guard only vs div-by-zero. */
        int32_t z_for_size = ROT_Z_INT(orp->z);
        if (z_for_size < 1) z_for_size = 1;

        /* Project Y boundaries from room top/bottom (same PROJ_Y_SCALE as walls).
         * orp->z is 24.8 fixed-point; shift numerator to compensate. */
        int32_t clip_top_y = 0, clip_bot_y = 0;
        renderer_project_zone_world_clip_y(r,
                                           entry->zone_top_world,
                                           entry->zone_bot_world,
                                           y_off,
                                           orp->z,
                                           (int)entry->ignore_top_clip,
                                           &clip_top_y,
                                           &clip_bot_y);
        if (clip_top_y >= clip_bot_y) continue;

        /* Project to screen X (PROJ_X_SCALE/2 = horizontal focal length). */
        int32_t obj_vx_fine = orp->x_fine;
        int scr_x = project_x_to_pixels(obj_vx_fine, orp->z);

        /* Get brightness + distance attenuation (uses integer Z) */
        int16_t obj_bright = rd16(obj + 2);
        int bright = (ROT_Z_INT(orp->z) >> 7) + obj_bright;
        if (bright < 0) bright = 0;

        int8_t obj_number = (int8_t)obj[16];
        int16_t vect_num, frame_num;
        int drawing_dead = 0;
        if (obj_number == OBJ_NBR_DEAD &&
            renderer_dead_object_is_runtime_corpse(obj)) {
            /* Death animation: original type in type_data[1]; death frame number in OBJ_DEADH (raw+8) */
            int8_t original_type = (int8_t)obj[19];
            if (original_type < 0 || original_type > 20) continue;
            drawing_dead = 1;
            obj_number = original_type;  /* for fallback and barrel check below */
            vect_num = -1;               /* force fallback to get vect from type */
            frame_num = rd16(obj + 8);   /* OBJ_DEADH = death animation frame */
        } else {
            vect_num = rd16(obj + 8);
            frame_num = rd16(obj + 10);
        }

        /* Resolve sprite graphic (vect_num). Fallback by obj_number when vect missing/invalid
         * or when level left vect 0 for a non-alien (so all enemies don't draw as alien). */
        int use_fallback = (vect_num < 0 || vect_num >= MAX_SPRITE_TYPES ||
                            !r->sprite_wad[vect_num] || !r->sprite_ptr[vect_num]);
        if (!use_fallback && vect_num == 0 && obj_number != OBJ_NBR_ALIEN &&
            obj_number >= OBJ_NBR_ROBOT && obj_number <= OBJ_NBR_FLAME_MARINE)
            use_fallback = 1;
        if (use_fallback) {
            vect_num = renderer_fallback_vect_for_obj_number(obj_number);
            if (!drawing_dead) frame_num = 0;
        }
        if (vect_num < 0 || vect_num >= MAX_SPRITE_TYPES) continue;

        int world_w = 32;
        int world_h = 32;
        renderer_resolve_billboard_world_size_for_spill(obj,
                                                        (entry_src == DRAW_SRC_SHOT) ? 1 : 0,
                                                        &world_w, &world_h);

        /* Screen size: width from Amiga (byte*128/z)*RENDER_SCALE; height uses proj_y_scale so billboard Y matches floor projection scale. */
        int explosion_billboard = 0;
        if (vect_num == 8) {
            if (obj_number == OBJ_NBR_BARREL) {
                explosion_billboard = 1;
            } else if (entry_src == DRAW_SRC_SHOT && (int8_t)obj[30] != 0) {
                /* Popping shot using explosion sheet (e.g. RockPop/FlamePop). */
                explosion_billboard = 1;
            }
        }

        int sprite_scale_x = renderer_sprite_scale_x();
        int world_h_for_proj = world_h;
        if (explosion_billboard) {
            sprite_scale_x = explosion_sprite_scale_x;
            world_h_for_proj = explosion_world_h_to_port(&g_renderer, world_h, explosion_sprite_scale_x);
        }
        int sprite_w = (int)((int32_t)world_w * sprite_scale_x / z_for_size) * SPRITE_SIZE_MULTIPLIER;
        int sprite_h = (int)((int64_t)world_h_for_proj * (int64_t)g_renderer.proj_y_scale * (int64_t)RENDER_SCALE / z_for_size) * SPRITE_SIZE_MULTIPLIER;
        if (explosion_billboard) {
            sprite_w *= EXPLOSION_SIZE_CORRECTION;
            sprite_h *= EXPLOSION_SIZE_CORRECTION;
        }
        if (sprite_w < 1) sprite_w = 1;
        if (sprite_h < 1) sprite_h = 1;

        /* Source dimensions: columns and rows from object data offsets 14, 15. */
        int src_cols = (int)obj[14];
        int src_rows = (int)obj[15];
        if (src_cols < 1) src_cols = 32;
        if (src_rows < 1) src_rows = 32;

        /* Some enemy classes use non-32 source decode dimensions in the
         * original code path (writes to object bytes 14/15). Enforce those
         * here as a safety net so first-frame or stale-object states don't
         * decode wide sheets as 32x32 strips. */
        switch ((ObjNumber)obj_number) {
        case OBJ_NBR_WORM:
            src_cols = 45; src_rows = 50;
            break;
        case OBJ_NBR_HUGE_RED_THING:
            src_cols = 63; src_rows = 63;
            break;
        case OBJ_NBR_SMALL_RED_THING:
            src_cols = 64; src_rows = 64;
            break;
        case OBJ_NBR_EYEBALL:
            src_cols = 15; src_rows = 31;
            break;
        default:
            break;
        }

        /* BIGSCARYALIEN (vect 11) uses 128-column frames in the PTR table.
         * With the renderer's eff_cols = src_cols*2 mapping, src_cols/src_rows
         * below 64 decode only a quarter/half of the sprite. */
        if (vect_num == 11) {
            if (src_cols < 64) src_cols = 64;
            if (src_rows < 64) src_rows = 64;
        }


        /* Look up frame info from FRAMES table (Amiga: 2(a0) indexes frame; frame gives DOWN_STRIP for strip offset). */
        uint32_t ptr_off = 0;
        uint16_t down_strip = 0;
        const SpriteFrame *ft = sprite_frames_table[vect_num].frames;
        int ft_count = sprite_frames_table[vect_num].count;
        if (ft && frame_num >= 0 && frame_num < ft_count) {
            ptr_off = ft[frame_num].ptr_off;
            down_strip = ft[frame_num].down_strip;
        }

        /* ObjDraw3 projects object Y directly from obj[4] (no floor reconstruction). */
        int16_t obj_y = rd16(obj + 4);
        int32_t rel_y = (((int32_t)obj_y) << 7) - y_off;
        int32_t rel_y_8 = rel_y >> WORLD_Y_FRAC_BITS;
        int center_y = g_renderer.height / 2;
        int scr_y = (int)(((int64_t)rel_y_8 * (int64_t)g_renderer.proj_y_scale * (int64_t)RENDER_SCALE << ROT_Z_FRAC_BITS) / (int64_t)orp->z) + center_y;

        const uint8_t *obj_pal = r->sprite_pal_data[vect_num];
        size_t obj_pal_size = r->sprite_pal_size[vect_num];
        {
            uint8_t pick_player_id = 0;
            if (entry_src == DRAW_SRC_OBJECT) {
                if (obj_number == OBJ_NBR_PLR1) pick_player_id = 1;
                else if (obj_number == OBJ_NBR_PLR2) pick_player_id = 2;
            }
            ctx->pick_player_id = pick_player_id;
        }

        {
            int32_t orp_z_int = ROT_Z_INT(orp->z);
            int source_type = (entry_src == DRAW_SRC_OBJECT) ? DRAW_SRC_OBJECT : DRAW_SRC_SHOT;
            int32_t billboard_world_x = 0;
            int32_t billboard_world_z = 0;
            int sprite_left = scr_x - sprite_w / 2;
            int sprite_right = sprite_left + sprite_w;
            int16_t spill_zones[RENDERER_F2_MAX_ZONE_LIST];
            int spill_zone_count = 0;

            if (orp_z_int > 32767) orp_z_int = 32767;

            if (level->object_points && pt_num >= 0 && pt_num < num_pts) {
                const uint8_t *pt_ptr = level->object_points + (size_t)(uint16_t)pt_num * 8u;
                billboard_world_x = (int32_t)rd16(pt_ptr + 0);
                billboard_world_z = (int32_t)rd16(pt_ptr + 4);
            }

            if (enhancement_enabled) {
                spill_zone_count = renderer_collect_spill_zones_recursive(ctx,
                                                                          state,
                                                                          entry->source_zone,
                                                                          billboard_world_x,
                                                                          billboard_world_z,
                                                                          world_w,
                                                                          billboard_view_right_x,
                                                                          billboard_view_right_z,
                                                                          sprite_left,
                                                                          sprite_right,
                                                                          sprite_w,
                                                                          spill_zones,
                                                                          RENDERER_F2_MAX_ZONE_LIST);
            }

            renderer_f2_pick_note_sprite_draw(level,
                                              source_type,
                                              obj_idx,
                                              entry->source_zone,
                                              entry->source_zone,
                                              0);

            renderer_draw_sprite_ctx_timed(ctx,
                                           (int16_t)scr_x, (int16_t)scr_y,
                                           (int16_t)sprite_w, (int16_t)sprite_h,
                                           (int16_t)orp_z_int,
                                           r->sprite_wad[vect_num], r->sprite_wad_size[vect_num],
                                           r->sprite_ptr[vect_num], r->sprite_ptr_size[vect_num],
                                           obj_pal, obj_pal_size,
                                           ptr_off, down_strip,
                                           src_cols, src_rows,
                                           (int16_t)bright, vect_num,
                                           draw_clip_left, draw_clip_right,
                                           entry->zone_top_world, entry->zone_bot_world,
                                           clip_top_y, clip_bot_y,
                                           0,
                                           enhancement_enabled ? level : NULL,
                                           enhancement_enabled ? entry->source_zone : -1,
                                           enhancement_enabled ? billboard_world_x : 0,
                                           enhancement_enabled ? billboard_world_z : 0,
                                           enhancement_enabled ? world_w : 0,
                                           enhancement_enabled ? billboard_view_right_x : 0,
                                           enhancement_enabled ? billboard_view_right_z : 0);

            if (enhancement_enabled && spill_zone_count > 0) {
                for (int si = 0; si < spill_zone_count; si++) {
                    int16_t spill_zone = spill_zones[si];
                    int32_t spill_zone_top;
                    int32_t spill_zone_bot;
                    int16_t spill_clip_left;
                    int16_t spill_clip_right;
                    int32_t spill_clip_top_world;
                    int32_t spill_clip_bot_world;
                    int spill_ignore_top;
                    int32_t spill_clip_top_y;
                    int32_t spill_clip_bot_y;

                    if (spill_zone < 0 || spill_zone == entry->source_zone) continue;
                    if (!renderer_resolve_zone_section_world_bounds(level,
                                                                    spill_zone,
                                                                    level_filter,
                                                                    &spill_zone_top,
                                                                    &spill_zone_bot)) {
                        continue;
                    }
                    if (!renderer_resolve_sprite_zone_draw_clip(ctx,
                                                                state,
                                                                spill_zone,
                                                                spill_zone_top,
                                                                spill_zone_bot,
                                                                0,
                                                                &spill_clip_left,
                                                                &spill_clip_right,
                                                                &spill_clip_top_world,
                                                                &spill_clip_bot_world,
                                                                &spill_ignore_top)) {
                        continue;
                    }
                    (void)spill_ignore_top;
                    if (spill_clip_left >= spill_clip_right) continue;

                    renderer_project_zone_world_clip_y(r,
                                                       spill_clip_top_world,
                                                       spill_clip_bot_world,
                                                       y_off,
                                                       orp->z,
                                                       0,
                                                       &spill_clip_top_y,
                                                       &spill_clip_bot_y);
                    if (spill_clip_top_y >= spill_clip_bot_y) continue;

                    renderer_f2_pick_note_sprite_draw(level,
                                                      source_type,
                                                      obj_idx,
                                                      entry->source_zone,
                                                      spill_zone,
                                                      1);

                    renderer_draw_sprite_ctx_timed(ctx,
                                                   (int16_t)scr_x, (int16_t)scr_y,
                                                   (int16_t)sprite_w, (int16_t)sprite_h,
                                                   (int16_t)orp_z_int,
                                                   r->sprite_wad[vect_num], r->sprite_wad_size[vect_num],
                                                   r->sprite_ptr[vect_num], r->sprite_ptr_size[vect_num],
                                                   obj_pal, obj_pal_size,
                                                   ptr_off, down_strip,
                                                   src_cols, src_rows,
                                                   (int16_t)bright, vect_num,
                                                   spill_clip_left, spill_clip_right,
                                                   spill_clip_top_world, spill_clip_bot_world,
                                                   spill_clip_top_y, spill_clip_bot_y,
                                                   1,
                                                   NULL,
                                                   -1,
                                                   0,
                                                   0,
                                                   0,
                                                   0,
                                                   0);
                }
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Draw a single zone
 *
 * Translated from AB3DI.s DoThisRoom (lines 3814-3925) and polyloop.
 *
 * The zone graphics data (from LEVELGRAPHICS) is a stream of entries:
 *   - First word: zone number (consumed before polyloop)
 *   - Then type words followed by type-specific data
 *
 * Type dispatch (from polyloop, lines 3828-3852):
 *   0  = wall        -> itsawalldraw (26 bytes of data after type)
 *   1  = floor       -> itsafloordraw (variable: ypos + sides + points + extra)
 *   2  = roof        -> itsafloordraw (same format)
 *   3  = clip setter -> no data (clipping done separately via ListOfGraphRooms)
 *   4  = object      -> ObjDraw (1 word: draw mode)
 *   5  = arc         -> CurveDraw (variable)
 *   6  = light beam  -> LightDraw (variable)
 *   7  = water       -> itsafloordraw (same as floor)
 *   8  = chunky floor-> itsafloordraw
 *   9  = bumpy floor -> itsafloordraw
 *  12  = backdrop    -> putinbackdrop (no extra data)
 *  13  = see-wall    -> itsawalldraw (same 26 bytes)
 *  <0  = end of list
 *
 * Wall entry (26 bytes after type word):
 *   +0: point1 (word)       - index into Points/Rotated array
 *   +2: point2 (word)       - index into Points/Rotated array
 *   +4: strip_start (word)  - leftend texture column
 *   +6: strip_end (word)    - rightend texture column
 *   +8: texture_tile (word) - tile offset (*16)
 *  +10: totalyoff (word)    - vertical texture offset
 *  +12: texture_id (word)   - index into walltiles[] array
 *  +14: VALAND (byte)       - vertical texture AND mask
 *  +15: VALSHIFT (byte)     - vertical texture shift
 *  +16: HORAND (word)       - horizontal texture AND mask
 *  +18: topofwall (long)    - wall top height (world coords)
 *  +22: botofwall (long)    - wall bottom height (world coords)
 *  +26: wallbrightoff (word)- brightness offset
 *  Total: 28 bytes after type word
 *
 * Floor/Roof entry (variable bytes after type word):
 *   +0: ypos (word)        - floor/ceiling height (>>6 for world)
 *   +2: num_sides-1 (word) - polygon sides minus 1
 *   +4: point indices (2 bytes * (sides))
 *   +4+2*sides: 4 bytes padding + 6 bytes extra data = 10 bytes
 *  Total: 2+2+2*sides+10 bytes after type word
 *
 * Object entry (2 bytes after type word):
 *   +0: draw_mode (word) - 0=before water, 1=after water, 2=full room
 * ----------------------------------------------------------------------- */
#define MAX_DOOR_ENTRIES 256
#define MAX_LIFT_ENTRIES 256
#define LIFT_ENTRY_SIZE  20
#define DOOR_ENTRY_SIZE  22

static int zone_has_door(const uint8_t *door_data, int16_t zone_id)
{
    if (!door_data) return 0;
    for (int i = 0; i < MAX_DOOR_ENTRIES; i++) {
        int16_t d_zone = rd16(door_data);
        if (d_zone < 0) return 0;  /* End of list */
        if (d_zone == zone_id) return 1;
        door_data += 22;
    }
    return 0;  /* Safety: avoid reading past buffer if format is wrong */
}

/* Zones are "tagged" as lift zones by appearing in the lift table (first word = zone_id). */
static int zone_has_lift(const uint8_t *lift_data, int16_t zone_id)
{
    if (!lift_data) return 0;
    for (int i = 0; i < MAX_LIFT_ENTRIES; i++) {
        int16_t l_zone = rd16(lift_data);
        if (l_zone < 0) return 0;
        if (l_zone == zone_id) return 1;
        lift_data += LIFT_ENTRY_SIZE;
    }
    return 0;
}

static int renderer_edge_match(int16_t a1, int16_t a2, int16_t b1, int16_t b2)
{
    return ((a1 == b1 && a2 == b2) || (a1 == b2 && a2 == b1)) ? 1 : 0;
}

static int renderer_level4_adjacent_door_wall_match(int16_t p1, int16_t p2, int16_t tex_id)
{
    /* Level 4 zone 124 doorway cluster:
     * points 266..269 form the door ring and can appear in adjacent streams
     * with different wall textures depending on view/zone side. */
    if (tex_id == 1 || tex_id == 6 || tex_id == 8) {
        int16_t d = (int16_t)(p1 - p2);
        if (d < 0) d = (int16_t)(-d);
        if (p1 >= 266 && p1 <= 269 &&
            p2 >= 266 && p2 <= 269 &&
            (d == 1 || d == 3)) {
            return 1;
        }
    }

    if (tex_id == 1) {
        if (renderer_edge_match(p1, p2, 25, 23)) return 1;
        if (renderer_edge_match(p1, p2, 24, 26)) return 1;
        if (renderer_edge_match(p1, p2, 33, 32)) return 1;
        if (renderer_edge_match(p1, p2, 269, 268)) return 1; /* zone 124 doorway */
        if (renderer_edge_match(p1, p2, 267, 266)) return 1; /* zone 124 doorway */
    }
    if (tex_id == 6) {
        if (renderer_edge_match(p1, p2, 23, 21)) return 1;
        if (renderer_edge_match(p1, p2, 22, 24)) return 1;
        if (renderer_edge_match(p1, p2, 34, 33)) return 1;
    }
    if (tex_id == 8) {
        if (renderer_edge_match(p1, p2, 24, 23)) return 1;
    }
    return 0;
}

static int renderer_wall_gfx_off_matches_door_zone(const LevelState *level,
                                                    uint32_t gfx_off,
                                                    int16_t door_zone_id)
{
    if (!level || !level->door_wall_list || !level->door_wall_list_offsets || !level->door_data)
        return 0;
    if (level->num_doors <= 0)
        return 0;

    const uint8_t *lst = level->door_wall_list;
    for (int di = 0; di < level->num_doors; di++) {
        int16_t zone_id = rd16(level->door_data + (size_t)di * 22u + 0u);
        if (zone_id != door_zone_id)
            continue;

        uint32_t start = level->door_wall_list_offsets[di];
        uint32_t end = level->door_wall_list_offsets[di + 1];
        for (uint32_t j = start; j < end; j++) {
            const uint8_t *ent = lst + (size_t)j * 10u;
            uint32_t ent_gfx = (uint32_t)rd32(ent + 2);
            if (ent_gfx == gfx_off)
                return 1;
        }
    }
    return 0;
}

static void renderer_reset_level_sky_cache_internal(void)
{
    free(g_level_sky_cache_polys);
    g_level_sky_cache_polys = NULL;
    g_level_sky_cache_poly_count = 0;
    g_level_sky_cache_poly_cap = 0;

    free(g_level_sky_cache_buckets);
    g_level_sky_cache_buckets = NULL;
    g_level_sky_cache_zone_slots = 0;
}

void renderer_reset_level_sky_cache(void)
{
    renderer_reset_level_sky_cache_internal();
}

static int renderer_level_sky_cache_ensure_poly_cap(uint32_t want)
{
    if (want <= g_level_sky_cache_poly_cap) return 1;

    uint32_t new_cap = (g_level_sky_cache_poly_cap > 0) ? g_level_sky_cache_poly_cap : 256u;
    while (new_cap < want) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = want;
            break;
        }
        new_cap *= 2u;
    }

    RendererSkyCachePoly *np =
        (RendererSkyCachePoly *)realloc(g_level_sky_cache_polys, (size_t)new_cap * sizeof(RendererSkyCachePoly));
    if (!np) return 0;
    g_level_sky_cache_polys = np;
    g_level_sky_cache_poly_cap = new_cap;
    return 1;
}

static int renderer_level_sky_cache_push_poly(const int16_t *pt_indices, int sides)
{
    if (!pt_indices) return 0;
    if (sides < 3 || sides > 100) return 0;
    if (!renderer_level_sky_cache_ensure_poly_cap(g_level_sky_cache_poly_count + 1u)) return 0;

    RendererSkyCachePoly *dst = &g_level_sky_cache_polys[g_level_sky_cache_poly_count++];
    dst->sides = (int16_t)sides;
    memcpy(dst->pt_indices, pt_indices, (size_t)sides * sizeof(int16_t));
    return 1;
}

/* Bytes of payload after the type word (for forward-scanning the zone graphics stream). */
static size_t zone_gfx_entry_data_skip(int16_t entry_type, const uint8_t *data)
{
    switch (entry_type) {
    case 0:
    case 13: return 28;
    case 3:
    case 12: return 0;
    case 4: return 2;
    case 5: return 28;
    case 6: return 4;
    case 1:
    case 2:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    {
        int16_t num_sides_m1 = rd16(data + 2);
        int sides = (int)num_sides_m1 + 1;
        if (sides < 0) sides = 0;
        if (sides > 100) sides = 100;
        /* Payload layout after type:
         *   ypos(2) + sides_m1(2) + point_indices(2*sides) + extra(8) */
        return (size_t)(4 + 2 * sides + 8);
    }
    default:
        return 0;
    }
}

static int renderer_build_viewer_split_floor_occluder(const GameState *state,
                                                      int16_t viewer_zone,
                                                      int16_t col_start,
                                                      int16_t col_end,
                                                      int16_t *out_top)
{
    const LevelState *level;
    const uint8_t *zone_data;
    const uint8_t *zgraph;
    const uint8_t *gfx_data;
    const RendererState *r = &g_renderer;
    int32_t zone_off;
    int32_t gfx_off;
    int32_t zone_floor;
    int32_t zone_roof;
    int32_t zone_water;
    int32_t y_off;
    int w;
    int h;
    int half_h;
    int active = 0;
    int16_t *left_edge = NULL;
    int16_t *right_edge = NULL;

    if (!state || !out_top) return 0;
    level = &state->level;
    if (!level->data || !level->zone_adds || !level->zone_graph_adds || !level->graphics) return 0;

    w = g_renderer.width;
    h = g_renderer.height;
    if (w <= 0 || h <= 0) return 0;
    for (int x = 0; x < w; x++) out_top[x] = (int16_t)h;

    if (viewer_zone < 0 || viewer_zone >= level_zone_slot_count(level)) return 0;
    if (level->num_zone_graph_entries > 0 && viewer_zone >= level->num_zone_graph_entries) return 0;

    zone_off = rd32(level->zone_adds + viewer_zone * 4);
    if (zone_off < 0) return 0;
    if (level->data_byte_count > 0 && ((size_t)zone_off + 48u > level->data_byte_count)) return 0;
    zone_data = level->data + zone_off;

    zone_floor = rd32(zone_data + 2);
    zone_roof = rd32(zone_data + 6);
    zone_water = rd32(zone_data + 18);
    if (!(zone_water > zone_roof && zone_water < zone_floor)) return 0;

    zgraph = level->zone_graph_adds + viewer_zone * 8;
    gfx_off = rd32(zgraph);
    if (gfx_off <= 0) return 0;
    if (level->graphics_byte_count > 0 && ((size_t)gfx_off + 2u > level->graphics_byte_count)) return 0;
    gfx_data = level->graphics + (size_t)gfx_off;

    y_off = r->yoff;
    half_h = h / 2;
    if (col_start < 0) col_start = 0;
    if (col_end > w) col_end = (int16_t)w;
    if (col_start >= col_end) return 0;

    left_edge = (int16_t *)malloc((size_t)h * sizeof(*left_edge));
    right_edge = (int16_t *)malloc((size_t)h * sizeof(*right_edge));
    if (!left_edge || !right_edge) {
        free(left_edge);
        free(right_edge);
        return 0;
    }

    {
        const uint8_t *ptr = gfx_data + 2;
        int max_iter = 500;

        while (max_iter-- > 0) {
            int16_t entry_type = rd16(ptr);
            ptr += 2;
            if (entry_type < 0) break;

            if (entry_type != 1) {
                size_t skip = zone_gfx_entry_data_skip(entry_type, ptr);
                if (skip == 0) {
                    if (entry_type != 3 && entry_type != 12) break;
                } else {
                    ptr += skip;
                }
                continue;
            }

            int16_t ypos = rd16(ptr);
            (void)ypos;
            ptr += 2;
            int16_t num_sides_m1 = rd16(ptr);
            ptr += 2;
            int sides = num_sides_m1 + 1;
            int16_t pt_indices[100];
            if (sides < 0) sides = 0;
            if (sides > 100) sides = 100;
            for (int s = 0; s < sides; s++) {
                pt_indices[s] = rd16(ptr);
                ptr += 2;
            }
            ptr += 8;

            {
                int32_t draw_h_world = zone_floor;
                int32_t floor_y_dist = draw_h_world - y_off;
                int poly_top = h;
                int poly_bot = -1;
                int y_min_clamp = half_h;
                int y_max_clamp = h - 1;

                if (draw_h_world < zone_roof || draw_h_world > zone_floor) continue;
                if (floor_y_dist <= 0) continue;

                for (int row = 0; row < h; row++) {
                    left_edge[row] = renderer_clamp_edge_x_i16(w);
                    right_edge[row] = -1;
                }

                const int32_t rel_h = draw_h_world - y_off;
                const int32_t FLOOR_NEAR = ROT_Z_FROM_INT(1);
                for (int s = 0; s < sides; s++) {
                    int i1 = pt_indices[s];
                    int i2 = pt_indices[(s + 1) % sides];
                    if (i1 < 0 || i1 >= MAX_POINTS || i2 < 0 || i2 >= MAX_POINTS) continue;

                    int32_t z1 = r->rotated[i1].z;
                    int32_t z2 = r->rotated[i2].z;
                    int32_t rx1 = r->rotated[i1].x;
                    int32_t rx2 = r->rotated[i2].x;

                    if (z1 < FLOOR_NEAR && z2 < FLOOR_NEAR) continue;

                    int32_t ez1 = z1, ez2 = z2;
                    int32_t ex1 = rx1, ex2 = rx2;
                    if (ez1 < FLOOR_NEAR) {
                        int32_t dz = ez2 - ez1;
                        if (dz != 0) {
                            int32_t t = (int32_t)((int64_t)(FLOOR_NEAR - ez1) * 65536 / dz);
                            ex1 = rx1 + (int32_t)((int64_t)(rx2 - rx1) * t / 65536);
                        } else {
                            ex1 = (rx1 + rx2) / 2;
                        }
                        ez1 = FLOOR_NEAR;
                    }
                    if (ez2 < FLOOR_NEAR) {
                        int32_t dz = ez1 - ez2;
                        if (dz != 0) {
                            int32_t t = (int32_t)((int64_t)(FLOOR_NEAR - ez2) * 65536 / dz);
                            ex2 = rx2 + (int32_t)((int64_t)(rx1 - rx2) * t / 65536);
                        } else {
                            ex2 = (rx1 + rx2) / 2;
                        }
                        ez2 = FLOOR_NEAR;
                    }

                    int sx1 = project_x_to_pixels(ex1, ez1);
                    int sx2 = project_x_to_pixels(ex2, ez2);
                    int32_t rel_h_8 = rel_h >> WORLD_Y_FRAC_BITS;
                    int sy1_raw = project_y_to_pixels_round(rel_h_8, ez1, r->proj_y_scale, half_h);
                    int sy2_raw = project_y_to_pixels_round(rel_h_8, ez2, r->proj_y_scale, half_h);

                    int sy1 = sy1_raw;
                    int sy2 = sy2_raw;
                    if (sy1 < y_min_clamp) sy1 = y_min_clamp;
                    if (sy1 > y_max_clamp) sy1 = y_max_clamp;
                    if (sy2 < y_min_clamp) sy2 = y_min_clamp;
                    if (sy2 > y_max_clamp) sy2 = y_max_clamp;

                    int dy_raw = sy2_raw - sy1_raw;
                    if (dy_raw == 0) {
                        int row = sy1;
                        if (row >= y_min_clamp && row <= y_max_clamp) {
                            int lo = sx1 < sx2 ? sx1 : sx2;
                            int hi = sx1 > sx2 ? sx1 : sx2;
                            if (lo < left_edge[row]) left_edge[row] = renderer_clamp_edge_x_i16(lo);
                            if (hi > right_edge[row]) right_edge[row] = renderer_clamp_edge_x_i16(hi);
                            if (row < poly_top) poly_top = row;
                            if (row > poly_bot) poly_bot = row;
                        }
                        continue;
                    }

                    int row_start = (sy1_raw < sy2_raw) ? sy1_raw : sy2_raw;
                    int row_end = (sy1_raw > sy2_raw) ? sy1_raw : sy2_raw;
                    if (row_start < y_min_clamp) row_start = y_min_clamp;
                    if (row_end > y_max_clamp) row_end = y_max_clamp;

                    int64_t x_fp = (int64_t)sx1 << 16;
                    int64_t dx_fp = ((int64_t)(sx2 - sx1) << 16) / dy_raw;

                    if (sy1_raw < sy2_raw) {
                        x_fp += dx_fp * (row_start - sy1_raw);
                    } else {
                        x_fp = (int64_t)sx2 << 16;
                        dx_fp = ((int64_t)(sx1 - sx2) << 16) / (-dy_raw);
                        x_fp += dx_fp * (row_start - sy2_raw);
                    }

                    for (int row = row_start; row <= row_end; row++) {
                        if (row < 0 || row >= h) {
                            x_fp += dx_fp;
                            continue;
                        }
                        int left_x = renderer_fp16_x_floor_px(x_fp);
                        int right_x = renderer_fp16_x_ceil_px(x_fp);
                        if (left_x < left_edge[row]) left_edge[row] = renderer_clamp_edge_x_i16(left_x);
                        if (right_x > right_edge[row]) right_edge[row] = renderer_clamp_edge_x_i16(right_x);
                        if (row < poly_top) poly_top = row;
                        if (row > poly_bot) poly_bot = row;
                        x_fp += dx_fp;
                    }
                }

                if (poly_top < y_min_clamp) poly_top = y_min_clamp;
                if (poly_top >= h) poly_top = h - 1;
                if (poly_bot > y_max_clamp) poly_bot = y_max_clamp;
                if (poly_bot >= h) poly_bot = h - 1;
                if (poly_bot < 0) continue;

                for (int row = poly_top; row <= poly_bot; row++) {
                    int cle = left_edge[row];
                    int cre = right_edge[row];
                    if (cle >= w || cre < 0) continue;
                    if (cle < col_start) cle = col_start;
                    if (cre >= col_end) cre = col_end - 1;
                    if (cle > cre) continue;
                    for (int x = cle; x <= cre; x++) {
                        if (row < out_top[x]) out_top[x] = (int16_t)row;
                    }
                    active = 1;
                }
            }
        }
    }

    free(left_edge);
    free(right_edge);
    return active;
}

static void zone_poly_sort_indices(int16_t *vals, int count)
{
    for (int i = 1; i < count; i++) {
        int16_t key = vals[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > key) {
            vals[j + 1] = vals[j];
            j--;
        }
        vals[j + 1] = key;
    }
}

static int zone_poly_contains_vertex(const int16_t *vals, int count, int16_t v)
{
    for (int i = 0; i < count; i++) {
        if (vals[i] == v) return 1;
    }
    return 0;
}

/* Canonicalize polygon vertex ids for matching:
 * - remove consecutive duplicate ids
 * - remove closing duplicate (last == first)
 * - unique + sort (winding/start independent) */
static int zone_poly_canonicalize_vertices(const int16_t *in, int in_count, int16_t *out_vals)
{
    if (!in || !out_vals) return 0;
    if (in_count <= 0) return 0;
    if (in_count > 100) in_count = 100;

    int tmp_n = 0;
    for (int i = 0; i < in_count; i++) {
        int16_t v = in[i];
        if (tmp_n > 0 && v == out_vals[tmp_n - 1]) continue;
        out_vals[tmp_n++] = v;
    }
    if (tmp_n >= 2 && out_vals[0] == out_vals[tmp_n - 1]) tmp_n--;

    int uniq_n = 0;
    for (int i = 0; i < tmp_n; i++) {
        int16_t v = out_vals[i];
        if (!zone_poly_contains_vertex(out_vals, uniq_n, v)) {
            out_vals[uniq_n++] = v;
        }
    }
    if (uniq_n < 3) return 0;

    zone_poly_sort_indices(out_vals, uniq_n);
    return uniq_n;
}

/* Roof/floor polygons can be authored with:
 * - different winding/start vertex
 * - redundant points in one stream and not the other
 * Treat exact set and strict subset/superset as a match to avoid false hole detection. */
static int zone_poly_roof_floor_match(const int16_t *a, int a_count,
                                      const int16_t *b, int b_count)
{
    int16_t ca[100];
    int16_t cb[100];
    int na = zone_poly_canonicalize_vertices(a, a_count, ca);
    int nb = zone_poly_canonicalize_vertices(b, b_count, cb);
    if (na <= 0 || nb <= 0) return 0;

    if (na == nb) {
        int same = 1;
        for (int i = 0; i < na; i++) {
            if (ca[i] != cb[i]) { same = 0; break; }
        }
        if (same) return 1;
    }

    int a_in_b = 1;
    for (int i = 0; i < na; i++) {
        if (!zone_poly_contains_vertex(cb, nb, ca[i])) { a_in_b = 0; break; }
    }
    if (a_in_b) return 1;

    int b_in_a = 1;
    for (int i = 0; i < nb; i++) {
        if (!zone_poly_contains_vertex(ca, na, cb[i])) { b_in_a = 0; break; }
    }
    return b_in_a;
}

/* Returns 1 if this stream has a type-2 roof polygon matching floor footprint. */
static int zone_stream_has_matching_roof_polygon(const uint8_t *gfx_data,
                                                 const int16_t *floor_pts, int floor_sides)
{
    if (!gfx_data || !floor_pts) return 0;
    if (floor_sides <= 0 || floor_sides > 100) return 0;

    const uint8_t *scan = gfx_data + 2; /* skip gfx zone id word */
    int scan_iter = 500;
    while (scan_iter-- > 0) {
        int16_t t = rd16(scan);
        scan += 2;
        if (t < 0) break;

        if (t == 2) {
            int poly_sides = (int)rd16(scan + 2) + 1;
            if (poly_sides < 0) poly_sides = 0;
            if (poly_sides > 100) poly_sides = 100;
            if (poly_sides >= 3) {
                int16_t roof_pts[100];
                const uint8_t *pp = scan + 4;
                for (int i = 0; i < poly_sides; i++) {
                    roof_pts[i] = rd16(pp);
                    pp += 2;
                }
                if (zone_poly_roof_floor_match(floor_pts, floor_sides, roof_pts, poly_sides)) {
                    return 1;
                }
            }
        }

        {
            size_t skip = zone_gfx_entry_data_skip(t, scan);
            if (skip == 0) {
                if (t != 3 && t != 12) break;
                continue;
            }
            scan += skip;
        }
    }
    return 0;
}

/* Returns 1 if this stream has at least one authored type-2 roof polygon. */
static AB3D_ATTR_UNUSED int zone_stream_has_explicit_roof_polygon(const uint8_t *gfx_data)
{
    if (!gfx_data) return 0;

    const uint8_t *scan = gfx_data + 2; /* skip gfx zone id word */
    int scan_iter = 500;
    while (scan_iter-- > 0) {
        int16_t t = rd16(scan);
        scan += 2;
        if (t < 0) break;
        if (t == 2) return 1;

        {
            size_t skip = zone_gfx_entry_data_skip(t, scan);
            if (skip == 0) {
                if (t != 3 && t != 12) break;
                continue;
            }
            scan += skip;
        }
    }
    return 0;
}

#define ZONE_STREAM_ENTRY_TYPE_CACHE_SIZE 256u

typedef struct {
    const uint8_t *gfx_data;
    int16_t want_type;
    uint16_t sig0;
    uint16_t sig1;
    uint8_t valid;
    uint8_t result;
} ZoneStreamEntryTypeCacheEntry;

/* Per-thread cache: this lookup runs in hot zone loops (including worker threads). */
static AB3D_THREAD_LOCAL ZoneStreamEntryTypeCacheEntry
    g_zone_stream_entry_type_cache[ZONE_STREAM_ENTRY_TYPE_CACHE_SIZE];

static inline unsigned zone_stream_entry_type_cache_slot(const uint8_t *gfx_data, int16_t want_type)
{
    uintptr_t k = (uintptr_t)gfx_data;
    k ^= (uintptr_t)((uint32_t)(uint16_t)want_type * 0x9E3779B1u);
    k ^= (k >> 11);
    return (unsigned)(k & (ZONE_STREAM_ENTRY_TYPE_CACHE_SIZE - 1u));
}

static int zone_stream_has_entry_type(const uint8_t *gfx_data, int16_t want_type)
{
    unsigned slot;
    uint16_t sig0;
    uint16_t sig1;

    if (!gfx_data) return 0;

    /* Small signature guards against stale cache hits if allocator reuses an address. */
    sig0 = (uint16_t)rd16(gfx_data + 0);
    sig1 = (uint16_t)rd16(gfx_data + 2);
    slot = zone_stream_entry_type_cache_slot(gfx_data, want_type);
    {
        ZoneStreamEntryTypeCacheEntry *ce = &g_zone_stream_entry_type_cache[slot];
        if (ce->valid && ce->gfx_data == gfx_data &&
            ce->want_type == want_type && ce->sig0 == sig0 && ce->sig1 == sig1) {
            return ce->result ? 1 : 0;
        }
    }

    const uint8_t *scan = gfx_data + 2;
    int scan_iter = 500;
    int found = 0;
    while (scan_iter-- > 0) {
        int16_t t = rd16(scan);
        scan += 2;
        if (t < 0) break;
        if (t == want_type) {
            found = 1;
            break;
        }

        {
            size_t skip = zone_gfx_entry_data_skip(t, scan);
            if (skip == 0) {
                if (t != 3 && t != 12) break;
                continue;
            }
            scan += skip;
        }
    }

    {
        ZoneStreamEntryTypeCacheEntry *ce = &g_zone_stream_entry_type_cache[slot];
        ce->gfx_data = gfx_data;
        ce->want_type = want_type;
        ce->sig0 = sig0;
        ce->sig1 = sig1;
        ce->valid = 1;
        ce->result = (uint8_t)(found ? 1 : 0);
    }

    return found;
}

/* Rasterize a floor-outline polygon at zone_roof height as a sky ceiling.
 * pt_indices/sides come directly from the type-1 (floor) polygon already read. */
static void renderer_tessellate_sky_ceiling_ctx(RenderSliceContext *ctx,
                                                const int16_t *pt_indices, int sides,
                                                int32_t zone_roof, int32_t y_off)
{
    RendererState *r = &g_renderer;
    int h = r->height;
    int half_h = h / 2;
    int center = half_h;

    int32_t rel_h = zone_roof - y_off;
    if (rel_h >= 0) return; /* ceiling must be above camera eye */

    int16_t *left_edge  = (int16_t*)malloc((size_t)h * sizeof(int16_t));
    int16_t *right_edge = (int16_t*)malloc((size_t)h * sizeof(int16_t));
    if (!left_edge || !right_edge) { free(left_edge); free(right_edge); return; }

    for (int i = 0; i < h; i++) {
        left_edge[i]  = renderer_clamp_edge_x_i16(r->width);
        right_edge[i] = -1;
    }
    int poly_top = h;
    int poly_bot = -1;

    /* Ceiling: rows 0 .. half_h-1, further clamped to ctx clip band */
    int y_min = ctx->top_clip;
    int y_max = half_h - 1;
    if (y_max > ctx->bot_clip) y_max = ctx->bot_clip;

    const int32_t NEAR = ROT_Z_FROM_INT(1);
    int32_t rel_h_8 = rel_h >> WORLD_Y_FRAC_BITS;

    for (int s = 0; s < sides; s++) {
        int i1 = pt_indices[s];
        int i2 = pt_indices[(s + 1) % sides];
        if (i1 < 0 || i1 >= MAX_POINTS || i2 < 0 || i2 >= MAX_POINTS) continue;

        int32_t z1 = r->rotated[i1].z, z2 = r->rotated[i2].z;
        int32_t x1 = r->rotated[i1].x, x2 = r->rotated[i2].x;

        if (z1 < NEAR && z2 < NEAR) continue;

        int32_t ez1 = z1, ez2 = z2, ex1 = x1, ex2 = x2;
        if (ez1 < NEAR) {
            int32_t dz = ez2 - ez1;
            ex1 = dz ? (x1 + (int32_t)((int64_t)(x2-x1) * (NEAR-ez1) / dz)) : (x1+x2)/2;
            ez1 = NEAR;
        }
        if (ez2 < NEAR) {
            int32_t dz = ez1 - ez2;
            ex2 = dz ? (x2 + (int32_t)((int64_t)(x1-x2) * (NEAR-ez2) / dz)) : (x1+x2)/2;
            ez2 = NEAR;
        }

        int sx1 = project_x_to_pixels(ex1, ez1);
        int sx2 = project_x_to_pixels(ex2, ez2);
        int sy1_raw = project_y_to_pixels_round(rel_h_8, ez1, r->proj_y_scale, center);
        int sy2_raw = project_y_to_pixels_round(rel_h_8, ez2, r->proj_y_scale, center);

        int dy_raw = sy2_raw - sy1_raw;
        if (dy_raw == 0) {
            int row = sy1_raw;
            if (row < y_min || row > y_max) continue;
            int lo = sx1 < sx2 ? sx1 : sx2;
            int hi = sx1 > sx2 ? sx1 : sx2;
            if (lo < left_edge[row])  left_edge[row]  = renderer_clamp_edge_x_i16(lo);
            if (hi > right_edge[row]) right_edge[row] = renderer_clamp_edge_x_i16(hi);
            if (row < poly_top) poly_top = row;
            if (row > poly_bot) poly_bot = row;
            continue;
        }

        int row_start = sy1_raw < sy2_raw ? sy1_raw : sy2_raw;
        int row_end   = sy1_raw > sy2_raw ? sy1_raw : sy2_raw;
        if (row_start < y_min) row_start = y_min;
        if (row_end   > y_max) row_end   = y_max;

        int64_t x_fp = (int64_t)sx1 << 16;
        int64_t dx_fp = ((int64_t)(sx2 - sx1) << 16) / dy_raw;
        if (sy1_raw < sy2_raw) {
            x_fp += dx_fp * (row_start - sy1_raw);
        } else {
            x_fp = (int64_t)sx2 << 16;
            dx_fp = ((int64_t)(sx1 - sx2) << 16) / (-dy_raw);
            x_fp += dx_fp * (row_start - sy2_raw);
        }

        for (int row = row_start; row <= row_end; row++) {
            if (row < 0 || row >= h) { x_fp += dx_fp; continue; }
            int lx = renderer_fp16_x_floor_px(x_fp);
            int rx = renderer_fp16_x_ceil_px(x_fp);
            if (lx < left_edge[row])  left_edge[row]  = renderer_clamp_edge_x_i16(lx);
            if (rx > right_edge[row]) right_edge[row] = renderer_clamp_edge_x_i16(rx);
            if (row < poly_top) poly_top = row;
            if (row > poly_bot) poly_bot = row;
            x_fp += dx_fp;
        }
    }

    if (poly_top < y_min) poly_top = y_min;
    if (poly_bot > y_max) poly_bot = y_max;

    for (int row = poly_top; row <= poly_bot; row++) {
        if (row < 0 || row >= h) continue;
        int16_t le = left_edge[row];
        int16_t re = right_edge[row];
        if (le >= r->width || re < 0) continue;
        {
            int16_t cle = le < (int16_t)ctx->left_clip ? (int16_t)ctx->left_clip : le;
            int16_t cre = re >= (int16_t)ctx->right_clip ? (int16_t)(ctx->right_clip-1) : re;
            if (cle > cre) continue;
        }
        renderer_draw_sky_ceiling_span_ctx(ctx, (int16_t)row, le, re);
    }

    free(left_edge);
    free(right_edge);
}

static void renderer_build_zone_stream_backdrop_sky_cache(int use_upper,
                                                          const uint8_t *lower_stream_gfx_data,
                                                          const uint8_t *upper_stream_gfx_data,
                                                          RendererSkyBuildDebug *dbg)
{
    const uint8_t *floor_stream_gfx_data = use_upper ? upper_stream_gfx_data : lower_stream_gfx_data;
    if (!floor_stream_gfx_data) return;
    const uint8_t *roof_stream_primary = floor_stream_gfx_data;
    const uint8_t *roof_stream_secondary = use_upper ? NULL : upper_stream_gfx_data;

    const uint8_t *scan = floor_stream_gfx_data + 2; /* skip gfx zone id word */
    int scan_iter = 500;
    while (scan_iter-- > 0) {
        int16_t t = rd16(scan);
        scan += 2;
        if (t < 0) break;

        if (t == 1) { /* floor */
            int16_t num_sides_m1 = rd16(scan + 2);
            int sides = (int)num_sides_m1 + 1;
            if (sides < 0) sides = 0;
            if (sides > 100) sides = 100;
            if (sides >= 3) {
                int16_t pt_indices[100];
                const uint8_t *pp = scan + 4;
                for (int i = 0; i < sides; i++) {
                    pt_indices[i] = rd16(pp);
                    pp += 2;
                }
                if (dbg) dbg->floor_polys_seen++;

                int has_matching_roof =
                    zone_stream_has_matching_roof_polygon(roof_stream_primary, pt_indices, sides);
                if (!has_matching_roof && roof_stream_secondary) {
                    has_matching_roof =
                        zone_stream_has_matching_roof_polygon(roof_stream_secondary, pt_indices, sides);
                }
                if (has_matching_roof) {
                    if (dbg) dbg->floor_polys_with_matching_roof++;
                } else {
                    if (renderer_level_sky_cache_push_poly(pt_indices, sides)) {
                        if (dbg) {
                            dbg->floor_polys_sky_added++;
                            dbg->sky_added_missing_matching_roof++;
                        }
                    } else if (dbg) {
                        dbg->sky_push_failed++;
                    }
                }
            }
        }

        {
            size_t skip = zone_gfx_entry_data_skip(t, scan);
            if (skip == 0) {
                if (t != 3 && t != 12) break;
                continue;
            }
            scan += skip;
        }
    }
}

void renderer_build_level_sky_cache(const LevelState *level)
{
    renderer_reset_level_sky_cache_internal();
    if (!level || !level->data || !level->graphics || !level->zone_adds || !level->zone_graph_adds) return;

    int zone_slots = level_zone_slot_count(level);
    if (zone_slots <= 0) return;

    g_level_sky_cache_buckets = (RendererSkyCacheBucket *)calloc((size_t)zone_slots * 2u,
                                                                  sizeof(RendererSkyCacheBucket));
    if (!g_level_sky_cache_buckets) return;
    g_level_sky_cache_zone_slots = zone_slots;

    int zone_limit = zone_slots;
    if (level->num_zone_graph_entries > 0 && level->num_zone_graph_entries < zone_limit)
        zone_limit = level->num_zone_graph_entries;
    /* Do not clamp by num_zones: some real levels use extra zone_adds slots
     * (num_zone_slots > num_zones), and those slots can be valid runtime zones. */

    for (int zone_id = 0; zone_id < zone_limit; zone_id++) {
        int32_t zone_off = rd32(level->zone_adds + (size_t)zone_id * 4u);
        if (zone_off < 0) continue;
        if (level->data_byte_count > 0 && (size_t)zone_off + 20u > level->data_byte_count) continue;
        const uint8_t *zone_data = level->data + zone_off;

        const uint8_t *zgraph = level->zone_graph_adds + (size_t)zone_id * 8u;
        int32_t lower_gfx_off = rd32(zgraph);
        int32_t upper_gfx_off = rd32(zgraph + 4);
        int32_t lower_roof_y = rd32(zone_data + ZONE_OFF_ROOF);
        int32_t upper_roof_raw = rd32(zone_data + ZONE_OFF_UPPER_ROOF);
        const uint8_t *lower_gfx_data = NULL;
        const uint8_t *upper_gfx_data = NULL;
        int zone_backdrop_flag = (rd16(zone_data + ZONE_OFF_BACK) != 0);

        if (lower_gfx_off > 0 &&
            (level->graphics_byte_count == 0 || ((size_t)lower_gfx_off + 2u) <= level->graphics_byte_count)) {
            lower_gfx_data = level->graphics + lower_gfx_off;
        }
        if (upper_gfx_off > 0 &&
            (level->graphics_byte_count == 0 || ((size_t)upper_gfx_off + 2u) <= level->graphics_byte_count)) {
            upper_gfx_data = level->graphics + upper_gfx_off;
        }

        for (int use_upper = 0; use_upper <= 1; use_upper++) {
            int bucket_idx = zone_id * 2 + use_upper;
            uint32_t start = g_level_sky_cache_poly_count;
            g_level_sky_cache_buckets[bucket_idx].start = start;
            g_level_sky_cache_buckets[bucket_idx].count = 0;

            const uint8_t *gfx_data = use_upper ? upper_gfx_data : lower_gfx_data;
            if (!gfx_data) continue;

            int stream_has_backdrop_marker = 0;
            {
                const uint8_t *scan = gfx_data + 2;
                int scan_iter = 500;
                while (scan_iter-- > 0) {
                    int16_t t = rd16(scan);
                    scan += 2;
                    if (t < 0) break;
                    if (t == 12) stream_has_backdrop_marker = 1;

                    {
                        size_t skip = zone_gfx_entry_data_skip(t, scan);
                        if (skip == 0) {
                            if (t != 3 && t != 12) break;
                            continue;
                        }
                        scan += skip;
                    }
                }
            }

            int32_t stream_roof_y = use_upper ? upper_roof_raw : lower_roof_y;
            int roof_open_to_sky = (stream_roof_y < 0) ? 1 : 0;
            int synth_enabled =
                ((zone_backdrop_flag || stream_has_backdrop_marker) && roof_open_to_sky) ? 1 : 0;
            RendererSkyBuildDebug dbg = {0};

            if (synth_enabled) {
                renderer_build_zone_stream_backdrop_sky_cache(use_upper,
                                                              lower_gfx_data,
                                                              upper_gfx_data,
                                                              &dbg);
            }
            uint32_t added = g_level_sky_cache_poly_count - start;
            g_level_sky_cache_buckets[bucket_idx].count = added;

        }
    }

    if (g_level_sky_cache_poly_count > 0) {
        printf("[RENDERER] Sky cache: %u synthesized polygons\n",
               (unsigned)g_level_sky_cache_poly_count);
    }
}

static void renderer_draw_zone_backdrop_sky_ctx(RenderSliceContext *ctx,
                                                int16_t zone_id, int use_upper,
                                                int32_t zone_roof, int32_t y_off)
{
    /* Backdrop sky is only valid for open roofs (same rule for lower and upper). */
    if (zone_roof >= 0) return;
    if (!ctx || !g_level_sky_cache_buckets || !g_level_sky_cache_polys) return;
    if (zone_id < 0 || zone_id >= g_level_sky_cache_zone_slots) return;

    int bucket_idx = zone_id * 2 + (use_upper ? 1 : 0);
    RendererSkyCacheBucket b = g_level_sky_cache_buckets[bucket_idx];
    uint32_t end = b.start + b.count;
    if (end > g_level_sky_cache_poly_count) end = g_level_sky_cache_poly_count;

    for (uint32_t i = b.start; i < end; i++) {
        const RendererSkyCachePoly *poly = &g_level_sky_cache_polys[i];
        renderer_tessellate_sky_ceiling_ctx(ctx, poly->pt_indices, poly->sides, zone_roof, y_off);
    }
}

static void renderer_draw_zone_ctx(RenderSliceContext *ctx, GameState *state, int16_t zone_id, int use_upper)
{
    if (!ctx || !state) return;
    RendererState *r = &g_renderer;
    LevelState *level = &state->level;
    ctx->pick_zone_id = zone_id;
    ctx->pick_player_id = 0;

    if (!level->data || !level->zone_adds || !level->zone_graph_adds) return;
    {
        int zone_slots = level_zone_slot_count(level);
        if (zone_id < 0 || zone_id >= zone_slots) return;
        if (level->num_zone_graph_entries > 0 && zone_id >= level->num_zone_graph_entries)
            return;
    }

    /* Get zone data (same level->data that door_routine/lift_routine write to each frame). */
    int32_t zone_off = rd32(level->zone_adds + zone_id * 4);
    const uint8_t *zone_data = level->data + zone_off;

    /* Zone heights: upper room uses its own floor/roof stored at offsets +10/+14.
     * For lower room, ZD_FLOOR (2) and ZD_ROOF (6) are written each frame by door_routine and
     * lift_routine; re-read them when this zone is tagged as door or lift so we see the sine/lift. */
    int32_t zone_floor, zone_roof;
    if (use_upper) {
        /* Match Amiga DoThisRoom upper pass: use ToUpperFloor/ToUpperRoof directly. */
        zone_floor = rd32(zone_data + ZONE_OFF_UPPER_FLOOR);
        zone_roof  = rd32(zone_data + ZONE_OFF_UPPER_ROOF);
    } else {
        zone_floor = rd32(zone_data + 2);   /* ZD_FLOOR (ToZoneFloor) */
        zone_roof  = rd32(zone_data + 6);   /* ZD_ROOF  (ToZoneRoof)  */
        /* Door/lift zones: use live values (door sine wave and lift position write here each frame) */
        if (zone_has_door(level->door_data, zone_id) || zone_has_lift(level->lift_data, zone_id)) {
            zone_floor = rd32(zone_data + 2);
            zone_roof  = rd32(zone_data + 6);
        }
    }

    /* Get zone graphics data (the polygon list for this zone).
     * zone_graph_adds: 8 bytes per zone = lower gfx offset (long) + upper gfx offset (long). */
    const uint8_t *zgraph = level->zone_graph_adds + zone_id * 8;
    int32_t gfx_off = use_upper ? rd32(zgraph + 4) : rd32(zgraph);
    int32_t other_gfx_off = use_upper ? rd32(zgraph) : rd32(zgraph + 4);
    if (gfx_off <= 0 || !level->graphics) return;
    if (level->graphics_byte_count > 0 &&
        ((size_t)gfx_off + 2u > level->graphics_byte_count)) return;

    const uint8_t *gfx_data = level->graphics + (size_t)gfx_off;
    const uint8_t *other_gfx_data = NULL;
    if (other_gfx_off > 0 && level->graphics &&
        (level->graphics_byte_count == 0 || ((size_t)other_gfx_off + 2u <= level->graphics_byte_count))) {
        other_gfx_data = level->graphics + (size_t)other_gfx_off;
    }
    int32_t zone_water = rd32(zone_data + 18);  /* ToZoneWater */

    int32_t y_off = r->yoff;
    PlayerState *viewer = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
    int16_t viewer_zone = viewer->zone;
    int trace_floor_zone = (g_renderer_zone_trace_active && !use_upper && zone_id == viewer_zone) ? 1 : 0;
    int half_h = g_renderer.height / 2;

    int has_split_water = (zone_water > zone_roof && zone_water < zone_floor);

    /* ObjDraw beforewat/afterwat clip bounds (ObjDraw3.ChipRam.s BEFOREWAT and AFTERWAT labels).
     * Bounds swap depending on whether the viewer is above or below the water plane.
     * In dry/non-split rooms, keep both passes as full-room so authored clip mode 0/1
     * cannot collapse to a zero-height band. */
    int32_t before_wat_top, before_wat_bot, after_wat_top, after_wat_bot;
    if (has_split_water) {
        if (zone_water < y_off) {
            before_wat_top = zone_roof;
            before_wat_bot = zone_water;
            after_wat_top = zone_water;
            after_wat_bot = zone_floor;
        } else {
            before_wat_top = zone_water;
            before_wat_bot = zone_floor;
            after_wat_top = zone_roof;
            after_wat_bot = zone_water;
        }
    } else {
        before_wat_top = zone_roof;
        before_wat_bot = zone_floor;
        after_wat_top = zone_roof;
        after_wat_bot = zone_floor;
    }

    int zone_has_door_flag = zone_has_door(level->door_data, zone_id);
    int zone_has_lift_flag = zone_has_lift(level->lift_data, zone_id);
    int has_door_wall_list = (level->door_wall_list && level->door_wall_list_offsets && level->num_doors > 0);
    int has_lift_wall_list = (level->lift_wall_list && level->lift_wall_list_offsets && level->num_lifts > 0);
    int current_stream_has_object_entries = zone_stream_has_entry_type(gfx_data, 4);
    int zone_backdrop_flag = (rd16(zone_data + ZONE_OFF_BACK) != 0);
    int stream_has_backdrop_marker = zone_stream_has_entry_type(gfx_data, 12);
    if (!stream_has_backdrop_marker && other_gfx_data) {
        stream_has_backdrop_marker = zone_stream_has_entry_type(other_gfx_data, 12);
    }
    int zone_open_sky = ((zone_roof < 0) && (zone_backdrop_flag || stream_has_backdrop_marker)) ? 1 : 0;
    int is_multi_floor_zone = (other_gfx_data != NULL);
    int fallback_object_pass = (!current_stream_has_object_entries && !has_split_water) ? 1 : 0;

    /* Read zone number from graphics data (consumed before polyloop) */
    const uint8_t *ptr = gfx_data;
    /* int16_t gfx_zone = rd16(ptr); */
    ptr += 2;

    /* Wall column clip spans are only needed for object/sprite occlusion.
     * If this zone stream has no object entry at all, skip clip bookkeeping
     * for every wall in this zone to reduce per-column overhead. */
    {
        int8_t prev_update_column_clip = ctx->update_column_clip;
        if (prev_update_column_clip) {
            if (!current_stream_has_object_entries && !fallback_object_pass) {
                ctx->update_column_clip = 0;
            }
        }

    /* Zone brightness from level data (no table); anim applied via level_get_zone_brightness. */
    int16_t zone_bright = 0;
    if (zone_id >= 0 && zone_id < level_zone_slot_count(level))
        zone_bright = level_get_zone_brightness(level, zone_id, use_upper ? 1 : 0);

    renderer_draw_zone_backdrop_sky_ctx(ctx, zone_id, use_upper, zone_roof, y_off);

    /* Amiga: draw walls and arcs in stream order (no deferral). */

    int max_iter = 500; /* Safety limit */
    int stream_entry_index = 0;

        while (max_iter-- > 0) {
        int16_t entry_type = rd16(ptr);
        ptr += 2; /* Consume type word (matches ASM (a0)+) */

        if (entry_type < 0) break; /* End of list */

        switch (entry_type) {
        case 0:  /* Wall */
        case 13: /* See-through wall */
        {
            /* Wall entry: 28 bytes of data
             * Translated from WallRoutine3.ChipMem.s itsawalldraw (line 1761).
             * Amiga: draw immediately in stream order (polyloop calls itsawalldraw per wall). */
            int16_t p1       = rd16(ptr + 0);   /* point1 index */
            int16_t p2       = rd16(ptr + 2);   /* point2 index */
            int16_t leftend  = rd16(ptr + 4);   /* strip start */
            int16_t rightend = rd16(ptr + 6);   /* strip end */
            /* ASM line 1770-1772: fromtile = (ptr+8) << 4 */
            int16_t fromtile = rd16(ptr + 8) << 4; /* horiz texture offset */
            int16_t totalyoff = rd16(ptr + 10); /* vertical texture offset */
            uint8_t  valand    = ptr[14];             /* vert AND mask */
            uint8_t  valshift  = ptr[15];             /* vert shift */
            int16_t  horand    = rd16(ptr + 16);      /* horiz AND mask */
            int32_t topwall  = rd32(ptr + 18);  /* wall top height */
            int32_t botwall  = rd32(ptr + 22);  /* wall bottom height */
            int16_t wallbrightoff = rd16(ptr + 26);  /* ASM: move.w (a0)+,wallbrightoff */

            /* Subtract camera Y (ASM: sub.l d6,topofwall / sub.l d6,botofwall) */
            topwall -= y_off;
            botwall -= y_off;

            int16_t tex_id = rd16(ptr + 12);
            if (p1 >= 0 && p1 < MAX_POINTS && p2 >= 0 && p2 < MAX_POINTS)
            {
                uint32_t wall_gfx_off = (level->graphics != NULL)
                                            ? (uint32_t)((ptr - 2) - level->graphics)
                                            : 0u;
                int32_t rx1 = r->rotated[p1].x;
                int32_t rz1 = r->rotated[p1].z;
                int32_t rx2 = r->rotated[p2].x;
                int32_t rz2 = r->rotated[p2].z;
                const uint8_t *wall_tex = (tex_id >= 0 && tex_id < MAX_WALL_TILES) ? r->walltiles[tex_id] : NULL;
                /* Prefer level-authored VALAND/VALSHIFT. Some textures have ambiguous raw sizes
                 * (e.g. 64 vs 128 rows), and inferred file dimensions can misalign specific doors.
                 * Only fall back to loaded-file dimensions when level data is missing/zero. */
                uint8_t use_valand = valand, use_valshift = valshift;
                if ((use_valand == 0 || use_valshift == 0) &&
                           tex_id >= 0 && tex_id < MAX_WALL_TILES && r->wall_valshift[tex_id] != 0) {
                    use_valand = r->wall_valand[tex_id];
                    use_valshift = r->wall_valshift[tex_id];
                }

                /* Amiga wall brightness uses per-point brightness (pointBrightsPtr), then adds wallbrightoff.
                 * If point brightness data is unavailable, fall back to zone brightness. */
                int16_t point_bright_l = zone_bright;
                int16_t point_bright_r = zone_bright;
                if (level->point_brights) {
                    point_bright_l = level_get_point_brightness(level, p1, use_upper ? 1 : 0);
                    point_bright_r = level_get_point_brightness(level, p2, use_upper ? 1 : 0);
                }
                int16_t wall_bright_l = (int16_t)(point_bright_l + wallbrightoff);
                int16_t wall_bright_r = (int16_t)(point_bright_r + wallbrightoff);

                int16_t wall_top = (int16_t)(topwall >> 8);
                int16_t wall_bot = (int16_t)(botwall >> 8);
                /* Original level height for texture step (before any door override). */
                int16_t wall_height_for_tex = (int16_t)((botwall - topwall) >> 8);
                if (wall_height_for_tex < 1) wall_height_for_tex = 1;
                int32_t door_yoff_add = 0;
                int skip_this_wall = 0;

                /* Lift zone: clip all walls to current zone floor/roof so we don't draw wall below the platform. */
                if (!has_lift_wall_list && zone_has_lift_flag && tex_id != SWITCHES_WALL_TEX_ID) {
                    int32_t live_zone_roof = rd32(zone_data + 6);
                    int32_t live_zone_floor = rd32(zone_data + 2);
                    int32_t zone_roof_rel = live_zone_roof - y_off;
                    int32_t zone_floor_rel = live_zone_floor - y_off;
                    int32_t draw_top = topwall > zone_roof_rel ? topwall : zone_roof_rel;
                    int32_t draw_bot = botwall < zone_floor_rel ? botwall : zone_floor_rel;
                    if (draw_bot <= draw_top) {
                        skip_this_wall = 1;
                    } else {
                        wall_top = (int16_t)(draw_top >> 8);
                        wall_bot = (int16_t)(draw_bot >> 8);
                        int32_t wall_full_h = botwall - topwall;
                        if (wall_full_h > 0) {
                            int rows = 1 << use_valshift;
                            door_yoff_add = (int32_t)((int64_t)(draw_top - topwall) * rows / wall_full_h);
                        }
                        /* Use clipped height for texture so the visible segment doesn't stretch. */
                        wall_height_for_tex = (int16_t)((draw_bot - draw_top) >> 8);
                        if (wall_height_for_tex < 1) wall_height_for_tex = 1;
                    }
                }
                /* Fallback when wall list data is unavailable: old heuristic for door zones only. */
                else if (!has_door_wall_list && zone_has_door_flag && tex_id != SWITCHES_WALL_TEX_ID) {
                    int32_t live_zone_roof = rd32(zone_data + 6);
                    int32_t live_zone_floor = rd32(zone_data + 2);
                    int32_t zone_roof_rel = live_zone_roof - y_off;
                    int32_t zone_floor_rel = live_zone_floor - y_off;
                    int32_t top_abs = topwall + y_off, bot_abs = botwall + y_off;
                    const int32_t zone_match_margin = 512;
                    if (top_abs >= live_zone_roof - zone_match_margin &&
                        bot_abs <= live_zone_floor + zone_match_margin) {
                        wall_top = (int16_t)(zone_roof_rel >> 8);
                        wall_bot = (int16_t)(zone_floor_rel >> 8);
                        int32_t wall_full_h = botwall - topwall;
                        if (wall_full_h > 0) {
                            int32_t door_top_offset = live_zone_roof - topwall - y_off;
                            int rows = 1 << use_valshift;
                            door_yoff_add = (int32_t)((int64_t)door_top_offset * rows / wall_full_h);
                        }
                        /* Use clipped height for texture so the door panel doesn't stretch. */
                        wall_height_for_tex = (int16_t)((zone_floor_rel - zone_roof_rel) >> 8);
                        if (wall_height_for_tex < 1) wall_height_for_tex = 1;
                    }
                }

                /* Level 4 (1-indexed) doorway: fix panel vertical scale (not y-offset).
                 * Apply only on the known adjacent-door wall signatures. */
                if (state->current_level == 3 &&
                    renderer_level4_adjacent_door_wall_match(p1, p2, tex_id)) {
                    wall_height_for_tex = (int16_t)(wall_height_for_tex - 16);
                    if (wall_height_for_tex < 1) wall_height_for_tex = 1;
                }

                /* Level 6 (1-indexed), door zones 120/123:
                 * apply the same scale-only correction, keyed by door wall-list membership
                 * so adjacent-zone faces are included automatically. */
                if (state->current_level == 5) {
                    int l6_zone120 = renderer_wall_gfx_off_matches_door_zone(level, wall_gfx_off, 120);
                    int l6_zone123 = renderer_wall_gfx_off_matches_door_zone(level, wall_gfx_off, 123);
                    if (l6_zone120 || l6_zone123) {
                    wall_height_for_tex = (int16_t)(wall_height_for_tex - 16);
                    if (wall_height_for_tex < 1) wall_height_for_tex = 1;
                    if (l6_zone120 && !g_renderfix_l6_zone120_seen) {
                        g_renderfix_l6_zone120_seen = 1;
                        printf("[RENDERFIX] level 6 door zone 120 scale fix active (first hit: zone=%d p1=%d p2=%d tex=%d gfx_off=%u)\n",
                               (int)zone_id, (int)p1, (int)p2, (int)tex_id, (unsigned)wall_gfx_off);
                    }
                    if (l6_zone123 && !g_renderfix_l6_zone123_seen) {
                        g_renderfix_l6_zone123_seen = 1;
                        printf("[RENDERFIX] level 6 door zone 123 scale fix active (first hit: zone=%d p1=%d p2=%d tex=%d gfx_off=%u)\n",
                               (int)zone_id, (int)p1, (int)p2, (int)tex_id, (unsigned)wall_gfx_off);
                    }
                    }
                }

                /* Switch walls (tex_id 11): same texture has on/off states.
                 * State is in wall first word bit 1 (p1 & 2): set = on, clear = off.
                 * Texture layout: off = left half (fromtile), on = right half (fromtile + 32).
                 * Use V offset 0 for switches so full texture maps consistently. */
                int16_t eff_totalyoff = (tex_id == SWITCHES_WALL_TEX_ID) ? 0 : (int16_t)(totalyoff + door_yoff_add);
                int16_t eff_fromtile   = fromtile;

                if (tex_id >= 0 && tex_id < MAX_WALL_TILES)
                    ctx->cur_wall_pal = r->wall_palettes[tex_id];
                else
                    ctx->cur_wall_pal = NULL;
                /* Amiga seethru path (type 13) clamps d6 to 32; normal walls clamp to 64. */
                int16_t wall_d6_max = (entry_type == 13) ? 32 : 64;
                if (!skip_this_wall) {
                    /* Automap: tag this wall as seen (serialized via g_automap_mutex). */
                    if (level->points && level->graphics) {
                        /* door_wall_list gfx_off points to the wall record starting at the type word. */
                        int16_t wx1 = rd16(level->points + (size_t)p1 * 4u + 0u);
                        int16_t wz1 = rd16(level->points + (size_t)p1 * 4u + 2u);
                        int16_t wx2 = rd16(level->points + (size_t)p2 * 4u + 0u);
                        int16_t wz2 = rd16(level->points + (size_t)p2 * 4u + 2u);
                        int is_door = 0;
                        uint8_t key_id = automap_door_key_for_wall_gfx_off(level, wall_gfx_off,
                                                                            wx1, wz1, wx2, wz2, zone_id, &is_door);
                        automap_mark_seen(level, wall_gfx_off, zone_id, wx1, wz1, wx2, wz2,
                                          (uint8_t)(is_door ? 1 : 0), key_id);
                    }
                    {
                        /* See-through/sky wall polys should not occlude billboards. */
                        int8_t prev_wall_update_column_clip = ctx->update_column_clip;
                        if (entry_type == 13) ctx->update_column_clip = 0;
                        renderer_draw_wall_ctx(ctx, rx1, rz1, rx2, rz2,
                                               wall_top, wall_bot,
                                               wall_tex, leftend, rightend,
                                               wall_bright_l, wall_bright_r,
                                               use_valand, use_valshift, horand,
                                               eff_totalyoff, eff_fromtile, tex_id,
                                               wall_height_for_tex, wall_d6_max);
                        ctx->update_column_clip = prev_wall_update_column_clip;
                    }
                }
            }
            ptr += 28;
            break;
        }

        case 1:  /* Floor */
        case 2:  /* Roof */
        case 7:  /* Water */
        case 8:  /* Chunky floor */
        case 9:  /* Bumpy floor */
        case 10: /* Bumpy floor variant */
        case 11: /* Bumpy floor variant */
        {
            /* Floor/Roof/Water polygon entry (variable size)
             * Translated from AB3DI.s itsafloordraw (line 5066).
             *
             * Format after type word:
             *   ypos (word): floor height
             *   num_sides-1 (word): number of polygon sides minus 1
             *   point_indices (word * (sides)): vertex indices
             *   extra_data (10 bytes): skip over
             */
            int16_t ypos = rd16(ptr);
            ptr += 2;
            int16_t num_sides_m1 = rd16(ptr);
            ptr += 2;

            /* Skip point indices (2 bytes each) */
            int sides = num_sides_m1 + 1;
            if (sides < 0) sides = 0;
            if (sides > 100) sides = 100; /* safety */

            /* We need these point indices for proper polygon rendering.
             * For now, collect them for the fill algorithm. */
            int16_t pt_indices[100];
            for (int s = 0; s < sides; s++) {
                pt_indices[s] = rd16(ptr);
                ptr += 2;
            }
            int use_gour_floor = ((entry_type == 1 || entry_type == 2) && level->point_brights != NULL);
            int16_t pt_brights[100];
            if (use_gour_floor) {
                for (int s = 0; s < sides; s++) {
                    int16_t pidx = pt_indices[s];
                    if (pidx >= 0 && pidx < MAX_POINTS) {
                        pt_brights[s] = level_get_point_brightness(level, pidx, use_upper ? 1 : 0);
                    } else {
                        pt_brights[s] = 0;
                    }
                }
            }

            /* Extra data after point indices (ASM: pastsides, line 5891):
             *   +0: padding (2 bytes, consumed by sideloop peek + addq #2)
             *   +2: scaleval (word) - texture scale shift
             *   +4: whichtile (word) - byte offset into floortile sheet
             *   +6: brightness offset (word) - added to ZoneBright
             * Total: 8 bytes.  Note: dontdrawreturn uses lea 4+6(a0),a0
             * which skips past the last point index (2) + these 8 = 10. */
            ptr += 2; /* padding */
            int16_t scaleval = rd16(ptr); ptr += 2;
            int16_t whichtile = rd16(ptr); ptr += 2;
            int16_t floor_bright_off = rd16(ptr); ptr += 2;

            /* Polygon height in world coords (same scale as y_off: *256). */
            int32_t poly_h_world = (int32_t)ypos << 6; /* ASM: asl.l #6,d7 */
            /* Keep floor/roof tied to live zone values so moving lifts/doors update each frame. */
            int32_t draw_h_world = poly_h_world;
            if (entry_type == 1)
                draw_h_world = zone_floor;
            else if (entry_type == 2)
                draw_h_world = zone_roof;
            /* AB3DI itsafloordraw early reject:
             *   if (ypos_world < TOPOFROOM) skip (water path uses checkforwater)
             *   if (ypos_world > BOTOFROOM) skip */
            if (draw_h_world < zone_roof || draw_h_world > zone_floor) {
                if (entry_type == 7 && !use_upper && zone_id == viewer_zone &&
                    poly_h_world < zone_roof) {
                    ctx->fill_screen_water = 0x0F;
                }
                continue;
            }
            int32_t rel_h = draw_h_world - y_off; /* Relative to camera */

            /* Amiga fillscrnwater flag: mark underwater/half-submerged when drawing water in viewer zone. */
            if (entry_type == 7 && !use_upper && zone_id == viewer_zone) {
                int32_t rel_water = draw_h_world - y_off;
                if (rel_water < 0) {
                    ctx->fill_screen_water = 0x0F; /* strong underwater tint */
                } else if (rel_water <= (1 << WORLD_Y_FRAC_BITS) && ctx->fill_screen_water == 0) {
                    ctx->fill_screen_water = (int8_t)-1; /* weaker near-surface tint */
                }
            }

            /* Sign decides which half of screen (floor vs ceiling).
             * Keep water in world-space like rel_h so near-surface behavior
             * stays aligned with fillscrnwater tint thresholds. */
            int32_t floor_y_dist;
            if (entry_type == 1 || entry_type == 2 || entry_type == 7) {
                floor_y_dist = draw_h_world - y_off;
            } else {
                floor_y_dist = ((int32_t)ypos - (int32_t)r->flooryoff);
            }

            if (floor_y_dist == 0) {
                /* At eye level - skip */
                continue;
            }

            /* ---- Polygon scanline rasterization ----
             * Translated from AB3DI.s sideloop (line 5208).
             *
             * Build left/right edge tables from polygon edges,
             * then fill between the edges for each row.
             *
             * Screen Y at each vertex: sy = rel_h / z + center
             * Screen X from on_screen[] (already projected).
             */
            int h = g_renderer.height;
            int center = h / 2;  /* Match wall projection center */
            int16_t *left_edge = (int16_t*)malloc((size_t)h * sizeof(int16_t));
            int16_t *right_edge_tab = (int16_t*)malloc((size_t)h * sizeof(int16_t));
            int16_t *col_top_tab = NULL;
            int16_t *col_bot_tab = NULL;
            int16_t *left_bright_tab = NULL;
            int16_t *right_bright_tab = NULL;
            int col_x_min = g_renderer.width;
            int col_x_max = -1;
            const int allow_floor_col_fast =
                (AB3D_ENABLE_FLOOR_COL_FAST && AB3D_CW_COL_MAJOR);
            const int allow_floor_col_fast_poly =
                (allow_floor_col_fast && entry_type != 7 &&
                 (floor_y_dist > 0 || AB3D_FLOOR_COL_FAST_INCLUDE_CEILING));
            const int build_floor_col_bounds =
                (allow_floor_col_fast_poly && AB3D_FLOOR_FAST_USE_EDGE_BOUNDS);
            if (build_floor_col_bounds) {
                col_top_tab = (int16_t*)malloc((size_t)g_renderer.width * sizeof(int16_t));
                col_bot_tab = (int16_t*)malloc((size_t)g_renderer.width * sizeof(int16_t));
            }
            if (use_gour_floor) {
                left_bright_tab = (int16_t*)malloc((size_t)h * sizeof(int16_t));
                right_bright_tab = (int16_t*)malloc((size_t)h * sizeof(int16_t));
            }
            if (!left_edge || !right_edge_tab ||
                (build_floor_col_bounds && (!col_top_tab || !col_bot_tab)) ||
                (use_gour_floor && (!left_bright_tab || !right_bright_tab))) {
                free(left_edge);
                free(right_edge_tab);
                free(col_top_tab);
                free(col_bot_tab);
                free(left_bright_tab);
                free(right_bright_tab);
                break;
            }
            for (int i = 0; i < h; i++) {
                left_edge[i] = renderer_clamp_edge_x_i16(g_renderer.width);
                right_edge_tab[i] = -1;
                if (use_gour_floor) {
                    left_bright_tab[i] = 0;
                    right_bright_tab[i] = 0;
                }
            }
            if (build_floor_col_bounds) {
                for (int i = 0; i < g_renderer.width; i++) {
                    col_top_tab[i] = renderer_clamp_edge_y_i16(h);
                    col_bot_tab[i] = -1;
                }
            }
            int poly_top = h;
            int poly_bot = -1;

            /* Clamp Y range for floor vs ceiling. Multi-floor: also clamp to zone top_clip/bot_clip
             * so lower room does not draw above the split and upper room does not draw below it. */
            int y_min_clamp, y_max_clamp;
            int top_clip_for_poly = ctx->top_clip;
            if (entry_type == 8 || entry_type == 9) {
                top_clip_for_poly -= CHUNKY_TOPCLIP_BIAS;
            }
            if (floor_y_dist > 0) {
                y_min_clamp = half_h;       /* floor: center to bottom */
                y_max_clamp = h - 1;
            } else {
                y_min_clamp = 0;            /* ceiling: top to center */
                y_max_clamp = half_h - 1;
            }
            if (y_min_clamp < top_clip_for_poly) y_min_clamp = top_clip_for_poly;
            if (y_max_clamp > ctx->bot_clip) y_max_clamp = ctx->bot_clip;

            /* Walk each polygon edge and rasterize into edge tables (floor and ceiling/roof).
             * Near-plane clip edges so vertices behind the camera get proper
             * screen X values (otherwise on_screen[].screen_x is garbage and
             * the edge table doesn't reach the screen sides). */
            /* Water needs a tighter near clip than floor/roof so a surface just
             * above the camera can still project up to the top rows.
             * Z values are 24.8 fixed-point. */
            const int32_t FLOOR_NEAR = ROT_Z_FROM_INT(1);
            for (int s = 0; s < sides; s++) {
                int i1 = pt_indices[s];
                int i2 = pt_indices[(s + 1) % sides];
                if (i1 < 0 || i1 >= MAX_POINTS || i2 < 0 || i2 >= MAX_POINTS) continue;

                int32_t z1 = r->rotated[i1].z;
                int32_t z2 = r->rotated[i2].z;
                int32_t rx1 = r->rotated[i1].x;
                int32_t rx2 = r->rotated[i2].x;

                /* Skip edge entirely when both vertices are behind the near plane.
                 * Otherwise we would rasterize a fabricated edge at z=FLOOR_NEAR and extend the polygon incorrectly. */
                if (z1 < FLOOR_NEAR && z2 < FLOOR_NEAR)
                    continue;

                /* Near-plane clip: interpolate X at FLOOR_NEAR when vertex is behind camera.
                 * Clipped (ex,ez) are used for projection so we never divide by z < FLOOR_NEAR. */
                int32_t ez1 = z1, ez2 = z2;
                int32_t ex1 = rx1, ex2 = rx2;
                int32_t eb1 = 0, eb2 = 0;
                if (use_gour_floor) {
                    eb1 = pt_brights[s];
                    eb2 = pt_brights[(s + 1) % sides];
                }
                if (ez1 < FLOOR_NEAR) {
                    int32_t dz = ez2 - ez1;
                    if (dz != 0) {
                        int32_t t = (int32_t)((int64_t)(FLOOR_NEAR - ez1) * 65536 / dz);
                        ex1 = rx1 + (int32_t)((int64_t)(rx2 - rx1) * t / 65536);
                        if (use_gour_floor) {
                            eb1 = eb1 + (int32_t)((int64_t)(eb2 - eb1) * t / 65536);
                        }
                    } else {
                        ex1 = (rx1 + rx2) / 2;
                        if (use_gour_floor) eb1 = (eb1 + eb2) / 2;
                    }
                    ez1 = FLOOR_NEAR;
                }
                if (ez2 < FLOOR_NEAR) {
                    int32_t dz = ez1 - ez2;
                    if (dz != 0) {
                        int32_t t = (int32_t)((int64_t)(FLOOR_NEAR - ez2) * 65536 / dz);
                        ex2 = rx2 + (int32_t)((int64_t)(rx1 - rx2) * t / 65536);
                        if (use_gour_floor) {
                            eb2 = eb2 + (int32_t)((int64_t)(eb1 - eb2) * t / 65536);
                        }
                    } else {
                        ex2 = (rx1 + rx2) / 2;
                        if (use_gour_floor) eb2 = (eb1 + eb2) / 2;
                    }
                    ez2 = FLOOR_NEAR;
                }
                /* Project X from clipped (ex,ez) so values are consistent and safe (no division by zero or negative z). */
                int sx1 = project_x_to_pixels(ex1, ez1);
                int sx2 = project_x_to_pixels(ex2, ez2);

                /* Project Y: same rule so X and Y stay consistent (no jump when z crosses FLOOR_NEAR). */
                int32_t rel_h_8 = rel_h >> WORLD_Y_FRAC_BITS;
                int sy1_raw = project_y_to_pixels_round(rel_h_8, ez1, r->proj_y_scale, center);
                int sy2_raw = project_y_to_pixels_round(rel_h_8, ez2, r->proj_y_scale, center);

                if (build_floor_col_bounds) {
                    renderer_floor_column_bounds_add_edge(col_top_tab,
                                                          col_bot_tab,
                                                          ctx->left_clip,
                                                          ctx->right_clip - 1,
                                                          y_min_clamp,
                                                          y_max_clamp,
                                                          sx1,
                                                          sy1_raw,
                                                          sx2,
                                                          sy2_raw,
                                                          &col_x_min,
                                                          &col_x_max);
                }

                /* Clamp for horizontal edge check */
                int sy1 = sy1_raw;
                int sy2 = sy2_raw;
                if (sy1 < y_min_clamp) sy1 = y_min_clamp;
                if (sy1 > y_max_clamp) sy1 = y_max_clamp;
                if (sy2 < y_min_clamp) sy2 = y_min_clamp;
                if (sy2 > y_max_clamp) sy2 = y_max_clamp;

                /* DDA edge walk - use raw Y values to properly interpolate X
                 * across visible rows even when endpoints project off-screen */
                int dy_raw = sy2_raw - sy1_raw;
                if (dy_raw == 0) {
                    /* Truly horizontal edge in world space */
                    int row = sy1;
                    if (row >= y_min_clamp && row <= y_max_clamp) {
                        int lo = sx1 < sx2 ? sx1 : sx2;
                        int hi = sx1 > sx2 ? sx1 : sx2;
                        int32_t lo_b = (sx1 <= sx2) ? eb1 : eb2;
                        int32_t hi_b = (sx1 <= sx2) ? eb2 : eb1;
                        if (lo < left_edge[row]) {
                            left_edge[row] = renderer_clamp_edge_x_i16(lo);
                            if (use_gour_floor) left_bright_tab[row] = (int16_t)lo_b;
                        }
                        if (hi > right_edge_tab[row]) {
                            right_edge_tab[row] = renderer_clamp_edge_x_i16(hi);
                            if (use_gour_floor) right_bright_tab[row] = (int16_t)hi_b;
                        }
                        if (row < poly_top) poly_top = row;
                        if (row > poly_bot) poly_bot = row;
                    }
                    continue;
                }

                /* Walk visible rows, interpolating X using raw Y values */
                int row_start = (sy1_raw < sy2_raw) ? sy1_raw : sy2_raw;
                int row_end = (sy1_raw > sy2_raw) ? sy1_raw : sy2_raw;
                if (row_start < y_min_clamp) row_start = y_min_clamp;
                if (row_end > y_max_clamp) row_end = y_max_clamp;

                /* Use fixed-point for better precision */
                int64_t x_fp = (int64_t)sx1 << 16;
                int64_t dx_fp = ((int64_t)(sx2 - sx1) << 16) / dy_raw;
                int64_t b_fp = (int64_t)eb1 << 16;
                int64_t db_fp = ((int64_t)(eb2 - eb1) << 16) / dy_raw;
                
                /* Adjust starting position if row_start is clamped */
                if (sy1_raw < sy2_raw) {
                    x_fp += dx_fp * (row_start - sy1_raw);
                    b_fp += db_fp * (row_start - sy1_raw);
                } else {
                    x_fp = (int64_t)sx2 << 16;
                    dx_fp = ((int64_t)(sx1 - sx2) << 16) / (-dy_raw);
                    x_fp += dx_fp * (row_start - sy2_raw);
                    b_fp = (int64_t)eb2 << 16;
                    db_fp = ((int64_t)(eb1 - eb2) << 16) / (-dy_raw);
                    b_fp += db_fp * (row_start - sy2_raw);
                }
                
                for (int row = row_start; row <= row_end; row++) {
                    if (row < 0 || row >= h) { x_fp += dx_fp; b_fp += db_fp; continue; }
                    int32_t edge_bright = (int32_t)(b_fp >> 16);
                    int left_x = renderer_fp16_x_floor_px(x_fp);
                    int right_x = renderer_fp16_x_ceil_px(x_fp);
                    if (left_x < left_edge[row]) {
                        left_edge[row] = renderer_clamp_edge_x_i16(left_x);
                        if (use_gour_floor) left_bright_tab[row] = (int16_t)edge_bright;
                    }
                    if (right_x > right_edge_tab[row]) {
                        right_edge_tab[row] = renderer_clamp_edge_x_i16(right_x);
                        if (use_gour_floor) right_bright_tab[row] = (int16_t)edge_bright;
                    }
                    if (row < poly_top) poly_top = row;
                    if (row > poly_bot) poly_bot = row;
                    x_fp += dx_fp;
                    b_fp += db_fp;
                }
            }

            /* Clamp polygon bounds so row is always in [0, h-1] */
            if (poly_top < y_min_clamp) poly_top = y_min_clamp;
            if (poly_top >= h) poly_top = h - 1;
            if (poly_bot > y_max_clamp) poly_bot = y_max_clamp;
            if (poly_bot >= h) poly_bot = h - 1;
            if (poly_bot < 0) poly_bot = -1;


            /* Resolve floor texture/palette source.
             * Amiga uses separate paths for:
             *   1/2/7  -> FloorLine (floortile + FloorPalScaled)
             *   8/9    -> BumpLine chunky (BumpTile + BumpPalScaled)
             *   10/11  -> BumpLine smooth (SmoothBumpTile + SmoothBumpPalScaled) */
            const uint8_t *floor_tex = NULL;
            const uint8_t *floor_pal = r->floor_pal;
            if (entry_type == 8 || entry_type == 9) {
                if (r->bump_tile)
                    floor_tex = r->bump_tile + (uint16_t)whichtile;
                floor_pal = r->bump_pal ? r->bump_pal : r->floor_pal;
            } else if (entry_type == 10 || entry_type == 11) {
                floor_tex = r->smooth_bump_tile ? r->smooth_bump_tile : r->bump_tile;
                floor_pal = r->smooth_bump_pal ? r->smooth_bump_pal :
                            (r->bump_pal ? r->bump_pal : r->floor_pal);
            } else {
                if (r->floor_tile)
                    floor_tex = r->floor_tile + (uint16_t)whichtile;
            }

            /* Brightness: zone_bright + floor entry's brightness offset
             * ASM: move.w (a0)+,d6 / add.w ZoneBright,d6 */
            int16_t bright = zone_bright + floor_bright_off;
            RendererZoneTraceFloorStats floor_trace_stats;
            RendererZoneTraceFloorStats *saved_floor_trace_stats = ctx->active_floor_trace_stats;
            renderer_zone_trace_floor_stats_init(&floor_trace_stats,
                                                 trace_floor_zone && floor_y_dist > 0,
                                                 zone_id,
                                                 stream_entry_index,
                                                 entry_type,
                                                 (entry_type == 7) ? 1 : 0);
            renderer_zone_trace_floor_stats_accumulate_edges(&floor_trace_stats,
                                                             ctx,
                                                             left_edge,
                                                             right_edge_tab,
                                                             poly_top,
                                                             poly_bot);
            ctx->active_floor_trace_stats = floor_trace_stats.active ? &floor_trace_stats : NULL;

            int run_floor_col_fast_poly = allow_floor_col_fast_poly;
            if (run_floor_col_fast_poly && use_gour_floor &&
                left_bright_tab && right_bright_tab &&
                poly_top >= 0 && poly_top <= poly_bot) {
                int gour_flat = 1;
                for (int row = poly_top; row <= poly_bot; row++) {
                    if (left_edge[row] > right_edge_tab[row]) continue;
                    if (left_bright_tab[row] != right_bright_tab[row]) {
                        gour_flat = 0;
                        break;
                    }
                }
                run_floor_col_fast_poly = gour_flat;
            }

            if (run_floor_col_fast_poly && floor_tex && floor_pal) {
                if (ctx->profile_collect_stats) {
                    uint64_t floor_t0 = SDL_GetPerformanceCounter();
                    renderer_draw_floor_columns_ctx_fast(ctx, left_edge, right_edge_tab,
                                                         col_top_tab, col_bot_tab,
                                                         col_x_min, col_x_max,
                                                         left_bright_tab, right_bright_tab,
                                                         poly_top, poly_bot,
                                                         rel_h, floor_tex, floor_pal,
                                                         bright, use_gour_floor,
                                                         scaleval);
                    {
                        uint64_t floor_ticks = SDL_GetPerformanceCounter() - floor_t0;
                        ctx->workload_stats.ticks_floor += floor_ticks;
                    }
                } else {
                    renderer_draw_floor_columns_ctx_fast(ctx, left_edge, right_edge_tab,
                                                         col_top_tab, col_bot_tab,
                                                         col_x_min, col_x_max,
                                                         left_bright_tab, right_bright_tab,
                                                         poly_top, poly_bot,
                                                         rel_h, floor_tex, floor_pal,
                                                         bright, use_gour_floor,
                                                         scaleval);
                }
                renderer_zone_trace_floor_stats_finalize(&floor_trace_stats,
                                                         ctx,
                                                         left_edge,
                                                         right_edge_tab,
                                                         poly_top,
                                                         poly_bot);
                ctx->active_floor_trace_stats = saved_floor_trace_stats;
                renderer_zone_trace_floor_stats_log(&floor_trace_stats);
                free(left_edge);
                free(right_edge_tab);
                free(col_top_tab);
                free(col_bot_tab);
                free(left_bright_tab);
                free(right_bright_tab);
                break;
            }

            /* Fill between edges for each row (floor and ceiling/roof). Clamp to zone clip. */
            int row_start = poly_top;
            int row_end = poly_bot;
            int row_step = 1;
            if (entry_type == 7) {
                /* Water refraction samples offset rows:
                 * - floor half (row_dist>0): samples below -> draw top->bottom
                 * - ceiling half (row_dist<0): samples above -> draw bottom->top
                 * so we avoid sampling freshly written water rows in both cases. */
                if (floor_y_dist < 0) {
                    row_start = poly_bot;
                    row_end = poly_top;
                    row_step = -1;
                }
            }
            for (int row = row_start;
                 (row_step > 0) ? (row <= row_end) : (row >= row_end);
                 row += row_step) {
                if (row < 0 || row >= h) continue;
                if (row < ctx->top_clip || row > ctx->bot_clip) continue;  /* multi-floor: stay in zone band */
                int16_t le = left_edge[row];
                int16_t re = right_edge_tab[row];
                if (le >= g_renderer.width || re < 0) continue;
                /* Check whether any of this span falls within the column strip. */
                {
                    int16_t cle = (le < (int16_t)ctx->left_clip)  ? (int16_t)ctx->left_clip      : le;
                    int16_t cre = (re >= (int16_t)ctx->right_clip) ? (int16_t)(ctx->right_clip-1) : re;
                    if (cle > cre) continue;
                }
                /* le/re intentionally passed unclipped: renderer_draw_floor_span_ctx clips
                 * internally and needs the full polygon span width for correct Gouraud shading. */
                /* Water (entry_type 7) and floor drawn inline in stream order (Amiga itsafloordraw). */
                int16_t water_rows_left = 0;
                if (entry_type == 7) {
                    int bound = row_end;
                    if (bound < (int)ctx->top_clip) bound = (int)ctx->top_clip;
                    if (bound > (int)ctx->bot_clip) bound = (int)ctx->bot_clip;
                    int rows_left = (row_step > 0) ? (bound - row + 1) : (row - bound + 1);
                    if (rows_left < 0) rows_left = 0;
                    if (rows_left > 32767) rows_left = 32767;
                    water_rows_left = (int16_t)rows_left;
                }
                int16_t row_bright_l = bright;
                int16_t row_bright_r = bright;
                int16_t row_use_gour = 0;
                if (use_gour_floor) {
                    row_bright_l = left_bright_tab[row];
                    row_bright_r = right_bright_tab[row];
                    row_use_gour = 1;
                }
                if (ctx->profile_collect_stats) {
                    uint64_t floor_t0 = SDL_GetPerformanceCounter();
                    renderer_draw_floor_span_ctx(ctx, (int16_t)row, le, re,
                                                 le, re,
                                                 rel_h, floor_tex, floor_pal,
                                                 bright, row_bright_l, row_bright_r, row_use_gour,
                                                 scaleval, (entry_type == 7) ? 1 : 0,
                                                 water_rows_left);
                    {
                        uint64_t floor_ticks = SDL_GetPerformanceCounter() - floor_t0;
                        ctx->workload_stats.ticks_floor += floor_ticks;
                        if (entry_type == 7) ctx->workload_stats.ticks_water += floor_ticks;
                    }
                } else {
                    renderer_draw_floor_span_ctx(ctx, (int16_t)row, le, re,
                                                 le, re,
                                                 rel_h, floor_tex, floor_pal,
                                                 bright, row_bright_l, row_bright_r, row_use_gour,
                                                 scaleval, (entry_type == 7) ? 1 : 0,
                                                 water_rows_left);
                }
            }
            renderer_zone_trace_floor_stats_finalize(&floor_trace_stats,
                                                     ctx,
                                                     left_edge,
                                                     right_edge_tab,
                                                     poly_top,
                                                     poly_bot);
            ctx->active_floor_trace_stats = saved_floor_trace_stats;
            renderer_zone_trace_floor_stats_log(&floor_trace_stats);
            free(left_edge);
            free(right_edge_tab);
            free(col_top_tab);
            free(col_bot_tab);
            free(left_bright_tab);
            free(right_bright_tab);
            break;
        }

        case 3: /* Clip setter */
            /* No additional data consumed in polyloop
             * (clipping is done from ListOfGraphRooms, not here) */
            break;

        case 4: /* Object (sprite) */
        {
            /* Amiga ObjDraw clip mode:
             *   <1 = before water, ==1 = after water, >1 = full room. */
            int16_t obj_clip_mode = rd16(ptr);
            ptr += 2;

            int32_t obj_top = zone_roof;
            int32_t obj_bot = zone_floor;
            if (obj_clip_mode < 1) {
                obj_top = before_wat_top;
                obj_bot = before_wat_bot;
            } else if (obj_clip_mode == 1) {
                obj_top = after_wat_top;
                obj_bot = after_wat_bot;
            }
            if (obj_top > obj_bot) {
                int32_t t = obj_top;
                obj_top = obj_bot;
                obj_bot = t;
            }
            int32_t zone_upper_gfx = rd32(level->zone_graph_adds + zone_id * 8 + 4);
            int is_multi_floor = (zone_upper_gfx > 0) &&
                                 (level->graphics_byte_count == 0 ||
                                  ((size_t)zone_upper_gfx + 2u <= level->graphics_byte_count));
            int ignore_sky_top_clip = (zone_open_sky && obj_top == zone_roof) ? 1 : 0;
            draw_zone_objects_ctx(ctx, state, zone_id, obj_top, obj_bot,
                                  is_multi_floor ? use_upper : -1,
                                  ignore_sky_top_clip);
            break;
        }

        case 5: /* Arc (curved wall) */
        {
            /* Arc entry: 28 bytes of data after type word.
             * Translated from WallRoutine3.ChipMem.s CurveDraw (lines 498-538):
             *   center_pt(2) + edge_pt(2) + bitmap_start(2) + bitmap_end(2) +
             *   angle(2) + subdivide_idx(2) + walltiles_offset(4) +
             *   basebright(2) + brightmult(2) + topofwall(4) + botofwall(4) = 28
             */
            int16_t center_pt = rd16(ptr + 0);
            int16_t edge_pt   = rd16(ptr + 2);
            int16_t bmp_start = rd16(ptr + 4);
            int16_t bmp_end   = rd16(ptr + 6);
            /* int16_t arc_angle = rd16(ptr + 8); */
            int16_t subdiv_idx = rd16(ptr + 10);
            int32_t tex_offset = rd32(ptr + 12);
            int16_t base_bright = rd16(ptr + 16);
            /* int16_t bright_mult = rd16(ptr + 18); */
            int32_t topwall = rd32(ptr + 20);
            int32_t botwall = rd32(ptr + 24);
            
            topwall -= y_off;
            botwall -= y_off;
            
            /* Subdivide lookup: index -> (shift, count) */
            static const int subdiv_counts[] = { 4, 8, 16, 32, 64 };
            int num_segments = 4;
            if (subdiv_idx >= 0 && subdiv_idx < 5) {
                num_segments = subdiv_counts[subdiv_idx];
            }
            
            /* Get center and edge in rotated space */
            if (center_pt >= 0 && center_pt < MAX_POINTS &&
                edge_pt >= 0 && edge_pt < MAX_POINTS)
            {
                int32_t cx = r->rotated[center_pt].x;
                int32_t cz = r->rotated[center_pt].z;
                int32_t ex = r->rotated[edge_pt].x;
                int32_t ez = r->rotated[edge_pt].z;
                
                /* Compute radius vector from center to edge */
                int32_t dx = ex - cx;
                int32_t dz = ez - cz;
                
                /* Get texture - tex_offset is byte offset into walltiles array */
                int16_t tex_id = (int16_t)(tex_offset / 4096); /* Approximate: each texture is 4K */
                const uint8_t *arc_tex = (tex_id >= 0 && tex_id < MAX_WALL_TILES) ? r->walltiles[tex_id] : NULL;
                
                /* Draw arc as series of wall segments */
                int32_t prev_x = ex;
                int32_t prev_z = ez;
                int32_t prev_t = bmp_start;
                
                for (int seg = 1; seg <= num_segments; seg++) {
                    /* Compute angle for this segment (0 to 2*PI over num_segments) */
                    int angle = (seg * 1024) / num_segments; /* 0-1024 represents 0-360 degrees */

                    /* Use game sine table (4096 entries, byte-indexed 0-8191 for 360 deg).
                     * Map 0-1024 arc steps to 0-8192 byte-index: multiply by 8.
                     * Divide table value (range approx -32767..32767) by 128 to get -256..255 scale. */
                    int byte_angle = angle << 3;
                    int32_t s = sin_lookup(byte_angle) >> 7;
                    int32_t c = cos_lookup(byte_angle) >> 7;

                    /* Rotate radius vector by angle */
                    int32_t rx = (dx * c - dz * s) / 256;
                    int32_t rz = (dx * s + dz * c) / 256;
                    
                    int32_t new_x = (cx + rx);
                    int32_t new_z = cz + rz;
                    int32_t new_t = bmp_start + ((bmp_end - bmp_start) * seg / num_segments);
                    
                    /* Amiga stream order: draw arc segment immediately */
                    if (new_z > 0 && prev_z > 0) {
                        int16_t wall_ht = (int16_t)((botwall - topwall) >> 8);
                        if (wall_ht < 1) wall_ht = 1;
                        if (tex_id >= 0 && tex_id < MAX_WALL_TILES)
                            ctx->cur_wall_pal = r->wall_palettes[tex_id];
                        else
                            ctx->cur_wall_pal = NULL;
                        renderer_draw_wall_ctx(ctx, prev_x, prev_z,
                                               new_x, new_z,
                                               (int16_t)(topwall >> 8), (int16_t)(botwall >> 8),
                                               arc_tex, (int16_t)prev_t, (int16_t)new_t,
                                               base_bright, base_bright, 63, 6, 255,
                                               0, 0, tex_id, wall_ht, 64);
                    }
                    
                    prev_x = new_x;
                    prev_z = new_z;
                    prev_t = new_t;
                }
            }
            
            ptr += 28;
            break;
        }

        case 6: /* Light beam */
        {
            /* Light beams: 4 bytes of data after type word.
             * Translated from AB3DI.s LightDraw (lines 4364-4365):
             *   point1(2) + point2(2) = 4 bytes */
            ptr += 4;
            break;
        }

        case 12: /* Backdrop */
        {
            /* putinbackdrop - no extra stream data.
             * Sky synthesis is geometry-driven: add only where floor polygons have no matching roof. */
            break;
        }

        default:
            /* Unknown type - skip nothing (type word already consumed) */
            break;
        }
        stream_entry_index++;
        }

        if (fallback_object_pass) {
            int ignore_sky_top_clip = zone_open_sky;
            draw_zone_objects_ctx(ctx, state, zone_id, zone_roof, zone_floor,
                                  is_multi_floor_zone ? (use_upper ? 1 : 0) : -1,
                                  ignore_sky_top_clip);
        }

        ctx->update_column_clip = prev_update_column_clip;
    }
}

void renderer_draw_zone(GameState *state, int16_t zone_id, int use_upper)
{
    RenderSliceContext ctx;
    render_slice_context_init(&ctx, g_renderer.left_clip, g_renderer.right_clip,
                              g_renderer.top_clip, g_renderer.bot_clip);
    ctx.wall_top_clip = g_renderer.wall_top_clip;
    ctx.wall_bot_clip = g_renderer.wall_bot_clip;
    renderer_draw_zone_ctx(&ctx, state, zone_id, use_upper);
}

static int renderer_compute_zone_clip_span(GameState *state, int16_t zone_id,
                                           uint32_t frame_idx, int trace_clip,
                                           int16_t *out_left_px, int16_t *out_right_px)
{
    RendererState *r = &g_renderer;
    if (!state || !out_left_px || !out_right_px) return 0;

    const uint8_t *lgr = state->view_list_of_graph_rooms;
    if (!lgr || !state->level.clips) return 0;

    const uint8_t *lgr_entry = NULL;
    while (rd16(lgr) >= 0) {
        int16_t entry_zone = -1;
        if (renderer_resolve_lgr_entry_zone_id(&state->level, rd16(lgr), &entry_zone) &&
            entry_zone == zone_id) {
            lgr_entry = lgr;
            break;
        }
        lgr += 8;
    }

    if (!lgr_entry) {
        if (trace_clip) {
            printf("[CLIP][frame %u] zone=%d SKIP not_in_lgr\n",
                   (unsigned)frame_idx, (int)zone_id);
        }
        return 0;
    }

    int left_clip_px = 0;
    int right_clip_px = g_renderer.width;
    int num_level_points = MAX_POINTS;
    if (state->level.data) {
        int npts = rd16(state->level.data + 14);
        if (npts > 0 && npts < MAX_POINTS) {
            num_level_points = npts;
        }
    }
    size_t clip_word_count = 0;
    if (state->level.clips && state->level.connect_table &&
        state->level.connect_table >= state->level.clips) {
        clip_word_count = (size_t)(state->level.connect_table - state->level.clips) / 2u;
    }
    int16_t clip_off = rd16(lgr_entry + 2);
    if (trace_clip) {
        printf("[CLIP][frame %u] zone=%d clip_off=%d\n",
               (unsigned)frame_idx, (int)zone_id, (int)clip_off);
    }

    if (clip_off >= 0) {
        const uint8_t *connect_table = state->level.connect_table;
        if (clip_word_count > 0 && (size_t)clip_off >= clip_word_count) {
            if (trace_clip) {
                printf("[CLIP][frame %u] zone=%d clip_off=%d out_of_range max=%zu\n",
                       (unsigned)frame_idx, (int)zone_id, (int)clip_off, clip_word_count);
            }
            clip_off = -1;
        }
        const uint8_t *clip_ptr = (clip_off >= 0) ? (state->level.clips + clip_off * 2) : NULL;

        int invalid_clip = 0;
        int guard = 0;

        /* --- Left clip boundary points --- */
        while (clip_ptr && rd16(clip_ptr) >= 0) {
            if (++guard > 1024) { invalid_clip = 1; break; }
            if (clip_word_count > 0) {
                size_t clip_idx = (size_t)(clip_ptr - state->level.clips) / 2u;
                if (clip_idx >= clip_word_count) { invalid_clip = 1; break; }
            }
            int16_t pt = rd16(clip_ptr);
            clip_ptr += 2;
            if (pt < 0 || pt >= num_level_points) continue;
            if (!renderer_ensure_level_point_rotated(state, pt)) {
                if (trace_clip) {
                    printf("[CLIP][frame %u] zone=%d left pt=%d rotate_failed\n",
                           (unsigned)frame_idx, (int)zone_id, (int)pt);
                }
                continue;
            }
            int32_t pz = r->rotated[pt].z;
            if (pz > 0) {
                int sxpx = project_x_to_pixels(r->rotated[pt].x, pz);
                int allow = 1;
                if (connect_table) {
                    int16_t cpt = rd16(connect_table + (size_t)pt * 4u + 2u);
                    if (cpt >= 0 && cpt < num_level_points) {
                        if (!renderer_ensure_level_point_rotated(state, cpt)) {
                            if (trace_clip) {
                                printf("[CLIP][frame %u] zone=%d left pt=%d cpt=%d rotate_failed\n",
                                       (unsigned)frame_idx, (int)zone_id, (int)pt, (int)cpt);
                            }
                            allow = 0;
                        }
                    }
                    if (allow && cpt >= 0 && cpt < num_level_points) {
                        int csxpx = project_x_to_pixels(r->rotated[cpt].x, r->rotated[cpt].z);
                        if (csxpx > sxpx) allow = 0;
                    } else if (trace_clip) {
                        printf("[CLIP][frame %u] zone=%d left pt=%d bad cpt=%d\n",
                               (unsigned)frame_idx, (int)zone_id, (int)pt, (int)cpt);
                    }
                }
                if (allow && sxpx > left_clip_px) {
                    left_clip_px = sxpx;
                }
            } else if (connect_table) {
                /* Portal vertex behind camera — clip edge to near plane for a tighter bound. */
                int16_t cpt = rd16(connect_table + (size_t)pt * 4u + 2u);
                if (cpt >= 0 && cpt < num_level_points &&
                    renderer_ensure_level_point_rotated(state, cpt)) {
                    int32_t cz = r->rotated[cpt].z;
                    if (cz > 0) {
                        int32_t dz = cz - pz;
                        if (dz > 0) {
                            int32_t clip_x = r->rotated[pt].x +
                                (int32_t)((int64_t)(r->rotated[cpt].x - r->rotated[pt].x) * (ROT_Z_FROM_INT(1) - pz) / dz);
                            int sxpx = project_x_to_pixels(clip_x, ROT_Z_FROM_INT(1));
                            if (sxpx > left_clip_px)
                                left_clip_px = sxpx;
                        }
                    }
                }
            }
        }
        if (invalid_clip) {
            if (trace_clip) {
                printf("[CLIP][frame %u] zone=%d invalid left clip sequence; fallback full span\n",
                       (unsigned)frame_idx, (int)zone_id);
            }
            left_clip_px = 0;
            right_clip_px = g_renderer.width;
            clip_ptr = NULL;
        } else if (clip_ptr) {
            clip_ptr += 2;
        }

        /* --- Right clip boundary points --- */
        while (clip_ptr && rd16(clip_ptr) >= 0) {
            if (++guard > 2048) { invalid_clip = 1; break; }
            if (clip_word_count > 0) {
                size_t clip_idx = (size_t)(clip_ptr - state->level.clips) / 2u;
                if (clip_idx >= clip_word_count) { invalid_clip = 1; break; }
            }
            int16_t pt = rd16(clip_ptr);
            clip_ptr += 2;
            if (pt < 0 || pt >= num_level_points) continue;
            if (!renderer_ensure_level_point_rotated(state, pt)) {
                if (trace_clip) {
                    printf("[CLIP][frame %u] zone=%d right pt=%d rotate_failed\n",
                           (unsigned)frame_idx, (int)zone_id, (int)pt);
                }
                continue;
            }
            int32_t pz = r->rotated[pt].z;
            if (pz > 0) {
                int sxpx = project_x_to_pixels(r->rotated[pt].x, pz);
                int allow = 1;
                if (connect_table) {
                    int16_t cpt = rd16(connect_table + (size_t)pt * 4u);
                    if (cpt >= 0 && cpt < num_level_points) {
                        if (!renderer_ensure_level_point_rotated(state, cpt)) {
                            if (trace_clip) {
                                printf("[CLIP][frame %u] zone=%d right pt=%d cpt=%d rotate_failed\n",
                                       (unsigned)frame_idx, (int)zone_id, (int)pt, (int)cpt);
                            }
                            allow = 0;
                        }
                    }
                    if (allow && cpt >= 0 && cpt < num_level_points) {
                        int csxpx = project_x_to_pixels(r->rotated[cpt].x, r->rotated[cpt].z);
                        if (csxpx < sxpx) allow = 0;
                    } else if (trace_clip) {
                        printf("[CLIP][frame %u] zone=%d right pt=%d bad cpt=%d\n",
                               (unsigned)frame_idx, (int)zone_id, (int)pt, (int)cpt);
                    }
                }
                if (allow && sxpx < right_clip_px) {
                    /* Bounds are already in render pixels; keep right edge as a 1px-exclusive bound. */
                    right_clip_px = sxpx + 1;
                }
            } else if (connect_table) {
                /* Portal vertex behind camera — clip edge to near plane for a tighter bound. */
                int16_t cpt = rd16(connect_table + (size_t)pt * 4u);
                if (cpt >= 0 && cpt < num_level_points &&
                    renderer_ensure_level_point_rotated(state, cpt)) {
                    int32_t cz = r->rotated[cpt].z;
                    if (cz > 0) {
                        int32_t dz = cz - pz;
                        if (dz > 0) {
                            int32_t clip_x = r->rotated[pt].x +
                                (int32_t)((int64_t)(r->rotated[cpt].x - r->rotated[pt].x) * (ROT_Z_FROM_INT(1) - pz) / dz);
                            int sxpx = project_x_to_pixels(clip_x, ROT_Z_FROM_INT(1));
                            int sxpx_r = sxpx + 1;
                            if (sxpx_r < right_clip_px)
                                right_clip_px = sxpx_r;
                        }
                    }
                }
            }
        }
        if (invalid_clip && trace_clip) {
            printf("[CLIP][frame %u] zone=%d invalid right clip sequence; fallback full span\n",
                   (unsigned)frame_idx, (int)zone_id);
        }
        if (invalid_clip) {
            left_clip_px = 0;
            right_clip_px = g_renderer.width;
        }
    }

    if (left_clip_px >= g_renderer.width || right_clip_px <= 0 || left_clip_px >= right_clip_px) {
        if (trace_clip) {
            printf("[CLIP][frame %u] zone=%d SKIP invalid lpx=%d rpx=%d\n",
                   (unsigned)frame_idx, (int)zone_id, left_clip_px, right_clip_px);
        }
        return 0;
    }

    if (left_clip_px < 0) left_clip_px = 0;
    if (right_clip_px > g_renderer.width) right_clip_px = g_renderer.width;

    *out_left_px = (int16_t)left_clip_px;
    *out_right_px = (int16_t)right_clip_px;

    if (trace_clip) {
        printf("[CLIP][frame %u] zone=%d prepass_clip_px=[%d,%d)\n",
               (unsigned)frame_idx, (int)zone_id, left_clip_px, right_clip_px);
    }
    return 1;
}

static void renderer_build_world_zone_prepass(GameState *state, uint32_t frame_idx,
                                              int trace_clip, RendererWorldZonePrepass *out)
{
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!state) return;

    PlayerState *view_plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;

    int count = state->zone_order_count;
    if (count < 0) count = 0;
    if (count > RENDERER_MAX_ZONE_ORDER) count = RENDERER_MAX_ZONE_ORDER;
    out->count = count;

    for (int i = 0; i < count; i++) {
        int16_t zone_id = state->zone_order_zones[i];
        out->zone_ids[i] = zone_id;
        if (zone_id < 0) continue;

        int16_t left_px = 0;
        int16_t right_px = (int16_t)g_renderer.width;
        if (renderer_compute_zone_clip_span(state, zone_id, frame_idx, trace_clip,
                                            &left_px, &right_px)) {
            RendererZoneSectionClip lower_clip;
            RendererZoneSectionClip upper_clip;
            RendererZoneSectionClip full_clip = renderer_make_zone_section_clip_full(g_renderer.height);
            uint8_t draw_upper_first = 0;
            renderer_build_zone_section_clips(state, zone_id,
                                              &lower_clip, &upper_clip,
                                              &draw_upper_first);

            {
                int clip_width = (int)right_px - (int)left_px;
                int disable_flat_section_clip = 0;
                if (zone_id == view_plr->zone) {
                    disable_flat_section_clip = 1;
                } else if (clip_width > ((g_renderer.width * 3) / 4)) {
                    disable_flat_section_clip = 1;
                }
                if (disable_flat_section_clip) {
                    if (lower_clip.valid) lower_clip = full_clip;
                    if (upper_clip.valid) upper_clip = full_clip;
                }
            }

            out->lower_clip[i] = lower_clip;
            out->upper_clip[i] = upper_clip;
            out->draw_upper_first[i] = draw_upper_first;
            if (lower_clip.valid || upper_clip.valid) {
                out->valid[i] = 1;
                out->left_px[i] = left_px;
                out->right_px[i] = right_px;
            }
        }
    }
}

static void renderer_draw_world_slice(GameState *state,
                                      const RendererWorldZonePrepass *zone_prepass,
                                      int16_t col_start, int16_t col_end,
                                      uint32_t frame_idx, int trace_clip,
                                      int8_t *out_fill_screen_water,
                                      RendererWorkloadStats *out_workload_stats)
{
    RenderSliceContext frame_ctx;
    int16_t *viewer_floor_occlude_top = NULL;
    int have_viewer_floor_occlude = 0;
    if (out_workload_stats) renderer_workload_stats_reset(out_workload_stats);
    if (!state || !zone_prepass || col_start >= col_end) {
        if (out_fill_screen_water) *out_fill_screen_water = 0;
        return;
    }

    int w = g_renderer.width;
    int h = g_renderer.height;
    if (w <= 0 || h <= 0) {
        if (out_fill_screen_water) *out_fill_screen_water = 0;
        return;
    }

    int cs = (int)col_start;
    int ce = (int)col_end;
    if (cs < 0) cs = 0;
    if (ce > w) ce = w;
    if (cs >= ce) {
        if (out_fill_screen_water) *out_fill_screen_water = 0;
        return;
    }

    int16_t strip_top = 0;
    int16_t strip_bot = (int16_t)(h - 1);

    render_slice_context_init(&frame_ctx, (int16_t)cs, (int16_t)ce, strip_top, strip_bot);
    frame_ctx.zone_prepass = zone_prepass;
    /* Column strips assign disjoint x ranges to workers; single-threaded full-width is one worker. */
    frame_ctx.update_column_clip = 1;
    PlayerState *plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
    const int trace_zone = (g_renderer_zone_trace_active && cs == 0 && ce == w) ? 1 : 0;

    viewer_floor_occlude_top = (int16_t *)malloc((size_t)w * sizeof(*viewer_floor_occlude_top));
    if (viewer_floor_occlude_top) {
        have_viewer_floor_occlude = renderer_build_viewer_split_floor_occluder(state, plr->zone,
                                                                               (int16_t)cs, (int16_t)ce,
                                                                               viewer_floor_occlude_top);
    }

    int zone_count = zone_prepass->count;
    if (zone_count < 0) zone_count = 0;
    if (zone_count > RENDERER_MAX_ZONE_ORDER) zone_count = RENDERER_MAX_ZONE_ORDER;

    for (int i = zone_count - 1; i >= 0; i--) {
        int16_t zone_id = zone_prepass->zone_ids[i];
        if (zone_id < 0 || !zone_prepass->valid[i]) continue;
        {
            int zs = level_zone_slot_count(&state->level);
            if (zone_id >= zs) continue;
            if (state->level.num_zone_graph_entries > 0 && zone_id >= state->level.num_zone_graph_entries)
                continue;
        }

        int left_clip_px = (int)zone_prepass->left_px[i];
        int right_clip_px = (int)zone_prepass->right_px[i];
        if (left_clip_px < 0) left_clip_px = 0;
        if (right_clip_px > w) right_clip_px = w;
        if (left_clip_px < cs) left_clip_px = cs;
        if (right_clip_px > ce) right_clip_px = ce;
        if (left_clip_px >= right_clip_px) {
            continue;
        }

        render_slice_context_reset(&frame_ctx, (int16_t)left_clip_px, (int16_t)right_clip_px,
                                   strip_top, strip_bot);

        frame_ctx.left_clip = (int16_t)left_clip_px;
        frame_ctx.right_clip = (int16_t)right_clip_px;
        frame_ctx.foreground_floor_occlude_top =
            (have_viewer_floor_occlude && zone_id != plr->zone) ? viewer_floor_occlude_top : NULL;
        if (trace_clip) {
            printf("[CLIP][frame %u] zone=%d slice_clip_px=[%d,%d)\n",
                   (unsigned)frame_idx, (int)zone_id, left_clip_px, right_clip_px);
        }

        RendererWorkloadStats zone_before_stats;
        RendererWorkloadStats zone_delta_stats;
        uint64_t zone_t0 = 0;
        int zone_drew_upper = 0;
        int zone_drew_lower = 0;
        int zone_upper_first = 0;
        if (trace_zone) {
            zone_before_stats = frame_ctx.workload_stats;
            zone_t0 = SDL_GetPerformanceCounter();
        }

        {
            const RendererZoneSectionClip *lower_clip = &zone_prepass->lower_clip[i];
            const RendererZoneSectionClip *upper_clip = &zone_prepass->upper_clip[i];
            zone_upper_first = zone_prepass->draw_upper_first[i] ? 1 : 0;

            if (zone_upper_first) {
                if (upper_clip->valid) {
                    renderer_apply_zone_section_clip(&frame_ctx, upper_clip);
                    renderer_draw_zone_ctx(&frame_ctx, state, zone_id, 1);
                    zone_drew_upper = 1;
                }
                if (lower_clip->valid) {
                    renderer_apply_zone_section_clip(&frame_ctx, lower_clip);
                    renderer_draw_zone_ctx(&frame_ctx, state, zone_id, 0);
                    zone_drew_lower = 1;
                }
            } else {
                if (lower_clip->valid) {
                    renderer_apply_zone_section_clip(&frame_ctx, lower_clip);
                    renderer_draw_zone_ctx(&frame_ctx, state, zone_id, 0);
                    zone_drew_lower = 1;
                }
                if (upper_clip->valid) {
                    renderer_apply_zone_section_clip(&frame_ctx, upper_clip);
                    renderer_draw_zone_ctx(&frame_ctx, state, zone_id, 1);
                    zone_drew_upper = 1;
                }
            }
        }
        if (!zone_drew_lower && !zone_drew_upper) {
            continue;
        }
        if (trace_zone) {
            renderer_workload_stats_diff(&zone_delta_stats, &frame_ctx.workload_stats, &zone_before_stats);
            renderer_log_world_zone_draw(frame_idx, i, zone_id,
                                         left_clip_px, right_clip_px,
                                         zone_drew_lower, zone_drew_upper, zone_upper_first,
                                         SDL_GetPerformanceCounter() - zone_t0,
                                         &zone_delta_stats);
        }
    }

    if (trace_zone) fflush(stdout);

    if (out_fill_screen_water) *out_fill_screen_water = frame_ctx.fill_screen_water;
    if (out_workload_stats) *out_workload_stats = frame_ctx.workload_stats;
    free(viewer_floor_occlude_top);
}

static void renderer_apply_underwater_tint_slice(int8_t fill_screen_water,
                                                  int16_t row_start, int16_t row_end,
                                                  int16_t col_start, int16_t col_end,
                                                  const uint32_t *src_rgb, const uint16_t *src_cw,
                                                  uint32_t *dst_rgb, uint16_t *dst_cw)
{
    if (fill_screen_water == 0) return;
    if (!src_rgb || !src_cw || !dst_rgb || !dst_cw) return;

    const int w = g_renderer.width;
    const int h = g_renderer.height;
    if (w <= 0 || h <= 0) return;

    int x0 = (int)col_start;
    int x1 = (int)col_end;
    if (x0 < 0) x0 = 0;
    if (x1 > w) x1 = w;
    if (x0 >= x1) return;

    int y0 = row_start;
    int y1 = row_end;
    if (y0 < 0) y0 = 0;
    if (y1 > h) y1 = h;
    if (y0 >= y1) return;

    /* AB3DI fillscrnwater post-pass:
     * AND #$00FF on copper color words. Strong applies to whole view (4*20 lines),
     * weak applies to bottom half only (2*20 lines). */
    const int strong = (fill_screen_water > 0);
    int tint_start = strong ? 0 : (h / 2);
    if (y0 < tint_start) y0 = tint_start;

    for (int y = y0; y < y1; y++) {
        size_t row = (size_t)y * (size_t)w;
        for (int x = x0; x < x1; x++) {
            size_t i = row + (size_t)x;
            uint16_t c12 = (uint16_t)(renderer_cw_load_xy(src_cw, x, y, w, h) & 0x00FFu);
            renderer_cw_store_xy(dst_cw, x, y, w, h, c12);
            if (g_renderer_rgb_raster_expand)
                dst_rgb[i] = amiga12_to_argb(c12);
        }
    }
}

static void renderer_apply_underwater_tint(int8_t fill_screen_water)
{
    if (!g_renderer.rgb_buffer || !g_renderer.cw_buffer) return;
    renderer_apply_underwater_tint_slice(fill_screen_water,
                                         0, (int16_t)g_renderer.height,
                                         0, (int16_t)g_renderer.width,
                                         g_renderer.rgb_buffer, g_renderer.cw_buffer,
                                         g_renderer.rgb_buffer, g_renderer.cw_buffer);
}

/* -----------------------------------------------------------------------
 * DrawDisplay - Main rendering entry point
 *
 * Translated from AB3DI.s DrawDisplay (lines 3395-3693).
 *
 * This is called once per frame to render the entire 3D scene.
 * ----------------------------------------------------------------------- */
void renderer_draw_display(GameState *state)
{
    RendererState *r = &g_renderer;
    int pick_this_frame = (g_pick_capture_armed || g_debug_spill_visualize) ? 1 : 0;
    int zone_trace = 0;
    int trace_clip = 0;
    g_pick_capture_armed = 0;
    g_pick_capture_active = pick_this_frame;
    if (pick_this_frame) {
        renderer_f2_pick_snapshot_clear(&g_renderer_f2_pick_snapshot);
    }
    if (!r->buffer) {
        g_renderer_zone_trace_active = 0;
        g_renderer_profile_collect_stats = 0;
        return;
    }
    if (!state) {
        g_renderer_zone_trace_active = 0;
        g_renderer_profile_collect_stats = 0;
        return;
    }
    int prof_on = renderer_profile_env_enabled() ? renderer_profile_enabled() : 0;
    uint64_t frame_perf_freq = 0;
    uint64_t t0 = 0;
    uint64_t t_after_setup = 0;
    uint64_t t_after_world = 0;
    uint64_t t_after_tint = 0;
    uint64_t t_after_gun = 0;
    uint64_t t_after_swap = 0;
    int world_workers = 0;
    int tint_workers = 0;
    int prepass_total_zones = 0;
    int prepass_valid_zones = 0;
    int prepass_clip_pixels = 0;
    int prepass_draw_zones = 0;
    int prepass_draw_lower = 0;
    int prepass_draw_upper = 0;
    int prepass_draw_order_count = 0;
    int16_t prepass_draw_order[RENDERER_MAX_ZONE_ORDER];
    RendererWorkloadStats world_workload_stats;
    renderer_workload_stats_reset(&world_workload_stats);
    if (prof_on || pick_this_frame) {
        frame_perf_freq = SDL_GetPerformanceFrequency();
        if (frame_perf_freq == 0) frame_perf_freq = 1;
        t0 = SDL_GetPerformanceCounter();
    }
    uint32_t frame_idx = g_render_frame_counter++;
    g_renderer_zone_trace_active = 0;
    g_renderer_profile_collect_stats = (prof_on || pick_this_frame) ? 1 : 0;

    /* 1. Projection setup.
     * Horizontal projection uses a normalized logical width so supersampling changes
     * detail only, and UHD (same aspect) keeps the same on-screen geometry. */
    int w = (r->width  > 0) ? r->width  : 1;
    int h = (r->height > 0) ? r->height : 1;
    if (state) g_proj_base_width = renderer_proj_effective_base_width_from_state(state);
    else g_proj_base_width = renderer_clamp_base_width(RENDER_DEFAULT_WIDTH);
#ifndef AB3D_NO_THREADS
    if (state) {
        int n = (int)state->cfg_render_threads_max;
        if (n < 0) n = 0;
        if (n > RENDERER_MAX_THREADS) n = RENDERER_MAX_THREADS;
        g_renderer_thread_max_workers = n;
    } else {
        g_renderer_thread_max_workers = 0;
    }
#endif

    /* Vertical scale per frame: same as (float)base / (640/h) but integer-only (avoid float
     * rounding that could make 100% vs 150% collapse to the same proj_y_scale after multiply). */
    {
        int32_t base_py = (int32_t)((int64_t)PROJ_Y_NUMERATOR / (int64_t)PROJ_Y_DENOM);
        r->proj_y_scale = (int32_t)(((int64_t)base_py * (int64_t)h) / 640);
        if (r->proj_y_scale < 1)
            r->proj_y_scale = 1;
    }

    /* ab3d.ini y_proj_scale: percent of computed proj_y_scale (100 = unchanged).
     * Floor/ceiling UV clamps must scale the same way: raw dist grows with proj_y_scale,
     * but pastfloorbright used fixed 30000/32000 caps tuned for 100% — without scaling
     * those caps, values over 100% saturate dist and break floor/ceiling perspective. */
    {
        int ypct = 100;
        if (state) {
            ypct = (int)state->cfg_y_proj_scale;
            if (ypct < 1) ypct = 1;
            r->proj_y_scale = (int32_t)(((int64_t)r->proj_y_scale * (int64_t)ypct + 50) / 100);
            if (r->proj_y_scale < 1)
                r->proj_y_scale = 1;
        }
        r->floor_uv_dist_max = (int32_t)(((int64_t)30000 * (int64_t)ypct + 50) / 100);
        r->floor_uv_dist_near = (int32_t)(((int64_t)32000 * (int64_t)ypct + 50) / 100);
    }
    renderer_floor_prepare_row_recip_table(h);

    /* 2. Setup view transform (from AB3DI.s DrawDisplay lines 3399-3438) */
    PlayerState *plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
    PlayerState *plr_sky = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
    int16_t sky_angpos = (int16_t)plr_sky->angpos;
    r->sky_frame_angpos = sky_angpos;

    int16_t ang = (int16_t)(plr->angpos & 0x3FFF); /* 14-bit angle */
    r->sinval = sin_lookup(ang);
    r->cosval = cos_lookup(ang);

    /* Extract integer part of 16.16 fixed-point position for rendering.
     * On Amiga: .w operations on big-endian 32-bit values read the high word. */
    r->xoff = (int16_t)(plr->xoff >> 16);
    r->zoff = (int16_t)(plr->zoff >> 16);
    r->yoff = plr->yoff;

    /* wallyoff = (yoff >> 8) + 224, masked to 0-255 */
    int32_t y_shifted = r->yoff >> 8;
    r->wallyoff = (int16_t)((y_shifted + 256 - 32) & 255);
    r->flooryoff = (int16_t)(y_shifted << 2);

    /* xoff34 = xoff * 3/4, zoff34 = zoff * 3/4 */
    r->xoff34 = (int16_t)((r->xoff * 3) >> 2);
    r->zoff34 = (int16_t)((r->zoff * 3) >> 2);

    /* xwobble from head bob */
    r->xwobble = 0; /* Would be set from plr->bob_frame */

    /* 3. Initialize column clipping (per-column top/bot/z for floor and sprite clipping) */
    {
        int16_t invalid_top = (int16_t)h;
        int16_t invalid_bot = -1;
        if (r->clip.top && r->clip.bot && r->clip.top2 && r->clip.bot2) {
            for (int i = 0; i < w; i++) {
                r->clip.top[i] = invalid_top;
                r->clip.bot[i] = invalid_bot;
                r->clip.top2[i] = invalid_top;
                r->clip.bot2[i] = invalid_bot;
            }
        }
        if (r->clip.z) memset(r->clip.z, 0, (size_t)w * sizeof(int32_t));
        if (r->clip.z2) memset(r->clip.z2, 0, (size_t)w * sizeof(int32_t));
    }
    if (pick_this_frame) {
        renderer_pick_clear_active_buffers();
    }
    if (g_debug_spill_visualize) {
        renderer_debug_spill_reset_sprite_rects();
    }

    {
        const int have_zone_state = (state && (state->zone_order_count > 0 || state->view_list_of_graph_rooms != NULL)) ? 1 : 0;
        const int have_zone_trace_state = (state && state->zone_order_count > 0 && state->view_list_of_graph_rooms != NULL) ? 1 : 0;
        const int trace_filters_match = have_zone_state ? renderer_trace_filters_match(state) : 0;
        if (have_zone_trace_state && g_renderer_zone_trace_requested) {
            zone_trace = 1;
            g_renderer_zone_trace_requested = 0;
            g_renderer_zone_trace_active = 1;
            g_renderer_zone_trace_frame_idx = frame_idx;
        } else if (have_zone_trace_state && trace_filters_match && renderer_peek_zone_trace_slot()) {
            zone_trace = 1;
            renderer_consume_zone_trace_slot();
            g_renderer_zone_trace_active = 1;
            g_renderer_zone_trace_frame_idx = frame_idx;
        }
        if (have_zone_trace_state && trace_filters_match && renderer_peek_clip_trace_slot()) {
            trace_clip = 1;
            renderer_consume_clip_trace_slot();
        }
        g_renderer_profile_collect_stats = (prof_on || zone_trace || pick_this_frame) ? 1 : 0;
        if (trace_clip) {
            printf("[CLIP][frame %u] begin\n", (unsigned)frame_idx);
        }
    }

    /* 4. Rotate geometry */
    renderer_rotate_level_pts(state);
    renderer_rotate_object_pts(state);
    RendererWorldZonePrepass world_zone_prepass;
    renderer_build_world_zone_prepass(state, frame_idx, trace_clip, &world_zone_prepass);
    prepass_total_zones = world_zone_prepass.count;
    if (prepass_total_zones < 0) prepass_total_zones = 0;
    if (prepass_total_zones > RENDERER_MAX_ZONE_ORDER) prepass_total_zones = RENDERER_MAX_ZONE_ORDER;
    for (int i = 0; i < prepass_total_zones; i++) {
        if (!world_zone_prepass.valid[i]) continue;
        prepass_valid_zones++;
        if (world_zone_prepass.lower_clip[i].valid) prepass_draw_lower++;
        if (world_zone_prepass.upper_clip[i].valid) prepass_draw_upper++;
        if (world_zone_prepass.lower_clip[i].valid || world_zone_prepass.upper_clip[i].valid) {
            prepass_draw_zones++;
            if (prepass_draw_order_count < RENDERER_MAX_ZONE_ORDER) {
                prepass_draw_order[prepass_draw_order_count++] = world_zone_prepass.zone_ids[i];
            }
        }
        {
            int wclip = (int)world_zone_prepass.right_px[i] - (int)world_zone_prepass.left_px[i];
            if (wclip > 0) prepass_clip_pixels += wclip;
        }
    }
    if (zone_trace) {
        renderer_log_world_zone_prepass(state, &world_zone_prepass, frame_idx);
    }
    if (prof_on) t_after_setup = SDL_GetPerformanceCounter();

    int8_t fill_screen_water = 0;
    int used_threaded_world = 0;
#ifndef AB3D_NO_THREADS
    if (state->cfg_render_threads && !zone_trace && !pick_this_frame) {
        used_threaded_world = renderer_dispatch_threaded_world(state, &world_zone_prepass,
                                                               frame_idx, trace_clip,
                                                               &fill_screen_water,
                                                               &world_workload_stats);
        if (!used_threaded_world) {
            static int s_thread_unavailable_logged = 0;
            if (!s_thread_unavailable_logged) {
                printf("[RENDERER] threading requested but unavailable (cpu_count=%d, workers=%d)\n",
                       g_renderer_thread_pool.cpu_count, g_renderer_thread_pool.worker_count);
                s_thread_unavailable_logged = 1;
            }
        }
    }
#endif
    if (used_threaded_world) {
#ifndef AB3D_NO_THREADS
        world_workers = (g_prof_last_world_workers > 0) ? g_prof_last_world_workers : 0;
#endif
    }
    if (!used_threaded_world) {
        renderer_clear(0);
        renderer_draw_world_slice(state, &world_zone_prepass,
                                  0, (int16_t)g_renderer.width,
                                  frame_idx, trace_clip,
                                  &fill_screen_water, &world_workload_stats);
        world_workers = 1;
    }
    if (prof_on) t_after_world = SDL_GetPerformanceCounter();

    int8_t tint_water = fill_screen_water;
    if (state && !state->cfg_post_tint)
        tint_water = 0;

    /* Save tint value for the GL post-pass (read by display.c after this call). */
    g_renderer.last_fill_screen_water = tint_water;

    /* 6. Underwater fillscrnwater post-pass — skipped when GL multiply handles it (or weapon GL). */
    int used_threaded_tint = 0;
    if (!s_weapon_post_gl_active && !s_gl_water_tint_post_active) {
#ifndef AB3D_NO_THREADS
        if (state->cfg_render_threads) {
            used_threaded_tint = renderer_dispatch_threaded_underwater_tint(tint_water);
        }
#endif
        if (!used_threaded_tint) {
            renderer_apply_underwater_tint(tint_water);
            tint_workers = (tint_water != 0) ? 1 : 0;
        } else {
#ifndef AB3D_NO_THREADS
            tint_workers = (g_prof_last_tint_workers > 0) ? g_prof_last_tint_workers : 0;
#endif
        }
    }
    if (prof_on) t_after_tint = SDL_GetPerformanceCounter();

    if (g_debug_spill_visualize) {
        renderer_apply_spill_visualize_debug_overlay(&world_zone_prepass, frame_idx);
    }

    /* 7. Draw gun overlay — skipped when the GL path will handle it.
     * In threaded-world mode the gun is already drawn per worker column strip. */
    if (!used_threaded_world && state->cfg_weapon_draw && !s_weapon_post_gl_active) {
        renderer_draw_gun(state);
    }
    if (prof_on) t_after_gun = SDL_GetPerformanceCounter();

    /* 8. Swap buffers (the just-drawn buffer becomes the display buffer) */
    /* Flush non-temporal store write-combine buffers before the display path reads cw_buffer. */
    AB3D_NT_STORE_FENCE();
    renderer_swap();
    g_pick_last_frame_valid = pick_this_frame;
    g_pick_capture_active = 0;

    uint64_t frame_end_counter = 0;
    double frame_ms = 0.0;
    if (prof_on || pick_this_frame) {
        frame_end_counter = SDL_GetPerformanceCounter();
        if (t0 > 0 && frame_end_counter >= t0) {
            frame_ms = ((double)(frame_end_counter - t0) * 1000.0) / (double)frame_perf_freq;
        }
    }

    if (pick_this_frame) {
        g_renderer_f2_pick_snapshot.valid = 1;
        g_renderer_f2_pick_snapshot.frame_idx = frame_idx;
        g_renderer_f2_pick_snapshot.width = w;
        g_renderer_f2_pick_snapshot.height = h;
        g_renderer_f2_pick_snapshot.frame_ms = frame_ms;
        g_renderer_f2_pick_snapshot.prepass_total_zones = prepass_total_zones;
        g_renderer_f2_pick_snapshot.prepass_valid_zones = prepass_valid_zones;
        g_renderer_f2_pick_snapshot.zones_drawn = prepass_draw_zones;
        g_renderer_f2_pick_snapshot.zones_drawn_lower = prepass_draw_lower;
        g_renderer_f2_pick_snapshot.zones_drawn_upper = prepass_draw_upper;
        g_renderer_f2_pick_snapshot.zone_draw_order_count = prepass_draw_order_count;
        if (prepass_draw_order_count > 0) {
            memcpy(g_renderer_f2_pick_snapshot.zone_draw_order,
                   prepass_draw_order,
                   (size_t)prepass_draw_order_count * sizeof(int16_t));
        }
        g_renderer_f2_pick_snapshot.workload = world_workload_stats;
    }

    /* Automap is drawn via SDL in display.c after the frame is composited. */
    if (prof_on) {
        RendererProfileState *ps = &g_renderer_profile;
        t_after_swap = frame_end_counter ? frame_end_counter : SDL_GetPerformanceCounter();
        ps->frames++;
        if (used_threaded_world) ps->threaded_world_frames++;
        if (used_threaded_tint) ps->threaded_tint_frames++;
        if (world_workers > 0) {
            ps->world_workers_total += (uint64_t)world_workers;
            ps->world_workers_samples++;
        }
        if (tint_workers > 0) {
            ps->tint_workers_total += (uint64_t)tint_workers;
            ps->tint_workers_samples++;
        }
        ps->ticks_total += (t_after_swap - t0);
        ps->ticks_setup_prepass += (t_after_setup - t0);
        ps->ticks_world += (t_after_world - t_after_setup);
        ps->ticks_tint += (t_after_tint - t_after_world);
        ps->ticks_gun += (t_after_gun - t_after_tint);
        ps->ticks_swap += (t_after_swap - t_after_gun);
        ps->prepass_zone_total += (uint64_t)prepass_total_zones;
        ps->prepass_zone_valid_total += (uint64_t)prepass_valid_zones;
        ps->prepass_clip_pixels_total += (uint64_t)prepass_clip_pixels;
        if (fill_screen_water > 0) ps->fill_water_strong_frames++;
        else if (fill_screen_water < 0) ps->fill_water_weak_frames++;
        ps->screen_pixels_total += (uint64_t)w * (uint64_t)h;
        renderer_workload_stats_add(&ps->workload_totals, &world_workload_stats);

        {
            double frame_ms = ((double)(t_after_swap - t0) * 1000.0) / (double)ps->perf_freq;
            if (ps->slow_threshold_ms > 0.0 && frame_ms >= ps->slow_threshold_ms) {
                ps->slow_frames++;
            }
            if (frame_ms > ps->worst_total_ms) {
                ps->worst_total_ms = frame_ms;
                ps->worst_frame_idx = frame_idx;
                ps->worst_setup_ms = ((double)(t_after_setup - t0) * 1000.0) / (double)ps->perf_freq;
                ps->worst_world_ms = ((double)(t_after_world - t_after_setup) * 1000.0) / (double)ps->perf_freq;
                ps->worst_tint_ms = ((double)(t_after_tint - t_after_world) * 1000.0) / (double)ps->perf_freq;
                ps->worst_gun_ms = ((double)(t_after_gun - t_after_tint) * 1000.0) / (double)ps->perf_freq;
                ps->worst_swap_ms = ((double)(t_after_swap - t_after_gun) * 1000.0) / (double)ps->perf_freq;
                ps->worst_prepass_total = prepass_total_zones;
                ps->worst_prepass_valid = prepass_valid_zones;
                ps->worst_prepass_clip_pixels = prepass_clip_pixels;
                ps->worst_world_workers = world_workers;
                ps->worst_tint_workers = tint_workers;
                ps->worst_fill_screen_water = fill_screen_water;
                ps->worst_screen_pixels = (uint64_t)w * (uint64_t)h;
                ps->worst_workload = world_workload_stats;
            }
        }
        renderer_profile_maybe_report(t_after_swap);
    }
    g_renderer_profile_collect_stats = 0;
    g_renderer_zone_trace_active = 0;
}
