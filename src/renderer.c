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
 *   3. RotateLevelPts: transform level vertices to view space
 *   4. RotateObjectPts: transform object positions to view space
 *   5. For each zone (back-to-front from OrderZones):
 *      a. Set left/right clip from LEVELCLIPS
 *      b. Determine split (upper/lower room)
 *      c. DoThisRoom: iterate zone graph data, dispatch:
 *         - Walls  -> column-by-column textured drawing
 *         - Floors -> span-based textured drawing
 *         - Objects -> scaled sprite drawing
 *   6. Draw gun overlay
 *   7. Swap buffers
 */

#include "renderer.h"
#include "renderer_3dobj.h"
#include "level.h"
#include "math_tables.h"
#include "game_data.h"
#include "game_types.h"
#include "visibility.h"
#include "audio.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

/* Floor/ceiling UV step per pixel: d1>>FLOOR_STEP_SHIFT (same at any width so texture scale is correct). */
#define FLOOR_STEP_SHIFT  (6 + RENDER_SCALE_LOG2)  /* d1>>9 at RENDER_SCALE=8 */
/* Camera UV scale in fixed-point, matching Amiga sxoff/szoff setup in pastsides:
 * xoff/zoff words are promoted to 16.16 before entering pastfloorbright. */
#define FLOOR_CAM_UV_SCALE  65536  /* 1<<16 */
/* Strict Amiga parity: no extra edge expansion beyond geometry rasterization. */
#define FLOOR_EDGE_EXTRA  0
#define CEILING_EDGE_EXTRA 0
#define PORTAL_EDGE_EXTRA 0
/* AB3DI itsachunkyfloor does `sub.w #12,topclip` before itsafloordraw.
 * Scale to current render resolution. */
#define CHUNKY_TOPCLIP_BIAS  (12 * RENDER_SCALE)
/* Minimum z in view space; vertices behind this are clipped. Used for walls and floor polygons. */
#define RENDERER_NEAR_PLANE 4
/* Amiga ObjDraw3 BitMapObj/PolygonObj: cmp.w #50,d1 ; ble objbehind.
 * Keep the same near cutoff for billboards so close sprites don't over-scale. */
#define SPRITE_NEAR_CLIP_Z 50

/* Raise view height for rendering only (gameplay uses plr->yoff unchanged).
 * Makes the camera draw from higher so the floor appears further away, matching Amiga. */
#define VIEW_HEIGHT_LIFT  (2 * 1024)

/* -----------------------------------------------------------------------
 * Global renderer state
 * ----------------------------------------------------------------------- */
RendererState g_renderer;

/* -----------------------------------------------------------------------
 * Big-endian read helpers (level data is Amiga big-endian)
 * ----------------------------------------------------------------------- */
static inline int16_t rd16(const uint8_t *p) {
    return (int16_t)((p[0] << 8) | p[1]);
}
static inline int32_t rd32(const uint8_t *p) {
    return (int32_t)((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
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

/* Water animation / assets (Amiga: watertouse, wtan, wateroff, fillscrnwater). */
static uint16_t g_water_wtan = 0;
static uint8_t g_water_off = 0;
static int8_t g_fill_screen_water = 0;
static uint8_t g_water_anim_cursor = 0;
static uint16_t g_water_src_off = 0;
/* Interpolated per-display-frame refraction phase (smooths 50Hz step on 60Hz+ displays). */
static uint16_t g_water_wtan_draw = 0;
/* Millisecond remainder used to keep average animation speed at exactly 50Hz. */
static uint32_t g_water_ms_remainder = 0;
/* Gameplay tweak: run water animation at half speed. */
static uint32_t g_water_speed_ms_remainder = 0;
static const uint8_t *g_water_file = NULL;
static size_t g_water_file_size = 0;
static const uint8_t *g_water_brighten = NULL;
static size_t g_water_brighten_size = 0;
static const uint16_t g_water_src_offsets[8] = { 0, 2, 256, 258, 512, 514, 768, 770 };

/* AB3DI.s DrawDisplay:
 *   watertouse = *waterpt++;
 *   if (waterpt == end) waterpt = start;
 *   wtan += 640 (mod 8192), wateroff++ (mod 64). */
static void renderer_advance_water_anim(void)
{
    g_water_src_off = g_water_src_offsets[g_water_anim_cursor & 7u];
    g_water_anim_cursor = (uint8_t)((g_water_anim_cursor + 1u) & 7u);
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

/* -----------------------------------------------------------------------
 * Convert a 12-bit Amiga color word (0x0RGB) to ARGB8888.
 *
 * Only 4096 distinct input values exist, so we pre-build a lookup table
 * in renderer_init() and reduce every call site to a single array read.
 * ----------------------------------------------------------------------- */
static uint32_t amiga12_lut[4096];
/* Rounded 8-bit channel -> 4-bit nibble: exactly matches (v+8)/17 clamp. */
static uint8_t byte_to_nibble_lut[256];
/* Full RGB24 -> Amiga12 lookup table (16,777,216 entries, ~32 MiB). */
static uint16_t *argb24_to_amiga12_lut = NULL;
static int argb24_to_amiga12_lut_ready = 0;
/* Exact floor(x/255) for x in [0, 130050] used by blend_argb. */
#define DIV255_LUT_MAX 130050u
static uint16_t div255_lut[DIV255_LUT_MAX + 1u];
static uint32_t g_render_frame_counter = 0;

/* Debug helper:
 * Set AB3D_CLIP_TRACE_FRAMES=N to print portal clip windows for first N frames. */
static int renderer_take_clip_trace_slot(void)
{
    static int initialized = 0;
    static int frames_left = 0;
    if (!initialized) {
        const char *env = getenv("AB3D_CLIP_TRACE_FRAMES");
        if (env) frames_left = atoi(env);
        initialized = 1;
    }
    if (frames_left > 0) {
        frames_left--;
        return 1;
    }
    return 0;
}

/* Project rotated X/Z to current render pixel-space X, matching wall path math. */
static inline int project_x_to_pixels(int32_t vx, int32_t vz)
{
    if (vz <= 0) return (vx >= 0) ? g_renderer.width : -g_renderer.width;
    int center_x = (g_renderer.width * 47) / 96;
    return (int)((int64_t)vx * (int64_t)RENDER_SCALE / (int64_t)vz) + center_x;
}

/* Project world Y to current render pixel-space Y using nearest-pixel rounding.
 * Matching wall/floor rounding avoids 1px seams along shared edges. */
static inline int project_y_to_pixels_round(int32_t vy, int32_t vz, int32_t proj_y_scale, int center_y)
{
    int64_t den = (vz > 0) ? (int64_t)vz : 1;
    int64_t num = (int64_t)vy * (int64_t)proj_y_scale * (int64_t)RENDER_SCALE;
    int64_t q = (num >= 0) ? ((num + den / 2) / den) : ((num - den / 2) / den);
    return (int)q + center_y;
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

#define FLOOR_PAL_LEVEL_COUNT 15
static const uint8_t *g_floor_span_pal_cache_src = NULL;
static uint8_t  g_floor_span_pal_cache_valid[FLOOR_PAL_LEVEL_COUNT];
static uint32_t g_floor_span_argb_cache[FLOOR_PAL_LEVEL_COUNT][256];
static uint16_t g_floor_span_cw_cache[FLOOR_PAL_LEVEL_COUNT][256];

static void floor_span_prepare_pal_cache(const uint8_t *pal_lut_src, int level,
                                         const uint32_t **argb_out,
                                         const uint16_t **cw_out)
{
    if (!pal_lut_src || level < 0 || level >= FLOOR_PAL_LEVEL_COUNT) {
        *argb_out = NULL;
        *cw_out = NULL;
        return;
    }

    if (g_floor_span_pal_cache_src != pal_lut_src) {
        g_floor_span_pal_cache_src = pal_lut_src;
        memset(g_floor_span_pal_cache_valid, 0, sizeof(g_floor_span_pal_cache_valid));
    }

    if (!g_floor_span_pal_cache_valid[level]) {
        const uint8_t *lut = pal_lut_src + level * 512;
        for (int ti = 0; ti < 256; ti++) {
            uint16_t cw = (uint16_t)((lut[ti * 2] << 8) | lut[ti * 2 + 1]);
            g_floor_span_argb_cache[level][ti] = amiga12_to_argb(cw);
            g_floor_span_cw_cache[level][ti] = cw;
        }
        g_floor_span_pal_cache_valid[level] = 1;
    }

    *argb_out = g_floor_span_argb_cache[level];
    *cw_out = g_floor_span_cw_cache[level];
}

void renderer_set_water_assets(const uint8_t *water_file, size_t water_file_size,
                               const uint8_t *water_brighten, size_t water_brighten_size)
{
    g_water_file = water_file;
    g_water_file_size = water_file_size;
    g_water_brighten = water_brighten;
    g_water_brighten_size = water_brighten_size;
    g_water_anim_cursor = 0;
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
    free(g_renderer.buffer);
    g_renderer.buffer = NULL;
    free(g_renderer.back_buffer);
    g_renderer.back_buffer = NULL;
    free(g_renderer.rgb_buffer);
    g_renderer.rgb_buffer = NULL;
    free(g_renderer.rgb_back_buffer);
    g_renderer.rgb_back_buffer = NULL;
    free(g_renderer.cw_buffer);
    g_renderer.cw_buffer = NULL;
    free(g_renderer.cw_back_buffer);
    g_renderer.cw_back_buffer = NULL;
    free(g_renderer.depth_buffer);
    g_renderer.depth_buffer = NULL;
    free(g_renderer.clip.top);
    g_renderer.clip.top = NULL;
    free(g_renderer.clip.bot);
    g_renderer.clip.bot = NULL;
    free(g_renderer.clip.z);
    g_renderer.clip.z = NULL;
}

static void allocate_buffers(int w, int h)
{
    g_renderer.width = w;
    g_renderer.height = h;
    g_renderer.present_width = w;
    g_renderer.present_height = h;

    size_t buf_size = (size_t)w * h;
    g_renderer.buffer = (uint8_t*)calloc(1, buf_size);
    g_renderer.back_buffer = (uint8_t*)calloc(1, buf_size);

    size_t rgb_size = buf_size * sizeof(uint32_t);
    g_renderer.rgb_buffer = (uint32_t*)calloc(1, rgb_size);
    g_renderer.rgb_back_buffer = (uint32_t*)calloc(1, rgb_size);

    size_t cw_size = buf_size * sizeof(uint16_t);
    g_renderer.cw_buffer = (uint16_t*)calloc(1, cw_size);
    g_renderer.cw_back_buffer = (uint16_t*)calloc(1, cw_size);

    g_renderer.depth_buffer = NULL;  /* Amiga: no depth buffer; painter's + stream order only */

    size_t clip_size = (size_t)w * sizeof(int16_t);
    g_renderer.clip.top = (int16_t*)calloc(1, clip_size);
    g_renderer.clip.bot = (int16_t*)calloc(1, clip_size);
    g_renderer.clip.z = (int32_t*)calloc(1, (size_t)w * sizeof(int32_t));

    g_renderer.top_clip = 0;
    g_renderer.bot_clip = (int16_t)(h - 1);
    g_renderer.wall_top_clip = -1;
    g_renderer.wall_bot_clip = -1;
    g_renderer.left_clip = 0;
    g_renderer.right_clip = (int16_t)w;
}

void renderer_init(void)
{
    build_amiga12_lut();
    memset(&g_renderer, 0, sizeof(g_renderer));
    allocate_buffers(RENDER_WIDTH, RENDER_HEIGHT);
    printf("[RENDERER] Initialized: %dx%d\n", g_renderer.width, g_renderer.height);
}

void renderer_resize(int w, int h)
{
    if (w < 96) w = 96;
    if (h < 80) h = 80;
    if (w > 2048) w = 2048;
    if (h > 2048) h = 2048;
    free_buffers();
    allocate_buffers(w, h);
}

void renderer_shutdown(void)
{
    free_buffers();
    free(argb24_to_amiga12_lut);
    argb24_to_amiga12_lut = NULL;
    argb24_to_amiga12_lut_ready = 0;
    printf("[RENDERER] Shutdown\n");
}

/* Row templates for fast RGB clear (avoid per-pixel loops). Max width from renderer_resize. */
#define CLEAR_ROW_MAX 2048
static uint32_t s_clear_sky_row[CLEAR_ROW_MAX];
static uint16_t s_clear_sky_cw_row[CLEAR_ROW_MAX];
static int s_clear_rows_inited = 0;

static void init_clear_rows(void)
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

static inline uint32_t sky_fetch_argb_mode0(int tx, int ty)
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

void renderer_draw_sky_pass(int16_t angpos)
{
    uint8_t *buf = g_renderer.buffer;
    uint32_t *rgb = g_renderer.rgb_buffer;
    uint16_t *cw = g_renderer.cw_buffer;
    if (!buf || !rgb || !cw) return;
    int w = g_renderer.width;
    int h = g_renderer.height;
    if (w < 1 || h < 1) return;

    int u0 = ((int)(angpos & 8191) * SKY_PAN_WIDTH) / 8192;
    /* Amiga puts backdrop in top 38 of 80 rows; keep same proportion at any output height. */
    int sky_h = (h * 38) / 80;
    if (sky_h < 1) sky_h = 1;
    /* Amiga draws one sky source column per 3D screen column (96 visible columns). */
    const int sky_view_cols = 96;

    if (s_sky_pixels && s_sky_mode == 0) {
        int th = s_sky_tex_h;
        int tw = s_sky_tex_w;
#if SKY_BILINEAR_FILTER
        if (th > 1 && tw > 1 && w > 1 && sky_h > 1) {
            const int64_t pan_period_fp = ((int64_t)SKY_PAN_WIDTH << 16);
            const int64_t sx_step_fp = (((int64_t)(sky_view_cols - 1)) << 16) / (int64_t)(w - 1);
            const int64_t v_den = (int64_t)(sky_h > 1 ? sky_h - 1 : 1);
            for (int y = 0; y < sky_h; y++) {
                int64_t v_fp = (((int64_t)y * (th - 1)) << 16) / v_den;
                int v0 = (int)(v_fp >> 16);
                int v1 = (v0 < th - 1) ? (v0 + 1) : v0;
                uint32_t fy = (uint32_t)(v_fp & 0xFFFF);
                size_t row = (size_t)y * (size_t)w;
                int64_t sx_fp = 0;
                for (int x = 0; x < w; x++, sx_fp += sx_step_fp) {
                    int64_t pan_fp = ((int64_t)u0 << 16) + sx_fp;
                    pan_fp %= pan_period_fp;
                    if (pan_fp < 0) pan_fp += pan_period_fp;
                    int64_t tx_fp = (pan_fp * (int64_t)tw) / (int64_t)SKY_PAN_WIDTH;
                    int tx0 = (int)(tx_fp >> 16);
                    uint32_t fx = (uint32_t)(tx_fp & 0xFFFF);
                    int tx1 = tx0 + 1;
                    if (tx1 >= tw) tx1 = 0;
                    uint32_t c00 = sky_fetch_argb_mode0(tx0, v0);
                    uint32_t c10 = sky_fetch_argb_mode0(tx1, v0);
                    uint32_t c01 = sky_fetch_argb_mode0(tx0, v1);
                    uint32_t c11 = sky_fetch_argb_mode0(tx1, v1);
                    uint32_t px = sky_bilinear_argb(c00, c10, c01, c11, fx, fy);
                    size_t p = row + (size_t)x;
                    buf[p] = 0;
                    rgb[p] = px;
                    cw[p] = argb_to_amiga12(px);
                }
            }
        } else
#endif
        {
            /* Nearest-neighbour path (or filtering disabled). */
            int bpc = s_sky_bytes_per_col;
            size_t row_stride = (size_t)tw * 2u;
            const int64_t r_den = (int64_t)(sky_h > 1 ? sky_h - 1 : 1);
            for (int y = 0; y < sky_h; y++) {
                int rpix = (int)((int64_t)y * (th - 1) / r_den);
                if (rpix < 0) rpix = 0;
                if (rpix >= th) rpix = th - 1;
                size_t row = (size_t)y * (size_t)w;
                for (int x = 0; x < w; x++) {
                    int sx = (int)((int64_t)x * sky_view_cols / w);
                    if (sx < 0) sx = 0;
                    if (sx >= sky_view_cols) sx = sky_view_cols - 1;
                    int c = (u0 + sx) % SKY_PAN_WIDTH;
                    if (c < 0) c += SKY_PAN_WIDTH;
                    c = (c * tw) / SKY_PAN_WIDTH;
                    if (c >= tw) c = tw - 1;
                    size_t off;
                    if (s_sky_row_major) off = (size_t)rpix * row_stride + (size_t)c * 2u;
                    else                 off = (size_t)c * (size_t)bpc + (size_t)rpix * 2u;
                    if (off + 2u > s_sky_data_bytes) continue;
                    uint16_t cw12 = (uint16_t)((s_sky_pixels[off] << 8) | s_sky_pixels[off + 1]);
                    uint32_t px = amiga12_to_argb(cw12);
                    size_t p = row + (size_t)x;
                    buf[p] = 0;
                    rgb[p] = px;
                    cw[p] = argb_to_amiga12(px);
                }
            }
        }
    } else if (s_sky_pixels && s_sky_mode == 1) {
        int th = s_sky_tex_h;
        int tw = s_sky_tex_w;
#if SKY_BILINEAR_FILTER
        if (th > 1 && tw > 1 && w > 1 && sky_h > 1) {
            const int64_t pan_period_fp = ((int64_t)SKY_PAN_WIDTH << 16);
            const int64_t sx_step_fp = (((int64_t)(sky_view_cols - 1)) << 16) / (int64_t)(w - 1);
            const int64_t v_den = (int64_t)(sky_h > 1 ? sky_h - 1 : 1);
            for (int y = 0; y < sky_h; y++) {
                int64_t v_fp = (((int64_t)y * (th - 1)) << 16) / v_den;
                int v0 = (int)(v_fp >> 16);
                int v1 = (v0 < th - 1) ? (v0 + 1) : v0;
                uint32_t fy = (uint32_t)(v_fp & 0xFFFF);
                size_t row = (size_t)y * (size_t)w;
                int64_t sx_fp = 0;
                for (int x = 0; x < w; x++, sx_fp += sx_step_fp) {
                    int64_t pan_fp = ((int64_t)u0 << 16) + sx_fp;
                    pan_fp %= pan_period_fp;
                    if (pan_fp < 0) pan_fp += pan_period_fp;
                    int64_t tx_fp = (pan_fp * (int64_t)tw) / (int64_t)SKY_PAN_WIDTH;
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
                    rgb[p] = px;
                    cw[p] = argb_to_amiga12(px);
                }
            }
        } else
#endif
        {
            /* Nearest-neighbour path (or filtering disabled). */
            const int64_t v_den = (int64_t)(sky_h > 1 ? sky_h - 1 : 1);
            for (int y = 0; y < sky_h; y++) {
                int v = (int)((int64_t)y * (th - 1) / v_den);
                if (v < 0) v = 0;
                if (v >= th) v = th - 1;
                size_t row = (size_t)y * (size_t)w;
                for (int x = 0; x < w; x++) {
                    int sx = (int)((int64_t)x * sky_view_cols / w);
                    if (sx < 0) sx = 0;
                    if (sx >= sky_view_cols) sx = sky_view_cols - 1;
                    int tu = (u0 + sx) % SKY_PAN_WIDTH;
                    if (tu < 0) tu += SKY_PAN_WIDTH;
                    tu = (tu * tw) / SKY_PAN_WIDTH;
                    if (tu >= tw) tu = tw - 1;
                    size_t toff = (size_t)v * (size_t)tw + (size_t)tu;
                    if (toff >= (size_t)tw * (size_t)th) continue;
                    uint8_t idx = s_sky_pixels[toff];
                    size_t p = row + (size_t)x;
                    buf[p] = 0;
                    rgb[p] = s_sky_argb[idx];
                    cw[p] = s_sky_cw[idx];
                }
            }
        }
    } else {
        for (int y = 0; y < sky_h; y++) {
            int t = (y * 255) / (sky_h > 1 ? sky_h - 1 : 1);
            size_t row = (size_t)y * (size_t)w;
            for (int x = 0; x < w; x++) {
                int sx = (int)((int64_t)x * sky_view_cols / w);
                if (sx < 0) sx = 0;
                if (sx >= sky_view_cols) sx = sky_view_cols - 1;
                int u = (u0 + sx) % SKY_PAN_WIDTH;
                if (u < 0) u += SKY_PAN_WIDTH;
                int shade = t + (u * 40) / SKY_PAN_WIDTH;
                if (shade > 255) shade = 255;
                int r = (shade * 40) / 255;
                int g = (shade * 90) / 255;
                int b = 30 + (shade * 200) / 255;
                uint32_t px = RENDER_RGB_RASTER_PIXEL(((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b);
                size_t p = row + (size_t)x;
                buf[p] = 0;
                rgb[p] = px;
                cw[p] = argb_to_amiga12(px);
            }
        }
    }

    /* Below sky band: same clear as renderer_clear (empty canvas for world draw). */
    {
        init_clear_rows();
        const uint32_t below_px = RENDER_RGB_CLEAR_SKY_PIXEL;
        const uint16_t below_cw = 0x0EEEu;
        size_t row_bytes_rgb = (size_t)w * sizeof(uint32_t);
        size_t row_bytes_cw = (size_t)w * sizeof(uint16_t);
        if (w <= CLEAR_ROW_MAX) {
            for (int y = sky_h; y < h; y++) {
                size_t row = (size_t)y * (size_t)w;
                memset(buf + row, 0, (size_t)w);
                memcpy(rgb + row, s_clear_sky_row, row_bytes_rgb);
                memcpy(cw + row, s_clear_sky_cw_row, row_bytes_cw);
            }
        } else {
            for (int y = sky_h; y < h; y++) {
                size_t row = (size_t)y * (size_t)w;
                for (int x = 0; x < w; x++) {
                    size_t p = row + (size_t)x;
                    buf[p] = 0;
                    rgb[p] = below_px;
                    cw[p] = below_cw;
                }
            }
        }
    }
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
}

const uint8_t *renderer_get_buffer(void)
{
    return g_renderer.back_buffer; /* The just-drawn frame */
}

const uint32_t *renderer_get_rgb_buffer(void)
{
    return g_renderer.rgb_back_buffer; /* The just-drawn RGB frame */
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
static inline void put_pixel(uint8_t *buf, int x, int y, uint8_t color)
{
    int w = g_renderer.width, h = g_renderer.height;
    if (x >= 0 && x < w && y >= 0 && y < h) {
        buf[y * w + x] = color;
    }
}

static inline void draw_vline(uint8_t *buf, int x, int y_top, int y_bot,
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
     * view_z = dx * sin + dz * cos    (d1 = d0*d6_swapped + d1*d6) */
    int32_t vx = (int32_t)dx * cos_v - (int32_t)dz * sin_v;
    vx <<= 1;              /* add.l d2,d2 in ASM */
    int16_t vx16 = (int16_t)(vx >> 16);  /* swap d2 */
    int32_t vx_fine = (int32_t)vx16 << 7; /* asl.l #7 */
    vx_fine += r->xwobble;

    int32_t vz = (int32_t)dx * sin_v + (int32_t)dz * cos_v;
    vz <<= 2;              /* asl.l #2 in ASM */
    int16_t vz16 = (int16_t)(vz >> 16);  /* swap d1 */

    r->rotated[idx].x = vx_fine;
    r->rotated[idx].z = (int32_t)vz16;

    /* Project to screen column in Amiga 96-column space.
     * ASM: divs d1,d2 ; add.w #47,d2 */
    if (vz16 > 0) {
        int32_t screen_x = (vx_fine / vz16) + 47;
        r->on_screen[idx].screen_x = (int16_t)screen_x;
        r->on_screen[idx].flags = 0;
    } else {
        /* Behind camera.
         * Amiga uses hard edge sentinels (0 or right edge), with tst.w on d2. */
        r->on_screen[idx].screen_x = (int16_t)(((int16_t)vx_fine > 0) ? 96 : 0);
        r->on_screen[idx].flags = 1;
    }
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

    const uint8_t *pts = state->level.object_points;

    /* Amiga ObjDraw: ObjRotated is indexed by POINT number, not object index.
     * Rotate every point; when drawing, object uses (object_data[0]) as pt num
     * to look up ObjRotated[pt_num]. So keys (and others) scale correctly. */
    for (int pt = 0; pt < num_pts; pt++) {
        int16_t px = rd16(pts + pt * 8);
        int16_t pz = rd16(pts + pt * 8 + 4);

        int16_t dx = (int16_t)(px - cam_x);
        int16_t dz = (int16_t)(pz - cam_z);

        /* Same rotation as level points */
        int32_t vx = (int32_t)dx * cos_v - (int32_t)dz * sin_v;
        vx <<= 1;
        int16_t vx16 = (int16_t)(vx >> 16);

        int32_t vz = (int32_t)dx * sin_v + (int32_t)dz * cos_v;
        vz <<= 2;
        int16_t vz16 = (int16_t)(vz >> 16);

        int32_t vx_fine = (int32_t)vx16 << 7;
        vx_fine += r->xwobble;

        r->obj_rotated[pt].x = vx16;
        r->obj_rotated[pt].z = (int32_t)vz16;
        r->obj_rotated[pt].x_fine = vx_fine;
    }
}

/* Wall texture index for switches (io.c wall_texture_table). Must be before draw_wall_column. */
#define SWITCHES_WALL_TEX_ID  11

/* -----------------------------------------------------------------------
 * Wall rendering (column-by-column)
 *
 * Translated from WallRoutine3.ChipMem.s ScreenWallstripdraw.
 *
 * Draws a vertical strip of a textured wall.
 * Parameters:
 *   x         - screen column
 *   y_top     - top of wall on screen
 *   y_bot     - bottom of wall on screen
 *   tex_col   - texture column to sample (0-127)
 *   texture   - pointer to wall pixel data (past 2048-byte LUT header)
 *   amiga_d6  - Amiga dimming index (0-32): 0=brightest (close), 32=dimmest (far)
 *
 * Uses g_renderer.cur_wall_pal as the per-texture 2048-byte LUT.
 * ----------------------------------------------------------------------- */
static void draw_wall_column(int x, int y_top, int y_bot,
                             int y_top_tex,
                             int tex_col, const uint8_t *texture,
                             int amiga_d6,
                             uint8_t valand, uint8_t valshift,
                             int16_t totalyoff, int32_t col_z,
                             int16_t wall_height_world, int16_t tex_id,
                             int16_t d6_max)
{
    uint8_t *buf = g_renderer.buffer;
    uint32_t *rgb = g_renderer.rgb_buffer;
    uint16_t *cw = g_renderer.cw_buffer;
    if (!buf || !rgb || !cw) return;
    if (x < g_renderer.left_clip || x >= g_renderer.right_clip) return;

    int width = g_renderer.width;

    /* Short walls (e.g. step risers) can project to same row; ensure at least 1 pixel height */
    int yt = y_top, yb = y_bot;
    if (yb <= yt) yb = yt + 1;

    /* Clip to screen. wall_top_clip/wall_bot_clip (multi-floor) let walls extend to meet ceiling/floor. */
    int effective_top = (g_renderer.wall_top_clip >= 0) ? (int)g_renderer.wall_top_clip : g_renderer.top_clip;
    int ct = (yt < effective_top) ? effective_top : yt;
    int effective_bot = (g_renderer.wall_bot_clip >= 0) ? (int)g_renderer.wall_bot_clip : g_renderer.bot_clip;
    int cb = (yb > effective_bot) ? effective_bot : yb;
    if (ct > cb) return;

    int wall_height = cb - ct;
    if (wall_height <= 0) return;

    /* Clamp d6 to SCALE table range.
     * See-through wall path in Amiga clamps to 0..32; normal walls clamp to 0..64. */
    if (d6_max < 0) d6_max = 0;
    if (d6_max > 64) d6_max = 64;
    if (amiga_d6 < 0) amiga_d6 = 0;
    if (amiga_d6 > d6_max) amiga_d6 = d6_max;

    /* Get the brightness block offset from the SCALE table */
    uint16_t lut_block_off = wall_scale_table[amiga_d6];

    /* Pointer to the per-texture LUT (NULL if no palette loaded) */
    const uint8_t *pal = g_renderer.cur_wall_pal;

    /* --- Packed texture addressing ---
     *
     * Amiga wall textures pack 3 five-bit texels per 16-bit word:
     *   bits [4:0]   = texel A (PACK 0)
     *   bits [9:5]   = texel B (PACK 1)
     *   bits [14:10] = texel C (PACK 2)
     *
     * The texture data is arranged in vertical strips.  Each strip
     * covers 3 adjacent columns and has (1 << valshift) rows, with
     * 2 bytes per row (one 16-bit packed word).
     *
     *   strip_index  = tex_col / 3
     *   pack_mode    = tex_col % 3
     *   strip_offset = strip_index << (valshift + 1)
     *   row_word     = data[strip_offset + (ty & valand) * 2 .. +1]
     *
     * (Derived from WallRoutine3.ChipMem.s ScreenWallstripdraw)   */
    int strip_index = tex_col / 3;
    int pack_mode   = tex_col % 3;
    /* strip_offset = strip_index * 2 << valshift  =  strip_index << (valshift+1) */
    int strip_offset = strip_index << (valshift + 1);

    /* Texture mapping uses logical wall height (y_top_tex..y_bot). Caller may pass y_top one row
     * above projected top to close ceiling gaps; tex_step and tex_y use y_top_tex so the extra
     * row samples one texel above yoff (no stretch). */
    int wall_pixels = yb - y_top_tex;
    if (wall_pixels < 1) wall_pixels = 1;
    int rows = 1 << valshift;
    int h = (int)wall_height_world;
    if (h < 1) h = 1;
    /* Full-height short walls: show full texture.
     * Switch texture is 32 rows; map exactly one switch texture vertically. */
    if (tex_id == SWITCHES_WALL_TEX_ID) h = rows;
    else if (rows < 64 && h < 64) h = 64;
    /* Amiga d4 progression is independent of VALSHIFT/VALAND; those only affect row masking/addressing.
     * Scaling by rows over-wraps 128-row door textures vertically. */
    int32_t tex_step = (int32_t)(((int64_t)h << 16) / wall_pixels);
    /* Vertical texture start offset (masked to texture row count). */
    int32_t yoff = (int32_t)((unsigned)totalyoff & (unsigned)valand);
    int32_t tex_y = (ct - y_top_tex) * tex_step + ((int32_t)yoff << 16);

    /* Cache 32-entry wall shade table by (palette pointer, LUT block). */
    static const uint8_t *cache_pal = NULL;
    static uint16_t cache_block_off = 0xFFFFu;
    static uint32_t cache_argb[32];
    static uint16_t cache_cw[32];
    static int cache_valid = 0;

    uint32_t fallback = 0;
    uint16_t fallback_cw = 0;
    if (texture && pal) {
        if (!cache_valid || cache_pal != pal || cache_block_off != lut_block_off) {
            for (int ti = 0; ti < 32; ti++) {
                int lut_off = lut_block_off + ti * 2;
                uint16_t c12 = ((uint16_t)pal[lut_off] << 8) | pal[lut_off + 1];
                cache_argb[ti] = amiga12_to_argb(c12);
                cache_cw[ti] = c12;
            }
            cache_pal = pal;
            cache_block_off = lut_block_off;
            cache_valid = 1;
        }
    } else {
        int gray = (64 - amiga_d6) * 255 / 64;
        if (gray < 0)   gray = 0;
        if (gray > 255) gray = 255;
        fallback = RENDER_RGB_RASTER_PIXEL(((uint32_t)gray << 16) | ((uint32_t)gray << 8) | (uint32_t)gray);
        fallback_cw = argb_to_amiga12(fallback);
    }

    size_t pix = (size_t)ct * (size_t)width + (size_t)x;
    if (texture && pal && strip_offset >= 0) {
        if (pack_mode == 0) {
            for (int y = ct; y <= cb; y++) {
                int byte_off = strip_offset + (((int)(tex_y >> 16) & valand) << 1);
                uint16_t word = ((uint16_t)texture[byte_off] << 8)
                              | (uint16_t)texture[byte_off + 1];
                uint8_t texel5 = (uint8_t)(word & 31u);
                buf[pix] = 2; /* tag: wall */
                rgb[pix] = cache_argb[texel5];
                cw[pix] = cache_cw[texel5];
                pix += (size_t)width;
                tex_y += tex_step;
            }
        } else if (pack_mode == 1) {
            for (int y = ct; y <= cb; y++) {
                int byte_off = strip_offset + (((int)(tex_y >> 16) & valand) << 1);
                uint16_t word = ((uint16_t)texture[byte_off] << 8)
                              | (uint16_t)texture[byte_off + 1];
                uint8_t texel5 = (uint8_t)((word >> 5) & 31u);
                buf[pix] = 2; /* tag: wall */
                rgb[pix] = cache_argb[texel5];
                cw[pix] = cache_cw[texel5];
                pix += (size_t)width;
                tex_y += tex_step;
            }
        } else {
            for (int y = ct; y <= cb; y++) {
                int byte_off = strip_offset + (((int)(tex_y >> 16) & valand) << 1);
                uint16_t word = ((uint16_t)texture[byte_off] << 8)
                              | (uint16_t)texture[byte_off + 1];
                uint8_t texel5 = (uint8_t)((word >> 10) & 31u);
                buf[pix] = 2; /* tag: wall */
                rgb[pix] = cache_argb[texel5];
                cw[pix] = cache_cw[texel5];
                pix += (size_t)width;
                tex_y += tex_step;
            }
        }
    } else if (texture && pal) {
        uint32_t argb0 = cache_argb[0];
        uint16_t cw0 = cache_cw[0];
        for (int y = ct; y <= cb; y++) {
            buf[pix] = 2; /* tag: wall */
            rgb[pix] = argb0;
            cw[pix] = cw0;
            pix += (size_t)width;
            tex_y += tex_step;
        }
    } else {
        for (int y = ct; y <= cb; y++) {
            buf[pix] = 2; /* tag: wall */
            rgb[pix] = fallback;
            cw[pix] = fallback_cw;
            pix += (size_t)width;
            tex_y += tex_step;
        }
    }

    /* Extend strip by one column left/right with edge texel colours (cf. floor span horizontal extend). */
    {
        for (int row = ct; row <= cb; row++) {
            size_t mid = (size_t)row * (size_t)width + (size_t)x;
            uint32_t c = rgb[mid];
            uint16_t wv = cw[mid];
            if (x > 0) {
                size_t L = mid - 1;
                buf[L] = 2;
                rgb[L] = c;
                cw[L] = wv;
            }
            if (x + 1 < width) {
                size_t R = mid + 1;
                buf[R] = 2;
                rgb[R] = c;
                cw[R] = wv;
            }
        }
    }

    /* Update column clip (walls occlude floor/ceiling/sprites behind).
     * Store wall depth so sprites only skip when actually behind the wall (sprite_z >= clip.z). */
    if (y_top > g_renderer.clip.top[x]) {
        g_renderer.clip.top[x] = (int16_t)y_top;
    }
    if (y_bot < g_renderer.clip.bot[x]) {
        g_renderer.clip.bot[x] = (int16_t)y_bot;
    }
    if (g_renderer.clip.z) {
        g_renderer.clip.z[x] = col_z;
    }
}

/* Wall column loop uses inverse-Z in 8.24 (INVZ_ONE). Projecting Y as world_y*K/z with an
 * integer z from z = INVZ_ONE/inv_z truncates twice vs. world_y*K*inv_z/INVZ_ONE (one divide).
 * The latter matches true perspective along the span and reduces stair-steps at floor/ceiling. */
static int wall_proj_y_screen_invz(int16_t world_y, int64_t inv_z_fp, int32_t proj_y_scale, int height)
{
    const int64_t INVZ_ONE = (1LL << 24);
    if (inv_z_fp <= 0) inv_z_fp = 1;
    int64_t num = (int64_t)world_y * (int64_t)proj_y_scale * (int64_t)RENDER_SCALE * inv_z_fp;
    int64_t q = (num >= 0)
        ? ((num + INVZ_ONE / 2) / INVZ_ONE)
        : ((num - INVZ_ONE / 2) / INVZ_ONE);
    return (int)q + height / 2;
}

/* Reference Z used to project two-level zone split height to screen Y (same scale as orp->z). */
#define TWO_LEVEL_SPLIT_REF_Z 400

/* -----------------------------------------------------------------------
 * Draw a wall segment between two rotated endpoints
 *
 * Translated from WallRoutine3.ChipMem.s walldraw/screendivide.
 *
 * Takes two endpoints in view space, projects them, and draws
 * columns from left to right with perspective-correct texturing.
 * ----------------------------------------------------------------------- */
void renderer_draw_wall(int32_t x1, int32_t z1, int32_t x2, int32_t z2,
                        int16_t top, int16_t bot,
                        const uint8_t *texture, int16_t tex_start,
                        int16_t tex_end, int16_t left_brightness, int16_t right_brightness,
                        uint8_t valand, uint8_t valshift, int16_t horand,
                        int16_t totalyoff, int16_t fromtile,
                        int16_t tex_id, int16_t wall_height_for_tex,
                        int16_t d6_max)
{
    RendererState *r = &g_renderer;

    /* Both behind camera - skip */
    if (z1 <= 0 && z2 <= 0) return;

    /* Clip to near plane */
    int32_t cx1 = x1, cz1 = z1;
    int32_t cx2 = x2, cz2 = z2;
    int16_t ct1 = tex_start, ct2 = tex_end;

    /* Amiga wall clip tests use z>0; keep near plane at 1 for parity. */
    const int32_t NEAR_PLANE = 1;
    
    if (cz1 < NEAR_PLANE) {
        /* Clip left endpoint to near plane */
        int32_t dz = cz2 - cz1;
        if (dz == 0) { cz1 = NEAR_PLANE; cx1 = (int16_t)((cx1 + cx2) / 2); ct1 = (int16_t)((ct1 + ct2) / 2); }
        else {
            int32_t t = (NEAR_PLANE - cz1) * 65536 / dz;
            cx1 = (int32_t)(cx1 + (int64_t)(cx2 - cx1) * t / 65536);
            cz1 = NEAR_PLANE;
            ct1 = (int16_t)(ct1 + (int32_t)(ct2 - ct1) * t / 65536);
        }
    }
    if (cz2 < NEAR_PLANE) {
        int32_t dz = cz1 - cz2;
        if (dz == 0) { cz2 = NEAR_PLANE; cx2 = cx1; ct2 = ct1; }
        else {
            int32_t t = (NEAR_PLANE - cz2) * 65536 / dz;
            cx2 = (int32_t)(cx2 + (int64_t)(cx1 - cx2) * t / 65536);
            cz2 = NEAR_PLANE;
            ct2 = (int16_t)(ct2 + (int32_t)(ct1 - ct2) * t / 65536);
        }
    }

    /* Project in fine pixel space for smooth motion. */
    int center_x = (g_renderer.width * 47) / 96;
    int scr_x1 = (int)((int64_t)cx1 * (int64_t)RENDER_SCALE / cz1) + center_x;
    int scr_x2 = (int)((int64_t)cx2 * (int64_t)RENDER_SCALE / cz2) + center_x;

    /* If endpoints project in reverse order, swap them for left-to-right drawing.
     * This can happen after near-plane clipping. All endpoint data must stay in sync. */
    if (scr_x1 > scr_x2) {
        int tmp;
        tmp = scr_x1; scr_x1 = scr_x2; scr_x2 = tmp;
        { int32_t t32 = cx1; cx1 = cx2; cx2 = t32; }
        { int32_t t32 = cz1; cz1 = cz2; cz2 = t32; }
        tmp = ct1; ct1 = (int16_t)ct2; ct2 = (int16_t)tmp;
    }

    if (scr_x2 < r->left_clip || scr_x1 >= r->right_clip) return;

    /* Clamp drawn range to clip region so we only iterate over visible columns.
     * This avoids int32 overflow in interpolation when the wall extends far off the left
     * (col = left_clip - scr_x1 can be huge, and col * 65536 overflows). */
    int draw_start = scr_x1;
    if (draw_start < r->left_clip) draw_start = r->left_clip;
    int draw_end = scr_x2;
    if (draw_end >= r->right_clip) draw_end = r->right_clip - 1;
    if (draw_start > draw_end) return;

    /* Wall span for interpolation (use at least 1 to avoid division by zero) */
    int span = scr_x2 - scr_x1;
    if (span <= 0) span = 1;

    /* Perspective-correct: interpolate 1/z and tex/z with higher fixed-point precision.
     * 16.16 inv_z causes visible quantization on far walls; use 8.24 here. */
    const int64_t INVZ_ONE = (1LL << 24); /* 8.24 */
    int64_t inv_z1_fp = INVZ_ONE / cz1;
    int64_t inv_z2_fp = INVZ_ONE / cz2;
    int64_t tex_over_z1_fp = (int64_t)ct1 * INVZ_ONE / cz1;
    int64_t tex_over_z2_fp = (int64_t)ct2 * INVZ_ONE / cz2;

    for (int screen_x = draw_start; screen_x <= draw_end; screen_x++) {
        /* t linear in screen x, 0..65535 (stable, no sensitive denominator) */
        int64_t t_fp = (int64_t)(screen_x - scr_x1) * 65536LL / span;
        if (t_fp < 0) t_fp = 0;
        if (t_fp > 65535) t_fp = 65535;

        int64_t inv_z_fp = inv_z1_fp + ((inv_z2_fp - inv_z1_fp) * t_fp) / 65536;
        if (inv_z_fp <= 0) inv_z_fp = 1;
        /* Rounded reciprocal: closer to true z than INVZ_ONE/inv_z truncation alone. */
        int32_t col_z = (int32_t)((INVZ_ONE + inv_z_fp / 2) / inv_z_fp);
        if (col_z < 1) col_z = 1;

        int32_t wall_bright = left_brightness +
            (int32_t)(((int64_t)(right_brightness - left_brightness) * t_fp) / 65536LL);
        int amiga_d6 = (col_z >> 7) + (wall_bright * 2);
        if (amiga_d6 < 0) amiga_d6 = 0;
        if (amiga_d6 > 64) amiga_d6 = 64;

        int y_top = wall_proj_y_screen_invz(top, inv_z_fp, g_renderer.proj_y_scale, g_renderer.height);
        int y_bot = wall_proj_y_screen_invz(bot, inv_z_fp, g_renderer.proj_y_scale, g_renderer.height);
        /* Strict Amiga parity: do not apply port-side wall-top seam expansion. */
        int ext = 0;
        int y_top_draw = y_top - ext;

        /* tex = (tex/z) * z in the same 8.24 domain, then convert to integer tex units. */
        int64_t tex_over_z_fp = tex_over_z1_fp + ((tex_over_z2_fp - tex_over_z1_fp) * t_fp) / 65536;
        int64_t tex_t_fp64 = tex_over_z_fp * (int64_t)col_z; /* 8.24 */
        int32_t tex_t_int = (int32_t)(tex_t_fp64 >> 24);
        int tex_col = ((int32_t)(tex_t_int & horand)) + fromtile;

        /* Switch walls: depth bias so they draw in front of the wall behind them. */
        int32_t depth_z = col_z;
        if (tex_id == SWITCHES_WALL_TEX_ID && col_z > 16) depth_z = col_z - 16;

        draw_wall_column(screen_x, y_top_draw, y_bot, y_top, tex_col, texture,
                         amiga_d6, valand, valshift, totalyoff, depth_z,
                         wall_height_for_tex, tex_id, d6_max);
    }
}

/* Extend floor/ceiling span by one column left/right with edge colours (cf. draw_wall_column +1 row). */
static void floor_span_extend_horizontal_edges(int16_t y, int xl, int xr, int width,
                                                uint8_t *buf, uint32_t *rgb, uint16_t *cwbuf)
{
    if (!buf || !rgb || !cwbuf || xl > xr || width < 1) return;
    if (y < 0 || y >= g_renderer.height) return;
    size_t row = (size_t)y * (size_t)width;
    size_t li = row + (size_t)xl;
    size_t ri = row + (size_t)xr;
    uint32_t cL = rgb[li];
    uint32_t cR = rgb[ri];
    uint16_t wL = cwbuf[li];
    uint16_t wR = cwbuf[ri];
    uint8_t tL = buf[li];
    uint8_t tR = buf[ri];
    if (xl > 0) {
        size_t L = li - 1;
        buf[L] = tL;
        rgb[L] = cL;
        cwbuf[L] = wL;
    }
    if (xr + 1 < width) {
        size_t R = ri + 1;
        buf[R] = tR;
        rgb[R] = cR;
        cwbuf[R] = wR;
    }
}

/* -----------------------------------------------------------------------
 * Floor/ceiling span rendering
 *
 * Translated from BumpMap.s / AB3DI.s itsafloordraw.
 *
 * Draws a horizontal span of floor or ceiling at a given height.
 * ----------------------------------------------------------------------- */
void renderer_draw_floor_span(int16_t y, int16_t x_left, int16_t x_right,
                              int32_t floor_height, const uint8_t *texture, const uint8_t *floor_pal,
                              int16_t brightness, int16_t left_brightness, int16_t right_brightness,
                              int16_t use_gouraud,
                              int16_t scaleval, int is_water,
                              int16_t water_rows_left)
{
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
    uint8_t *buf = rs->buffer;
    uint32_t *rgb = rs->rgb_buffer;
    uint16_t *cwbuf = rs->cw_buffer;
    if (!buf || !rgb || !cwbuf) return;
    if (y < 0 || y >= rs->height) return;

    int xl = (x_left < rs->left_clip) ? rs->left_clip : x_left;
    int xr = (x_right >= rs->right_clip) ? rs->right_clip - 1 : x_right;
    if (xl > xr) return;

    int center = rs->height / 2;  /* Match wall/floor projection center */
    int row_dist = y - center;
    if (row_dist == 0) row_dist = (y < center) ? -1 : 1;
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
    int32_t fh_8 = floor_height >> (WORLD_Y_FRAC_BITS - 1);
    int32_t dist;
    if (abs_row_dist <= 3) {
        dist = 32000;
    } else {
        dist = (int32_t)((int64_t)fh_8 * g_renderer.proj_y_scale * RENDER_SCALE / row_dist);
        if (dist < 0) dist = -dist;
        if (dist < 16) dist = 16;
        if (dist > 30000) dist = 30000;
    }

    /* Amiga formula: d6 = (dist >> 7) + zone_bright. Higher d6 = darker. */
    int amiga_d6 = (dist >> 7) + zone_d6;
    if (amiga_d6 < 0) amiga_d6 = 0;
    if (amiga_d6 > 64) amiga_d6 = 64;
    int gray = (64 - amiga_d6) * 255 / 64;

    /* ---- ASM pastfloorbright (line 6660) ----
     * d1 = d0 * cosval (change in U across whole width)
     * d2 = -(d0 * sinval) (change in V across whole width) */
    /* AB3DI pastfloorbright uses SineTable/bigsine values (~32767 magnitude).
     * Runtime sin/cos lookup is half-scale (~16384), so promote here for all
     * floor/ceiling/water span UV stepping. */
    int32_t cos_v = ((int32_t)rs->cosval) << 1;
    int32_t sin_v = ((int32_t)rs->sinval) << 1;
    int32_t d1 = (int32_t)(((int64_t)dist * cos_v));
    int32_t d2 = (int32_t)(-((int64_t)dist * sin_v));
    if (scaleval != 0) {
        /* Amiga AB3DI.s scaleprog (around line 6667), used for floor/roof/water:
         *   scaleval > 0 => asl.l d1/d2 (larger texels)
         *   scaleval < 0 => asr.l d1/d2 (smaller texels) */
        int s = (int)scaleval;
        if (s > 0) {
            if (s > 15) s = 15;
            {
                int64_t t1 = (int64_t)d1 << s;
                int64_t t2 = (int64_t)d2 << s;
                if (t1 > INT32_MAX) t1 = INT32_MAX;
                if (t1 < INT32_MIN) t1 = INT32_MIN;
                if (t2 > INT32_MAX) t2 = INT32_MAX;
                if (t2 < INT32_MIN) t2 = INT32_MIN;
                d1 = (int32_t)t1;
                d2 = (int32_t)t2;
            }
        } else {
            s = -s;
            if (s > 15) s = 15;
            d1 >>= s;
            d2 >>= s;
        }
    }

    /* Step per pixel: fixed so each pixel = same world extent at any width.
     * No def_w/w scaling - in widescreen we show more tiles, texture scale stays correct. */
    int w = rs->width;
    if (w < 1) w = 1;
    int32_t u_step = (int32_t)(d1 >> FLOOR_STEP_SHIFT);
    int32_t v_step = (int32_t)(d2 >> FLOOR_STEP_SHIFT);

    /* Center UV at screen middle (pixel w/2): U_center = -d2, V_center = d1.
     * So at pixel 0: start_u = -d2 - (w/2)*u_step, start_v = d1 - (w/2)*v_step.
     * Compute in 64-bit to avoid overflow, then take low 32 bits (UV wraps for tiling). */
    int half_w = w / 2;
    int64_t start_u64 = -(int64_t)d2 - (int64_t)half_w * u_step;
    int64_t start_v64 = (int64_t)d1 - (int64_t)half_w * v_step;

    /* Camera position offset: UV per world unit = FLOOR_CAM_UV_SCALE (matches fixed u_step). */
    int32_t cam_scale = FLOOR_CAM_UV_SCALE;
    if (scaleval != 0) {
        /* Match Amiga sxoff/szoff scaling done in pastsides before pastfloorbright. */
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
    start_u64 += (int64_t)rs->xoff * cam_scale;
    start_v64 += (int64_t)rs->zoff * cam_scale;

    /* Offset to left edge of span (xl pixels from left of screen). */
    if (xl > 0) {
        start_u64 += (int64_t)xl * u_step;
        start_v64 += (int64_t)xl * v_step;
    }

    const uint8_t *pal_lut_src = floor_pal ? floor_pal : rs->floor_pal;
    int floor_pal_level = 0;
    if (pal_lut_src && !use_gour) {
        int bright_idx = brightness + 5 + (dist >> 8);
        if (bright_idx < 0) bright_idx = 0;
        if (bright_idx > 28) bright_idx = 28;
        floor_pal_level = floor_bright_level_table[bright_idx];
    }

    const uint32_t *span_argb = NULL;
    const uint16_t *span_cw = NULL;
    const int use_span_lut = (texture != NULL && pal_lut_src != NULL && !is_water && !use_gour);
    if (use_span_lut) {
        floor_span_prepare_pal_cache(pal_lut_src, floor_pal_level, &span_argb, &span_cw);
    }

    /* UV accumulators: 32-bit wrapping is sufficient - we only need (fp>>16)&63 for tile coords.
     * The 64-bit start computation above already handles the large intermediate values;
     * truncating to 32 bits here halves the per-pixel addition cost. */
    uint32_t u_fp = (uint32_t)(int32_t)start_u64;
    uint32_t v_fp = (uint32_t)(int32_t)start_v64;

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

        /* Safety guard: keep source sampling on not-yet-written rows in current pass.
         * This is only a guard for high-res rasterization and does not alter non-water paths. */
        if (water_rows_left > 0) {
            int max_mag = (int)water_rows_left - 1;
            if (max_mag < 0) max_mag = 0;
            int32_t max_mag_fp = (int32_t)max_mag << 8;
            if (refr_y_off_fp > max_mag_fp) refr_y_off_fp = max_mag_fp;
            if (refr_y_off_fp < -max_mag_fp) refr_y_off_fp = -max_mag_fp;
        }
        water_refr_y_off_fp = refr_y_off_fp;
    }

    const int span_len = xr - xl + 1;
    uint8_t *row8 = buf + (size_t)y * w + (size_t)xl;
    uint32_t *row32 = rgb + (size_t)y * w + (size_t)xl;
    uint16_t *row16 = cwbuf + (size_t)y * w + (size_t)xl;
    size_t water_refr_base0 = 0;
    size_t water_refr_base1 = 0;
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
    }

    /* Fast non-water textured path: no per-pixel branching. */
    if (use_span_lut && span_argb && span_cw) {
        uint8_t *p8 = row8;
        uint32_t *p32 = row32;
        uint16_t *p16 = row16;
        for (int i = 0; i < span_len; i++) {
            uint8_t texel = texture[((((u_fp >> 16) & 63u)) << 2) | ((((v_fp >> 16) & 63u)) << 10)];
            u_fp += (uint32_t)u_step;
            v_fp += (uint32_t)v_step;

            *p8++ = 1;
            *p32++ = span_argb[texel];
            *p16++ = span_cw[texel];
        }
        floor_span_extend_horizontal_edges(y, xl, xr, w, buf, rgb, cwbuf);
        return;
    }

    if (is_water) {
        uint8_t *p8 = row8;
        uint32_t *p32 = row32;
        uint16_t *p16 = row16;
        size_t bg_i0 = water_refr_base0 + (size_t)xl;
        size_t bg_i1 = water_refr_base1 + (size_t)xl;
        const int water_has_file = (g_water_file && g_water_file_size >= 65536u);
        const int water_has_brighten = (g_water_brighten && g_water_brighten_size >= 512u);
        uint32_t dist_off = (((uint32_t)dist) & 0xFF00u) << 1;
        if (dist_off > (uint32_t)(12 * 512)) dist_off = (uint32_t)(12 * 512);

        for (int i = 0; i < span_len; i++, bg_i0++, bg_i1++) {
            uint8_t u8 = (uint8_t)((u_fp >> 16) & 0xFFu);
            uint8_t v8 = (uint8_t)((v_fp >> 16) & 0xFFu);
            u_fp += (uint32_t)u_step;
            v_fp += (uint32_t)v_step;

            /* Amiga-style textured water: refract existing pixels instead of drawing a solid color. */
            uint16_t d5_word = (uint16_t)(((uint16_t)v8 << 8) | (uint16_t)u8);
            uint16_t water_d5 = (uint16_t)(d5_word + (uint16_t)g_water_off);
            water_d5 &= 0x3F3Fu;
            uint8_t water_level = 0;

            if (water_has_file) {
                size_t wi = ((size_t)water_d5 << 2) + (size_t)g_water_src_off;
                if (wi < g_water_file_size) {
                    water_level = g_water_file[wi];
                }
            } else if (texture) {
                uint32_t tex_idx = (((uint32_t)v8 & 63u) << 10) | (((uint32_t)water_d5 & 63u) << 2);
                water_level = (uint8_t)(texture[tex_idx] >> 4);
            }

            uint32_t bg0 = rgb[bg_i0];
            uint16_t bg_cw0 = cwbuf[bg_i0];
            if (buf[bg_i0] == 0 && water_has_back_buffers) {
                /* AB3DI texturedwater samples from display memory while floor lines are streamed.
                 * When refraction points at rows not written yet this frame, those pixels still
                 * contain prior-frame values; mirror that by sampling back-buffer content. */
                bg0 = rs->rgb_back_buffer[bg_i0];
                bg_cw0 = rs->cw_back_buffer[bg_i0];
            }
            uint8_t bg_sample0 = (uint8_t)(bg_cw0 & 0xFFu);

            uint32_t bg1 = bg0;
            uint16_t bg_cw1 = bg_cw0;
            uint8_t bg_sample1 = bg_sample0;
            if (water_has_next_refr) {
                bg1 = rgb[bg_i1];
                bg_cw1 = cwbuf[bg_i1];
                if (buf[bg_i1] == 0 && water_has_back_buffers) {
                    bg1 = rs->rgb_back_buffer[bg_i1];
                    bg_cw1 = rs->cw_back_buffer[bg_i1];
                }
                bg_sample1 = (uint8_t)(bg_cw1 & 0xFFu);
            }

            uint32_t out;
            uint16_t out_cw;
            if (water_has_brighten) {
                /* Amiga texturedwater:
                 *   d0 = WaterFile word, then move.b sampled_pixel_lowbyte,d0,
                 *   output = brightentab[d0]. */
                size_t bi0 = (size_t)dist_off + ((size_t)water_level << 9) + (size_t)bg_sample0 * 2u;
                uint32_t out0;
                uint16_t out_cw0;
                if (bi0 + 1u < g_water_brighten_size) {
                    out_cw0 = (uint16_t)((g_water_brighten[bi0] << 8) | g_water_brighten[bi0 + 1u]);
                    out0 = amiga12_to_argb(out_cw0);
                } else {
                    out_cw0 = bg_cw0;
                    out0 = bg0;
                }

                if (water_has_next_refr) {
                    size_t bi1 = (size_t)dist_off + ((size_t)water_level << 9) + (size_t)bg_sample1 * 2u;
                    uint32_t out1;
                    uint16_t out_cw1;
                    if (bi1 + 1u < g_water_brighten_size) {
                        out_cw1 = (uint16_t)((g_water_brighten[bi1] << 8) | g_water_brighten[bi1 + 1u]);
                        out1 = amiga12_to_argb(out_cw1);
                    } else {
                        out_cw1 = bg_cw1;
                        out1 = bg1;
                    }
                    out = blend_argb(out0, out1, (uint32_t)water_refr_frac);
                    out_cw = argb_to_amiga12(out);
                } else {
                    out = out0;
                    out_cw = out_cw0;
                }
            } else {
                /* Fallback when brighten table is missing: keep prior blended behavior. */
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
            }

            *p8++ = 4; /* water tag: avoid wall-join fill smearing */
            *p32++ = out;
            *p16++ = out_cw;
        }
        floor_span_extend_horizontal_edges(y, xl, xr, w, buf, rgb, cwbuf);
        return;
    }

    /* Gouraud floor path (Amiga GOURSEL floor/roof): interpolate brightness levels across the span. */
    int64_t gour_level_fp = 0;
    int64_t gour_level_step = 0;
    int64_t gour_bright_fp = 0;
    int64_t gour_bright_step = 0;
    const int dist_add = (dist >> 7);
    if (use_gour) {
        int span_w = xr - xl;
        int left_level = left_brightness + dist_add;
        int right_level = right_brightness + dist_add;
        if (left_level < 0) left_level = 0;
        if (right_level < 0) right_level = 0;
        left_level >>= 1;
        right_level >>= 1;
        if (left_level > 14) left_level = 14;
        if (right_level > 14) right_level = 14;
        gour_level_fp = (int64_t)left_level << 16;
        gour_level_step = (span_w > 0) ? (((int64_t)(right_level - left_level) << 16) / span_w) : 0;
        gour_bright_fp = (int64_t)left_brightness << 16;
        gour_bright_step = (span_w > 0) ? (((int64_t)(right_brightness - left_brightness) << 16) / span_w) : 0;
    }

    if (texture && pal_lut_src) {
        uint8_t *p8 = row8;
        uint32_t *p32 = row32;
        uint16_t *p16 = row16;

        if (use_gour) {
            const uint32_t *gour_argb_levels[FLOOR_PAL_LEVEL_COUNT];
            const uint16_t *gour_cw_levels[FLOOR_PAL_LEVEL_COUNT];
            for (int level = 0; level < FLOOR_PAL_LEVEL_COUNT; level++) {
                floor_span_prepare_pal_cache(pal_lut_src, level,
                                             &gour_argb_levels[level],
                                             &gour_cw_levels[level]);
            }

            for (int i = 0; i < span_len; i++) {
                int gour_level = (int)(gour_level_fp >> 16);
                gour_level_fp += gour_level_step;
                if (gour_level < 0) gour_level = 0;
                if (gour_level > 14) gour_level = 14;

                uint8_t texel = texture[((((u_fp >> 16) & 63u)) << 2) | ((((v_fp >> 16) & 63u)) << 10)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;

                *p8++ = 1;
                *p32++ = gour_argb_levels[gour_level][texel];
                *p16++ = gour_cw_levels[gour_level][texel];
            }
            floor_span_extend_horizontal_edges(y, xl, xr, w, buf, rgb, cwbuf);
            return;
        }

        {
            const uint8_t *lut = pal_lut_src + floor_pal_level * 512;
            for (int i = 0; i < span_len; i++) {
                uint8_t texel = texture[((((u_fp >> 16) & 63u)) << 2) | ((((v_fp >> 16) & 63u)) << 10)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;
                uint16_t out_cw = (uint16_t)((lut[texel * 2] << 8) | lut[texel * 2 + 1]);

                *p8++ = 1;
                *p32++ = amiga12_to_argb(out_cw);
                *p16++ = out_cw;
            }
            floor_span_extend_horizontal_edges(y, xl, xr, w, buf, rgb, cwbuf);
            return;
        }
    }

    if (texture) {
        uint8_t *p8 = row8;
        uint32_t *p32 = row32;
        uint16_t *p16 = row16;

        if (use_gour) {
            for (int i = 0; i < span_len; i++) {
                int gour_bright = (int)(gour_bright_fp >> 16);
                gour_bright_fp += gour_bright_step;
                int d6 = dist_add + (gour_bright * 2);
                if (d6 < 0) d6 = 0;
                if (d6 > 64) d6 = 64;
                int gour_gray = (64 - d6) * 255 / 64;

                uint8_t texel = texture[((((u_fp >> 16) & 63u)) << 2) | ((((v_fp >> 16) & 63u)) << 10)];
                u_fp += (uint32_t)u_step;
                v_fp += (uint32_t)v_step;

                int lit = ((int)texel * gour_gray) >> 8;
                uint32_t argb = RENDER_RGB_RASTER_PIXEL(((uint32_t)lit << 16) | ((uint32_t)lit << 8) | (uint32_t)lit);
                *p8++ = 1;
                *p32++ = argb;
                *p16++ = argb_to_amiga12(argb);
            }
            floor_span_extend_horizontal_edges(y, xl, xr, w, buf, rgb, cwbuf);
            return;
        }

        for (int i = 0; i < span_len; i++) {
            uint8_t texel = texture[((((u_fp >> 16) & 63u)) << 2) | ((((v_fp >> 16) & 63u)) << 10)];
            u_fp += (uint32_t)u_step;
            v_fp += (uint32_t)v_step;
            int lit = ((int)texel * gray) >> 8;
            uint32_t argb = RENDER_RGB_RASTER_PIXEL(((uint32_t)lit << 16) | ((uint32_t)lit << 8) | (uint32_t)lit);
            *row8++ = 1;
            *row32++ = argb;
            *row16++ = argb_to_amiga12(argb);
        }
        floor_span_extend_horizontal_edges(y, xl, xr, w, buf, rgb, cwbuf);
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
            *row32++ = argb;
            *row16++ = argb_to_amiga12(argb);
        }
    } else {
        uint32_t argb = RENDER_RGB_RASTER_PIXEL(((uint32_t)gray << 16) | ((uint32_t)gray << 8) | (uint32_t)gray);
        uint16_t out_cw = argb_to_amiga12(argb);
        for (int i = 0; i < span_len; i++) {
            *row8++ = 1;
            *row32++ = argb;
            *row16++ = out_cw;
        }
    }
    floor_span_extend_horizontal_edges(y, xl, xr, w, buf, rgb, cwbuf);
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
    (void)sprite_type;
    uint8_t *buf = g_renderer.buffer;
    uint32_t *rgb = g_renderer.rgb_buffer;
    uint16_t *cw = g_renderer.cw_buffer;
    if (!buf || !rgb || !cw) return;
    if (z <= SPRITE_NEAR_CLIP_Z) return;
    if (!wad || !ptr_data) return;
    int rw = g_renderer.width, rh = g_renderer.height;
    if (src_cols < 1) src_cols = 32;
    if (src_rows < 1) src_rows = 32;

    /* ASM: sub.w d3,d0 (left = center_x - half_w) before doubling d3.
     * ASM: sub.w d4,d2 (top = center_y - half_h) before doubling d4.
     * width/height passed in are already doubled (full size).
     * sx/sy are set after we may reduce to draw_w/draw_h for texture aspect. */
    int sx, sy;

    /* Brightness -> palette byte offset via objscalecols (ObjDraw3.ChipRam.s line 572).
     * Use caller-provided d6 brightness (distance + object brightness), clamped to
     * the valid Amiga table index range. */
    int bright_idx = brightness;
    if (bright_idx < 0) bright_idx = 0;
    if (bright_idx > 61) bright_idx = 61;
    uint32_t pal_level_off = obj_scale_cols[bright_idx];
    if (pal && pal_size < 960) pal_level_off = 0;  /* single-level or small palette */
    int gray = (bright_idx * 255) / 62;
    if (gray < 90) gray = 90;  /* floor so no-palette / fallback path stays visible */

    /* Draw column-by-column (matching Amiga's column-strip approach).
     *
     * The PTR table has src_cols*2 columns per frame (confirmed by frame offset
     * spacing: e.g. alien frames are 64*4 bytes apart = 64 columns, src_cols=32).
     * Each column's WAD data has src_rows*2 words.
     *
     * Amiga blank-strip logic (line 763): "move.l (a5),d1 / beq blankstrip"
     * skips only when ALL 4 bytes of the PTR entry are zero (mode==0 AND wad_off==0).
     * wad_off==0 with mode!=0 is a VALID column (data at start of WAD). */
    int eff_cols = src_cols * 2;
    int eff_rows = src_rows * 2;
    uint32_t max_col = (uint32_t)((ptr_size > ptr_offset) ? (ptr_size - ptr_offset) / 4u : 0);

    sx = screen_x - width / 2;
    sy = screen_y - height / 2;

    int clip_left = g_renderer.left_clip;
    int clip_right = g_renderer.right_clip;
    if (clip_left < 0) clip_left = 0;
    if (clip_right > rw) clip_right = rw;
    int dx_start = 0;
    int dx_end = width;
    int col_start = sx;
    if (col_start < clip_left) col_start = clip_left;
    if (col_start < 0) col_start = 0;
    dx_start = col_start - sx;
    if (dx_start < 0) dx_start = 0;
    int col_end = sx + width;
    if (col_end > clip_right) col_end = clip_right;
    if (col_end > rw) col_end = rw;
    dx_end = col_end - sx;
    if (dx_end > width) dx_end = width;
    if (dx_end < 0) dx_end = 0;
    for (int dx = dx_start; dx < dx_end; dx++) {
        int screen_col = sx + dx;
        if (screen_col < 0 || screen_col >= rw) continue;

        /* Map screen column to source column 0..eff_cols-1 */
        int src_col = (width > 1) ? (dx * eff_cols) / width : 0;
        if (src_col >= eff_cols) src_col = eff_cols - 1;
        /* Clamp to valid PTR range */
        if (max_col > 0 && (uint32_t)src_col >= max_col) src_col = (int)(max_col - 1);

        /* Read PTR entry for this source column */
        uint32_t entry_off = ptr_offset + (uint32_t)src_col * 4;
        if (entry_off + 4 > ptr_size) continue;

        const uint8_t *entry = ptr_data + entry_off;
        uint8_t mode = entry[0];
        uint32_t wad_off = ((uint32_t)entry[1] << 16)
                         | ((uint32_t)entry[2] << 8)
                         | (uint32_t)entry[3];

        /* Amiga: beq blankstrip - skip when entire 4-byte entry is zero */
        if (mode == 0 && wad_off == 0) continue;
        if (wad_off >= wad_size) continue;

        const uint8_t *src = wad + wad_off;

        for (int dy = 0; dy < height; dy++) {
            int screen_row = sy + dy;
            if (screen_row < 0 || screen_row >= rh) continue;
            /* Room band clip: do not draw above ceiling or below floor (floor covers feet). */
            if (clip_top_sy < clip_bot_sy && (screen_row < clip_top_sy || screen_row > clip_bot_sy)) continue;

            /* Billboards are drawn last in the zone (after all walls), so we draw on top with no clip test. */
            /* Map screen row to source row 0..eff_rows-1 */
            int src_row = (height > 1) ? (dy * eff_rows) / height : 0;
            if (src_row >= eff_rows) src_row = eff_rows - 1;
            int row_idx = (int)down_strip + src_row;

            /* Bounds check: each row is 2 bytes (one 16-bit word) */
            if (wad_off + (size_t)(row_idx + 1) * 2 > wad_size) continue;

            /* Decode 5-bit pixel from packed word (match gun: build 16-bit word big-endian). */
            uint16_t w = (uint16_t)((src[row_idx * 2] << 8) | src[row_idx * 2 + 1]);
            uint8_t texel = 0;
            if (mode == 0) {
                texel = (uint8_t)(w & 0x1F);
            } else if (mode == 1) {
                texel = (uint8_t)((w >> 5) & 0x1F);
            } else {
                texel = (uint8_t)((w >> 10) & 0x1F);
            }
            if (texel == 0) continue;  /* transparent */

            uint8_t *row8 = buf + (size_t)screen_row * rw;
            uint32_t *row32 = rgb + (size_t)screen_row * rw;
            uint16_t *row16 = cw + (size_t)screen_row * rw;

            /* Geometry tag buffer is used by wall-join post-pass:
             * 1=floor/ceiling, 2=wall. Sprites must use a neutral tag so
             * they are never treated as wall/floor spans. */
            row8[screen_col] = 3;

            /* Color from .pal brightness palette (15 levels x 64 bytes or single 64-byte block).
             * Amiga .pal is big-endian 12-bit words. Try little-endian if colors look wrong. */
            if (pal && pal_size >= 64) {
                uint32_t level_off = (pal_level_off + 64 <= pal_size) ? pal_level_off : 0;
                uint32_t ci = level_off + (uint32_t)texel * 2;
                if (ci + 1 < pal_size) {
                    uint16_t c12 = (uint16_t)((pal[ci] << 8) | pal[ci + 1]);
                    row32[screen_col] = amiga12_to_argb(c12);
                    row16[screen_col] = c12;
                } else {
                    int shade = (gray * (int)texel) / 31;
                    uint32_t c = RENDER_RGB_RASTER_PIXEL(((uint32_t)shade << 16) | ((uint32_t)shade << 8) | (uint32_t)shade);
                    row32[screen_col] = c;
                    row16[screen_col] = argb_to_amiga12(c);
                }
            } else {
                /* No palette: use texel for shading so sprite shape is visible */
                int shade = (gray * (int)texel) / 31;
                uint32_t c = RENDER_RGB_RASTER_PIXEL(((uint32_t)shade << 16) | ((uint32_t)shade << 8) | (uint32_t)shade);
                row32[screen_col] = c;
                row16[screen_col] = argb_to_amiga12(c);
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Draw gun overlay
 *
 * Translated from AB3DI.s DrawInGun (lines 2426-2535).
 * Amiga: gun graphic from Objects+9, GUNYOFFS=20, 3 chunks x 32 = 96 wide,
 * 78-GUNYOFFS = 58 lines tall. If gun graphics are not loaded, nothing is drawn.
 * ----------------------------------------------------------------------- */
void renderer_draw_gun(GameState *state)
{
    uint8_t *buf = g_renderer.buffer;
    uint32_t *rgb = g_renderer.rgb_buffer;
    uint16_t *cw = g_renderer.cw_buffer;
    if (!buf || !rgb || !cw) return;

    int rw = g_renderer.width, rh = g_renderer.height;

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
            for (int sy = gy; sy < gy + gun_h_draw && sy < rh; sy++) {
                if (sy < 0) continue;
                int src_row = (int)((int64_t)(sy - gy) * (int64_t)gun_h_src / gun_h_draw);
                if (src_row >= gun_h_src) continue;
                for (int sx = gx; sx < gx + gun_w_draw && sx < rw; sx++) {
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
                    uint32_t c = amiga12_to_argb(c12);
                    buf[sy * rw + sx] = 15;
                    rgb[sy * rw + sx] = c;
                    cw[sy * rw + sx] = c12;
                }
            }
            return;
        }
    }

    /* Do not draw placeholder gun when real gun data is missing or slot unused */
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

/* RockPop/Explosion world sizes are authored from Amiga BitMapObj tables where
 * both axes use the same <<7 scale. This port projects billboard height with
 * proj_y_scale, so convert Amiga-authored explosion height into the port's
 * Y-scale domain before projection.
 *
 * This keeps explosion billboard sizing consistent across resize/aspect changes
 * without relying on a fixed heuristic multiplier. */
static inline int explosion_world_h_to_port(const RendererState *r, int world_h_amiga)
{
    int32_t py = r->proj_y_scale;
    if (py < 1) py = 1;
    int64_t corrected = ((int64_t)world_h_amiga * 128 + (py / 2)) / py;
    if (corrected < 1) corrected = 1;
    if (corrected > 32767) corrected = 32767;
    return (int)corrected;
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
static void draw_zone_objects(GameState *state, int16_t zone_id,
                              int32_t top_of_room, int32_t bot_of_room, int level_filter)
{
    RendererState *r = &g_renderer;
    LevelState *level = &state->level;
    if (!level->object_data || !level->object_points) return;

    int32_t y_off = r->yoff;

    /* Build one depth-sorted list for sprites and particles so painter order is consistent. */
    enum {
        DRAW_SRC_OBJECT = 0,
        DRAW_SRC_SHOT = 1,
        DRAW_SRC_EXPLOSION = 2
    };
    typedef struct {
        int src;
        int idx;
        int32_t z;
    } ObjEntry;
    ObjEntry objs[80 + 40 + MAX_EXPLOSIONS];
    const int max_draw_entries = (int)(sizeof(objs) / sizeof(objs[0]));
    int obj_count = 0;

    int num_pts = level->num_object_points;
    if (num_pts > MAX_OBJ_POINTS) num_pts = MAX_OBJ_POINTS;

    /* Object in_top at offset 63: 0 = lower floor, non-zero = upper floor (kept in sync by movement/objects). */
    const int obj_off_in_top = 63;

    /* Iterate by object index; each object has point number at offset 0 (ObjDraw: move.w (a0)+,d0).
     * Use that to look up ObjRotated[pt_num] so keys and other pickups use correct position/z. */
    for (int obj_idx = 0; obj_idx < 80 && obj_count < max_draw_entries; obj_idx++) {
        const uint8_t *obj = level->object_data + obj_idx * OBJECT_SIZE;
        int16_t pt_num = rd16(obj);
        if (pt_num < 0) break; /* End of object list */

        if ((unsigned)pt_num >= (unsigned)num_pts) continue; /* invalid point number */

        /* Only draw objects that are currently in this zone (obj_zone is updated by movement). */
        int16_t obj_zone = rd16(obj + 12);
        int in_this_zone = (obj_zone >= 0 && obj_zone == (int16_t)zone_id);
        if (!in_this_zone) continue;

        /* Multi-floor: only render objects when we are drawing their floor. Use object's in_top
         * so upper/lower is consistent with game logic (objects.c, movement). */
        if (level_filter >= 0) {
            int obj_on_upper = (obj[obj_off_in_top] != 0);
            if ((level_filter == 1 && !obj_on_upper) || (level_filter == 0 && obj_on_upper))
                continue;
        }

        ObjRotatedPoint *orp = &r->obj_rotated[pt_num];
        if (orp->z <= SPRITE_NEAR_CLIP_Z) continue; /* Amiga ObjDraw near clip */

        objs[obj_count].src = DRAW_SRC_OBJECT;
        objs[obj_count].idx = obj_idx;
        objs[obj_count].z = orp->z;
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
            if (rd16(obj + 12) < 0) continue; /* OBJ_ZONE */
            if ((int8_t)obj[16] != OBJ_NBR_BULLET) continue;
            int16_t pt_num = rd16(obj);
            if (pt_num < 0 || (unsigned)pt_num >= (unsigned)num_pts) continue;
            if (rd16(obj + 12) != (int16_t)zone_id) continue;
            if (level_filter >= 0) {
                int shot_on_upper = (obj[obj_off_in_top] != 0);
                if ((level_filter == 1 && !shot_on_upper) || (level_filter == 0 && shot_on_upper))
                    continue;
            }
            ObjRotatedPoint *orp = &r->obj_rotated[pt_num];
            if (orp->z <= SPRITE_NEAR_CLIP_Z) continue;
            objs[obj_count].src = DRAW_SRC_SHOT;
            objs[obj_count].idx = slot + ((pool == 0) ? 0 : NASTY_SHOT_SLOT_COUNT);
            objs[obj_count].z = orp->z;
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
            if (state->explosions[ei].zone != (int16_t)zone_id) continue;
            if (state->explosions[ei].start_delay > 0) continue;
            if ((int)state->explosions[ei].frame >= 9) continue;
            if (level_filter >= 0) {
                int expl_on_upper = (state->explosions[ei].in_top != 0);
                if ((level_filter == 1 && !expl_on_upper) || (level_filter == 0 && expl_on_upper))
                    continue;
            }

            int16_t dx = (int16_t)(state->explosions[ei].x - cam_x);
            int16_t dz = (int16_t)(state->explosions[ei].z - cam_z);
            int32_t vz = (int32_t)dx * sin_v + (int32_t)dz * cos_v;
            vz <<= 2;
            int32_t orp_z = (int32_t)(int16_t)(vz >> 16);
            if (orp_z <= SPRITE_NEAR_CLIP_Z) continue;

            objs[obj_count].src = DRAW_SRC_EXPLOSION;
            objs[obj_count].idx = ei;
            objs[obj_count].z = orp_z;
            obj_count++;
        }
    }

    /* Insertion sort by Z descending (farthest first - painter's algorithm).
     * Uses >= in the shift condition to match the original selection sort's
     * tie-breaking: equal-depth items end up in reverse-input order (later
     * entries draw first, i.e. appear earlier in the sorted array). */
    for (int i = 1; i < obj_count; i++) {
        ObjEntry key = objs[i];
        int j = i - 1;
        while (j >= 0 && objs[j].z <= key.z) {
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
    for (int oi = 0; oi < obj_count; oi++) {
        int entry_src = objs[oi].src;
        if (entry_src == DRAW_SRC_EXPLOSION) {
            int ei = objs[oi].idx;
            int16_t sin_v = r->sinval;
            int16_t cos_v = r->cosval;
            int16_t cam_x = r->xoff;
            int16_t cam_z = r->zoff;
            int scale = (int)state->explosions[ei].size_scale;
            if (scale <= 0) scale = 100;
            int frame_num = state->explosions[ei].frame;
            if (frame_num >= expl_ft_count) frame_num = expl_ft_count - 1;
            if (frame_num < 0) continue;
            /* Use Amiga RockPop world size progression (shot_size 2 pop frames 0-8)
             * instead of a fixed heuristic explosion size. */
            int expl_w = 100;
            int expl_h = 100;
            if (bullet_pop_tables[2]) {
                const BulletAnimFrame *pf = &bullet_pop_tables[2][frame_num];
                if (pf->width > 0 && pf->height > 0) {
                    expl_w = pf->width;
                    expl_h = pf->height;
                }
            }
            expl_w = (expl_w * scale) / 100;
            expl_h = (expl_h * scale) / 100;
            if (expl_w < 1) expl_w = 1;
            if (expl_h < 1) expl_h = 1;

            int16_t dx = (int16_t)(state->explosions[ei].x - cam_x);
            int16_t dz = (int16_t)(state->explosions[ei].z - cam_z);
            int32_t vx = (int32_t)dx * cos_v - (int32_t)dz * sin_v;
            vx <<= 1;
            int16_t vx16 = (int16_t)(vx >> 16);
            int32_t vz = (int32_t)dx * sin_v + (int32_t)dz * cos_v;
            vz <<= 2;
            int16_t vz16 = (int16_t)(vz >> 16);
            int32_t vx_fine = (int32_t)vx16 << 7;
            vx_fine += r->xwobble;
            int32_t orp_z = (int32_t)vz16;
            if (orp_z <= SPRITE_NEAR_CLIP_Z) continue;

            int center_x = (r->width * 47) / 96;
            int scr_x = (int)(vx_fine * RENDER_SCALE / (int32_t)orp_z) + center_x;
            int32_t rel_y_8 = (state->explosions[ei].y_floor - y_off) >> WORLD_Y_FRAC_BITS;
            int center_y = r->height / 2;
            int scr_y = (int)((int64_t)rel_y_8 * (int64_t)r->proj_y_scale * (int32_t)RENDER_SCALE / (int32_t)orp_z) + center_y;
            int z_for_size = orp_z;
            if (z_for_size < 1) z_for_size = 1;
            int expl_h_port = explosion_world_h_to_port(r, expl_h);
            int sprite_w = (int)((int32_t)expl_w * SPRITE_SIZE_SCALE / z_for_size) * SPRITE_SIZE_MULTIPLIER;
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

            int32_t clip_top_y = (int)((int64_t)((top_of_room - y_off) >> WORLD_Y_FRAC_BITS) * r->proj_y_scale * RENDER_SCALE / orp_z) + center_y;
            int32_t clip_bot_y = (int)((int64_t)((bot_of_room - y_off) >> WORLD_Y_FRAC_BITS) * r->proj_y_scale * RENDER_SCALE / orp_z) + center_y;
            int bright = (orp_z >> 7);
            const uint8_t *obj_pal = r->sprite_pal_data[expl_vect];
            size_t obj_pal_size = r->sprite_pal_size[expl_vect];
            renderer_draw_sprite((int16_t)scr_x, (int16_t)scr_y,
                                 (int16_t)sprite_w, (int16_t)sprite_h,
                                 (int16_t)(orp_z > 32767 ? 32767 : orp_z),
                                 r->sprite_wad[expl_vect], r->sprite_wad_size[expl_vect],
                                 r->sprite_ptr[expl_vect], r->sprite_ptr_size[expl_vect],
                                 obj_pal, obj_pal_size,
                                 ptr_off, down_strip,
                                 32, 32,
                                 (int16_t)bright, expl_vect,
                                 clip_top_y, clip_bot_y);
            continue;
        }

        int obj_idx = objs[oi].idx;
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
            draw_3d_vector_object(obj, orp3d, state);
            continue;
        }

        int16_t pt_num = rd16(obj);
        if ((unsigned)pt_num >= (unsigned)num_pts) continue;
        ObjRotatedPoint *orp = &r->obj_rotated[pt_num];

        /* Use actual view Z for size so sprites scale at all distances. Guard only vs div-by-zero. */
        int32_t z_for_size = orp->z;
        if (z_for_size < 1) z_for_size = 1;

        /* Project Y boundaries from room top/bottom (same PROJ_Y_SCALE as walls). */
        int32_t clip_top_y = (int)((int64_t)((top_of_room - y_off) >> WORLD_Y_FRAC_BITS) * g_renderer.proj_y_scale * RENDER_SCALE / orp->z) + (g_renderer.height / 2);
        int32_t clip_bot_y = (int)((int64_t)((bot_of_room - y_off) >> WORLD_Y_FRAC_BITS) * g_renderer.proj_y_scale * RENDER_SCALE / orp->z) + (g_renderer.height / 2);
        if (clip_top_y >= clip_bot_y) continue;

        /* Project to screen X (PROJ_X_SCALE/2 = horizontal focal length). */
        int32_t obj_vx_fine = orp->x_fine;
        int center_x = (g_renderer.width * 47) / 96;
        int scr_x = (int)(obj_vx_fine * RENDER_SCALE / (int32_t)orp->z) + center_x;

        /* Get brightness + distance attenuation
         * ASM: asr.w #7,d6 ; add.w (a0)+,d6 (distance>>7 + obj brightness) */
        /* Raw brightness d6 = (z >> 7) + objVectBright.
         * Passed to renderer_draw_sprite which uses objscalecols to map to palette level. */
        int16_t obj_bright = rd16(obj + 2);  /* objVectBright */
        int bright = (orp->z >> 7) + obj_bright;
        if (bright < 0) bright = 0;

        int8_t obj_number = (int8_t)obj[16];
        int16_t vect_num, frame_num;
        int drawing_dead = 0;
        if (obj_number == OBJ_NBR_DEAD) {
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
            switch ((ObjNumber)obj_number) {
                case OBJ_NBR_ALIEN:           vect_num = 0;  break;
                case OBJ_NBR_MEDIKIT:         vect_num = 1;  break;
                case OBJ_NBR_BULLET:          vect_num = 2;  break;
                case OBJ_NBR_BIG_GUN:         vect_num = 9;  break;
                case OBJ_NBR_KEY:             vect_num = 5;  break;
                case OBJ_NBR_PLR1:
                case OBJ_NBR_PLR2:
                case OBJ_NBR_MARINE:          vect_num = 10; break;
                case OBJ_NBR_ROBOT:           vect_num = 10; break;
                case OBJ_NBR_BIG_NASTY:       vect_num = 11; break;
                case OBJ_NBR_FLYING_NASTY:    vect_num = 4;  break;
                case OBJ_NBR_AMMO:            vect_num = 1;  break; /* PICKUPS sprite */
                case OBJ_NBR_BARREL:          vect_num = 7;  break;
                case OBJ_NBR_WORM:            vect_num = 13; break;
                case OBJ_NBR_HUGE_RED_THING:  vect_num = 14; break;
                case OBJ_NBR_SMALL_RED_THING: vect_num = 15; break;
                case OBJ_NBR_TREE:            vect_num = 15; break;
                case OBJ_NBR_EYEBALL:         vect_num = 0;  break;
                case OBJ_NBR_TOUGH_MARINE:    vect_num = 16; break;
                case OBJ_NBR_FLAME_MARINE:    vect_num = 17; break;
                default:                      vect_num = 0;  break;
            }
            if (!drawing_dead) frame_num = 0;
        }
        if (vect_num < 0 || vect_num >= MAX_SPRITE_TYPES) continue;

        /* ObjDraw3 uses move.b for width/height, i.e. unsigned bytes. */
        int world_w = (int)obj[6];
        int world_h = (int)obj[7];
        if (world_w <= 0) world_w = 32;
        if (world_h <= 0) world_h = 32;

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

        int sprite_w = (int)((int32_t)world_w * SPRITE_SIZE_SCALE / z_for_size) * SPRITE_SIZE_MULTIPLIER;
        int world_h_for_proj = world_h;
        if (explosion_billboard) {
            world_h_for_proj = explosion_world_h_to_port(&g_renderer, world_h);
        }
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
        int scr_y = (int)((int64_t)rel_y_8 * (int64_t)g_renderer.proj_y_scale * (int32_t)RENDER_SCALE / (int32_t)orp->z) + center_y;

        /* Use dedicated .pal if loaded; no fallback to WAD header because
         * sprite .pal format (15 levels x 32 x 2 bytes = 960) differs from
         * the wall LUT format in the WAD header (17 blocks x 32 x 2 = 2048). */
        const uint8_t *obj_pal = r->sprite_pal_data[vect_num];
        size_t obj_pal_size = r->sprite_pal_size[vect_num];

        renderer_draw_sprite((int16_t)scr_x, (int16_t)scr_y,
                             (int16_t)sprite_w, (int16_t)sprite_h,
                             (int16_t)(orp->z > 32767 ? 32767 : orp->z),
                             r->sprite_wad[vect_num], r->sprite_wad_size[vect_num],
                             r->sprite_ptr[vect_num], r->sprite_ptr_size[vect_num],
                             obj_pal, obj_pal_size,
                             ptr_off, down_strip,
                             src_cols, src_rows,
                             (int16_t)bright, vect_num,
                             clip_top_y, clip_bot_y);
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

void renderer_draw_zone(GameState *state, int16_t zone_id, int use_upper)
{
    RendererState *r = &g_renderer;
    LevelState *level = &state->level;

    if (!level->data || !level->zone_adds || !level->zone_graph_adds) return;

    /* Get zone data (same level->data that door_routine/lift_routine write to each frame). */
    int32_t zone_off = rd32(level->zone_adds + zone_id * 4);
    const uint8_t *zone_data = level->data + zone_off;

    /* Zone heights: upper room uses its own floor/roof stored at offsets +10/+14.
     * For lower room, ZD_FLOOR (2) and ZD_ROOF (6) are written each frame by door_routine and
     * lift_routine; re-read them when this zone is tagged as door or lift so we see the sine/lift. */
    int32_t zone_floor, zone_roof;
    if (use_upper) {
        int32_t uf = rd32(zone_data + 10);  /* ZD_UPPER_FLOOR */
        int32_t ur = rd32(zone_data + 14);  /* ZD_UPPER_ROOF  */
        /* Fallback: if upper floor/roof not set, use lower values */
        zone_floor = (uf != 0) ? uf : rd32(zone_data + 2);
        zone_roof  = (ur != 0) ? ur : rd32(zone_data + 6);
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
    if (gfx_off == 0 || !level->graphics) return;

    const uint8_t *gfx_data = level->graphics + gfx_off;
    int32_t zone_water = rd32(zone_data + 18);  /* ToZoneWater */

    int32_t y_off = r->yoff;
    PlayerState *viewer = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
    int16_t viewer_zone = viewer->zone;
    int half_h = g_renderer.height / 2;

    /* ObjDraw beforewat/afterwat clip bounds (ObjDraw3.ChipRam.s BEFOREWAT and AFTERWAT labels).
     * Bounds swap depending on whether the viewer is above or below the water plane. */
    int32_t before_wat_top, before_wat_bot, after_wat_top, after_wat_bot;
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

    int zone_has_door_flag = zone_has_door(level->door_data, zone_id);
    int zone_has_lift_flag = zone_has_lift(level->lift_data, zone_id);
    int has_door_wall_list = (level->door_wall_list && level->door_wall_list_offsets && level->num_doors > 0);
    int has_lift_wall_list = (level->lift_wall_list && level->lift_wall_list_offsets && level->num_lifts > 0);

    /* Read zone number from graphics data (consumed before polyloop) */
    const uint8_t *ptr = gfx_data;
    /* int16_t gfx_zone = rd16(ptr); */
    ptr += 2;

    /* Zone brightness from level data (no table); anim applied via level_get_zone_brightness. */
    int16_t zone_bright = 0;
    if (zone_id >= 0 && zone_id < level->num_zones)
        zone_bright = level_get_zone_brightness(level, zone_id, use_upper ? 1 : 0);


    /* Amiga: draw walls and arcs in stream order (no deferral). */

    int max_iter = 500; /* Safety limit */

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

                /* Switch walls (tex_id 11): same texture has on/off states.
                 * State is in wall first word bit 1 (p1 & 2): set = on, clear = off.
                 * Texture layout: off = left half (fromtile), on = right half (fromtile + 32).
                 * Use V offset 0 for switches so full texture maps consistently. */
                int16_t eff_totalyoff = (tex_id == SWITCHES_WALL_TEX_ID) ? 0 : (int16_t)(totalyoff + door_yoff_add);
                int16_t eff_fromtile   = fromtile;

                if (tex_id >= 0 && tex_id < MAX_WALL_TILES)
                    r->cur_wall_pal = r->wall_palettes[tex_id];
                else
                    r->cur_wall_pal = NULL;
                /* Amiga seethru path (type 13) clamps d6 to 32; normal walls clamp to 64. */
                int16_t wall_d6_max = (entry_type == 13) ? 32 : 64;
                if (!skip_this_wall)
                    renderer_draw_wall(rx1, rz1, rx2, rz2,
                                      wall_top, wall_bot,
                                      wall_tex, leftend, rightend,
                                      wall_bright_l, wall_bright_r,
                                      use_valand, use_valshift, horand,
                                      eff_totalyoff, eff_fromtile, tex_id,
                                      wall_height_for_tex, wall_d6_max);
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
                    g_fill_screen_water = 0x0F;
                }
                continue;
            }
            int32_t rel_h = draw_h_world - y_off; /* Relative to camera */

            /* Amiga fillscrnwater flag: mark underwater/half-submerged when drawing water in viewer zone. */
            if (entry_type == 7 && !use_upper && zone_id == viewer_zone) {
                int32_t rel_water = draw_h_world - y_off;
                if (rel_water < 0) {
                    g_fill_screen_water = 0x0F; /* strong underwater tint */
                } else if (rel_water <= (1 << WORLD_Y_FRAC_BITS) && g_fill_screen_water == 0) {
                    g_fill_screen_water = (int8_t)-1; /* weaker near-surface tint */
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
            int16_t *left_bright_tab = NULL;
            int16_t *right_bright_tab = NULL;
            if (use_gour_floor) {
                left_bright_tab = (int16_t*)malloc((size_t)h * sizeof(int16_t));
                right_bright_tab = (int16_t*)malloc((size_t)h * sizeof(int16_t));
            }
            if (!left_edge || !right_edge_tab ||
                (use_gour_floor && (!left_bright_tab || !right_bright_tab))) {
                free(left_edge);
                free(right_edge_tab);
                free(left_bright_tab);
                free(right_bright_tab);
                break;
            }
            for (int i = 0; i < h; i++) {
                left_edge[i] = (int16_t)g_renderer.width;
                right_edge_tab[i] = -1;
                if (use_gour_floor) {
                    left_bright_tab[i] = 0;
                    right_bright_tab[i] = 0;
                }
            }
            int poly_top = h;
            int poly_bot = -1;

            /* Clamp Y range for floor vs ceiling. Multi-floor: also clamp to zone top_clip/bot_clip
             * so lower room does not draw above the split and upper room does not draw below it. */
            int y_min_clamp, y_max_clamp;
            int top_clip_for_poly = r->top_clip;
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
            if (y_max_clamp > r->bot_clip) y_max_clamp = r->bot_clip;

            /* Walk each polygon edge and rasterize into edge tables (floor and ceiling/roof).
             * Near-plane clip edges so vertices behind the camera get proper
             * screen X values (otherwise on_screen[].screen_x is garbage and
             * the edge table doesn't reach the screen sides). */
            /* Water needs a tighter near clip than floor/roof so a surface just
             * above the camera can still project up to the top rows. */
            const int32_t FLOOR_NEAR = 1;
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
                        int32_t t = ((int32_t)(FLOOR_NEAR - ez1) << 16) / dz;
                        ex1 = rx1 + (int32_t)((int64_t)(rx2 - rx1) * t / 65536);
                        if (use_gour_floor) {
                            eb1 = eb1 + (int32_t)((int64_t)(eb2 - eb1) * t / 65536);
                        }
                    } else {
                        ex1 = (rx1 + rx2) / 2;  /* degenerate edge: use midpoint */
                        if (use_gour_floor) eb1 = (eb1 + eb2) / 2;
                    }
                    ez1 = FLOOR_NEAR;
                }
                if (ez2 < FLOOR_NEAR) {
                    int32_t dz = ez1 - ez2;
                    if (dz != 0) {
                        int32_t t = ((int32_t)(FLOOR_NEAR - ez2) << 16) / dz;
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
                int center_x = (g_renderer.width * 47) / 96;
                int sx1 = (int)((int64_t)ex1 * RENDER_SCALE / ez1) + center_x;
                int sx2 = (int)((int64_t)ex2 * RENDER_SCALE / ez2) + center_x;

                /* Project Y: same rule so X and Y stay consistent (no jump when z crosses FLOOR_NEAR). */
                int32_t rel_h_8 = rel_h >> WORLD_Y_FRAC_BITS;
                int sy1_raw = project_y_to_pixels_round(rel_h_8, ez1, r->proj_y_scale, center);
                int sy2_raw = project_y_to_pixels_round(rel_h_8, ez2, r->proj_y_scale, center);

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
                        if (lo < r->left_clip) lo = r->left_clip;
                        if (hi >= r->right_clip) hi = r->right_clip - 1;
                        if (lo < left_edge[row]) {
                            left_edge[row] = (int16_t)lo;
                            if (use_gour_floor) left_bright_tab[row] = (int16_t)lo_b;
                        }
                        if (hi > right_edge_tab[row]) {
                            right_edge_tab[row] = (int16_t)hi;
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
                    int x = (int)(x_fp >> 16);
                    int32_t edge_bright = (int32_t)(b_fp >> 16);
                    int left_x = x;
                    int right_x = x;
                    if (left_x < r->left_clip) left_x = r->left_clip;
                    if (right_x >= r->right_clip) right_x = r->right_clip - 1;
                    if (left_x < left_edge[row]) {
                        left_edge[row] = (int16_t)left_x;
                        if (use_gour_floor) left_bright_tab[row] = (int16_t)edge_bright;
                    }
                    if (right_x > right_edge_tab[row]) {
                        right_edge_tab[row] = (int16_t)right_x;
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
                if (row < r->top_clip || row > r->bot_clip) continue;  /* multi-floor: stay in zone band */
                int16_t le = left_edge[row];
                int16_t re = right_edge_tab[row];
                if (le >= g_renderer.width || re < 0) continue;
                if (le < r->left_clip) le = (int16_t)r->left_clip;
                if (re >= r->right_clip) re = (int16_t)(r->right_clip - 1);
                if (le > re) continue;
                /* Water (entry_type 7) and floor drawn inline in stream order (Amiga itsafloordraw). */
                int16_t water_rows_left = 0;
                if (entry_type == 7) {
                    int rows_left = (row_step > 0) ? (row_end - row + 1) : (row - row_end + 1);
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
                renderer_draw_floor_span((int16_t)row, le, re,
                                         rel_h, floor_tex, floor_pal,
                                         bright, row_bright_l, row_bright_r, row_use_gour,
                                         scaleval, (entry_type == 7) ? 1 : 0,
                                         water_rows_left);
            }
            free(left_edge);
            free(right_edge_tab);
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
            int is_multi_floor = (rd32(level->zone_graph_adds + zone_id * 8 + 4) != 0);
            draw_zone_objects(state, zone_id, obj_top, obj_bot,
                             is_multi_floor ? use_upper : -1);
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
                            r->cur_wall_pal = r->wall_palettes[tex_id];
                        else
                            r->cur_wall_pal = NULL;
                        renderer_draw_wall(prev_x, prev_z,
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
            /* putinbackdrop - no extra stream data. PC port draws sky once per frame
             * in renderer_draw_display (after clear), so this entry is a no-op here. */
            break;
        }

        default:
            /* Unknown type - skip nothing (type word already consumed) */
            break;
        }
    }
}

static void renderer_apply_underwater_tint(void)
{
    if (g_fill_screen_water == 0) return;
    if (!g_renderer.rgb_buffer || !g_renderer.cw_buffer) return;

    /* AB3DI fillscrnwater post-pass:
     * AND #$00FF on copper color words. Strong applies to whole view (4*20 lines),
     * weak applies to bottom half only (2*20 lines). */
    const int strong = (g_fill_screen_water > 0);
    const int w = g_renderer.width;
    const int h = g_renderer.height;
    int y0 = strong ? 0 : (h / 2);
    if (y0 < 0) y0 = 0;
    if (y0 > h) y0 = h;

    uint16_t *cw = g_renderer.cw_buffer;
    uint32_t *rgb = g_renderer.rgb_buffer;
    for (int y = y0; y < h; y++) {
        size_t row = (size_t)y * (size_t)w;
        for (int x = 0; x < w; x++) {
            size_t i = row + (size_t)x;
            uint16_t c12 = (uint16_t)(cw[i] & 0x00FFu);
            cw[i] = c12;
            rgb[i] = amiga12_to_argb(c12);
        }
    }
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
    if (!r->buffer) return;
    uint32_t frame_idx = g_render_frame_counter++;
    int trace_clip = renderer_take_clip_trace_slot();
    if (trace_clip) {
        printf("[CLIP][frame %u] begin\n", (unsigned)frame_idx);
    }

    /* 1. Clear framebuffer */
    renderer_clear(0);
    g_fill_screen_water = 0; /* Amiga DrawDisplay: clr.b fillscrnwater */

    /* Sky backdrop first (Amiga Anims.s putinbackdrop / data/gfx/backfile), before world draw. */
    {
        PlayerState *plr_sky = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
        renderer_draw_sky_pass((int16_t)plr_sky->angpos);
    }

    /* Vertical scale per frame: denominator scaled by screen aspect ratio (w/h vs default). */
    int w = (r->width  > 0) ? r->width  : 1;
    int h = (r->height > 0) ? r->height : 1;
    r->proj_y_scale = (int32_t)((int64_t)PROJ_Y_NUMERATOR / (int64_t)PROJ_Y_DENOM);
    //if (r->proj_y_scale < 1) r->proj_y_scale = 1;

    float screen_rescsale = ((float)640 / (float)h);
    r->proj_y_scale = (int32_t)((float)r->proj_y_scale / screen_rescsale);

    /* 2. Setup view transform (from AB3DI.s DrawDisplay lines 3399-3438) */
    PlayerState *plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;

    int16_t ang = (int16_t)(plr->angpos & 0x3FFF); /* 14-bit angle */
    r->sinval = sin_lookup(ang);
    r->cosval = cos_lookup(ang);

    /* Extract integer part of 16.16 fixed-point position for rendering.
     * On Amiga: .w operations on big-endian 32-bit values read the high word.
     * Raise camera Y for drawing only so floor appears further away (match Amiga). */
    r->xoff = (int16_t)(plr->xoff >> 16);
    r->zoff = (int16_t)(plr->zoff >> 16);
    r->yoff = plr->yoff - VIEW_HEIGHT_LIFT;

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
        memset(r->clip.top, 0, (size_t)w * sizeof(int16_t));
        int16_t bot_val = (int16_t)(h - 1);
        int16_t *bot = r->clip.bot;
        for (int i = 0; i < w; i++) bot[i] = bot_val;
        if (r->clip.z) memset(r->clip.z, 0, (size_t)w * sizeof(int32_t));
    }

    /* 4. Rotate geometry */
    renderer_rotate_level_pts(state);
    renderer_rotate_object_pts(state);

    /* 5. Amiga replication: painter's algorithm only, no depth buffer.
     *
     * Zones: drawn in OrderZones order (zone_order_zones[0]=farthest .. [n-1]=nearest),
     * so far zones are drawn first, near zones overwrite (painter's).
     *
     * Within each zone (AB3DI.s polyloop / DoThisRoom): strict stream order.
     * Each primitive is drawn as it appears (wall, floor, roof, arc, type-4 objects, etc.);
     * no deferral. Objects are drawn when the type-4 entry is encountered (Amiga ObjDraw).
     *
     * For each zone: apply LEVELCLIPS, then renderer_draw_zone (stream parse + draw).
     */
    const uint8_t *lgr_base = state->view_list_of_graph_rooms;

    for (int i = state->zone_order_count - 1; i >= 0; i--) {
        int16_t zone_id = state->zone_order_zones[i];
        if (zone_id < 0) continue;

        /* Reset clip to full screen for each zone */
        r->left_clip = 0;
        r->right_clip = (int16_t)g_renderer.width;
        r->top_clip = 0;
        r->bot_clip = (int16_t)(g_renderer.height - 1);
        r->wall_top_clip = -1;
        r->wall_bot_clip = -1;

        /* Apply zone clipping from ListOfGraphRooms + LEVELCLIPS (portal clipping).
         * Match Amiga NEWsetlclip/NEWsetrclip behavior:
         * - test clip point z > 0
         * - compare against CONNECT_TABLE paired point on screen
         * - right clip uses sx + 1
         * - skip zone when clip is invalid (no full-screen fallback) */
        {
            const uint8_t *lgr = lgr_base;
        if (!lgr || !state->level.clips) {
            continue;
        }
        {
            const uint8_t *connect_table = state->level.connect_table;
            int left_clip_px = 0;
            int right_clip_px = g_renderer.width;

            /* Find this zone's entry in ListOfGraphRooms */
            int found = 0;
            while (rd16(lgr) >= 0) {
                if (rd16(lgr) == zone_id) {
                    found = 1;
                    break;
                }
                lgr += 8;
            }

            if (found) {
                int16_t clip_off = rd16(lgr + 2);
                if (trace_clip) {
                    printf("[CLIP][frame %u] zone=%d clip_off=%d\n",
                           (unsigned)frame_idx, (int)zone_id, (int)clip_off);
                }
                if (clip_off >= 0) {
                    const uint8_t *clip_ptr = state->level.clips + clip_off * 2;

                    /* Left clips (Amiga NEWsetlclip) */
                    while (rd16(clip_ptr) >= 0) {
                        int16_t pt = rd16(clip_ptr);
                        clip_ptr += 2;
                        if (pt >= 0 && pt < MAX_POINTS) {
                            int16_t z = (int16_t)r->rotated[pt].z;
                            if (z > 0) {
                                int sxpx = project_x_to_pixels(r->rotated[pt].x, r->rotated[pt].z);
                                int allow = 1;
                                if (connect_table) {
                                    int16_t cpt = rd16(connect_table + (size_t)pt * 4u + 2u);
                                    if (cpt >= 0 && cpt < MAX_POINTS) {
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
                            }
                        }
                    }
                    clip_ptr += 2; /* Skip -1 terminator */

                    /* Right clips (Amiga NEWsetrclip) */
                    while (rd16(clip_ptr) >= 0) {
                        int16_t pt = rd16(clip_ptr);
                        clip_ptr += 2;
                        if (pt >= 0 && pt < MAX_POINTS) {
                            int16_t z = (int16_t)r->rotated[pt].z;
                            if (z > 0) {
                                int sxpx = project_x_to_pixels(r->rotated[pt].x, r->rotated[pt].z);
                                int allow = 1;
                                if (connect_table) {
                                    int16_t cpt = rd16(connect_table + (size_t)pt * 4u);
                                    if (cpt >= 0 && cpt < MAX_POINTS) {
                                        int csxpx = project_x_to_pixels(r->rotated[cpt].x, r->rotated[cpt].z);
                                        if (csxpx < sxpx) allow = 0;
                                    } else if (trace_clip) {
                                        printf("[CLIP][frame %u] zone=%d right pt=%d bad cpt=%d\n",
                                               (unsigned)frame_idx, (int)zone_id, (int)pt, (int)cpt);
                                    }
                                }
                                if (allow && sxpx < right_clip_px) {
                                    right_clip_px = sxpx + RENDER_SCALE;
                                }
                            }
                        }
                    }

                    /* Amiga dontbothercantseeit guard: skip this zone if clip is invalid. */
                    if (left_clip_px >= g_renderer.width || right_clip_px <= 0 ||
                        left_clip_px >= right_clip_px) {
                        if (trace_clip) {
                            printf("[CLIP][frame %u] zone=%d SKIP invalid lpx=%d rpx=%d\n",
                                   (unsigned)frame_idx, (int)zone_id,
                                   left_clip_px, right_clip_px);
                        }
                        continue;
                    }

                    if (left_clip_px < 0) left_clip_px = 0;
                    if (right_clip_px > g_renderer.width) right_clip_px = g_renderer.width;
                    r->left_clip = (int16_t)left_clip_px;
                    r->right_clip = (int16_t)right_clip_px;
                    if (trace_clip) {
                        printf("[CLIP][frame %u] zone=%d clip_px=[%d,%d)\n",
                               (unsigned)frame_idx, (int)zone_id,
                               (int)r->left_clip, (int)r->right_clip);
                    }
                }
            } else {
                /* Zone not present in ListOfGraphRooms: do not draw it. */
                if (trace_clip) {
                    printf("[CLIP][frame %u] zone=%d SKIP not_in_lgr\n",
                           (unsigned)frame_idx, (int)zone_id);
                }
                continue;
            }
        }
        }

        /* Multi-floor zone ordering (Amiga DrawDisplay, AB3DI.s 3543-3630):
         *   if (yoff >= ToZoneRoof): draw upper first, then lower
         *   else:                    draw lower first, then upper
         * No screen-space split clipping is applied in the original path. */
        if (state->level.zone_adds && state->level.zone_graph_adds) {
            int32_t zone_off = rd32(state->level.zone_adds + zone_id * 4);
            const uint8_t *zgraph = state->level.zone_graph_adds + zone_id * 8;
            int32_t upper_gfx = rd32(zgraph + 4);
            if (upper_gfx != 0 && zone_off >= 0 && state->level.data) {
                const uint8_t *zd = state->level.data + zone_off;
                int32_t split_height = rd32(zd + 6);  /* ToZoneRoof */
                /* Amiga compares SplitHeight against yoff that is loaded directly from PLR*_yoff.
                 * Keep this compare in raw player space (no render-only view lift offset). */
                int draw_upper_first = (plr->yoff >= split_height);

                if (draw_upper_first) {
                    renderer_draw_zone(state, zone_id, 1);  /* upper */
                    renderer_draw_zone(state, zone_id, 0);  /* lower */
                } else {
                    renderer_draw_zone(state, zone_id, 0);  /* lower */
                    renderer_draw_zone(state, zone_id, 1);  /* upper */
                }
                continue;
            }
        }
        renderer_draw_zone(state, zone_id, 0);
    }

    /* 6. Draw gun overlay */
    renderer_draw_gun(state);
    renderer_apply_underwater_tint();

    /* 7. Swap buffers (the just-drawn buffer becomes the display buffer) */
    renderer_swap();
}

