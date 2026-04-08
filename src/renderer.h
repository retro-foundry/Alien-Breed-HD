/*
 * Alien Breed 3D I - PC Port
 * renderer.h - Software 3D renderer (chunky buffer)
 *
 * Translated from: AB3DI.s (DrawDisplay, RotateLevelPts, RotateObjectPts),
 *                  WallRoutine3.ChipMem.s, ObjDraw3.ChipRam.s, BumpMap.s
 *
 * The renderer draws into a chunky (8-bit indexed) pixel buffer.
 * Displaying that buffer to the actual screen is platform-specific
 * and handled by the display backend (SDL/OpenGL, etc.).
 */

#ifndef RENDERER_H
#define RENDERER_H

#include "game_state.h"
#include <stdint.h>
#include <stddef.h>

/* -----------------------------------------------------------------------
 * Framebuffer dimensions (from AB3DI.s)
 *
 * The Amiga version renders 96 columns of game view (columns 0-95)
 * into a buffer that is 104 longwords wide (416 bytes).
 * Screen height is 80 lines.
 * -----------------------------------------------------------------------*/
#define RENDER_SCALE     8     /* Resolution multiplier (1 = Amiga); halve display window scale when doubling so window size unchanged */
#define RENDER_SCALE_LOG2 3    /* log2(RENDER_SCALE); keep in sync when changing RENDER_SCALE (1->0, 2->1, 4->2, 8->3) */
#define RENDER_WIDTH         (96  * RENDER_SCALE)   /* Visible game columns */
#define RENDER_HEIGHT        (80  * RENDER_SCALE)   /* Visible game lines   */
#define RENDER_DEFAULT_WIDTH  RENDER_WIDTH          /* Default/resize reference width  */
#define RENDER_DEFAULT_HEIGHT RENDER_HEIGHT         /* Default/resize reference height */
/* Maximum internal framebuffer width/height (base size × supersampling). */
#define RENDER_INTERNAL_MAX_DIM 8192
/* Sky filtering for HD output.
 * 1 = bilinear sample (smoother in high resolutions), 0 = nearest (Amiga-style blocky). */
#ifndef SKY_BILINEAR_FILTER
#define SKY_BILINEAR_FILTER 1
#endif
/* 1 = draw sky backdrop (backfile / procedural); 0 = skip sky. */
#ifndef RENDER_SKY
#define RENDER_SKY 1
#endif
/* 1 = clear framebuffer (renderer_clear) and empty-canvas fills when sky is off; 0 = leave buffer. */
#ifndef RENDER_CLEAR
#define RENDER_CLEAR 0
#endif

/* Wall vertical texture: divisor so one world-space wall height maps to one texture repeat at any resolution. */
#define WALL_VERTICAL_TEX_DIVISOR  ((int32_t)PROJ_Y_SCALE * (int32_t)RENDER_SCALE)
#define RENDER_STRIDE    RENDER_WIDTH            /* Bytes per line (1 byte per pixel tag) */
#define RENDER_BUF_SIZE  (RENDER_STRIDE * RENDER_HEIGHT)
#define RENDER_STRIDE    RENDER_WIDTH   /* stride = width; runtime uses renderer width */
#define RENDER_BUF_SIZE  (RENDER_STRIDE * RENDER_HEIGHT)

/* Alpha tagging for the ARGB game buffer:
 *   RENDER_RGB_SKY_MASK_A (1) = cleared / empty sky.
 *   RENDER_RGB_RASTER_A (0)  = anything drawn by the 3D rasterizer.
 * display.c forces A=0xFF when uploading to SDL (RGB unchanged). */
#ifndef RENDER_RGB_SKY_MASK_A
#define RENDER_RGB_SKY_MASK_A 0x01u
#endif
#define RENDER_RGB_CLEAR_SKY_PIXEL \
    ((uint32_t)(((uint32_t)RENDER_RGB_SKY_MASK_A << 24) | 0x00EEEEEEu))

#ifndef RENDER_RGB_RASTER_A
#define RENDER_RGB_RASTER_A 0x00u
#endif
#define RENDER_RGB_RASTER_PIXEL(rgb24) \
    (((uint32_t)RENDER_RGB_RASTER_A << 24) | ((uint32_t)(rgb24) & 0x00FFFFFFu))

/* Projection scales.
 *
 * PROJ_X_SCALE: horizontal projection (256 = Amiga default). The horizontal
 *   focal length is PROJ_X_SCALE / 2 (= 128). Do not change unless you want
 *   to alter horizontal FOV.
 *
 * PROJ_Y: vertical projection. PROJ_Y_NUMERATOR / PROJ_Y_DENOM gives base scale;
 *   the divide by PROJ_Y_DENOM is applied per frame with height scaling. */
#define PROJ_X_SCALE     256
#define PROJ_Y_NUMERATOR (256 * 15)
#define PROJ_Y_DENOM     38                /* adjust this value to change the vertical FOV, can need tweaking */
#define PROJ_Y_SCALE     (PROJ_Y_NUMERATOR / PROJ_Y_DENOM)

/* World Y fixed-point: zone/floor/object heights use this many fractional bits for projection. */
#define WORLD_Y_FRAC_BITS  8
#define WORLD_Y_SUBUNITS   (1 << WORLD_Y_FRAC_BITS)

/* Rotation Z precision: view-space Z stored as 24.8 fixed-point instead of integer.
 * Eliminates single-integer-step jitter in wall edges, floor/ceiling boundaries,
 * and sprite positions that was visible at high render resolutions. */
#define ROT_Z_FRAC_BITS    8
#define ROT_Z_ONE           (1 << ROT_Z_FRAC_BITS)
#define ROT_Z_INT(z)        ((int32_t)(z) >> ROT_Z_FRAC_BITS)
#define ROT_Z_FROM_INT(i)   ((int32_t)(i) << ROT_Z_FRAC_BITS)

/* Sprite size: (world * SPRITE_SIZE_SCALE / z) * SPRITE_SIZE_MULTIPLIER.
 * Keep multiplier at 1 so billboard scale matches the existing projection path. */
#define SPRITE_SIZE_SCALE      (128 * RENDER_SCALE)
#define SPRITE_SIZE_MULTIPLIER 1
/* Extra scalar on top of the Amiga->port explosion size conversion.
 * Keep at 1 for exact conversion; >1 or <1 is an explicit gameplay/style tweak. */
#define EXPLOSION_SIZE_CORRECTION 1

/* -----------------------------------------------------------------------
 * Rotated point arrays
 *
 * RotateLevelPts writes to Rotated[] and OnScreen[].
 * RotateObjectPts writes to ObjRotated[].
 * ----------------------------------------------------------------------- */
#define MAX_POINTS       2048
#define MAX_OBJ_POINTS   1024

typedef struct {
    int32_t x;       /* View-space X (scaled, high precision: vx>>9 + wobble) */
    int32_t z;       /* View-space Z (depth, 24.8 fixed-point — use ROT_Z_INT for integer) */
} RotatedPoint;

typedef struct {
    int16_t screen_x; /* Screen column */
    int16_t flags;    /* Behind camera, etc. */
} OnScreenPoint;

typedef struct {
    int16_t x;       /* View-space X (16-bit, legacy — prefer x_fine) */
    int32_t z;       /* View-space Z (depth, 24.8 fixed-point — use ROT_Z_INT for integer) */
    int32_t x_fine;  /* View-space X (high precision: vx>>9 + wobble) */
} ObjRotatedPoint;

/* -----------------------------------------------------------------------
 * Per-column clipping table
 *
 * Each column stores up to two wall spans (top/bot + depth). This keeps
 * sprite-vs-wall occlusion lightweight while handling split upper/lower
 * coverage better than a single collapsed span.
 * Allocated with width elements (renderer width).
 * ----------------------------------------------------------------------- */
typedef struct {
    int16_t *top;
    int16_t *bot;
    int32_t *z;    /* span 0 depth; 0 = invalid span */
    int16_t *top2;
    int16_t *bot2;
    int32_t *z2;   /* span 1 depth; 0 = invalid span */
} ColumnClip;

/* -----------------------------------------------------------------------
 * Wall texture table (walltiles)
 *
 * From WallChunk.s: 40 texture slots.
 * Each texture is 64x32 palette (2048 bytes) + 64x32 chunky (2048 bytes).
 * PaletteAddr = walltiles[id]
 * ChunkAddr   = walltiles[id] + 64*32
 * ----------------------------------------------------------------------- */
#define MAX_WALL_TILES  40
#define WALL_TEX_SIZE   (64 * 32 * 2)  /* palette + pixels */

/* -----------------------------------------------------------------------
 * Object/sprite graphics (Objects table from ObjDraw3.ChipRam.s)
 *
 * objVectNumber (object data offset 8) indexes this table.
 * Each slot can have a .wad (chunky pixels); we use 32x32 per frame.
 * ----------------------------------------------------------------------- */
#define MAX_SPRITE_TYPES  20
#define MAX_SPRITE_FRAMES 40  /* max frames per sprite type */

/* -----------------------------------------------------------------------
 * Renderer state
 * ----------------------------------------------------------------------- */
typedef struct {
    /* Current framebuffer size (matches window when resizable) */
    int width;
    int height;
    /* Actual display/output size requested by SDL. Can differ from framebuffer
     * when renderer clamps to an internal max dimension. */
    int present_width;
    int present_height;

    /* Vertical projection scale: numerator / (PROJ_Y_DENOM * width/RENDER_DEFAULT_WIDTH) * height/RENDER_DEFAULT_HEIGHT, recomputed each frame. */
    int32_t proj_y_scale;
    /* Floor/ceiling UV distance clamps (pastfloorbright): scale with ab3d.ini y_proj_scale so values >100% are not stuck at the default cap. */
    int32_t floor_uv_dist_max;
    int32_t floor_uv_dist_near;

    /* Framebuffer (double-buffered); base pointers are cache-line-aligned. */
    uint8_t *buffer;          /* Current render target (width * height bytes) */
    uint8_t *back_buffer;     /* Back buffer for swap */

    /* 32-bit ARGB framebuffer (double-buffered).
     * Size: width * height * sizeof(uint32_t). */
    uint32_t *rgb_buffer;
    uint32_t *rgb_back_buffer;

    /* 12-bit Amiga color-word framebuffer (double-buffered).
     * Each pixel stores the final 0x0RGB word used for water byte sampling. */
    uint16_t *cw_buffer;
    uint16_t *cw_back_buffer;

    /* View transform */
    int16_t sinval, cosval;   /* Sin/cos of view angle */
    int16_t xoff, zoff;       /* Camera X/Z */
    int32_t yoff;             /* Camera Y (fixed) */
    int16_t wallyoff;         /* Wall texture Y offset */
    int16_t flooryoff;        /* Floor Y offset */
    int16_t xoff34, zoff34;   /* 3/4 offsets */
    int32_t xwobble;          /* Head bob X wobble */
    int16_t sky_frame_angpos; /* sky pan angle for current frame (backdrop ceiling polys) */

    /* Rotated geometry */
    RotatedPoint  rotated[MAX_POINTS];
    OnScreenPoint on_screen[MAX_POINTS];
    /* Per-point frame stamp: non-zero when rotated[]/on_screen[] for that
     * index were produced in rotate_stamp frame. */
    uint32_t      rotated_stamp[MAX_POINTS];
    uint32_t      rotate_stamp;
    ObjRotatedPoint obj_rotated[MAX_OBJ_POINTS];

    /* Column clipping */
    ColumnClip clip;

    /* Zone rendering state */
    int16_t left_clip;
    int16_t right_clip;
    int16_t top_clip;
    int16_t bot_clip;
    /* Multi-floor: optional extra clip for walls only. -1 = use normal top_clip/bot_clip. */
    int16_t wall_top_clip;  /* lower room: 0 so walls can extend to top when very close */
    int16_t wall_bot_clip;  /* upper room: split_y so walls meet floor at split */

    /* Current room rendering state */
    int32_t top_of_room;
    int32_t bot_of_room;
    int32_t split_height;

    /* Wall texture table (from WallChunk.s walltiles) */
    const uint8_t *walltiles[MAX_WALL_TILES]; /* Pointers to pixel data (past 2048-byte LUT) */

    /* Object sprite graphics (from LoadFromDisk.s LoadObjects, ObjDraw3 Objects).
     *
     * Amiga format: .wad contains packed pixel data (3 five-bit pixels per
     * 16-bit word).  .ptr contains column pointer table (4 bytes per column:
     * byte 0 = which "third" 0/1/2, bytes 1-3 = offset into .wad).
     * .pal contains brightness-graded palette (15 levels × 32 colors × 2 bytes).
     *
     * sprite_wad[vect] = raw .wad pixel data
     * sprite_ptr[vect] = column pointer table (.ptr file)
     * sprite_pal_data[vect] = brightness palette (.pal file) */
    const uint8_t *sprite_wad[MAX_SPRITE_TYPES];
    size_t         sprite_wad_size[MAX_SPRITE_TYPES];
    const uint8_t *sprite_ptr[MAX_SPRITE_TYPES];
    size_t         sprite_ptr_size[MAX_SPRITE_TYPES];
    const uint8_t *sprite_pal_data[MAX_SPRITE_TYPES];
    size_t         sprite_pal_size[MAX_SPRITE_TYPES];

    /* Wall palette/LUT table.
     * Each entry points to the 2048-byte brightness LUT at the START of
     * the .wad file data.  Indexed as:
     *   color_word = lut[SCALE[d6] + texel5 * 2]  (big-endian 16-bit)
     * The word is a 12-bit Amiga color (0x0RGB). */
    const uint8_t *wall_palettes[MAX_WALL_TILES];

    /* Wall texture dimensions from loaded file (rows = 1<<wall_valshift[i], valand = rows-1).
     * Set by io_load_walls from actual file size; 0 means use level data. */
    uint8_t wall_valand[MAX_WALL_TILES];
    uint8_t wall_valshift[MAX_WALL_TILES];

    /* Current wall palette pointer (set per-wall in draw_zone) */
    const uint8_t *cur_wall_pal;

    /* Floor tile texture (from floortile - 256x256 sheet, 8-bit texels).
     * Individual 64x64 tiles are at floortile + whichtile offset. */
    const uint8_t *floor_tile;

    /* Floor brightness LUT (from FloorPalScaled - 15 levels * 512 bytes).
     * Maps texel -> brightness-scaled color. NULL if not loaded. */
    const uint8_t *floor_pal;

    /* Bump-mapped floor assets (AB3DI BumpLine path for types 8-11). */
    const uint8_t *bump_tile;         /* data/gfx/BumpTile (chunky floors) */
    const uint8_t *smooth_bump_tile;  /* data/gfx/SmoothBumpTile (smooth bumps) */
    const uint8_t *bump_pal;          /* data/pal/BumpPalScaled */
    const uint8_t *smooth_bump_pal;   /* data/pal/SmoothBumpPalScaled */

    /* Gun overlay (newgunsinhand.wad + .ptr + .pal). NULL if not loaded. */
    const uint8_t *gun_wad;   /* raw graphic data */
    const uint8_t *gun_ptr;   /* 96 columns × 4 bytes (mode + 24-bit offset per column) per frame */
    const uint8_t *gun_pal;   /* 32 × 16-bit 12-bit Amiga color (64 bytes) */
    size_t         gun_wad_size;

    /* Palette (legacy, used as fallback) */
    uint32_t palette[256];
} RendererState;

/* Global renderer instance */
extern RendererState g_renderer;

/* -----------------------------------------------------------------------
 * API
 * ----------------------------------------------------------------------- */

/* Initialize renderer (allocate buffers) */
void renderer_init(void);

/* Shutdown renderer (free buffers) */
void renderer_shutdown(void);

/* Resize framebuffer (call on window resize) */
void renderer_resize(int w, int h);

/* Clear the framebuffer to a color */
void renderer_clear(uint8_t color);

/* Sky backdrop (Amiga data/gfx/backfile). Drawn first each frame after clear.
 * Standard backfile: 32832 bytes = 432 x 38 x 2 (big-endian 12-bit words), interpreted in
 * strict Amiga column-major parity mode. Optional rgb_palette_768 is only for non-standard
 * 8-bit indexed sky assets (tex_w*tex_h == data_bytes). */
void renderer_set_sky_assets(const uint8_t *chunky_pixels, int tex_w, int tex_h, size_t data_bytes,
                             const uint8_t *rgb_palette_768);

/* Swap front/back buffers */
void renderer_swap(void);

/* Main entry point: renders the full 3D scene to the chunky buffer.
 * Translated from AB3DI.s DrawDisplay.
 * After this call, g_renderer.buffer contains the rendered frame. */
void renderer_draw_display(GameState *state);
/* Build/reset synthesized backdrop sky-hole polygons for the currently loaded level.
 * Build once after level parse/load, then draw from cache each frame. */
void renderer_build_level_sky_cache(const LevelState *level);
void renderer_reset_level_sky_cache(void);

/* Optional water assets from Amiga helper files.
 * Pass NULL/0 to disable and use fallback water shading. */
void renderer_set_water_assets(const uint8_t *water_file, size_t water_file_size,
                               const uint8_t *water_brighten, size_t water_brighten_size);
void renderer_step_water_anim(int steps);
void renderer_step_water_anim_ms(uint32_t elapsed_ms);
int renderer_toggle_floor_gouraud_debug_view(void);
int renderer_get_floor_gouraud_debug_view(void);
/* Automap: change world-units-per-pixel (PgUp = zoom in, PgDn = zoom out). */
void renderer_automap_adjust_scale(int delta_steps);
/* Stable dedupe key for one seen wall segment (stored as key+1; 0 reserved empty). */
uint32_t renderer_automap_seen_key_plus1(uint32_t gfx_off,
                                         int16_t x1, int16_t z1,
                                         int16_t x2, int16_t z2);
/* Serialize automap hash/wall list (threaded world draw vs overlay / level unload). */
void renderer_automap_lock(void);
void renderer_automap_unlock(void);
/* Collect line segments in internal render pixel coords (for SDL overlay in display.c). */
/* c12 uses low 12 bits for Amiga color and optional segment flags in high bits. */
#define RENDERER_AUTOMAP_SEGFLAG_INTERNAL 0x8000u
int renderer_automap_collect_line_segments(GameState *state,
                                           int *x0, int *y0, int *x1, int *y1,
                                           uint16_t *c12, int max_lines);
/* HUD: Amiga12 tint for key condition bit 0..3 (same palette as automap key sprites). */
uint16_t renderer_key_condition_bit_color_c12(const GameState *state, int bit_index);
/* HUD: key sprite frame index 0..3 for condition bit (from level key objects / fallback). */
int renderer_key_sprite_frame_for_condition_bit(const GameState *state, int bit_index);
uintptr_t renderer_key_sprite_hud_cache_tag(const GameState *state);
/* Rasterize key sprite frame to 32 rows; stride_pixels >= 32; ARGB with alpha 0 = transparent. */
int renderer_key_sprite_rasterize_frame_argb(int frame_index, uint32_t *out, int stride_pixels);
/* Return nearest wall-span depth at (col,row), or 0 when no wall span covers that pixel. */
int32_t renderer_column_clip_nearest_z_at(int col, int row);

/* Sub-routines (called by draw_display) */
void renderer_rotate_level_pts(GameState *state);
void renderer_rotate_object_pts(GameState *state);
void renderer_draw_zone(GameState *state, int16_t zone_id, int use_upper);
void renderer_draw_wall(int32_t x1, int32_t z1, int32_t x2, int32_t z2,
                        int16_t top, int16_t bot,
                        const uint8_t *texture, int16_t tex_start,
                        int16_t tex_end, int16_t left_brightness, int16_t right_brightness,
                        uint8_t valand, uint8_t valshift, int16_t horand,
                        int16_t totalyoff, int16_t fromtile,
                        int16_t tex_id, int16_t wall_height_for_tex,
                        int16_t d6_max);
void renderer_draw_floor_span(int16_t y, int16_t x_left, int16_t x_right,
                              int32_t floor_height, const uint8_t *texture, const uint8_t *floor_pal,
                              int16_t brightness, int16_t left_brightness, int16_t right_brightness,
                              int16_t use_gouraud,
                              int16_t scaleval, int is_water,
                              int16_t water_rows_left);
void renderer_draw_sprite(int16_t screen_x, int16_t screen_y,
                          int16_t width, int16_t height, int16_t z,
                          const uint8_t *wad, size_t wad_size,
                          const uint8_t *ptr_data, size_t ptr_size,
                          const uint8_t *pal, size_t pal_size,
                          uint32_t ptr_offset, uint16_t down_strip,
                          int src_cols, int src_rows,
                          int16_t brightness, int sprite_type,
                          int32_t clip_top_sy, int32_t clip_bot_sy);
void renderer_draw_gun(GameState *state);

/* Get pointer to the current rendered frame for display */
const uint8_t *renderer_get_buffer(void);
const uint32_t *renderer_get_rgb_buffer(void);
/* Completed frame: 16-bit packed 4444, 0x0RGB (high nibble unused). Matches SDL ARGB4444 with A=0. */
const uint16_t *renderer_get_cw_buffer(void);
uint32_t *renderer_get_active_rgb_target(void);
uint16_t *renderer_get_active_cw_target(void);
/* When zero (default), raster skips per-pixel ARGB expansion; cw_buffer is authoritative for display. */
void renderer_set_rgb_raster_expand(int enabled);
int renderer_get_rgb_raster_expand(void);
uint16_t renderer_argb_to_amiga12(uint32_t argb);
int renderer_get_width(void);
int renderer_get_height(void);
int renderer_get_stride(void);
void renderer_set_present_size(int w, int h);

#endif /* RENDERER_H */
