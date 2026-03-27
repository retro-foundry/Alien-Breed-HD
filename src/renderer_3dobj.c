/*
 * Alien Breed 3D I - PC Port
 * renderer_3dobj.c - 3D vector/polygon object rendering
 *
 * Translated from ObjDraw3.ChipRam.s PolygonObj / doapoly.
 *
 * Pipeline:
 *   1. Read objVectFacing (obj[30]) and viewer angle → compute relative angle.
 *   2. For each vertex in the active frame: rotate around Y axis using the
 *      relative angle (matching the Amiga rotobj loop).
 *   3. Translate+project each rotated vertex to screen (matching convtoscr).
 *   4. Depth-sort polygon parts by their sort-point's Z distance from camera
 *      (matching PutinParts / insertion-sort into 32-slot buffer).
 *   5. For each part, iterate its polygon list (matching polyloo).
 *      For each polygon: decode edge-stream perimeter, back-face cull, then
 *      flat-shade filled scan-line fill.
 */

#include "renderer_3dobj.h"
#include "renderer.h"
#include "math_tables.h"
#include "game_types.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include "logging.h"
#define printf ab3d_log_printf

/* -----------------------------------------------------------------------
 * Global POLYOBJECTS table
 * ----------------------------------------------------------------------- */
VecObject g_poly_objects[POLY_OBJECTS_COUNT];

/* Shared polygon texture assets (provided by io.c). */
static const uint8_t *g_poly_tex_maps = NULL;
static size_t         g_poly_tex_maps_size = 0;
static const uint8_t *g_poly_tex_pal = NULL;
static size_t         g_poly_tex_pal_size = 0;
static int32_t       *g_poly_obj_depth = NULL;
static size_t         g_poly_obj_depth_cap = 0;
/* Generation counter: increment per object instead of clearing the depth buffer.
 * A depth entry is only valid when its stamp matches g_depth_gen. */
static uint32_t      *g_poly_obj_depth_gen = NULL;
static uint32_t       g_depth_gen = 0;
/* Default in PC port: animated object frame enabled (can be forced static via CLI). */
static int            g_poly_use_object_frame = 1;

void poly_obj_set_texture_assets(const uint8_t *texture_maps, size_t texture_maps_size,
                                 const uint8_t *texture_pal, size_t texture_pal_size)
{
    g_poly_tex_maps = texture_maps;
    g_poly_tex_maps_size = texture_maps_size;
    g_poly_tex_pal = texture_pal;
    g_poly_tex_pal_size = texture_pal_size;
}

void poly_obj_set_use_object_frame(int enabled)
{
    g_poly_use_object_frame = enabled ? 1 : 0;
}

/* -----------------------------------------------------------------------
 * Big-endian read helper
 * ----------------------------------------------------------------------- */
static inline int16_t vec_rd16(const uint8_t *p)
{
    return (int16_t)(((uint16_t)p[0] << 8) | p[1]);
}

/* -----------------------------------------------------------------------
 * poly_obj_load: parse a .vec binary into a slot
 * ----------------------------------------------------------------------- */
int poly_obj_load(int slot, const uint8_t *data, size_t size)
{
    if (slot < 0 || slot >= POLY_OBJECTS_COUNT) return 0;
    VecObject *vo = &g_poly_objects[slot];
    memset(vo, 0, sizeof(*vo));

    if (!data || size < 6) return 0;

    vo->data = data;
    vo->size = size;
    vo->num_points = (int)((uint16_t)((data[0] << 8) | data[1]));
    vo->num_frames = (int)((uint16_t)((data[2] << 8) | data[3]));

    if (vo->num_points <= 0 || vo->num_frames <= 0) return 0;
    if (vo->num_points > MAX_POLY_POINTS) vo->num_points = MAX_POLY_POINTS;
    int nf = vo->num_frames;
    if (nf > MAX_POLY_FRAMES) nf = MAX_POLY_FRAMES;
    vo->num_frames = nf;

    /* Frame pointer table at offset 4 */
    for (int f = 0; f < nf; f++) {
        int off = 4 + f * 2;
        if (off + 2 > (int)size) { vo->num_frames = f; break; }
        vo->frame_off[f] = (uint16_t)(((uint16_t)data[off] << 8) | data[off + 1]);
    }

    /* Part list starts at offset 4 + num_frames*2 */
    int part_off = 4 + nf * 2;
    int num_parts = 0;
    while (part_off + 4 <= (int)size && num_parts < MAX_POLY_PARTS) {
        uint16_t poly_data_off = (uint16_t)(((uint16_t)data[part_off] << 8) | data[part_off + 1]);
        if (poly_data_off == 0xFFFFu) break;
        uint16_t sort_off = (uint16_t)(((uint16_t)data[part_off + 2] << 8) | data[part_off + 3]);
        vo->part_poly_off[num_parts] = poly_data_off;
        vo->part_sort_off[num_parts] = sort_off;
        num_parts++;
        part_off += 4;
    }
    vo->num_parts = num_parts;

    printf("[3DOBJ] Slot %d: %d pts, %d frames, %d parts (size=%zu)\n",
           slot, vo->num_points, vo->num_frames, vo->num_parts, size);
    return 1;
}

/* -----------------------------------------------------------------------
 * Per-object base colours (approximated from Amiga visuals).
 * Index matches POLYOBJECTS slot: 0=robot, 1=medipac, 2=exitsign, 3=crate,
 * 4=terminal, 5=blueind, 6=greenind, 7=redind, 8=yellowind, 9=gaspipe.
 * ----------------------------------------------------------------------- */
static const uint32_t obj_base_rgb[POLY_OBJECTS_COUNT] = {
    0xBBBBBBu,   /* 0 robot     - metallic grey */
    0xEEEEEEu,   /* 1 medipac   - white         */
    0xFFCC22u,   /* 2 exitsign  - amber          */
    0xBB7733u,   /* 3 crate     - warm brown     */
    0x88AA88u,   /* 4 terminal  - grey-green     */
    0x3366FFu,   /* 5 blueind   - blue           */
    0x33CC44u,   /* 6 greenind  - green          */
    0xFF3333u,   /* 7 redind    - red            */
    0xFFEE22u,   /* 8 yellowind - yellow         */
    0x999999u,   /* 9 gaspipe   - pipe grey      */
};

/*
 * texture_map_index tint table.
 * The Amiga stores a texture_map_index word in each polygon's extra data.
 * We use it to pick a face tint (multiplied with the base colour so objects
 * have recognisable multi-colour detail – e.g. the yellow indicator's light-cap
 * appears brighter than its base).
 * Values are 0..255 per channel stored as 0xRRGGBB.
 */
static uint32_t tex_tint_from_word(uint16_t tex_map_word)
{
    switch (tex_map_word) {
        case 0x0000: return 0xB8B8B8u; /* neutral metal/pipe */
        case 0x0001: return 0xD8D8D8u;
        case 0x0002: return 0xFFD080u;
        case 0x0003: return 0x9A9A9Au;
        case 0x0101: return 0xF0F0F0u;
        case 0x0102: return 0xFFFFFFu; /* indicator face/body */
        case 0x0202: return 0xBBDDBBu;
        case 0x0203: return 0xB6B6B6u; /* indicator cap/rim */
        case 0x0300: return 0xB0B8C8u;
        case 0x0301: return 0xE0D0A0u;
        case 0x0302: return 0xC18A56u;
        case 0x0303: return 0xDFDFDFu;
        default: {
            /* Stable fallback for unknown map words. */
            static const uint32_t fallback[8] = {
                0xFFFFFFu, 0xD0D0D0u, 0xC0A080u, 0xA0C0E0u,
                0xE0A0A0u, 0xA0D0A0u, 0xE0C080u, 0xB0B0B0u
            };
            unsigned idx = (unsigned)(((tex_map_word >> 8) ^ tex_map_word) & 7u);
            return fallback[idx];
        }
    }
}

static int obj_bright_to_level(int raw_brightness)
{
    if (raw_brightness <= 1) return 0;
    if (raw_brightness < 0) return 0;

    /* ObjDraw3 objscalecols shape: 0..1 -> level0, then 4 indices per level. */
    int level = 1 + (raw_brightness - 2) / 4;
    if (level > 14) level = 14;
    return level;
}

/* -----------------------------------------------------------------------
 * Per-vertex screen coordinate cache (shared across all polygons in one
 * draw call, reset for each object).
 * ----------------------------------------------------------------------- */
typedef struct { int32_t x, y; int16_t z; } BoxRot;
typedef struct { int32_t x, y, z; int16_t vb; } ObjVertex;
typedef struct { int32_t x, y, z; int32_t u, v; int16_t vb; } PolyVertex;

static BoxRot    s_boxrot[MAX_POLY_POINTS];
static int16_t   s_boxbrights[MAX_POLY_POINTS];
static ObjVertex s_world[MAX_POLY_POINTS];

/* -----------------------------------------------------------------------
 * Polygon scanline rasteriser
 *
 * Draws a convex (or near-convex) polygon into
 * g_renderer.rgb_buffer.  Uses a simple span-table (min_x/max_x per row).
 * ----------------------------------------------------------------------- */
#define POLY_MAX_HEIGHT   (RENDER_HEIGHT)
#define MAX_POLY_VERTS    16
#define MAX_CLIP_VERTS    (MAX_POLY_VERTS + 2)
#define OBJ_NEAR_Z        4

static PolyVertex intersect_near_plane(const PolyVertex *a, const PolyVertex *b,
                                       int32_t near_z);
static int clip_polygon_to_near(const PolyVertex *in, int in_count,
                                PolyVertex *out, int max_out, int32_t near_z);
static int poly_textures_ready(void);
static uint8_t sample_poly_texel_index(uint16_t tex_map_word,
                                       int32_t u_fixed, int32_t v_fixed);
static uint32_t sample_poly_palette(uint8_t pal_idx, int shade_level);
static void draw_textured_polygon(const int *sx, const int *sy,
                                  const int32_t *sz,
                                  const int32_t *u, const int32_t *v,
                                  int n, uint16_t tex_map_word, int shade_level,
                                  const int16_t *shade_values,
                                  int use_gouraud, int use_holes,
                                  int clip_left, int clip_right,
                                  int clip_top, int clip_bot);
static int ensure_poly_depth_buffer(size_t pixels);

static int s_span_min[POLY_MAX_HEIGHT];
static int s_span_max[POLY_MAX_HEIGHT];

static void trace_edge(int x0, int y0, int x1, int y1,
                       int y_base, int y_limit)
{
    if (y0 == y1) {
        /* Horizontal edge – update span at that row */
        if (y0 < y_base || y0 > y_limit) return;
        int idx = y0 - y_base;
        if (x0 < s_span_min[idx]) s_span_min[idx] = x0;
        if (x0 > s_span_max[idx]) s_span_max[idx] = x0;
        if (x1 < s_span_min[idx]) s_span_min[idx] = x1;
        if (x1 > s_span_max[idx]) s_span_max[idx] = x1;
        return;
    }
    if (y0 > y1) {
        int t = x0; x0 = x1; x1 = t;
        t = y0; y0 = y1; y1 = t;
    }
    int dy = y1 - y0;
    int dx = x1 - x0;
    int sy = y0 > y_base ? y0 : y_base;
    int ey = y1 < y_limit ? y1 : y_limit;
    for (int y = sy; y <= ey; y++) {
        int x = x0 + (dy > 0 ? ((y - y0) * dx) / dy : 0);
        int idx = y - y_base;
        if (x < s_span_min[idx]) s_span_min[idx] = x;
        if (x > s_span_max[idx]) s_span_max[idx] = x;
    }
}

static void draw_filled_polygon(const int *sx, const int *sy, int n,
                                uint32_t color,
                                int clip_left, int clip_right,
                                int clip_top, int clip_bot)
{
    if (n < 3 || !g_renderer.rgb_buffer) return;

    int min_y = sy[0], max_y = sy[0];
    for (int i = 1; i < n; i++) {
        if (sy[i] < min_y) min_y = sy[i];
        if (sy[i] > max_y) max_y = sy[i];
    }
    if (min_y < clip_top)  min_y = clip_top;
    if (max_y > clip_bot)  max_y = clip_bot;
    if (min_y > max_y) return;

    int range = max_y - min_y + 1;
    if (range > POLY_MAX_HEIGHT) range = POLY_MAX_HEIGHT;
    max_y = min_y + range - 1;

    for (int i = 0; i < range; i++) {
        s_span_min[i] = clip_right + 1;
        s_span_max[i] = clip_left  - 1;
    }
    for (int i = 0; i < n; i++) {
        trace_edge(sx[i], sy[i], sx[(i + 1) % n], sy[(i + 1) % n],
                   min_y, max_y);
    }

    int W = g_renderer.width;
    uint32_t *rgb = g_renderer.rgb_buffer;
    for (int y = min_y; y <= max_y; y++) {
        int x0 = s_span_min[y - min_y];
        int x1 = s_span_max[y - min_y];
        if (x0 < clip_left)  x0 = clip_left;
        if (x1 > clip_right) x1 = clip_right;
        if (x0 > x1) continue;
        uint32_t *row = rgb + (size_t)y * W;
        for (int x = x0; x <= x1; x++)
            row[x] = color;
    }
}

/* -----------------------------------------------------------------------
 * Blend a base colour with a tint colour and apply brightness shading.
 * brightness : 0 = closest/brightest, 14 = farthest/darkest.
 * ----------------------------------------------------------------------- */
static uint32_t make_poly_color(int slot, uint16_t tex_map_word, int shade_level)
{
    if (slot < 0 || slot >= POLY_OBJECTS_COUNT) slot = 0;
    uint32_t base = obj_base_rgb[slot];
    uint32_t tint = tex_tint_from_word(tex_map_word);

    /* Blend base × tint / 255 (component-wise) */
    unsigned br = ((base >> 16) & 0xFF) * ((tint >> 16) & 0xFF) / 255;
    unsigned bg = ((base >>  8) & 0xFF) * ((tint >>  8) & 0xFF) / 255;
    unsigned bb = ( base        & 0xFF) * ( tint        & 0xFF) / 255;

    /* Shade: 0 = full bright, 14 = darkest. */
    if (shade_level < 0)  shade_level = 0;
    if (shade_level > 14) shade_level = 14;
    unsigned shade = (unsigned)(255 - shade_level * 17);

    br = br * shade / 255;
    bg = bg * shade / 255;
    bb = bb * shade / 255;

    return RENDER_RGB_RASTER_PIXEL((br << 16) | (bg << 8) | bb);
}

/* -----------------------------------------------------------------------
 * draw_3d_vector_object
 *
 * Main entry point – implements the PolygonObj pipeline from
 * ObjDraw3.ChipRam.s.
 * ----------------------------------------------------------------------- */
void draw_3d_vector_object(const uint8_t *obj, const ObjRotatedPoint *orp, GameState *state)
{
    RendererState *r = &g_renderer;

    /* ---- 0. Object identification ------------------------------------ */
    /* After the Amiga's `move.w (a0)+,d0` at the start of PolygonObj,
     * a0 is advanced by 2.  Subsequent `6(a0)` accesses byte 8 of the
     * original object → that is objVectNumber (the POLYOBJECTS index). */
    int16_t vect_num = vec_rd16(obj + 8);
    if (vect_num < 0 || vect_num >= POLY_OBJECTS_COUNT) return;

    VecObject *vo = &g_poly_objects[vect_num];
    if (!vo->data || vo->num_points == 0 || vo->num_parts == 0) return;

    /* ---- 1. Brightness (objBright) ----------------------------------- */
    /* Amiga: d3 = zpos>>7; d2 = objVectBright + d3. */
    int16_t obj_bright = vec_rd16(obj + 2);    /* objVectBright */
    int32_t z_mid      = orp->z;
    int obj_bright_base = (int)(z_mid >> 7) + (int)obj_bright;
    if (obj_bright_base < 0)  obj_bright_base = 0;
    if (obj_bright_base > 14) obj_bright_base = 14;

    /* ---- 2. Relative rotation angle ---------------------------------- */
    /* Amiga: angpos = (objAng - 2048 - viewer_angpos) & 8191
     * where angpos uses byte-indexed sine table (even entries only).
     * We read objVectFacing from offset 30 (a2-field before a0 advanced). */
    int16_t facing  = vec_rd16(obj + 30);  /* objVectFacing */
    PlayerState *plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
    int viewer_ang  = (int)(plr->angpos & ANGLE_MASK); /* 0..8191 */
    /* Amiga subtracts 2048 (= ANGLE_90 bytes = 90 degrees) before viewer angle */
    int rel_ang = (int)facing - ANGLE_90 - viewer_ang;
    rel_ang &= ANGLE_MASK;   /* wrap to 0..8191 */

    int16_t sin_v = sin_lookup(rel_ang);
    int16_t cos_v = cos_lookup(rel_ang);

    /* ---- 3. Y-axis offset for object --------------------------------- */
    /* ObjDraw3 PolygonObj @ convtoscr: move.w 2(a0),d2 / ext.l d2 / asl.l #7,d2 / sub.l yoff,d2
     * (a0 points at object+2 after move.w (a0)+,d0 at PolygonObj entry). */
    int16_t obj_y4 = vec_rd16(obj + 4);
    int32_t y_adjust = ((int32_t)obj_y4 << 7) - r->yoff;

    /* ---- 4. Select animation frame ----------------------------------- */
    /* ObjDraw3 PolygonObj currently forces frame 0:
     *   moveq #0,d5
     *   move.w (a4,d5.w*2),d5
     * The objVectFrameNumber path exists only in commented-out code. */
    int frame_num = 0;
    if (g_poly_use_object_frame) {
        frame_num = (int)vec_rd16(obj + 10);
        if (frame_num < 0 || frame_num >= vo->num_frames) frame_num = 0;
    }
    int frame_byte_off = (int)vo->frame_off[frame_num];
    if (frame_byte_off + vo->num_points * 6 > (int)vo->size) return;
    const uint8_t *pts = vo->data + frame_byte_off;

    /* ---- 5. rotobj: rotate all vertices ------------------------------ */
    /* Amiga doubles each raw coordinate before muls (add.w d,d), then:
     *   boxrot[i].x = (x*2*sin - z*2*cos) >> 8 = (x*sin - z*cos) >> 7  (int32)
     *   boxrot[i].y = y*2 << 7                  = y << 8               (int32)
     *   boxrot[i].z = ((x*cos + z*sin)*4) >> 16 = (x*cos+z*sin) >> 14  (int16)
     */
    int np = vo->num_points;
    for (int i = 0; i < np; i++) {
        int16_t lx = vec_rd16(pts + i * 6 + 0);
        int16_t ly = vec_rd16(pts + i * 6 + 2);
        int16_t lz = vec_rd16(pts + i * 6 + 4);

        int32_t rx = ((int32_t)lx * sin_v - (int32_t)lz * cos_v) >> 7;
        int32_t ry = (int32_t)ly << 8;
        int16_t rz = (int16_t)(((int32_t)lx * cos_v + (int32_t)lz * sin_v) >> 14);

        s_boxrot[i].x = rx;
        s_boxrot[i].y = ry;
        s_boxrot[i].z = rz;

        /* Per-vertex brightness: (rz + 20) / 4, clamped 0..13 (Amiga add.w #20 / asr.w #2) */
        int vb = ((int)rz + 20) >> 2;
        if (vb < 0)  vb = 0;
        if (vb > 13) vb = 13;
        s_boxbrights[i] = (int16_t)vb;
    }

    /* ---- 6. convtoscr prep: translate vertices to world/view --------- */
    /* Keep world-space X/Y/Z per vertex, then clip each polygon against
     * a near plane before projection. This avoids dropping whole faces when
     * only part of a polygon crosses behind the camera. */
    int32_t xpos_mid = orp->x_fine;     /* 32-bit view X of object centre */
    int32_t zpos_mid = orp->z;          /* view Z of object centre  */
    int W = r->width, H = r->height;
    int half_w = W / 2, half_h = H / 2;
    int32_t proj_ys = r->proj_y_scale;
    size_t pix_count = (size_t)W * (size_t)H;
    if (!ensure_poly_depth_buffer(pix_count)) return;
    /* Advance generation stamp instead of clearing all W*H entries.
     * On overflow back to zero, reset the stamp array so no stale entries match. */
    g_depth_gen++;
    if (g_depth_gen == 0) {
        g_depth_gen = 1;
        if (g_poly_obj_depth_gen)
            memset(g_poly_obj_depth_gen, 0, pix_count * sizeof(uint32_t));
    }

    for (int i = 0; i < np; i++) {
        int32_t worldX = s_boxrot[i].x + xpos_mid;
        int32_t worldY = s_boxrot[i].y + y_adjust;
        int32_t worldZ = (int32_t)s_boxrot[i].z + zpos_mid;
        s_world[i].x = worldX;
        s_world[i].y = worldY;
        s_world[i].z = worldZ;
        s_world[i].vb = s_boxbrights[i];
    }

    /* ---- 7. PutinParts: depth-sort polygon parts --------------------- */
    /* Amiga uses a 32-slot insertion-sort depth buffer keyed by squared
     * distance of each part's sort-point from the object origin.
     * For simplicity we use a direct array + insertion sort.
     * sort key = z of the sort point's world-space Z (before projection).
     */
    int nparts = vo->num_parts;
    if (nparts > MAX_POLY_PARTS) nparts = MAX_POLY_PARTS;

    typedef struct { int32_t sort_key; int part_idx; } PartEntry;
    PartEntry sorted[MAX_POLY_PARTS];
    int sorted_count = 0;

    for (int p = 0; p < nparts; p++) {
        /* sort_off is a byte offset into boxrot[] where each entry is 10 bytes
         * (4 + 4 + 2).  Recover the point index as sort_off / 10. */
        int sort_pt = (int)vo->part_sort_off[p] / 10;
        if (sort_pt < 0 || sort_pt >= np) sort_pt = 0;

        /* Amiga PutinParts key:
         *   key = (x>>7)^2 + (y>>7)^2 + z^2
         * where x/y/z are from boxrot[] (object space after facing/view rotation,
         * before object translation). */
        int16_t sx = (int16_t)(s_boxrot[sort_pt].x >> 7);
        int16_t sy = (int16_t)(s_boxrot[sort_pt].y >> 7);
        int16_t sz = (int16_t)s_boxrot[sort_pt].z;
        int32_t key = (int32_t)sx * (int32_t)sx +
                      (int32_t)sy * (int32_t)sy +
                      (int32_t)sz * (int32_t)sz;

        /* Insertion sort: farthest first (painter's) */
        int ins = sorted_count;
        while (ins > 0 && sorted[ins - 1].sort_key < key) ins--;
        for (int k = sorted_count; k > ins; k--) sorted[k] = sorted[k - 1];
        sorted[ins].sort_key = key;
        sorted[ins].part_idx = p;
        if (sorted_count < MAX_POLY_PARTS) sorted_count++;
    }

    /* ---- 8. Partloop / polyloo: draw each polygon -------------------- */
    int clip_l = (int)r->left_clip;
    int clip_r = (int)r->right_clip - 1;
    int clip_t = (int)r->top_clip;
    int clip_b = (int)r->bot_clip;
    if (clip_l < 0) clip_l = 0;
    if (clip_r >= W) clip_r = W - 1;
    if (clip_t < 0) clip_t = 0;
    if (clip_b >= H) clip_b = H - 1;

    for (int si = 0; si < sorted_count; si++) {   /* farthest first (painter's) */
        int pi = sorted[si].part_idx;
        int poly_off = (int)vo->part_poly_off[pi];
        if (poly_off + 2 > (int)vo->size) continue;
        const uint8_t *poly_ptr = vo->data + poly_off;

        /* Iterate polygon list for this part */
        int bytes_left = (int)vo->size - poly_off;
        while (bytes_left >= 2) {
            const uint8_t *poly_start = poly_ptr;
            int16_t lines_to_draw = vec_rd16(poly_start);
            if (lines_to_draw < 0) break;  /* end-of-list marker */

            int poly_size = 18 + lines_to_draw * 4;
            if (poly_size <= 0 || poly_size > bytes_left) break;

            poly_ptr   += poly_size;
            bytes_left -= poly_size;

            /* ObjDraw3 encodes polygons as a sliding edge stream.
             * Unique perimeter vertices = lines_to_draw + 1.
             * The next record repeats the first point for closure. */
            int num_verts = lines_to_draw + 1;
            if (num_verts < 3 || num_verts > MAX_POLY_VERTS) continue;

            int vi[MAX_POLY_VERTS];
            int bad_index = 0;
            for (int v = 0; v < num_verts; v++) {
                const uint8_t *vrec = poly_start + 4 + v * 4;
                vi[v] = (int)(uint16_t)vec_rd16(vrec);
                if (vi[v] < 0 || vi[v] >= np) bad_index = 1;
            }
            if (bad_index) continue;

            /* texture_map_index, brightness divisor, pregour live at:
             *   poly + 12 + d0*4, +14 + d0*4, +16 + d0*4 */
            uint16_t preholes_word = (uint16_t)vec_rd16(poly_start + 2);
            const uint8_t *extra = poly_start + 12 + lines_to_draw * 4;
            uint16_t tex_map = 0;
            int shade_div = 1;
            uint16_t pregour_word = 0;
            if (extra + 6 <= vo->data + vo->size) {
                tex_map = (uint16_t)vec_rd16(extra);
                shade_div = (int)vec_rd16(extra + 2);
                pregour_word = (uint16_t)vec_rd16(extra + 4);
            }
            if (shade_div < 0) shade_div = -shade_div;
            if (shade_div == 0) shade_div = 1;
            /* On Amiga the word write lands on adjacent bytes:
             * low byte of preholes -> Holes, low byte of pregour -> Gouraud. */
            int use_holes = ((int)preholes_word & 0x00FF) != 0;
            int use_gouraud = ((int)pregour_word & 0x00FF) != 0;

            PolyVertex in_poly[MAX_CLIP_VERTS];
            for (int v = 0; v < num_verts; v++) {
                const uint8_t *vrec = poly_start + 4 + v * 4;
                const ObjVertex *sv = &s_world[vi[v]];
                in_poly[v].x = sv->x;
                in_poly[v].y = sv->y;
                in_poly[v].z = sv->z;
                in_poly[v].u = (int32_t)vrec[2] << 16;
                in_poly[v].v = (int32_t)vrec[3] << 16;
                in_poly[v].vb = sv->vb;
            }

            PolyVertex clipped_poly[MAX_CLIP_VERTS];
            int clipped_n = clip_polygon_to_near(in_poly, num_verts, clipped_poly,
                                                 MAX_CLIP_VERTS, OBJ_NEAR_Z);
            if (clipped_n < 3) continue;

            int sx[MAX_CLIP_VERTS], sy[MAX_CLIP_VERTS];
            int32_t sz[MAX_CLIP_VERTS];
            int32_t su[MAX_CLIP_VERTS], svt[MAX_CLIP_VERTS];
            int16_t shade_values[MAX_CLIP_VERTS];
            for (int v = 0; v < clipped_n; v++) {
                int32_t wz = clipped_poly[v].z;
                if (wz <= 0) wz = 1;
                sx[v] = (int)((clipped_poly[v].x * RENDER_SCALE) / wz) + half_w;
                sy[v] = (int)((int64_t)(clipped_poly[v].y >> WORLD_Y_FRAC_BITS) * proj_ys * RENDER_SCALE
                              / wz) + half_h;
                sz[v] = wz;
                su[v] = clipped_poly[v].u;
                svt[v] = clipped_poly[v].v;
                int vb = (int)clipped_poly[v].vb;
                if (vb < 0) vb = 0;
                if (vb > 14) vb = 14;
                shade_values[v] = (int16_t)vb;
            }

            /* Back-face culling – matches Amiga doapoly cross product:
             *   muls (x2-x1),(y0-y1)  →  sub  muls (x0-x1),(y2-y1)  → ble skip
             * i.e. (x2-x1)*(y0-y1) - (x0-x1)*(y2-y1) > 0 is front-facing. */
            int cross = (sx[2] - sx[1]) * (sy[0] - sy[1]) - (sx[0] - sx[1]) * (sy[2] - sy[1]);
            if (cross <= 0) continue;  /* back-facing or degenerate */

            /* ObjDraw3 face brightness:
             *   d1 = (polybright * 8) / shade_div
             *   shade_raw = objBright + 14 - d1
             * polybright is measured in Amiga screen pixels, so normalize for
             * higher render scales to keep similar contrast. */
            int cross_norm = cross / (RENDER_SCALE * RENDER_SCALE);
            int face_term = (cross_norm * 8) / shade_div;
            if (face_term < 0) face_term = 0;
            if (face_term > 64) face_term = 64;
            int shade_raw = obj_bright_base + 14 - face_term;

            int shade_level = obj_bright_to_level(shade_raw);
            if (poly_textures_ready()) {
                draw_textured_polygon(sx, sy, sz, su, svt, clipped_n,
                                      tex_map, shade_level,
                                      shade_values, use_gouraud, use_holes,
                                      clip_l, clip_r, clip_t, clip_b);
            } else {
                uint32_t color = make_poly_color(vect_num, tex_map, shade_level);
                draw_filled_polygon(sx, sy, clipped_n, color,
                                    clip_l, clip_r, clip_t, clip_b);
            }
        }
    }
}

static inline uint32_t amiga12_to_argb_local(uint16_t w)
{
    uint32_t r4 = (uint32_t)((w >> 8) & 0xF);
    uint32_t g4 = (uint32_t)((w >> 4) & 0xF);
    uint32_t b4 = (uint32_t)(w & 0xF);
    return RENDER_RGB_RASTER_PIXEL((r4 * 0x11u << 16) | (g4 * 0x11u << 8) | (b4 * 0x11u));
}

static int poly_textures_ready(void)
{
    return g_poly_tex_maps && g_poly_tex_pal &&
           g_poly_tex_maps_size >= 65536 &&
           g_poly_tex_pal_size >= (15u * 512u);
}

static uint8_t sample_poly_texel_index(uint16_t tex_map_word,
                                       int32_t u_fixed, int32_t v_fixed)
{
    if (!poly_textures_ready()) return 0;

    int u = (int)(u_fixed >> 16) & 63;
    int v = (int)(v_fixed >> 16) & 63;
    size_t uv = ((size_t)v << 8) | (size_t)u;  /* v:high byte, u:low byte */
    size_t tex_off = (uv << 2) + (size_t)tex_map_word;
    if (tex_off >= g_poly_tex_maps_size) tex_off %= g_poly_tex_maps_size;

    return g_poly_tex_maps[tex_off];
}

static uint32_t sample_poly_palette(uint8_t pal_idx, int shade_level)
{
    if (!poly_textures_ready()) return 0;
    if (shade_level < 0) shade_level = 0;
    if (shade_level > 14) shade_level = 14;

    size_t pal_off = (size_t)shade_level * 512u + (size_t)pal_idx * 2u;
    if (pal_off + 1u >= g_poly_tex_pal_size) return 0;
    uint16_t cw = (uint16_t)(((uint16_t)g_poly_tex_pal[pal_off] << 8) |
                              (uint16_t)g_poly_tex_pal[pal_off + 1u]);
    return amiga12_to_argb_local(cw);
}

static void draw_textured_triangle(const int *sx, const int *sy,
                                   const int32_t *sz, const int32_t *u, const int32_t *v,
                                   const int16_t *shade_values,
                                   int use_gouraud, int use_holes,
                                   uint16_t tex_map_word, int shade_level,
                                   int clip_left, int clip_right,
                                   int clip_top, int clip_bot)
{
    if (!g_renderer.rgb_buffer) return;

    int min_x = sx[0], max_x = sx[0];
    int min_y = sy[0], max_y = sy[0];
    for (int i = 1; i < 3; i++) {
        if (sx[i] < min_x) min_x = sx[i];
        if (sx[i] > max_x) max_x = sx[i];
        if (sy[i] < min_y) min_y = sy[i];
        if (sy[i] > max_y) max_y = sy[i];
    }
    if (min_x < clip_left) min_x = clip_left;
    if (max_x > clip_right) max_x = clip_right;
    if (min_y < clip_top) min_y = clip_top;
    if (max_y > clip_bot) max_y = clip_bot;
    if (min_x > max_x || min_y > max_y) return;

    double x0 = (double)sx[0], y0 = (double)sy[0];
    double x1 = (double)sx[1], y1 = (double)sy[1];
    double x2 = (double)sx[2], y2 = (double)sy[2];
    double area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (area == 0.0) return;

    double inv_area = 1.0 / area;
    int W = g_renderer.width;
    uint32_t *rgb = g_renderer.rgb_buffer;

    for (int y = min_y; y <= max_y; y++) {
        uint32_t *row = rgb + (size_t)y * W;
        double py = (double)y + 0.5;
        for (int x = min_x; x <= max_x; x++) {
            double px = (double)x + 0.5;
            double w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * inv_area;
            double w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * inv_area;
            double w2 = 1.0 - w0 - w1;
            if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) continue;

            int32_t zf = (int32_t)(w0 * (double)sz[0] + w1 * (double)sz[1] + w2 * (double)sz[2]);
            size_t didx = (size_t)y * (size_t)W + (size_t)x;
            if (g_poly_obj_depth && g_poly_obj_depth_gen &&
                g_poly_obj_depth_gen[didx] == g_depth_gen &&
                zf >= g_poly_obj_depth[didx]) continue;
            int32_t uf = (int32_t)(w0 * (double)u[0] + w1 * (double)u[1] + w2 * (double)u[2]);
            int32_t vf = (int32_t)(w0 * (double)v[0] + w1 * (double)v[1] + w2 * (double)v[2]);
            uint8_t pal_idx = sample_poly_texel_index(tex_map_word, uf, vf);
            if (use_holes && pal_idx == 0) continue;
            int pixel_shade = shade_level;
            if (use_gouraud && shade_values) {
                double shade_f = w0 * (double)shade_values[0] +
                                 w1 * (double)shade_values[1] +
                                 w2 * (double)shade_values[2];
                pixel_shade = (int)(shade_f + 0.5);
                if (pixel_shade < 0) pixel_shade = 0;
                if (pixel_shade > 14) pixel_shade = 14;
            }
            if (g_poly_obj_depth) {
                g_poly_obj_depth[didx] = zf;
                if (g_poly_obj_depth_gen) g_poly_obj_depth_gen[didx] = g_depth_gen;
            }
            row[x] = sample_poly_palette(pal_idx, pixel_shade);
        }
    }
}

static void draw_textured_polygon(const int *sx, const int *sy,
                                  const int32_t *sz,
                                  const int32_t *u, const int32_t *v,
                                  int n, uint16_t tex_map_word, int shade_level,
                                  const int16_t *shade_values,
                                  int use_gouraud, int use_holes,
                                  int clip_left, int clip_right,
                                  int clip_top, int clip_bot)
{
    if (n < 3) return;

    int tsx[3], tsy[3];
    int32_t tz[3];
    int32_t tu[3], tv[3];
    int16_t tshade[3];
    for (int i = 1; i < n - 1; i++) {
        tsx[0] = sx[0];  tsy[0] = sy[0];
        tsx[1] = sx[i];  tsy[1] = sy[i];
        tsx[2] = sx[i + 1]; tsy[2] = sy[i + 1];

        tz[0] = sz[0];
        tz[1] = sz[i];
        tz[2] = sz[i + 1];
        tu[0] = u[0];  tv[0] = v[0];
        tu[1] = u[i];  tv[1] = v[i];
        tu[2] = u[i + 1]; tv[2] = v[i + 1];
        if (shade_values) {
            tshade[0] = shade_values[0];
            tshade[1] = shade_values[i];
            tshade[2] = shade_values[i + 1];
        } else {
            tshade[0] = tshade[1] = tshade[2] = (int16_t)shade_level;
        }

        draw_textured_triangle(tsx, tsy, tz, tu, tv,
                               tshade, use_gouraud, use_holes,
                               tex_map_word, shade_level,
                               clip_left, clip_right, clip_top, clip_bot);
    }
}

static int ensure_poly_depth_buffer(size_t pixels)
{
    if (pixels == 0) return 0;
    if (pixels > g_poly_obj_depth_cap) {
        int32_t *new_buf = (int32_t *)realloc(g_poly_obj_depth, pixels * sizeof(int32_t));
        if (!new_buf) return 0;
        uint32_t *new_gen = (uint32_t *)realloc(g_poly_obj_depth_gen, pixels * sizeof(uint32_t));
        if (!new_gen) { free(new_buf); return 0; }
        /* Zero new stamp entries so they never accidentally match g_depth_gen. */
        memset(new_gen + g_poly_obj_depth_cap, 0,
               (pixels - g_poly_obj_depth_cap) * sizeof(uint32_t));
        g_poly_obj_depth = new_buf;
        g_poly_obj_depth_gen = new_gen;
        g_poly_obj_depth_cap = pixels;
    }
    return 1;
}

static PolyVertex intersect_near_plane(const PolyVertex *a, const PolyVertex *b,
                                       int32_t near_z)
{
    PolyVertex out = *a;
    int64_t dz = (int64_t)b->z - (int64_t)a->z;
    if (dz == 0) {
        out.z = near_z;
        return out;
    }

    int64_t num = (int64_t)near_z - (int64_t)a->z;
    out.x = a->x + (int32_t)(((int64_t)(b->x - a->x) * num) / dz);
    out.y = a->y + (int32_t)(((int64_t)(b->y - a->y) * num) / dz);
    out.z = near_z;
    out.u = a->u + (int32_t)(((int64_t)(b->u - a->u) * num) / dz);
    out.v = a->v + (int32_t)(((int64_t)(b->v - a->v) * num) / dz);
    out.vb = (int16_t)(a->vb + (int32_t)(((int64_t)(b->vb - a->vb) * num) / dz));
    return out;
}

static int clip_polygon_to_near(const PolyVertex *in, int in_count,
                                PolyVertex *out, int max_out, int32_t near_z)
{
    if (!in || !out || in_count < 3 || max_out < 3) return 0;

    int out_count = 0;
    for (int i = 0; i < in_count; i++) {
        const PolyVertex *s = &in[i];
        const PolyVertex *e = &in[(i + 1) % in_count];
        int s_in = (s->z >= near_z);
        int e_in = (e->z >= near_z);

        if (s_in && e_in) {
            if (out_count >= max_out) return 0;
            out[out_count++] = *e;
        } else if (s_in && !e_in) {
            if (out_count >= max_out) return 0;
            out[out_count++] = intersect_near_plane(s, e, near_z);
        } else if (!s_in && e_in) {
            if (out_count + 1 >= max_out) return 0;
            out[out_count++] = intersect_near_plane(s, e, near_z);
            out[out_count++] = *e;
        }
    }

    return out_count;
}
