/*
 * Alien Breed 3D I - PC Port
 * movement.c - Object movement, collision, and physics (full implementation)
 *
 * Translated from: ObjectMove.s, Fall.s
 */

#include "movement.h"
#include "game_data.h"
#include "math_tables.h"
#include "level.h"
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

 /* -----------------------------------------------------------------------
  * Constants from the original code
  * ----------------------------------------------------------------------- */
#define GRAVITY_ACCEL       256     /* gravity per frame (Fall.s: add.l #256,d2) */
#define GRAVITY_DECEL       512     /* ground decel (Fall.s: sub.l #512,d2) */
#define WATER_MAX_VELOCITY  512     /* max fall speed in water */

  /* Floor line structure offsets (from Defs.i / raw data analysis):
   * 0:  int16  x1         - start X
   * 2:  int16  z1         - start Z
   * 4:  int16  xlen       - X length (direction * scale)
   * 6:  int16  zlen       - Z length (direction * scale)
   * 8:  int16  connected_zone  - zone on other side (-1 = solid wall)
   * 10: int16  line_length     - magnitude of direction vector
   * 12: int16  normal_or_angle - precomputed value (unused here)
   * 14: int8   awayfromwall_shift (unused here)
   * 15: int8   reserved
   * Total: 16 bytes per floor line
   */
#define FLINE_SIZE      16
#define FLINE_X         0
#define FLINE_Z         2
#define FLINE_XLEN      4
#define FLINE_ZLEN      6
#define FLINE_CONNECT   8
#define FLINE_LENGTH    10
#define FLINE_NORMAL    12
#define FLINE_AWAY      14

   /* Zone data offsets */
#define ZONE_FLOOR_HEIGHT       2
#define ZONE_ROOF_HEIGHT        6
#define ZONE_UPPER_FLOOR        10
#define ZONE_UPPER_ROOF         14
#define ZONE_EXIT_LIST          32
#define ZONE_WALL_LIST          28

enum {
    MOVE_PASS_WALLS = 1,  /* Amiga checkwalls */
    MOVE_PASS_OTHER = 2,  /* Amiga checkotherwalls */
    MOVE_PASS_BRUTE = 3
};

/* -----------------------------------------------------------------------
 * Helper: read big-endian int16/int32 from level data
 * ----------------------------------------------------------------------- */
static int16_t read_be16(const uint8_t* p)
{
    return (int16_t)((p[0] << 8) | p[1]);
}

static void write_be16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v >> 8);
    p[1] = (uint8_t)(v & 0xFFu);
}

static int32_t read_be32(const uint8_t* p)
{
    return (int32_t)((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
}

static int move_debug_enabled(void)
{
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        const char *v = getenv("AB3D_DEBUG_MOVE");
        enabled = (v && v[0] != '\0' && v[0] != '0') ? 1 : 0;
        init = 1;
    }
    return enabled;
}

/* Optional filter to one collision id. -1 means all. */
static int move_debug_cid_filter(void)
{
    static int init = 0;
    static int cid = -1;
    if (!init) {
        const char *v = getenv("AB3D_DEBUG_MOVE_CID");
        if (v && v[0] != '\0')
            cid = atoi(v);
        init = 1;
    }
    return cid;
}

static int move_debug_obj_type_by_cid(const LevelState *level, int16_t coll_id)
{
    if (!level || !level->object_data || coll_id < 0) return -1;
    for (int i = 0; i < 256; i++) {
        const GameObject *obj = (const GameObject *)(level->object_data + (size_t)i * OBJECT_SIZE);
        int16_t cid = OBJ_CID(obj);
        if (cid < 0) break;
        if (cid == coll_id)
            return (int)(int8_t)obj->raw[16];
    }
    return -1;
}

static bool move_debug_is_enemy_type(int obj_type)
{
    switch (obj_type) {
    case OBJ_NBR_ALIEN:
    case OBJ_NBR_ROBOT:
    case OBJ_NBR_BIG_NASTY:
    case OBJ_NBR_FLYING_NASTY:
    case OBJ_NBR_MARINE:
    case OBJ_NBR_WORM:
    case OBJ_NBR_HUGE_RED_THING:
    case OBJ_NBR_SMALL_RED_THING:
    case OBJ_NBR_TREE:
    case OBJ_NBR_EYEBALL:
    case OBJ_NBR_TOUGH_MARINE:
    case OBJ_NBR_FLAME_MARINE:
        return true;
    default:
        return false;
    }
}

static bool move_ctx_is_enemy(const MoveContext *ctx, const LevelState *level, int *out_obj_type)
{
    if (out_obj_type) *out_obj_type = -1;
    if (!ctx || !level) return false;
    {
        int obj_type = move_debug_obj_type_by_cid(level, ctx->coll_id);
        if (out_obj_type) *out_obj_type = obj_type;
        return move_debug_is_enemy_type(obj_type);
    }
}

static bool move_debug_should_log(const MoveContext *ctx, const LevelState *level, int *out_obj_type)
{
    if (out_obj_type) *out_obj_type = -1;
    if (!move_debug_enabled() || !ctx || !level) return false;

    {
        int filter = move_debug_cid_filter();
        if (filter >= 0 && (int)ctx->coll_id != filter) return false;
    }

    return move_ctx_is_enemy(ctx, level, out_obj_type);
}

static int move_debug_line_index(const LevelState *level, const uint8_t *fline)
{
    if (!level || !level->floor_lines || !fline) return -1;
    if (fline < level->floor_lines) return -1;
    return (int)((fline - level->floor_lines) / FLINE_SIZE);
}

/* Amiga MoveObject exit passability test (ObjectMove.s chkstepup/botinsidebot):
 *  1) target room clearance must be strictly greater than thingheight
 *  2) d1 = mover_y + thingheight - target_floor
 *     if d1 > 0  -> d1 < StepUpVal
 *     else       -> -d1 < StepDownVal
 *  3) mover_y - target_roof must be >= 0
 */
static bool transition_height_ok_single(const MoveContext *ctx, int32_t mover_y,
    int32_t target_floor, int32_t target_roof)
{
    int64_t clearance = (int64_t)target_floor - (int64_t)target_roof;
    if (clearance <= (int64_t)ctx->thing_height) return false;

    {
        int64_t d1 = (int64_t)mover_y + (int64_t)ctx->thing_height - (int64_t)target_floor;
        if (d1 > 0) {
            if (d1 >= (int64_t)ctx->step_up_val) return false;
        } else {
            if ((-d1) >= (int64_t)ctx->step_down_val) return false;
        }
    }

    if ((int64_t)mover_y - (int64_t)target_roof < 0) return false;
    return true;
}

/* Amiga checkotherwalls first tests lower floor/roof, then upper floor/roof. */
static bool transition_height_ok_zone(const MoveContext *ctx, int32_t mover_y,
    const uint8_t *target_zone, int8_t *out_in_top)
{
    if (!ctx || !target_zone) return false;

    {
        int32_t floor_h = read_be32(target_zone + ZONE_FLOOR_HEIGHT);
        int32_t roof_h = read_be32(target_zone + ZONE_ROOF_HEIGHT);
        if (transition_height_ok_single(ctx, mover_y, floor_h, roof_h)) {
            if (out_in_top) *out_in_top = 0;
            return true;
        }
    }

    {
        int32_t floor_h = read_be32(target_zone + ZONE_UPPER_FLOOR);
        int32_t roof_h = read_be32(target_zone + ZONE_UPPER_ROOF);
        if (transition_height_ok_single(ctx, mover_y, floor_h, roof_h)) {
            if (out_in_top) *out_in_top = 1;
            return true;
        }
    }

    return false;
}

static void mark_floorline_touch_flag(const MoveContext *ctx, const uint8_t *fline);

static void get_floorline_shift_offsets(const MoveContext *ctx, const uint8_t *fline,
    int32_t *out_a4, int32_t *out_a6)
{
    int32_t a4 = 0;
    int32_t a6 = 0;

    if (ctx && fline) {
        int8_t shift = (int8_t)ctx->awayfromwall;
        if (shift >= 0) {
            a4 = (int32_t)(int8_t)fline[FLINE_NORMAL + 0];
            a6 = (int32_t)(int8_t)fline[FLINE_NORMAL + 1];
            if (shift > 0) {
                if (shift > 15) shift = 15;
                a4 <<= shift;
                a6 <<= shift;
            }
        }
    }

    if (out_a4) *out_a4 = a4;
    if (out_a6) *out_a6 = a6;
}

static int32_t compute_crossing_height_asm(const MoveContext *ctx,
    int64_t cross_new, int64_t cross_old, int32_t line_len_plus_ext)
{
    int64_t hit_y = (int64_t)ctx->newy;

    if (line_len_plus_ext != 0) {
        int64_t new_d = cross_new / (int64_t)line_len_plus_ext;
        int64_t old_d = cross_old / (int64_t)line_len_plus_ext;
        int64_t denom = old_d - new_d;
        if (denom <= 0) denom = 1;

        {
            int64_t dy = (int64_t)ctx->newy - (int64_t)ctx->oldy;
            if (dy != 0) {
                hit_y = (int64_t)ctx->newy + (dy / denom) * new_d;
            }
        }
    }

    if (hit_y < (int64_t)INT32_MIN) hit_y = (int64_t)INT32_MIN;
    if (hit_y > (int64_t)INT32_MAX) hit_y = (int64_t)INT32_MAX;
    return (int32_t)hit_y;
}

static bool pass1_vertical_hit_asm(const MoveContext *ctx, const uint8_t *target_zone, int32_t hit_y)
{
    int32_t lower_floor = read_be32(target_zone + ZONE_FLOOR_HEIGHT);
    int32_t lower_roof = read_be32(target_zone + ZONE_ROOF_HEIGHT);
    int32_t upper_floor = read_be32(target_zone + ZONE_UPPER_FLOOR);
    int32_t upper_roof = read_be32(target_zone + ZONE_UPPER_ROOF);

    int64_t d6 = (int64_t)hit_y + (int64_t)ctx->thing_height - (int64_t)ctx->step_up_val;

    if (d6 >= (int64_t)lower_floor) return true;
    if ((int64_t)hit_y > (int64_t)lower_roof) return false;
    if ((int64_t)hit_y < (int64_t)upper_roof) return true;
    if (d6 < (int64_t)upper_floor) return false;
    return true;
}

static bool pass1_exit_is_nonblocking(const MoveContext *ctx, const uint8_t *target_zone,
    int32_t lx, int32_t lz, int16_t lxlen, int16_t lzlen, int32_t a4, int32_t a6,
    int32_t orig_newx, int32_t orig_newz, int32_t line_len)
{
    int64_t dx = (int64_t)lxlen - (int64_t)a4 - (int64_t)a6;
    int64_t dz = (int64_t)lzlen + (int64_t)a4 - (int64_t)a6;

    int64_t new_rel_x = (int64_t)orig_newx - (int64_t)lx - (int64_t)a4;
    int64_t new_rel_z = (int64_t)orig_newz - (int64_t)lz - (int64_t)a6;
    int64_t old_rel_x = (int64_t)ctx->oldx - (int64_t)lx - (int64_t)a4;
    int64_t old_rel_z = (int64_t)ctx->oldz - (int64_t)lz - (int64_t)a6;

    int64_t cross_new = new_rel_x * dz - new_rel_z * dx;
    int64_t cross_old = old_rel_x * dz - old_rel_z * dx;

    /* Same side as Amiga checkwalls -> not a blocking crossing. */
    if (cross_new > 0) return true;

    {
        int32_t hit_y = compute_crossing_height_asm(ctx, cross_new, cross_old, line_len + ctx->extlen);
        return !pass1_vertical_hit_asm(ctx, target_zone, hit_y);
    }
}

static int check_wall_line_otherpass_asm(MoveContext *ctx, const uint8_t *fline,
    int32_t lx, int32_t lz, int16_t lxlen, int16_t lzlen, int32_t a4, int32_t a6,
    int32_t *xdiff, int32_t *zdiff)
{
    int16_t sx = (int16_t)ctx->newx;
    int16_t sz = (int16_t)ctx->newz;
    int16_t ox = (int16_t)ctx->oldx;
    int16_t oz = (int16_t)ctx->oldz;
    int16_t lxw = (int16_t)lx;
    int16_t lzw = (int16_t)lz;
    int16_t a4w = (int16_t)a4;
    int16_t a6w = (int16_t)a6;

    int16_t dx = (int16_t)((int16_t)lxlen - a4w - a6w);
    int16_t dz = (int16_t)((int16_t)lzlen + a4w - a6w);

    int16_t rel_nx = (int16_t)(sx - lxw - a4w);
    int16_t rel_nz = (int16_t)(sz - lzw - a6w);
    int32_t cross_new = (int32_t)dz * (int32_t)rel_nx - (int32_t)dx * (int32_t)rel_nz;
    if (cross_new >= 0) return 0;

    {
        int16_t h = (int16_t)(sx - ox);
        int16_t g = (int16_t)(sz - oz);
        int16_t ea = (int16_t)(ox - lxw - a4w);
        int16_t bf = (int16_t)(lzw + a6w - oz);

        int32_t d1 = (int32_t)g * (int32_t)ea + (int32_t)h * (int32_t)bf;
        int32_t d4 = (int32_t)g * (int32_t)dx - (int32_t)h * (int32_t)dz;

        if (d4 == 0) return 0;

        if (d4 < 0) {
            if (d1 > 0) return 0;
            if (d4 > d1) return 0;
        } else {
            if (d1 < 0) return 0;
            if (d4 < d1) return 0;
        }
    }

    {
        int16_t den = (int16_t)((int16_t)read_be16(fline + FLINE_LENGTH) + (int16_t)ctx->extlen);
        if (den == 0) return 0;

        int16_t d7 = (int16_t)(cross_new / (int32_t)den);
        d7 = (int16_t)(d7 - 3);

        {
            int32_t t6 = (int32_t)(int16_t)d7 * (int32_t)dz;
            int32_t t7 = (int32_t)(int16_t)d7 * (int32_t)dx;
            int16_t d6q = (int16_t)(t6 / (int32_t)den);
            int16_t d7q = (int16_t)(t7 / (int32_t)den);
            int16_t hit_x = (int16_t)(sx - d6q);
            int16_t hit_z = (int16_t)(sz + d7q);

            int16_t rel_ox = (int16_t)(ox - lxw - a4w);
            int16_t rel_oz = (int16_t)(oz - lzw - a6w);
            int32_t cross_old = (int32_t)dz * (int32_t)rel_ox - (int32_t)dx * (int32_t)rel_oz;

            if (cross_old < 0) return 0;

            ctx->newx = (int32_t)hit_x;
            ctx->newz = (int32_t)hit_z;
            mark_floorline_touch_flag(ctx, fline);
            ctx->hitwall = 1;
            ctx->wall_hit_y = ctx->newy;
            *xdiff = ctx->newx - ctx->oldx;
            *zdiff = ctx->newz - ctx->oldz;
            return 2;
        }
    }
}

static int firstpass_othercheck_hit(int16_t hit_x, int16_t hit_z,
    int16_t lx, int16_t lz, int16_t a4, int16_t a6, int16_t d2, int16_t d5)
{
    int16_t d6 = (int16_t)(hit_x - lx - a4);
    int16_t d7 = (int16_t)(hit_z - lz - a6);
    int16_t ad2 = (d2 < 0) ? (int16_t)-d2 : d2;
    int16_t ad5 = (d5 < 0) ? (int16_t)-d5 : d5;

    if (ad5 > ad2) {
        /* Use Z coord (ObjectMove.s UseZ path). */
        if (d7 <= 0) {
            int16_t t = d5;
            if (t > 4) return 0;
            t = (int16_t)(t - 4);
            if (d7 < t) return 0;
            return 1;
        } else {
            int16_t t = d5;
            if (t < -4) return 0;
            t = (int16_t)(t + 4);
            if (d7 > t) return 0;
            return 1;
        }
    }

    /* Use X coord. */
    if (d6 <= 0) {
        int16_t t = d2;
        if (t > 4) return 0;
        t = (int16_t)(t - 4);
        if (d6 < t) return 0;
        return 1;
    } else {
        int16_t t = d2;
        if (t < -4) return 0;
        t = (int16_t)(t + 4);
        if (d6 > t) return 0;
        return 1;
    }
}

static int check_wall_line_walls_asm(MoveContext *ctx, const uint8_t *fline,
    int32_t lx, int32_t lz, int16_t lxlen, int16_t lzlen, int32_t line_len,
    int32_t a4, int32_t a6, const uint8_t *target_zone,
    int32_t *xdiff, int32_t *zdiff)
{
    int16_t sx = (int16_t)ctx->newx;
    int16_t sz = (int16_t)ctx->newz;
    int16_t ox = (int16_t)ctx->oldx;
    int16_t oz = (int16_t)ctx->oldz;
    int16_t lxw = (int16_t)lx;
    int16_t lzw = (int16_t)lz;
    int16_t a4w = (int16_t)a4;
    int16_t a6w = (int16_t)a6;

    int16_t d2 = (int16_t)((int16_t)lxlen - a4w - a6w);
    int16_t d5 = (int16_t)((int16_t)lzlen + a4w - a6w);

    int16_t rel_nx = (int16_t)(sx - lxw - a4w);
    int16_t rel_nz = (int16_t)(sz - lzw - a6w);
    int32_t cross_new = (int32_t)d5 * (int32_t)rel_nx - (int32_t)d2 * (int32_t)rel_nz;

    if (cross_new > 0) return 0;

    {
        int16_t den = (int16_t)((int16_t)line_len + (int16_t)ctx->extlen);
        if (den == 0) den = 1;

        int16_t d7 = (int16_t)(cross_new / (int32_t)den);
        int16_t rel_ox = (int16_t)(ox - lxw - a4w);
        int16_t rel_oz = (int16_t)(oz - lzw - a6w);
        int32_t cross_old = (int32_t)d5 * (int32_t)rel_ox - (int32_t)d2 * (int32_t)rel_oz;
        int32_t hit_y = compute_crossing_height_asm(ctx, cross_new, cross_old, den);

        if (target_zone && !pass1_vertical_hit_asm(ctx, target_zone, hit_y)) {
            return 0;
        }

        {
            int16_t hit_x = sx;
            int16_t hit_z = sz;

            if (ctx->wallbounce || ctx->exitfirst) {
                int16_t h = (int16_t)(sx - ox);
                int16_t g = (int16_t)(sz - oz);
                int16_t otherd = (int16_t)(cross_old / (int32_t)den);
                int16_t total = (int16_t)(otherd - d7);
                if (total <= 0) total = 1;
                hit_x = (int16_t)(sx + (int16_t)(((int32_t)h * (int32_t)d7) / (int32_t)total));
                hit_z = (int16_t)(sz + (int16_t)(((int32_t)g * (int32_t)d7) / (int32_t)total));
            } else {
                /* Amiga checkwalls divides by (line_length + extlen) here too. */
                int16_t hit_den = den;
                if (hit_den == 0) return 0;
                {
                    int16_t off_x = (int16_t)(((int32_t)d7 * (int32_t)d5) / (int32_t)hit_den);
                    int16_t off_z = (int16_t)(((int32_t)d7 * (int32_t)d2) / (int32_t)hit_den);
                    hit_x = (int16_t)(sx - off_x);
                    hit_z = (int16_t)(sz + off_z);
                }
            }

            if (!firstpass_othercheck_hit(hit_x, hit_z, lxw, lzw, a4w, a6w, d2, d5)) {
                return 0;
            }

            ctx->newx = (int32_t)hit_x;
            ctx->newz = (int32_t)hit_z;
            mark_floorline_touch_flag(ctx, fline);
            ctx->hitwall = 1;
            ctx->wall_hit_y = hit_y;
            ctx->wall_xsize = d2;
            ctx->wall_zsize = d5;
            ctx->wall_length = den;
            *xdiff = ctx->newx - ctx->oldx;
            *zdiff = ctx->newz - ctx->oldz;
            return ctx->exitfirst ? 3 : 2;
        }
    }
}

/* Amiga MoveObject ORs wallflags into floorline word 14(a2) when the mover
 * is close to, or collides with, that wall line. Door/Lift routines consume
 * these bits to detect per-player interaction. */
static void mark_floorline_touch_flag(const MoveContext *ctx, const uint8_t *fline)
{
    if (!ctx || !fline) return;
    if (ctx->wall_flags == 0) return;
    uint8_t *fl = (uint8_t *)fline;
    uint16_t flags = (uint16_t)read_be16(fl + FLINE_AWAY);
    flags |= ctx->wall_flags;
    write_be16(fl + FLINE_AWAY, flags);
}

/* Amiga MoveObject "near wall" test that tags floorline +14 with wallflags:
 * d = ((newx-lx)*zlen - (newz-lz)*xlen) / (line_length + extlen)
 * if (d > 0 && d < 32) or.w wallflags,14(a2)
 *
 * Our coordinates may be shifted by pos_shift, so scale the 32 threshold too. */
static void mark_floorline_touch_if_near(const MoveContext *ctx, const uint8_t *fline,
    int32_t lx, int32_t lz, int16_t lxlen, int16_t lzlen, int ps)
{
    if (!ctx || !fline) return;
    if (ctx->wall_flags == 0) return;

    {
        int64_t side = (int64_t)(ctx->newx - lx) * (int64_t)lzlen -
                       (int64_t)(ctx->newz - lz) * (int64_t)lxlen;
        if (side <= 0) return;

        int32_t line_len = (int32_t)(int16_t)read_be16(fline + FLINE_LENGTH);
        if (line_len < 0) line_len = -line_len;
        int32_t denom = line_len + ctx->extlen;
        if (denom <= 0) denom = 1;

        int64_t d = side / denom;
        int64_t near_thresh = (int64_t)32 << ps;
        if (d < near_thresh)
            mark_floorline_touch_flag(ctx, fline);
    }
}

/* -----------------------------------------------------------------------
 * Geometry helpers (robust, radius-aware wall collision)
 * ----------------------------------------------------------------------- */
static double clamp01d(double t)
{
    if (t < 0.0) return 0.0;
    if (t > 1.0) return 1.0;
    return t;
}

static double orient2d(double ax, double az, double bx, double bz, double cx, double cz)
{
    return (bx - ax) * (cz - az) - (bz - az) * (cx - ax);
}

static bool on_segment(double px, double pz, double qx, double qz, double rx, double rz, double eps)
{
    return (qx >= fmin(px, rx) - eps && qx <= fmax(px, rx) + eps &&
        qz >= fmin(pz, rz) - eps && qz <= fmax(pz, rz) + eps);
}

static bool segments_intersect(double ax, double az, double bx, double bz,
    double cx, double cz, double dx, double dz)
{
    /* Standard 2D segment intersection including colinear touching. */
    const double eps = 1e-9;

    double o1 = orient2d(ax, az, bx, bz, cx, cz);
    double o2 = orient2d(ax, az, bx, bz, dx, dz);
    double o3 = orient2d(cx, cz, dx, dz, ax, az);
    double o4 = orient2d(cx, cz, dx, dz, bx, bz);

    if (((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps)) &&
        ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps))) {
        return true;
    }

    if (fabs(o1) <= eps && on_segment(ax, az, cx, cz, bx, bz, eps)) return true;
    if (fabs(o2) <= eps && on_segment(ax, az, dx, dz, bx, bz, eps)) return true;
    if (fabs(o3) <= eps && on_segment(cx, cz, ax, az, dx, dz, eps)) return true;
    if (fabs(o4) <= eps && on_segment(cx, cz, bx, bz, dx, dz, eps)) return true;

    return false;
}

static double point_segment_distance_and_normal(double px, double pz,
    double ax, double az,
    double bx, double bz,
    double* out_nx, double* out_nz)
{
    double dx = bx - ax;
    double dz = bz - az;
    double len2 = dx * dx + dz * dz;

    double t = 0.0;
    if (len2 > 0.0) {
        t = ((px - ax) * dx + (pz - az) * dz) / len2;
        t = clamp01d(t);
    }

    double cx = ax + t * dx;
    double cz = az + t * dz;

    double nx = px - cx;
    double nz = pz - cz;
    double d2 = nx * nx + nz * nz;

    if (d2 <= 0.0) {
        /* Exact on segment: choose a stable perpendicular normal. */
        double nnx = dz;
        double nnz = -dx;
        double nlen2 = nnx * nnx + nnz * nnz;
        if (nlen2 > 0.0) {
            double inv = 1.0 / sqrt(nlen2);
            nnx *= inv;
            nnz *= inv;
        }
        else {
            nnx = 1.0;
            nnz = 0.0;
        }
        if (out_nx) *out_nx = nnx;
        if (out_nz) *out_nz = nnz;
        return 0.0;
    }

    {
        double d = sqrt(d2);
        if (out_nx) *out_nx = nx / d;
        if (out_nz) *out_nz = nz / d;
        return d;
    }
}

static double segment_segment_distance(double ax, double az, double bx, double bz,
    double cx, double cz, double dx, double dz)
{
    if (segments_intersect(ax, az, bx, bz, cx, cz, dx, dz)) return 0.0;

    {
        double best = DBL_MAX;
        double tmp;

        tmp = point_segment_distance_and_normal(ax, az, cx, cz, dx, dz, NULL, NULL);
        if (tmp < best) best = tmp;

        tmp = point_segment_distance_and_normal(bx, bz, cx, cz, dx, dz, NULL, NULL);
        if (tmp < best) best = tmp;

        tmp = point_segment_distance_and_normal(cx, cz, ax, az, bx, bz, NULL, NULL);
        if (tmp < best) best = tmp;

        tmp = point_segment_distance_and_normal(dx, dz, ax, az, bx, bz, NULL, NULL);
        if (tmp < best) best = tmp;

        return best;
    }
}

static bool do_wall_slide_radius(MoveContext* ctx,
    double ax, double az,
    double bx, double bz,
    double radius_scaled)
{
    if (ctx->wallbounce) {
        ctx->newx = ctx->oldx;
        ctx->newz = ctx->oldz;
        return true;
    }

    {
        double px = (double)ctx->newx;
        double pz = (double)ctx->newz;

        double nx = 0.0, nz = 0.0;
        double dist = point_segment_distance_and_normal(px, pz, ax, az, bx, bz, &nx, &nz);

        if (dist >= radius_scaled) return false;

        /* Push out by penetration depth + small epsilon to avoid re-hitting due to rounding. */
        {
            const double eps = 1.0;
            double penetration = radius_scaled - dist;
            px += nx * (penetration + eps);
            pz += nz * (penetration + eps);
        }

        /* Slide: remove inward component along the collision normal. */
        {
            double vx = px - (double)ctx->oldx;
            double vz = pz - (double)ctx->oldz;
            double vn = vx * nx + vz * nz;

            if (vn < 0.0) {
                vx -= vn * nx;
                vz -= vn * nz;
                px = (double)ctx->oldx + vx;
                pz = (double)ctx->oldz + vz;
            }
        }

        /* Final safety: ensure outside after slide too. */
        {
            const double eps = 1.0;
            dist = point_segment_distance_and_normal(px, pz, ax, az, bx, bz, &nx, &nz);
            if (dist < radius_scaled) {
                double penetration = radius_scaled - dist;
                px += nx * (penetration + eps);
                pz += nz * (penetration + eps);
            }
        }

        ctx->newx = (int32_t)llround(px);
        ctx->newz = (int32_t)llround(pz);
        return true;
    }
}

/* -----------------------------------------------------------------------
 * True iff path (oldx,oldz)->(newx,newz) intersects the wall SEGMENT
 * (lx,lz) to (lx+wx, lz+wz) where all coordinates are in the same
 * (shifted) coordinate space. Returns false for the infinite line extension.
 * ----------------------------------------------------------------------- */
static int path_hits_segment(int32_t oldx, int32_t oldz, int32_t newx, int32_t newz,
    int32_t lx, int32_t lz, int32_t wx, int32_t wz)
{
    int64_t dx = (int64_t)(newx - oldx);
    int64_t dz = (int64_t)(newz - oldz);
    int64_t old_cross = (int64_t)(oldx - lx) * wz - (int64_t)(oldz - lz) * wx;
    int64_t denom = dx * (int64_t)wz - dz * (int64_t)wx;
    if (denom == 0) return 0;

    /* t = -old_cross / denom = param along path (0 = old, 1 = new) */
    {
        int64_t t_num = -old_cross;
        if (denom > 0) {
            if (t_num < 0 || t_num > denom) return 0;
        }
        else {
            if (t_num > 0 || t_num < denom) return 0;
        }

        /* Param s on wall segment */
        {
            int64_t wall_len_sq = (int64_t)wx * wx + (int64_t)wz * wz;
            if (wall_len_sq == 0) return 0;

            int64_t s_num = (int64_t)(oldx - lx) * wx + (int64_t)(oldz - lz) * wz;
            s_num += (t_num * (dx * (int64_t)wx + dz * (int64_t)wz)) / denom;

            if (s_num < 0 || s_num > wall_len_sq) return 0;
            return 1;
        }
    }
}

/* Amiga find_room checkifcrossed gate:
 *  - new position must be on "left" side of line (cross < 0)
 *  - both line endpoints must straddle the old->new movement vector
 * Uses 16-bit coordinate arithmetic like ObjectMove.s. */
static int find_room_crosses_exit_asm(const MoveContext *ctx,
    int32_t lx, int32_t lz, int16_t lxlen, int16_t lzlen,
    int32_t *out_cross_new, int32_t *out_cross_old)
{
    int16_t sx = (int16_t)ctx->newx;
    int16_t sz = (int16_t)ctx->newz;
    int16_t ox = (int16_t)ctx->oldx;
    int16_t oz = (int16_t)ctx->oldz;
    int16_t lxw = (int16_t)lx;
    int16_t lzw = (int16_t)lz;

    int16_t rel_nx = (int16_t)(sx - lxw);
    int16_t rel_nz = (int16_t)(sz - lzw);
    int32_t cross_new = (int32_t)rel_nx * (int32_t)lzlen - (int32_t)rel_nz * (int32_t)lxlen;
    if (out_cross_new) *out_cross_new = cross_new;
    if (cross_new >= 0) return 0;

    {
        int16_t dx = (int16_t)(sx - ox);
        int16_t dz = (int16_t)(sz - oz);

        int16_t e0x = (int16_t)(lxw - ox);
        int16_t e0z = (int16_t)(lzw - oz);
        int32_t side0 = (int32_t)dz * (int32_t)e0x - (int32_t)dx * (int32_t)e0z;
        if (side0 > 0) return 0;

        {
            int16_t ex1 = (int16_t)(lxw + lxlen);
            int16_t ez1 = (int16_t)(lzw + lzlen);
            int16_t e1x = (int16_t)(ex1 - ox);
            int16_t e1z = (int16_t)(ez1 - oz);
            int32_t side1 = (int32_t)dz * (int32_t)e1x - (int32_t)dx * (int32_t)e1z;
            if (side1 < 0) return 0;
        }
    }

    if (out_cross_old) {
        int16_t rel_ox = (int16_t)(ox - lxw);
        int16_t rel_oz = (int16_t)(oz - lzw);
        *out_cross_old = (int32_t)rel_ox * (int32_t)lzlen - (int32_t)rel_oz * (int32_t)lxlen;
    }

    return 1;
}

/* -----------------------------------------------------------------------
 * Helper: Newton-Raphson square root (from ObjectMove.s CalcDist)
 * ----------------------------------------------------------------------- */
static int32_t newton_sqrt(int32_t dx, int32_t dz)
{
    if (dx < 0) dx = -dx;
    if (dz < 0) dz = -dz;

    if (dx == 0 && dz == 0) return 0;

    {
        int32_t guess = (dx + dz) / 2;
        int64_t sum_sq;

        if (guess == 0) guess = 1;

        sum_sq = (int64_t)dx * dx + (int64_t)dz * dz;

        for (int i = 0; i < 3; i++) {
            if (guess == 0) break;
            guess = (int32_t)((guess + sum_sq / guess) / 2);
        }

        return guess;
    }
}

/* -----------------------------------------------------------------------
 * move_context_init
 * ----------------------------------------------------------------------- */
void move_context_init(MoveContext* ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->extlen = 40;
    ctx->step_up_val = 40 * 256;
    ctx->step_down_val = 0x1000000;
    ctx->coll_id = -1;
    ctx->wall_xsize = 0;
    ctx->wall_zsize = 0;
    ctx->wall_length = 0;
}

/* -----------------------------------------------------------------------
 * player_fall - Gravity physics
 * ----------------------------------------------------------------------- */
void player_fall(int32_t* yoff, int32_t* yvel, int32_t tyoff,
    int32_t water_level, bool in_water)
{
    int32_t d0 = tyoff - *yoff;
    int32_t d1 = *yoff;
    int32_t d2 = *yvel;

    if (d0 > 0) {
        /* Above ground - falling */
        d1 += d2;
        d2 += GRAVITY_ACCEL;

        if (in_water && d1 >= water_level) {
            if (d2 > WATER_MAX_VELOCITY) d2 = WATER_MAX_VELOCITY;
        }
    }
    else {
        /* At or below ground */
        d2 -= GRAVITY_DECEL;
        /* Amiga Fall.s:
         *   sub.l #512,d2
         *   blt.s .notfast
         *   move.l #0,d2
         * Clamp only non-negative upward velocity to 0; keep negative lift velocity
         * so stand-up / upward correction can continue toward tyoff. */
        if (d2 >= 0) d2 = 0;

        d1 += d2;
        d0 = tyoff - d1;

        if (d0 > 0) {
            d2 = 0;
            d1 += d0;
        }
    }

    *yvel = d2;
    *yoff = d1;
}

/* -----------------------------------------------------------------------
 * check_wall_line - Phase 1: Check one floor line for wall collision.
 *
 * Returns:
 *   0 = no hit
 *   2 = wall hit, slid (restart scan)
 *   3 = wall hit, reverted (stop)
 * ----------------------------------------------------------------------- */
static int check_wall_line(MoveContext* ctx, LevelState* level,
    const uint8_t* fline, const uint8_t* zone_data,
    int32_t* xdiff, int32_t* zdiff, int pass_kind)
{
    int ps = ctx->pos_shift;
    int pass_is_walls = (pass_kind == MOVE_PASS_WALLS || pass_kind == MOVE_PASS_BRUTE) ? 1 : 0;
    int pass_is_other = (pass_kind == MOVE_PASS_OTHER) ? 1 : 0;

    int16_t connect = (int16_t)read_be16(fline + FLINE_CONNECT);
    int connect_index = level_connect_to_zone_index(level, connect);
    int zone_slots = level_zone_slot_count(level);
    int32_t lx = (int32_t)(int16_t)read_be16(fline + FLINE_X) << ps;
    int32_t lz = (int32_t)(int16_t)read_be16(fline + FLINE_Z) << ps;
    int16_t lxlen = (int16_t)read_be16(fline + FLINE_XLEN);
    int16_t lzlen = (int16_t)read_be16(fline + FLINE_ZLEN);
    int32_t line_len = (int32_t)(int16_t)read_be16(fline + FLINE_LENGTH);
    int32_t a4 = 0;
    int32_t a6 = 0;
    int32_t orig_newx = ctx->newx;
    int32_t orig_newz = ctx->newz;

    int32_t wx = (int32_t)lxlen << ps;
    int32_t wz = (int32_t)lzlen << ps;
    const uint8_t *target_zone = NULL;
    uint8_t *target_room = NULL;
    int has_target_zone = 0;

    get_floorline_shift_offsets(ctx, fline, &a4, &a6);

    /* Amiga tags touch flags before deciding whether an exit is passable. */
    mark_floorline_touch_if_near(ctx, fline, lx, lz, lxlen, lzlen, ps);

    /* ---- Exit line handling. ---- */
    if (zone_data && level->zone_adds && connect_index >= 0 && connect_index < zone_slots) {
        int32_t target_zone_off = read_be32(level->zone_adds + (size_t)connect_index * 4);
        target_zone = level->data + target_zone_off;
        target_room = (uint8_t*)(level->data + target_zone_off);
        has_target_zone = 1;
        int dbg_type = -1;
        int is_enemy = move_ctx_is_enemy(ctx, level, &dbg_type) ? 1 : 0;

        if (pass_is_other) {
            int8_t target_in_top = 0;
            if (transition_height_ok_zone(ctx, ctx->newy, target_zone, &target_in_top)) {
                if (!(ctx->no_transition_back && target_room == ctx->no_transition_back)) {
                    /* checkotherwalls: passable exit -> skip wall collision. */
                    return 0;
                }
                if (move_debug_enabled()) {
                    if (is_enemy) {
                        int cur_zone = level_zone_index_from_room_ptr(level, zone_data);
                        int line_idx = move_debug_line_index(level, fline);
                        printf("[MOVEDBG] no_back_block cid=%d type=%d line=%d zone=%d->%d connect=%d in_top=%d pos=(%d,%d) y=%d\n",
                               (int)ctx->coll_id, dbg_type, line_idx, cur_zone, connect_index, (int)connect,
                               (int)target_in_top, (int)ctx->newx, (int)ctx->newz, (int)ctx->newy);
                    }
                }
                /* If blocked by no_transition_back, treat as wall below. */
            } else {
                if (move_debug_enabled() && is_enemy) {
                    int cur_zone = level_zone_index_from_room_ptr(level, zone_data);
                    int line_idx = move_debug_line_index(level, fline);

                    int32_t lf = read_be32(target_zone + ZONE_FLOOR_HEIGHT);
                    int32_t lr = read_be32(target_zone + ZONE_ROOF_HEIGHT);
                    int32_t uf = read_be32(target_zone + ZONE_UPPER_FLOOR);
                    int32_t ur = read_be32(target_zone + ZONE_UPPER_ROOF);

                    long long lclear = (long long)lf - (long long)lr;
                    long long ld1 = (long long)ctx->newy + (long long)ctx->thing_height - (long long)lf;
                    long long uclear = (long long)uf - (long long)ur;
                    long long ud1 = (long long)ctx->newy + (long long)ctx->thing_height - (long long)uf;
                    int lroof_ok = ((long long)ctx->newy - (long long)lr >= 0) ? 1 : 0;
                    int uroof_ok = ((long long)ctx->newy - (long long)ur >= 0) ? 1 : 0;

                    printf("[MOVEDBG] step_block cid=%d type=%d line=%d zone=%d->%d connect=%d pos=(%d,%d) y=%d h=%d up=%d down=%d "
                           "L{f=%d r=%d clr=%lld d1=%lld roof_ok=%d} U{f=%d r=%d clr=%lld d1=%lld roof_ok=%d}\n",
                           (int)ctx->coll_id, dbg_type, line_idx, cur_zone, connect_index, (int)connect,
                           (int)ctx->newx, (int)ctx->newz, (int)ctx->newy, (int)ctx->thing_height,
                           (int)ctx->step_up_val, (int)ctx->step_down_val,
                           (int)lf, (int)lr, lclear, ld1, lroof_ok,
                           (int)uf, (int)ur, uclear, ud1, uroof_ok);
                }
            }
        }
    }

    /* Amiga checkotherwalls uses its own integer hit test (no radius slide). */
    if (pass_is_other && ctx->extlen != 0) {
        return check_wall_line_otherpass_asm(ctx, fline, lx, lz, lxlen, lzlen, a4, a6, xdiff, zdiff);
    }

    /* Amiga checkwalls uses integer wall math for all movers, including extlen==0 bullets. */
    if (pass_is_walls) {
        return check_wall_line_walls_asm(ctx, fline, lx, lz, lxlen, lzlen, line_len,
            a4, a6, has_target_zone ? target_zone : NULL, xdiff, zdiff);
    }

    /* ---- Wall collision with extents ----
     * Treat mover as a circle of radius (ctx->extlen / 2) so small gaps are passable.
     * If extlen == 0, fall back to old point-crossing behavior.
     */
    {
        int32_t ext = ctx->extlen;
        if (ext < 0) ext = 0;
        ext = ext / 2;  /* use half extlen for wall radius */

        if (ext == 0) {
            /* Point-crossing fallback (your previous behavior). */
            int64_t new_cross = (int64_t)(ctx->newx - lx) * lzlen -
                (int64_t)(ctx->newz - lz) * lxlen;
            int64_t old_cross = (int64_t)(ctx->oldx - lx) * lzlen -
                (int64_t)(ctx->oldz - lz) * lxlen;

            if ((new_cross ^ old_cross) >= 0) return 0;
            if (old_cross == 0 || new_cross == 0) return 0;

            if (!path_hits_segment(ctx->oldx, ctx->oldz, ctx->newx, ctx->newz, lx, lz, wx, wz)) {
                return 0;
            }

            {
                int64_t den = old_cross - new_cross;
                if (den != 0) {
                    int64_t dx = (int64_t)ctx->newx - (int64_t)ctx->oldx;
                    int64_t dz = (int64_t)ctx->newz - (int64_t)ctx->oldz;
                    int64_t dy = (int64_t)ctx->newy - (int64_t)ctx->oldy;
                    int64_t hit_x = (int64_t)ctx->oldx + (dx * old_cross) / den;
                    int64_t hit_z = (int64_t)ctx->oldz + (dz * old_cross) / den;
                    int64_t hit_y = (int64_t)ctx->oldy + (dy * old_cross) / den;
                    ctx->newx = (int32_t)hit_x;
                    ctx->newz = (int32_t)hit_z;
                    ctx->wall_hit_y = (int32_t)hit_y;
                } else {
                    ctx->newx = ctx->oldx;
                    ctx->newz = ctx->oldz;
                    ctx->wall_hit_y = ctx->newy;
                }
            }
            mark_floorline_touch_flag(ctx, fline);
            ctx->hitwall = 1;
            if (pass_is_walls && has_target_zone &&
                pass1_exit_is_nonblocking(ctx, target_zone, lx, lz, lxlen, lzlen, a4, a6,
                    orig_newx, orig_newz, line_len)) {
                ctx->newx = orig_newx;
                ctx->newz = orig_newz;
                ctx->hitwall = 0;
                ctx->wall_hit_y = ctx->newy;
                return 0;
            }
            return 3;
        }

        {
            double radius = (double)((int64_t)ext << ps);
            int64_t old_cross = (int64_t)(ctx->oldx - lx) * lzlen -
                                (int64_t)(ctx->oldz - lz) * lxlen;
            int64_t new_cross = (int64_t)(ctx->newx - lx) * lzlen -
                                (int64_t)(ctx->newz - lz) * lxlen;
            int crossed_blocked_side = (old_cross >= 0 && new_cross < 0);

            double ax = (double)lx;
            double az = (double)lz;
            double bx = (double)(lx + wx);
            double bz = (double)(lz + wz);

            double p0x = (double)ctx->oldx;
            double p0z = (double)ctx->oldz;
            double p1x = (double)ctx->newx;
            double p1z = (double)ctx->newz;

            /* Quick reject: if movement segment stays farther than radius, no hit. */
            double d_path = segment_segment_distance(p0x, p0z, p1x, p1z, ax, az, bx, bz);
            if (d_path >= radius) return 0;

            /* Optional "moving away" reject (helps avoid sticky edges if already close). */
            {
                double d_old = point_segment_distance_and_normal(p0x, p0z, ax, az, bx, bz, NULL, NULL);
                double d_new = point_segment_distance_and_normal(p1x, p1z, ax, az, bx, bz, NULL, NULL);
                if (!crossed_blocked_side && d_old < radius && d_new > d_old) return 0;
            }

            if (do_wall_slide_radius(ctx, ax, az, bx, bz, radius)) {
                int64_t slide_cross = (int64_t)(ctx->newx - lx) * lzlen -
                                      (int64_t)(ctx->newz - lz) * lxlen;
                mark_floorline_touch_flag(ctx, fline);
                ctx->hitwall = 1;
                ctx->wall_hit_y = ctx->newy;
                if (pass_is_walls && has_target_zone &&
                    pass1_exit_is_nonblocking(ctx, target_zone, lx, lz, lxlen, lzlen, a4, a6,
                        orig_newx, orig_newz, line_len)) {
                    ctx->newx = orig_newx;
                    ctx->newz = orig_newz;
                    ctx->hitwall = 0;
                    ctx->wall_hit_y = ctx->newy;
                    return 0;
                }
                if (crossed_blocked_side && slide_cross < 0) {
                    /* Keep solid/blocked lines one-sided: never accept a result
                     * that leaves us on the far side of this segment. */
                    ctx->newx = ctx->oldx;
                    ctx->newz = ctx->oldz;
                    return 3;
                }
                *xdiff = ctx->newx - ctx->oldx;
                *zdiff = ctx->newz - ctx->oldz;
                return 2;
            }

            mark_floorline_touch_flag(ctx, fline);
            ctx->newx = ctx->oldx;
            ctx->newz = ctx->oldz;
            ctx->hitwall = 1;
            ctx->wall_hit_y = ctx->newy;
            if (pass_is_walls && has_target_zone &&
                pass1_exit_is_nonblocking(ctx, target_zone, lx, lz, lxlen, lzlen, a4, a6,
                    orig_newx, orig_newz, line_len)) {
                ctx->newx = orig_newx;
                ctx->newz = orig_newz;
                ctx->hitwall = 0;
                ctx->wall_hit_y = ctx->newy;
                return 0;
            }
            return 3;
        }
    }
}

/* -----------------------------------------------------------------------
 * find_room - Phase 3: Determine which room the final position is in.
 * ----------------------------------------------------------------------- */
static void find_room(MoveContext* ctx, LevelState* level,
    const uint8_t** zone_data_ptr)
{
    const uint8_t* zone_data = *zone_data_ptr;
    if (!zone_data || !level->zone_adds) return;
    int zone_slots = level_zone_slot_count(level);
    if (zone_slots <= 0) return;

    {
        int ps = ctx->pos_shift;
        int16_t list_off = read_be16(zone_data + ZONE_EXIT_LIST);
        const uint8_t* list_ptr = zone_data + list_off;

        for (int i = 0; i < 128; i++) {
            int16_t entry = read_be16(list_ptr + i * 2);
            if (entry < 0) break; /* -1 ends exit portion */
            if (entry >= level->num_floor_lines) continue;

            {
                const uint8_t* fline = level->floor_lines + entry * FLINE_SIZE;
                int16_t connect = (int16_t)read_be16(fline + FLINE_CONNECT);
                int connect_index = level_connect_to_zone_index(level, connect);
                if (connect_index < 0) continue;
                if (connect_index >= zone_slots) continue;

                int32_t lx = (int32_t)(int16_t)read_be16(fline + FLINE_X) << ps;
                int32_t lz = (int32_t)(int16_t)read_be16(fline + FLINE_Z) << ps;
                int16_t lxlen = (int16_t)read_be16(fline + FLINE_XLEN);
                int16_t lzlen = (int16_t)read_be16(fline + FLINE_ZLEN);
                int32_t cross_new = 0;
                int32_t cross_old = 0;
                if (!find_room_crosses_exit_asm(ctx, lx, lz, lxlen, lzlen, &cross_new, &cross_old))
                    continue;

                {
                    int32_t target_zone_off = read_be32(level->zone_adds + (size_t)connect_index * 4);
                    const uint8_t* target_zone = level->data + target_zone_off;
                    int8_t target_in_top = 0;
                    {
                        /* Amiga find_room computes crossing height and sets StoodInTop
                         * from target lower-roof compare:
                         *   cmp.l ToZoneRoof(a4),d4 ; slt StoodInTop */
                        int32_t line_len = (int32_t)(int16_t)read_be16(fline + FLINE_LENGTH);
                        int32_t cross_y = ctx->newy;
                        if (line_len != 0) {
                            cross_y = compute_crossing_height_asm(ctx, (int64_t)cross_new, (int64_t)cross_old, line_len);
                        }

                        {
                            int32_t target_roof = read_be32(target_zone + ZONE_ROOF_HEIGHT);
                            target_in_top = (cross_y < target_roof) ? 1 : 0;
                        }
                    }

                    {
                        uint8_t* target_room = (uint8_t*)(level->data + target_zone_off);
                        if (ctx->no_transition_back && target_room == ctx->no_transition_back) {
                            if (move_debug_enabled()) {
                                int dbg_type = -1;
                                if (move_debug_should_log(ctx, level, &dbg_type)) {
                                    int cur_zone = level_zone_index_from_room_ptr(level, zone_data);
                                    printf("[MOVEDBG] transition_rejected_no_back cid=%d type=%d line=%d zone=%d->%d pos=(%d,%d) y=%d\n",
                                           (int)ctx->coll_id, dbg_type, (int)entry, cur_zone, connect_index,
                                           (int)ctx->newx, (int)ctx->newz, (int)ctx->newy);
                                }
                            }
                            continue;
                        }

                        if (move_debug_enabled()) {
                            int dbg_type = -1;
                            if (move_debug_should_log(ctx, level, &dbg_type)) {
                                int cur_zone = level_zone_index_from_room_ptr(level, zone_data);
                                printf("[MOVEDBG] transition cid=%d type=%d line=%d zone=%d->%d pos=(%d,%d) y=%d in_top=%d\n",
                                       (int)ctx->coll_id, dbg_type, (int)entry, cur_zone, connect_index,
                                       (int)ctx->newx, (int)ctx->newz, (int)ctx->newy,
                                       (int)target_in_top);
                            }
                        }

                        ctx->objroom = target_room;
                        ctx->stood_in_top = target_in_top;
                        *zone_data_ptr = target_zone;
                        return;
                    }
                }
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * move_object_substepped - Sub-step movement to avoid tunneling through walls
 * ----------------------------------------------------------------------- */
void move_object_substepped(MoveContext* ctx, LevelState* level)
{
    int32_t start_x = ctx->oldx;
    int32_t start_z = ctx->oldz;
    int32_t target_x = ctx->newx;
    int32_t target_z = ctx->newz;
    int32_t dx = target_x - start_x;
    int32_t dz = target_z - start_z;
    int32_t adx = (dx < 0) ? -dx : dx;
    int32_t adz = (dz < 0) ? -dz : dz;
    int32_t maxd = (adx > adz) ? adx : adz;
    int steps = (int)(maxd / 4) + 1;
    if (steps > 32) steps = 32;

    int32_t cur_x = start_x;
    int32_t cur_z = start_z;
    uint8_t* cur_room = ctx->objroom;
    int8_t cur_top = ctx->stood_in_top;
    int8_t any_hit = 0;
    int32_t hit_y = ctx->newy;
    int16_t hit_wx = 0;
    int16_t hit_wz = 0;
    int16_t hit_wlen = 0;

    for (int i = 1; i <= steps; i++) {
        MoveContext step = *ctx;
        step.objroom = cur_room;
        step.stood_in_top = cur_top;
        step.oldx = cur_x;
        step.oldz = cur_z;
        step.newx = start_x + (dx * i) / steps;
        step.newz = start_z + (dz * i) / steps;
        step.xdiff = step.newx - step.oldx;
        step.zdiff = step.newz - step.oldz;

        move_object(&step, level);

        if (step.hitwall) {
            any_hit = 1;
            hit_y = step.wall_hit_y;
            hit_wx = step.wall_xsize;
            hit_wz = step.wall_zsize;
            hit_wlen = step.wall_length;
            if (step.exitfirst) {
                cur_x = step.newx;
                cur_z = step.newz;
                cur_room = step.objroom;
                cur_top = step.stood_in_top;
                break;
            }
        }
        cur_x = step.newx;
        cur_z = step.newz;
        cur_room = step.objroom;
        cur_top = step.stood_in_top;
    }

    ctx->newx = cur_x;
    ctx->newz = cur_z;
    ctx->objroom = cur_room;
    ctx->stood_in_top = cur_top;
    ctx->hitwall = any_hit;
    ctx->wall_hit_y = hit_y;
    ctx->wall_xsize = hit_wx;
    ctx->wall_zsize = hit_wz;
    ctx->wall_length = hit_wlen;
}

/* -----------------------------------------------------------------------
 * move_object - Zone-based collision against current room's walls and exits
 * ----------------------------------------------------------------------- */
void move_object(MoveContext* ctx, LevelState* level)
{
    if (!level || !level->data || !level->floor_lines) {
        ctx->hitwall = 0;
        ctx->wall_hit_y = ctx->newy;
        return;
    }

    ctx->hitwall = 0;
    ctx->wall_hit_y = ctx->newy;
    ctx->wall_xsize = 0;
    ctx->wall_zsize = 0;
    ctx->wall_length = 0;

    {
        int32_t xdiff = ctx->newx - ctx->oldx;
        int32_t zdiff = ctx->newz - ctx->oldz;
        if (xdiff == 0 && zdiff == 0) return;

        const uint8_t* zone_data = ctx->objroom;

        int total_iterations = 0;
        const int max_total = 200;

        if (zone_data) {
        gobackanddoitallagain:
            if (total_iterations >= max_total) return;

            if (total_iterations >= max_total) goto phase3;

            {
                int16_t list_off = read_be16(zone_data + ZONE_EXIT_LIST);
                const uint8_t* list_ptr = zone_data + list_off;

                for (int i = 0; i < 128; i++, total_iterations++) {
                    if (total_iterations >= max_total) goto phase3;

                    int16_t entry = read_be16(list_ptr + i * 2);
                    if (entry < 0) break; /* Amiga checkwalls: stop at first negative */
                    if (entry >= level->num_floor_lines) continue;

                    {
                        const uint8_t* fline = level->floor_lines + entry * FLINE_SIZE;
                        int result = check_wall_line(ctx, level, fline, zone_data, &xdiff, &zdiff, MOVE_PASS_WALLS);

                        /* Amiga MoveObject continues scanning after a slide-adjusted hit
                         * (hitthewall -> checkwalls/checkotherwalls), it does not restart
                         * the pass from the beginning. */
                        if (result == 2) continue;
                        if (result == 3) goto phase3;
                    }
                }
            }

            if (ctx->extlen != 0) {
                if (total_iterations >= max_total) goto phase3;

                {
                    int16_t list_off = read_be16(zone_data + ZONE_EXIT_LIST);
                    const uint8_t* list_ptr = zone_data + list_off;

                    for (int i = 0; i < 128; i++, total_iterations++) {
                        if (total_iterations >= max_total) goto phase3;

                        int16_t entry = read_be16(list_ptr + i * 2);
                        if (entry < 0) {
                            if (entry == -2) break; /* Amiga checkotherwalls terminator */
                            continue;               /* -1 separator */
                        }
                        if (entry >= level->num_floor_lines) continue;

                        {
                            const uint8_t* fline = level->floor_lines + entry * FLINE_SIZE;
                            int result = check_wall_line(ctx, level, fline, zone_data, &xdiff, &zdiff, MOVE_PASS_OTHER);

                            if (result == 2) continue;
                            if (result == 3) goto phase3;
                        }
                    }
                }
            }

        phase3:
            {
                const uint8_t* prev_zone = zone_data;
                find_room(ctx, level, &zone_data);

                if (zone_data != prev_zone) {
                    goto gobackanddoitallagain;
                }
            }
        }
        else {
            int32_t num_lines = (int32_t)level->num_floor_lines;
            if (num_lines <= 0) return;

            if (total_iterations >= max_total) return;

            for (int32_t i = 0; i < num_lines; i++, total_iterations++) {
                if (total_iterations >= max_total) return;

                {
                    const uint8_t* fline = level->floor_lines + i * FLINE_SIZE;
                    int result = check_wall_line(ctx, level, fline, zone_data, &xdiff, &zdiff, MOVE_PASS_BRUTE);

                    if (result == 2) continue;
                    if (result == 3) return;
                }
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * collision_check - Object-to-object collision
 * ----------------------------------------------------------------------- */
void collision_check(MoveContext* ctx, LevelState* level)
{
    ctx->hitwall = 0;
    if (!level->object_data) return;

    /* Determine mover width. Prefer table width if we can locate the mover object by coll_id.
     * Otherwise fall back to ctx->extlen (which matches your default player value).
     */
    {
        int32_t mover_width = ctx->extlen;
        if (mover_width < 0) mover_width = 0;

        if (ctx->coll_id >= 0) {
            int find_i = 0;
            while (1) {
                GameObject* o = (GameObject*)(level->object_data + find_i * OBJECT_SIZE);
                if (OBJ_CID(o) < 0) break;
                if (OBJ_CID(o) == ctx->coll_id) {
                    int mt = o->obj.number;
                    if (mt >= 0 && mt <= 20) mover_width = col_box_table[mt].width;
                    break;
                }
                find_i++;
            }
        }

        /* Iterate all objects and test collision. */
        {
            int obj_index = 0;
            while (1) {
                GameObject* obj = (GameObject*)(level->object_data + obj_index * OBJECT_SIZE);

                if (OBJ_CID(obj) < 0) break;

                if (OBJ_CID(obj) == ctx->coll_id && ctx->coll_id >= 0) {
                    obj_index++;
                    continue;
                }

                if (OBJ_ZONE(obj) < 0) {
                    obj_index++;
                    continue;
                }

                {
                    int obj_type = obj->obj.number;
                    if (obj_type < 0 || obj_type > 20) {
                        obj_index++;
                        continue;
                    }

                    {
                        uint32_t type_bit = 1u << obj_type;
                        if (!(ctx->collide_flags & type_bit)) {
                            obj_index++;
                            continue;
                        }
                    }

                    /* Amiga Collision skips objects with zero lives. */
                    if (NASTY_LIVES(*obj) == 0) {
                        obj_index++;
                        continue;
                    }

                    if (ctx->stood_in_top != obj->obj.in_top) {
                        obj_index++;
                        continue;
                    }

                    /* Object position */
                    {
                        int16_t ox = 0, oz = 0;
                        if (level->object_points) {
                            int16_t cid = OBJ_CID(obj);
                            if (cid < 0) {
                                obj_index++;
                                continue;
                            }
                            const uint8_t* p = level->object_points + cid * 8;
                            ox = obj_w(p);
                            oz = obj_w(p + 4);
                        }

                        const CollisionBox* box = &col_box_table[obj_type];

                        int32_t dx = ctx->newx - ox;
                        int32_t dz = ctx->newz - oz;
                        if (dx < 0) dx = -dx;
                        if (dz < 0) dz = -dz;

                        /* Y overlap */
                        {
                            int16_t obj_y = (int16_t)((obj->raw[4] << 8) | obj->raw[5]);
                            int16_t mover_bot = (int16_t)(ctx->newy >> 7);
                            int16_t mover_top = (int16_t)((ctx->newy + ctx->thing_height) >> 7);

                            int16_t obj_bot = obj_y - box->half_height;
                            int16_t obj_top = obj_y + box->half_height;

                            if (mover_top < obj_bot || mover_bot > obj_top) {
                                obj_index++;
                                continue;
                            }
                        }

                        /* Amiga ObjectMove.s Collision axis check:
                         * if |dz|>|dx| use z axis only; else use x axis and then
                         * apply moving-closer check. */
                        if (dz > dx) {
                            int32_t z_pen = dz - mover_width;
                            if (z_pen > (int32_t)box->width) {
                                obj_index++;
                                continue;
                            }
                        } else {
                            int32_t x_pen = dx - mover_width;
                            if (x_pen > (int32_t)box->width) {
                                obj_index++;
                                continue;
                            }

                            /* Moving closer check (Amiga .checkx branch only). */
                            {
                                int32_t new_dx = (int32_t)ox - ctx->newx;
                                int32_t new_dz = (int32_t)oz - ctx->newz;
                                int32_t new_dist_sq = new_dx * new_dx + new_dz * new_dz;

                                int32_t old_dx = (int32_t)ox - ctx->oldx;
                                int32_t old_dz = (int32_t)oz - ctx->oldz;
                                int32_t old_dist_sq = old_dx * old_dx + old_dz * old_dz;

                                if (new_dist_sq > old_dist_sq) {
                                    obj_index++;
                                    continue;
                                }
                            }
                        }

                        ctx->hitwall = 1;
                        return;
                    }
                }

            }
        }
    }
}

/* -----------------------------------------------------------------------
 * head_towards - Move toward a target point
 * ----------------------------------------------------------------------- */
void head_towards(MoveContext* ctx, int32_t target_x, int32_t target_z,
    int16_t speed)
{
    int32_t dx = target_x - ctx->oldx;
    int32_t dz = target_z - ctx->oldz;
    int32_t dist = newton_sqrt(dx, dz);

    if (dist == 0) {
        ctx->newx = ctx->oldx;
        ctx->newz = ctx->oldz;
        return;
    }

    if (dist <= speed) {
        ctx->newx = target_x;
        ctx->newz = target_z;
        return;
    }

    ctx->newx = ctx->oldx + (dx * speed) / dist;
    ctx->newz = ctx->oldz + (dz * speed) / dist;
}

/* -----------------------------------------------------------------------
 * head_towards_angle - Steer facing toward target and move forward
 * ----------------------------------------------------------------------- */
void head_towards_angle(MoveContext* ctx, int16_t* facing,
    int32_t target_x, int32_t target_z,
    int16_t speed, int16_t turn_speed)
{
    int32_t dx = target_x - ctx->oldx;
    int32_t dz = target_z - ctx->oldz;

    if (dx == 0 && dz == 0) return;

    {
        double angle_rad = atan2((double)-dx, (double)-dz);
        int16_t target_angle = (int16_t)(angle_rad * (4096.0 / (2.0 * 3.14159265)));
        target_angle = (target_angle * 2) & ANGLE_MASK;

        int16_t current = *facing;
        int16_t diff = target_angle - current;

        while (diff > 4096) diff -= 8192;
        while (diff < -4096) diff += 8192;

        if (diff > turn_speed) diff = turn_speed;
        if (diff < -turn_speed) diff = -turn_speed;

        current += diff;
        current &= ANGLE_MASK;
        *facing = current;

        {
            int16_t sin_val = sin_lookup(current);
            int16_t cos_val = cos_lookup(current);

            ctx->newx = ctx->oldx - ((int32_t)sin_val * speed) / 16384;
            ctx->newz = ctx->oldz - ((int32_t)cos_val * speed) / 16384;
        }
    }
}

/* -----------------------------------------------------------------------
 * check_teleport - Zone teleporter check
 * ----------------------------------------------------------------------- */
bool check_teleport(MoveContext* ctx, LevelState* level, int16_t zone_id)
{
    if (!level->data || !level->zone_adds) return false;
    int zone_slots = level_zone_slot_count(level);
    if (zone_id < 0 || zone_id >= zone_slots) return false;

    {
        int32_t zone_off = read_be32(level->zone_adds + zone_id * 4);
        const uint8_t* zone_data = level->data + zone_off;

        int16_t tel_zone = read_be16(zone_data + 38);
        if (tel_zone < 0) return false;
        if (tel_zone >= zone_slots) return false;

        {
            int16_t tel_x = read_be16(zone_data + 40);
            int16_t tel_z = read_be16(zone_data + 42);

            int32_t src_floor = read_be32(zone_data + ZONE_FLOOR_HEIGHT);

            int32_t dest_zone_off = read_be32(level->zone_adds + tel_zone * 4);
            const uint8_t* dest_zone = level->data + dest_zone_off;
            int32_t dest_floor = read_be32(dest_zone + ZONE_FLOOR_HEIGHT);

            int32_t floor_diff = dest_floor - src_floor;

            ctx->newx = tel_x;
            ctx->newz = tel_z;
            ctx->newy += floor_diff;

            {
                uint32_t old_flags = ctx->collide_flags;
                ctx->collide_flags = 0x7FFFF;
                collision_check(ctx, level);
                ctx->collide_flags = old_flags;
            }

            if (ctx->hitwall) {
                ctx->newy -= floor_diff;
                ctx->hitwall = 0;
                return false;
            }

            ctx->objroom = (uint8_t*)(level->data + dest_zone_off);
            return true;
        }
    }
}

/* -----------------------------------------------------------------------
 * find_close_room - Locate the room nearest to a target point
 * ----------------------------------------------------------------------- */
void find_close_room(MoveContext* ctx, LevelState* level, int16_t distance)
{
    (void)distance;

    if (!level->data || !level->zone_adds) return;

    ctx->step_up_val = 0x1000000;
    ctx->step_down_val = 0x1000000;
    ctx->thing_height = 0;
    ctx->extlen = 0;
    ctx->exitfirst = 1;

    move_object(ctx, level);
}
