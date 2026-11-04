#ifndef AB3D_RENDERER_FIXED_H
#define AB3D_RENDERER_FIXED_H
#include <stdint.h>

/* Nearest integer from signed 8.24 projection, ties away from zero.
 * WallRoutine3 projection stays in range of the original bias addition. */
static inline int32_t renderer_project_round_24(int64_t numerator)
{
    return (int32_t)((numerator + (numerator < 0 ? 8388607 : 8388608)) >> 24);
}

/* Exact signed 64/32 division. Most projection/interpolation quotients need
 * one 386 DIV; the high-word step also preserves genuinely wide results.
 * Unlike __divdi3, the divisor's upper word is known to be zero here. */
static inline int64_t renderer_divide_i64_i32(int64_t numerator, int32_t divisor)
{
#ifdef AB3D_DOS
    int negative = (numerator < 0) != (divisor < 0);
    uint64_t magnitude = numerator < 0 ? UINT64_C(0) - (uint64_t)numerator : (uint64_t)numerator;
    uint32_t denominator = divisor < 0 ? 0u - (uint32_t)divisor : (uint32_t)divisor;
    uint32_t high = magnitude >> 32, low = magnitude, quotient_high = 0;
    if (high >= denominator) {
        quotient_high = high / denominator;
        high %= denominator;
    }
    uint32_t quotient, remainder;
    __asm__("divl %4" : "=a"(quotient), "=d"(remainder)
            : "a"(low), "d"(high), "rm"(denominator) : "cc");
    uint64_t result = ((uint64_t)quotient_high << 32) | quotient;
    return (int64_t)(negative ? UINT64_C(0) - result : result);
#else
    return numerator / divisor;
#endif
}

/* Endpoint-anchored wall interpolation: zero/full-span positions and constant
 * attributes already have their exact result, including signed deltas. */
static inline int64_t renderer_wall_interpolate_delta(int64_t delta, int32_t position, int32_t span)
{
#ifdef AB3D_DOS
    if (!position) return 0;
    if (position == span) return delta;
    if (!delta) return 0;
#endif
    return renderer_divide_i64_i32(delta * position, span);
}

/* Nearest projection quotient, ties away from zero. The depth is positive
 * and the projection numerator leaves room for its half-depth bias. */
static inline int64_t renderer_divide_round_i64_i32(int64_t numerator, int32_t depth)
{
    return renderer_divide_i64_i32(numerator >= 0 ? numerator + depth / 2 : numerator - depth / 2, depth);
}

typedef struct {
    int64_t numerator;
    int32_t depth, quotient;
    int valid;
} RendererRoundCache;

/* Adjacent polygon edges share an endpoint. Cache the full division inputs,
 * keeping the screen centre outside the cache so view changes remain exact. */
static inline int32_t renderer_divide_round_cached(RendererRoundCache *cache, int64_t numerator, int32_t depth)
{
    if (cache->valid && cache->numerator == numerator && cache->depth == depth)
        return cache->quotient;
    int32_t quotient = (int32_t)renderer_divide_round_i64_i32(numerator, depth);
    cache->numerator = numerator;
    cache->depth = depth;
    cache->quotient = quotient;
    cache->valid = 1;
    return quotient;
}

/* AB3DI.s texturedwater: cancel the common factor in sine*256/(depth*64).
 * The doubled sine table is bounded by +/-65536, so the numerator fits 32 bits. */
static inline int32_t renderer_water_vertical_refraction(int32_t sine, int32_t distance)
{
    return sine * 4 / (distance + 300);
}

/* WallRoutine3.ChipMem.s wall texture column selection, using the port's
 * perspective-correct tex/z interpolation. inv_z is positive, at most 2^24
 * (the one-world-unit near plane); the resulting column fits int32_t.
 * floor(trunc(tex_z * 2^24 / inv_z) / 2^24) == floor(tex_z / inv_z)
 * for these integer inputs, including negative texture coordinates.
 * The reduced quotient fits a single 386 IDIV instead of libgcc's 64-bit
 * division. Keep the original expression as the portable implementation. */
static inline int32_t renderer_wall_tex_column(int64_t tex_z, int32_t inv_z)
{
#ifdef AB3D_DOS
    int32_t quotient, remainder;
    __asm__("idivl %4"
            : "=a"(quotient), "=d"(remainder)
            : "a"((uint32_t)tex_z), "d"((int32_t)(tex_z >> 32)), "rm"(inv_z)
            : "cc");
    return quotient - (remainder < 0);
#else
    return (int32_t)(((tex_z * INT64_C(16777216)) / inv_z) >> 24);
#endif
}

typedef struct {
    uint32_t quotient, quotient_step, remainder, remainder_step;
    uint32_t fraction, fraction_step, divisor;
} RendererWallTextureWalk;

/* ScreenWallstripdraw with constant inverse Z. Keep the original 16-bit
 * interpolation fraction and Euclidean remainder separately: their carries
 * reproduce floor((phase >> 16) / inv_z), including negative steps. */
static inline int renderer_wall_texture_walk_init(RendererWallTextureWalk *walk,
                                                  int64_t phase, int64_t step, int32_t inv_z)
{
    int64_t integer_step = step >> 16;
    if (inv_z <= 0 || integer_step < INT32_MIN || integer_step > INT32_MAX) return 0;
    int32_t quotient = renderer_wall_tex_column(phase >> 16, inv_z);
    int32_t quotient_step = renderer_wall_tex_column(integer_step, inv_z);
    walk->quotient = (uint32_t)quotient;
    walk->quotient_step = (uint32_t)quotient_step;
    walk->remainder = (uint32_t)((phase >> 16) - (int64_t)quotient * inv_z);
    walk->remainder_step = (uint32_t)(integer_step - (int64_t)quotient_step * inv_z);
    walk->fraction = (uint32_t)phase & 65535;
    walk->fraction_step = (uint32_t)step & 65535;
    walk->divisor = (uint32_t)inv_z;
    return 1;
}

static inline void renderer_wall_texture_walk_advance(RendererWallTextureWalk *walk)
{
    uint32_t fraction = walk->fraction + walk->fraction_step;
    uint32_t remainder = walk->remainder + walk->remainder_step + (fraction >> 16);
    walk->fraction = fraction & 65535;
    walk->quotient += walk->quotient_step;
    if (remainder >= walk->divisor) {
        remainder -= walk->divisor;
        walk->quotient++;
    }
    walk->remainder = remainder;
}
#endif
