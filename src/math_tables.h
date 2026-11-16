/*
 * Alien Breed 3D I - PC Port
 * math_tables.h - Sine/cosine table and math utilities
 *
 * The original game uses a 4096-entry sine table stored in data/math/bigsine.
 * Angles are represented over 8192 units per turn (0-8191).
 * Historically only even indices were sampled (effective 4096 steps), which can
 * introduce visible fixed-point wobble in modern high-resolution rendering.
 *
 * This port keeps the original 4096-entry table and builds an expanded 8192-entry
 * table by inserting midpoint samples between each original entry. This preserves
 * original values at even angles while improving odd-angle smoothness.
 *
 * Cosine is just sine offset by 1024 entries (2048 bytes = 90 degrees).
 *
 * Access pattern:
 *   sin(angle) = fine_sine[angle & 8191]
 *   cos(angle) = fine_sine[(angle + 2048) & 8191]
 */

#ifndef MATH_TABLES_H
#define MATH_TABLES_H

#include <stdint.h>

/* Table size constants */
#define ANGLE_TABLE_SIZE      4096
#define ANGLE_FINE_TABLE_SIZE 8192
#define ANGLE_MASK            (ANGLE_FINE_TABLE_SIZE - 1) /* full-resolution wrap */
#define ANGLE_EVEN_MASK       8190                        /* legacy even-angle mask */
#define ANGLE_90              (ANGLE_FINE_TABLE_SIZE / 4)
#define ANGLE_180             (ANGLE_FINE_TABLE_SIZE / 2)
#define ANGLE_270             ((ANGLE_FINE_TABLE_SIZE * 3) / 4)
#define ANGLE_FULL            ANGLE_FINE_TABLE_SIZE

/* Original and expanded sine tables. */
#ifdef AB3D_DOS
extern const int16_t sine_table[ANGLE_TABLE_SIZE];
extern const uint32_t interpolation_phase[20];
#else
extern int16_t sine_table[ANGLE_TABLE_SIZE];
#endif
extern int16_t sine_table_fine[ANGLE_FINE_TABLE_SIZE];

/* Initialize the sine table (generates values matching Amiga binary) */
void math_tables_init(void);

/* Existing PC HeadTowards/ViewpointToDraw angle, in even byte-angle units.
 * DOS evaluates it with integer vector rotation; desktop retains atan2. */
int16_t math_angle_to_target(int32_t dx, int32_t dz);
/* Truncated Euclidean length used by the PC enemy movement path. */
int32_t math_distance_floor(int32_t dx, int32_t dz);
/* Rounded panorama columns for the renderer's unchanged horizontal FOV. */
int math_sky_view_columns(int left,int right,int focal,int pan_width);

/*
 * Lookup sin/cos by 0..8191 angle index.
 * angle is masked internally.
 */
static inline int16_t sin_lookup(int angle)
{
    return sine_table_fine[angle & ANGLE_MASK];
}

static inline int16_t cos_lookup(int angle)
{
    return sine_table_fine[(angle + ANGLE_90) & ANGLE_MASK];
}

/*
 * Distance calculation (from ObjectMove.s CalcDist).
 * Returns approximate distance using |dx|+|dz| with correction.
 */
static inline int32_t calc_dist_approx(int32_t dx, int32_t dz)
{
    if (dx < 0) dx = -dx;
    if (dz < 0) dz = -dz;
    /* The original uses max(|dx|,|dz|) + min(|dx|,|dz|)/2 approximation */
    if (dx > dz)
        return dx + (dz >> 1);
    else
        return dz + (dx >> 1);
}

/*
 * Euclidean distance (integer sqrt of dx²+dz²).
 * Used for blast radius so splash matches Amiga ComputeBlast (which uses sqrt).
 */
static inline int32_t calc_dist_euclidean(int32_t dx, int32_t dz)
{
    if (dx < 0) dx = -dx;
    if (dz < 0) dz = -dz;
    if (dx == 0 && dz == 0) return 0;
    {
        int64_t sum_sq = (int64_t)dx * dx + (int64_t)dz * dz;
        int32_t guess = (int32_t)((dx + dz) / 2);
        if (guess == 0) guess = 1;
        for (int i = 0; i < 3; i++) {
            if (guess == 0) break;
            guess = (int32_t)((guess + sum_sq / guess) / 2);
        }
        return guess;
    }
}

#endif /* MATH_TABLES_H */
