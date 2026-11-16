/*
 * Alien Breed 3D I - PC Port
 * math_tables.c - Sine table generation
 *
 * Generates:
 *   - a 4096-entry legacy sine table (original parity values)
 *   - an expanded 8192-entry table used by runtime lookups
 *
 * Values are in the range ~[-16384, 16384].
 * The original Amiga table is stored in data/math/bigsine as raw binary.
 * We generate it from scratch using the same scale factor.
 *
 * The Amiga 68000 code uses 16-bit signed words multiplied together
 * (muls instruction), so the sine table values are scaled to ~16384
 * (2^14) to fit nicely in 16-bit math:
 *   sin_table[i] = (int16_t)(sin(i * 2 * PI / 4096) * 16384)
 */

#include "math_tables.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef AB3D_DOS
#include "math_tables_dos.h"
const int16_t sine_table[ANGLE_TABLE_SIZE] = AB3D_DOS_SINE_VALUES;
const uint32_t interpolation_phase[20] = AB3D_DOS_INTERPOLATION_VALUES;
#else
int16_t sine_table[ANGLE_TABLE_SIZE];
#endif
int16_t sine_table_fine[ANGLE_FINE_TABLE_SIZE];

void math_tables_init(void)
{
#ifndef AB3D_DOS
    /* Build the PC port's 4096-entry table. DOS prepares these same values
     * offline, keeping startup and the lookup path integer-only. */
    for (int i = 0; i < ANGLE_TABLE_SIZE; i++) {
        double angle = (double)i * 2.0 * M_PI / (double)ANGLE_TABLE_SIZE;
        double val = sin(angle) * 16384.0;
        /* Round to nearest and clamp to int16_t range */
        if (val >= 0)
            sine_table[i] = (int16_t)(val + 0.5);
        else
            sine_table[i] = (int16_t)(val - 0.5);
    }

#endif
    /* Expand to 8192 by inserting midpoint samples between legacy entries.
     * Even slots are exact legacy values, odd slots are rounded averages. */
    for (int i = 0; i < ANGLE_TABLE_SIZE; i++) {
        int16_t a = sine_table[i];
        int16_t b = sine_table[(i + 1) & (ANGLE_TABLE_SIZE - 1)];
        int32_t sum = (int32_t)a + (int32_t)b;
        int16_t mid = (int16_t)((sum >= 0) ? ((sum + 1) >> 1) : ((sum - 1) >> 1));

        int fine = i << 1;
        sine_table_fine[fine] = a;
        sine_table_fine[fine + 1] = mid;
    }
}
