#ifndef AB3D_RENDERER_FIXED_H
#define AB3D_RENDERER_FIXED_H
#include <stdint.h>

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
#endif
