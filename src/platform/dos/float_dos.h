#ifndef AB3D_FLOAT_DOS_H
#define AB3D_FLOAT_DOS_H
#include <stdint.h>
#include <limits.h>
#include <string.h>

/* SDL3 exposes mouse/display numbers as binary32. Decode their integer
 * part without loading x87; match DJGPP's truncation and overflow result. */
static inline int32_t dos_float_to_i32(const float *value)
{
    uint32_t bits;memcpy(&bits,value,sizeof(bits));
    unsigned exponent=(bits>>23)&255;
    if(exponent<127)return 0;
    if(exponent>=158)return INT32_MIN;
    uint32_t mantissa=(bits&UINT32_C(0x7fffff))|UINT32_C(0x800000);
    int shift=(int)exponent-150;
    uint32_t magnitude=shift>=0?mantissa<<shift:mantissa>>-shift;
    return bits>>31?-(int32_t)magnitude:(int32_t)magnitude;
}
#endif
