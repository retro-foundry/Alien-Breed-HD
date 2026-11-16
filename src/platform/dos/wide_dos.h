#ifndef AB3D_WIDE_DOS_H
#define AB3D_WIDE_DOS_H
#include <stdint.h>
#include <limits.h>
#include "renderer_fixed.h"

/* Exact 128-bit intermediates for the 32-bit compiler. */
typedef struct { uint64_t low,high; } DosWideInteger;

static inline DosWideInteger dos_wide_unsigned_product(uint64_t a,uint64_t b)
{
    uint64_t ll=(uint64_t)(uint32_t)a*(uint32_t)b;
    uint64_t lh=(uint64_t)(uint32_t)a*(b>>32);
    uint64_t hl=(a>>32)*(uint32_t)b;
    DosWideInteger result={ll,(a>>32)*(b>>32)+(lh>>32)+(hl>>32)};
    uint64_t previous=result.low;result.low+=lh<<32;result.high+=result.low<previous;
    previous=result.low;result.low+=hl<<32;result.high+=result.low<previous;
    return result;
}

static inline DosWideInteger dos_wide_product(int64_t a,int32_t b)
{
    uint64_t magnitude=a<0?-(uint64_t)a:(uint64_t)a;
    uint32_t factor=b<0?-(uint32_t)b:(uint32_t)b;
    DosWideInteger result=dos_wide_unsigned_product(magnitude,factor);
    if((a<0)!=(b<0)) {
        result.low=~result.low+1;
        result.high=~result.high+(result.low==0);
    }
    return result;
}

static inline DosWideInteger dos_wide_add(DosWideInteger a,DosWideInteger b)
{
    DosWideInteger sum={a.low+b.low,a.high+b.high};
    sum.high+=sum.low<a.low;
    return sum;
}

static inline DosWideInteger dos_wide_subtract(DosWideInteger a,DosWideInteger b)
{
    DosWideInteger difference={a.low-b.low,a.high-b.high-(a.low<b.low)};
    return difference;
}

static inline int dos_wide_compare(DosWideInteger a,DosWideInteger b)
{
    if(a.high!=b.high)return a.high>b.high?1:-1;
    return a.low==b.low?0:a.low>b.low?1:-1;
}

static inline DosWideInteger dos_wide_multiply_u32(DosWideInteger a,uint32_t b)
{
    DosWideInteger result=dos_wide_unsigned_product(a.low,b);
    result.high+=a.high*b;
    return result;
}

static inline int64_t dos_wide_divide(DosWideInteger value,uint64_t divisor)
{
    if((!value.high && value.low<=INT64_MAX) ||
       (value.high==UINT64_MAX && value.low>INT64_MAX)) {
        int64_t numerator=(int64_t)value.low;
        if(divisor<=INT32_MAX)return renderer_divide_i64_i32(numerator,(int32_t)divisor);
        return numerator/(int64_t)divisor;
    }
    int negative=(int64_t)value.high<0;
    if(negative) {
        value.low=~value.low+1;
        value.high=~value.high+(value.low==0);
    }
    uint64_t quotient=0;
    /* Callers keep the quotient in the signed 64-bit range. */
    for(int shift=63;shift>=0;shift--) {
        uint64_t high=shift?divisor>>(64-shift):0,low=divisor<<shift;
        if(value.high>high || (value.high==high && value.low>=low)) {
            uint64_t previous=value.low;
            value.low-=low;value.high-=high+(previous<low);
            quotient|=UINT64_C(1)<<shift;
        }
    }
    return negative?(int64_t)(-quotient):(int64_t)quotient;
}
#endif
