/* Independent arithmetic and former floating sampler oracles. */
#include "platform/dos/triangle_dos.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define CHECK(condition) do { if(!(condition)) { \
    printf("DOS triangle failure at line %d: %s\n",__LINE__,#condition); \
    fflush(stdout);exit(1); } } while(0)

static uint32_t seed=0xd4c1a372;
static uint32_t next(void) { seed=seed*1664525u+1013904223u;return seed; }

static void check_wide(void)
{
    DosTriangle triangle={0};
    triangle.denominator=INT64_C(1)<<50;
    int64_t weights[]={triangle.denominator/2,triangle.denominator/4,triangle.denominator/4};
    int32_t constant[]={INT32_MAX,INT32_MAX,INT32_MAX};
    CHECK(dos_triangle_attribute(&triangle,weights,constant,0)==INT32_MAX);
    for(int i=0;i<3;i++)constant[i]=INT32_MIN;
    CHECK(dos_triangle_attribute(&triangle,weights,constant,0)==INT32_MIN);
    int32_t mixed[]={INT32_MAX,-INT32_MAX,INT32_MAX};
    CHECK(dos_triangle_attribute(&triangle,weights,mixed,0)==INT32_MAX/2);
#ifdef __SIZEOF_INT128__
    for(unsigned i=0;i<100000;i++) {
        int64_t a=(int64_t)((uint64_t)next()<<32|next());
        int64_t b=(int64_t)((uint64_t)next()<<32|next());
        int32_t av=(int32_t)next(),bv=(int32_t)next();
        uint64_t divisor=((uint64_t)next()<<32|next())|UINT64_C(0x10000000000);
        divisor&=INT64_MAX;
        __int128 expected=(__int128)a*av+(__int128)b*bv;
        DosWideInteger actual=dos_wide_add(dos_wide_product(a,av),dos_wide_product(b,bv));
        CHECK(actual.low==(uint64_t)expected);
        CHECK(actual.high==(uint64_t)((unsigned __int128)expected>>64));
        CHECK(dos_wide_divide(actual,divisor)==(int64_t)(expected/divisor));
    }
#endif
}

int main(void)
{
    check_wide();
    seed=0xd4c1a372;
    unsigned samples=0,integer_boundaries=0;
    for(unsigned i=0;i<1000;i++) {
        int sx[3],sy[3];int32_t values[3];
        for(int j=0;j<3;j++) {
            sx[j]=(int)(next()%257)-112;sy[j]=(int)(next()%257)-112;
            values[j]=(int32_t)(next()%16777217)-8388608;
        }
        DosTriangle triangle;
        double area=((double)sx[1]-sx[0])*((double)sy[2]-sy[0])-
                    ((double)sx[2]-sx[0])*((double)sy[1]-sy[0]);
        int initialized=dos_triangle_init(&triangle,sx,sy);
        CHECK(initialized==(area!=0));if(!initialized)continue;
        for(int y=0;y<32;y++)for(int x=0;x<32;x++) {
            double px=x+0.5,py=y+0.5;
            double w0=((sx[1]-px)*(sy[2]-py)-(sx[2]-px)*(sy[1]-py))/area;
            double w1=((sx[2]-px)*(sy[0]-py)-(sx[0]-px)*(sy[2]-py))/area;
            double w2=1.0-w0-w1;
            int64_t weights[3];int inside=dos_triangle_weights(&triangle,x,y,weights);
            CHECK(inside==(w0>=-1e-6 && w1>=-1e-6 && w2>=-1e-6));
            if(!inside)continue;
            int64_t numerator=weights[0]*values[0]+weights[1]*values[1]+weights[2]*values[2];
            int32_t actual=dos_triangle_attribute(&triangle,weights,values,0);
            CHECK(actual==numerator/triangle.denominator);
            int32_t former=(int32_t)(w0*values[0]+w1*values[1]+w2*values[2]);
            if(actual!=former) {
                CHECK(abs(actual-former)<=1);
                CHECK(numerator%triangle.denominator==0);
                integer_boundaries++;
            }
            CHECK(dos_triangle_attribute(&triangle,weights,values,1)==
                  (numerator+triangle.denominator/2)/triangle.denominator);
            samples++;
        }
    }
    printf("DOS triangles: wide products/divisions and %u covered samples passed (%u exact integer boundaries)\n",
           samples,integer_boundaries);
    return 0;
}
