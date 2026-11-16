/* Compare the DOS integer/offline maths with the unchanged PC formulae. */
#include "math_tables.h"
#include "math_tables_dos.h"
#include "audio.h"
#include "renderer_fixed.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/* DOS shells do not all support stderr redirection. Keep failed oracle
 * assertions in the same redirected log as successful checks. */
#undef assert
#define assert(condition) do { if(!(condition)) { \
    printf("DOS maths assertion failed at line %d: %s\n",__LINE__,#condition); \
    fflush(stdout);exit(1); } } while(0)

static unsigned checked, exact_boundary_differences;

static void check_volume(int volume, int dx, int dz)
{
    uint64_t squared=(uint64_t)((int64_t)dx*dx)+(uint64_t)((int64_t)dz*dz);
    int expected;
    int64_t numerator=(int64_t)volume*64;
    if(numerator>32767)numerator=32767;
    if(volume<=0)expected=0;
    else if(!squared)expected=255;
    else {
        double gain=(double)numerator/(sqrt((double)squared)/4.0+1.0);
        expected=gain>=64.0?255:(int)(gain*255.0/64.0);
    }
    int actual=audio_distance_volume(volume,dx,dz);
    if(actual!=expected) {
        /* Old floating-point evaluation can round an exact integer gain
         * down by one. Accept only that mathematically exact boundary. */
        assert(actual==expected+1 && volume>0);
        uint64_t remaining=(uint64_t)numerator*255-64u*actual;
        assert(squared*(256u*actual*actual)==remaining*remaining);
        exact_boundary_differences++;
    }
    checked++;
}

int main(void)
{
    math_tables_init();
    for(unsigned ms=0;ms<20;ms++) {
        volatile float alpha=(float)ms/20.0f; /* GameState stores binary32. */
        assert(interpolation_phase[ms]==(uint32_t)((double)alpha*4294967296.0));
        for(int from=-40000;from<=40000;from+=101)
            for(int delta=-10000;delta<=10000;delta+=127) {
                int32_t expected=(int32_t)((double)from+(double)delta*(double)alpha);
                assert(renderer_interp_fixed_i32(from,from+delta,interpolation_phase[ms])==expected);
                assert(renderer_interp_fixed_delta_i32(delta,interpolation_phase[ms])==
                       (int32_t)((double)delta*(double)alpha));
            }
    }
    for(int i=0;i<ANGLE_TABLE_SIZE;i++) {
        double value=sin((double)i*2.0*3.14159265358979323846/ANGLE_TABLE_SIZE)*16384.0;
        int16_t expected=(int16_t)(value+(value>=0?0.5:-0.5));
        assert(sine_table[i]==expected);
        assert(sine_table_fine[2*i]==expected);
        int32_t sum=(int32_t)sine_table[i]+sine_table[(i+1)&(ANGLE_TABLE_SIZE-1)];
        assert(sine_table_fine[2*i+1]==(int16_t)((sum>=0?sum+1:sum-1)>>1));
    }
    double k=tan(3.141592653589793*2200.0/8007);
    double norm=1.0/(1.0+1.41421356237*k+k*k);
    assert(AB3D_DOS_FILTER_B0==(int32_t)(k*k*norm*16384));
    assert(AB3D_DOS_FILTER_A1==(int32_t)(2*(k*k-1)*norm*16384));
    assert(AB3D_DOS_FILTER_A2==(int32_t)((1-1.41421356237*k+k*k)*norm*16384));
    for(int x=-128;x<=128;x++)for(int z=-128;z<=128;z++)
        for(int v=0;v<=512;v+=16)check_volume(v,x,z);
    uint32_t seed=0x34d6c721;
    for(unsigned i=0;i<500000;i++) {
        seed=seed*1664525u+1013904223u;int x=(int)(seed>>16)-32768;
        seed=seed*1664525u+1013904223u;int z=(int)(seed>>16)-32768;
        seed=seed*1664525u+1013904223u;check_volume((int)(seed%1025),x,z);
    }
    const int limits[]={INT_MIN,INT_MIN+1,-65535,-1,0,1,65535,INT_MAX};
    for(unsigned x=0;x<sizeof(limits)/sizeof(*limits);x++)
        for(unsigned z=0;z<sizeof(limits)/sizeof(*limits);z++)
            check_volume(INT_MAX,limits[x],limits[z]);
    printf("DOS maths: 4096 sine values, 8192 lookups, filter coefficients, %u distance gains passed (%u exact integer boundaries)\n",
           checked,exact_boundary_differences);
    return 0;
}
