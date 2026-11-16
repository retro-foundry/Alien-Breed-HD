/* Compare the DOS integer/offline maths with the unchanged PC formulae. */
#include "math_tables.h"
#include "math_tables_dos.h"
#include "audio.h"
#include "renderer_fixed.h"
#include "platform/dos/float_dos.h"
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
static unsigned angle_cases,distance_cases;
static unsigned progress_cases,float_cases;
static unsigned sky_cases;

static void check_progress(uint32_t position,uint32_t length)
{
    volatile float phase=length?(float)position/(float)length:1.0f;
    if(phase>1.0f)phase=1.0f;
    AudioProgress actual=audio_progress_ratio(position,length);
    uint32_t expected=phase>=1.0f?UINT32_MAX:(uint32_t)((double)phase*4294967296.0);
    if(actual!=expected)printf("progress %lu/%lu expected=%lu actual=%lu\n",(unsigned long)position,(unsigned long)length,(unsigned long)expected,(unsigned long)actual);
    assert(actual==expected);
    assert(audio_progress_alpha(actual,220)==(int)(phase*220.0f+0.5f));progress_cases++;
}

static void check_direction(int32_t dx,int32_t dz)
{
    double radians=atan2(-(double)dx,-(double)dz);
    int16_t raw=(int16_t)(radians*(4096.0/(2.0*3.14159265)));
    int16_t expected=(int16_t)((raw*2)&ANGLE_MASK);
    int16_t actual=math_angle_to_target(dx,dz);
    if(actual!=expected)printf("angle %ld,%ld expected=%d actual=%d\n",(long)dx,(long)dz,expected,actual);
    assert(actual==expected);angle_cases++;
    double distance=sqrt((double)dx*dx+(double)dz*dz);
    int32_t length=math_distance_floor(dx,dz);
    assert(length==(int32_t)distance);distance_cases++;
}

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
    for(int width=96;width<=640;width++)for(int scale=1;scale<=8;scale++)
        for(int centered=0;centered<2;centered++) {
            int left=centered?width/2:width*47/96,right=width-1-left;
            double fov=atan((double)left/(64*scale))+atan((double)right/(64*scale));
            int expected=(int)(432.0*fov/(2.0*3.14159265358979323846)+0.5);
            if(expected<1)expected=1;
            assert(math_sky_view_columns(left,right,64*scale,432)==expected);sky_cases++;
        }
    const uint32_t lengths[]={0,1,20,44100,1000000,16777217,UINT32_MAX};
    for(unsigned l=0;l<sizeof(lengths)/sizeof(*lengths);l++) {
        uint32_t length=lengths[l];
        for(unsigned alpha=0;alpha<220;alpha++) {
            uint32_t middle=(uint32_t)((uint64_t)length*(alpha*2+1)/440);
            for(int delta=-2;delta<=2;delta++)check_progress(middle+delta,length);
        }
        check_progress(0,length);check_progress(length,length);
    }
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
    for(unsigned x=0;x<sizeof(limits)/sizeof(*limits);x++)
        for(unsigned z=0;z<sizeof(limits)/sizeof(*limits);z++)check_direction(limits[x],limits[z]);
    for(int x=-128;x<=128;x++)for(int z=-128;z<=128;z++)check_direction(x,z);
    for(unsigned i=0;i<100000;i++) {
        seed=seed*1664525u+1013904223u;int32_t x=(int)(seed%131071)-65535;
        seed=seed*1664525u+1013904223u;int32_t z=(int)(seed%131071)-65535;
        check_direction(x,z);
        check_progress(seed,seed^UINT32_C(0x9e3781a2));
        uint32_t bits=seed;volatile float input;memcpy((void *)&input,&bits,sizeof(bits));
        int32_t expected=(int32_t)input;
        assert(dos_float_to_i32((const float *)&input)==expected);float_cases++;
    }
    /* Challenge each truncation threshold from both sides at real word
     * coordinate scales, rather than comparing only arbitrary angles. */
    const int radii[]={17,257,32767,65535};
    for(int angle=-2048;angle<=2048;angle++)for(unsigned r=0;r<4;r++) {
        double radians=angle*(2.0*3.14159265)/4096.0;
        int32_t x=(int32_t)(-sin(radians)*radii[r]);
        int32_t z=(int32_t)(-cos(radians)*radii[r]);
        for(int delta=-1;delta<=1;delta++){check_direction(x+delta,z);check_direction(x,z+delta);}
    }
    printf("DOS integer directions: %u angle and %u distance cases passed\n",angle_cases,distance_cases);
    printf("DOS integer transport: %u fade phases and %u binary32 input cases passed\n",progress_cases,float_cases);
    printf("DOS integer sky: %u projection/FOV cases passed\n",sky_cases);
    printf("DOS maths: 4096 sine values, 8192 lookups, filter coefficients, %u distance gains passed (%u exact integer boundaries)\n",
           checked,exact_boundary_differences);
    return 0;
}
