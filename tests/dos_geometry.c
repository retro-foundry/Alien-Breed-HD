/* Compare the actual renderer helpers with their original wide-arithmetic
 * definitions, including collinear/touching edges and degenerate segments.
 * Whole-program linking discards the unused renderer and its asset dependencies. */
#include "../src/renderer.c"
#include <assert.h>
#include <stdio.h>

static int64_t reference_orientation(int32_t ax,int32_t az,int32_t bx,int32_t bz,
                                      int32_t cx,int32_t cz)
{
    return ((int64_t)bx-ax)*((int64_t)cz-az)-((int64_t)bz-az)*((int64_t)cx-ax);
}

static int reference_bounds(int32_t ax,int32_t az,int32_t bx,int32_t bz,int32_t x,int32_t z)
{
    return x>=(ax<bx?ax:bx) && x<=(ax>bx?ax:bx) &&
           z>=(az<bz?az:bz) && z<=(az>bz?az:bz);
}

static int reference_intersection(const int32_t *p)
{
    int64_t a=reference_orientation(p[0],p[1],p[2],p[3],p[4],p[5]);
    int64_t b=reference_orientation(p[0],p[1],p[2],p[3],p[6],p[7]);
    int64_t c=reference_orientation(p[4],p[5],p[6],p[7],p[0],p[1]);
    int64_t d=reference_orientation(p[4],p[5],p[6],p[7],p[2],p[3]);
    if(((a>0 && b<0)||(a<0 && b>0)) && ((c>0 && d<0)||(c<0 && d>0)))return 1;
    return (!a && reference_bounds(p[0],p[1],p[2],p[3],p[4],p[5])) ||
           (!b && reference_bounds(p[0],p[1],p[2],p[3],p[6],p[7])) ||
           (!c && reference_bounds(p[4],p[5],p[6],p[7],p[0],p[1])) ||
           (!d && reference_bounds(p[4],p[5],p[6],p[7],p[2],p[3]));
}

/* Original capsule test, before the bounding-box rejection. Endpoint deltas
 * below are signed 16-bit floor-line lengths, so the interior square fits. */
static int reference_radius(int32_t px,int32_t pz,int32_t x1,int32_t z1,
                             int32_t x2,int32_t z2,int32_t radius)
{
    int64_t dx=(int64_t)x2-x1,dz=(int64_t)z2-z1;
    int64_t vx=(int64_t)px-x1,vz=(int64_t)pz-z1;
    int64_t length=dx*dx+dz*dz,distance;
    if(radius<0)radius=0;
    if(length<=0)distance=vx*vx+vz*vz;
    else {
        int64_t t=vx*dx+vz*dz;
        if(t<=0)distance=vx*vx+vz*vz;
        else if(t>=length) {
            int64_t wx=(int64_t)px-x2,wz=(int64_t)pz-z2;
            distance=wx*wx+wz*wz;
        } else {
            distance=vx*vx+vz*vz-t*t/length;
            if(distance<0)distance=0;
        }
    }
    return distance<=(int64_t)radius*radius;
}

static unsigned checked;
static void check_case(const int32_t *p)
{
    int expected=reference_intersection(p);
    assert(renderer_segments_intersect_i32(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7])==expected);
    assert(renderer_segments_intersect_i32(p[2],p[3],p[0],p[1],p[6],p[7],p[4],p[5])==expected);
    assert(renderer_segments_intersect_i32(p[4],p[5],p[6],p[7],p[0],p[1],p[2],p[3])==expected);
    int on=reference_orientation(p[4],p[5],p[6],p[7],p[0],p[1])==0 &&
           reference_bounds(p[4],p[5],p[6],p[7],p[0],p[1]);
    assert(renderer_point_on_segment_i32(p[0],p[1],p[4],p[5],p[6],p[7])==on);
    int32_t x1=(int16_t)p[2],z1=(int16_t)p[3];
    int32_t x2=x1+(int16_t)p[4],z2=z1+(int16_t)p[5];
    int32_t radius=checked%7==0?-1:(int32_t)((uint32_t)p[6]&32767);
    int32_t px=checked&1?(int16_t)p[0]:p[0],pz=checked&1?(int16_t)p[1]:p[1];
    assert(renderer_point_within_segment_radius_i32(px,pz,x1,z1,x2,z2,radius)==
           reference_radius(px,pz,x1,z1,x2,z2,radius));
    assert(renderer_point_within_segment_radius_i32(x1,z1,x1,z1,x2,z2,0));
    assert(renderer_point_within_segment_radius_i32(x2,z2,x1,z1,x2,z2,0));
    checked++;
}

int main(void)
{
    setvbuf(stdout,NULL,_IOLBF,0);
    int32_t p[8];
    for(unsigned n=0;n<6561;n++) {
        unsigned v=n;
        for(unsigned i=0;i<8;i++){p[i]=(int32_t)(v%3)-1;v/=3;}
        check_case(p);
    }
    uint32_t seed=0x21875133;
    for(unsigned n=0;n<32768;n++) {
        for(unsigned i=0;i<8;i++) {
            seed=seed*1664525u+1013904223u;
            /* Differences and both orientation products stay within int64. */
            p[i]=(int32_t)(seed&0x7fffffffu)-0x40000000;
        }
        check_case(p);
    }
    printf("PASS: %u geometry cases retain intersections, endpoint order, collinearity, point ownership and capsule admission\n",checked);
    puts("DOS geometry regression passed");
    return 0;
}
