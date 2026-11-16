#ifndef AB3D_TRIANGLE_DOS_H
#define AB3D_TRIANGLE_DOS_H
#include "wide_dos.h"

/* The PC polygon sampler evaluates barycentric weights at pixel centres.
 * Doubling the edge equations represents those half pixels exactly. Keep
 * wide attribute products for near-plane polygons extending off screen. */
typedef struct {
    int64_t denominator,minimum;
    int64_t dx0,dy0,dx1,dy1;
    int64_t x1,y1,x2,y2;
    int direction;
} DosTriangle;

static inline int dos_triangle_init(DosTriangle *triangle,const int *x,const int *y)
{
    int64_t area=((int64_t)x[1]-x[0])*((int64_t)y[2]-y[0])-
                 ((int64_t)x[2]-x[0])*((int64_t)y[1]-y[0]);
    if(!area)return 0;
    triangle->direction=area<0?-1:1;
    triangle->denominator=(area<0?-area:area)*2;
    triangle->minimum=-(triangle->denominator/1000000);
    triangle->dx0=(int64_t)x[2]-x[1];triangle->dy0=(int64_t)y[2]-y[1];
    triangle->dx1=(int64_t)x[0]-x[2];triangle->dy1=(int64_t)y[0]-y[2];
    triangle->x1=x[1];triangle->y1=y[1];triangle->x2=x[2];triangle->y2=y[2];
    return 1;
}

static inline int dos_triangle_weights(const DosTriangle *triangle,int x,int y,int64_t *weights)
{
    int64_t px=(int64_t)x*2+1,py=(int64_t)y*2+1;
    weights[0]=((triangle->x1*2-px)*triangle->dy0-
                (triangle->y1*2-py)*triangle->dx0)*triangle->direction;
    weights[1]=((triangle->x2*2-px)*triangle->dy1-
                (triangle->y2*2-py)*triangle->dx1)*triangle->direction;
    weights[2]=triangle->denominator-weights[0]-weights[1];
    return weights[0]>=triangle->minimum && weights[1]>=triangle->minimum && weights[2]>=triangle->minimum;
}

static inline int32_t dos_triangle_attribute(const DosTriangle *triangle,const int64_t *weights,
                                             const int32_t *values,int round_half_up)
{
    DosWideInteger value=dos_wide_add(dos_wide_product(weights[0],values[0]),
                                    dos_wide_product(weights[1],values[1]));
    value=dos_wide_add(value,dos_wide_product(weights[2],values[2]));
    if(round_half_up) {
        DosWideInteger half={(uint64_t)triangle->denominator/2,0};
        value=dos_wide_add(value,half);
    }
    return (int32_t)dos_wide_divide(value,(uint64_t)triangle->denominator);
}
#endif
