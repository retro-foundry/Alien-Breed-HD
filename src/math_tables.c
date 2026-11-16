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

#ifdef AB3D_DOS
static int64_t dos_atan_positive(int64_t x,int64_t y,const int64_t steps[60])
{
    if(!x && !y)return 0;
    int64_t angle=0;
    uint64_t largest=(uint64_t)(x>y?x:y);
    unsigned shift=0;
    while(largest<(UINT64_C(1)<<59)){largest<<=1;shift++;}
    x<<=shift;y<<=shift;
    for(unsigned i=0;i<60 && y;i++) {
        int64_t previous_x=x;
        if(y>0){x+=y>>i;y-=previous_x>>i;angle+=steps[i];}
        else {x-=y>>i;y+=previous_x>>i;angle-=steps[i];}
    }
    return angle;
}
#endif

int16_t math_angle_to_target(int32_t dx, int32_t dz)
{
#ifdef AB3D_DOS
    /* Preserve the existing atan2(-dx,-dz) * (4096/(2*3.14159265))
     * truncation, including its slightly short PI constant. Replacing it
     * with the rounded sine table moves exact direction boundaries. */
    static const int64_t steps[60] = AB3D_DOS_ATAN_VALUES;
    int64_t x=-(int64_t)dz,y=-(int64_t)dx;
    int negative=y<0,back=x<0;
    if(!x && !y)return 0; /* DJGPP atan2 returns zero for the coincident case. */
    int64_t angle=dos_atan_positive(x<0?-x:x,y<0?-y:y,steps);
    if(back)angle=AB3D_DOS_PI_ANGLE-angle;
    int raw=(int)(angle>>50);
    return (int16_t)(((negative?-raw:raw)*2)&ANGLE_MASK);
#else
    double angle=atan2(-(double)dx,-(double)dz);
    int16_t raw=(int16_t)(angle*(4096.0/(2.0*3.14159265)));
    return (int16_t)((raw*2)&ANGLE_MASK);
#endif
}

int math_sky_view_columns(int left,int right,int focal,int pan_width)
{
#ifdef AB3D_DOS
    /* FOV uses true PI, unlike the historical enemy angle expression. */
    static const int64_t steps[60]=AB3D_DOS_TURN_ATAN_VALUES;
    int64_t phase=dos_atan_positive(focal,left,steps)+dos_atan_positive(focal,right,steps);
    int columns=(int)((phase*pan_width+(INT64_C(1)<<49))>>50);
#else
    double fov=atan((double)left/focal)+atan((double)right/focal);
    int columns=(int)(((double)pan_width*fov)/(2.0*3.14159265358979323846)+0.5);
#endif
    if(columns<1)columns=1;
    if(columns>pan_width)columns=pan_width;
    return columns;
}

int32_t math_distance_floor(int32_t dx,int32_t dz)
{
#ifdef AB3D_DOS
    uint64_t value=(uint64_t)((int64_t)dx*dx)+(uint64_t)((int64_t)dz*dz);
    uint64_t root=0,bit=UINT64_C(1)<<62;
    while(bit>value)bit>>=2;
    while(bit) {
        if(value>=root+bit){value-=root+bit;root=(root>>1)+bit;}
        else root>>=1;
        bit>>=2;
    }
    return root>INT32_MAX?INT32_MIN:(int32_t)root;
#else
    return (int32_t)sqrt((double)dx*dx+(double)dz*dz);
#endif
}

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
