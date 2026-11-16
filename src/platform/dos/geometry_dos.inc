/* Integer form of renderer_zone_clip_segment_to_dx_ranges. The zone outline
 * still ends at ObjectMove.s checkwalls' first negative word. Preserve the
 * PC port's fractional segment and pixel-centre ownership without rounding
 * intermediate world points onto an edge. Included privately by renderer.c. */
#include "wide_dos.h"

typedef struct { uint64_t numerator,denominator; } DosGeometryFraction;

static int dos_geometry_fraction_compare(DosGeometryFraction a,DosGeometryFraction b)
{
    return dos_wide_compare(dos_wide_unsigned_product(a.numerator,b.denominator),
                            dos_wide_unsigned_product(b.numerator,a.denominator));
}

static int dos_geometry_fraction_near(DosGeometryFraction a,DosGeometryFraction b)
{
    DosWideInteger an=dos_wide_unsigned_product(a.numerator,b.denominator);
    DosWideInteger bn=dos_wide_unsigned_product(b.numerator,a.denominator);
    DosWideInteger difference=dos_wide_compare(an,bn)<0?dos_wide_subtract(bn,an):dos_wide_subtract(an,bn);
    /* Same 1e-9 parameter tolerance as the PC implementation. A signed
     * 32-bit segment crossed with a 16-bit floor edge has <48-bit products;
     * this comparison, including the billion multiplier, fits in 128 bits. */
    return dos_wide_compare(dos_wide_multiply_u32(difference,1000000000),
                            dos_wide_unsigned_product(a.denominator,b.denominator))<=0;
}

static void dos_geometry_add_fraction(DosGeometryFraction *values,int *count,int64_t numerator,int64_t denominator)
{
    if(!denominator)return;
    if(denominator<0) { denominator=-denominator;numerator=-numerator; }
    if(numerator<0)numerator=0;
    if(numerator>denominator)numerator=denominator;
    DosGeometryFraction value={(uint64_t)numerator,(uint64_t)denominator};
    for(int i=0;i<*count;i++)if(dos_geometry_fraction_near(values[i],value))return;
    if(*count<RENDERER_GEOM_CLIP_MAX_T_VALUES)values[(*count)++]=value;
}

static int dos_geometry_sign(int64_t value,int64_t derivative)
{
    if(value)return value>0?1:-1;
    return derivative==0?0:derivative>0?1:-1;
}

/* Test p(t+0), choosing the interior of the interval when a pixel centre
 * coincides exactly with its left endpoint. Nonzero integer numerators are
 * at least 1/(2*width) from an edge, beyond the old 1e-7 edge tolerance. */
static int dos_geometry_contains(const LevelState *level,int16_t zone,
                                  int32_t x0,int32_t z0,int64_t dx,int64_t dz,
                                  int numerator,int denominator)
{
    if(!level || !level->data || !level->zone_adds || !level->floor_lines)return 0;
    int slots=level_zone_slot_count(level);
    if(zone<0 || zone>=slots)return 0;
    const uint8_t *list=renderer_zone_exit_list_ptr(level,zone);
    if(!list)return 0;
    int inside=0,edges=0;
    int64_t px=(int64_t)x0*denominator+dx*numerator;
    int64_t pz=(int64_t)z0*denominator+dz*numerator;
    for(int i=0;i<256;i++) {
        if(level->data_byte_count && (size_t)(list-level->data)+(size_t)i*2+2>level->data_byte_count)break;
        int16_t entry=rd16(list+(size_t)i*2);
        if(entry<0)break;
        if(entry>=level->num_floor_lines)continue;
        const uint8_t *line=level->floor_lines+(size_t)(uint16_t)entry*RENDERER_FLINE_SIZE;
        int64_t x1=rd16(line+RENDERER_FLINE_X_OFF),z1=rd16(line+RENDERER_FLINE_Z_OFF);
        int64_t ex=rd16(line+RENDERER_FLINE_XLEN_OFF),ez=rd16(line+RENDERER_FLINE_ZLEN_OFF);
        int64_t x2=x1+ex,z2=z1+ez;
        int64_t cross=(px-x1*denominator)*ez-(pz-z1*denominator)*ex;
        int64_t derivative=dx*ez-dz*ex;
        edges++;
        if(!cross && !derivative &&
           dos_geometry_sign(px-(x1<x2?x1:x2)*denominator,dx)>=0 &&
           dos_geometry_sign(px-(x1>x2?x1:x2)*denominator,dx)<=0 &&
           dos_geometry_sign(pz-(z1<z2?z1:z2)*denominator,dz)>=0 &&
           dos_geometry_sign(pz-(z1>z2?z1:z2)*denominator,dz)<=0)return 1;
        int above1=dos_geometry_sign(z1*denominator-pz,-dz)>0;
        int above2=dos_geometry_sign(z2*denominator-pz,-dz)>0;
        if(above1!=above2) {
            int side=dos_geometry_sign(cross,derivative);
            if((ez>0 && side<0)||(ez<0 && side>0))inside^=1;
        }
    }
    return edges?inside:0;
}

static int renderer_zone_contains_point(const LevelState *level,int16_t zone,int32_t x,int32_t z)
{
    return dos_geometry_contains(level,zone,x,z,0,0,0,1);
}

static int dos_geometry_column(DosGeometryFraction value,int width)
{
    int64_t numerator=(int64_t)(value.numerator*(uint64_t)width*2)-value.denominator;
    int64_t denominator=(int64_t)value.denominator*2;
    return (int)(numerator>=0?(numerator+denominator-1)/denominator:numerator/denominator);
}

static int renderer_zone_clip_segment_to_dx_ranges(const LevelState *level,int16_t zone,
    int32_t x0,int32_t z0,int32_t x1,int32_t z1,int width,int clip_start,int clip_end,
    int *starts,int *ends,double *t_starts,double *t_ends,int capacity)
{
    if(!level || !level->data || !level->zone_adds || !level->floor_lines)return 0;
    if(!starts || !ends || capacity<=0 || width<=0 || clip_end<=clip_start)return 0;
    const uint8_t *list=renderer_zone_exit_list_ptr(level,zone);
    if(!list)return 0;
    DosGeometryFraction values[RENDERER_GEOM_CLIP_MAX_T_VALUES];
    int count=0,output=0;
    int64_t dx=(int64_t)x1-x0,dz=(int64_t)z1-z0;
    dos_geometry_add_fraction(values,&count,0,1);dos_geometry_add_fraction(values,&count,1,1);
    for(int i=0;i<256;i++) {
        if(level->data_byte_count && (size_t)(list-level->data)+(size_t)i*2+2>level->data_byte_count)break;
        int16_t entry=rd16(list+(size_t)i*2);
        if(entry<0)break;
        if(entry>=level->num_floor_lines)continue;
        const uint8_t *line=level->floor_lines+(size_t)(uint16_t)entry*RENDERER_FLINE_SIZE;
        int32_t ex0=rd16(line+RENDERER_FLINE_X_OFF),ez0=rd16(line+RENDERER_FLINE_Z_OFF);
        int32_t ex=rd16(line+RENDERER_FLINE_XLEN_OFF),ez=rd16(line+RENDERER_FLINE_ZLEN_OFF);
        int32_t ex1=ex0+ex,ez1=ez0+ez;
        if(!renderer_segments_intersect_i32(x0,z0,x1,z1,ex0,ez0,ex1,ez1))continue;
        int64_t denominator=dx*ez-dz*ex;
        if(denominator) {
            int64_t numerator=((int64_t)ex0-x0)*ez-((int64_t)ez0-z0)*ex;
            dos_geometry_add_fraction(values,&count,numerator,denominator);
        } else if(!renderer_orient2d_i64(x0,z0,x1,z1,ex0,ez0) &&
                  !renderer_orient2d_i64(x0,z0,x1,z1,ex1,ez1)) {
            if(llabs(dx)>=llabs(dz) && dx) {
                dos_geometry_add_fraction(values,&count,(int64_t)ex0-x0,dx);
                dos_geometry_add_fraction(values,&count,(int64_t)ex1-x0,dx);
            } else if(dz) {
                dos_geometry_add_fraction(values,&count,(int64_t)ez0-z0,dz);
                dos_geometry_add_fraction(values,&count,(int64_t)ez1-z0,dz);
            }
        }
    }
    for(int i=1;i<count;i++) {
        DosGeometryFraction value=values[i];int j=i-1;
        while(j>=0 && dos_geometry_fraction_compare(values[j],value)>0) {
            values[j+1]=values[j];j--;
        }
        values[j+1]=value;
    }
    for(int i=0;i+1<count;i++) {
        DosGeometryFraction a=values[i],b=values[i+1];
        if(dos_geometry_fraction_near(a,b))continue;
        int start=dos_geometry_column(a,width),end=dos_geometry_column(b,width);
        if(start<0)start=0;if(end>width)end=width;
        if(start<clip_start)start=clip_start;if(end>clip_end)end=clip_end;
        if(end<=start)continue;
        if(!dos_geometry_contains(level,zone,x0,z0,dx,dz,2*start+1,2*width))continue;
        if(output && start<=ends[output-1]) {
            if(end>ends[output-1])ends[output-1]=end;
            if(t_ends)t_ends[output-1]=(double)b.numerator/(double)b.denominator;
            continue;
        }
        if(output>=capacity)break;
        starts[output]=start;ends[output]=end;
        /* Optional diagnostic outputs retain the existing double contract.
         * Production calls supply NULL; no floating-point operations occur. */
        if(t_starts)t_starts[output]=(double)a.numerator/(double)a.denominator;
        if(t_ends)t_ends[output]=(double)b.numerator/(double)b.denominator;
        output++;
    }
    return output;
}
