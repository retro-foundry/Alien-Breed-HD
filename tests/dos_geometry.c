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

static void clip_word(uint8_t *p,int value)
{ p[0]=(uint8_t)((unsigned)value>>8);p[1]=(uint8_t)value; }

static void check_clip_columns(void)
{
    enum { CELLS=7,ROOM_BYTES=64 };
    uint8_t data[CELLS*ROOM_BYTES]={0},adds[CELLS*4]={0},lines[(CELLS*4+1)*16]={0};
    LevelState level={0};level.data=data;level.data_byte_count=sizeof(data);
    level.zone_adds=adds;level.floor_lines=lines;level.num_floor_lines=CELLS*4+1;
    level.num_zones=level.num_zone_slots=CELLS;
    for(int zone=0;zone<CELLS;zone++) {
        unsigned offset=zone*ROOM_BYTES;
        adds[zone*4+2]=(uint8_t)(offset>>8);adds[zone*4+3]=(uint8_t)offset;
        uint8_t *room=data+offset;clip_word(room,zone);clip_word(room+32,40);
        const int corners[4][2]={{zone*16,-16},{(zone+1)*16,-16},{(zone+1)*16,16},{zone*16,16}};
        for(int edge=0;edge<4;edge++) {
            int index=zone*4+edge;uint8_t *line=lines+index*16;
            clip_word(line,corners[edge][0]);clip_word(line+2,corners[edge][1]);
            clip_word(line+4,corners[(edge+1)%4][0]-corners[edge][0]);
            clip_word(line+6,corners[(edge+1)%4][1]-corners[edge][1]);
            clip_word(room+40+edge*2,index);
        }
        clip_word(room+48,-1);clip_word(room+50,CELLS*4);clip_word(room+52,-2);
    }
    /* A neighbor-collision line behind -1 must not enter the primary outline. */
    clip_word(lines+CELLS*4*16,30);clip_word(lines+CELLS*4*16+2,-16);
    clip_word(lines+CELLS*4*16+6,32);
    assert(renderer_zone_contains_point(&level,0,8,0));
    assert(!renderer_zone_contains_point(&level,0,20,0));
    const int widths[]={4,11,127,224,320,768};
    unsigned columns=0;
    for(unsigned w=0;w<sizeof(widths)/sizeof(*widths);w++) {
        int width=widths[w],owners[768]={0};
        for(int zone=0;zone<CELLS;zone++) {
            int starts[2],ends[2];
            int count=renderer_zone_clip_segment_to_dx_ranges(&level,(int16_t)zone,
                0,0,112,0,width,0,width,starts,ends,NULL,NULL,2);
            for(int i=0;i<count;i++)for(int x=starts[i];x<ends[i];x++) {
                assert(((2*x+1)*112)/(2*width)/16==zone);owners[x]++;columns++;
            }
        }
        for(int x=0;x<width;x++)assert(owners[x]==1);
    }
    uint32_t seed=0x76c3a1ef;unsigned intervals=0;
    for(unsigned n=0;n<3000;n++) {
        int point[4];
        for(int i=0;i<4;i++) { seed=seed*1664525u+1013904223u;point[i]=(int)(seed%513)-256; }
        for(unsigned w=0;w<sizeof(widths)/sizeof(*widths);w++)for(int zone=0;zone<CELLS;zone++) {
            int width=widths[w],clip_start=width/7,clip_end=width-width/11;
            /* Independent slab intersection with this rectangular cell. */
            double first=0,last=1;int valid=1;
            for(int axis=0;axis<2;axis++) {
                int lower=axis?-16:zone*16,upper=axis?16:(zone+1)*16;
                int delta=point[axis+2]-point[axis];
                if(!delta) { if(point[axis]<lower || point[axis]>upper)valid=0; }
                else {
                    double a=(double)(lower-point[axis])/delta,b=(double)(upper-point[axis])/delta;
                    if(a>b) { double swap=a;a=b;b=swap; }
                    if(a>first)first=a;if(b<last)last=b;
                }
            }
            int expected_start=(int)ceil(first*width-0.5),expected_end=(int)ceil(last*width-0.5);
            if(expected_start<clip_start)expected_start=clip_start;
            if(expected_end>clip_end)expected_end=clip_end;
            int expected=valid && last-first>1e-9 && expected_end>expected_start;
            int starts[2],ends[2];int count=renderer_zone_clip_segment_to_dx_ranges(&level,(int16_t)zone,
                point[0],point[1],point[2],point[3],width,clip_start,clip_end,starts,ends,NULL,NULL,2);
            assert(count==expected);
            if(count) { assert(starts[0]==expected_start);assert(ends[0]==expected_end); }
            intervals++;
        }
    }
    DosGeometryFraction a={0,1},b={1,1000000000},c={1,999999999};
    assert(dos_geometry_fraction_near(a,b));assert(!dos_geometry_fraction_near(a,c));
    DosGeometryFraction wide={(UINT64_C(1)<<47)-1,(UINT64_C(1)<<48)-1},half={1,2};
    assert(dos_geometry_fraction_compare(wide,half)<0);assert(dos_geometry_fraction_near(wide,half));
    printf("PASS: %u pixel-centre owners and %u independent slab intersections retain sprite clipping\n",columns,intervals);
}

int main(void)
{
    setvbuf(stdout,NULL,_IOLBF,0);
    check_clip_columns();
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
    static const int32_t coordinates[]={-0x40000000,-32768,0,32767,0x3fffffff};
    for(unsigned a=0;a<5;a++)for(unsigned b=0;b<5;b++)for(unsigned c=0;c<5;c++)
    for(unsigned d=0;d<5;d++)for(unsigned e=0;e<5;e++)for(unsigned f=0;f<5;f++)
    for(unsigned axes=0;axes<4;axes++) {
        /* Every horizontal/vertical pairing, reversed endpoints and points,
         * with the same independent wide reference at large coordinates. */
        p[0]=axes&1?coordinates[c]:coordinates[a];
        p[1]=axes&1?coordinates[a]:coordinates[c];
        p[2]=axes&1?coordinates[c]:coordinates[b];
        p[3]=axes&1?coordinates[b]:coordinates[c];
        p[4]=axes&2?coordinates[f]:coordinates[d];
        p[5]=axes&2?coordinates[d]:coordinates[f];
        p[6]=axes&2?coordinates[f]:coordinates[e];
        p[7]=axes&2?coordinates[e]:coordinates[f];
        check_case(p);
    }
    static const int32_t deltas[]={-32768,-32767,0,1,32767};
    static const int32_t offsets[]={-32768,-1,0,1,32767};
    static const int32_t radii[]={INT32_MIN,-1,0,1,32767,INT32_MAX};
    unsigned radius_checked=0;
    for(unsigned x=0;x<5;x++)for(unsigned z=0;z<5;z++)
    for(unsigned ox=0;ox<5;ox++)for(unsigned oz=0;oz<5;oz++)
    for(unsigned r=0;r<6;r++) {
        /* (-32768,-32768) has squared length 2^31. Include its neighbours,
         * zero length and radii whose inflated bounds exceed signed 32 bits. */
        int32_t x1=-32768,z1=32767,x2=x1+deltas[x],z2=z1+deltas[z];
        int32_t px=x1+deltas[x]/2+offsets[ox],pz=z1+deltas[z]/2+offsets[oz];
        assert(renderer_point_within_segment_radius_i32(px,pz,x1,z1,x2,z2,radii[r])==
               reference_radius(px,pz,x1,z1,x2,z2,radii[r]));
        radius_checked++;
    }
    printf("PASS: %u geometry cases retain intersections, endpoint order, collinearity, point ownership and capsule admission\n",checked);
    printf("PASS: %u capsule cases retain squared-length boundaries and signed radius extremes\n",radius_checked);
    static const int16_t adjacent_values[] = {
        INT16_MIN,-257,-256,-1,0,1,31,32,63,64,95,96,127,128,159,160,
        191,192,223,224,254,255,256,257,300,511,512,1024,32766,INT16_MAX,0,255
    };
    static const unsigned adjacent_counts[] = {0,1,2,7,16,32};
    unsigned adjacent_checked = 0;
    for (unsigned pattern = 0; pattern < 3; pattern++) {
        int16_t zones[32];
        for (unsigned i = 0; i < 32; i++)
            zones[i] = adjacent_values[pattern == 0 ? i : pattern == 1 ? 31-i : i%4];
        for (unsigned c = 0; c < sizeof(adjacent_counts)/sizeof(*adjacent_counts); c++) {
            unsigned count = adjacent_counts[c];
            uint32_t mask[8] = {0};
            for (unsigned i = 0; i < count; i++) {
                unsigned zone = (uint16_t)zones[i];
                if (zone < 256) mask[zone>>5] |= 1u << (zone&31);
            }
            for (int zone = INT16_MIN; zone <= INT16_MAX; zone++) {
                int expected = 0;
                for (unsigned i = 0; i < count; i++)
                    if (zones[i] == zone) expected = 1;
                assert(renderer_dos_adjacent_contains(mask,zones,count,(int16_t)zone) == expected);
                adjacent_checked++;
            }
        }
    }
    printf("PASS: %u adjacent-room queries retain all signed zone IDs, empty lists and duplicates\n",adjacent_checked);
    puts("DOS geometry regression passed");
    return 0;
}
