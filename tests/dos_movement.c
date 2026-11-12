/* Execute production movement math against its original division oracles. */
#include "../src/movement.c"
#include <assert.h>

static uint32_t touch_seed=0x82a7c4e1;
static uint32_t touch_random(void)
{
    touch_seed=touch_seed*1664525u+1013904223u;
    return touch_seed;
}

static void check_touch(int32_t lx,int32_t lz,int32_t x,int32_t z,
                         int16_t vx,int16_t vz,int16_t length,int16_t extent,
                         int shift,uint16_t flags)
{
    uint8_t line[16],expected[16];
    for(unsigned i=0;i<sizeof(line);i++)line[i]=(uint8_t)(touch_random()>>24);
    write_be16(line+FLINE_XLEN,(uint16_t)vx);write_be16(line+FLINE_ZLEN,(uint16_t)vz);
    write_be16(line+FLINE_LENGTH,(uint16_t)length);
    memcpy(expected,line,sizeof(line));
    MoveContext ctx={0};ctx.newx=x;ctx.newz=z;ctx.extlen=extent;
    ctx.pos_shift=(int8_t)shift;ctx.wall_flags=flags;
    int64_t side=((int64_t)x-lx)*vz-((int64_t)z-lz)*vx;
    if(flags && side>0) {
        int divisor=length<0?-(int)length:length;divisor+=extent;
        if(divisor<=0)divisor=1;
        /* Volatile retains the original wide division in the independent
         * oracle rather than letting the optimizer rewrite both predicates. */
        volatile int64_t wide_divisor=divisor;
        int64_t distance=side/wide_divisor;
        if(distance<((int64_t)32<<shift)) {
            uint16_t old=(uint16_t)(((unsigned)expected[14]<<8)|expected[15]);
            expected[14]=(uint8_t)((old|flags)>>8);expected[15]=(uint8_t)(old|flags);
        }
    }
    int positive_word_side=mark_floorline_touch_if_near(&ctx,line,lx,lz,vx,vz,shift);
    assert(!memcmp(line,expected,sizeof(line)));
    /* Repeated movers can encounter a line whose requested flags are set. */
    assert(mark_floorline_touch_if_near(&ctx,line,lx,lz,vx,vz,shift)==positive_word_side);
    assert(!memcmp(line,expected,sizeof(line)));
    int64_t rx=(int64_t)x-lx,rz=(int64_t)z-lz;
    int expected_side=flags && side>0 && rx>=INT16_MIN && rx<=INT16_MAX &&
                      rz>=INT16_MIN && rz<=INT16_MAX;
    assert(positive_word_side==expected_side);
    if(positive_word_side) {
        /* Independent ObjectMove.s checkwalls word subtraction/multiply.
         * A proven side must take its original immediate rejection. */
        int16_t word_x=(int16_t)((int16_t)x-(int16_t)lx);
        int16_t word_z=(int16_t)((int16_t)z-(int16_t)lz);
        int32_t word_cross=(int32_t)word_x*vz-(int32_t)word_z*vx;
        assert(word_cross>0);
    }
}

void check_movement_touch(void)
{
    static const int16_t lengths[][2]={{1,0},{17,0},{-32768,0},{32767,32767},
                                      {-32768,-32768},{0,-32768},{0,0}};
    static const uint16_t flags[]={0,1,0x400,0xffff};
    unsigned checked=0;
    for(unsigned mode=0;mode<2;mode++)for(unsigned n=0;n<7;n++)
    for(unsigned f=0;f<4;f++) {
        int shift=mode?16:0;
        int divisor=lengths[n][0]<0?-(int)lengths[n][0]:lengths[n][0];
        divisor+=lengths[n][1];if(divisor<=0)divisor=1;
        int vector=divisor>32767?divisor/2:divisor;
        int32_t boundary=(divisor/vector)*(32<<shift);
        for(int delta=-2;delta<=2;delta++) {
            check_touch(0,0,boundary,-delta,1,(int16_t)vector,
                        lengths[n][0],lengths[n][1],shift,flags[f]);checked++;
        }
        for(int delta=-1;delta<=1;delta++) {
            check_touch(0,0,0,delta,1,1,lengths[n][0],lengths[n][1],shift,flags[f]);checked++;
        }
    }
    for(unsigned i=0;i<65536;i++) {
        int shift=i&1?16:0;
        int32_t scale=(int32_t)1<<shift;
        int32_t lx=((int16_t)(touch_random()>>16)/4)*scale;
        int32_t lz=((int16_t)(touch_random()>>16)/4)*scale;
        int32_t x=((int16_t)(touch_random()>>16)/4)*scale;
        int32_t z=((int16_t)(touch_random()>>16)/4)*scale;
        int16_t vx=(int16_t)(touch_random()>>16),vz=(int16_t)(touch_random()>>16);
        int16_t length=(int16_t)(touch_random()>>16),extent=(int16_t)(touch_random()>>16);
        uint16_t flag=(uint16_t)touch_random();
        check_touch(lx,lz,x,z,vx,vz,length,extent,shift,flag);checked++;
    }
    static const int32_t word_edges[]={-65536,-32769,-32768,-32767,-1,0,1,32766,32767,32768,65535};
    static const int16_t vectors[][2]={{-32768,-32768},{-32768,32767},{32767,-32768},{32767,32767},
                                      {0,1},{1,0},{-1,0},{0,-1}};
    for(unsigned mode=0;mode<2;mode++)for(unsigned f=0;f<4;f++)
    for(unsigned v=0;v<8;v++)for(unsigned x=0;x<11;x++)for(unsigned z=0;z<11;z++) {
        check_touch(17,-23,17+word_edges[x],-23+word_edges[z],vectors[v][0],vectors[v][1],
                    (int16_t)(v&1?-32768:32767),(int16_t)(v&2?-32768:32767),mode?16:0,flags[f]);
        checked++;
    }
    /* The largest denominator's threshold, with unit changes in the cross
     * while every coordinate/direction still fits a signed word. */
    for(unsigned f=0;f<4;f++)for(int delta=-2;delta<=2;delta++) {
        check_touch(17,-23,17+32-delta,-23+32+delta,-32768,32767,
                    -32768,32767,0,flags[f]);checked++;
    }
    uint8_t empty_line[16]={0};MoveContext empty_ctx={0};
    assert(!mark_floorline_touch_if_near(NULL,empty_line,0,0,1,1,0));
    assert(!mark_floorline_touch_if_near(&empty_ctx,NULL,0,0,1,1,0));
    printf("PASS: %u movement touch cases preserve the original wide division, threshold boundaries, signed lengths/extents, both position scales and all line bytes\n",checked);
    puts("PASS: positive-side proofs match original signed-word crosses, including word overflow boundaries and absent touch flags");

    for(unsigned i=0;i<32768;i++) {
        MoveContext ctx={0};
        ctx.oldy=(int32_t)(touch_random()%2000000001u)-1000000000;
        ctx.newy=i%3==0?ctx.oldy:(int32_t)(touch_random()%2000000001u)-1000000000;
        int32_t cross_new=(int32_t)touch_random(),cross_old=(int32_t)touch_random();
        int32_t divisor=(int16_t)(touch_random()>>16);
        if(i%7==0)divisor=0;
        int64_t height=ctx.newy;
        if(divisor) {
            volatile int64_t wide_divisor=divisor;
            int64_t new_distance=(int64_t)cross_new/wide_divisor;
            int64_t old_distance=(int64_t)cross_old/wide_divisor;
            int64_t span=old_distance-new_distance;if(span<=0)span=1;
            int64_t dy=(int64_t)ctx.newy-ctx.oldy;
            if(dy)height=ctx.newy+(dy/span)*new_distance;
        }
        if(height<INT32_MIN)height=INT32_MIN;
        if(height>INT32_MAX)height=INT32_MAX;
        assert(compute_crossing_height_asm(&ctx,cross_new,cross_old,divisor)==height);
    }
    puts("PASS: 32768 movement crossing heights preserve signed division order, zero vertical motion/divisors and clamping");
}
