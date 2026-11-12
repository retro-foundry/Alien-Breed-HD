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

static void path_write_long(uint8_t *out,int32_t value)
{
    uint32_t bits=(uint32_t)value;
    out[0]=(uint8_t)(bits>>24);out[1]=(uint8_t)(bits>>16);
    out[2]=(uint8_t)(bits>>8);out[3]=(uint8_t)bits;
}

void check_movement_paths(void)
{
    enum { DATA_BYTES=1024, LINE_BYTES=16*12, GUARD=16, ROOM0=64, ROOM1=448,
           CASES=8192, EDGES=32*4 };
    static uint8_t data_store[DATA_BYTES+GUARD*2],line_store[LINE_BYTES+GUARD*2];
    static uint8_t data_seed[sizeof(data_store)],line_seed[sizeof(line_store)];
    static uint8_t data_expected[sizeof(data_store)],line_expected[sizeof(line_store)];
    uint8_t adds[8];LevelState level={0};
    uint8_t *data=data_store+GUARD,*lines=line_store+GUARD;
    level.data=data;level.data_byte_count=DATA_BYTES;level.floor_lines=lines;
    level.zone_adds=adds;level.num_zones=level.num_zone_slots=2;level.num_floor_lines=8;
    path_write_long(adds,ROOM0);path_write_long(adds+4,ROOM1);
    unsigned proved=0,rejected=0;
    static const int32_t origins[]={-24000,0,24000};
    static const int16_t shapes[3][4][2]={
        {{-512,-512},{-512,512},{512,512},{512,-512}},
        {{-512,0},{0,512},{512,0},{0,-512}},
        {{-768,-512},{-256,512},{768,512},{256,-512}}
    };
    MoveContext empty={0};
    assert(!movement_clear_room_path(NULL,&level));
    assert(!movement_clear_room_path(&empty,NULL));
    for(unsigned test=0;test<CASES+EDGES;test++) {
        level.data=data;level.floor_lines=lines;
        memset(data_store,0xa5,sizeof(data_store));memset(line_store,0x96,sizeof(line_store));
        memset(data,0,DATA_BYTES);memset(lines,0,LINE_BYTES);
        int32_t origin=test>=CASES?0:origins[(test/3)%3];
        unsigned shape=test>=CASES?0:(test/9)%3,room=test>=CASES?0:test&1;
        for(unsigned zone=0;zone<2;zone++) {
            uint8_t *header=data+(zone?ROOM1:ROOM0);
            write_be16(header,(uint16_t)zone);
            path_write_long(header+ZONE_FLOOR_HEIGHT,0);
            path_write_long(header+ZONE_ROOF_HEIGHT,-8192);
            write_be16(header+ZONE_EXIT_LIST,64);
            for(unsigned edge=0;edge<4;edge++) {
                unsigned next=(edge+1)%4,index=zone*4+edge;
                int32_t x0=origin+(int32_t)zone*1024+shapes[shape][edge][0];
                int32_t z0=shapes[shape][edge][1];
                int32_t vx=shapes[shape][next][0]-shapes[shape][edge][0];
                int32_t vz=shapes[shape][next][1]-shapes[shape][edge][1];
                uint8_t *line=lines+index*16;
                write_be16(line+FLINE_X,(uint16_t)x0);write_be16(line+FLINE_Z,(uint16_t)z0);
                write_be16(line+FLINE_XLEN,(uint16_t)vx);write_be16(line+FLINE_ZLEN,(uint16_t)vz);
                write_be16(line+FLINE_LENGTH,(uint16_t)(int)sqrt((double)vx*vx+(double)vz*vz));
                int connect=shape==0 && ((zone==0 && edge==2)||(zone==1 && edge==0))?1-(int)zone:-1;
                write_be16(line+FLINE_CONNECT,(uint16_t)connect);
                write_be16(line+FLINE_AWAY,(uint16_t)touch_random());
                line[FLINE_NORMAL]=(uint8_t)(test%5?0:1);line[FLINE_NORMAL+1]=0;
                write_be16(header+64+edge*2,(uint16_t)index);
            }
            write_be16(header+72,(uint16_t)-1);write_be16(header+74,(uint16_t)-2);
        }
        MoveContext seed;memset(&seed,0,sizeof(seed));
        int32_t center=origin+(int32_t)room*1024;
        int range=test%4==0?201:1401;
        seed.oldx=center+(int32_t)(touch_random()%(unsigned)range)-range/2;
        seed.oldz=(int32_t)(touch_random()%(unsigned)range)-range/2;
        seed.newx=center+(int32_t)(touch_random()%(unsigned)range)-range/2;
        seed.newz=(int32_t)(touch_random()%(unsigned)range)-range/2;
        seed.oldy=-2048;seed.newy=-1024-(int32_t)(test%512);
        seed.xdiff=123456;seed.zdiff=-654321;seed.thing_height=128;
        seed.step_up_val=256;seed.step_down_val=0x1000000;
        seed.objroom=data+(room?ROOM1:ROOM0);seed.awayfromwall=-1;
        seed.stood_in_top=(int8_t)(test&1);seed.coll_id=-1;seed.collide_flags=0x13579bdf;
        seed.wall_flags=test%3==0?0:test%3==1?0x400:0xffff;
        seed.hitwall=1;seed.wall_hit_y=12345;seed.wall_xsize=123;seed.wall_zsize=-456;seed.wall_length=789;
        seed.exitfirst=(int8_t)(test&1);seed.wallbounce=(int8_t)((test>>1)&1);
        if(test%17==0)seed.extlen=(int16_t)(test%2?80:-1);
        if(test%19==0)seed.awayfromwall=(int8_t)(test%3);
        if(test%23==0)seed.no_transition_back=data+(room?ROOM0:ROOM1);
        if(test%29==0)seed.objroom=NULL;
        if(test%31==0) {
            seed.pos_shift=16;seed.oldx=seed.oldz=0;
            seed.newx=65536;seed.newz=-65536;
        }
        if(test%37==0)seed.newx=seed.oldx+3,seed.newz=seed.oldz-2;
        if(test<CASES && test%41==0) {
            uint8_t *list=data+(room?ROOM1:ROOM0)+64;
            write_be16(list,999);write_be16(list+2,(uint16_t)-1); /* Invalid entry is skipped. */
        }
        if(test<CASES && test%43==0)write_be16(data+(room?ROOM1:ROOM0)+64,(uint16_t)-2);
        int expected_proof=-1;
        if(test>=CASES) {
            unsigned kind=(test-CASES)/4,flag=(test-CASES)%4;
            static const uint16_t flags[]={0,1,0x400,0xffff};
            memset(&seed,0,sizeof(seed));
            seed.objroom=data+ROOM0;seed.awayfromwall=-1;seed.wall_flags=flags[flag];
            seed.oldy=-2048;seed.newy=-1024;seed.newx=16;
            seed.thing_height=128;seed.step_up_val=256;seed.step_down_val=0x1000000;
            seed.coll_id=-1;seed.stood_in_top=(int8_t)(flag&1);
            seed.exitfirst=(int8_t)(flag&1);seed.wallbounce=(int8_t)(flag>>1);
            seed.hitwall=1;seed.wall_hit_y=12345;seed.wall_xsize=123;seed.wall_zsize=-456;seed.wall_length=789;
            uint8_t *list=data+ROOM0+64,*line=lines;
            expected_proof=1;
            if(kind==1)seed.newx=11,expected_proof=0;
            if(kind==2)seed.newx=12;
            if(kind==3)seed.oldx=-500,seed.newx=-484,expected_proof=!seed.wall_flags;
            if(kind==4)seed.oldx=-460,seed.newx=-500,expected_proof=!seed.wall_flags;
            if(kind==5)seed.oldx=-500,seed.newx=-530,expected_proof=0;
            if(kind==6)seed.oldx=500,seed.newx=530,expected_proof=0;
            if(kind>=7 && kind<=10) {
                write_be16(list,(uint16_t)-1);
                seed.oldx=kind==7?32767:kind==8?32768:kind==9?-32768:-32769;
                seed.newx=seed.oldx+(kind<9?-12:12);expected_proof=kind==7 || kind==9;
            }
            if(kind>=11 && kind<=15) {
                write_be16(list,0);write_be16(list+2,(uint16_t)-1);
                int32_t base=kind==11?-30000:kind==12?-32767:kind==13?-32768:32767;
                write_be16(line+FLINE_X,(uint16_t)base);write_be16(line+FLINE_Z,0);
                write_be16(line+FLINE_XLEN,0);write_be16(line+FLINE_ZLEN,(uint16_t)(kind>=14?-1000:1000));
                write_be16(line+FLINE_LENGTH,1000);write_be16(line+FLINE_CONNECT,(uint16_t)-1);
                seed.oldx=kind==11?30000:kind==14?-1:kind==15?-2:0;
                seed.newx=seed.oldx+(kind==12 || kind==13?-12:12);
                expected_proof=kind==12 || kind==14;
            }
            if(kind==16)seed.extlen=1,expected_proof=0;
            if(kind==17)seed.extlen=-1,expected_proof=0;
            if(kind==18)seed.awayfromwall=0,expected_proof=0;
            if(kind==19)seed.awayfromwall=1,expected_proof=0;
            if(kind==20)seed.pos_shift=16,expected_proof=0;
            if(kind==21)seed.objroom=NULL,expected_proof=0;
            if(kind==22)level.floor_lines=NULL,expected_proof=0;
            if(kind==23)level.data=NULL,expected_proof=0;
            if(kind==24)write_be16(list,(uint16_t)-1);
            if(kind==25)write_be16(list,999),write_be16(list+2,(uint16_t)-1);
            if(kind==26) {
                write_be16(list,0);write_be16(list+2,(uint16_t)-1);
                write_be16(line+FLINE_XLEN,0);write_be16(line+FLINE_ZLEN,0);expected_proof=0;
            }
            if(kind==27 || kind==28 || kind==31) {
                write_be16(list,0);write_be16(list+2,(uint16_t)-1);
                write_be16(line+FLINE_X,0);write_be16(line+FLINE_Z,0);
                write_be16(line+FLINE_XLEN,(uint16_t)(kind==27?32767:-32768));
                write_be16(line+FLINE_ZLEN,(uint16_t)(kind==27?-32768:32767));
                write_be16(line+FLINE_LENGTH,(uint16_t)-32768);write_be16(line+FLINE_CONNECT,(uint16_t)-1);
                seed.oldx=seed.oldz=-32768;seed.newx=seed.newz=-32756;
                expected_proof=kind==27;
                if(kind==31) {
                    /* Tight diagonal margins distinguish endpoints from a
                     * proof that also covers independently rounded steps. */
                    seed.oldx=32;seed.oldz=2+(int32_t)(flag&1);
                    seed.newx=48;seed.newz=seed.oldz-16;
                    expected_proof=!seed.wall_flags || (flag&1);
                }
            }
            if(kind==29 || kind==30) {
                for(unsigned i=0;i<128;i++)write_be16(list+i*2,0);
                if(kind==30)write_be16(list+127*2,(uint16_t)-1);
                expected_proof=kind==30;
            }
        }
        MoveContext reference=seed,actual=seed;
        memcpy(data_seed,data_store,sizeof(data_store));memcpy(line_seed,line_store,sizeof(line_store));
        int clear=movement_clear_room_path(&seed,&level);
        assert(!memcmp(&seed,&reference,sizeof(seed)));
        assert(!memcmp(data_store,data_seed,sizeof(data_store)));
        assert(!memcmp(line_store,line_seed,sizeof(line_store)));
        if(expected_proof>=0 && clear!=expected_proof)
            fprintf(stderr,"Movement proof case=%u kind=%u actual=%d expected=%d\n",
                    test,(test-CASES)/4,clear,expected_proof);
        if(expected_proof>=0)assert(clear==expected_proof);
        proved+=clear;rejected+=!clear;
        if(seed.pos_shift || seed.extlen || seed.awayfromwall>=0 || !seed.objroom)assert(!clear);
        move_object_substep_loop(&reference,&level);
        memcpy(data_expected,data_store,sizeof(data_store));memcpy(line_expected,line_store,sizeof(line_store));
        memcpy(data_store,data_seed,sizeof(data_store));memcpy(line_store,line_seed,sizeof(line_store));
        move_object_substepped(&actual,&level);
        if(memcmp(&actual,&reference,sizeof(actual)) ||
           memcmp(data_store,data_expected,sizeof(data_store)) ||
           memcmp(line_store,line_expected,sizeof(line_store))) {
            fprintf(stderr,"Movement path mismatch case=%u clear=%d shape=%u room=%u\n",test,clear,shape,room);
            assert(0);
        }
        if(clear) {
            assert(!actual.hitwall && actual.newx==seed.newx && actual.newz==seed.newz);
            assert(!memcmp(line_store,line_seed,sizeof(line_store)));
        }
    }
    assert(proved && rejected);
    printf("PASS: %u complete substep comparisons (%u clear proofs, %u original paths) retain every context/room/floorline byte, transitions, touches, collision modes and guards\n",CASES+EDGES,proved,rejected);
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
