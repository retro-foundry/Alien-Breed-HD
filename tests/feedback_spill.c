/* Cross-zone billboards are an optional port enhancement. ObjDraw3's ObjDraw
 * owns objects by GraphicRoom; BitMapObj supplies the camera-facing extent.
 * These fixtures use ObjectMove's boundary/-1/neighbor-collision/-2 format. */
#include "../src/renderer.c"
#include "sb_decompress.h"
#include "io.h"
#include "objects.h"
#include <assert.h>

GameState g_state;
enum { CELLS = 12, CELL_BYTES = 64 };
static GameState fixture;
static uint8_t data[CELLS * CELL_BYTES], adds[CELLS * 4], lines[CELLS * 4 * 16];

static void strip(int count, int closed_edge)
{
    memset(&fixture, 0, sizeof(fixture));
    memset(data, 0, sizeof(data)); memset(lines, 0, sizeof(lines));
    memset(g_renderer_connected_zone_cache, 0, sizeof(g_renderer_connected_zone_cache));
    memset(g_renderer_portal_line_cache, 0, sizeof(g_renderer_portal_line_cache));
    LevelState *level = &fixture.level;
    level->data=data; level->data_byte_count=sizeof(data); level->zone_adds=adds;
    level->floor_lines=lines; level->num_floor_lines=count*4;
    level->num_zones=level->num_zone_slots=count;
    fixture.cfg_billboard_sprite_rendering_enhancement=true;
    fixture.zone_order_count=count;
    for(int z=0;z<count;z++) {
        uint8_t *room=data+z*CELL_BYTES;
        obj_sl(adds+z*4,z*CELL_BYTES); obj_sw(room,z);
        obj_sl(room+2,32*256); obj_sl(room+6,-32*256);
        obj_sw(room+32,40); fixture.zone_order_zones[z]=(int16_t)(count-1-z);
        const int corners[4][2]={{z*16,-16},{(z+1)*16,-16},{(z+1)*16,16},{z*16,16}};
        int n=0;
        for(int e=0;e<4;e++) {
            uint8_t *line=lines+(z*4+e)*16;
            obj_sw(line,corners[e][0]); obj_sw(line+2,corners[e][1]);
            obj_sw(line+4,corners[(e+1)%4][0]-corners[e][0]);
            obj_sw(line+6,corners[(e+1)%4][1]-corners[e][1]);
            int connect=-1;
            if(e==1 && z+1<count && z!=closed_edge) connect=z+1;
            if(e==3 && z>0 && z-1!=closed_edge) connect=z-1;
            obj_sw(line+8,connect);
            obj_sw(room+40+2*n++,z*4+e);
        }
        obj_sw(room+40+2*n++,-1);
        obj_sw(room+40+2*n,-2);
    }
}

static void check_multizone(void)
{
    strip(7,-1);
    int16_t zones[CELLS];
    int found=renderer_collect_billboard_zones(&fixture.level,3,56,0,112,
                                               16384,0,zones,CELLS);
    for(int zone=0;zone<7;zone++) if(zone!=3) {
        if(!renderer_zone_list_contains(zones,found,(int16_t)zone)) {
            fprintf(stderr,"missing multi-zone footprint: source=3 destination=%d found=%d\n",zone,found);
            abort();
        }
    }
}

static unsigned checks;

static void check_primary_outline(void)
{
    strip(7,-1);
    uint8_t *line=lines+28*16;
    obj_sw(line,30);obj_sw(line+2,-16);obj_sw(line+4,0);obj_sw(line+6,32);
    fixture.level.num_floor_lines=29;
    obj_sw(data+50,28);obj_sw(data+52,-2);
    assert(renderer_zone_contains_point(&fixture.level,0,8,0));
    assert(!renderer_zone_contains_point(&fixture.level,0,20,0));
    assert(!renderer_zone_contains_point(&fixture.level,0,-.25,.25));
    assert(renderer_zone_contains_point(&fixture.level,0,.25,.25));
    int starts[4],ends[4];
    int count=renderer_zone_clip_segment_to_dx_ranges(&fixture.level,0,0,0,112,0,
                                                       224,0,224,starts,ends,NULL,NULL,4);
    assert(count==1 && starts[0]==0 && ends[0]==32);
    puts("PASS: neighboring collision edges are excluded from the outline; fractional midpoint ownership stays exact");
}

static void check_portals_and_order(void)
{
    for(int closed=-1;closed<6;closed++) {
        strip(7,closed);
        int16_t zones[CELLS];
        int count=renderer_collect_billboard_zones(&fixture.level,3,56,0,112,16384,0,zones,CELLS);
        for(int zone=0;zone<7;zone++) {
            int expected=closed<0 || (zone<=closed)==(3<=closed);
            assert(renderer_zone_list_contains(zones,count,(int16_t)zone)==expected);
            checks++;
        }
    }
    strip(7,-1);
    /* Only right-to-left portal records: reverse connections still work. */
    for(int zone=0;zone<7;zone++) obj_sw(lines+(zone*4+1)*16+8,-1);
    memset(g_renderer_connected_zone_cache,0,sizeof(g_renderer_connected_zone_cache));
    memset(g_renderer_portal_line_cache,0,sizeof(g_renderer_portal_line_cache));
    int16_t zones[CELLS];
    assert(renderer_collect_billboard_zones(&fixture.level,3,56,0,112,16384,0,zones,CELLS)==7);
    assert(renderer_collect_billboard_zones(&fixture.level,3,56,32,112,16384,0,zones,CELLS)==1);
    assert(renderer_collect_billboard_zones(&fixture.level,3,56,0,8,16384,0,zones,CELLS)==1);
    assert(renderer_collect_billboard_zones(&fixture.level,3,56,0,112,0,16384,zones,CELLS)==1);
    fixture.zone_order_count=256;
    for(int order=0;order<3;order++) {
        for(int i=0;i<256;i++) fixture.zone_order_zones[i]=-1;
        fixture.zone_order_zones[order==0?0:127]=3;
        fixture.zone_order_zones[order==0?255:0]=0;
        assert(renderer_spill_draw_still_valid(&fixture,3,0,56,0,112,16384,0,0,224,112,224));
        checks++;
    }
    puts("PASS: multi-hop, closed/disconnected, reverse-only, off-plane, rotated and order-independent portal cases");
}

static void check_column_ownership(void)
{
    strip(7,-1);
    static const int widths[]={4,11,127,224,320,768};
    for(unsigned w=0;w<sizeof(widths)/sizeof(*widths);w++) {
        int width=widths[w], owners[768]={0};
        for(int zone=0;zone<7;zone++) {
            int starts[16],ends[16];
            int count=renderer_zone_clip_segment_to_dx_ranges(&fixture.level,(int16_t)zone,
                            0,0,112,0,width,0,width,starts,ends,NULL,NULL,16);
            for(int i=0;i<count;i++) for(int dx=starts[i];dx<ends[i];dx++) {
                assert((int)(((2*dx+1)*112)/(2*width))/16==zone);
                owners[dx]++; checks++;
            }
        }
        for(int dx=0;dx<width;dx++) assert(owners[dx]==1);
    }
    puts("PASS: every column has exactly one owner across seven cells at six view widths");
}

static void check_interpolation_and_index(void)
{
    strip(7,-1);
    static uint8_t objects[256*OBJECT_SIZE], points[8], previous[8];
    static uint8_t nasty[NASTY_SHOT_SLOT_COUNT*OBJECT_SIZE], player[PLAYER_SHOT_SLOT_COUNT*OBJECT_SIZE];
    LevelState *level=&fixture.level;
    level->object_data=objects;level->object_points=points;level->prev_object_points=previous;
    level->num_object_points=1;level->nasty_shot_data=nasty;level->player_shot_data=player;
    obj_sw(points,80);obj_sw(points+4,8);obj_sw(previous,32);obj_sw(previous+4,8);
    memset(objects,0,sizeof(objects));memset(nasty,0,sizeof(nasty));memset(player,0,sizeof(player));
    for(int i=0;i<256;i++) { obj_sw(objects+i*OBJECT_SIZE+12,3);objects[i*OBJECT_SIZE+6]=112; }
    for(int pool=0;pool<2;pool++) {
        uint8_t *p=pool?player:nasty;int count=pool?PLAYER_SHOT_SLOT_COUNT:NASTY_SHOT_SLOT_COUNT;
        for(int i=0;i<count;i++) {obj_sw(p+i*OBJECT_SIZE+12,3);p[i*OBJECT_SIZE+16]=OBJ_NBR_BULLET;}
    }
    static const float alphas[]={0,.125f,.5f,.875f,1};
    g_renderer.width=224;g_renderer.height=200;g_renderer.cosval=32767;g_renderer.sinval=0;
    g_renderer.xoff=56;g_renderer.zoff=-128;g_renderer.view_center_x=112;
    for(unsigned a=0;a<sizeof(alphas)/sizeof(*alphas);a++) {
        fixture.obj_interp_alpha=alphas[a];
        int32_t x,z;renderer_object_world_position(&fixture,0,&x,&z);
        assert(x==(alphas[a]>0 ? 32+(int)(48*alphas[a]) : 80) && z==8);
        renderer_rotate_object_pts(&fixture);
        int32_t expected_x=(((int32_t)(x-56)*32767)*2)>>9;
        assert(g_renderer.obj_rotated[0].x_fine==expected_x);
        checks++;
    }
    renderer_objects_prepare(level,g_renderer.obj_rotated,ROT_Z_FROM_INT(SPRITE_NEAR_CLIP_Z));
    const unsigned selected[]={0,255,256,355,356,375,376,423};
    for(unsigned i=0;i<sizeof(selected)/sizeof(*selected);i++)
        renderer_objects_add_room(level,6,selected[i]);
    uint8_t actual[256];unsigned count=renderer_objects_for_room(level,6,NULL,0,actual);
    assert(count==2 && actual[0]==0 && actual[1]==255);
    unsigned counts[2];renderer_objects_shots_for_room(level,6,NULL,0,counts,actual);
    assert(counts[0]==2 && counts[1]==2);
    assert(actual[0]==0 && actual[1]==99 && actual[2]==0 && actual[3]==19);
    assert(renderer_objects_in_room(level,6,376) && renderer_objects_in_room(level,6,423));
    renderer_objects_prepare(level,g_renderer.obj_rotated,ROT_Z_FROM_INT(SPRITE_NEAR_CLIP_Z));
    assert(!renderer_objects_for_room(level,6,NULL,0,actual));
    assert(!renderer_objects_in_room(level,6,376));
    /* Gameplay has crossed into cell 5 while the visible center is still in
     * cell 3. A billboard parallel to the portal cannot discover cell 3 by
     * walking from the new owner along its lateral segment. */
    for(int i=0;i<256;i++) obj_sw(objects+i*OBJECT_SIZE+12,5);
    obj_sw(points,84);obj_sw(previous,36);fixture.obj_interp_alpha=.5f;
    g_renderer.cosval=0;g_renderer.sinval=32767;g_renderer.xoff=-128;g_renderer.zoff=8;
    renderer_rotate_object_pts(&fixture);
    int32_t x,z;renderer_object_world_position(&fixture,0,&x,&z);
    assert(x==60 && renderer_billboard_display_zone(&fixture,0,5,x,z)==3);
    renderer_prepare_object_rooms(&fixture);
    assert(renderer_objects_for_room(level,3,NULL,0,actual)==256);
    g_renderer_object_index_state=NULL;
    puts("PASS: interpolation matches rotation; object/shot/explosion slots remain separate and reset per frame");
}

static void check_raster_fragments(void)
{
    strip(7,-1);
    static RendererPixel pixels[224*16];
    static uint8_t pal[64], tags[224*16], wad[4], ptr[8]={1,0,0,0,1,0,0,0};
    obj_sw(pal+62,0xabc);
    sprite_word_write(wad,31<<5);sprite_word_write(wad+2,31<<5);
    g_renderer.width=224;g_renderer.height=16;g_renderer.cw_buffer=pixels;g_renderer.buffer=tags;
    g_renderer_rgb_raster_expand=0;
    for(int zone=0;zone<7;zone++) for(int local=0;local<2;local++) for(int band=0;band<3;band++) {
        memset(pixels,0,sizeof(pixels));
        RenderSliceContext ctx={0};ctx.left_clip=0;ctx.right_clip=224;ctx.bot_clip=15;ctx.profile_collect_stats=1;
        renderer_draw_sprite_ctx(&ctx,112,8,224,8,256,wad,sizeof(wad),ptr,sizeof(ptr),pal,sizeof(pal),
                                 0,0,1,1,0,0,0,224,0,256,7,8-band,!local,
                                 &fixture.level,(int16_t)zone,56,0,112,16384,0);
        for(int y=0;y<16;y++) for(int x=0;x<224;x++) {
            int expected=(x/32==zone && y>=7 && y<=8-band);
            if ((pixels[renderer_cw_index_xy(x,y,224,16)]!=0)!=expected) {
                fprintf(stderr,"raster mismatch zone=%d local=%d x=%d y=%d expected=%d pixel=%x\n",
                        zone,local,x,y,expected,pixels[renderer_cw_index_xy(x,y,224,16)]);
                abort();
            }
            checks++;
        }
    }
    g_renderer.cw_buffer=NULL;g_renderer.buffer=NULL;
    puts("PASS: owned and spill raster fragments use the same footprint, including two-row, one-row and empty room bands");
}

/* Independent floating-point ray cast: test the sample itself, never a rounded
 * midpoint between intersections. Boundary samples are left to the exact strip
 * ownership cases above. */
static int reference_contains(const LevelState *level,int zone,double x,double z)
{
    const uint8_t *room=level->data+obj_l(level->zone_adds+zone*4);
    const uint8_t *list=room+obj_w(room+32);
    int inside=0;
    for(int i=0;i<256;i++) {
        int index=obj_w(list+i*2);
        if(index<0) break;
        if(index>=level->num_floor_lines) continue;
        const uint8_t *line=level->floor_lines+index*16;
        double x0=obj_w(line),z0=obj_w(line+2),x1=x0+obj_w(line+4),z1=z0+obj_w(line+6);
        double cross=(x-x0)*(z1-z0)-(z-z0)*(x1-x0);
        if(fabs(cross)<1e-7 && x>=fmin(x0,x1) && x<=fmax(x0,x1) &&
           z>=fmin(z0,z1) && z<=fmax(z0,z1)) return -1;
        if((z0>z)!=(z1>z) && x<x0+(z-z0)*(x1-x0)/(z1-z0)) inside^=1;
    }
    return inside;
}

static void check_authored_levels(void)
{
    unsigned cases=0,samples=0,multi=0,portal_cases=0,portal_multi=0;
    math_tables_init();
    for(int map=0;map<16;map++) {
        static GameState state;memset(&state,0,sizeof(state));
        char path[128];size_t bytes;
        snprintf(path,sizeof(path),"amiga/data/levels/level_%c/twolev.bin",'a'+map);
        assert(!sb_load_file(path,&state.level.data,&bytes));state.level.data_byte_count=bytes;
        snprintf(path,sizeof(path),"amiga/data/levels/level_%c/twolev.graph.bin",'a'+map);
        assert(!sb_load_file(path,&state.level.graphics,&bytes));state.level.graphics_byte_count=bytes;
        assert(!level_parse(&state.level));
        object_init_world_sizes_from_types(&state.level);
        memset(g_renderer_connected_zone_cache,0,sizeof(g_renderer_connected_zone_cache));
        memset(g_renderer_portal_line_cache,0,sizeof(g_renderer_portal_line_cache));
        LevelState *level=&state.level;
        for(int slot=0;slot<256;slot++) {
            const uint8_t *obj=level->object_data+slot*OBJECT_SIZE;
            int point=obj_w(obj),source=obj_w(obj+12);
            if(point<0)break;
            if(obj[6]==OBJ_3D_SPRITE || source<0 || point>=level->num_object_points) continue;
            int world_w,world_h;int32_t x,z;
            renderer_resolve_billboard_world_size_for_spill(obj,0,&world_w,&world_h);
            renderer_object_world_position(&state,point,&x,&z);
            for(int angle=0;angle<8192;angle+=1024) {
                int16_t rx=cos_lookup((int16_t)angle)/2,rz=-sin_lookup((int16_t)angle)/2;
                int16_t rooms[256];
                int count=renderer_collect_billboard_zones(level,(int16_t)source,x,z,world_w,rx,rz,rooms,256);
                int32_t x0,z0,x1,z1;
                assert(renderer_compute_billboard_world_segment(x,z,world_w,rx,rz,&x0,&z0,&x1,&z1));
                int16_t neighbors[256];
                int adjacent=renderer_collect_connected_zones_fast(level,(int16_t)source,neighbors,256);
                for(int ri=0;ri<count;ri++) {
                    int zone=rooms[ri],starts[256],ends[256];
                    if(zone!=source && !renderer_zone_list_contains(neighbors,adjacent,(int16_t)zone)) multi++;
                    int spans=renderer_zone_clip_segment_to_dx_ranges(level,(int16_t)zone,x0,z0,x1,z1,
                                                                      127,0,127,starts,ends,NULL,NULL,256);
                    for(int dx=0;dx<127;dx++) {
                        double t=(dx+.5)/127.0;
                        int expected=reference_contains(level,zone,x0+(x1-x0)*t,z0+(z1-z0)*t);
                        if(expected<0)continue;
                        int actual=0;for(int i=0;i<spans;i++) if(dx>=starts[i] && dx<ends[i])actual=1;
                        if(actual!=expected) {
                            fprintf(stderr,"authored clip mismatch level=%d object=%d source=%d zone=%d angle=%d dx=%d actual=%d expected=%d segment=(%d,%d)-(%d,%d)\n",
                                    map+1,slot,source,zone,angle,dx,actual,expected,x0,z0,x1,z1);
                            fprintf(stderr,"direct contains=%d spans=%d sample=%.9f,%.9f\n",renderer_zone_contains_point(level,(int16_t)zone,x0+(x1-x0)*t,z0+(z1-z0)*t),spans,x0+(x1-x0)*t,z0+(z1-z0)*t);
                            for(int i=0;i<spans;i++)fprintf(stderr,"span %d %d\n",starts[i],ends[i]);
                            const uint8_t *list=renderer_zone_exit_list_ptr(level,(int16_t)zone);
                            for(int i=0;i<256 && obj_w(list+2*i)!=-2;i++) {int li=obj_w(list+2*i);if(li<0)continue;const uint8_t *fl=level->floor_lines+li*16;fprintf(stderr,"edge %d: %d %d %d %d\n",li,obj_w(fl),obj_w(fl+2),obj_w(fl+4),obj_w(fl+6));}
                            abort();
                        }
                        samples++;
                    }
                }
                cases++;
            }
        }
        /* Move the original widest footprint along real portal corners and
         * midpoints. Starting poses alone do not exercise a crossing enemy. */
        int widest=0;
        for(int type=0;type<21;type++) if(default_object_world_size[type].w>widest)
            widest=default_object_world_size[type].w;
        for(int source=0;source<level_zone_slot_count(level);source++) {
            const uint8_t *list=renderer_zone_exit_list_ptr(level,(int16_t)source);
            if(!list)continue;
            int16_t adjacent[256];
            int neighbors=renderer_collect_connected_zones_fast(level,(int16_t)source,adjacent,256);
            for(int e=0;e<128 && obj_w(list+e*2)>=0;e++) {
                int line_id=obj_w(list+e*2);if(line_id>=level->num_floor_lines)continue;
                const uint8_t *line=level->floor_lines+line_id*16;
                if(obj_w(line+8)<0)continue;
                for(int along=0;along<2;along++) for(int angle=0;angle<8192;angle+=1024) {
                    int32_t x=obj_w(line)+along*obj_w(line+4)/2;
                    int32_t z=obj_w(line+2)+along*obj_w(line+6)/2;
                    int16_t rx=cos_lookup((int16_t)angle)/2,rz=-sin_lookup((int16_t)angle)/2,rooms[256];
                    int found=renderer_collect_billboard_zones(level,(int16_t)source,x,z,widest,rx,rz,rooms,256);
                    int32_t x0,z0,x1,z1;
                    assert(renderer_compute_billboard_world_segment(x,z,widest,rx,rz,&x0,&z0,&x1,&z1));
                    for(int i=0;i<found;i++) if(rooms[i]!=source && !renderer_zone_list_contains(adjacent,neighbors,rooms[i])) {
                        int starts[256],ends[256];
                        int spans=renderer_zone_clip_segment_to_dx_ranges(level,rooms[i],x0,z0,x1,z1,127,0,127,
                                                                            starts,ends,NULL,NULL,256);
                        if(spans>0)portal_multi++;
                    }
                    portal_cases++;
                }
            }
        }
        io_release_level_memory(level);
    }
    printf("PASS: %u authored billboard orientations, %u pixel samples and %u multi-hop pieces across all sixteen levels\n",cases,samples,multi);
    printf("PASS: %u real portal-corner/midpoint orientations include %u nonempty multi-hop fragments\n",portal_cases,portal_multi);
}

int main(void)
{
    setvbuf(stdout,NULL,_IOLBF,0);
    check_multizone();
    check_primary_outline();
    check_portals_and_order();
    check_column_ownership();
    check_interpolation_and_index();
    check_raster_fragments();
    check_authored_levels();
    printf("PASS: %u spill geometry and position checkpoints\n",checks);
    puts("Billboard spill regression passed");
    return 0;
}
