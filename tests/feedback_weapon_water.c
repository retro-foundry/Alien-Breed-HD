/* Replay AB3DI.s DrawInGun -> fillscrnwater using authored water and gun data. */
#include "../src/renderer.c"
#include "sb_decompress.h"
#include <assert.h>

#ifndef AB3D_WEAPON_WATER_EXTERNAL_STATE
GameState g_state;
#endif
static uint8_t water_graphics[256];
static int32_t water_plane;

void weapon_water_scene_init(void)
{
    LevelState *level=&g_state.level;
    size_t bytes;
    assert(!sb_load_file("amiga/data/levels/level_a/twolev.bin",&level->data,&bytes));
    level->data_byte_count=bytes;
    assert(!sb_load_file("amiga/data/levels/level_a/twolev.graph.bin",&level->graphics,&bytes));
    level->graphics_byte_count=bytes;
    assert(!sb_load_file("amiga/data/levels/level_a/twolev.clips",&level->clips,&bytes));
    level->clips_byte_count=bytes;
    assert(!level_parse(level));
    level_assign_clips(level,level->num_zones);
    const int zone=85;
    const uint8_t *entry=level->graphics+rd32(level->zone_graph_adds+zone*8)+2;
    while(rd16(entry)!=7){assert(rd16(entry)>=0);entry+=2+zone_gfx_entry_data_skip(rd16(entry),entry+2);}
    size_t size=2+zone_gfx_entry_data_skip(7,entry+2);
    assert(size+6<=sizeof(water_graphics));
    obj_sw(water_graphics+2,zone);memcpy(water_graphics+4,entry,size);obj_sw(water_graphics+4+size,-1);
    water_plane=(int32_t)rd16(entry+2)*64;
    int count=rd16(entry+4)+1,x=0,z=0;
    assert(count>=3 && count<=100);
    for(int i=0;i<count;i++) {
        int point=rd16(entry+6+i*2);x+=rd16(level->points+point*4);z+=rd16(level->points+point*4+2);
    }
    uint8_t *adds=calloc((size_t)level_zone_slot_count(level),8);assert(adds);
    obj_sl(adds+zone*8,2);level->zone_graph_adds=adds;
    level->graphics=water_graphics;level->graphics_byte_count=size+6;level->object_data=NULL;
    uint8_t *water,*brighten,*wad,*ptr,*pal;
    size_t water_size,brighten_size,wad_size,ptr_size,pal_size;
    assert(!sb_load_file("amiga/data/helper/WaterFile",&water,&water_size));
    assert(!sb_load_file("amiga/data/helper/OldBrightenFile",&brighten,&brighten_size));
    assert(!sb_load_file("amiga/data/includes/newgunsinhand.wad",&wad,&wad_size));
    assert(!sb_load_file("amiga/data/includes/newgunsinhand.ptr",&ptr,&ptr_size));
    assert(!sb_load_file("amiga/data/pal/NewGunsInHand.pal",&pal,&pal_size));
    renderer_set_water_assets(water,water_size,brighten,brighten_size);
    g_renderer.gun_wad=wad;g_renderer.gun_wad_size=wad_size;
    g_renderer.gun_ptr=ptr;g_renderer.gun_ptr_size=ptr_size;g_renderer.gun_pal=pal;
    for(int i=0;i<2;i++) {
        PlayerState *p=i?&g_state.plr2:&g_state.plr1;
        p->zone=zone;p->roompt=(int32_t)(level_get_zone_data_ptr(level,zone)-level->data);
        p->xoff=p->oldxoff=x/count*65536;p->zoff=p->oldzoff=z/count*65536;
        p->angpos=p->oldangpos=0;p->look_y=p->oldlook_y=0;
    }
    g_state.zone_order_count=1;g_state.zone_order_zones[0]=zone;
    g_state.view_list_of_graph_rooms=level->data+g_state.plr1.roompt+48;
    g_state.cfg_y_proj_scale=100;g_state.cfg_mouse_look=true;g_state.cfg_render_threads_max=2;
    g_state.cfg_weapon_draw=true;g_state.cfg_post_tint=true;
    g_state.plr1.energy=g_state.plr2.energy=100;
}

void weapon_water_scene_height(int relative)
{
    g_state.plr1.yoff=g_state.plr1.oldyoff=water_plane-relative;
    g_state.plr2.yoff=g_state.plr2.oldyoff=water_plane-relative;
}

void weapon_water_scene_reset(void)
{
    renderer_clear_frame_history();g_render_frame_counter=0;
}

#ifndef AB3D_WEAPON_WATER_SCENE_LIB
enum { WW=192,WH=160 };
static RendererPixel dry_background[WW*WH],dry_weapon[WW*WH],wet_background[WW*WH];
static void capture(RendererPixel *pixels,int gun,int tint)
{
    g_state.cfg_weapon_draw=gun!=0;g_state.cfg_post_tint=tint!=0;
    weapon_water_scene_reset();renderer_draw_display(&g_state);
    if(pixels)memcpy(pixels,renderer_get_cw_buffer(),sizeof(dry_background));
}

int main(void)
{
    assert(SDL_Init(SDL_INIT_TIMER)==0);
    math_tables_init();renderer_init(WW,WH);weapon_water_scene_init();
    const int guns[]={0,1,2,3,4,7},heights[]={256,0,-256};
    unsigned cases=0,red_pixels=0;
    int thread_modes=1;
#ifndef AB3D_NO_THREADS
    if(SDL_GetCPUCount()>1){assert(g_renderer_thread_pool.worker_count>0);thread_modes=2;}
#endif
    for(int threaded=0;threaded<thread_modes;threaded++)for(int expand=0;expand<2;expand++)
    for(int player=0;player<2;player++)for(int gun=0;gun<6;gun++)for(int frame=0;frame<4;frame++)
    for(int height=0;height<3;height++) {
        g_state.cfg_render_threads=threaded!=0;g_state.mode=player?MODE_SLAVE:MODE_SINGLE;
        PlayerState *p=player?&g_state.plr2:&g_state.plr1;p->gun_selected=guns[gun];p->gun_frame=frame;
        renderer_set_rgb_raster_expand(expand);weapon_water_scene_height(heights[height]);
        renderer_set_weapon_post_gl_active(0);renderer_set_gl_water_tint_post_active(0);
        capture(dry_background,0,0);capture(dry_weapon,1,0);
        unsigned dry_red=0,wet_visible=0;
        for(int i=0;i<WW*WH;i++)dry_red+=((renderer_pixel_to_c12(dry_weapon[i])^renderer_pixel_to_c12(dry_background[i]))&0xf00)!=0;
        assert(dry_red>0);red_pixels+=dry_red;
        capture(wet_background,0,1);capture(NULL,1,1);
        const RendererPixel *actual=renderer_get_cw_buffer();
        /* At the exact plane, world clipping may omit the coplanar polygon;
         * both no-water and at-surface flags deliberately leave colors dry. */
        if(heights[height]<0) {
            if(renderer_get_last_fill_screen_water()!=15) {
                RendererWorldZonePrepass prepass;renderer_build_world_zone_prepass(&g_state,0,0,&prepass);
                fprintf(stderr,"Water fixture: fill=%d zone=%d camera=%d,%d,%d plane=%d count=%d valid=%d lower=%d upper=%d\n",
                    renderer_get_last_fill_screen_water(),p->zone,g_renderer.xoff,g_renderer.yoff,g_renderer.zoff,water_plane,
                    prepass.count,prepass.valid[0],prepass.lower_clip[0].valid,prepass.upper_clip[0].valid);
            }
            assert(renderer_get_last_fill_screen_water()==15);
        }
        else assert(renderer_get_last_fill_screen_water()<=0);
        if(heights[height]<0) {
            for(int i=0;i<WW*WH;i++) {
                assert((renderer_pixel_to_c12(actual[i])&0xf00)==(renderer_pixel_to_c12(wet_background[i])&0xf00));
                wet_visible+=actual[i]!=wet_background[i];
                if(expand)assert((renderer_get_rgb_buffer()[i]&0xff0000)==0);
            }
            assert(wet_visible>0);
        } else assert(!memcmp(actual,dry_weapon,sizeof(dry_weapon)));
        /* A GL-owned weapon must not also appear in worker/main CPU pixels. */
        renderer_set_weapon_post_gl_active(1);renderer_set_gl_water_tint_post_active(1);
        capture(NULL,1,1);
        assert(!memcmp(renderer_get_cw_buffer(),dry_background,sizeof(dry_background)));
#ifndef AB3D_NO_THREADS
        if(threaded)assert(g_prof_last_world_workers>0);
#endif
        cases++;
    }
    renderer_shutdown();SDL_Quit();
    printf("PASS: %u weapon/water cases, %u authored red pixels; %d thread mode(s), ARGB/CW, both players, GL ownership\n",cases,red_pixels,thread_modes);
    return 0;
}
#endif
