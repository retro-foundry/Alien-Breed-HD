/* Exercise the release renderer source and prepared DOS gun/water assets.
 * Include the renderer to parse authored graphics and freeze capture history. */
#include "../src/renderer.c"
#include "../src/game_loop.h"
#include "../src/level.h"
static void weapon_water_tick(GameState *, GameLoopCtx *);
#define game_loop_tick weapon_water_tick
#include "../src/main.c"
#undef game_loop_tick

enum { SAMPLES=2048 };
static RendererPixel background[SAMPLES],dry_weapon[SAMPLES];
static int32_t plane;

static void require(int condition,const char *message)
{
    if(!condition){fprintf(stderr,"Weapon water: %s\n",message);exit(4);}
}

static void place_camera(GameState *state)
{
    LevelState *level=&state->level;
    const int zone=85;
    const uint8_t *entry=level->graphics+obj_l(level->zone_graph_adds+zone*8)+2;
    /* Use the authored water record, as in feedback_weapon_water. */
    while(obj_w(entry)!=7) {
        int type=obj_w(entry);require(type>=0,"missing authored water");
        entry+=2+zone_gfx_entry_data_skip(type,entry+2);
    }
    plane=(int32_t)obj_w(entry+2)*64;
    int count=obj_w(entry+4)+1,x=0,z=0;
    require(count>=3 && count<=100,"invalid water polygon");
    for(int i=0;i<count;i++) {
        int point=obj_w(entry+6+i*2);
        x+=obj_w(level->points+point*4);z+=obj_w(level->points+point*4+2);
    }
    for(int i=0;i<2;i++) {
        PlayerState *p=i?&state->plr2:&state->plr1;
        p->zone=zone;p->roompt=(int32_t)(level_get_zone_data_ptr(level,zone)-level->data);
        p->xoff=p->oldxoff=x/count*65536;p->zoff=p->oldzoff=z/count*65536;
        p->angpos=p->oldangpos=0;p->look_y=p->oldlook_y=0;p->energy=100;
    }
    state->zone_order_count=1;state->zone_order_zones[0]=zone;
    state->view_list_of_graph_rooms=level->data+state->plr1.roompt+48;
    state->cfg_y_proj_scale=100;state->cfg_mouse_look=true;
}

static void capture(GameState *state,RendererPixel *samples,int gun,int tint)
{
    state->cfg_weapon_draw=gun!=0;state->cfg_post_tint=tint!=0;
    renderer_clear_frame_history();g_render_frame_counter=0;renderer_draw_display(state);
    if(samples)for(int i=0;i<SAMPLES;i++)
        samples[i]=renderer_get_cw_buffer()[(size_t)i*g_renderer.width*g_renderer.height/SAMPLES];
}

static void weapon_water_tick(GameState *state,GameLoopCtx *ctx)
{
    const int guns[]={0,1,2,3,4,7};
    int frame=ctx->frame_count;
    if(!frame)place_camera(state);
    state->mode=frame<24?MODE_SINGLE:MODE_SLAVE;
    PlayerState *p=frame<24?&state->plr1:&state->plr2;
    p->gun_selected=guns[(frame/4)%6];p->gun_frame=frame%4;
    int relative=frame<48?-256:frame<56?0:256;
    state->plr1.yoff=state->plr1.oldyoff=state->plr2.yoff=state->plr2.oldyoff=plane-relative;
    capture(state,background,0,0);capture(state,dry_weapon,1,0);capture(state,NULL,1,1);
    require(relative>=0 || renderer_get_last_fill_screen_water()==15,"camera must be submerged");
    const RendererPixel *actual=renderer_get_cw_buffer();
    int changed=0;
    for(int i=0;i<SAMPLES;i++) {
        if(dry_weapon[i]==background[i] || renderer_pixel_without_red(dry_weapon[i])==dry_weapon[i])continue;
        RendererPixel expected=relative<0?renderer_pixel_without_red(dry_weapon[i]):dry_weapon[i];
        require(actual[(size_t)i*g_renderer.width*g_renderer.height/SAMPLES]==expected,"weapon tint differs from original red mask");
        changed++;
    }
    require(changed>0,"source weapon colors must be visible");
    display_draw_display(state);ctx->frame_count++;
    printf("[WEAPONWATER] frame=%d samples=%d height=%d passed\n",frame,changed,relative);
}
