/* Capture the actual SDL/OpenGL composition before presentation. */
#include <SDL.h>
#include "../src/math_tables.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint32_t *captured;
static int captured_width,captured_height;
static SDL_Window *hidden_window(const char *title,int x,int y,int w,int h,Uint32 flags)
{return SDL_CreateWindow(title,x,y,w,h,flags|SDL_WINDOW_HIDDEN);}
static void capture_present(SDL_Renderer *renderer)
{
    assert(SDL_GetRendererOutputSize(renderer,&captured_width,&captured_height)==0);
    assert(captured_width==320 && captured_height==240);
    assert(SDL_RenderReadPixels(renderer,NULL,SDL_PIXELFORMAT_ARGB8888,captured,captured_width*4)==0);
    SDL_RenderPresent(renderer);
}
#define SDL_CreateWindow hidden_window
#define SDL_RenderPresent capture_present
#include "../src/display.c"
#undef SDL_CreateWindow
#undef SDL_RenderPresent

extern void weapon_water_scene_init(void);
extern void weapon_water_scene_height(int relative);
extern void weapon_water_scene_reset(void);
enum { PIXELS=320*240 };
static uint32_t dry_background[PIXELS],dry_weapon[PIXELS],wet_background[PIXELS],actual[PIXELS];
static void capture(uint32_t *pixels,int gun,int tint)
{
    captured=pixels;g_state.cfg_weapon_draw=gun!=0;g_state.cfg_post_tint=tint!=0;
    weapon_water_scene_reset();display_draw_display(&g_state);
}

int main(int argc,char **argv)
{
    setvbuf(stdout,NULL,_IONBF,0);
    int gl=argc>1 && !strcmp(argv[1],"--gl");
#ifdef __EMSCRIPTEN__
    SDL_setenv("AB3D_RENDER_DRIVER","",1);
#else
    SDL_setenv("AB3D_RENDER_DRIVER",gl?"opengl":"software",1);
#endif
    SDL_setenv("AB3D_DISABLE_GL_UNPACK",gl?"0":"1",1);
    SDL_setenv("AB3D_WEB_USE_INI_RES","1",1);
    math_tables_init();game_state_init(&g_state);game_state_setup_default(&g_state);
    g_state.cfg_render_width=320;g_state.cfg_render_height=240;g_state.cfg_supersampling=1;
    g_state.cfg_display_mode=0;g_state.cfg_render_threads=false;g_state.cfg_hud_bars=false;
    display_init(&g_state);assert(g_window && g_sdl_ren);
    if(gl)assert(g_gl_unpack_ok && g_gl_hud_ok);
    else assert(!g_gl_unpack_ok);
    weapon_water_scene_init();display_upload_gun_gl_textures();
    const int guns[]={0,1,2,3,4,7},heights[]={256,0,-256};
    unsigned cases=0,red_samples=0;
    int thread_modes=2;
#ifdef AB3D_NO_THREADS
    thread_modes=1;
#endif
    for(int threaded=0;threaded<thread_modes;threaded++)for(int post_weapon=0;post_weapon<2;post_weapon++)
    for(int player=0;player<2;player++)for(int gun=0;gun<6;gun++)for(int frame=0;frame<4;frame++)
    for(int height=0;height<3;height++) {
        g_state.cfg_render_threads=threaded!=0;g_state.cfg_weapon_post_gl=post_weapon!=0;
        g_state.mode=player?MODE_SLAVE:MODE_SINGLE;
        PlayerState *p=player?&g_state.plr2:&g_state.plr1;p->gun_selected=guns[gun];p->gun_frame=frame;
        weapon_water_scene_height(heights[height]);
        capture(dry_background,0,0);capture(dry_weapon,1,0);
        unsigned red_changes=0,wet_visible=0,hud_red=0;
        for(int i=0;i<PIXELS;i++)red_changes+=((dry_background[i]^dry_weapon[i])&0xff0000)!=0;
        assert(red_changes>0);red_samples+=red_changes;
        capture(wet_background,0,1);capture(actual,1,1);
        if(heights[height]<0) {
            for(int i=0;i<PIXELS;i++) {
                if((actual[i]&0xff0000)!=(wet_background[i]&0xff0000)) {
                    fprintf(stderr,"Display fixture: GL=%d threads=%d GLgun=%d player=%d gun=%d frame=%d height=%d fill=%d pixel=%d,%d background=%08x actual=%08x dry=%08x cw=%04x\n",
                        gl,threaded,post_weapon,player,guns[gun],frame,heights[height],renderer_get_last_fill_screen_water(),i%320,i/320,
                        wet_background[i],actual[i],dry_weapon[i],renderer_pixel_to_c12(renderer_get_cw_buffer()[i]));
                }
                assert((actual[i]&0xff0000)==(wet_background[i]&0xff0000));
                wet_visible+=((actual[i]^wet_background[i])&0xffff)!=0;
                hud_red+=(wet_background[i]&0xff0000)!=0;
            }
            assert(wet_visible>0 && hud_red>0);
        } else for(int i=0;i<PIXELS;i++)assert((actual[i]&0xffffff)==(dry_weapon[i]&0xffffff));
        cases++;
    }
    display_shutdown();renderer_shutdown();SDL_Quit();
    printf("PASS: %s composition, %u cases, %u authored red pixels; weapon tinted, HUD preserved\n",gl?"OpenGL":"SDL software",cases,red_samples);
    return 0;
}
