/* Real New Game, briefing, level loader and player control. Only keyboard
 * navigation is injected; all camera motion and clicks come through INT 33h. */
#define main ab3d_standard_main
#include "../src/main.c"
#undef main
#include <assert.h>

static int front_key(GameLoopCtx *ctx,SDL_Scancode key)
{
    assert(Ab3dDos_PushKey(key,true));
    int result=game_loop_front_menu_tick(&g_state,ctx);
    assert(Ab3dDos_PushKey(key,false));
    game_loop_front_menu_tick(&g_state,ctx);
    return result;
}

static void game_key(GameLoopCtx *ctx,SDL_Scancode key)
{
    assert(Ab3dDos_PushKey(key,true));game_loop_tick(&g_state,ctx);
    assert(Ab3dDos_PushKey(key,false));game_loop_tick(&g_state,ctx);
}

static void physical_input(GameLoopCtx *ctx,int phase)
{
    unsigned directions=0;
    printf("READY: game mouse axes %d\n",phase);fflush(NULL);
    Uint64 deadline=SDL_GetTicks()+10000;
    while(directions!=15 && SDL_GetTicks()<deadline) {
        int angle=g_state.plr1.s_angpos,look=g_state.plr1.s_look_y;
        game_loop_tick(&g_state,ctx);
        int turn=(g_state.plr1.s_angpos-angle)&ANGLE_MASK;
        if(turn>ANGLE_MASK/2)turn-=ANGLE_MASK+1;
        if(turn>0)directions|=1;if(turn<0)directions|=2;
        if(g_state.plr1.s_look_y>look)directions|=4;
        if(g_state.plr1.s_look_y<look)directions|=8;
    }
    printf("[MOUSEGAME] phase=%d camera directions=%u\n",phase,directions);fflush(NULL);
    assert(directions==15);
    /* input.c AMIGA_KEY_RALT/SPACE: the actual player fire/operate bindings.
     * A released tap during a drawing gap must remain for one control tick. */
    const unsigned keys[]={0x65,0x40};
    for(unsigned button=0;button<2;button++) {
        input_consume_mouse_deltas();
        printf("READY: game mouse button %d %u\n",phase,button);fflush(NULL);
        deadline=SDL_GetTicks()+1500;
        while(SDL_GetTicks()<deadline){audio_dos_pump();SDL_Delay(5);}
        input_update(g_state.key_map,&g_state.last_pressed);
        MouseState mouse;input_read_mouse(&mouse);
        printf("[MOUSEGAME] phase=%d button=%u key=%d latched=%d/%d\n",phase,button,
            g_state.key_map[keys[button]],mouse.left_button,mouse.right_button);fflush(NULL);
        assert(g_state.key_map[keys[button]] && (button?mouse.right_button:mouse.left_button));
        assert(!(SDL_GetMouseState(NULL,NULL)&(SDL_BUTTON(SDL_BUTTON_LEFT)|SDL_BUTTON(SDL_BUTTON_RIGHT))));
        input_consume_mouse_deltas();input_update(g_state.key_map,&g_state.last_pressed);
        input_read_mouse(&mouse);
        assert(!g_state.key_map[keys[button]] && !mouse.left_button && !mouse.right_button);
    }
    printf("[MOUSEGAME] phase=%d real host fire/operate taps passed\n",phase);fflush(NULL);
}

int main(int argc,char **argv)
{
    dos_platform_bootstrap(argv[0]);assert(ab3d_log_init_file());
    dos_platform_init(argc,argv);
#ifdef AB3D_DOS_INDEXED
    dos_palette_load(-1);
#endif
    setup_game(&g_state);play_game_load_shared_assets(&g_state);
    GameLoopCtx ctx;game_loop_front_menu_init(&ctx,&g_state);
    front_key(&ctx,SDL_SCANCODE_RETURN);front_key(&ctx,SDL_SCANCODE_UP);
    assert(front_key(&ctx,SDL_SCANCODE_RETURN)==GAME_LOOP_FRONT_MENU_NEW_GAME);
    assert(play_game_apply_front_menu_result(&g_state,GAME_LOOP_FRONT_MENU_NEW_GAME));
    assert(play_the_game_should_show_level_text(&g_state));
    play_the_game_present_level_text(&g_state,255);
    bool ready=false;play_the_game_prepare_level(&g_state,&ready);
    assert(play_the_game_drain_level_text_input(&g_state)==0);
    assert(Ab3dDos_PushKey(SDL_SCANCODE_RETURN,true));
    assert(play_the_game_poll_level_text_input(&g_state)==1);
    assert(Ab3dDos_PushKey(SDL_SCANCODE_RETURN,false));play_the_game_clear_level_text(&g_state);
    game_loop_ctx_init(&ctx,&g_state);game_loop_tick(&g_state,&ctx);
    bool disabled=getenv("AB3D_TEST_MOUSE_DISABLED")!=NULL;
    printf("[MOUSEGAME] capture after briefing=%d expected=%d\n",SDL_GetRelativeMouseMode(),!disabled);fflush(NULL);
    assert(SDL_GetRelativeMouseMode()==!disabled);
    if(getenv("AB3D_TEST_MOUSE_LIVE"))physical_input(&ctx,0);
    game_key(&ctx,SDL_SCANCODE_ESCAPE);assert(ctx.ingame_menu_open && !SDL_GetRelativeMouseMode());
    game_key(&ctx,SDL_SCANCODE_RETURN);assert(!ctx.ingame_menu_open);
    assert(SDL_GetRelativeMouseMode()==!disabled);
    if(getenv("AB3D_TEST_MOUSE_LIVE"))physical_input(&ctx,1);
    dos_capture_frame("FRAME.BMP");
    play_the_game_finalize_session(&g_state);tear_down_game(&g_state);
    printf("[MOUSEGAME] PASS real New Game/briefing/menu capture lifecycle\n");
    ab3d_log_shutdown();return 0;
}
