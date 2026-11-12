/* Use the real setup, New Game and level-loader path around the text mode.
 * Compare with/without the briefing at the same fixed ticks in the host tool. */
#define main ab3d_standard_main
#include "../src/main.c"
#undef main
#include "../src/platform/dos/text_dos.h"

int main(int argc,char **argv)
{
    int show=1,quit_in_text=0;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--without-briefing"))show=0;
        if(!strcmp(argv[i],"--quit-in-briefing"))quit_in_text=1;
    }
    dos_platform_bootstrap(argv[0]);
    if(!ab3d_log_init_file())return 2;
    dos_platform_init(argc,argv);
#ifdef AB3D_DOS_INDEXED
    dos_palette_load(-1);
#endif
    setup_game(&g_state);
    play_game_load_shared_assets(&g_state);
    if(!play_game_apply_front_menu_result(&g_state,GAME_LOOP_FRONT_MENU_NEW_GAME) ||
       !play_the_game_should_show_level_text(&g_state))return 3;
    if(show) {
        play_the_game_present_level_text(&g_state,255);
        if(!dos_text_active())return 4;
        dos_capture_frame("LOADING.BMP");
    }
    bool ready=false;
    play_the_game_prepare_level(&g_state,&ready);
    dos_memory_report("briefing level loaded");
    if(show) {
        if(!dos_text_active())return 5;
        dos_capture_frame("LOADED.BMP");
        if(quit_in_text) {
            play_the_game_finalize_session(&g_state);
            if(dos_text_active())return 6;
            tear_down_game(&g_state);
            printf("[BRIEFINGCHECK] quit restored graphics and released the loaded level\n");
            ab3d_log_shutdown();return 0;
        }
        if(play_the_game_drain_level_text_input(&g_state)<0 ||
           !Ab3dDos_PushKey(SDL_SCANCODE_RETURN,true) ||
           play_the_game_poll_level_text_input(&g_state)!=1 ||
           !Ab3dDos_PushKey(SDL_SCANCODE_RETURN,false))return 7;
        play_the_game_clear_level_text(&g_state);
        if(dos_text_active())return 8;
        display_present_text_screen_alpha(0);
    }
    GameLoopCtx ctx;game_loop_ctx_init(&ctx,&g_state);srand(1);
    audio_dos_reset_stats();
    for(int frame=0;frame<4;frame++)game_loop_tick(&g_state,&ctx);
    dos_capture_frame("FRAME.BMP");
    printf("[BRIEFINGCHECK] level=%d draws=%d show=%d; loading text and gameplay mode restored\n",
           g_state.current_level+1,ctx.frame_count,show);
    audio_dos_report_stats();
    play_the_game_finalize_session(&g_state);
    tear_down_game(&g_state);ab3d_log_shutdown();return 0;
}
