/* Exercise the real control_loop.c outcome callbacks and native presentation.
 * AB3DI.s playWellDone/playGameOver supplies the music lifecycle; the tint
 * colors and timing are the existing PC port's behavior. */
#define main ab3d_standard_main
#include "../src/main.c"
#undef main
#include "../src/control_loop.c"

static uint32_t world_hash(void)
{
    uint32_t hash=2166136261u;
    const unsigned char *pixels=(const unsigned char *)renderer_get_cw_buffer();
    size_t size=(size_t)renderer_get_width()*renderer_get_height()*sizeof(RendererPixel);
    for(size_t i=0;i<size;i++)hash=(hash^pixels[i])*16777619u;
    return hash;
}

int main(int argc,char **argv)
{
    int death=0;
    for(int i=1;i<argc;i++)if(!strcmp(argv[i],"--death"))death=1;
    dos_platform_bootstrap(argv[0]);
    if(!ab3d_log_init_file())return 2;
    dos_platform_init(argc,argv);
#ifdef AB3D_DOS_INDEXED
    dos_palette_load(-1);
#endif
    setup_game(&g_state);
    play_game_load_shared_assets(&g_state);
    play_game_apply_front_menu_result(&g_state,GAME_LOOP_FRONT_MENU_NEW_GAME);
    bool ready=false;
    play_the_game_prepare_level(&g_state,&ready);
    GameLoopCtx ctx;game_loop_ctx_init(&ctx,&g_state);srand(1);
    for(int frame=0;frame<4;frame++)game_loop_tick(&g_state,&ctx);
    audio_mt_end();
    uint32_t hash=world_hash();
#ifdef AB3D_DOS_INDEXED
    FILE *palette=fopen("PALETTE.BIN","wb");
    if(!palette)return 3;
    if(fwrite(dos_c12_to_index,1,sizeof(dos_c12_to_index),palette)!=sizeof(dos_c12_to_index) ||
       fwrite(dos_index_to_c12,1,sizeof(dos_index_to_c12),palette)!=sizeof(dos_index_to_c12))return 4;
    if(fclose(palette))return 5;
#endif
    const float progress[]={0.0f,0.25f,0.5f,1.0f,0.5f};
    const char *names[]={"WHITE0.BMP","WHITE1.BMP","WHITE2.BMP","WHITE3.BMP","WHITEREP.BMP"};
    for(unsigned i=0;i<sizeof(progress)/sizeof(progress[0]);i++) {
        control_level_complete_fade_tick(progress[i],&g_state);
        dos_capture_frame(names[i]);
        if(world_hash()!=hash)return 6;
    }
    control_game_over_fade_tick(0.5f,&g_state);dos_capture_frame("REDMID.BMP");
    control_game_over_fade_tick(1.0f,&g_state);dos_capture_frame("REDEND.BMP");
    control_level_complete_fade_tick(1.0f,&g_state);dos_capture_frame("WHITEEND.BMP");
    display_clear_screen_tint();
    display_present_last_frame(NULL);dos_capture_frame("WHITECLR.BMP");
    if(world_hash()!=hash)return 7;

    int level=g_state.current_level;
    g_state.finished_level=death?0:1;
    if(death)g_state.plr1.energy=0;
    if(play_the_game_after_game_loop(&g_state))return 8;
    play_the_game_finalize_session(&g_state);
    audio_dos_reset_stats();
    int again=play_game_outer_should_continue(&g_state);
    audio_dos_report_stats();
    display_present_last_frame(NULL);dos_capture_frame("AFTER.BMP");
    if(world_hash()!=hash)return 9;
    if(death) {
        if(!again || !play_game_front_menu_requested())return 10;
    } else if(level==MAX_LEVELS-1) {
        if(again)return 11;
    } else {
        if(!again || g_state.current_level!=level+1 || !s_show_transition_level_text)return 12;
        play_the_game_prepare_level(&g_state,&ready);
        game_loop_ctx_init(&ctx,&g_state);
        game_loop_tick(&g_state,&ctx);
        dos_capture_frame("NEXT.BMP");
    }
    printf("[FADECHECK] callbacks preserve world pixels; outcome=%s level=%d; tint cleared\n",
           death?"death":level==MAX_LEVELS-1?"final":"next",level+1);
    play_the_game_finalize_session(&g_state);
    tear_down_game(&g_state);ab3d_log_shutdown();return 0;
}
