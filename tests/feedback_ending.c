/* Browser harness for the real death/ending fades and title-menu lifecycle.
 * Link against the web game's objects except main/control_loop/emscripten_loop. */
#include <emscripten.h>
#define main ab3d_normal_main
#include "../src/main.c"
#undef main
#include "../src/control_loop.c"
#include "../src/emscripten_loop.c"

EMSCRIPTEN_KEEPALIVE void feedback_finish_game(void)
{
    g_state.current_level = MAX_LEVELS - 1;
    g_state.finished_level = 1;
    g_state.running = false;
    /* Run the real level cleanup and begin the end music/fade, then advance
     * only its clock. The regular browser callback must return to the menu. */
    g_em_phase = EM_AFTER;
    g_em_outer_sub = 0;
    em_frame();
    em_frame();
    if (g_em_phase == EM_OUTER_POST && g_em_outer_sub == 1)
        s_em_fade_t0 = SDL_GetTicks() - audio_music_duration_ms();
}

EMSCRIPTEN_KEEPALIVE void feedback_die(void)
{
    /* The browser callback must handle the last game frame, cleanup, the
     * complete GameOver track/fade and the title menu without skipping phases. */
    g_state.plr1.energy = 0;
}

EMSCRIPTEN_KEEPALIVE void feedback_save_autosave(void)
{
    player_save_autosave(&g_state);
}

EMSCRIPTEN_KEEPALIVE int feedback_current_level(void)
{
    return g_state.current_level + 1;
}

EMSCRIPTEN_KEEPALIVE int feedback_at_title(void)
{
    return g_em_phase == EM_FRONT_MENU && play_game_front_menu_requested();
}

EMSCRIPTEN_KEEPALIVE int feedback_started_new_game(void)
{
    int start_level = (g_state.cfg_start_level >= 0 && g_state.cfg_start_level < MAX_LEVELS)
                    ? g_state.cfg_start_level : 0;
    return g_em_phase == EM_GAME && g_state.current_level == start_level &&
           !g_state.finished_level && !play_game_front_menu_requested();
}

EMSCRIPTEN_KEEPALIVE int feedback_title_has_continue(void)
{
    return g_em_front_menu_ctx.ingame_menu_autosave_count > 0;
}

int main(void)
{
    SDL_SetMainReady();
    setup_game(&g_state);
    play_game_load_shared_assets(&g_state);
    play_game_apply_front_menu_result(&g_state, GAME_LOOP_FRONT_MENU_NEW_GAME);
    bool copper_ready = false;
    play_the_game_prepare_level(&g_state, &copper_ready);
    game_loop_ctx_init(&g_em_gl_ctx, &g_state);
    g_em_st = &g_state;
    g_em_phase = EM_GAME;
    emscripten_set_main_loop(em_frame, 0, 1);
    return 0;
}
