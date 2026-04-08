/*
 * Alien Breed 3D I - PC Port
 * game_state.c - Global game state initialization
 */

#include "game_state.h"
#include "game_data.h"
#include <string.h>

/* Global game state instance */
GameState g_state;

/* Use the end_zones table from game_data.c */

static void game_state_init_single_player_loadout(PlayerState *plr)
{
    memset(plr->gun_data, 0, sizeof(plr->gun_data));

    /* Amiga ControlLoop.s SetupDefaultGame:
     * pistol visible with 160 internal ammo, other main guns hidden/empty. */
    plr->gun_data[0].visible = -1;
    plr->gun_data[0].ammo = 160;
    plr->gun_selected = 0;
}

void game_state_init_player(PlayerState *plr)
{
    memset(plr, 0, sizeof(*plr));
    plr->energy = PLAYER_MAX_ENERGY;
    plr->height = PLAYER_HEIGHT;
    plr->s_height = PLAYER_HEIGHT;
    plr->s_targheight = PLAYER_HEIGHT;
    plr->no_transition_back_roompt = -1;
}

void game_state_init(GameState *state)
{
    memset(state, 0, sizeof(*state));

    state->mode = MODE_SINGLE;
    state->mp_mode = 0;
    state->running = true;

    game_state_init_player(&state->plr1);
    game_state_init_player(&state->plr2);

    /* Default control: mouse+kbd for both players */
    state->plr1_control.mouse_kbd = true;
    state->plr2_control.mouse_kbd = true;

    /* Default prefs (from AB3DI.s: 'k4nxs') */
    strncpy(state->prefs_file, "k4nxs", sizeof(state->prefs_file));

    /* Copy end zones table */
    memcpy(state->end_zones, end_zones, sizeof(state->end_zones));

    /* Level */
    state->current_level = 0;
    state->max_level = 0;
    state->finished_level = 0;

    state->do_anything = true;
    state->nasty = true; /* enemies active in single player */

    state->cfg_start_level = -1;
    state->infinite_health = false;
    state->infinite_ammo = false;
    state->cfg_all_weapons = true;
    state->cfg_all_keys = false;
    state->cfg_display_mode = -1;
    state->cfg_render_width = 1280;
    state->cfg_render_height = 720;
    state->cfg_supersampling = 1;
    state->cfg_render_threads = true;
    state->cfg_render_threads_max = 0;
    state->cfg_volume = 100;
    state->cfg_y_proj_scale = 100;
    state->cfg_billboard_sprite_rendering_enhancement = true;
    state->cfg_show_fps = false;
    state->fps_display = 0;
    state->cfg_weapon_draw = true;
    state->cfg_post_tint = true;
    state->cfg_weapon_post_gl = false;
}

/*
 * SetupDefaultGame equivalent (from ControlLoop.s)
 * Called for single player and coop mode.
 */
void game_state_setup_default(GameState *state)
{
    game_state_init_player(&state->plr1);
    game_state_init_player(&state->plr2);
    game_state_init_single_player_loadout(&state->plr1);
    game_state_init_single_player_loadout(&state->plr2);
    state->nasty = true;
}

/*
 * SetupTwoPlayerGame equivalent
 * Called for versus multiplayer.
 */
void game_state_setup_two_player(GameState *state)
{
    game_state_init_player(&state->plr1);
    game_state_init_player(&state->plr2);
    state->nasty = false;  /* no enemies in versus mode */
}
