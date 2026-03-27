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
    state->cfg_render_width = 1920;
    state->cfg_render_height = 1080;
}

/*
 * SetupDefaultGame equivalent (from ControlLoop.s)
 * Called for single player and coop mode.
 */
void game_state_setup_default(GameState *state)
{
    game_state_init_player(&state->plr1);
    game_state_init_player(&state->plr2);
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
