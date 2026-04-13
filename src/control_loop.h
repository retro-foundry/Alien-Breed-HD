/*
 * Alien Breed 3D I - PC Port
 * control_loop.h - Outer control loop (menu, level management)
 *
 * Translated from: ControlLoop.s
 *
 * This is the outer loop that:
 * 1. Shows title screen / plays title music
 * 2. Loads shared assets (walls, floor, objects, SFX)
 * 3. Presents menu and handles option selection
 * 4. Loads levels and launches PlayTheGame
 * 5. Returns to menu after death/level complete
 */

#ifndef CONTROL_LOOP_H
#define CONTROL_LOOP_H

#include "game_state.h"

#include <stdbool.h>

/* Shared asset load used by play_game and the Emscripten browser driver. */
void play_game_load_shared_assets(GameState *state);

/* One inner iteration of PlayTheGame (level load + conditions); used by Emscripten main loop. */
void play_the_game_prepare_level(GameState *state, bool *copper_screen_ready);

/* After game_loop returns: 1 = F9 reload (caller runs another prepare), 0 = level session done. */
int play_the_game_after_game_loop(GameState *state);

/* Cleanup at end of play_the_game (copper screen, level mem, flags). */
void play_the_game_finalize_session(GameState *state);

/* After a full play_the_game: 1 = continue play_game outer loop, 0 = exit to panel release. */
#if !defined(__EMSCRIPTEN__)
int play_game_outer_should_continue(GameState *state);
#else
/* Browser: outer post is split across rAF frames (music + fade cannot block one callback). */
int play_game_outer_emscripten_begin(GameState *state);
int play_game_outer_emscripten_fade_frame(GameState *state);
int play_game_outer_emscripten_finish(GameState *state);
#endif

/* PlayGame - the top-level game loop (ControlLoop.s: PlayGame) */
void play_game(GameState *state);

/* PlayTheGame - level gameplay (AB3DI.s: PlayTheGame) */
void play_the_game(GameState *state);

/* Menu */
int  read_main_menu(GameState *state);

/* Password system (ControlLoop.s CalcPassword, PassLineToGame, GetStats) */
void calc_password(GameState *state);
int  pass_line_to_game(GameState *state, const char *password);

#endif /* CONTROL_LOOP_H */
