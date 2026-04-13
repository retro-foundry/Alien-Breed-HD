#ifndef EMSCRIPTEN_LOOP_H
#define EMSCRIPTEN_LOOP_H

#include "game_state.h"

/* Browser build: load assets, then drive play_game/play_the_game/game_loop via
 * emscripten_set_main_loop (see emscripten_loop.c). */
void emscripten_run_game(GameState *state);

#endif
