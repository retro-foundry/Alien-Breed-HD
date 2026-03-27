/*
 * Alien Breed 3D I - PC Port
 * player.h - Player control and state management
 *
 * Translates from: Plr1Control.s, Plr2Control.s, PlayerShoot.s
 *
 * The original code has multiple control methods:
 *   - Mouse (PLR1_mouse_control)
 *   - Mouse+KBD (PLR1_mousekbd_control)
 *   - Keyboard only (PLR1_keyboard_control)
 *   - Joystick (PLR1_JoyStick_control)
 *   - Follow path (PLR1_follow_path) - debug/demo
 *
 * All are stubbed for now, but the structure mirrors the original dispatch.
 */

#ifndef PLAYER_H
#define PLAYER_H

#include "game_state.h"

/* Top-level control dispatch (from AB3DI.s PLR1_Control / PLR2_Control) */
void player1_control(GameState *state);
void player2_control(GameState *state);

/* Init player positions from level data (from LevelData2.s InitPlayer) */
void player_init_from_level(GameState *state);

/* Shooting (from PlayerShoot.s) */
void player1_shoot(GameState *state);
void player2_shoot(GameState *state);

/* Copy player state to per-frame snapshot (from mainLoop in AB3DI.s) */
void player1_snapshot(GameState *state);
void player2_snapshot(GameState *state);

/* Debug: save current player position/orientation to data/debug_save.bin (F5). */
void player_debug_save_position(GameState *state);

/* Load F5 save file into live player state (F9; also used on level init in debug builds). */
bool player_debug_load_save_from_file(GameState *state);

/* If debug_save.bin contains a saved level, set current_level before level load (restart). */
void player_debug_apply_saved_level_before_load(GameState *state);

#endif /* PLAYER_H */
