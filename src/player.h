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

/* Debug: save full game + level runtime state to data/debug_save.bin (F5). */
void player_debug_save_position(GameState *state);

typedef enum {
    PLAYER_DEBUG_LOAD_FAILED = 0,
    PLAYER_DEBUG_LOAD_APPLIED,
    PLAYER_DEBUG_LOAD_NEED_LEVEL_RELOAD
} PlayerDebugLoadResult;

/* Read debug_save.bin. Full saves stage a pending restore and request level reload. */
PlayerDebugLoadResult player_debug_load_save_from_file(GameState *state);

/* After level reload, apply the pending full restore (or legacy position payload). */
void player_debug_apply_save_payload_after_level_load(GameState *state);

#endif /* PLAYER_H */
