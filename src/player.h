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
 *   - Joystick (PLR1_JoyStick_control; mapped through the SDL gamepad input path)
 *   - Follow path (PLR1_follow_path) - debug/demo
 *
 * The port keeps this dispatch shape while routing implemented controls through
 * player.c and the SDL input layer.
 */

#ifndef PLAYER_H
#define PLAYER_H

#include "game_state.h"

/* Top-level control dispatch (from AB3DI.s PLR1_Control / PLR2_Control) */
void player1_control(GameState *state);
void player2_control(GameState *state);

/* Init player positions from level data (from LevelData2.s InitPlayer) */
void player_init_from_level(GameState *state);

/* Clear modern mouse-look view/aim state when mouse look is disabled. */
void player_clear_mouse_look_aim_state(GameState *state);

/* Apply ab3d.ini weapon availability toggles to player inventory. */
void player_apply_weapon_config(GameState *state);

/* Inventory ammo for a valid gun index. The starting rifle and PC
 * flamethrower share the rifle's counter, in the original eight units/round. */
int16_t *player_weapon_ammo_ptr(PlayerState *plr, int gun_idx);

/* Shooting (from PlayerShoot.s) */
void player1_shoot(GameState *state);
void player2_shoot(GameState *state);

/* Copy player state to per-frame snapshot (from mainLoop in AB3DI.s) */
void player1_snapshot(GameState *state);
void player2_snapshot(GameState *state);

/* Save full game + level runtime state to savegame.bin beside the executable. */
void player_save_position(GameState *state);

#define PLAYER_AUTOSAVE_SLOT_COUNT 5

typedef enum {
    PLAYER_SAVE_LOAD_FAILED = 0,
    PLAYER_SAVE_LOAD_APPLIED,
    PLAYER_SAVE_LOAD_NEED_LEVEL_RELOAD
} PlayerSaveLoadResult;

typedef struct {
    bool    present;
    int16_t level;
    char    timestamp[32];
} PlayerAutosaveInfo;

/* Read savegame.bin. Full saves stage a pending restore and request level reload. */
PlayerSaveLoadResult player_load_save_from_file(GameState *state);

/* Autosave is separate from F5/F9 quicksave; slot 0 uses autosave.bin. */
void player_save_autosave(GameState *state);
PlayerSaveLoadResult player_load_autosave_from_file(GameState *state, int slot);
bool player_read_autosave_info(int slot, PlayerAutosaveInfo *info);

/* After level reload, apply the pending full restore (or legacy position payload). */
void player_apply_save_payload_after_level_load(GameState *state);

#endif /* PLAYER_H */
