/*
 * Alien Breed 3D I - PC Port
 * input.h - SDL2 input backend
 *
 * Implements keyboard/mouse/joystick handling via SDL2 for the PC port.
 */

#ifndef INPUT_H
#define INPUT_H

#include "game_types.h"
#include <stdint.h>
#include <stdbool.h>

/* Lifecycle */
void input_init(void);
void input_shutdown(void);

/* Per-frame polling */
void input_update(uint8_t *key_map, uint8_t *last_pressed);

/* Mouse */
typedef struct {
    int16_t dx;
    int16_t dy;
    int16_t wheel_y;
    bool    left_button;
    bool    right_button;
} MouseState;

void input_read_mouse(MouseState *out);

/* Joystick */
typedef struct {
    int16_t dx;
    int16_t dy;
    bool    fire;
} JoyState;

void input_read_joy1(JoyState *out);
void input_read_joy2(JoyState *out);

/* Convenience: check if a specific key is pressed */
bool input_key_pressed(const uint8_t *key_map, uint8_t keycode);

/* Clear keyboard state */
void input_clear_keyboard(uint8_t *key_map);

/* F5 save current game state. Returns true once when F5 was pressed (caller should save). */
bool input_f5_save_requested(void);
/* F9 load savegame.bin (same file as F5). Returns true once when F9 was pressed. */
bool input_f9_load_requested(void);
bool input_f6_gouraud_visualize_requested(void);
bool input_f7_spill_visualize_requested(void);
/* F2: log center-pick debug info (once per key press). */
bool input_f2_pick_log_requested(void);
/* Tab: toggle automap overlay (once per key press). */
bool input_automap_toggle_requested(void);
/* PgUp/PgDn: automap zoom (once per key press). */
bool input_automap_pgup_requested(void);
bool input_automap_pgdn_requested(void);
/* F12 (and F11): toggle fullscreen desktop (once per key press). */
bool input_fullscreen_toggle_requested(void);

#endif /* INPUT_H */
