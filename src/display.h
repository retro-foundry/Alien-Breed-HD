/*
 * Alien Breed 3D I - PC Port
 * display.h - SDL2 display frontend
 *
 * Provides the SDL2-backed replacement for the original Amiga display
 * system: copper/list setup, chunky framebuffer management, texture
 * uploads, HUD drawing, etc.
 */

#ifndef DISPLAY_H
#define DISPLAY_H

#include "game_state.h"

/* Lifecycle (pass state for ab3d.ini render_width/render_height/supersampling; may be NULL -> 1280x720 at 1x) */
void display_init(GameState *state);
void display_shutdown(void);
int display_is_fullscreen(void);  /* non-zero if window is fullscreen */
void display_toggle_fullscreen(void); /* borderless ini fullscreen or windowed <-> desktop fullscreen */
/* Emscripten: poll after canvas fullscreen (no-op on native). */
void display_emscripten_frame_resize_poll(void);
void display_on_resize(int w, int h);  /* call on window resize to resize framebuffer */
void display_handle_resize(void);      /* query renderer output size and resize (use on SDL_WINDOWEVENT_RESIZED) */

/* Screen management */
void display_alloc_text_screen(void);
void display_release_text_screen(void);
void display_alloc_copper_screen(void);
void display_release_copper_screen(void);
void display_alloc_title_memory(void);
void display_release_title_memory(void);
void display_alloc_panel_memory(void);
void display_release_panel_memory(void);

/* Title screen */
void display_setup_title_screen(void);
void display_load_title_screen(void);
void display_clear_opt_screen(void);
void display_draw_opt_screen(int screen_num);
void display_fade_up_title(int amount);
void display_fade_down_title(int amount);
void display_clear_title_palette(void);

/* Gun GL textures: decode all 32 gun frame slots and upload to GPU once.
 * Call after io_load_gun_graphics().  Safe to call multiple times (re-uploads). */
void display_upload_gun_gl_textures(void);

/* In-game rendering */
void display_init_copper_screen(void);
void display_draw_display(GameState *state);  /* Renders 3D scene; presents 12-bit cw framebuffer */
void display_present_last_frame(GameState *state); /* Presents the last rendered frame without re-rendering/swap */
void display_swap_buffers(void);
void display_wait_vblank(void);
void display_set_screen_tint(int r, int g, int b, int alpha);
void display_clear_screen_tint(void);

/* HUD */
void display_energy_bar(int16_t energy);
void display_ammo_bar(int16_t ammo);

/* Text */
void display_draw_line_of_text(const char *text, int line);
void display_clear_text_screen(void);

#endif /* DISPLAY_H */
