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
/* Emscripten: poll after canvas/window resize or fullscreen (no-op on native). */
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
void display_present_title_screen_alpha(int text_alpha, int dim_alpha);

/* Gun GL textures: decode all 32 gun frame slots and upload to GPU once.
 * Call after io_load_gun_graphics().  Safe to call multiple times (re-uploads). */
void display_upload_gun_gl_textures(void);

/* In-game rendering */
void display_init_copper_screen(void);
void display_draw_display(GameState *state);  /* Renders 3D scene; presents 12-bit cw framebuffer */
void display_present_last_frame(GameState *state); /* Presents the last rendered frame without re-rendering/swap */
void display_suppress_next_present(void); /* Render/upload the next frame without showing it */
void display_present_text_screen(void); /* Presents the retained text screen without a 3D frame */
void display_present_text_screen_alpha(int alpha); /* Same as above with text alpha 0..255 */
void display_swap_buffers(void);
void display_wait_vblank(void);
void display_set_screen_tint(int r, int g, int b, int alpha);
void display_clear_screen_tint(void);

/* Text */
void display_draw_line_of_text(const char *text, int line);
void display_draw_line_of_text_alpha(const char *text, int line, int alpha);
void display_clear_text_screen(void);
#ifdef AB3D_DOS
void display_text_reset_pages(void);
int display_text_next_page(void);
void display_dos_check_text_layouts(void);
void display_dos_check_text_screen(const char *label, int allow_pages);
void display_dos_check_hud_bounds(GameState *state);
#endif
int display_text_line_at_point(int window_x, int window_y, int title_screen,
                               int *out_line);

#endif /* DISPLAY_H */
