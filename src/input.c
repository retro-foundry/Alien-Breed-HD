/*
 * Alien Breed 3D I - PC Port
 * input.c - SDL2 input backend
 *
 * Maps SDL2 keyboard scancodes to Amiga raw keycodes that the
 * game's key_map[] array expects.
 */

#include "input.h"
#include "display.h"
#include <SDL.h>
#include <stdio.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Amiga rawkey codes used by the game (from player.c default_keys
 * and game_loop.c KEY_ESC, ControlLoop.s).
 * ----------------------------------------------------------------------- */
#define AMIGA_KEY_ESC       0x45
#define AMIGA_KEY_SPACE     0x40
#define AMIGA_KEY_UP        0x4C
#define AMIGA_KEY_DOWN      0x4D
#define AMIGA_KEY_LEFT      0x4F
#define AMIGA_KEY_RIGHT     0x4E
#define AMIGA_KEY_RSHIFT    0x61
#define AMIGA_KEY_RALT      0x65
#define AMIGA_KEY_RAMIGA    0x67
#define AMIGA_KEY_D         0x22
#define AMIGA_KEY_L         0x28
#define AMIGA_KEY_PERIOD    0x39   /* . > */
#define AMIGA_KEY_SLASH     0x3A   /* / ? */
#define AMIGA_KEY_P         0x19   /* Pause */
#define AMIGA_KEY_1         0x01
#define AMIGA_KEY_2         0x02
#define AMIGA_KEY_3         0x03
#define AMIGA_KEY_4         0x04
#define AMIGA_KEY_5         0x05
#define AMIGA_KEY_6         0x06
#define AMIGA_KEY_7         0x07
#define AMIGA_KEY_8         0x08

/* -----------------------------------------------------------------------
 * SDL scancode -> Amiga rawkey mapping
 * ----------------------------------------------------------------------- */
static uint8_t sdl_to_amiga(SDL_Scancode sc)
{
    switch (sc) {
    case SDL_SCANCODE_ESCAPE:   return AMIGA_KEY_ESC;
    case SDL_SCANCODE_SPACE:    return AMIGA_KEY_SPACE;
    case SDL_SCANCODE_UP:       return AMIGA_KEY_UP;
    case SDL_SCANCODE_W:        return AMIGA_KEY_UP;       /* WASD: W = forward */
    case SDL_SCANCODE_DOWN:     return AMIGA_KEY_DOWN;
    case SDL_SCANCODE_S:        return AMIGA_KEY_DOWN;     /* WASD: S = backward */
    case SDL_SCANCODE_LEFT:     return AMIGA_KEY_LEFT;
    case SDL_SCANCODE_RIGHT:    return AMIGA_KEY_RIGHT;
    case SDL_SCANCODE_A:        return AMIGA_KEY_PERIOD;   /* WASD: A = strafe left */
    case SDL_SCANCODE_D:        return AMIGA_KEY_SLASH;    /* WASD: D = strafe right -- override duck */
    case SDL_SCANCODE_LSHIFT:   return AMIGA_KEY_RSHIFT;   /* Shift = run */
    case SDL_SCANCODE_RSHIFT:   return AMIGA_KEY_RSHIFT;
    case SDL_SCANCODE_LALT:     return AMIGA_KEY_RALT;     /* Alt = fire */
    case SDL_SCANCODE_RALT:     return AMIGA_KEY_RALT;
    case SDL_SCANCODE_LCTRL:    return AMIGA_KEY_RALT;     /* Ctrl = fire too */
    case SDL_SCANCODE_RCTRL:    return AMIGA_KEY_RALT;
    case SDL_SCANCODE_PERIOD:   return AMIGA_KEY_PERIOD;   /* . = strafe left */
    case SDL_SCANCODE_SLASH:    return AMIGA_KEY_SLASH;    /* / = strafe right */
    case SDL_SCANCODE_C:        return AMIGA_KEY_D;        /* C = duck (original D) */
    case SDL_SCANCODE_L:        return AMIGA_KEY_L;        /* L = look behind */
    case SDL_SCANCODE_P:        return AMIGA_KEY_P;        /* P = pause */
    case SDL_SCANCODE_1:        return AMIGA_KEY_1;
    case SDL_SCANCODE_2:        return AMIGA_KEY_2;
    case SDL_SCANCODE_3:        return AMIGA_KEY_3;
    case SDL_SCANCODE_4:        return AMIGA_KEY_4;
    case SDL_SCANCODE_5:        return AMIGA_KEY_5;
    case SDL_SCANCODE_6:        return AMIGA_KEY_6;
    case SDL_SCANCODE_7:        return AMIGA_KEY_7;
    case SDL_SCANCODE_8:        return AMIGA_KEY_8;
    case SDL_SCANCODE_KP_1:     return AMIGA_KEY_1;
    case SDL_SCANCODE_KP_2:     return AMIGA_KEY_2;
    case SDL_SCANCODE_KP_3:     return AMIGA_KEY_3;
    case SDL_SCANCODE_KP_4:     return AMIGA_KEY_4;
    case SDL_SCANCODE_KP_5:     return AMIGA_KEY_5;
    case SDL_SCANCODE_KP_6:     return AMIGA_KEY_6;
    case SDL_SCANCODE_KP_7:     return AMIGA_KEY_7;
    case SDL_SCANCODE_KP_8:     return AMIGA_KEY_8;
    default:                    return 0xFF; /* unmapped */
    }
}

/* -----------------------------------------------------------------------
 * Mouse state
 * ----------------------------------------------------------------------- */
static MouseState g_mouse = {0};
static bool g_quit_requested = false;
static bool g_f5_save_requested = false;
static bool g_f9_load_requested = false;
static bool g_f6_gouraud_visualize_requested = false;
static bool g_f7_spill_visualize_requested = false;
static bool g_f2_pick_log_requested = false;
static bool g_automap_toggle_requested = false;
static bool g_automap_pgup_requested = false;
static bool g_automap_pgdn_requested = false;
static bool g_fullscreen_toggle_requested = false;

/* True after we successfully enable relative mode (click-to-play). The browser may
 * exit pointer lock before SDL_KEYDOWN(Escape) is delivered, so SDL_GetRelativeMouseMode()
 * can already be false when handling Esc — we still must clear keys and show the cursor. */
static SDL_bool g_input_capture_active = SDL_FALSE;

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
EM_JS(void, input_emscripten_canvas_cursor_default, (void), {
    var c = Module['canvas'];
    if (c) c.style.cursor = 'default';
});
/* Browser exits pointer lock on Esc before SDL may see KEYDOWN; sync release here. */
EM_JS(void, input_emscripten_install_pointer_lock_listener, (void), {
    Module['ab3dPointerLockLost'] = 0;
    document.addEventListener('pointerlockchange', function() {
        if (!document.pointerLockElement && Module['canvas']) {
            Module['ab3dPointerLockLost'] = 1;
        }
    });
});
#else
static void input_emscripten_canvas_cursor_default(void) { }
static void input_emscripten_install_pointer_lock_listener(void) { }
#endif

/* Unwind SDL's cursor hide stack (ShowCursor is reference-counted). */
static void input_force_cursor_visible(void)
{
    int n = 0;
    while (SDL_ShowCursor(SDL_QUERY) == SDL_DISABLE && n++ < 16) {
        SDL_ShowCursor(SDL_ENABLE);
    }
}

/* Relative mode hides the OS cursor; when not captured, keep the pointer visible.
 * Releasing capture clears mouse state and the full key_map (if provided) so movement
 * and fire keys do not stay latched after Esc or focus loss. */
static void input_apply_relative_mouse(SDL_bool want_capture, uint8_t *key_map)
{
    if (SDL_SetRelativeMouseMode(want_capture) != 0 && want_capture) {
        static int logged_fail = 0;
        if (!logged_fail) {
            printf("[INPUT] SDL_SetRelativeMouseMode(1) failed: %s\n", SDL_GetError());
            logged_fail = 1;
        }
        want_capture = SDL_FALSE;
    }
    if (!want_capture) {
        g_mouse.left_button = false;
        g_mouse.right_button = false;
        g_mouse.wheel_y = 0;
        g_mouse.dx = 0;
        g_mouse.dy = 0;
        if (key_map) {
            memset(key_map, 0, 128);
        }
        SDL_CaptureMouse(SDL_FALSE);
        g_input_capture_active = SDL_FALSE;
        SDL_ShowCursor(SDL_ENABLE);
        input_force_cursor_visible();
        input_emscripten_canvas_cursor_default();
    } else {
        g_input_capture_active = SDL_TRUE;
        SDL_ShowCursor(SDL_DISABLE);
    }
}

/* -----------------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------------- */
void input_init(void)
{
    printf("[INPUT] SDL2 input init\n");
    g_quit_requested = false;
    g_f7_spill_visualize_requested = false;
    g_f2_pick_log_requested = false;
    /* Pointer visible until user clicks in the window to capture (fullscreen or windowed). */
    input_apply_relative_mouse(SDL_FALSE, NULL);
    input_emscripten_install_pointer_lock_listener();
}

void input_shutdown(void)
{
    input_apply_relative_mouse(SDL_FALSE, NULL);
    printf("[INPUT] SDL2 input shutdown\n");
}

/* -----------------------------------------------------------------------
 * Per-frame polling
 * ----------------------------------------------------------------------- */
void input_update(uint8_t *key_map, uint8_t *last_pressed)
{
    /* Reset mouse deltas each frame */
    g_mouse.dx = 0;
    g_mouse.dy = 0;
    g_mouse.wheel_y = 0;

#if defined(__EMSCRIPTEN__)
    /* Pointer lock often ends here before SDL gets KEYDOWN(Escape). */
    {
        int lost = EM_ASM_INT({
            var v = Module['ab3dPointerLockLost'] | 0;
            Module['ab3dPointerLockLost'] = 0;
            return v;
        });
        if (lost) {
            input_apply_relative_mouse(SDL_FALSE, key_map);
        }
    }
#endif

    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
        switch (ev.type) {
        case SDL_QUIT:
            g_quit_requested = true;
            /* Map to ESC so the game exits cleanly */
            if (key_map) key_map[AMIGA_KEY_ESC] = 1;
            break;

        case SDL_KEYDOWN:
        {
            if (ev.key.repeat) {
                break;
            }
            if (ev.key.keysym.scancode == SDL_SCANCODE_ESCAPE) {
                /* One Esc always: release relative mode, clear keyboard, show pointer.
                 * (Browser often drops pointer lock before SDL_GetRelativeMouseMode() is
                 * false here — do not branch on that.) */
                SDL_bool had_capture = g_input_capture_active || SDL_GetRelativeMouseMode();
                input_apply_relative_mouse(SDL_FALSE, key_map);
                /* Quit to menu only when Esc was not releasing in-game capture */
                if (!had_capture && key_map) {
                    key_map[AMIGA_KEY_ESC] = 1;
                    if (last_pressed) *last_pressed = AMIGA_KEY_ESC;
                }
                break;
            }
            if (ev.key.keysym.scancode == SDL_SCANCODE_F2) {
                g_f2_pick_log_requested = true;
            } else if (ev.key.keysym.scancode == SDL_SCANCODE_F5) {
                g_f5_save_requested = true;
            } else if (ev.key.keysym.scancode == SDL_SCANCODE_F9) {
                g_f9_load_requested = true;
            } else if (ev.key.keysym.scancode == SDL_SCANCODE_F6) {
                g_f6_gouraud_visualize_requested = true;
            } else if (ev.key.keysym.scancode == SDL_SCANCODE_F7) {
                g_f7_spill_visualize_requested = true;
#if !defined(__EMSCRIPTEN__)
            } else if (ev.key.keysym.scancode == SDL_SCANCODE_F11 ||
                       ev.key.keysym.scancode == SDL_SCANCODE_F12) {
                /* Browser: F11 is handled in display.c (Fullscreen API); SDL often never sees it. */
                g_fullscreen_toggle_requested = true;
#endif
            } else if (ev.key.keysym.scancode == SDL_SCANCODE_TAB) {
                g_automap_toggle_requested = true;
            } else if (ev.key.keysym.scancode == SDL_SCANCODE_PAGEUP) {
                g_automap_pgup_requested = true;
            } else if (ev.key.keysym.scancode == SDL_SCANCODE_PAGEDOWN) {
                g_automap_pgdn_requested = true;
            }
            uint8_t amiga = sdl_to_amiga(ev.key.keysym.scancode);
            if (amiga != 0xFF && key_map) {
                key_map[amiga] = 1;
                if (last_pressed) *last_pressed = amiga;
            }
            break;
        }

        case SDL_KEYUP:
        {
            /* Same PollEvent batch often has KEYDOWN(Esc) then KEYUP(Esc); clearing ESC here
             * undoes quit in one frame. Release-capture path memset on KEYDOWN; never need
             * KEYUP to clear Esc in key_map. */
            if (ev.key.keysym.scancode == SDL_SCANCODE_ESCAPE) {
                break;
            }
            uint8_t amiga = sdl_to_amiga(ev.key.keysym.scancode);
            if (amiga != 0xFF && key_map) {
                key_map[amiga] = 0;
            }
            break;
        }

        case SDL_MOUSEMOTION:
            /* Only use mouse motion for look when captured */
            if (SDL_GetRelativeMouseMode()) {
                g_mouse.dx += (int16_t)ev.motion.xrel;
                g_mouse.dy += (int16_t)ev.motion.yrel;
            }
            break;

        case SDL_MOUSEBUTTONDOWN:
            /* First left-click captures; that click is not treated as fire/jump */
            if (!SDL_GetRelativeMouseMode()) {
                if (ev.button.button == SDL_BUTTON_LEFT) {
                    input_apply_relative_mouse(SDL_TRUE, NULL);
                }
                break;
            }
            /* Mouse buttons only affect gameplay while captured */
            if (ev.button.button == SDL_BUTTON_LEFT) {
                g_mouse.left_button = true;
                if (key_map) key_map[AMIGA_KEY_RALT] = 1;
            }
            if (ev.button.button == SDL_BUTTON_RIGHT) {
                g_mouse.right_button = true;
                if (key_map) key_map[AMIGA_KEY_SPACE] = 1;
            }
            break;

        case SDL_MOUSEBUTTONUP:
            if (ev.button.button == SDL_BUTTON_LEFT) {
                g_mouse.left_button = false;
                if (SDL_GetRelativeMouseMode() && key_map) key_map[AMIGA_KEY_RALT] = 0;
            }
            if (ev.button.button == SDL_BUTTON_RIGHT) {
                g_mouse.right_button = false;
                if (SDL_GetRelativeMouseMode() && key_map) key_map[AMIGA_KEY_SPACE] = 0;
            }
            break;

        case SDL_MOUSEWHEEL:
            if (!SDL_GetRelativeMouseMode()) break;
            {
                int16_t wheel_y = (int16_t)ev.wheel.y;
                if (ev.wheel.direction == SDL_MOUSEWHEEL_FLIPPED) {
                    wheel_y = (int16_t)-wheel_y;
                }
                g_mouse.wheel_y = (int16_t)(g_mouse.wheel_y + wheel_y);
            }
            break;

        case SDL_WINDOWEVENT:
            switch (ev.window.event) {
            case SDL_WINDOWEVENT_RESIZED:
                display_handle_resize();  /* use renderer output size, not window logical size */
                break;
            case SDL_WINDOWEVENT_FOCUS_LOST:
                /* Same as Esc: always sync release (lock may already be gone). */
                input_apply_relative_mouse(SDL_FALSE, key_map);
                break;
            default:
                break;
            }
            break;
        }
    }
    /* Pointer lock can drop before SDL_KEYDOWN(Escape); sync state next frame. */
    if (g_input_capture_active && !SDL_GetRelativeMouseMode()) {
        input_apply_relative_mouse(SDL_FALSE, key_map);
    }
    /* No gameplay from mouse while uncaptured (belt-and-suspenders vs stale buttons). */
    if (!SDL_GetRelativeMouseMode()) {
        g_mouse.left_button = false;
        g_mouse.right_button = false;
    }
}

void input_read_mouse(MouseState *out)
{
    if (out) *out = g_mouse;
}

void input_read_joy1(JoyState *out)
{
    if (out) memset(out, 0, sizeof(*out));
}

void input_read_joy2(JoyState *out)
{
    if (out) memset(out, 0, sizeof(*out));
}

bool input_key_pressed(const uint8_t *key_map, uint8_t keycode)
{
    if (!key_map || keycode >= 128) return false;
    return key_map[keycode] != 0;
}

void input_clear_keyboard(uint8_t *key_map)
{
    if (key_map) memset(key_map, 0, 128);
}

bool input_f5_save_requested(void)
{
    if (!g_f5_save_requested) return false;
    g_f5_save_requested = false;
    return true;
}

bool input_f9_load_requested(void)
{
    if (!g_f9_load_requested) return false;
    g_f9_load_requested = false;
    return true;
}

bool input_f6_gouraud_visualize_requested(void)
{
    if (!g_f6_gouraud_visualize_requested) return false;
    g_f6_gouraud_visualize_requested = false;
    return true;
}

bool input_f7_spill_visualize_requested(void)
{
    if (!g_f7_spill_visualize_requested) return false;
    g_f7_spill_visualize_requested = false;
    return true;
}

bool input_f2_pick_log_requested(void)
{
    if (!g_f2_pick_log_requested) return false;
    g_f2_pick_log_requested = false;
    return true;
}

bool input_automap_toggle_requested(void)
{
    if (!g_automap_toggle_requested) return false;
    g_automap_toggle_requested = false;
    return true;
}

bool input_automap_pgup_requested(void)
{
    if (!g_automap_pgup_requested) return false;
    g_automap_pgup_requested = false;
    return true;
}

bool input_automap_pgdn_requested(void)
{
    if (!g_automap_pgdn_requested) return false;
    g_automap_pgdn_requested = false;
    return true;
}

bool input_fullscreen_toggle_requested(void)
{
    if (!g_fullscreen_toggle_requested) return false;
    g_fullscreen_toggle_requested = false;
    return true;
}
