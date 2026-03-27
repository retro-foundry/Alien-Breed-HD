/*
 * Alien Breed 3D I - PC Port
 * display.c - SDL2 display backend
 *
 * Creates a window, takes the chunky buffer from the software renderer,
 * converts it through a palette to RGB, and presents it on screen.
 *
 * Internal render size comes from ab3d.ini (render_width/render_height).
 * The window is scaled independently; the image is letterboxed, centered,
 * aspect preserved.
 */

#include "display.h"
#include "renderer.h"
#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * SDL2 state
 * ----------------------------------------------------------------------- */
static SDL_Window   *g_window   = NULL;
static SDL_Renderer *g_sdl_ren  = NULL;
static SDL_Texture  *g_texture  = NULL;
static int g_present_width = 0;
static int g_present_height = 0;
static SDL_Rect g_present_dst_rect;
static int g_internal_w = RENDER_WIDTH;
static int g_internal_h = RENDER_HEIGHT;
static int g_use_fixed_renderer_size = 1;

/* Initial window size (pixels); does not affect internal render resolution */
#define DISPLAY_DEFAULT_WINDOW_W 1280
#define DISPLAY_DEFAULT_WINDOW_H 720

#ifdef AB3D_RELEASE
/* Release: optional fullscreen desktop after window creation */
#endif

/* Legacy palette no longer needed - colors come from the .wad LUT data
 * and are written directly to the rgb_buffer as ARGB8888 pixels. */

/* -----------------------------------------------------------------------
 * Letterbox: scale internal image to fit window, centered, aspect kept
 * ----------------------------------------------------------------------- */
static void display_update_letterbox(int win_w, int win_h)
{
    int rw = g_internal_w;
    int rh = g_internal_h;
    if (win_w < 1) win_w = 1;
    if (win_h < 1) win_h = 1;
    if (rw < 1) rw = 1;
    if (rh < 1) rh = 1;

    double sx = (double)win_w / (double)rw;
    double sy = (double)win_h / (double)rh;
    double sc = (sx < sy) ? sx : sy;
    int dw = (int)(rw * sc + 0.5);
    int dh = (int)(rh * sc + 0.5);
    if (dw < 1) dw = 1;
    if (dh < 1) dh = 1;
    if (dw > win_w) dw = win_w;
    if (dh > win_h) dh = win_h;

    g_present_dst_rect.x = (win_w - dw) / 2;
    g_present_dst_rect.y = (win_h - dh) / 2;
    g_present_dst_rect.w = dw;
    g_present_dst_rect.h = dh;

    renderer_set_present_size(dw, dh);
}

/* -----------------------------------------------------------------------
 * Renderer scaling helper
 * ----------------------------------------------------------------------- */
static void display_set_renderer_target_size(int w, int h)
{
    if (w < 1 || h < 1) return;
    printf("[DISPLAY] renderer target: %dx%d\n", w, h);
    renderer_resize(w, h);
    renderer_set_present_size(w, h);
    g_internal_w = w;
    g_internal_h = h;
    if (g_texture) SDL_DestroyTexture(g_texture);
    g_texture = SDL_CreateTexture(g_sdl_ren,
        SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
        g_internal_w, g_internal_h);
    if (g_texture) SDL_SetTextureScaleMode(g_texture, SDL_ScaleModeNearest);
}

/* -----------------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------------- */
void display_init(GameState *state)
{
    int rw = 1920;
    int rh = 1080;
    if (state) {
        rw = (int)state->cfg_render_width;
        rh = (int)state->cfg_render_height;
    }
    if (rw < 96) rw = 96;
    if (rh < 80) rh = 80;
    if (rw > 4096) rw = 4096;
    if (rh > 4096) rh = 4096;

    g_internal_w = rw;
    g_internal_h = rh;

    if (!SDL_WasInit(SDL_INIT_VIDEO)) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            printf("[DISPLAY] SDL_Init failed: %s\n", SDL_GetError());
            return;
        }
    }

    printf("[DISPLAY] SDL2 init (internal render %dx%d, window scaled to fit)\n",
           g_internal_w, g_internal_h);

    renderer_init();

    int window_w = DISPLAY_DEFAULT_WINDOW_W;
    int window_h = DISPLAY_DEFAULT_WINDOW_H;

    g_window = SDL_CreateWindow(
        "Alien Breed 3D I",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        window_w, window_h,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
    );
    if (!g_window) {
        printf("[DISPLAY] SDL_CreateWindow failed: %s\n", SDL_GetError());
        return;
    }

    g_sdl_ren = SDL_CreateRenderer(g_window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!g_sdl_ren) {
        printf("[DISPLAY] SDL_CreateRenderer failed: %s\n", SDL_GetError());
        return;
    }

    display_set_renderer_target_size(rw, rh);

    int out_w = window_w;
    int out_h = window_h;
    if (SDL_GetRendererOutputSize(g_sdl_ren, &out_w, &out_h) != 0) {
        out_w = window_w;
        out_h = window_h;
    }
    if (out_w < 1) out_w = 1;
    if (out_h < 1) out_h = 1;
    display_update_letterbox(out_w, out_h);
    g_present_width = out_w;
    g_present_height = out_h;

#ifdef AB3D_RELEASE
    if (SDL_SetWindowFullscreen(g_window, SDL_WINDOW_FULLSCREEN_DESKTOP) != 0) {
        printf("[DISPLAY] SDL_SetWindowFullscreenDesktop failed: %s\n", SDL_GetError());
    }
    SDL_PumpEvents();
    if (SDL_GetRendererOutputSize(g_sdl_ren, &out_w, &out_h) == 0 &&
        out_w >= 1 && out_h >= 1) {
        display_update_letterbox(out_w, out_h);
        g_present_width = out_w;
        g_present_height = out_h;
    }
#endif

    printf("[DISPLAY] SDL2 ready: window %dx%d, present rect %dx%d at (%d,%d)\n",
           window_w, window_h,
           g_present_dst_rect.w, g_present_dst_rect.h,
           g_present_dst_rect.x, g_present_dst_rect.y);
}

void display_on_resize(int w, int h)
{
    if (w < 1 || h < 1) return;
    printf("[DISPLAY] resize: %dx%d\n", w, h);
    display_update_letterbox(w, h);
    g_present_width = w;
    g_present_height = h;
}

void display_handle_resize(void)
{
    if (!g_sdl_ren) return;
    int out_w = 0, out_h = 0;
#ifdef AB3D_RELEASE
    if (g_window && (SDL_GetWindowFlags(g_window) & SDL_WINDOW_FULLSCREEN_DESKTOP) != 0) {
        SDL_DisplayMode dm;
        if (SDL_GetWindowDisplayMode(g_window, &dm) == 0 && dm.w >= 96 && dm.h >= 80) {
            out_w = dm.w;
            out_h = dm.h;
        }
    }
#endif
    if (out_w < 96 || out_h < 80) {
        if (SDL_GetRendererOutputSize(g_sdl_ren, &out_w, &out_h) != 0) return;
    }
    if (out_w < 1) out_w = 1;
    if (out_h < 1) out_h = 1;
    printf("[DISPLAY] handle_resize: output %dx%d\n", out_w, out_h);
    display_update_letterbox(out_w, out_h);
    g_present_width = out_w;
    g_present_height = out_h;
}

int display_is_fullscreen(void)
{
    if (!g_window) return 0;
    return (SDL_GetWindowFlags(g_window) & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_FULLSCREEN_DESKTOP)) != 0;
}

void display_shutdown(void)
{
    renderer_shutdown();
    if (g_texture)  SDL_DestroyTexture(g_texture);
    if (g_sdl_ren)  SDL_DestroyRenderer(g_sdl_ren);
    if (g_window)   SDL_DestroyWindow(g_window);
    SDL_Quit();
    printf("[DISPLAY] SDL2 shutdown\n");
}

/* -----------------------------------------------------------------------
 * Screen management (no-ops for SDL2)
 * ----------------------------------------------------------------------- */
void display_alloc_text_screen(void)        { }
void display_release_text_screen(void)      { }
void display_alloc_copper_screen(void)      { }
void display_release_copper_screen(void)    { }
void display_alloc_title_memory(void)       { }
void display_release_title_memory(void)     { }
void display_alloc_panel_memory(void)       { }
void display_release_panel_memory(void)     { }

void display_setup_title_screen(void)       { }
void display_load_title_screen(void)        { }
void display_clear_opt_screen(void)         { }
void display_draw_opt_screen(int screen_num) { (void)screen_num; }
void display_fade_up_title(int amount)      { (void)amount; }
void display_fade_down_title(int amount)    { (void)amount; }
void display_clear_title_palette(void)      { }

void display_init_copper_screen(void)       { }

/* -----------------------------------------------------------------------
 * Main rendering
 * ----------------------------------------------------------------------- */
void display_draw_display(GameState *state)
{
    /* 1. Software-render the 3D scene into the rgb buffer */
    renderer_draw_display(state);

    /* 2. Copy 32-bit ARGB rgb_buffer directly to SDL texture */
    if (!g_texture || !g_sdl_ren) return;

    const uint32_t *src = renderer_get_rgb_buffer();
    if (!src) return;

    uint32_t *pixels;
    int pitch;
    if (SDL_LockTexture(g_texture, NULL, (void**)&pixels, &pitch) < 0) return;

    int w = renderer_get_width(), h = renderer_get_height();
    /* Game buffer: A=0 raster, A=1 clear/sky (see renderer.h). SDL upload forces opaque. */
    if (pitch == (int)(w * sizeof(uint32_t))) {
        const size_t n = (size_t)w * (size_t)h;
        for (size_t i = 0; i < n; i++)
            pixels[i] = src[i] | 0xFF000000u;
    } else {
        for (int y = 0; y < h; y++) {
            uint32_t *dst_row = (uint32_t*)((uint8_t*)pixels + (size_t)y * pitch);
            const uint32_t *src_row = src + (size_t)y * w;
            for (int x = 0; x < w; x++)
                dst_row[x] = src_row[x] | 0xFF000000u;
        }
    }

    SDL_UnlockTexture(g_texture);

    /* 3. Letterbox: clear, then blit centered with aspect ratio */
    SDL_SetRenderDrawColor(g_sdl_ren, 0, 0, 0, 255);
    SDL_RenderClear(g_sdl_ren);
    SDL_RenderCopy(g_sdl_ren, g_texture, NULL, &g_present_dst_rect);
    SDL_RenderPresent(g_sdl_ren);

    /* Debug: show player position in window title (throttled) */
    if (state && g_window) {
        static int title_frame = 0;
        if ((++title_frame % 30) == 0) {
            PlayerState *dbg_plr = &state->plr1;
            char title[128];
            snprintf(title, sizeof(title), "AB3D - Pos(%d,%d) Zone=%d Ang=%d",
                     (int)(dbg_plr->xoff >> 16), (int)(dbg_plr->zoff >> 16),
                     dbg_plr->zone, dbg_plr->angpos);
            SDL_SetWindowTitle(g_window, title);
        }
    }
}

void display_swap_buffers(void)
{
    /* Handled in display_draw_display */
}

void display_wait_vblank(void)
{
    /* VSync is handled by SDL_RENDERER_PRESENTVSYNC */
}

/* -----------------------------------------------------------------------
 * HUD
 * ----------------------------------------------------------------------- */
void display_energy_bar(int16_t energy)
{
    uint32_t *rgb = g_renderer.rgb_buffer;
    if (!rgb) return;
    int w = g_renderer.width, h = g_renderer.height;
    int bar_y = h - 2;
    int bar_w = (energy > 0) ? ((int)energy * (w - 4) / 127) : 0;
    for (int x = 2; x < 2 + bar_w && x < w - 2; x++) {
        rgb[bar_y * w + x] = RENDER_RGB_RASTER_PIXEL(0x00CC00u);
        rgb[(bar_y - 1) * w + x] = RENDER_RGB_RASTER_PIXEL(0x00CC00u);
    }
}

void display_ammo_bar(int16_t ammo)
{
    uint32_t *rgb = g_renderer.rgb_buffer;
    if (!rgb) return;
    int w = g_renderer.width, h = g_renderer.height;
    int bar_y = h - 5;
    int max_ammo = 999;
    int bar_w = (ammo > 0) ? ((int)ammo * (w - 4) / max_ammo) : 0;
    if (bar_w > w - 4) bar_w = w - 4;
    for (int x = 2; x < 2 + bar_w && x < w - 2; x++) {
        rgb[bar_y * w + x] = RENDER_RGB_RASTER_PIXEL(0xCCCC00u);
        rgb[(bar_y - 1) * w + x] = RENDER_RGB_RASTER_PIXEL(0xCCCC00u);
    }
}

/* -----------------------------------------------------------------------
 * Text (minimal for now)
 * ----------------------------------------------------------------------- */
void display_draw_line_of_text(const char *text, int line)
{
    (void)text;
    (void)line;
}

void display_clear_text_screen(void)
{
}
