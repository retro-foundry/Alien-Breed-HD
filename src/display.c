/*
 * Alien Breed 3D I - PC Port
 * display.c - SDL2 display backend
 *
 * Creates a window, takes the chunky buffer from the software renderer,
 * converts it through a palette to RGB, and presents it on screen.
 *
 * Internal render size comes from ab3d.ini (render_width/render_height).
 * The window is created at that size (1:1); if resized, the image is
 * letterboxed, centered, aspect preserved.
 */

#include "display.h"
#include "renderer.h"
#include <SDL.h>
/* SDL_GL_BindTexture / glGenerateMipmap: framebuffer minification when window < internal size */
#if SDL_VERSION_ATLEAST(2, 0, 12)
#include <SDL_opengl.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if SDL_VERSION_ATLEAST(2, 0, 12)
typedef void (APIENTRY *DisplayGlGenMipmapFn)(GLenum target);
typedef void (APIENTRY *DisplayGlTexParameteriFn)(GLenum target, GLenum pname, GLint param);
#endif

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
static int g_release_borderless_desktop = 0;

#if SDL_VERSION_ATLEAST(2, 0, 12)
static DisplayGlGenMipmapFn     g_gl_generate_mipmap;
static DisplayGlTexParameteriFn g_gl_tex_parameteri;
static int g_fb_mipmap_ok_logged;
static int g_fb_mipmap_fail_logged;
#endif

#ifdef AB3D_RELEASE
/* Release: run in borderless desktop-sized window (never request display mode changes). */
#endif

/* Legacy palette no longer needed - colors come from the .wad LUT data
 * and are written directly to the rgb_buffer as ARGB8888 pixels. */

#if SDL_VERSION_ATLEAST(2, 0, 12)
/* GL enums (avoid pulling full GL headers); values from GL spec */
#define DGL_TEXTURE_2D 0x0DE1
#define DGL_TEXTURE_MIN_FILTER 0x2801
#define DGL_TEXTURE_MAG_FILTER 0x2800
#define DGL_TEXTURE_WRAP_S 0x2802
#define DGL_TEXTURE_WRAP_T 0x2803
#define DGL_LINEAR_MIPMAP_LINEAR 0x2703
#define DGL_LINEAR 0x2601
#define DGL_CLAMP_TO_EDGE 0x812F

static void display_load_gl_mipmap_procs(void)
{
    g_gl_generate_mipmap = (DisplayGlGenMipmapFn)
        SDL_GL_GetProcAddress("glGenerateMipmap");
    g_gl_tex_parameteri = (DisplayGlTexParameteriFn)
        SDL_GL_GetProcAddress("glTexParameteri");
}

/* After streaming upload: rebuild mip chain so minification uses trilinear mips.
 * Resolves GL entry points after the renderer's context exists (lazy first frame). */
static void display_regenerate_framebuffer_mipmaps_if_downscaled(int tex_w, int tex_h)
{
    if (!g_texture || !g_sdl_ren) return;
    if (g_present_dst_rect.w >= tex_w && g_present_dst_rect.h >= tex_h)
        return;

    if (!g_gl_generate_mipmap || !g_gl_tex_parameteri)
        display_load_gl_mipmap_procs();
    if (!g_gl_generate_mipmap || !g_gl_tex_parameteri) {
        if (!g_fb_mipmap_fail_logged) {
            printf("[DISPLAY] framebuffer mipmaps: glGenerateMipmap unavailable (linear scale only)\n");
            g_fb_mipmap_fail_logged = 1;
        }
        return;
    }

    float tw, th;
    if (SDL_GL_BindTexture(g_texture, &tw, &th) != 0) {
        if (!g_fb_mipmap_fail_logged) {
            SDL_RendererInfo ri;
            const char *drv = "?";
            if (SDL_GetRendererInfo(g_sdl_ren, &ri) == 0) drv = ri.name;
            printf("[DISPLAY] framebuffer mipmaps: SDL_GL_BindTexture failed (driver=%s; OpenGL enables mips)\n",
                   drv);
            g_fb_mipmap_fail_logged = 1;
        }
        return;
    }

    if (!g_fb_mipmap_ok_logged) {
        printf("[DISPLAY] framebuffer mipmaps: active (trilinear minification when window < internal res)\n");
        g_fb_mipmap_ok_logged = 1;
    }

    g_gl_tex_parameteri(DGL_TEXTURE_2D, DGL_TEXTURE_MIN_FILTER, (GLint)DGL_LINEAR_MIPMAP_LINEAR);
    g_gl_tex_parameteri(DGL_TEXTURE_2D, DGL_TEXTURE_MAG_FILTER, (GLint)DGL_LINEAR);
    g_gl_tex_parameteri(DGL_TEXTURE_2D, DGL_TEXTURE_WRAP_S, (GLint)DGL_CLAMP_TO_EDGE);
    g_gl_tex_parameteri(DGL_TEXTURE_2D, DGL_TEXTURE_WRAP_T, (GLint)DGL_CLAMP_TO_EDGE);
    g_gl_generate_mipmap(DGL_TEXTURE_2D);
    SDL_GL_UnbindTexture(g_texture);
}
#else
#define display_regenerate_framebuffer_mipmaps_if_downscaled(w, h) ((void)0)
#endif

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
    if (g_texture) {
        /* Linear mag; minification uses GL mip chain when OpenGL + downscale */
        SDL_SetTextureScaleMode(g_texture, SDL_ScaleModeLinear);
    }
}

/* -----------------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------------- */
void display_init(GameState *state)
{
    int rw = 1280;
    int rh = 720;
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

#ifdef AB3D_RELEASE
    printf("[DISPLAY] SDL2 init (internal render %dx%d, startup: borderless desktop window)\n",
           g_internal_w, g_internal_h);
#else
    printf("[DISPLAY] SDL2 init (internal render %dx%d, window opens at that size; resize to letterbox)\n",
           g_internal_w, g_internal_h);
#endif

    renderer_init();

    int window_w = rw;
    int window_h = rh;
    Uint32 window_flags = SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;
    int window_x = SDL_WINDOWPOS_CENTERED;
    int window_y = SDL_WINDOWPOS_CENTERED;
#ifdef AB3D_RELEASE
    SDL_Rect desktop_bounds;
    SDL_DisplayMode desktop_mode;
    if (SDL_GetDesktopDisplayMode(0, &desktop_mode) == 0 &&
        desktop_mode.w >= 96 && desktop_mode.h >= 80) {
        window_w = desktop_mode.w;
        window_h = desktop_mode.h;
    }
    if (SDL_GetDisplayBounds(0, &desktop_bounds) == 0) {
        window_x = desktop_bounds.x;
        window_y = desktop_bounds.y;
        if (desktop_bounds.w >= 96 && desktop_bounds.h >= 80) {
            window_w = desktop_bounds.w;
            window_h = desktop_bounds.h;
        }
    }
    window_flags = SDL_WINDOW_SHOWN | SDL_WINDOW_BORDERLESS | SDL_WINDOW_OPENGL;
    printf("[DISPLAY] Release OpenGL path: borderless desktop window + SDL_WINDOW_OPENGL\n");
    g_release_borderless_desktop = 1;
#endif

    g_window = SDL_CreateWindow(
        "Alien Breed 3D I",
        window_x, window_y,
        window_w, window_h,
        window_flags
    );
    if (!g_window) {
        printf("[DISPLAY] SDL_CreateWindow failed: %s\n", SDL_GetError());
        return;
    }

    /* Prefer OpenGL so SDL_GL_BindTexture + glGenerateMipmap work for framebuffer mips */
    SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengl");
    g_sdl_ren = SDL_CreateRenderer(g_window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!g_sdl_ren) {
        SDL_SetHint(SDL_HINT_RENDER_DRIVER, "");
        g_sdl_ren = SDL_CreateRenderer(g_window, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    }
    if (!g_sdl_ren) {
        printf("[DISPLAY] SDL_CreateRenderer failed: %s\n", SDL_GetError());
        return;
    }
    {
        SDL_RendererInfo ri;
        if (SDL_GetRendererInfo(g_sdl_ren, &ri) == 0)
            printf("[DISPLAY] SDL renderer driver: %s\n", ri.name);
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
    if (SDL_GetRendererOutputSize(g_sdl_ren, &out_w, &out_h) != 0) return;
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
#ifdef AB3D_RELEASE
    if (g_release_borderless_desktop) return 1;
#endif
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

#if SDL_VERSION_ATLEAST(2, 0, 12)
    display_regenerate_framebuffer_mipmaps_if_downscaled(w, h);
#endif

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
