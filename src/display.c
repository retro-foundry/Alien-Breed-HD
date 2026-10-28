/*
 * Alien Breed 3D I - PC Port
 * display.c - SDL2 display backend
 *
 * Creates a window and presents the 12-bit Amiga color-word framebuffer.
 * Default: SDL OpenGL render driver + GL_R16UI + fragment shader (raw uint16 upload, no SDL BlitNtoN).
 * Fallback: Direct3D + SDL texture (UpdateTexture / RenderCopy may use internal blit). Opt-out: AB3D_DISABLE_GL_UNPACK=1.
 *
 * Base window size comes from ab3d.ini (render_width/render_height).
 * Internal render size is base size multiplied by supersampling.
 * The final image is letterboxed, centered, aspect preserved.
 *
 * Emscripten: internal resolution defaults from ab3d.ini, then (unless AB3D_WEB_USE_INI_RES=1)
 * is overridden to a preset matching the physical screen aspect (nearest of 16:9, 16:10, …).
 * The SDL canvas is created at the base size; letterboxing scales to the drawable area.
 */

#include "display.h"
#ifdef AB3D_DOS
#include "platform_dos.h"
#endif
#include "renderer.h"
#include "game_data.h"
#include "io.h"
#include <SDL.h>
#include <SDL_opengl.h>
/* SDL_GL_BindTexture / glGenerateMipmap: framebuffer minification when window < internal size */
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>

/* SDL_SetWindowFullscreen is unreliable with the HTML5 canvas; use the Fullscreen API and
 * sync letterboxing when the canvas size changes. F11 often does not reach SDL (browser). */
static void display_emscripten_install_fullscreen_listeners(void)
{
    MAIN_THREAD_EM_ASM(({
        function ab3dToggleCanvasFullscreen() {
            var c = Module['canvas'];
            if (!c) return;
            var el = document.fullscreenElement || document.webkitFullscreenElement ||
                     document.mozFullScreenElement || document.msFullscreenElement;
            if (el === c) {
                var ex = document.exitFullscreen || document.webkitExitFullscreen ||
                         document.mozCancelFullScreen || document.msExitFullscreen;
                if (ex) ex.call(document);
            } else {
                var req = c.requestFullscreen || c.webkitRequestFullscreen ||
                          c.mozRequestFullScreen || c.msRequestFullscreen;
                if (req) req.call(c).catch(function() {});
            }
        }
        Module['ab3dToggleCanvasFullscreen'] = ab3dToggleCanvasFullscreen;
        Module['ab3dFullscreenResizePending'] = 0;
        function ab3dSyncCanvasSizeToClient() {
            var c = Module['canvas'];
            var w = 0;
            var h = 0;
            if (c) {
                w = c.clientWidth | 0;
                h = c.clientHeight | 0;
            }
            if (w < 1) w = window.innerWidth | 0;
            if (h < 1) h = window.innerHeight | 0;
            if (w < 1) w = 1;
            if (h < 1) h = 1;
            if (c) {
                if ((c.width | 0) !== w) c.width = w;
                if ((c.height | 0) !== h) c.height = h;
            }
            Module['ab3dFullscreenResizePending'] = 1;
        }
        Module['ab3dSyncCanvasSizeToClient'] = ab3dSyncCanvasSizeToClient;
        var mark = function() { Module['ab3dFullscreenResizePending'] = 1; };
        window.addEventListener('resize', mark, false);
        if (window.visualViewport)
            window.visualViewport.addEventListener('resize', mark, false);
        document.addEventListener('fullscreenchange', mark, false);
        document.addEventListener('webkitfullscreenchange', mark, false);
        window.addEventListener('keydown', function(e) {
            if (e.code !== 'F11' && e.key !== 'F11' && e.keyCode !== 122) return;
            e.preventDefault();
            e.stopPropagation();
            ab3dToggleCanvasFullscreen();
        }, true);
    }));
}

static void display_emscripten_canvas_fullscreen_toggle(void)
{
    MAIN_THREAD_EM_ASM(({
        if (Module['ab3dToggleCanvasFullscreen'])
            Module['ab3dToggleCanvasFullscreen']();
    }));
}

/* Pick render_width/height from the nearest common aspect to screen.width/screen.height. */
static void display_emscripten_apply_screen_aspect_resolution(int *out_w,
                                                              int *out_h)
{
    MAIN_THREAD_EM_ASM(({
        var screenW = screen.width;
        var screenH = screen.height;
        if (screenW < 1 || screenH < 1) {
            screenW = window.innerWidth || 1920;
            screenH = window.innerHeight || 1080;
        }
        var measured = screenW / screenH;
        var common = [[16, 9], [16, 10], [3, 2], [4, 3], [21, 9], [32, 9]];
        var bestIdx = 0;
        var bestDiff = 1e9;
        for (var i = 0; i < common.length; i++) {
            var cw = common[i][0], ch = common[i][1];
            var diff = Math.abs(measured - (cw / ch));
            if (diff < bestDiff) {
                bestDiff = diff;
                bestIdx = i;
            }
        }
        /* Preset pixel sizes (same aspect as common[bestIdx]); clamped later in C if needed. */
        var res = [
            [1920, 1080],
            [1920, 1200],
            [1920, 1280],
            [1920, 1440],
            [2560, 1080],
            [3840, 1080]
        ];
        var tw = res[bestIdx][0];
        var th = res[bestIdx][1];
        HEAP32[$0 >> 2] = tw;
        HEAP32[$1 >> 2] = th;
    }), out_w, out_h);
}
#endif

#ifndef AB3D_DOS
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

/* OpenGL 3.0+ integer texture (not always in SDL's bundled gl.h) */
#ifndef GL_R16UI
#define GL_R16UI 0x8232
#endif
#ifndef GL_RED_INTEGER
#define GL_RED_INTEGER 0x8D94
#endif
#ifndef GL_COLOR_BUFFER_BIT
#define GL_COLOR_BUFFER_BIT 0x00004000
#endif
#ifndef GL_TRIANGLE_STRIP
#define GL_TRIANGLE_STRIP 0x0005
#endif
#ifndef GL_VERTEX_SHADER
#define GL_VERTEX_SHADER 0x8B31
#endif
#ifndef GL_FRAGMENT_SHADER
#define GL_FRAGMENT_SHADER 0x8B30
#endif
#ifndef GL_COMPILE_STATUS
#define GL_COMPILE_STATUS 0x8B81
#endif
#ifndef GL_LINK_STATUS
#define GL_LINK_STATUS 0x8B82
#endif
#ifndef GL_INFO_LOG_LENGTH
#define GL_INFO_LOG_LENGTH 0x8B84
#endif
#ifndef GL_TEXTURE0
#define GL_TEXTURE0 0x84C0
#endif
#ifndef GL_UNPACK_ALIGNMENT
#define GL_UNPACK_ALIGNMENT 0x0CF5
#endif

#if SDL_VERSION_ATLEAST(2, 0, 12)
typedef void (APIENTRY *DisplayGlGenMipmapFn)(GLenum target);
typedef void (APIENTRY *DisplayGlTexParameteriFn)(GLenum target, GLenum pname, GLint param);
#endif

#define DISPLAY_AUTOMAP_MAX_SEGS 4096

/* -----------------------------------------------------------------------
 * SDL2 state
 * ----------------------------------------------------------------------- */
static SDL_Window   *g_window   = NULL;
static SDL_Renderer *g_sdl_ren  = NULL;
static SDL_Texture  *g_texture  = NULL;
/* True when g_texture accepts native packed 0x0RGB words directly (XRGB4444/ARGB4444). */
static int g_texture_is_4444_direct = 0;
#ifdef AB3D_DOS
static SDL_Surface *g_dos_frame_surfaces[2];
static const uint16_t *g_dos_frame_pixels[2];
static int g_dos_frame_next;
static unsigned char *g_dos_map_scratch;
static int g_dos_map_capacity;

/* Renderer output already matches SDL and VESA: native RGB565, row-major.
 * Surface descriptors borrow the two engine buffers. No conversion or scaling. */
static void display_dos_copy_cw(const uint16_t *src,int w,int h,const SDL_Rect *rect)
{
    SDL_Surface *surface=SDL_GetWindowSurface(g_window);
    if(!surface || surface->format!=SDL_PIXELFORMAT_RGB565) {
        fprintf(stderr,"DOS RGB565 framebuffer unavailable: %s\n",SDL_GetError());exit(1);
    }
    if(rect->w!=w || rect->h!=h || w>surface->w || h>surface->h) {
        fprintf(stderr,"DOS requires a 1:1 framebuffer fitting the VESA mode\n");exit(1);
    }
    int slot;
    if(g_dos_frame_pixels[0]==src)slot=0;
    else if(g_dos_frame_pixels[1]==src)slot=1;
    else {slot=g_dos_frame_next;g_dos_frame_next^=1;}
    SDL_Surface *frame=g_dos_frame_surfaces[slot];
    if(g_dos_frame_pixels[slot]!=src || !frame || frame->w!=w || frame->h!=h) {
        if(frame)SDL_DestroySurface(frame);
        frame=SDL_CreateSurfaceFrom(w,h,SDL_PIXELFORMAT_RGB565,(void *)src,w*2);
        if(!frame){fprintf(stderr,"DOS native surface failed: %s\n",SDL_GetError());exit(1);}
        SDL_SetSurfaceBlendMode(frame,SDL_BLENDMODE_NONE);
        g_dos_frame_surfaces[slot]=frame;g_dos_frame_pixels[slot]=src;
    }
    SDL_Rect dst=*rect;
    if(!SDL_BlitSurface(frame,NULL,surface,&dst)) {
        fprintf(stderr,"DOS native framebuffer blit failed: %s\n",SDL_GetError());exit(1);
    }
}
#endif
/* Key HUD: one SDL texture per key sprite frame 0..3 (invalidated when level/assets change). */
static SDL_Texture *g_key_hud_tex[4];
static uintptr_t    g_key_hud_tex_tag;
static SDL_Texture *g_bar_hud_tex[2];
static void display_key_hud_free_textures(void);

/* HUD digit strips: 0 = health %, 1 = ammo count (fonts, e.g. health_digits.png). */
#define DISPLAY_HUD_DIGIT_KIND_COUNT  2
#define DISPLAY_HUD_DIGIT_COUNT       10
#define DISPLAY_HUD_DIGIT_CELL_W      9
#define DISPLAY_HUD_DIGIT_CELL_H      11
#define DISPLAY_HUD_DIGIT_STRIDE      11
#define DISPLAY_HUD_DIGIT_SHEET_W     108
static SDL_Texture *g_hud_digit_tex[DISPLAY_HUD_DIGIT_KIND_COUNT];
static int          g_hud_digit_tex_w[DISPLAY_HUD_DIGIT_KIND_COUNT];
static int          g_hud_digit_tex_h[DISPLAY_HUD_DIGIT_KIND_COUNT];
static GLuint       g_hud_digit_array_tex[DISPLAY_HUD_DIGIT_KIND_COUNT];
static int          g_hud_digit_array_ok[DISPLAY_HUD_DIGIT_KIND_COUNT];
static int          g_hud_digit_array_upload_attempted[DISPLAY_HUD_DIGIT_KIND_COUNT];
static int          g_hud_digits_load_attempted;
static const char *const g_hud_digit_paths[DISPLAY_HUD_DIGIT_KIND_COUNT] = {
    "fonts/health_digits.png",
    "fonts/ammo_digits.png"
};
static void         display_hud_digits_free(void);
static void         display_hud_digits_ensure_loaded(void);
static void         display_hud_digit_upload_array_texture(int kind);
static int          hud_digit_tex_scale_k(int tex_w, int tex_h);

/* Printable ASCII bitmap font from fonts/ascii_font/ascii_font_green_black_metadata_v2.json. */
#define DISPLAY_ASCII_FONT_PATH       "fonts/ascii_font/ascii_font_green_black_atlas_v2.png"
#define DISPLAY_ASCII_FIRST           32
#define DISPLAY_ASCII_LAST            126
#define DISPLAY_ASCII_COUNT           (DISPLAY_ASCII_LAST - DISPLAY_ASCII_FIRST + 1)
#define DISPLAY_ASCII_COLS            16
#ifdef AB3D_DOS
/* DOS packaging removes cell padding that the glyph blitter never samples. */
#define DISPLAY_ASCII_CELL_W          8
#define DISPLAY_ASCII_CELL_H          13
#define DISPLAY_ASCII_DRAW_X          0
#define DISPLAY_ASCII_DRAW_Y          0
#else
#define DISPLAY_ASCII_CELL_W          12
#define DISPLAY_ASCII_CELL_H          17
#define DISPLAY_ASCII_DRAW_X          2
#define DISPLAY_ASCII_DRAW_Y          2
#endif
#define DISPLAY_ASCII_DRAW_W          8
#define DISPLAY_ASCII_DRAW_H          13
#define DISPLAY_ASCII_XADVANCE        8
#define DISPLAY_ASCII_LINE_ADVANCE    14
#define DISPLAY_ASCII_SCALE_ONE       256
#define DISPLAY_TEXT_MAX_LINES        32
#define DISPLAY_TEXT_MAX_CHARS        128
#ifdef AB3D_DOS
#define DISPLAY_TEXT_LAYOUT_MAX_RUNS 128
static int g_dos_text_page,g_dos_text_pages=1,g_dos_text_page_rows;
void display_text_reset_pages(void) { g_dos_text_page=0;g_dos_text_pages=1; }
int display_text_next_page(void) {
    if(g_dos_text_page+1>=g_dos_text_pages)return 0;
    g_dos_text_page++;return 1;
}
#else
#define DISPLAY_TEXT_LAYOUT_MAX_RUNS  DISPLAY_TEXT_MAX_LINES
#endif
static SDL_Texture *g_ascii_font_tex;
static int          g_ascii_font_tex_w;
static int          g_ascii_font_tex_h;
static int          g_ascii_font_load_attempted;
static GLuint       g_ascii_font_array_tex;
static int          g_ascii_font_array_ok;
static int          g_ascii_font_array_upload_attempted;
static char         g_text_lines[DISPLAY_TEXT_MAX_LINES][DISPLAY_TEXT_MAX_CHARS];
static uint8_t      g_text_line_used[DISPLAY_TEXT_MAX_LINES];
static uint8_t      g_text_line_alpha[DISPLAY_TEXT_MAX_LINES];
static void         display_ascii_font_free(void);
static void         display_ascii_font_upload_array_texture(void);
static void         display_text_screen_draw(Uint8 alpha);
static void         display_text_screen_draw_in_rect(Uint8 alpha, SDL_Rect r);

/* Amiga ControlLoop.s LoadTitleScrn loads disk/includes/titlescrnraw into
 * seven 10240-byte bitplanes, and TitleCop.s supplies TITLEPAL. */
#define DISPLAY_TITLE_RAW_PATH      "data/includes/titlescrnraw"
#define DISPLAY_TITLE_PAL_PATH      "data/pal/TitleScrnPal"
#define DISPLAY_TITLE_W             320
#define DISPLAY_TITLE_H             256
#define DISPLAY_TITLE_PLANES        7
#define DISPLAY_TITLE_ROW_BYTES     (DISPLAY_TITLE_W / 8)
#define DISPLAY_TITLE_PLANE_BYTES   (DISPLAY_TITLE_ROW_BYTES * DISPLAY_TITLE_H)
#define DISPLAY_TITLE_RAW_BYTES     (DISPLAY_TITLE_PLANES * DISPLAY_TITLE_PLANE_BYTES)
#define DISPLAY_TITLE_PAL_COLORS    (1 << DISPLAY_TITLE_PLANES)
#define DISPLAY_TITLE_PAL_ENTRY_BYTES 4
#define DISPLAY_TITLE_PAL_BYTES     (DISPLAY_TITLE_PAL_COLORS * DISPLAY_TITLE_PAL_ENTRY_BYTES)
#define DISPLAY_TITLE_FULL_FADE_SCALE 252
static SDL_Texture *g_title_screen_tex;
static GLuint       g_title_screen_gl_tex;
static int          g_title_screen_load_attempted;
static void         display_title_screen_free(void);

#ifdef AB3D_DOS
static const int g_gl_unpack_ok=0;
#else
static int g_gl_unpack_ok; /* OpenGL R16UI + shader path active */
#endif
static int g_present_width = 0;
static int g_present_height = 0;
static SDL_Rect g_present_dst_rect;
static int g_internal_w = RENDER_WIDTH;
static int g_internal_h = RENDER_HEIGHT;
static int g_release_borderless_desktop = 0;
static int g_debug_window_title = 0;
static int g_framebuffer_mipmap_enabled = 0;
static int g_screen_tint_enabled = 0;
static Uint8 g_screen_tint_r = 0;
static Uint8 g_screen_tint_g = 0;
static Uint8 g_screen_tint_b = 0;
static Uint8 g_screen_tint_a = 0;
static int g_suppress_next_present_count = 0;
/* Reused for GL automap line NDC conversion (avoid per-frame malloc/free). */
static float *g_gl_lines_ndc_scratch = NULL;
static size_t g_gl_lines_ndc_scratch_cap = 0;

#if SDL_VERSION_ATLEAST(2, 0, 12)
static DisplayGlGenMipmapFn     g_gl_generate_mipmap;
static DisplayGlTexParameteriFn g_gl_tex_parameteri;
static int g_fb_mipmap_ok_logged;
static int g_fb_mipmap_fail_logged;
#endif

/* ini display_mode=fullscreen: same SDL window flags as windowed (no SDL fullscreen APIs).
 * Window client area is sized to the primary display bounds; internal render size is still
 * from ab3d.ini (render_width/height × supersampling), letterboxed into the window. */

/* Non-GL path: g_texture is XRGB4444/ARGB4444 direct (preferred) or ARGB8888 fallback. */

#if SDL_VERSION_ATLEAST(2, 0, 12)
/* GL enums (avoid pulling full GL headers); values from GL spec */
#define DGL_TEXTURE_2D 0x0DE1
#define DGL_TEXTURE_MIN_FILTER 0x2801
#define DGL_TEXTURE_MAG_FILTER 0x2800
#define DGL_TEXTURE_WRAP_S 0x2802
#define DGL_TEXTURE_WRAP_T 0x2803
#define DGL_TEXTURE_WRAP_R 0x8072
#define DGL_TEXTURE_2D_ARRAY 0x8C1A
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
    if (!g_framebuffer_mipmap_enabled) return;
    if (g_gl_unpack_ok) return;
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
 * OpenGL: unpack 12-bit Amiga color words (GPU) — GL_R16UI + fragment shader
 * Fallback: direct CPU upload to 4444 texture (ARGB8888 fallback expands).
 * ----------------------------------------------------------------------- */
#ifndef GL_NEAREST
#define GL_NEAREST 0x2600
#endif

typedef void   (APIENTRY *DisplayGlActiveTextureFn)(GLenum texture);
typedef void   (APIENTRY *DisplayGlBindBufferFn)(GLenum target, GLuint buffer);
typedef void   (APIENTRY *DisplayGlBindTextureFn)(GLenum target, GLuint texture);
typedef void   (APIENTRY *DisplayGlBindVertexArrayFn)(GLuint array);
typedef void   (APIENTRY *DisplayGlBufferDataFn)(GLenum target, ptrdiff_t size, const void *data, GLenum usage);
typedef void   (APIENTRY *DisplayGlClearFn)(GLbitfield mask);
typedef void   (APIENTRY *DisplayGlClearColorFn)(GLfloat r, GLfloat g, GLfloat b, GLfloat a);
typedef void   (APIENTRY *DisplayGlDeleteBuffersFn)(GLsizei n, const GLuint *buffers);
typedef void   (APIENTRY *DisplayGlDeleteProgramFn)(GLuint program);
typedef void   (APIENTRY *DisplayGlDeleteShaderFn)(GLuint shader);
typedef void   (APIENTRY *DisplayGlDeleteTexturesFn)(GLsizei n, const GLuint *textures);
typedef void   (APIENTRY *DisplayGlDeleteVertexArraysFn)(GLsizei n, const GLuint *arrays);
typedef void   (APIENTRY *DisplayGlDrawArraysFn)(GLenum mode, GLint first, GLsizei count);
typedef void   (APIENTRY *DisplayGlEnableVertexAttribArrayFn)(GLuint index);
typedef void   (APIENTRY *DisplayGlGenBuffersFn)(GLsizei n, GLuint *buffers);
typedef void   (APIENTRY *DisplayGlGenTexturesFn)(GLsizei n, GLuint *textures);
typedef void   (APIENTRY *DisplayGlGenVertexArraysFn)(GLsizei n, GLuint *arrays);
typedef GLint  (APIENTRY *DisplayGlGetAttribLocationFn)(GLuint program, const GLchar *name);
typedef GLenum (APIENTRY *DisplayGlGetErrorFn)(void);
typedef void   (APIENTRY *DisplayGlGetProgramivFn)(GLuint program, GLenum pname, GLint *params);
typedef void   (APIENTRY *DisplayGlGetShaderivFn)(GLuint shader, GLenum pname, GLint *params);
typedef void   (APIENTRY *DisplayGlGetShaderInfoLogFn)(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef GLint  (APIENTRY *DisplayGlGetUniformLocationFn)(GLuint program, const GLchar *name);
typedef void   (APIENTRY *DisplayGlLinkProgramFn)(GLuint program);
typedef void   (APIENTRY *DisplayGlPixelStoreiFn)(GLenum pname, GLint param);
typedef void   (APIENTRY *DisplayGlShaderSourceFn)(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
typedef void   (APIENTRY *DisplayGlTexImage2DFn)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
typedef void   (APIENTRY *DisplayGlTexImage3DFn)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels);
typedef void   (APIENTRY *DisplayGlTexParameteriFn2)(GLenum target, GLenum pname, GLint param);
typedef void   (APIENTRY *DisplayGlTexSubImage2DFn)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels);
typedef void   (APIENTRY *DisplayGlUniform1iFn)(GLint location, GLint v0);
typedef void   (APIENTRY *DisplayGlUseProgramFn)(GLuint program);
typedef void   (APIENTRY *DisplayGlVertexAttribPointerFn)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
typedef void   (APIENTRY *DisplayGlViewportFn)(GLint x, GLint y, GLsizei width, GLsizei height);
typedef void   (APIENTRY *DisplayGlGetIntegervFn)(GLenum pname, GLint *params);
typedef GLuint (APIENTRY *DisplayGlCreateProgramFn)(void);
typedef GLuint (APIENTRY *DisplayGlCreateShaderFn)(GLenum type);
typedef void   (APIENTRY *DisplayGlAttachShaderFn)(GLuint program, GLuint shader);
typedef void   (APIENTRY *DisplayGlCompileShaderFn)(GLuint shader);
typedef void   (APIENTRY *DisplayGlGetProgramInfoLogFn)(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef void   (APIENTRY *DisplayGlEnableFn)(GLenum cap);
typedef void   (APIENTRY *DisplayGlDisableFn)(GLenum cap);
typedef void   (APIENTRY *DisplayGlBlendFuncFn)(GLenum sfactor, GLenum dfactor);
typedef void   (APIENTRY *DisplayGlUniform4fFn)(GLint location, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void   (APIENTRY *DisplayGlUniform1fFn)(GLint location, GLfloat v0);
typedef void   (APIENTRY *DisplayGlDisableVertexAttribArrayFn)(GLuint index);

#ifndef GL_BLEND
#define GL_BLEND 0x0BE2
#endif
#ifndef GL_SRC_ALPHA
#define GL_SRC_ALPHA 0x0302
#endif
#ifndef GL_ONE_MINUS_SRC_ALPHA
#define GL_ONE_MINUS_SRC_ALPHA 0x0303
#endif
#ifndef GL_DEPTH_TEST
#define GL_DEPTH_TEST 0x0B71
#endif
#ifndef GL_CULL_FACE
#define GL_CULL_FACE 0x0B44
#endif
#ifndef GL_SCISSOR_TEST
#define GL_SCISSOR_TEST 0x0C11
#endif
#ifndef GL_TRIANGLES
#define GL_TRIANGLES 0x0004
#endif
#ifndef GL_LINES
#define GL_LINES 0x0001
#endif
#ifndef GL_STREAM_DRAW
#define GL_STREAM_DRAW 0x88E0
#endif
#ifndef GL_RGBA
#define GL_RGBA 0x1908
#endif
#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif
#ifndef GL_UNSIGNED_BYTE
#define GL_UNSIGNED_BYTE 0x1401
#endif
#ifndef GL_MAX_TEXTURE_SIZE
#define GL_MAX_TEXTURE_SIZE 0x0D33
#endif
#ifndef GL_DST_COLOR
#define GL_DST_COLOR 0x0306
#endif
#ifndef GL_ZERO
#define GL_ZERO 0
#endif

static GLuint g_gl_tex_cw;
static GLuint g_gl_prog;
static GLuint g_gl_vao;
static GLuint g_gl_vbo;

/* Pre-baked gun frame GL textures: 8 guns × 4 frames = 32 slots, each 96×58 RGBA.
 * Uploaded once at asset-load time; zero means not loaded. */
#define GUN_GL_FRAME_COUNT 32
static GLuint g_gun_gl_frames[GUN_GL_FRAME_COUNT];

/* HUD overlays: SDL_Renderer GL backend does not reliably mix with raw glUseProgram;
 * draw native GL overlay textures with our own GLSL in the 12-bit present context. */
static int g_gl_hud_ok;
static GLuint g_gl_prog_hud_tex;
static GLuint g_gl_prog_hud_font;
static GLuint g_gl_prog_hud_solid;
static GLuint g_gl_hud_vao_tex;
static GLuint g_gl_hud_vao_solid;
static GLuint g_gl_hud_vbo;
static GLint g_gl_hud_loc_tex;
static GLint g_gl_hud_loc_color_tex;
static GLint g_gl_hud_loc_rb_swap;
static GLint g_gl_hud_loc_font_tex;
static GLint g_gl_hud_loc_font_color;
static GLint g_gl_hud_loc_font_layer;
static GLint g_gl_hud_loc_color_solid;
static int g_gl_overlay_win_w;
static int g_gl_overlay_win_h;
static DisplayGlEnableFn                   g_gl_enable;
static DisplayGlDisableFn                  g_gl_disable;
static DisplayGlBlendFuncFn                g_gl_blend_func;
static DisplayGlUniform4fFn                g_gl_uniform4f;
static DisplayGlUniform1fFn                g_gl_uniform1f;
static DisplayGlDisableVertexAttribArrayFn g_gl_disable_vertex_attrib_array;

static DisplayGlGetIntegervFn              g_gl_get_integerv;
static DisplayGlActiveTextureFn            g_gl_active_texture;
static DisplayGlBindBufferFn               g_gl_bind_buffer;
static DisplayGlBindTextureFn              g_gl_bind_texture;
static DisplayGlBindVertexArrayFn          g_gl_bind_vertex_array;
static DisplayGlBufferDataFn               g_gl_buffer_data;
static DisplayGlClearFn                    g_gl_clear;
static DisplayGlClearColorFn               g_gl_clear_color;
static DisplayGlDeleteBuffersFn            g_gl_delete_buffers;
static DisplayGlDeleteProgramFn            g_gl_delete_program;
static DisplayGlDeleteShaderFn             g_gl_delete_shader;
static DisplayGlDeleteTexturesFn           g_gl_delete_textures;
static DisplayGlDeleteVertexArraysFn       g_gl_delete_vertex_arrays;
static DisplayGlDrawArraysFn               g_gl_draw_arrays;
static DisplayGlEnableVertexAttribArrayFn  g_gl_enable_vertex_attrib_array;
static DisplayGlGenBuffersFn               g_gl_gen_buffers;
static DisplayGlGenTexturesFn              g_gl_gen_textures;
static DisplayGlGenVertexArraysFn          g_gl_gen_vertex_arrays;
static DisplayGlGetAttribLocationFn        g_gl_get_attrib_location;
static DisplayGlGetErrorFn                 g_gl_get_error;
static DisplayGlGetProgramivFn           g_gl_get_programiv;
static DisplayGlGetShaderivFn              g_gl_get_shaderiv;
static DisplayGlGetShaderInfoLogFn        g_gl_get_shader_info_log;
static DisplayGlGetUniformLocationFn      g_gl_get_uniform_location;
static DisplayGlLinkProgramFn            g_gl_link_program;
static DisplayGlPixelStoreiFn              g_gl_pixel_storei;
static DisplayGlShaderSourceFn             g_gl_shader_source;
static DisplayGlTexImage2DFn               g_gl_tex_image_2d;
static DisplayGlTexImage3DFn               g_gl_tex_image_3d;
static DisplayGlTexParameteriFn2           g_gl_tex_parameteri2;
static DisplayGlTexSubImage2DFn            g_gl_tex_sub_image_2d;
static DisplayGlUniform1iFn               g_gl_uniform1i;
static DisplayGlUseProgramFn               g_gl_use_program;
static DisplayGlVertexAttribPointerFn      g_gl_vertex_attrib_pointer;
static DisplayGlViewportFn                 g_gl_viewport;
static DisplayGlCreateProgramFn            g_gl_create_program;
static DisplayGlCreateShaderFn             g_gl_create_shader;
static DisplayGlAttachShaderFn             g_gl_attach_shader;
static DisplayGlCompileShaderFn            g_gl_compile_shader;
static DisplayGlGetProgramInfoLogFn       g_gl_get_program_info_log;

static inline size_t display_cw_index_xy(int x, int y, int w, int h)
{
#if AB3D_CW_COL_MAJOR
    (void)w;
    return (size_t)x * (size_t)h + (size_t)y;
#else
    (void)h;
    return (size_t)y * (size_t)w + (size_t)x;
#endif
}

#if defined(__EMSCRIPTEN__)
static const char display_gl_vs_src[] =
    "#version 300 es\n"
    "precision highp float;\n"
    "layout(location = 0) in vec2 a_pos;\n"
    "layout(location = 1) in vec2 a_uv;\n"
    "out vec2 v_uv;\n"
    "void main() {\n"
    "  v_uv = a_uv;\n"
    "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "}\n";

static const char display_gl_fs_src[] =
    "#version 300 es\n"
    "precision highp float;\n"
    "precision highp int;\n"
    "uniform highp usampler2D u_cw;\n"
    "uniform highp int u_fb_w;\n"
    "uniform highp int u_fb_h;\n"
    "in highp vec2 v_uv;\n"
    "out highp vec4 o_col;\n"
    "void main() {\n"
    "  highp int fb_w = (u_fb_w > 0) ? u_fb_w : 1;\n"
    "  highp int fb_h = (u_fb_h > 0) ? u_fb_h : 1;\n"
    "  highp int ox = int(clamp(v_uv.x * float(fb_w), 0.0, float(fb_w - 1)));\n"
    "  highp int oy = int(clamp(v_uv.y * float(fb_h), 0.0, float(fb_h - 1)));\n"
#if AB3D_CW_COL_MAJOR
    "  highp int lin = ox * fb_h + oy;\n"
    "  highp ivec2 tp = ivec2(lin % fb_w, lin / fb_w);\n"
#else
    "  highp ivec2 tp = ivec2(ox, oy);\n"
#endif
    "  highp uint w = texelFetch(u_cw, tp, 0).r;\n"
    "  highp uint c = w & 0xFFFu;\n"
    "  highp uint r4 = (c >> 8) & 0xFu;\n"
    "  highp uint g4 = (c >> 4) & 0xFu;\n"
    "  highp uint b4 = c & 0xFu;\n"
    "  float r = float(r4) * (1.0 / 15.0);\n"
    "  float g = float(g4) * (1.0 / 15.0);\n"
    "  float b = float(b4) * (1.0 / 15.0);\n"
    "  o_col = vec4(r, g, b, 1.0);\n"
    "}\n";
#else
static const char display_gl_vs_src[] =
    "#version 330 core\n"
    "layout(location = 0) in vec2 a_pos;\n"
    "layout(location = 1) in vec2 a_uv;\n"
    "out vec2 v_uv;\n"
    "void main() {\n"
    "  v_uv = a_uv;\n"
    "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "}\n";

static const char display_gl_fs_src[] =
    "#version 330 core\n"
    "uniform usampler2D u_cw;\n"
    "uniform int u_fb_w;\n"
    "uniform int u_fb_h;\n"
    "in vec2 v_uv;\n"
    "out vec4 o_col;\n"
    "void main() {\n"
    "  int fb_w = (u_fb_w > 0) ? u_fb_w : 1;\n"
    "  int fb_h = (u_fb_h > 0) ? u_fb_h : 1;\n"
    "  int ox = int(clamp(v_uv.x * float(fb_w), 0.0, float(fb_w - 1)));\n"
    "  int oy = int(clamp(v_uv.y * float(fb_h), 0.0, float(fb_h - 1)));\n"
#if AB3D_CW_COL_MAJOR
    "  int lin = ox * fb_h + oy;\n"
    "  ivec2 tp = ivec2(lin % fb_w, lin / fb_w);\n"
#else
    "  ivec2 tp = ivec2(ox, oy);\n"
#endif
    "  uint w = texelFetch(u_cw, tp, 0).r;\n"
    "  uint c = w & 0xFFFu;\n"
    "  uint r4 = (c >> 8) & 0xFu;\n"
    "  uint g4 = (c >> 4) & 0xFu;\n"
    "  uint b4 = c & 0xFu;\n"
    "  float r = float(r4) * (1.0 / 15.0);\n"
    "  float g = float(g4) * (1.0 / 15.0);\n"
    "  float b = float(b4) * (1.0 / 15.0);\n"
    "  o_col = vec4(r, g, b, 1.0);\n"
    "}\n";
#endif

static int display_gl_load_procs(void)
{
    g_gl_active_texture = (DisplayGlActiveTextureFn)SDL_GL_GetProcAddress("glActiveTexture");
    g_gl_bind_buffer = (DisplayGlBindBufferFn)SDL_GL_GetProcAddress("glBindBuffer");
    g_gl_bind_texture = (DisplayGlBindTextureFn)SDL_GL_GetProcAddress("glBindTexture");
    g_gl_bind_vertex_array = (DisplayGlBindVertexArrayFn)SDL_GL_GetProcAddress("glBindVertexArray");
    g_gl_buffer_data = (DisplayGlBufferDataFn)SDL_GL_GetProcAddress("glBufferData");
    g_gl_clear = (DisplayGlClearFn)SDL_GL_GetProcAddress("glClear");
    g_gl_clear_color = (DisplayGlClearColorFn)SDL_GL_GetProcAddress("glClearColor");
    g_gl_delete_buffers = (DisplayGlDeleteBuffersFn)SDL_GL_GetProcAddress("glDeleteBuffers");
    g_gl_delete_program = (DisplayGlDeleteProgramFn)SDL_GL_GetProcAddress("glDeleteProgram");
    g_gl_delete_shader = (DisplayGlDeleteShaderFn)SDL_GL_GetProcAddress("glDeleteShader");
    g_gl_delete_textures = (DisplayGlDeleteTexturesFn)SDL_GL_GetProcAddress("glDeleteTextures");
    g_gl_delete_vertex_arrays = (DisplayGlDeleteVertexArraysFn)SDL_GL_GetProcAddress("glDeleteVertexArrays");
    g_gl_draw_arrays = (DisplayGlDrawArraysFn)SDL_GL_GetProcAddress("glDrawArrays");
    g_gl_enable_vertex_attrib_array = (DisplayGlEnableVertexAttribArrayFn)SDL_GL_GetProcAddress("glEnableVertexAttribArray");
    g_gl_gen_buffers = (DisplayGlGenBuffersFn)SDL_GL_GetProcAddress("glGenBuffers");
    g_gl_gen_textures = (DisplayGlGenTexturesFn)SDL_GL_GetProcAddress("glGenTextures");
    g_gl_gen_vertex_arrays = (DisplayGlGenVertexArraysFn)SDL_GL_GetProcAddress("glGenVertexArrays");
    g_gl_get_attrib_location = (DisplayGlGetAttribLocationFn)SDL_GL_GetProcAddress("glGetAttribLocation");
    g_gl_get_error = (DisplayGlGetErrorFn)SDL_GL_GetProcAddress("glGetError");
    g_gl_get_programiv = (DisplayGlGetProgramivFn)SDL_GL_GetProcAddress("glGetProgramiv");
    g_gl_get_shaderiv = (DisplayGlGetShaderivFn)SDL_GL_GetProcAddress("glGetShaderiv");
    g_gl_get_shader_info_log = (DisplayGlGetShaderInfoLogFn)SDL_GL_GetProcAddress("glGetShaderInfoLog");
    g_gl_get_uniform_location = (DisplayGlGetUniformLocationFn)SDL_GL_GetProcAddress("glGetUniformLocation");
    g_gl_link_program = (DisplayGlLinkProgramFn)SDL_GL_GetProcAddress("glLinkProgram");
    g_gl_pixel_storei = (DisplayGlPixelStoreiFn)SDL_GL_GetProcAddress("glPixelStorei");
    g_gl_shader_source = (DisplayGlShaderSourceFn)SDL_GL_GetProcAddress("glShaderSource");
    g_gl_tex_image_2d = (DisplayGlTexImage2DFn)SDL_GL_GetProcAddress("glTexImage2D");
    g_gl_tex_image_3d = (DisplayGlTexImage3DFn)SDL_GL_GetProcAddress("glTexImage3D");
    g_gl_tex_parameteri2 = (DisplayGlTexParameteriFn2)SDL_GL_GetProcAddress("glTexParameteri");
    g_gl_tex_sub_image_2d = (DisplayGlTexSubImage2DFn)SDL_GL_GetProcAddress("glTexSubImage2D");
    g_gl_uniform1i = (DisplayGlUniform1iFn)SDL_GL_GetProcAddress("glUniform1i");
    g_gl_use_program = (DisplayGlUseProgramFn)SDL_GL_GetProcAddress("glUseProgram");
    g_gl_vertex_attrib_pointer = (DisplayGlVertexAttribPointerFn)SDL_GL_GetProcAddress("glVertexAttribPointer");
    g_gl_viewport = (DisplayGlViewportFn)SDL_GL_GetProcAddress("glViewport");
    g_gl_create_program = (DisplayGlCreateProgramFn)SDL_GL_GetProcAddress("glCreateProgram");
    g_gl_create_shader = (DisplayGlCreateShaderFn)SDL_GL_GetProcAddress("glCreateShader");
    g_gl_attach_shader = (DisplayGlAttachShaderFn)SDL_GL_GetProcAddress("glAttachShader");
    g_gl_compile_shader = (DisplayGlCompileShaderFn)SDL_GL_GetProcAddress("glCompileShader");
    g_gl_get_program_info_log = (DisplayGlGetProgramInfoLogFn)SDL_GL_GetProcAddress("glGetProgramInfoLog");
    g_gl_get_integerv = (DisplayGlGetIntegervFn)SDL_GL_GetProcAddress("glGetIntegerv");

    return (g_gl_active_texture && g_gl_bind_buffer && g_gl_bind_texture && g_gl_bind_vertex_array &&
            g_gl_buffer_data && g_gl_clear && g_gl_clear_color && g_gl_delete_buffers && g_gl_delete_program &&
            g_gl_delete_shader && g_gl_delete_textures && g_gl_delete_vertex_arrays && g_gl_draw_arrays &&
            g_gl_enable_vertex_attrib_array && g_gl_gen_buffers && g_gl_gen_textures && g_gl_gen_vertex_arrays &&
            g_gl_get_attrib_location && g_gl_get_error && g_gl_get_programiv && g_gl_get_shaderiv &&
            g_gl_get_shader_info_log && g_gl_get_uniform_location && g_gl_link_program && g_gl_pixel_storei &&
            g_gl_shader_source && g_gl_tex_image_2d && g_gl_tex_parameteri2 && g_gl_tex_sub_image_2d &&
            g_gl_uniform1i && g_gl_use_program && g_gl_vertex_attrib_pointer && g_gl_viewport &&
            g_gl_create_program && g_gl_create_shader && g_gl_attach_shader && g_gl_compile_shader &&
            g_gl_get_program_info_log) ? 1 : 0;
}

static GLuint display_gl_compile_shader(GLenum type, const char *src)
{
    GLuint sh = g_gl_create_shader(type);
    if (!sh) return 0;
    g_gl_shader_source(sh, 1, &src, NULL);
    g_gl_compile_shader(sh);
    GLint ok = 0;
    g_gl_get_shaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        log[0] = 0;
        g_gl_get_shader_info_log(sh, (GLsizei)sizeof(log), NULL, log);
        printf("[DISPLAY] GL shader compile failed: %s\n", log);
        g_gl_delete_shader(sh);
        return 0;
    }
    return sh;
}

/* Letterbox + HUD rects use SDL_GetRendererOutputSize (same space as SDL_RenderCopy). Prefer it
 * over SDL_GL_GetDrawableSize so scaling matches the non-GL path on HiDPI / mixed-DPI setups. */
static void display_gl_output_size(int *out_w, int *out_h)
{
    if (g_sdl_ren && SDL_GetRendererOutputSize(g_sdl_ren, out_w, out_h) == 0) {
        if (*out_w < 1) *out_w = 1;
        if (*out_h < 1) *out_h = 1;
        return;
    }
    SDL_GL_GetDrawableSize(g_window, out_w, out_h);
    if (*out_w < 1) *out_w = 1;
    if (*out_h < 1) *out_h = 1;
}

#if defined(__EMSCRIPTEN__)
static const char hud_tex_vs_src[] =
    "#version 300 es\n"
    "precision highp float;\n"
    "layout(location=0) in vec2 a_pos;\n"
    "layout(location=1) in vec2 a_uv;\n"
    "out vec2 v_uv;\n"
    "void main() {\n"
    "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "  v_uv = a_uv;\n"
    "}\n";

static const char hud_tex_fs_src[] =
    "#version 300 es\n"
    "precision mediump float;\n"
    "uniform sampler2D u_tex;\n"
    "uniform vec4 u_color;\n"
    "uniform float u_rb_swap;\n"
    "in vec2 v_uv;\n"
    "out vec4 o_col;\n"
    "void main() {\n"
    "  vec4 s = texture(u_tex, v_uv);\n"
    "  vec4 t = mix(s, vec4(s.b, s.g, s.r, s.a), u_rb_swap);\n"
    "  o_col = t * u_color;\n"
    "}\n";

static const char hud_font_fs_src[] =
    "#version 300 es\n"
    "precision mediump float;\n"
    "uniform sampler2DArray u_tex;\n"
    "uniform vec4 u_color;\n"
    "uniform float u_layer;\n"
    "in vec2 v_uv;\n"
    "out vec4 o_col;\n"
    "void main() {\n"
    "  o_col = texture(u_tex, vec3(v_uv, u_layer)) * u_color;\n"
    "}\n";

static const char hud_solid_vs_src[] =
    "#version 300 es\n"
    "precision highp float;\n"
    "layout(location=0) in vec2 a_pos;\n"
    "void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }\n";

static const char hud_solid_fs_src[] =
    "#version 300 es\n"
    "precision mediump float;\n"
    "uniform vec4 u_color;\n"
    "out vec4 o_col;\n"
    "void main() { o_col = u_color; }\n";
#else
static const char hud_tex_vs_src[] =
    "#version 330 core\n"
    "layout(location=0) in vec2 a_pos;\n"
    "layout(location=1) in vec2 a_uv;\n"
    "out vec2 v_uv;\n"
    "void main() {\n"
    "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "  v_uv = a_uv;\n"
    "}\n";

static const char hud_tex_fs_src[] =
    "#version 330 core\n"
    "uniform sampler2D u_tex;\n"
    "uniform vec4 u_color;\n"
    "uniform float u_rb_swap;\n"
    "in vec2 v_uv;\n"
    "out vec4 o_col;\n"
    "void main() {\n"
    "  vec4 s = texture(u_tex, v_uv);\n"
    "  vec4 t = mix(s, vec4(s.b, s.g, s.r, s.a), u_rb_swap);\n"
    "  o_col = t * u_color;\n"
    "}\n";

static const char hud_font_fs_src[] =
    "#version 330 core\n"
    "uniform sampler2DArray u_tex;\n"
    "uniform vec4 u_color;\n"
    "uniform float u_layer;\n"
    "in vec2 v_uv;\n"
    "out vec4 o_col;\n"
    "void main() {\n"
    "  o_col = texture(u_tex, vec3(v_uv, u_layer)) * u_color;\n"
    "}\n";

static const char hud_solid_vs_src[] =
    "#version 330 core\n"
    "layout(location=0) in vec2 a_pos;\n"
    "void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }\n";

static const char hud_solid_fs_src[] =
    "#version 330 core\n"
    "uniform vec4 u_color;\n"
    "out vec4 o_col;\n"
    "void main() { o_col = u_color; }\n";
#endif

static void display_gl_wnd_ndc(float x, float y, int win_w, int win_h, float *nx, float *ny)
{
    *nx = (2.0f * x / (float)win_w) - 1.0f;
    *ny = 1.0f - (2.0f * y / (float)win_h);
}

static void display_gl_hud_shutdown(void)
{
    if (!g_gl_hud_ok) return;
    if (g_gl_delete_vertex_arrays && g_gl_hud_vao_tex) g_gl_delete_vertex_arrays(1, &g_gl_hud_vao_tex);
    if (g_gl_delete_vertex_arrays && g_gl_hud_vao_solid) g_gl_delete_vertex_arrays(1, &g_gl_hud_vao_solid);
    g_gl_hud_vao_tex = 0;
    g_gl_hud_vao_solid = 0;
    if (g_gl_delete_buffers && g_gl_hud_vbo) g_gl_delete_buffers(1, &g_gl_hud_vbo);
    g_gl_hud_vbo = 0;
    if (g_gl_delete_program && g_gl_prog_hud_tex) g_gl_delete_program(g_gl_prog_hud_tex);
    if (g_gl_delete_program && g_gl_prog_hud_font) g_gl_delete_program(g_gl_prog_hud_font);
    if (g_gl_delete_program && g_gl_prog_hud_solid) g_gl_delete_program(g_gl_prog_hud_solid);
    g_gl_prog_hud_tex = 0;
    g_gl_prog_hud_font = 0;
    g_gl_prog_hud_solid = 0;
    g_gl_hud_loc_font_tex = -1;
    g_gl_hud_loc_font_color = -1;
    g_gl_hud_loc_font_layer = -1;
    g_gl_hud_ok = 0;
}

static void display_gl_hud_try_init_font_program(void)
{
    g_gl_prog_hud_font = 0;
    g_gl_hud_loc_font_tex = -1;
    g_gl_hud_loc_font_color = -1;
    g_gl_hud_loc_font_layer = -1;

    GLuint vs = display_gl_compile_shader(GL_VERTEX_SHADER, hud_tex_vs_src);
    GLuint fs = display_gl_compile_shader(GL_FRAGMENT_SHADER, hud_font_fs_src);
    if (!vs || !fs) {
        if (vs) g_gl_delete_shader(vs);
        if (fs) g_gl_delete_shader(fs);
        return;
    }

    GLuint prog = g_gl_create_program();
    if (!prog) {
        g_gl_delete_shader(vs);
        g_gl_delete_shader(fs);
        return;
    }
    g_gl_attach_shader(prog, vs);
    g_gl_attach_shader(prog, fs);
    g_gl_delete_shader(vs);
    g_gl_delete_shader(fs);
    g_gl_link_program(prog);

    GLint linked = 0;
    g_gl_get_programiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[1024];
        log[0] = 0;
        if (g_gl_get_program_info_log)
            g_gl_get_program_info_log(prog, (GLsizei)sizeof(log), NULL, log);
        printf("[DISPLAY] HUD GL font array program link failed: %s\n", log);
        g_gl_delete_program(prog);
        return;
    }

    g_gl_hud_loc_font_tex = g_gl_get_uniform_location(prog, "u_tex");
    g_gl_hud_loc_font_color = g_gl_get_uniform_location(prog, "u_color");
    g_gl_hud_loc_font_layer = g_gl_get_uniform_location(prog, "u_layer");
    if (g_gl_hud_loc_font_tex < 0 ||
        g_gl_hud_loc_font_color < 0 ||
        g_gl_hud_loc_font_layer < 0) {
        printf("[DISPLAY] HUD GL font array program missing uniforms\n");
        g_gl_delete_program(prog);
        g_gl_hud_loc_font_tex = -1;
        g_gl_hud_loc_font_color = -1;
        g_gl_hud_loc_font_layer = -1;
        return;
    }

    g_gl_prog_hud_font = prog;
}

static int display_gl_hud_try_init(void)
{
    g_gl_hud_ok = 0;
    if (!g_sdl_ren || !g_window) return 0;

    g_gl_enable = (DisplayGlEnableFn)SDL_GL_GetProcAddress("glEnable");
    g_gl_disable = (DisplayGlDisableFn)SDL_GL_GetProcAddress("glDisable");
    g_gl_blend_func = (DisplayGlBlendFuncFn)SDL_GL_GetProcAddress("glBlendFunc");
    g_gl_uniform4f = (DisplayGlUniform4fFn)SDL_GL_GetProcAddress("glUniform4f");
    g_gl_uniform1f = (DisplayGlUniform1fFn)SDL_GL_GetProcAddress("glUniform1f");
    g_gl_disable_vertex_attrib_array = (DisplayGlDisableVertexAttribArrayFn)SDL_GL_GetProcAddress("glDisableVertexAttribArray");
    if (!g_gl_enable || !g_gl_disable || !g_gl_blend_func || !g_gl_uniform4f || !g_gl_uniform1f ||
        !g_gl_disable_vertex_attrib_array)
        return 0;

    GLuint v1 = display_gl_compile_shader(GL_VERTEX_SHADER, hud_tex_vs_src);
    GLuint f1 = display_gl_compile_shader(GL_FRAGMENT_SHADER, hud_tex_fs_src);
    if (!v1 || !f1) {
        if (v1) g_gl_delete_shader(v1);
        if (f1) g_gl_delete_shader(f1);
        return 0;
    }
    g_gl_prog_hud_tex = g_gl_create_program();
    if (!g_gl_prog_hud_tex) {
        g_gl_delete_shader(v1);
        g_gl_delete_shader(f1);
        return 0;
    }
    g_gl_attach_shader(g_gl_prog_hud_tex, v1);
    g_gl_attach_shader(g_gl_prog_hud_tex, f1);
    g_gl_delete_shader(v1);
    g_gl_delete_shader(f1);
    g_gl_link_program(g_gl_prog_hud_tex);
    GLint linked = 0;
    g_gl_get_programiv(g_gl_prog_hud_tex, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[1024];
        log[0] = 0;
        if (g_gl_get_program_info_log) g_gl_get_program_info_log(g_gl_prog_hud_tex, (GLsizei)sizeof(log), NULL, log);
        printf("[DISPLAY] HUD GL tex program link failed: %s\n", log);
        g_gl_delete_program(g_gl_prog_hud_tex);
        g_gl_prog_hud_tex = 0;
        return 0;
    }
    g_gl_hud_loc_tex = g_gl_get_uniform_location(g_gl_prog_hud_tex, "u_tex");
    g_gl_hud_loc_color_tex = g_gl_get_uniform_location(g_gl_prog_hud_tex, "u_color");
    g_gl_hud_loc_rb_swap = g_gl_get_uniform_location(g_gl_prog_hud_tex, "u_rb_swap");

    v1 = display_gl_compile_shader(GL_VERTEX_SHADER, hud_solid_vs_src);
    f1 = display_gl_compile_shader(GL_FRAGMENT_SHADER, hud_solid_fs_src);
    if (!v1 || !f1) {
        if (v1) g_gl_delete_shader(v1);
        if (f1) g_gl_delete_shader(f1);
        g_gl_delete_program(g_gl_prog_hud_tex);
        g_gl_prog_hud_tex = 0;
        return 0;
    }
    g_gl_prog_hud_solid = g_gl_create_program();
    if (!g_gl_prog_hud_solid) {
        g_gl_delete_shader(v1);
        g_gl_delete_shader(f1);
        g_gl_delete_program(g_gl_prog_hud_tex);
        g_gl_prog_hud_tex = 0;
        return 0;
    }
    g_gl_attach_shader(g_gl_prog_hud_solid, v1);
    g_gl_attach_shader(g_gl_prog_hud_solid, f1);
    g_gl_delete_shader(v1);
    g_gl_delete_shader(f1);
    g_gl_link_program(g_gl_prog_hud_solid);
    linked = 0;
    g_gl_get_programiv(g_gl_prog_hud_solid, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[1024];
        log[0] = 0;
        if (g_gl_get_program_info_log) g_gl_get_program_info_log(g_gl_prog_hud_solid, (GLsizei)sizeof(log), NULL, log);
        printf("[DISPLAY] HUD GL solid program link failed: %s\n", log);
        g_gl_delete_program(g_gl_prog_hud_solid);
        g_gl_prog_hud_solid = 0;
        g_gl_delete_program(g_gl_prog_hud_tex);
        g_gl_prog_hud_tex = 0;
        return 0;
    }
    g_gl_hud_loc_color_solid = g_gl_get_uniform_location(g_gl_prog_hud_solid, "u_color");

    display_gl_hud_try_init_font_program();

    g_gl_gen_vertex_arrays(1, &g_gl_hud_vao_tex);
    g_gl_gen_vertex_arrays(1, &g_gl_hud_vao_solid);
    g_gl_gen_buffers(1, &g_gl_hud_vbo);

    g_gl_bind_vertex_array(g_gl_hud_vao_tex);
    g_gl_bind_buffer(0x8892 /* GL_ARRAY_BUFFER */, g_gl_hud_vbo);
    g_gl_vertex_attrib_pointer(0, 2, 0x1406 /* GL_FLOAT */, 0, (GLsizei)(4 * sizeof(float)), (const void*)0);
    g_gl_vertex_attrib_pointer(1, 2, 0x1406, 0, (GLsizei)(4 * sizeof(float)), (const void*)(uintptr_t)(2 * sizeof(float)));
    g_gl_enable_vertex_attrib_array(0);
    g_gl_enable_vertex_attrib_array(1);
    g_gl_bind_vertex_array(0);

    g_gl_bind_vertex_array(g_gl_hud_vao_solid);
    g_gl_bind_buffer(0x8892, g_gl_hud_vbo);
    g_gl_vertex_attrib_pointer(0, 2, 0x1406, 0, (GLsizei)(2 * sizeof(float)), (const void*)0);
    g_gl_enable_vertex_attrib_array(0);
    g_gl_disable_vertex_attrib_array(1);
    g_gl_bind_vertex_array(0);

    g_gl_hud_ok = 1;
    if (g_ascii_font_tex && !g_ascii_font_array_ok)
        display_ascii_font_upload_array_texture();
    for (int i = 0; i < DISPLAY_HUD_DIGIT_KIND_COUNT; i++) {
        if (g_hud_digit_tex[i] && !g_hud_digit_array_ok[i])
            display_hud_digit_upload_array_texture(i);
    }
    printf("[DISPLAY] HUD: GL overlay (native textures + GLSL)\n");
    return 1;
}

/* Upload all 32 gun frame slots to pre-baked GL textures.
 * Must be called after io_load_gun_graphics() and after GL is initialized.
 * Safe to re-call (deletes old textures first). */
void display_upload_gun_gl_textures(void)
{
    if (!g_gl_unpack_ok) return;  /* GL path not available */
    if (!g_gl_gen_textures || !g_gl_bind_texture || !g_gl_tex_image_2d ||
        !g_gl_tex_parameteri2 || !g_gl_delete_textures) return;

    int w = renderer_gun_src_width();
    int h = renderer_gun_src_height();
    if (w < 1 || h < 1) return;

    uint32_t *scratch = (uint32_t *)malloc((size_t)w * (size_t)h * sizeof(uint32_t));
    if (!scratch) return;

    for (int slot = 0; slot < GUN_GL_FRAME_COUNT; slot++) {
        /* Delete any previously uploaded texture for this slot */
        if (g_gun_gl_frames[slot]) {
            g_gl_delete_textures(1, &g_gun_gl_frames[slot]);
            g_gun_gl_frames[slot] = 0;
        }
        if (!renderer_decode_gun_frame_rgba(slot, scratch)) continue;

        g_gl_gen_textures(1, &g_gun_gl_frames[slot]);
        g_gl_bind_texture(0x0DE1 /* GL_TEXTURE_2D */, g_gun_gl_frames[slot]);
        g_gl_tex_parameteri2(0x0DE1, 0x2800 /* GL_TEXTURE_MAG_FILTER */, 0x2600 /* GL_NEAREST */);
        g_gl_tex_parameteri2(0x0DE1, 0x2801 /* GL_TEXTURE_MIN_FILTER */, 0x2600 /* GL_NEAREST */);
        g_gl_tex_parameteri2(0x0DE1, 0x2802 /* GL_TEXTURE_WRAP_S */,     0x812F /* GL_CLAMP_TO_EDGE */);
        g_gl_tex_parameteri2(0x0DE1, 0x2803 /* GL_TEXTURE_WRAP_T */,     0x812F /* GL_CLAMP_TO_EDGE */);
        g_gl_tex_image_2d(0x0DE1, 0, (GLint)GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, scratch);
    }
    g_gl_bind_texture(0x0DE1, 0);
    free(scratch);
    printf("[DISPLAY] Gun GL textures uploaded (%d slots, %dx%d each)\n",
           GUN_GL_FRAME_COUNT, w, h);
}

/* Draw a pre-baked gun GL texture as a quad over dst (window pixels). */
static void display_gl_gun_quad(int frame_slot, const SDL_Rect *dst)
{
    if (!g_gl_hud_ok || g_gl_overlay_win_w < 1) return;
    if (frame_slot < 0 || frame_slot >= GUN_GL_FRAME_COUNT) return;
    if (!g_gun_gl_frames[frame_slot]) return;
    if (!dst) return;

#if SDL_VERSION_ATLEAST(2, 0, 14)
    SDL_RenderFlush(g_sdl_ren);
#endif
    int wx = g_gl_overlay_win_w, wy = g_gl_overlay_win_h;
    float nx0, ny0, nx1, ny1, nx2, ny2, nx3, ny3;
    display_gl_wnd_ndc((float)dst->x,           (float)dst->y,           wx, wy, &nx0, &ny0);
    display_gl_wnd_ndc((float)(dst->x + dst->w), (float)dst->y,           wx, wy, &nx1, &ny1);
    display_gl_wnd_ndc((float)(dst->x + dst->w), (float)(dst->y + dst->h), wx, wy, &nx2, &ny2);
    display_gl_wnd_ndc((float)dst->x,           (float)(dst->y + dst->h), wx, wy, &nx3, &ny3);

    float buf[24] = {
        nx0, ny0, 0.0f, 0.0f,
        nx1, ny1, 1.0f, 0.0f,
        nx3, ny3, 0.0f, 1.0f,
        nx1, ny1, 1.0f, 0.0f,
        nx2, ny2, 1.0f, 1.0f,
        nx3, ny3, 0.0f, 1.0f,
    };

    g_gl_use_program(g_gl_prog_hud_tex);
    if (g_gl_hud_loc_tex >= 0)        g_gl_uniform1i(g_gl_hud_loc_tex, 0);
    if (g_gl_hud_loc_color_tex >= 0)  g_gl_uniform4f(g_gl_hud_loc_color_tex, 1.0f, 1.0f, 1.0f, 1.0f);
    if (g_gl_hud_loc_rb_swap >= 0)    g_gl_uniform1f(g_gl_hud_loc_rb_swap, 0.0f);
    g_gl_active_texture(GL_TEXTURE0);
    g_gl_bind_texture(0x0DE1, g_gun_gl_frames[frame_slot]);
    g_gl_bind_vertex_array(g_gl_hud_vao_tex);
    g_gl_bind_buffer(0x8892, g_gl_hud_vbo);
    g_gl_buffer_data(0x8892, (ptrdiff_t)sizeof(buf), buf, GL_STREAM_DRAW);
    g_gl_draw_arrays(GL_TRIANGLES, 0, 6);
    g_gl_bind_texture(0x0DE1, 0);
}

/* Apply the Amiga underwater tint (AND #$00FF = zero red channel) as a GL
 * multiply quad.  fill_screen_water > 0 → full viewport; < 0 → bottom half. */
static void display_gl_multiply_tint_rect(int8_t fill_screen_water)
{
    if (!g_gl_hud_ok || g_gl_overlay_win_w < 1) return;
    if (fill_screen_water == 0) return;

    /* Compute rect in window pixels over the present dst letterbox */
    SDL_Rect r = g_present_dst_rect;
    if (fill_screen_water < 0) {
        /* Weak tint: bottom half only */
        int half = r.h / 2;
        r.y += half;
        r.h -= half;
    }
    if (r.w < 1 || r.h < 1) return;

#if SDL_VERSION_ATLEAST(2, 0, 14)
    SDL_RenderFlush(g_sdl_ren);
#endif
    /* Color multiply: dst_out = src_color * dst_color.  Amiga AND #$00FF zeroes
     * the red channel, so multiply by (R=0, G=1, B=1). */
    int wx = g_gl_overlay_win_w, wy = g_gl_overlay_win_h;
    float nx0, ny0, nx1, ny1, nx2, ny2, nx3, ny3;
    display_gl_wnd_ndc((float)r.x,         (float)r.y,         wx, wy, &nx0, &ny0);
    display_gl_wnd_ndc((float)(r.x + r.w), (float)r.y,         wx, wy, &nx1, &ny1);
    display_gl_wnd_ndc((float)(r.x + r.w), (float)(r.y + r.h), wx, wy, &nx2, &ny2);
    display_gl_wnd_ndc((float)r.x,         (float)(r.y + r.h), wx, wy, &nx3, &ny3);

    float solid_buf[12] = {
        nx0, ny0, nx1, ny1, nx3, ny3,
        nx1, ny1, nx2, ny2, nx3, ny3,
    };
    g_gl_use_program(g_gl_prog_hud_solid);
    if (g_gl_hud_loc_color_solid >= 0)
        g_gl_uniform4f(g_gl_hud_loc_color_solid, 0.0f, 1.0f, 1.0f, 1.0f);
    g_gl_bind_vertex_array(g_gl_hud_vao_solid);
    g_gl_bind_buffer(0x8892, g_gl_hud_vbo);
    g_gl_buffer_data(0x8892, (ptrdiff_t)sizeof(solid_buf), solid_buf, GL_STREAM_DRAW);
    /* Temporarily switch to multiply blend */
    g_gl_blend_func(GL_DST_COLOR, GL_ZERO);
    g_gl_draw_arrays(GL_TRIANGLES, 0, 6);
    /* Restore normal alpha blend */
    g_gl_blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

static void display_gl_overlay_begin(void)
{
    display_gl_output_size(&g_gl_overlay_win_w, &g_gl_overlay_win_h);
    g_gl_viewport(0, 0, g_gl_overlay_win_w, g_gl_overlay_win_h);
    g_gl_disable(GL_CULL_FACE);
    g_gl_disable(GL_SCISSOR_TEST);
    g_gl_enable(GL_BLEND);
    g_gl_blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    g_gl_disable(GL_DEPTH_TEST);
}

static void display_gl_overlay_end(void)
{
    g_gl_use_program(0);
    g_gl_bind_vertex_array(0);
    g_gl_bind_buffer(0x8892 /* GL_ARRAY_BUFFER */, 0);
}

static void display_gl_reset_rgba_texture_unpack(int alignment)
{
    if (alignment != 1 && alignment != 2 && alignment != 4 && alignment != 8)
        alignment = 4;
    if (g_gl_bind_buffer)
        g_gl_bind_buffer(0x88EC /* GL_PIXEL_UNPACK_BUFFER */, 0);
    if (g_gl_pixel_storei) {
        g_gl_pixel_storei(GL_UNPACK_ALIGNMENT, alignment);
        g_gl_pixel_storei(0x0CF2 /* GL_UNPACK_ROW_LENGTH */, 0);
        g_gl_pixel_storei(0x0CF3 /* GL_UNPACK_SKIP_ROWS */, 0);
        g_gl_pixel_storei(0x0CF4 /* GL_UNPACK_SKIP_PIXELS */, 0);
        g_gl_pixel_storei(0x806E /* GL_UNPACK_IMAGE_HEIGHT */, 0);
        g_gl_pixel_storei(0x806D /* GL_UNPACK_SKIP_IMAGES */, 0);
    }
}

static void display_gl_texture_blit(SDL_Texture *tex, const SDL_Rect *src_opt, const SDL_Rect *dst)
{
    if (!tex || !dst || g_gl_overlay_win_w < 1) return;
#if SDL_VERSION_ATLEAST(2, 0, 14)
    SDL_RenderFlush(g_sdl_ren);
#endif
    float tw_s, th_s;
    if (SDL_GL_BindTexture(tex, &tw_s, &th_s) != 0)
        return;
    (void)tw_s;
    (void)th_s;
    int tw = 0, th = 0;
    if (SDL_QueryTexture(tex, NULL, NULL, &tw, &th) != 0 || tw < 1 || th < 1) {
        SDL_GL_UnbindTexture(tex);
        return;
    }
    SDL_Rect src;
    if (src_opt) {
        src = *src_opt;
    } else {
        src.x = 0;
        src.y = 0;
        src.w = tw;
        src.h = th;
    }
    Uint8 cr = 255, cg = 255, cb = 255, ca = 255;
    SDL_GetTextureColorMod(tex, &cr, &cg, &cb);
    SDL_GetTextureAlphaMod(tex, &ca);
    float fr = cr / 255.0f, fg = cg / 255.0f, fb = cb / 255.0f, fa = ca / 255.0f;

    float u0 = (float)src.x / (float)tw;
    float v0 = (float)src.y / (float)th;
    float u1 = (float)(src.x + src.w) / (float)tw;
    float v1 = (float)(src.y + src.h) / (float)th;

    int wx = g_gl_overlay_win_w, wy = g_gl_overlay_win_h;
    float nx0, ny0, nx1, ny1, nx2, ny2, nx3, ny3;
    display_gl_wnd_ndc((float)dst->x, (float)dst->y, wx, wy, &nx0, &ny0);
    display_gl_wnd_ndc((float)(dst->x + dst->w), (float)dst->y, wx, wy, &nx1, &ny1);
    display_gl_wnd_ndc((float)(dst->x + dst->w), (float)(dst->y + dst->h), wx, wy, &nx2, &ny2);
    display_gl_wnd_ndc((float)dst->x, (float)(dst->y + dst->h), wx, wy, &nx3, &ny3);

    float buf[24] = {
        nx0, ny0, u0, v0,
        nx1, ny1, u1, v0,
        nx3, ny3, u0, v1,
        nx1, ny1, u1, v0,
        nx2, ny2, u1, v1,
        nx3, ny3, u0, v1,
    };

    g_gl_use_program(g_gl_prog_hud_tex);
    if (g_gl_hud_loc_tex >= 0) g_gl_uniform1i(g_gl_hud_loc_tex, 0);
    if (g_gl_hud_loc_color_tex >= 0) g_gl_uniform4f(g_gl_hud_loc_color_tex, fr, fg, fb, fa);
#if defined(__EMSCRIPTEN__)
    /* SDL_GL_BindTexture + WebGL: SDL's texture bytes are BGRA vs our GL_RGBA gun path. */
    if (g_gl_hud_loc_rb_swap >= 0) g_gl_uniform1f(g_gl_hud_loc_rb_swap, 1.0f);
#else
    if (g_gl_hud_loc_rb_swap >= 0) g_gl_uniform1f(g_gl_hud_loc_rb_swap, 0.0f);
#endif
    g_gl_active_texture(GL_TEXTURE0);
    g_gl_bind_vertex_array(g_gl_hud_vao_tex);
    g_gl_bind_buffer(0x8892, g_gl_hud_vbo);
    g_gl_buffer_data(0x8892, (ptrdiff_t)sizeof(buf), buf, GL_STREAM_DRAW);
    g_gl_draw_arrays(GL_TRIANGLES, 0, 6);
    SDL_GL_UnbindTexture(tex);
}

static void display_gl_texture_id_blit(GLuint tex, const SDL_Rect *dst, Uint8 alpha)
{
    if (!tex || !dst || !g_gl_hud_ok || g_gl_overlay_win_w < 1) return;
#if SDL_VERSION_ATLEAST(2, 0, 14)
    SDL_RenderFlush(g_sdl_ren);
#endif
    int wx = g_gl_overlay_win_w, wy = g_gl_overlay_win_h;
    float nx0, ny0, nx1, ny1, nx2, ny2, nx3, ny3;
    display_gl_wnd_ndc((float)dst->x, (float)dst->y, wx, wy, &nx0, &ny0);
    display_gl_wnd_ndc((float)(dst->x + dst->w), (float)dst->y, wx, wy, &nx1, &ny1);
    display_gl_wnd_ndc((float)(dst->x + dst->w), (float)(dst->y + dst->h), wx, wy, &nx2, &ny2);
    display_gl_wnd_ndc((float)dst->x, (float)(dst->y + dst->h), wx, wy, &nx3, &ny3);

    float buf[24] = {
        nx0, ny0, 0.0f, 0.0f,
        nx1, ny1, 1.0f, 0.0f,
        nx3, ny3, 0.0f, 1.0f,
        nx1, ny1, 1.0f, 0.0f,
        nx2, ny2, 1.0f, 1.0f,
        nx3, ny3, 0.0f, 1.0f,
    };

    g_gl_use_program(g_gl_prog_hud_tex);
    if (g_gl_hud_loc_tex >= 0)       g_gl_uniform1i(g_gl_hud_loc_tex, 0);
    if (g_gl_hud_loc_color_tex >= 0) g_gl_uniform4f(g_gl_hud_loc_color_tex, 1.0f, 1.0f, 1.0f,
                                                   (float)alpha / 255.0f);
    if (g_gl_hud_loc_rb_swap >= 0)   g_gl_uniform1f(g_gl_hud_loc_rb_swap, 0.0f);
    g_gl_active_texture(GL_TEXTURE0);
    g_gl_bind_texture(DGL_TEXTURE_2D, tex);
    g_gl_bind_vertex_array(g_gl_hud_vao_tex);
    g_gl_bind_buffer(0x8892, g_gl_hud_vbo);
    g_gl_buffer_data(0x8892, (ptrdiff_t)sizeof(buf), buf, GL_STREAM_DRAW);
    g_gl_draw_arrays(GL_TRIANGLES, 0, 6);
    g_gl_bind_texture(DGL_TEXTURE_2D, 0);
}

static int display_gl_array_layer_blit(GLuint tex, int layer, int layer_count,
                                       float u0, float v0, float u1, float v1,
                                       const SDL_Rect *dst, Uint8 alpha)
{
    if (!dst || !tex || layer < 0 || layer >= layer_count)
        return 0;
    if (!g_gl_unpack_ok || !g_gl_hud_ok || !g_gl_prog_hud_font ||
        !g_gl_tex_parameteri2 || g_gl_overlay_win_w < 1) {
        return 0;
    }
    if (g_gl_get_error) {
        for (int i = 0; i < 8 && g_gl_get_error() != 0; i++) {
        }
    }

    int wx = g_gl_overlay_win_w, wy = g_gl_overlay_win_h;
    float nx0, ny0, nx1, ny1, nx2, ny2, nx3, ny3;
    display_gl_wnd_ndc((float)dst->x, (float)dst->y, wx, wy, &nx0, &ny0);
    display_gl_wnd_ndc((float)(dst->x + dst->w), (float)dst->y, wx, wy,
                       &nx1, &ny1);
    display_gl_wnd_ndc((float)(dst->x + dst->w),
                       (float)(dst->y + dst->h), wx, wy, &nx2, &ny2);
    display_gl_wnd_ndc((float)dst->x, (float)(dst->y + dst->h), wx, wy,
                       &nx3, &ny3);

    float buf[24] = {
        nx0, ny0, u0, v0,
        nx1, ny1, u1, v0,
        nx3, ny3, u0, v1,
        nx1, ny1, u1, v0,
        nx2, ny2, u1, v1,
        nx3, ny3, u0, v1,
    };

    g_gl_active_texture(GL_TEXTURE0);
    g_gl_bind_texture(DGL_TEXTURE_2D_ARRAY, tex);
    g_gl_tex_parameteri2(DGL_TEXTURE_2D_ARRAY, DGL_TEXTURE_MIN_FILTER,
                         (GLint)DGL_LINEAR_MIPMAP_LINEAR);
    g_gl_tex_parameteri2(DGL_TEXTURE_2D_ARRAY, DGL_TEXTURE_MAG_FILTER,
                         (GLint)GL_NEAREST);
    g_gl_use_program(g_gl_prog_hud_font);
    g_gl_uniform1i(g_gl_hud_loc_font_tex, 0);
    g_gl_uniform4f(g_gl_hud_loc_font_color, 1.0f, 1.0f, 1.0f,
                   (float)alpha / 255.0f);
    g_gl_uniform1f(g_gl_hud_loc_font_layer, (float)layer);
    g_gl_bind_vertex_array(g_gl_hud_vao_tex);
    g_gl_bind_buffer(0x8892, g_gl_hud_vbo);
    g_gl_buffer_data(0x8892, (ptrdiff_t)sizeof(buf), buf, GL_STREAM_DRAW);
    g_gl_draw_arrays(GL_TRIANGLES, 0, 6);
    if (g_gl_get_error && g_gl_get_error() != 0) {
        g_gl_bind_texture(DGL_TEXTURE_2D_ARRAY, 0);
        return 0;
    }
    g_gl_bind_texture(DGL_TEXTURE_2D_ARRAY, 0);
    return 1;
}

static int display_gl_font_glyph_blit(int glyph_idx, const SDL_Rect *dst, Uint8 alpha)
{
    if (glyph_idx < 0 || glyph_idx >= DISPLAY_ASCII_COUNT)
        return 0;
    if (!g_ascii_font_array_ok || !g_ascii_font_array_tex)
        return 0;

    float u0 = (float)DISPLAY_ASCII_DRAW_X / (float)DISPLAY_ASCII_CELL_W;
    float v0 = (float)DISPLAY_ASCII_DRAW_Y / (float)DISPLAY_ASCII_CELL_H;
    float u1 = (float)(DISPLAY_ASCII_DRAW_X + DISPLAY_ASCII_DRAW_W) /
               (float)DISPLAY_ASCII_CELL_W;
    float v1 = (float)(DISPLAY_ASCII_DRAW_Y + DISPLAY_ASCII_DRAW_H) /
               (float)DISPLAY_ASCII_CELL_H;

    return display_gl_array_layer_blit(g_ascii_font_array_tex, glyph_idx,
                                       DISPLAY_ASCII_COUNT, u0, v0, u1, v1,
                                       dst, alpha);
}

static void display_gl_solid_rect_fill(const SDL_Rect *dst, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    if (!dst || g_gl_overlay_win_w < 1) return;
#if SDL_VERSION_ATLEAST(2, 0, 14)
    SDL_RenderFlush(g_sdl_ren);
#endif
    float fr = r / 255.0f, fg = g / 255.0f, fb = b / 255.0f, fa = a / 255.0f;
    int wx = g_gl_overlay_win_w, wy = g_gl_overlay_win_h;
    float nx0, ny0, nx1, ny1, nx2, ny2, nx3, ny3;
    display_gl_wnd_ndc((float)dst->x, (float)dst->y, wx, wy, &nx0, &ny0);
    display_gl_wnd_ndc((float)(dst->x + dst->w), (float)dst->y, wx, wy, &nx1, &ny1);
    display_gl_wnd_ndc((float)(dst->x + dst->w), (float)(dst->y + dst->h), wx, wy, &nx2, &ny2);
    display_gl_wnd_ndc((float)dst->x, (float)(dst->y + dst->h), wx, wy, &nx3, &ny3);
    float buf[12] = {
        nx0, ny0, nx1, ny1, nx3, ny3,
        nx1, ny1, nx2, ny2, nx3, ny3,
    };
    g_gl_use_program(g_gl_prog_hud_solid);
    if (g_gl_hud_loc_color_solid >= 0) g_gl_uniform4f(g_gl_hud_loc_color_solid, fr, fg, fb, fa);
    g_gl_bind_vertex_array(g_gl_hud_vao_solid);
    g_gl_bind_buffer(0x8892, g_gl_hud_vbo);
    g_gl_buffer_data(0x8892, (ptrdiff_t)sizeof(buf), buf, GL_STREAM_DRAW);
    g_gl_draw_arrays(GL_TRIANGLES, 0, 6);
}

static void display_gl_line(int x0, int y0, int x1, int y1, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    if (g_gl_overlay_win_w < 1) return;
    float fr = r / 255.0f, fg = g / 255.0f, fb = b / 255.0f, fa = a / 255.0f;
    int wx = g_gl_overlay_win_w, wy = g_gl_overlay_win_h;
    float nx0, ny0, nx1, ny1;
    display_gl_wnd_ndc((float)x0, (float)y0, wx, wy, &nx0, &ny0);
    display_gl_wnd_ndc((float)x1, (float)y1, wx, wy, &nx1, &ny1);
    float buf[4] = { nx0, ny0, nx1, ny1 };
    g_gl_use_program(g_gl_prog_hud_solid);
    if (g_gl_hud_loc_color_solid >= 0) g_gl_uniform4f(g_gl_hud_loc_color_solid, fr, fg, fb, fa);
    g_gl_bind_vertex_array(g_gl_hud_vao_solid);
    g_gl_bind_buffer(0x8892, g_gl_hud_vbo);
    g_gl_buffer_data(0x8892, (ptrdiff_t)sizeof(buf), buf, GL_STREAM_DRAW);
    g_gl_draw_arrays(GL_LINES, 0, 2);
}

static int display_gl_lines_ensure_scratch(int nverts)
{
    size_t need_floats;
    size_t new_cap;
    float *new_buf;

    if (nverts <= 0) return 0;
    need_floats = (size_t)nverts * 2u;
    if (g_gl_lines_ndc_scratch && need_floats <= g_gl_lines_ndc_scratch_cap)
        return 1;

    new_cap = (g_gl_lines_ndc_scratch_cap > 0) ? g_gl_lines_ndc_scratch_cap : 1024u;
    while (new_cap < need_floats) {
        if (new_cap > SIZE_MAX / 2u) {
            new_cap = need_floats;
            break;
        }
        new_cap *= 2u;
    }

    new_buf = (float *)realloc(g_gl_lines_ndc_scratch, new_cap * sizeof(*new_buf));
    if (!new_buf) return 0;
    g_gl_lines_ndc_scratch = new_buf;
    g_gl_lines_ndc_scratch_cap = new_cap;
    return 1;
}

static void display_gl_lines_release_scratch(void)
{
    free(g_gl_lines_ndc_scratch);
    g_gl_lines_ndc_scratch = NULL;
    g_gl_lines_ndc_scratch_cap = 0;
}

static void display_gl_lines_xy(const float *xy, int nverts, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    if (!g_gl_hud_ok || g_gl_overlay_win_w < 1 || !xy || nverts < 2 || (nverts & 1) != 0) return;

    int wx = g_gl_overlay_win_w;
    int wy = g_gl_overlay_win_h;
    float fr = r / 255.0f;
    float fg = g / 255.0f;
    float fb = b / 255.0f;
    float fa = a / 255.0f;
    float *buf;
    if (!display_gl_lines_ensure_scratch(nverts)) return;
    buf = g_gl_lines_ndc_scratch;

    for (int i = 0; i < nverts; i++) {
        float nx, ny;
        display_gl_wnd_ndc(xy[i * 2], xy[i * 2 + 1], wx, wy, &nx, &ny);
        buf[i * 2] = nx;
        buf[i * 2 + 1] = ny;
    }

    g_gl_use_program(g_gl_prog_hud_solid);
    if (g_gl_hud_loc_color_solid >= 0) g_gl_uniform4f(g_gl_hud_loc_color_solid, fr, fg, fb, fa);
    g_gl_bind_vertex_array(g_gl_hud_vao_solid);
    g_gl_bind_buffer(0x8892, g_gl_hud_vbo);
    g_gl_buffer_data(0x8892, (ptrdiff_t)((size_t)nverts * 2u * sizeof(float)), buf, GL_STREAM_DRAW);
    g_gl_draw_arrays(GL_LINES, 0, nverts);
}

typedef struct DisplayAutomapGlLineBucket {
    Uint8 r;
    Uint8 g;
    Uint8 b;
    Uint8 a;
    int begin;
    int count;
    int write;
} DisplayAutomapGlLineBucket;

static int display_automap_gl_find_bucket(DisplayAutomapGlLineBucket *buckets, int *bucket_count,
                                          int bucket_cap, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    if (!buckets || !bucket_count || bucket_cap < 1) return -1;
    for (int i = 0; i < *bucket_count; i++) {
        DisplayAutomapGlLineBucket *bucket = &buckets[i];
        if (bucket->r == r && bucket->g == g && bucket->b == b && bucket->a == a)
            return i;
    }
    if (*bucket_count >= bucket_cap) return -1;
    {
        DisplayAutomapGlLineBucket *bucket = &buckets[*bucket_count];
        bucket->r = r;
        bucket->g = g;
        bucket->b = b;
        bucket->a = a;
        bucket->begin = 0;
        bucket->count = 0;
        bucket->write = 0;
    }
    (*bucket_count)++;
    return *bucket_count - 1;
}

static void display_automap_gl_append_bucket_line(float *xy, int max_floats,
                                                  DisplayAutomapGlLineBucket *bucket,
                                                  float x0, float y0, float x1, float y1)
{
    if (!xy || !bucket) return;
    if (bucket->write > max_floats - 4) return;
    xy[bucket->write++] = x0;
    xy[bucket->write++] = y0;
    xy[bucket->write++] = x1;
    xy[bucket->write++] = y1;
}

static void display_automap_gl_prepare_buckets(DisplayAutomapGlLineBucket *buckets, int bucket_count)
{
    int offset = 0;
    for (int i = 0; i < bucket_count; i++) {
        buckets[i].begin = offset;
        buckets[i].write = offset;
        offset += buckets[i].count;
    }
}

/* Window-pixel triangles (solid HUD shader). nverts: 3 or 6 (one or two triangles). */
static void display_gl_solid_triangles_xy(const float *xy, int nverts, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    if (!g_gl_hud_ok || g_gl_overlay_win_w < 1 || !xy || nverts < 3 || nverts > 6 || (nverts % 3) != 0) return;
#if SDL_VERSION_ATLEAST(2, 0, 14)
    SDL_RenderFlush(g_sdl_ren);
#endif
    int wx = g_gl_overlay_win_w, wy = g_gl_overlay_win_h;
    float buf[12];
    for (int i = 0; i < nverts; i++) {
        float nx, ny;
        display_gl_wnd_ndc(xy[i * 2], xy[i * 2 + 1], wx, wy, &nx, &ny);
        buf[i * 2] = nx;
        buf[i * 2 + 1] = ny;
    }
    float fr = r / 255.0f, fg = g / 255.0f, fb = b / 255.0f, fa = a / 255.0f;
    g_gl_use_program(g_gl_prog_hud_solid);
    if (g_gl_hud_loc_color_solid >= 0) g_gl_uniform4f(g_gl_hud_loc_color_solid, fr, fg, fb, fa);
    g_gl_bind_vertex_array(g_gl_hud_vao_solid);
    g_gl_bind_buffer(0x8892, g_gl_hud_vbo);
    g_gl_buffer_data(0x8892, (ptrdiff_t)(nverts * 2 * sizeof(float)), buf, GL_STREAM_DRAW);
    g_gl_draw_arrays(GL_TRIANGLES, 0, nverts);
}

static void display_overlay_copy(SDL_Texture *tex, const SDL_Rect *src_opt, const SDL_Rect *dst)
{
    if (!g_sdl_ren || !tex || !dst) return;
#ifdef AB3D_DOS
    if(dos_check_enabled && !dos_benchmark_frames) {
        int w,h;
        if(src_opt){w=src_opt->w;h=src_opt->h;}
        else if(SDL_QueryTexture(tex,NULL,NULL,&w,&h)<0) {
            fprintf(stderr,"DOS UI texture query failed: %s\n",SDL_GetError());exit(1);
        }
        if(w!=dst->w || h!=dst->h) {
            fprintf(stderr,"DOS UI attempted scaling %dx%d to %dx%d\n",w,h,dst->w,dst->h);exit(1);
        }
        /* Native title artwork is deliberately cropped; controls must fit. */
        if(tex!=g_title_screen_tex && (dst->x<0 || dst->y<0 ||
           dst->x+dst->w>g_present_width || dst->y+dst->h>g_present_height)) {
            fprintf(stderr,"DOS UI blit outside %dx%d: %d,%d %dx%d\n",
                g_present_width,g_present_height,dst->x,dst->y,dst->w,dst->h);exit(1);
        }
    }
#endif
    if (g_gl_unpack_ok && g_gl_hud_ok)
        display_gl_texture_blit(tex, src_opt, dst);
    else
        SDL_RenderCopy(g_sdl_ren, tex, src_opt, dst);
}


static void display_overlay_fill_rect_abs(const SDL_Rect *rect, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    if (!g_sdl_ren || !rect) return;
#ifdef AB3D_DOS
    if (!a) return;
    if(dos_check_enabled && !dos_benchmark_frames && (rect->x<0 || rect->y<0 ||
       rect->x+rect->w>g_present_width || rect->y+rect->h>g_present_height)) {
        fprintf(stderr,"DOS UI fill outside %dx%d: %d,%d %dx%d\n",
            g_present_width,g_present_height,rect->x,rect->y,rect->w,rect->h);exit(1);
    }
    a = 255;
#endif
    if (g_gl_unpack_ok && g_gl_hud_ok)
        display_gl_solid_rect_fill(rect, r, g, b, a);
    else {
#ifdef AB3D_DOS
        SDL_SetRenderDrawBlendMode(g_sdl_ren, SDL_BLENDMODE_NONE);
#else
        SDL_SetRenderDrawBlendMode(g_sdl_ren, SDL_BLENDMODE_BLEND);
#endif
        SDL_SetRenderDrawColor(g_sdl_ren, r, g, b, a);
        SDL_RenderFillRect(g_sdl_ren, rect);
    }
}

#ifndef AB3D_DOS
static unsigned char *display_load_png_rgba_asset(const char *rel_path,
                                                  int *out_w, int *out_h,
                                                  const char *label)
{
    char *base;
    char path[1024];
    int w = 0, h = 0, comp = 0;
    unsigned char *rgba = NULL;

    if (!rel_path || !*rel_path) return NULL;

#ifdef AB3D_DOS
    rel_path = dos_asset_alias(rel_path);
#endif
    base = SDL_GetBasePath();
    if (base) {
        int plen = snprintf(path, sizeof(path), "%s%s", base, rel_path);
        if (plen > 0 && plen < (int)sizeof(path))
            rgba = stbi_load(path, &w, &h, &comp, 4);
        SDL_free(base);
    }
    if (!rgba) {
        int plen = snprintf(path, sizeof(path), "%s", rel_path);
        if (plen > 0 && plen < (int)sizeof(path))
            rgba = stbi_load(path, &w, &h, &comp, 4);
    }
    if (!rgba) {
        printf("[DISPLAY] %s: could not load %s (%s)\n",
               label ? label : "PNG",
               rel_path,
               stbi_failure_reason() ? stbi_failure_reason() : "?");
        return NULL;
    }
    if (w < 1 || h < 1) {
        stbi_image_free(rgba);
        return NULL;
    }

    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return rgba;
}

#endif

static FILE *display_open_asset_file(const char *rel_path,char *path,size_t path_size,const char *label);
static SDL_Texture *display_load_ui_image_asset(const char *rel_path, int *out_w, int *out_h,
                                                  const char *label)
{
#ifdef AB3D_DOS
    char path[1024];unsigned char header[12];
    FILE *file=display_open_asset_file(rel_path,path,sizeof(path),label);
    if(!file || fread(header,1,12,file)!=12 || memcmp(header,"ABUI",4)) {
        fprintf(stderr,"Invalid/missing native DOS UI image: %s\n",rel_path);exit(1);
    }
    unsigned w=header[4]|(header[5]<<8),h=header[6]|(header[7]<<8);
    unsigned flags=header[8]|(header[9]<<8),key=header[10]|(header[11]<<8);
    if(!w || !h || flags>1){fprintf(stderr,"Invalid DOS UI dimensions/flags: %s\n",rel_path);exit(1);}
    DosUiImage *image=dos_ui_create_image(w,h,SDL_TEXTUREACCESS_STATIC);
    if(!image){fprintf(stderr,"Cannot allocate DOS UI image: %s (%s)\n",rel_path,SDL_GetError());exit(1);}
    for(unsigned y=0;y<h;y++)if(fread((uint8_t *)image->normal->pixels+y*image->normal->pitch,2,w,file)!=w) {
        fprintf(stderr,"Truncated DOS UI image: %s\n",rel_path);exit(1);
    }
    if(fgetc(file)!=EOF || ferror(file) || !SDL_SetSurfaceColorKey(image->normal,flags!=0,key) ||
       !SDL_SetSurfaceBlendMode(image->normal,SDL_BLENDMODE_NONE) || !SDL_SetSurfaceRLE(image->normal,true)) {
        fprintf(stderr,"DOS UI image format/setup failed: %s\n",rel_path);exit(1);
    }
    fclose(file);if(out_w)*out_w=w;if(out_h)*out_h=h;return image;
#else
    int w = 0, h = 0;
    unsigned char *rgba = NULL;

    if (!g_sdl_ren || !rel_path || !*rel_path) return NULL;

    rgba = display_load_png_rgba_asset(rel_path, &w, &h, label);
    if (!rgba) return NULL;

    size_t npix = (size_t)w * (size_t)h;
    uint32_t *argb = (uint32_t *)malloc(npix * sizeof(uint32_t));
    if (!argb) {
        stbi_image_free(rgba);
        return NULL;
    }
    for (size_t p = 0; p < npix; p++) {
        unsigned char r = rgba[p * 4 + 0];
        unsigned char g = rgba[p * 4 + 1];
        unsigned char b = rgba[p * 4 + 2];
        unsigned char a = rgba[p * 4 + 3];
        argb[p] = ((uint32_t)a << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
    }
    stbi_image_free(rgba);

    SDL_Texture *t = SDL_CreateTexture(g_sdl_ren, SDL_PIXELFORMAT_ARGB8888,
                                       SDL_TEXTUREACCESS_STATIC, w, h);
    if (!t || SDL_UpdateTexture(t, NULL, argb, (int)(w * (int)sizeof(uint32_t))) != 0) {
        if (t) SDL_DestroyTexture(t);
        free(argb);
        printf("[DISPLAY] %s: SDL texture failed for %s\n",
               label ? label : "PNG", rel_path);
        return NULL;
    }
    free(argb);

    SDL_SetTextureBlendMode(t, SDL_BLENDMODE_BLEND);
#if SDL_VERSION_ATLEAST(2, 0, 12)
    SDL_SetTextureScaleMode(t, SDL_ScaleModeNearest);
#endif
    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return t;
#endif
}

static FILE *display_open_asset_file(const char *rel_path, char *path,
                                     size_t path_size, const char *label)
{
    FILE *f = NULL;
    char *base;

    if (!rel_path || !*rel_path || !path || path_size < 1) return NULL;
    path[0] = '\0';

#ifdef AB3D_DOS
    rel_path = dos_asset_alias(rel_path);
#endif
    base = SDL_GetBasePath();
    if (base) {
        int plen = snprintf(path, path_size, "%s%s", base, rel_path);
        if (plen > 0 && plen < (int)path_size)
            f = fopen(path, "rb");
        SDL_free(base);
    }
    if (!f) {
        int plen = snprintf(path, path_size, "%s", rel_path);
        if (plen > 0 && plen < (int)path_size)
            f = fopen(path, "rb");
    }

    if (!f) {
        printf("[DISPLAY] %s: could not open %s\n",
               label ? label : "asset", rel_path);
    }
    return f;
}

static uint8_t *display_load_asset_exact(const char *rel_path,
                                         size_t expected_size,
                                         const char *label)
{
    char path[1024];
    FILE *f;
    long len;
    uint8_t *data;

    f = display_open_asset_file(rel_path, path, sizeof(path), label);
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    len = ftell(f);
    if (len < 0 || (size_t)len != expected_size) {
        printf("[DISPLAY] %s: expected %u bytes, got %ld (%s)\n",
               label ? label : "asset",
               (unsigned)expected_size,
               len,
               path);
        fclose(f);
        return NULL;
    }
    rewind(f);

    data = (uint8_t *)malloc(expected_size);
    if (!data) {
        fclose(f);
        return NULL;
    }
    if (fread(data, 1, expected_size, f) != expected_size) {
        printf("[DISPLAY] %s: read failed (%s)\n",
               label ? label : "asset", path);
        free(data);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return data;
}

static SDL_Rect display_fit_rect(int win_w, int win_h, int src_w, int src_h)
{
    SDL_Rect dst;
#ifdef AB3D_DOS
    dst.w=src_w;dst.h=src_h;
    dst.x=(win_w-src_w)/2;dst.y=(win_h-src_h)/2;
    return dst;
#endif
    double sx;
    double sy;
    double sc;

    if (win_w < 1) win_w = 1;
    if (win_h < 1) win_h = 1;
    if (src_w < 1) src_w = 1;
    if (src_h < 1) src_h = 1;

    sx = (double)win_w / (double)src_w;
    sy = (double)win_h / (double)src_h;
    sc = (sx < sy) ? sx : sy;

    dst.w = (int)((double)src_w * sc + 0.5);
    dst.h = (int)((double)src_h * sc + 0.5);
    if (dst.w < 1) dst.w = 1;
    if (dst.h < 1) dst.h = 1;
    if (dst.w > win_w) dst.w = win_w;
    if (dst.h > win_h) dst.h = win_h;
    dst.x = (win_w - dst.w) / 2;
    dst.y = (win_h - dst.h) / 2;
    return dst;
}

#ifdef AB3D_DOS
/* Artwork can be taller than the VESA mode. Text uses only its visible area. */
static SDL_Rect display_dos_visible_text_rect(SDL_Rect r)
{
    int right=r.x+r.w,bottom=r.y+r.h;
    if(r.x<0)r.x=0;
    if(r.y<0)r.y=0;
    if(right>g_present_width)right=g_present_width;
    if(bottom>g_present_height)bottom=g_present_height;
    r.w=right-r.x;r.h=bottom-r.y;
    return r;
}
#endif

static void display_title_screen_free(void)
{
    if (g_title_screen_gl_tex && g_gl_delete_textures) {
        g_gl_delete_textures(1, &g_title_screen_gl_tex);
    }
    g_title_screen_gl_tex = 0;

    if (g_title_screen_tex) {
        SDL_DestroyTexture(g_title_screen_tex);
        g_title_screen_tex = NULL;
    }
    g_title_screen_load_attempted = 0;
}

static uint8_t display_title_palette_component_to_rgb(int component)
{
    if (component < 0) component = 0;
    if (component > 255) component = 255;

    /* ControlLoop.s PUTIN32 multiplies each 8-bit component by FADEVAL and
     * shifts by 8 before splitting high/low nibbles into AGA color banks.
     * FadeUpTitle's full title display ends at FADEVAL=252. */
    component = (component * DISPLAY_TITLE_FULL_FADE_SCALE) >> 8;
    if (component < 0) component = 0;
    if (component > 255) component = 255;
    return (uint8_t)component;
}

static void display_title_screen_upload_gl_texture(const uint32_t *argb)
{
    if (!argb || !g_gl_unpack_ok) return;
    if (!g_gl_gen_textures || !g_gl_bind_texture || !g_gl_tex_image_2d ||
        !g_gl_tex_parameteri2 || !g_gl_delete_textures) {
        return;
    }

    size_t pixels = (size_t)DISPLAY_TITLE_W * (size_t)DISPLAY_TITLE_H;
    uint8_t *rgba = (uint8_t *)malloc(pixels * 4u);
    if (!rgba) return;

    for (size_t i = 0; i < pixels; i++) {
        uint32_t c = argb[i];
        rgba[i * 4u + 0u] = (uint8_t)((c >> 16) & 255u);
        rgba[i * 4u + 1u] = (uint8_t)((c >> 8) & 255u);
        rgba[i * 4u + 2u] = (uint8_t)(c & 255u);
        rgba[i * 4u + 3u] = (uint8_t)((c >> 24) & 255u);
    }

    if (g_title_screen_gl_tex)
        g_gl_delete_textures(1, &g_title_screen_gl_tex);
    g_title_screen_gl_tex = 0;

    if (g_gl_get_error) {
        for (int i = 0; i < 8 && g_gl_get_error() != 0; i++) {
        }
    }

    g_gl_gen_textures(1, &g_title_screen_gl_tex);
    if (g_title_screen_gl_tex) {
        g_gl_active_texture(GL_TEXTURE0);
        g_gl_bind_texture(DGL_TEXTURE_2D, g_title_screen_gl_tex);
        g_gl_tex_parameteri2(DGL_TEXTURE_2D, DGL_TEXTURE_MAG_FILTER, (GLint)GL_NEAREST);
        g_gl_tex_parameteri2(DGL_TEXTURE_2D, DGL_TEXTURE_MIN_FILTER, (GLint)GL_NEAREST);
        g_gl_tex_parameteri2(DGL_TEXTURE_2D, DGL_TEXTURE_WRAP_S, (GLint)DGL_CLAMP_TO_EDGE);
        g_gl_tex_parameteri2(DGL_TEXTURE_2D, DGL_TEXTURE_WRAP_T, (GLint)DGL_CLAMP_TO_EDGE);
        display_gl_reset_rgba_texture_unpack(1);
        g_gl_tex_image_2d(DGL_TEXTURE_2D, 0, (GLint)GL_RGBA8,
                          DISPLAY_TITLE_W, DISPLAY_TITLE_H, 0,
                          GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        display_gl_reset_rgba_texture_unpack(4);
        g_gl_bind_texture(DGL_TEXTURE_2D, 0);
    }

    if (g_gl_get_error && g_gl_get_error() != 0) {
        g_gl_delete_textures(1, &g_title_screen_gl_tex);
        g_title_screen_gl_tex = 0;
        printf("[DISPLAY] title screen: GL texture upload failed\n");
    }

    free(rgba);
}

#ifdef AB3D_DOS
static void display_title_screen_ensure_loaded(void)
{
    if(g_title_screen_load_attempted || !g_sdl_ren)return;
    g_title_screen_load_attempted=1;
    char path[1024];
    FILE *file=display_open_asset_file(DISPLAY_TITLE_RAW_PATH,path,sizeof(path),"title raw");
    uint8_t *pal=display_load_asset_exact(DISPLAY_TITLE_PAL_PATH,DISPLAY_TITLE_PAL_BYTES,"title palette");
    uint8_t *plane=malloc(DISPLAY_TITLE_PLANE_BYTES);
    DosUiImage *title=dos_ui_create_striped_image(DISPLAY_TITLE_W,DISPLAY_TITLE_H);
    if(!file || !pal || !plane || !title){fprintf(stderr,"Cannot load native DOS title assets\n");exit(1);}
    uint16_t colors[1<<DISPLAY_TITLE_PLANES];
    for(unsigned index=0;index<(1u<<DISPLAY_TITLE_PLANES);index++) {
        size_t off=index*DISPLAY_TITLE_PAL_ENTRY_BYTES;
        unsigned r=display_title_palette_component_to_rgb((pal[off]<<8)|pal[off+1]);
        unsigned g=display_title_palette_component_to_rgb(pal[off+2]);
        unsigned b=display_title_palette_component_to_rgb(pal[off+3]);
        colors[index]=(uint16_t)(((r>>3)<<11)|((g>>2)<<5)|(b>>3));
    }
    /* ControlLoop.s's original planar data: read one plane at a time into
     * the final surface, then resolve its indices through the same palette. */
    for(int bit=0;bit<DISPLAY_TITLE_PLANES;bit++) {
        if(fread(plane,1,DISPLAY_TITLE_PLANE_BYTES,file)!=DISPLAY_TITLE_PLANE_BYTES) {
            fprintf(stderr,"Truncated DOS title plane %d\n",bit);exit(1);
        }
        for(int y=0;y<DISPLAY_TITLE_H;y++) {
            uint16_t *row=(uint16_t *)((uint8_t *)title->strips[y/16]->pixels+(y%16)*title->strips[y/16]->pitch);
            for(int x=0;x<DISPLAY_TITLE_W;x++)
                row[x]|=((plane[y*DISPLAY_TITLE_ROW_BYTES+(x>>3)]>>(7-(x&7)))&1u)<<bit;
        }
    }
    for(int y=0;y<DISPLAY_TITLE_H;y++) {
        uint16_t *row=(uint16_t *)((uint8_t *)title->strips[y/16]->pixels+(y%16)*title->strips[y/16]->pitch);
        for(int x=0;x<DISPLAY_TITLE_W;x++)row[x]=colors[row[x]];
    }
    if(fgetc(file)!=EOF || ferror(file)){fprintf(stderr,"DOS title raw size/read mismatch\n");exit(1);}
    fclose(file);free(plane);free(pal);g_title_screen_tex=title;
}
#else
static void display_title_screen_ensure_loaded(void)
{
    uint8_t *raw;
    uint8_t *pal;
    uint32_t *argb;

    if (g_title_screen_load_attempted || !g_sdl_ren) return;
    g_title_screen_load_attempted = 1;

    raw = display_load_asset_exact(DISPLAY_TITLE_RAW_PATH,
                                   DISPLAY_TITLE_RAW_BYTES,
                                   "title screen raw");
    pal = display_load_asset_exact(DISPLAY_TITLE_PAL_PATH,
                                   DISPLAY_TITLE_PAL_BYTES,
                                   "title screen palette");
    if (!raw || !pal) {
        free(raw);
        free(pal);
        return;
    }

    argb = (uint32_t *)malloc((size_t)DISPLAY_TITLE_W *
                              (size_t)DISPLAY_TITLE_H *
                              sizeof(uint32_t));
    if (!argb) {
        free(raw);
        free(pal);
        return;
    }

    for (int y = 0; y < DISPLAY_TITLE_H; y++) {
        for (int x = 0; x < DISPLAY_TITLE_W; x++) {
            int byte_off = y * DISPLAY_TITLE_ROW_BYTES + (x >> 3);
            int bit = 7 - (x & 7);
            int index = 0;

            for (int plane = 0; plane < DISPLAY_TITLE_PLANES; plane++) {
                const uint8_t *plane_base =
                    raw + (size_t)plane * DISPLAY_TITLE_PLANE_BYTES;
                index |= ((plane_base[byte_off] >> bit) & 1) << plane;
            }

            {
                size_t pal_off = (size_t)index * DISPLAY_TITLE_PAL_ENTRY_BYTES;
                uint32_t r = display_title_palette_component_to_rgb(
                    ((int)pal[pal_off] << 8) | (int)pal[pal_off + 1]);
                uint32_t g = display_title_palette_component_to_rgb(
                    (int)pal[pal_off + 2]);
                uint32_t b = display_title_palette_component_to_rgb(
                    (int)pal[pal_off + 3]);
                argb[(size_t)y * DISPLAY_TITLE_W + (size_t)x] =
                    0xFF000000u | (r << 16) | (g << 8) | b;
            }
        }
    }

    display_title_screen_upload_gl_texture(argb);

    g_title_screen_tex = SDL_CreateTexture(g_sdl_ren, SDL_PIXELFORMAT_ARGB8888,
                                           SDL_TEXTUREACCESS_STATIC,
                                           DISPLAY_TITLE_W, DISPLAY_TITLE_H);
    if (!g_title_screen_tex ||
        SDL_UpdateTexture(g_title_screen_tex, NULL, argb,
                          DISPLAY_TITLE_W * (int)sizeof(uint32_t)) != 0) {
        if (g_title_screen_tex) SDL_DestroyTexture(g_title_screen_tex);
        g_title_screen_tex = NULL;
        printf("[DISPLAY] title screen: SDL texture upload failed\n");
    } else {
        SDL_SetTextureBlendMode(g_title_screen_tex, SDL_BLENDMODE_BLEND);
#if SDL_VERSION_ATLEAST(2, 0, 12)
        SDL_SetTextureScaleMode(g_title_screen_tex, SDL_ScaleModeNearest);
#endif
        printf("[DISPLAY] Title screen loaded from %s / %s\n",
               DISPLAY_TITLE_RAW_PATH, DISPLAY_TITLE_PAL_PATH);
    }

    free(argb);
    free(raw);
    free(pal);
}
#endif

static void display_ascii_font_free(void)
{
    if (g_ascii_font_array_tex && g_gl_delete_textures) {
        g_gl_delete_textures(1, &g_ascii_font_array_tex);
    }
    g_ascii_font_array_tex = 0;
    g_ascii_font_array_ok = 0;
    g_ascii_font_array_upload_attempted = 0;

    if (g_ascii_font_tex) {
        SDL_DestroyTexture(g_ascii_font_tex);
        g_ascii_font_tex = NULL;
    }
    g_ascii_font_tex_w = 0;
    g_ascii_font_tex_h = 0;
    g_ascii_font_load_attempted = 0;
}

static void display_ascii_font_upload_array_texture(void)
{
#if !defined(AB3D_DOS) && SDL_VERSION_ATLEAST(2, 0, 12)
    int atlas_w = 0, atlas_h = 0;
    unsigned char *atlas = NULL;
    unsigned char *layers = NULL;
    size_t layer_px = (size_t)DISPLAY_ASCII_CELL_W *
                      (size_t)DISPLAY_ASCII_CELL_H;
    size_t layer_bytes = layer_px * 4u;
    size_t total_bytes = layer_bytes * (size_t)DISPLAY_ASCII_COUNT;

    g_ascii_font_array_ok = 0;
    if (!g_gl_unpack_ok || !g_gl_hud_ok || !g_gl_prog_hud_font)
        return;
    if (!g_gl_gen_textures || !g_gl_bind_texture || !g_gl_tex_image_3d ||
        !g_gl_tex_parameteri2 || !g_gl_delete_textures || !g_gl_pixel_storei)
        return;
    if (g_ascii_font_array_upload_attempted)
        return;
    g_ascii_font_array_upload_attempted = 1;

    if (!g_gl_generate_mipmap)
        display_load_gl_mipmap_procs();
    if (!g_gl_generate_mipmap)
        return;

    atlas = display_load_png_rgba_asset(DISPLAY_ASCII_FONT_PATH,
                                        &atlas_w, &atlas_h,
                                        "ASCII font");
    if (!atlas)
        return;
    if (atlas_w != DISPLAY_ASCII_COLS * DISPLAY_ASCII_CELL_W ||
        atlas_h < DISPLAY_ASCII_CELL_H) {
        stbi_image_free(atlas);
        return;
    }

    layers = (unsigned char *)malloc(total_bytes);
    if (!layers) {
        stbi_image_free(atlas);
        return;
    }
    memset(layers, 0, total_bytes);

    for (int glyph = 0; glyph < DISPLAY_ASCII_COUNT; glyph++) {
        int col = glyph % DISPLAY_ASCII_COLS;
        int row = glyph / DISPLAY_ASCII_COLS;
        int src_x = col * DISPLAY_ASCII_CELL_W;
        int src_y = row * DISPLAY_ASCII_CELL_H;
        if (src_y + DISPLAY_ASCII_CELL_H > atlas_h)
            continue;

        unsigned char *dst_layer = layers + (size_t)glyph * layer_bytes;
        for (int y = 0; y < DISPLAY_ASCII_CELL_H; y++) {
            const unsigned char *src =
                atlas + ((size_t)(src_y + y) * (size_t)atlas_w +
                         (size_t)src_x) * 4u;
            unsigned char *dst =
                dst_layer + (size_t)y * (size_t)DISPLAY_ASCII_CELL_W * 4u;
            memcpy(dst, src, (size_t)DISPLAY_ASCII_CELL_W * 4u);
        }
    }

    if (g_ascii_font_array_tex) {
        g_gl_delete_textures(1, &g_ascii_font_array_tex);
        g_ascii_font_array_tex = 0;
    }

    if (g_gl_get_error) {
        for (int i = 0; i < 8 && g_gl_get_error() != 0; i++) {
        }
    }

    g_gl_gen_textures(1, &g_ascii_font_array_tex);
    if (!g_ascii_font_array_tex) {
        free(layers);
        stbi_image_free(atlas);
        return;
    }
    g_gl_bind_texture(DGL_TEXTURE_2D_ARRAY, g_ascii_font_array_tex);
    display_gl_reset_rgba_texture_unpack(1);
    g_gl_tex_parameteri2(DGL_TEXTURE_2D_ARRAY, DGL_TEXTURE_MIN_FILTER,
                         (GLint)DGL_LINEAR_MIPMAP_LINEAR);
    g_gl_tex_parameteri2(DGL_TEXTURE_2D_ARRAY, DGL_TEXTURE_MAG_FILTER,
                         (GLint)GL_NEAREST);
    g_gl_tex_parameteri2(DGL_TEXTURE_2D_ARRAY, DGL_TEXTURE_WRAP_S,
                         (GLint)DGL_CLAMP_TO_EDGE);
    g_gl_tex_parameteri2(DGL_TEXTURE_2D_ARRAY, DGL_TEXTURE_WRAP_T,
                         (GLint)DGL_CLAMP_TO_EDGE);
    g_gl_tex_image_3d(DGL_TEXTURE_2D_ARRAY, 0, (GLint)GL_RGBA8,
                      DISPLAY_ASCII_CELL_W, DISPLAY_ASCII_CELL_H,
                      DISPLAY_ASCII_COUNT, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                      layers);
    g_gl_generate_mipmap(DGL_TEXTURE_2D_ARRAY);
    display_gl_reset_rgba_texture_unpack(4);
    g_gl_bind_texture(DGL_TEXTURE_2D_ARRAY, 0);

    if (g_gl_get_error && g_gl_get_error() != 0) {
        g_gl_delete_textures(1, &g_ascii_font_array_tex);
        g_ascii_font_array_tex = 0;
    } else {
        g_ascii_font_array_ok = 1;
        printf("[DISPLAY] ASCII font: GL texture array mipmaps active\n");
    }

    free(layers);
    stbi_image_free(atlas);
#else
    (void)0;
#endif
}

static void display_ascii_font_enable_scaling(void)
{
    if (!g_ascii_font_tex) return;

#if SDL_VERSION_ATLEAST(2, 0, 12)
    SDL_SetTextureScaleMode(g_ascii_font_tex, SDL_ScaleModeNearest);
    display_ascii_font_upload_array_texture();
    if (g_gl_unpack_ok && g_gl_hud_ok && g_ascii_font_array_ok)
        return;
#endif
}

static void display_ascii_font_ensure_loaded(void)
{
    if (g_ascii_font_load_attempted || !g_sdl_ren) return;
    g_ascii_font_load_attempted = 1;

    g_ascii_font_tex = display_load_ui_image_asset(DISPLAY_ASCII_FONT_PATH,
                                                     &g_ascii_font_tex_w,
                                                     &g_ascii_font_tex_h,
                                                     "ASCII font");
    if (g_ascii_font_tex &&
        (g_ascii_font_tex_w != DISPLAY_ASCII_COLS * DISPLAY_ASCII_CELL_W ||
         g_ascii_font_tex_h < DISPLAY_ASCII_CELL_H)) {
        printf("[DISPLAY] ASCII font: unexpected atlas size %dx%d for %s\n",
               g_ascii_font_tex_w, g_ascii_font_tex_h, DISPLAY_ASCII_FONT_PATH);
    }
    display_ascii_font_enable_scaling();
}

typedef struct DisplayTextRun {
    const char *text;
    int start;
    int len;
    int source_line;
    Uint8 alpha;
} DisplayTextRun;

static int display_ascii_text_is_blank(const char *text)
{
    if (!text) return 1;
    while (*text) {
        if (*text != ' ' && *text != '\t')
            return 0;
        text++;
    }
    return 1;
}

static int display_text_has_lines(void)
{
    for (int i = 0; i < DISPLAY_TEXT_MAX_LINES; i++) {
        if (g_text_line_used[i] && !display_ascii_text_is_blank(g_text_lines[i]))
            return 1;
    }
    return 0;
}

static int display_text_scaled_px(int px, int scale_q)
{
    int scaled = (px * scale_q + DISPLAY_ASCII_SCALE_ONE / 2) / DISPLAY_ASCII_SCALE_ONE;
    return (scaled > 0) ? scaled : 1;
}

static int display_text_crisp_scale_q(int scale_q)
{
    if (scale_q <= DISPLAY_ASCII_SCALE_ONE)
        return scale_q;

    /* Fractional nearest-neighbor upscales make source pixels land at uneven
     * widths. Keep magnification to whole numbers, while sub-1x scales still
     * use mipmapped minification. */
    int whole = scale_q / DISPLAY_ASCII_SCALE_ONE;
    if (whole < 1) whole = 1;
    return whole * DISPLAY_ASCII_SCALE_ONE;
}

static int display_text_span_width_px_q(int len, int scale_q)
{
    if (len < 1 || scale_q < 1) return 0;
    return len * display_text_scaled_px(DISPLAY_ASCII_XADVANCE, scale_q);
}

static void display_bitmap_text_span_abs(const char *text, int start, int len,
                                         int x, int y, int scale_q, Uint8 alpha)
{
    if (!text || len < 1 || scale_q < 1) return;
    display_ascii_font_ensure_loaded();
    if (!g_ascii_font_tex) return;

    SDL_SetTextureColorMod(g_ascii_font_tex, 255, 255, 255);
    SDL_SetTextureAlphaMod(g_ascii_font_tex, alpha);

    int draw_w = display_text_scaled_px(DISPLAY_ASCII_DRAW_W, scale_q);
    int draw_h = display_text_scaled_px(DISPLAY_ASCII_DRAW_H, scale_q);
    int advance = display_text_scaled_px(DISPLAY_ASCII_XADVANCE, scale_q);

    int pen_x = x;
    for (int i = 0; i < len; i++) {
        unsigned char ch = (unsigned char)text[start + i];
        if (ch < DISPLAY_ASCII_FIRST || ch > DISPLAY_ASCII_LAST)
            ch = (unsigned char)'?';

        if (ch != (unsigned char)' ') {
            int idx = (int)ch - DISPLAY_ASCII_FIRST;
            int col = idx % DISPLAY_ASCII_COLS;
            int row = idx / DISPLAY_ASCII_COLS;
            SDL_Rect src;
            SDL_Rect dst;
            src.x = col * DISPLAY_ASCII_CELL_W + DISPLAY_ASCII_DRAW_X;
            src.y = row * DISPLAY_ASCII_CELL_H + DISPLAY_ASCII_DRAW_Y;
            src.w = DISPLAY_ASCII_DRAW_W;
            src.h = DISPLAY_ASCII_DRAW_H;
            dst.x = pen_x;
            dst.y = y;
            dst.w = draw_w;
            dst.h = draw_h;
            if (!display_gl_font_glyph_blit(idx, &dst, alpha)) {
                display_overlay_copy(g_ascii_font_tex, &src, &dst);
            }
        }

        pen_x += advance;
    }
}

static int display_text_margin_for_extent(int extent)
{
    int margin = extent / 24;
    if (margin < 8) margin = 8;
    if (margin > 64) margin = 64;
    if (extent < margin * 2 + 1) margin = (extent > 2) ? extent / 8 : 0;
    return margin;
}

static void display_text_trim_span(const char *text, int *start, int *len)
{
    int s = 0;
    int e = text ? (int)strlen(text) : 0;
    while (s < e && (text[s] == ' ' || text[s] == '\t')) s++;
    while (e > s && (text[e - 1] == ' ' || text[e - 1] == '\t')) e--;
    *start = s;
    *len = e - s;
}

static int display_text_append_run(DisplayTextRun *runs, int max_runs,
                                   int *count, const char *text, int start,
                                   int len, int source_line, Uint8 alpha)
{
    if (*count >= max_runs) return 0;
    runs[*count].text = text;
    runs[*count].start = start;
    runs[*count].len = len;
    runs[*count].source_line = source_line;
    runs[*count].alpha = alpha;
    (*count)++;
    return 1;
}

static int display_text_build_line_layout(DisplayTextRun *runs, int max_runs,
                                          int *out_longest_chars, int wrap_cols)
{
    int first_line = -1;
    int last_line = -1;
    for (int i = 0; i < DISPLAY_TEXT_MAX_LINES; i++) {
        if (g_text_line_used[i] && !display_ascii_text_is_blank(g_text_lines[i])) {
            if (first_line < 0) first_line = i;
            last_line = i;
        }
    }
    if (first_line < 0 || last_line < first_line) return 0;

    int count = 0;
    int longest = 1;
    for (int i = first_line; i <= last_line; i++) {
        int start = 0;
        int len = 0;
        display_text_trim_span(g_text_lines[i], &start, &len);
        if (!g_text_line_used[i] || display_ascii_text_is_blank(g_text_lines[i])) {
            start = 0;
            len = 0;
        }
        while(wrap_cols>0 && len>wrap_cols) {
            int take=wrap_cols;
            while(take>0 && g_text_lines[i][start+take]!=' ')take--;
            if(!take)take=wrap_cols;
            if(!display_text_append_run(runs,max_runs,&count,g_text_lines[i],start,take,i,g_text_line_alpha[i]))return 0;
            if(take>longest)longest=take;
            start+=take;len-=take;
            while(len && g_text_lines[i][start]==' '){start++;len--;}
        }
        if (!display_text_append_run(runs, max_runs, &count, g_text_lines[i],
                                     start, len, i, g_text_line_alpha[i])) {
            return 0;
        }
        if (len > longest) longest = len;
    }

    if (out_longest_chars) *out_longest_chars = longest;
    return count;
}

static int display_text_layout_in_rect(SDL_Rect r, DisplayTextRun *runs,
                                       int max_runs, int *out_run_count,
                                       int *out_scale_q, int *out_line_h,
                                       int *out_draw_h, int *out_top,
                                       int *out_margin_x, int *out_margin_y)
{
    if (!display_text_has_lines() || !runs || max_runs < 1) return 0;

    if (r.w < 1 || r.h < 1) {
        r.x = 0;
        r.y = 0;
        r.w = (g_present_width > 0) ? g_present_width : 640;
        r.h = (g_present_height > 0) ? g_present_height : 480;
    }

    int margin_x = display_text_margin_for_extent(r.w);
    int margin_y = display_text_margin_for_extent(r.h);
    int avail_w = r.w - margin_x * 2;
    int avail_h = r.h - margin_y * 2;
    if (avail_w < DISPLAY_ASCII_XADVANCE) {
        avail_w = r.w;
        margin_x = 0;
    }
    if (avail_h < DISPLAY_ASCII_LINE_ADVANCE) {
        avail_h = r.h;
        margin_y = 0;
    }

    int longest_chars = 1;
    int wrap_cols=0;
#ifdef AB3D_DOS
    wrap_cols=avail_w/DISPLAY_ASCII_XADVANCE;
#endif
    int run_count = display_text_build_line_layout(runs, max_runs,
                                                   &longest_chars,wrap_cols);
    if (run_count < 1) return 0;

#ifdef AB3D_DOS
    /* Native bitmap pixels on DOS, including menus, level text and dialogs. */
    int scale_q=DISPLAY_ASCII_SCALE_ONE;
#else
    int scale_q = DISPLAY_ASCII_SCALE_ONE * 4;
    if (longest_chars > 0) {
        int64_t natural_w = (int64_t)longest_chars *
                            (int64_t)DISPLAY_ASCII_XADVANCE;
        int width_q = (natural_w > 0) ?
            (int)(((int64_t)avail_w * DISPLAY_ASCII_SCALE_ONE) / natural_w) :
            scale_q;
        if (width_q < scale_q) scale_q = width_q;
    }
    if (run_count > 0) {
        int64_t natural_h = (int64_t)run_count *
                            (int64_t)DISPLAY_ASCII_LINE_ADVANCE;
        int height_q = (natural_h > 0) ?
            (int)(((int64_t)avail_h * DISPLAY_ASCII_SCALE_ONE) / natural_h) :
            scale_q;
        if (height_q < scale_q) scale_q = height_q;
    }
    if (scale_q < 1) scale_q = 1;

    while (scale_q > 1) {
        int total_h = run_count *
                      display_text_scaled_px(DISPLAY_ASCII_LINE_ADVANCE,
                                             scale_q);
        int total_w = display_text_span_width_px_q(longest_chars, scale_q);
        if (total_h <= avail_h && total_w <= avail_w)
            break;
        scale_q--;
    }

    scale_q = display_text_crisp_scale_q(scale_q);
#endif

    int line_h = display_text_scaled_px(DISPLAY_ASCII_LINE_ADVANCE, scale_q);
    int draw_h = display_text_scaled_px(DISPLAY_ASCII_DRAW_H, scale_q);
    int total_h = run_count * line_h;
#ifdef AB3D_DOS
    g_dos_text_page_rows=avail_h/line_h;
    if(g_dos_text_page_rows<1)g_dos_text_page_rows=1;
    if(run_count>g_dos_text_page_rows && g_dos_text_page_rows>1)g_dos_text_page_rows--;
    g_dos_text_pages=(run_count+g_dos_text_page_rows-1)/g_dos_text_page_rows;
    if(g_dos_text_page>=g_dos_text_pages)g_dos_text_page=g_dos_text_pages-1;
    int remaining=run_count-g_dos_text_page*g_dos_text_page_rows;
    total_h=(remaining<g_dos_text_page_rows?remaining:g_dos_text_page_rows)*line_h;
    if(g_dos_text_pages>1)total_h+=line_h;
#endif
    int top = r.y + (r.h - total_h) / 2;
    if (top < r.y + margin_y) top = r.y + margin_y;

    if (out_run_count) *out_run_count = run_count;
    if (out_scale_q) *out_scale_q = scale_q;
    if (out_line_h) *out_line_h = line_h;
    if (out_draw_h) *out_draw_h = draw_h;
    if (out_top) *out_top = top;
    if (out_margin_x) *out_margin_x = margin_x;
    if (out_margin_y) *out_margin_y = margin_y;
    return 1;
}

static void display_text_screen_draw_in_rect(Uint8 alpha, SDL_Rect r)
{
    if (!display_text_has_lines()) return;
#ifdef AB3D_DOS
    r=display_dos_visible_text_rect(r);
    if(r.w<1 || r.h<1)return;
#endif
    display_ascii_font_ensure_loaded();
    if (!g_ascii_font_tex) return;

    DisplayTextRun runs[DISPLAY_TEXT_LAYOUT_MAX_RUNS];
    int run_count = 0;
    int scale_q = DISPLAY_ASCII_SCALE_ONE;
    int line_h = DISPLAY_ASCII_LINE_ADVANCE;
    int draw_h = DISPLAY_ASCII_DRAW_H;
    int top = 0;
    int margin_x = 0;
    int margin_y = 0;

    if (!display_text_layout_in_rect(r, runs, DISPLAY_TEXT_LAYOUT_MAX_RUNS,
                                     &run_count, &scale_q, &line_h, &draw_h,
                                     &top, &margin_x, &margin_y)) {
        return;
    }

    int first_run=0,end_run=run_count;
#ifdef AB3D_DOS
    first_run=g_dos_text_page*g_dos_text_page_rows;
    if(end_run>first_run+g_dos_text_page_rows)end_run=first_run+g_dos_text_page_rows;
#endif
    for (int i = first_run; i < end_run; i++) {
        if (runs[i].len < 1) continue;

        int text_w = display_text_span_width_px_q(runs[i].len, scale_q);
        int x = r.x + (r.w - text_w) / 2;
        if (x < r.x + margin_x) x = r.x + margin_x;
        int y = top + (i-first_run) * line_h;
        if (y > r.y + r.h - margin_y - draw_h) continue;
        Uint8 run_alpha = (Uint8)(((int)alpha * (int)runs[i].alpha + 127) / 255);
#ifdef AB3D_DOS
        /* Screen transitions step on/off; only disabled labels use grey. */
        run_alpha = alpha ? (runs[i].alpha < 255 ? 128 : 255) : 0;
#endif
        display_bitmap_text_span_abs(runs[i].text, runs[i].start, runs[i].len,
                                     x, y, scale_q, run_alpha);
    }
#ifdef AB3D_DOS
    if(g_dos_text_pages>1) {
        const char *prompt=g_dos_text_page+1<g_dos_text_pages?"MORE - PRESS A KEY":"PRESS A KEY";
        int len=(int)strlen(prompt);
        display_bitmap_text_span_abs(prompt,0,len,r.x+(r.w-len*DISPLAY_ASCII_XADVANCE)/2,
            r.y+r.h-margin_y-DISPLAY_ASCII_DRAW_H,DISPLAY_ASCII_SCALE_ONE,alpha);
    }
#endif
}

static void display_text_screen_draw(Uint8 alpha)
{
#ifdef AB3D_DOS
    display_text_screen_draw_in_rect(alpha,(SDL_Rect){0,0,g_present_width,g_present_height});
#else
    display_text_screen_draw_in_rect(alpha, g_present_dst_rect);
#endif
}

static void display_automap_line_stroked_gl(int ax0, int ay0, int ax1, int ay1,
                                            Uint8 fr, Uint8 fg, Uint8 fb, Uint8 alpha,
                                            int black_k_max, int fg_k_max)
{
    int dx = ax1 - ax0;
    int dy = ay1 - ay0;
    double len = hypot((double)dx, (double)dy);
    if (len < 1e-6) return;
    double px = -dy / len;
    double py = dx / len;
    /* Do not SDL_RenderFlush here: automap draws thousands of stroked lines; per-line flush
     * stalls the WebGL main thread (tab freeze). Flush once in display_automap_sdl_overlay. */
    if (black_k_max < 0) black_k_max = 0;
    if (fg_k_max < 0) fg_k_max = 0;
    for (int k = -black_k_max; k <= black_k_max; k++) {
        double oxk = k * px, oyk = k * py;
        int ox = (int)(oxk + (oxk >= 0.0 ? 0.5 : -0.5));
        int oy = (int)(oyk + (oyk >= 0.0 ? 0.5 : -0.5));
        display_gl_line(ax0 + ox, ay0 + oy, ax1 + ox, ay1 + oy, 0, 0, 0, alpha);
    }
    for (int k = -fg_k_max; k <= fg_k_max; k++) {
        double oxk = k * px, oyk = k * py;
        int ox = (int)(oxk + (oxk >= 0.0 ? 0.5 : -0.5));
        int oy = (int)(oyk + (oyk >= 0.0 ? 0.5 : -0.5));
        display_gl_line(ax0 + ox, ay0 + oy, ax1 + ox, ay1 + oy, fr, fg, fb, alpha);
    }
}

static void display_gl_shutdown_unpack(void)
{
    if (g_gl_delete_textures) {
        for (int i = 0; i < GUN_GL_FRAME_COUNT; i++) {
            if (g_gun_gl_frames[i]) g_gl_delete_textures(1, &g_gun_gl_frames[i]);
            g_gun_gl_frames[i] = 0;
        }
    }
    display_gl_hud_shutdown();
    if (!g_gl_unpack_ok) return;
    if (g_gl_delete_vertex_arrays && g_gl_vao) g_gl_delete_vertex_arrays(1, &g_gl_vao);
    if (g_gl_delete_buffers && g_gl_vbo) g_gl_delete_buffers(1, &g_gl_vbo);
    if (g_gl_delete_program && g_gl_prog) g_gl_delete_program(g_gl_prog);
    if (g_gl_delete_textures && g_gl_tex_cw) g_gl_delete_textures(1, &g_gl_tex_cw);
    g_gl_vao = 0;
    g_gl_vbo = 0;
    g_gl_prog = 0;
    g_gl_tex_cw = 0;
#ifndef AB3D_DOS
    g_gl_unpack_ok = 0;
#endif
}

static int display_gl_try_init_unpack(void)
{
#ifdef AB3D_DOS
    return 0;
#else
    if (!g_sdl_ren || !g_window) return 0;
    SDL_RenderClear(g_sdl_ren);
#if SDL_VERSION_ATLEAST(2, 0, 14)
    SDL_RenderFlush(g_sdl_ren);
#endif
    if (!SDL_GL_GetCurrentContext()) {
        printf("[DISPLAY] 12-bit unpack: no GL context (use OpenGL renderer for GPU path)\n");
        return 0;
    }
    if (!display_gl_load_procs()) {
        printf("[DISPLAY] 12-bit unpack: failed to load OpenGL 3.0+ entry points\n");
        return 0;
    }

    GLuint vs = display_gl_compile_shader(GL_VERTEX_SHADER, display_gl_vs_src);
    GLuint fs = display_gl_compile_shader(GL_FRAGMENT_SHADER, display_gl_fs_src);
    if (!vs || !fs) return 0;

    g_gl_prog = g_gl_create_program();
    if (!g_gl_prog) {
        g_gl_delete_shader(vs);
        g_gl_delete_shader(fs);
        return 0;
    }
    g_gl_attach_shader(g_gl_prog, vs);
    g_gl_attach_shader(g_gl_prog, fs);
    g_gl_link_program(g_gl_prog);
    GLint linked = 0;
    g_gl_get_programiv(g_gl_prog, GL_LINK_STATUS, &linked);
    g_gl_delete_shader(vs);
    g_gl_delete_shader(fs);
    if (!linked) {
        char log[1024];
        log[0] = 0;
        if (g_gl_get_program_info_log) g_gl_get_program_info_log(g_gl_prog, (GLsizei)sizeof(log), NULL, log);
        printf("[DISPLAY] GL program link failed: %s\n", log);
        g_gl_delete_program(g_gl_prog);
        g_gl_prog = 0;
        return 0;
    }

    static const float quad[] = {
        -1.0f,  1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 0.0f,
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
    };
    g_gl_gen_vertex_arrays(1, &g_gl_vao);
    g_gl_gen_buffers(1, &g_gl_vbo);
    g_gl_bind_vertex_array(g_gl_vao);
    g_gl_bind_buffer(0x8892 /* GL_ARRAY_BUFFER */, g_gl_vbo);
    g_gl_buffer_data(0x8892, (ptrdiff_t)sizeof(quad), quad, 0x88E4 /* GL_STATIC_DRAW */);
    g_gl_vertex_attrib_pointer(0, 2, 0x1406 /* GL_FLOAT */, 0, (GLsizei)(4 * sizeof(float)), (const void*)0);
    g_gl_vertex_attrib_pointer(1, 2, 0x1406, 0, (GLsizei)(4 * sizeof(float)), (const void*)(uintptr_t)(2 * sizeof(float)));
    g_gl_enable_vertex_attrib_array(0);
    g_gl_enable_vertex_attrib_array(1);
    g_gl_bind_vertex_array(0);

    g_gl_gen_textures(1, &g_gl_tex_cw);
    g_gl_bind_texture(0x0DE1 /* GL_TEXTURE_2D */, g_gl_tex_cw);
    g_gl_tex_parameteri2(0x0DE1, 0x2801 /* GL_TEXTURE_MIN_FILTER */, (GLint)GL_NEAREST);
    g_gl_tex_parameteri2(0x0DE1, 0x2800 /* GL_TEXTURE_MAG_FILTER */, (GLint)GL_NEAREST);
    g_gl_tex_parameteri2(0x0DE1, 0x2802 /* GL_TEXTURE_WRAP_S */, (GLint)0x812F /* GL_CLAMP_TO_EDGE */);
    g_gl_tex_parameteri2(0x0DE1, 0x2803 /* GL_TEXTURE_WRAP_T */, (GLint)0x812F);
    g_gl_bind_texture(0x0DE1, 0);

    g_gl_unpack_ok = 1;
    if (!display_gl_hud_try_init()) {
        printf("[DISPLAY] HUD: GL overlay init failed; status HUD may be invisible until fixed or AB3D_DISABLE_GL_UNPACK=1\n");
    }
    printf("[DISPLAY] 12-bit unpack: GPU (GL_R16UI + shader)\n");
    return 1;
#endif
}

/* GL client-unpack state (pixel store + PBO) is global; SDL's renderer may leave
 * GL_UNPACK_ROW_LENGTH / SKIP_* non-zero or GL_PIXEL_UNPACK_BUFFER bound, which makes
 * glTex(Sub)Image2D read tightly packed CPU rows with the wrong stride (diagonal shear). */
static void display_gl_reset_client_pixel_unpack(void)
{
    if (g_gl_bind_buffer)
        g_gl_bind_buffer(0x88EC /* GL_PIXEL_UNPACK_BUFFER */, 0);
    if (g_gl_pixel_storei) {
        g_gl_pixel_storei(0x0CF5 /* GL_UNPACK_ALIGNMENT */, 2);
        g_gl_pixel_storei(0x0CF2 /* GL_UNPACK_ROW_LENGTH */, 0);
        g_gl_pixel_storei(0x0CF3 /* GL_UNPACK_SKIP_ROWS */, 0);
        g_gl_pixel_storei(0x0CF4 /* GL_UNPACK_SKIP_PIXELS */, 0);
    }
}

static void display_gl_resize_cw_texture(int w, int h)
{
    if (!g_gl_unpack_ok || w < 1 || h < 1) return;
    display_gl_reset_client_pixel_unpack();
    g_gl_active_texture(GL_TEXTURE0);
    g_gl_bind_texture(0x0DE1 /* GL_TEXTURE_2D */, g_gl_tex_cw);
    g_gl_tex_image_2d(0x0DE1, 0, (GLint)GL_R16UI, w, h, 0, GL_RED_INTEGER, 0x1403 /* GL_UNSIGNED_SHORT */, NULL);
    g_gl_bind_texture(0x0DE1, 0);
    if (g_gl_get_error && g_gl_get_error() != 0)
        printf("[DISPLAY] GL_R16UI glTexImage2D reported an error (driver may not support integer textures)\n");
}

static void display_gl_present_cw(const uint16_t *src, int w, int h)
{
    GLint loc_u;
    GLint loc_w;
    GLint loc_h;
    int win_w = 1, win_h = 1;

    if (!g_gl_unpack_ok || !src || w < 1 || h < 1) return;
    display_gl_output_size(&win_w, &win_h);

    display_gl_reset_client_pixel_unpack();
    g_gl_active_texture(GL_TEXTURE0);
    g_gl_bind_texture(0x0DE1 /* GL_TEXTURE_2D */, g_gl_tex_cw);
    g_gl_tex_sub_image_2d(0x0DE1, 0, 0, 0, w, h, GL_RED_INTEGER, 0x1403 /* GL_UNSIGNED_SHORT */, src);
    g_gl_bind_texture(0x0DE1, 0);

    SDL_SetRenderDrawColor(g_sdl_ren, 0, 0, 0, 255);
    SDL_RenderClear(g_sdl_ren);
#if SDL_VERSION_ATLEAST(2, 0, 14)
    SDL_RenderFlush(g_sdl_ren);
#endif

    {
        int dx = g_present_dst_rect.x;
        int dy = g_present_dst_rect.y;
        int dw = g_present_dst_rect.w;
        int dh = g_present_dst_rect.h;
        int gl_y = win_h - dy - dh;
        g_gl_viewport((GLint)dx, (GLint)gl_y, (GLsizei)dw, (GLsizei)dh);
    }

    g_gl_use_program(g_gl_prog);
    loc_u = g_gl_get_uniform_location(g_gl_prog, "u_cw");
    if (loc_u >= 0) {
        g_gl_active_texture(GL_TEXTURE0);
        g_gl_bind_texture(0x0DE1 /* GL_TEXTURE_2D */, g_gl_tex_cw);
        g_gl_uniform1i(loc_u, 0);
    }
    loc_w = g_gl_get_uniform_location(g_gl_prog, "u_fb_w");
    if (loc_w >= 0) g_gl_uniform1i(loc_w, w);
    loc_h = g_gl_get_uniform_location(g_gl_prog, "u_fb_h");
    if (loc_h >= 0) g_gl_uniform1i(loc_h, h);
    g_gl_bind_vertex_array(g_gl_vao);
    g_gl_draw_arrays(GL_TRIANGLE_STRIP, 0, 4);
    g_gl_bind_vertex_array(0);
    g_gl_use_program(0);

    /* SDL_Renderer HUD/automap/tint overlays use window pixel coords (0..drawable); they assume
     * glViewport matches the full drawable. A letterbox-only viewport clips or misplaces them. */
    g_gl_viewport(0, 0, (GLsizei)win_w, (GLsizei)win_h);
}

/* SDL2's OpenGL backend caches the "current shader" as an enum (texture vs solid, etc.), not the
 * actual GL program name. After raw glUseProgram(0), the GPU has no program while drawstate.shader
 * still says SHADER_TEXTURE — GL_SelectShader is skipped and SDL_RenderCopy HUD draws do nothing.
 * Queue a no-op solid draw (alpha 0 + blend) so the next texture draw forces GL_SelectShader. */
static void display_sdl_resync_after_raw_gl(void)
{
#if SDL_VERSION_ATLEAST(2, 0, 14)
    SDL_RenderFlush(g_sdl_ren);
#endif
    {
        int vw, vh;
        if (SDL_GetRendererOutputSize(g_sdl_ren, &vw, &vh) == 0) {
            SDL_Rect full;
            full.x = 0;
            full.y = 0;
            full.w = vw;
            full.h = vh;
            SDL_RenderSetViewport(g_sdl_ren, &full);
        }
    }
    SDL_SetRenderDrawBlendMode(g_sdl_ren, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(g_sdl_ren, 0, 0, 0, 0);
    SDL_RenderDrawPoint(g_sdl_ren, 0, 0);
}

static void display_cpu_unpack_cw_to_texture(const uint16_t *src, int w, int h)
{
    if (!g_texture || !src || w < 1 || h < 1) return;

    /* Fast path: cw_buffer is already packed 0x0RGB words. Push directly; no per-pixel CPU work. */
#if defined(__amigaos4__) || defined(__BIG_ENDIAN__)
    if (0) { /* Disabled on big-endian: byte order mismatch */
#else
    if (g_texture_is_4444_direct && !AB3D_CW_COL_MAJOR) {
#endif
        if (SDL_UpdateTexture(g_texture, NULL, src, (int)((size_t)w * sizeof(uint16_t))) != 0) {
            return;
        }
        return;
    }

    void *pixels;
    int pitch;
    if (SDL_LockTexture(g_texture, NULL, &pixels, &pitch) < 0) return;
    for (int y = 0; y < h; y++) {
        uint32_t *dst_row = (uint32_t*)((uint8_t*)pixels + (size_t)y * (size_t)pitch);
        for (int x = 0; x < w; x++) {
            uint32_t c = (uint32_t)(src[display_cw_index_xy(x, y, w, h)] & 0xFFFu);
            uint32_t r4 = (c >> 8) & 0xFu;
            uint32_t g4 = (c >> 4) & 0xFu;
            uint32_t b4 = c & 0xFu;
            #if defined(__amigaos4__) || defined(__BIG_ENDIAN__)
            dst_row[x] = 0xFF000000u | (b4 * 0x11u << 16) | (g4 * 0x11u << 8) | (r4 * 0x11u);
#else
            dst_row[x] = 0xFF000000u | (r4 * 0x11u << 16) | (g4 * 0x11u << 8) | (b4 * 0x11u);
#endif
        }
    }
    SDL_UnlockTexture(g_texture);
}

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

#ifdef AB3D_DOS
    int dw=rw,dh=rh;
    printf("[DOS VIDEO] render %dx%d; native RGB565 SDL surface; 1:1 viewport\n",rw,rh);
#else
    double sx = (double)win_w / (double)rw;
    double sy = (double)win_h / (double)rh;
    double sc = (sx < sy) ? sx : sy;
    int dw = (int)(rw * sc + 0.5);
    int dh = (int)(rh * sc + 0.5);
    if (dw < 1) dw = 1;
    if (dh < 1) dh = 1;
    if (dw > win_w) dw = win_w;
    if (dh > win_h) dh = win_h;
#endif

    g_present_dst_rect.x = (win_w - dw) / 2;
    g_present_dst_rect.y = (win_h - dh) / 2;
    g_present_dst_rect.w = dw;
    g_present_dst_rect.h = dh;

    renderer_set_present_size(dw, dh);

#if SDL_VERSION_ATLEAST(2, 0, 12)
    /* OpenGL render path: nearest when magnifying the software framebuffer; linear when
     * minifying (pairs with glGenerateMipmap + trilinear in display_regenerate_*). */
    if (g_texture && !g_gl_unpack_ok) {
        SDL_SetTextureScaleMode(g_texture,
            (sc > 1.0) ? SDL_ScaleModeNearest : SDL_ScaleModeLinear);
    }
#endif
}

static int display_sync_present_output_size(int *out_w, int *out_h)
{
    int rw = g_present_width;
    int rh = g_present_height;

    if (!g_sdl_ren) return 0;
    if (SDL_GetRendererOutputSize(g_sdl_ren, &rw, &rh) != 0) {
        rw = g_present_width;
        rh = g_present_height;
    }
    if (rw < 1) rw = 1;
    if (rh < 1) rh = 1;
    if (rw != g_present_width || rh != g_present_height) {
        display_update_letterbox(rw, rh);
        g_present_width = rw;
        g_present_height = rh;
    }

    if (out_w) *out_w = rw;
    if (out_h) *out_h = rh;
    return 1;
}

static void display_window_point_to_size(int window_x, int window_y,
                                         int target_w, int target_h,
                                         int *out_x, int *out_y)
{
    int win_w = 0;
    int win_h = 0;

    if (g_window) {
        SDL_GetWindowSize(g_window, &win_w, &win_h);
    }
    if (target_w < 1) target_w = 1;
    if (target_h < 1) target_h = 1;
    if (win_w < 1) win_w = target_w;
    if (win_h < 1) win_h = target_h;

    if (out_x) {
        *out_x = (int)(((int64_t)window_x * (int64_t)target_w) / win_w);
    }
    if (out_y) {
        *out_y = (int)(((int64_t)window_y * (int64_t)target_h) / win_h);
    }
}

static int display_window_point_to_output(int window_x, int window_y,
                                          int *out_x, int *out_y)
{
    int out_w = 0;
    int out_h = 0;

    if (!display_sync_present_output_size(&out_w, &out_h))
        return 0;

    display_window_point_to_size(window_x, window_y, out_w, out_h,
                                 out_x, out_y);
    return 1;
}

int display_text_line_at_point(int window_x, int window_y, int title_screen,
                               int *out_line)
{
    DisplayTextRun runs[DISPLAY_TEXT_LAYOUT_MAX_RUNS];
    SDL_Rect r;
    int px = 0;
    int py = 0;
    int out_w = 0;
    int out_h = 0;
    int run_count = 0;
    int scale_q = DISPLAY_ASCII_SCALE_ONE;
    int line_h = DISPLAY_ASCII_LINE_ADVANCE;
    int draw_h = DISPLAY_ASCII_DRAW_H;
    int top = 0;
    int margin_x = 0;

    if (out_line) *out_line = -1;
    if (!out_line || !g_sdl_ren)
        return 0;

    if (!display_window_point_to_output(window_x, window_y, &px, &py))
        return 0;

    if (!display_sync_present_output_size(&out_w, &out_h))
        return 0;

    if (title_screen) {
        r = display_fit_rect(out_w, out_h, DISPLAY_TITLE_W, DISPLAY_TITLE_H);
        if (g_gl_unpack_ok && g_gl_hud_ok) {
            display_gl_output_size(&out_w, &out_h);
            if (out_w < 1) out_w = g_present_width;
            if (out_h < 1) out_h = g_present_height;
            display_window_point_to_size(window_x, window_y, out_w, out_h,
                                         &px, &py);
            r = display_fit_rect(out_w, out_h, DISPLAY_TITLE_W, DISPLAY_TITLE_H);
        }
    } else {
        r = g_present_dst_rect;
    }

#ifdef AB3D_DOS
    r=display_dos_visible_text_rect(r);
    if(r.w<1 || r.h<1)return 0;
#endif
    if (!display_text_layout_in_rect(r, runs, DISPLAY_TEXT_LAYOUT_MAX_RUNS,
                                     &run_count, &scale_q, &line_h, &draw_h,
                                     &top, &margin_x, NULL)) {
        return 0;
    }
    (void)scale_q;
    (void)draw_h;

    if (px < r.x + margin_x || px >= r.x + r.w - margin_x)
        return 0;
    if (py < top || py >= top + run_count * line_h)
        return 0;

    int run_index = (py - top) / line_h;
    if (run_index < 0 || run_index >= run_count)
        return 0;
    if (runs[run_index].len < 1)
        return 0;

    *out_line = runs[run_index].source_line;
    return *out_line >= 0;
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
    if (g_texture) {
        SDL_DestroyTexture(g_texture);
        g_texture = NULL;
    }
    g_texture_is_4444_direct = 0;
    if (g_gl_unpack_ok) {
        display_gl_resize_cw_texture(w, h);
    } else {
#ifdef AB3D_DOS
        /* The DOS renderer owns native RGB565 framebuffers; no SDL texture. */
#else
#if defined(__amigaos4__) || defined(__BIG_ENDIAN__)
        g_texture = NULL; /* Force ARGB8888 on big-endian */
#else
        g_texture = SDL_CreateTexture(g_sdl_ren,
            SDL_PIXELFORMAT_XRGB4444, SDL_TEXTUREACCESS_STREAMING, g_internal_w, g_internal_h);
#endif
        if (g_texture) {
            g_texture_is_4444_direct = 1;
        } else {
#if defined(__amigaos4__) || defined(__BIG_ENDIAN__)
            g_texture = NULL; /* Force ARGB8888 on big-endian */
#else
            g_texture = SDL_CreateTexture(g_sdl_ren,
                SDL_PIXELFORMAT_ARGB4444, SDL_TEXTUREACCESS_STREAMING, g_internal_w, g_internal_h);
#endif
            if (g_texture) {
                g_texture_is_4444_direct = 1;
            } else {
                g_texture = SDL_CreateTexture(g_sdl_ren,
                    SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, g_internal_w, g_internal_h);
                if (g_texture) {
                    printf("[DISPLAY] CPU fallback texture: 4444 unavailable, using ARGB8888\n");
                }
            }
        }
        if (g_texture) {
            SDL_SetTextureScaleMode(g_texture, SDL_ScaleModeLinear);
            /* Opaque blit: cw uses XRGB4444 (high nibble unused). */
            SDL_SetTextureBlendMode(g_texture, SDL_BLENDMODE_NONE);
        }
#endif
    }
}

static void display_log_display_mode_snapshot(const char *tag)
{
    SDL_DisplayMode cur;
    SDL_DisplayMode desk;
    int cur_ok = (SDL_GetCurrentDisplayMode(0, &cur) == 0);
    int desk_ok = (SDL_GetDesktopDisplayMode(0, &desk) == 0);

    if (cur_ok && desk_ok) {
        printf("[DISPLAY] %s: current=%dx%d@%d desktop=%dx%d@%d\n",
               tag,
               cur.w, cur.h, (int)cur.refresh_rate,
               desk.w, desk.h, (int)desk.refresh_rate);
        return;
    }
    if (cur_ok) {
        printf("[DISPLAY] %s: current=%dx%d@%d desktop=unavailable\n",
               tag, cur.w, cur.h, (int)cur.refresh_rate);
        return;
    }
    if (desk_ok) {
        printf("[DISPLAY] %s: current=unavailable desktop=%dx%d@%d\n",
               tag, desk.w, desk.h, (int)desk.refresh_rate);
        return;
    }
    printf("[DISPLAY] %s: display mode query unavailable (%s)\n", tag, SDL_GetError());
}

static int display_use_fullscreen_desktop(const GameState *state)
{
    if (state) {
        if (state->cfg_display_mode == 1) return 1;
        if (state->cfg_display_mode == 0) return 0;
    }
#ifdef AB3D_RELEASE
    return 1;
#else
    return 0;
#endif
}

#if defined(__EMSCRIPTEN__)
/* Scale down so both dimensions stay <= max_dim (aspect preserved). Avoids WebGL tex limits + WASM OOM. */
static void emscripten_clamp_dims_aspect(int *w, int *h, int max_dim)
{
    if (!w || !h || max_dim < 96) return;
    if (*w < 1 || *h < 1) return;
    if (*w <= max_dim && *h <= max_dim) return;
    double sx = (double)max_dim / (double)*w;
    double sy = (double)max_dim / (double)*h;
    double s = (sx < sy) ? sx : sy;
    *w = (int)((double)*w * s + 0.5);
    *h = (int)((double)*h * s + 0.5);
    if (*w < 96) *w = 96;
    if (*h < 80) *h = 80;
}

#endif

/* -----------------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------------- */
void display_init(GameState *state)
{
    int base_rw = 1280;
    int base_rh = 720;
    int supersampling = 1;
    int fullscreen_desktop;
    int rw, rh;
    if (state) {
        base_rw = (int)state->cfg_render_width;
        base_rh = (int)state->cfg_render_height;
#if defined(__EMSCRIPTEN__)
        {
            const char *use_ini = SDL_getenv("AB3D_WEB_USE_INI_RES");
            if (!use_ini || use_ini[0] == '\0' ||
                (use_ini[0] != '1' && use_ini[0] != 'y' && use_ini[0] != 'Y')) {
                int aw = 0, ah = 0;
                display_emscripten_apply_screen_aspect_resolution(&aw, &ah);
                if (aw >= 96 && ah >= 80 && aw <= RENDER_INTERNAL_MAX_DIM && ah <= RENDER_INTERNAL_MAX_DIM) {
                    state->cfg_render_width = (int16_t)aw;
                    state->cfg_render_height = (int16_t)ah;
                    base_rw = aw;
                    base_rh = ah;
                    printf("[DISPLAY] Web: screen aspect → preset %dx%d (set AB3D_WEB_USE_INI_RES=1 to keep ab3d.ini)\n",
                           aw, ah);
                }
            }
        }
#endif
        supersampling = (int)state->cfg_supersampling;
    }
    if (!SDL_WasInit(SDL_INIT_VIDEO)) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            printf("[DISPLAY] SDL_Init failed: %s\n", SDL_GetError());
            return;
        }
    }
    fullscreen_desktop = display_use_fullscreen_desktop(state);
#ifdef AB3D_DOS
    fullscreen_desktop = 0;
#endif
    if (base_rw < 96) base_rw = 96;
    if (base_rh < 80) base_rh = 80;
    if (supersampling < 1) supersampling = 1;
#ifdef AB3D_DOS
    if(supersampling!=1){fprintf(stderr,"DOS 1:1 rendering requires supersampling=1\n");exit(1);}
#endif
    if (supersampling > 4) supersampling = 4;

    {
        /* Keep aspect ratio if a requested supersampled target exceeds renderer max size. */
        const double req_w = (double)base_rw * (double)supersampling;
        const double req_h = (double)base_rh * (double)supersampling;
        double fit = 1.0;
        if (req_w > (double)RENDER_INTERNAL_MAX_DIM || req_h > (double)RENDER_INTERNAL_MAX_DIM) {
            const double fit_w = (double)RENDER_INTERNAL_MAX_DIM / req_w;
            const double fit_h = (double)RENDER_INTERNAL_MAX_DIM / req_h;
            fit = (fit_w < fit_h) ? fit_w : fit_h;
            if (fit < 0.0) fit = 0.0;
        }
        rw = (int)(req_w * fit + 0.5);
        rh = (int)(req_h * fit + 0.5);
        if (rw < 96) rw = 96;
        if (rh < 80) rh = 80;
        if (rw > RENDER_INTERNAL_MAX_DIM) rw = RENDER_INTERNAL_MAX_DIM;
        if (rh > RENDER_INTERNAL_MAX_DIM) rh = RENDER_INTERNAL_MAX_DIM;
    }

#if defined(__EMSCRIPTEN__)
    /* Supersampling can push rw/rh past 4096 even when base is clamped; WebGL + heap need this. */
    {
        int rw0 = rw, rh0 = rh;
        emscripten_clamp_dims_aspect(&rw, &rh, 4096);
        if (rw != rw0 || rh != rh0) {
            if (state) {
                state->cfg_render_width = (int16_t)rw;
                state->cfg_render_height = (int16_t)rh;
                state->cfg_supersampling = 1;
            }
            printf("[DISPLAY] Web: supersampled size clamped to %dx%d (GPU/heap safe)\n", rw, rh);
        }
    }
#endif

    g_internal_w = rw;
    g_internal_h = rh;

    display_log_display_mode_snapshot("startup-before-window");

    /* Request GL 3.0+ so integer textures (GL_R16UI) work when using the OpenGL render driver.
     * Upload raw color words on the GPU and avoid SDL CPU blit (BlitNtoN) on present.
     *
     * On Windows/Linux use the compatibility profile: SDL2's OpenGL renderer still relies on
     * fixed-function state (glMatrixMode/glOrtho) for SDL_RenderClear and 2D overlays. A 3.0+
     * *core* context removes those entry points, which breaks mixing SDL_Renderer with our
     * custom GL unpack pass and corrupts the framebuffer. macOS only exposes core profiles
     * for GL 3.2+, so we keep core there. */
#if defined(__EMSCRIPTEN__)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#elif defined(__APPLE__)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
#else
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
#endif
#ifndef __EMSCRIPTEN__
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    if (fullscreen_desktop) {
        printf("[DISPLAY] SDL2 init (display_mode=fullscreen: desktop-sized window, base=%dx%d, internal render=%dx%d, supersampling=%d)\n",
               base_rw, base_rh, g_internal_w, g_internal_h, supersampling);
    } else {
#if defined(__EMSCRIPTEN__)
        printf("[DISPLAY] SDL2 init (display_mode=windowed, window %dx%d, internal render %dx%d, supersampling=%d; letterbox to fit)\n",
               base_rw, base_rh, g_internal_w, g_internal_h, supersampling);
#else
        printf("[DISPLAY] SDL2 init (display_mode=windowed, window %dx%d, internal render %dx%d, supersampling=%d; resize to letterbox)\n",
               base_rw, base_rh, g_internal_w, g_internal_h, supersampling);
#endif
    }

    renderer_init(rw, rh);

    const char *driver_override = SDL_getenv("AB3D_RENDER_DRIVER");
    /* Set AB3D_DISABLE_GL_UNPACK=1 to force D3D + SDL texture (CPU blit path). Otherwise prefer GL_R16UI. */
    const char *disable_gl_unpack = SDL_getenv("AB3D_DISABLE_GL_UNPACK");
    const char *debug_window_title = SDL_getenv("AB3D_DEBUG_WINDOW_TITLE");
    const char *framebuffer_mipmap = SDL_getenv("AB3D_FRAMEBUFFER_MIPMAPS");
    int prefer_gpu_unpack = 1;
    if (disable_gl_unpack && disable_gl_unpack[0] != '\0') {
        if (disable_gl_unpack[0] == '1' || disable_gl_unpack[0] == 'y' || disable_gl_unpack[0] == 'Y')
            prefer_gpu_unpack = 0;
    }
    g_debug_window_title = (debug_window_title &&
                            (debug_window_title[0] == '1' ||
                             debug_window_title[0] == 'y' ||
                             debug_window_title[0] == 'Y')) ? 1 : 0;
    g_framebuffer_mipmap_enabled = (framebuffer_mipmap &&
                                    (framebuffer_mipmap[0] == '1' ||
                                     framebuffer_mipmap[0] == 'y' ||
                                     framebuffer_mipmap[0] == 'Y')) ? 1 : 0;
    int window_w = base_rw;
    int window_h = base_rh;
#ifdef AB3D_DOS
    window_w=base_rw<320?320:base_rw;window_h=base_rh<200?200:base_rh;
#endif
    Uint32 window_flags = SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;
    int window_x = SDL_WINDOWPOS_CENTERED;
    int window_y = SDL_WINDOWPOS_CENTERED;
    g_release_borderless_desktop = 0;
    if (fullscreen_desktop) {
        /* Same flags as windowed — no SDL_WINDOW_FULLSCREEN* (those can alter the display mode). */
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
        window_flags = SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;
        g_release_borderless_desktop = 1;
    }
    {
        int explicit_render_driver = (driver_override && driver_override[0] != '\0' &&
                                      strcmp(driver_override, "auto") != 0);
        if (prefer_gpu_unpack && !explicit_render_driver) {
#if !defined(__amigaos4__)
            window_flags |= SDL_WINDOW_OPENGL;
#endif
        }
    }
    if (driver_override && strcmp(driver_override, "opengl") == 0) {
        window_flags |= SDL_WINDOW_OPENGL;
        if (fullscreen_desktop) {
            printf("[DISPLAY] OpenGL override: desktop-sized window + SDL_WINDOW_OPENGL\n");
        } else {
            printf("[DISPLAY] OpenGL override: windowed SDL_WINDOW_OPENGL enabled\n");
        }
    }

    g_window = SDL_CreateWindow(
        "Alien Breed 3D I",
        window_x, window_y,
        window_w, window_h,
        window_flags
    );
    if (!g_window) {
        printf("[DISPLAY] SDL_CreateWindow failed: %s\n", SDL_GetError());
#ifdef AB3D_DOS
        exit(1);
#endif
        return;
    }
    SDL_GetWindowSize(g_window, &window_w, &window_h);

    {
        const char *driver_try[4];
        int driver_try_count = 0;
        const char *selected_hint = "";

        if (driver_override && driver_override[0] != '\0' && strcmp(driver_override, "auto") != 0) {
            driver_try[driver_try_count++] = driver_override;
            driver_try[driver_try_count++] = "";
        } else if (prefer_gpu_unpack) {
            /* OpenGL first: enables GL_R16UI present (no SDL CPU blit for the 3D buffer). */
#if defined(__EMSCRIPTEN__)
            driver_try[driver_try_count++] = "opengles2";
            driver_try[driver_try_count++] = "";
#else
            driver_try[driver_try_count++] = "opengl";
            driver_try[driver_try_count++] = "direct3d11";
            driver_try[driver_try_count++] = "direct3d";
            driver_try[driver_try_count++] = "";
#endif
        } else {
            driver_try[driver_try_count++] = "direct3d11";
            driver_try[driver_try_count++] = "direct3d";
            driver_try[driver_try_count++] = "";
        }

        g_sdl_ren = NULL;
        for (int i = 0; i < driver_try_count; i++) {
            selected_hint = driver_try[i];
            SDL_SetHint(SDL_HINT_RENDER_DRIVER, selected_hint);
            g_sdl_ren = SDL_CreateRenderer(g_window, -1,
                SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
            if (g_sdl_ren) break;
        }
        if (!g_sdl_ren) {
            selected_hint = "";
        }
        printf("[DISPLAY] renderer hint request: %s\n",
               (selected_hint[0] != '\0') ? selected_hint : "auto");
    }
    if (!g_sdl_ren) {
        printf("[DISPLAY] SDL_CreateRenderer failed: %s\n", SDL_GetError());
#ifdef AB3D_DOS
        exit(1);
#endif
        return;
    }
    {
        SDL_RendererInfo ri;
        if (SDL_GetRendererInfo(g_sdl_ren, &ri) == 0) {
            printf("[DISPLAY] SDL renderer driver: %s\n", ri.name);
        }
    }

    display_gl_try_init_unpack();
    if (!g_gl_unpack_ok && prefer_gpu_unpack) {
        printf("[DISPLAY] GL_R16UI present unavailable (no GL 3.0+ context). Using SDL texture path; "
               "try AB3D_RENDER_DRIVER=opengl or install updated GPU drivers. "
               "To force D3D+texture: set AB3D_DISABLE_GL_UNPACK=1\n");
    }
#if defined(__EMSCRIPTEN__)
    /* GL_R16UI texture must be <= GL_MAX_TEXTURE_SIZE (often 4096 on WebGL); also sync cfg. */
    if (g_gl_unpack_ok) {
        int max_tex = 4096;
        if (g_gl_get_integerv) {
            GLint m = 0;
            g_gl_get_integerv(GL_MAX_TEXTURE_SIZE, &m);
            if (m > 0 && m < max_tex) max_tex = (int)m;
        }
        if (rw > max_tex || rh > max_tex) {
            emscripten_clamp_dims_aspect(&rw, &rh, max_tex);
            g_internal_w = rw;
            g_internal_h = rh;
            if (state) {
                state->cfg_render_width = (int16_t)rw;
                state->cfg_render_height = (int16_t)rh;
                state->cfg_supersampling = 1;
            }
            printf("[DISPLAY] Web: clamped internal render to %dx%d (GPU texture limit %d)\n",
                   rw, rh, max_tex);
        }
    }
#endif
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
    display_log_display_mode_snapshot("startup-after-window");
#ifdef AB3D_DOS
    /* Keep the small, always-needed HUD atlases ahead of level heap growth. */
    display_hud_digits_ensure_loaded();
#else
    (void)display_gl_lines_ensure_scratch(DISPLAY_AUTOMAP_MAX_SEGS * 14);
#endif
#if defined(__EMSCRIPTEN__)
    /* Canvas backing store may settle after layout; refresh letterbox once. */
    SDL_PumpEvents();
    display_handle_resize();
    display_emscripten_install_fullscreen_listeners();
#endif
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

void display_emscripten_frame_resize_poll(void)
{
#if defined(__EMSCRIPTEN__)
    if (!g_sdl_ren || !g_window) return;
    int css_w = 0;
    int css_h = 0;
    int canvas_w = 0;
    int canvas_h = 0;
    MAIN_THREAD_EM_ASM({
        var c = Module['canvas'];
        var w = 0;
        var h = 0;
        var bw = 0;
        var bh = 0;
        if (c) {
            w = c.clientWidth | 0;
            h = c.clientHeight | 0;
            bw = c.width | 0;
            bh = c.height | 0;
        }
        if (w < 1) w = window.innerWidth | 0;
        if (h < 1) h = window.innerHeight | 0;
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        HEAP32[$0 >> 2] = w;
        HEAP32[$1 >> 2] = h;
        HEAP32[$2 >> 2] = bw;
        HEAP32[$3 >> 2] = bh;
    }, &css_w, &css_h, &canvas_w, &canvas_h);

    int win_w = 0;
    int win_h = 0;
    SDL_GetWindowSize(g_window, &win_w, &win_h);
    int pending = 0;
    if (css_w > 0 && css_h > 0 &&
        (css_w != win_w || css_h != win_h ||
         css_w != canvas_w || css_h != canvas_h)) {
        MAIN_THREAD_EM_ASM({
            if (Module['ab3dSyncCanvasSizeToClient']) {
                Module['ab3dSyncCanvasSizeToClient']();
                return;
            }
            var c = Module['canvas'];
            if (c) {
                if ((c.width | 0) !== $0) c.width = $0;
                if ((c.height | 0) !== $1) c.height = $1;
            }
        }, css_w, css_h);
        SDL_SetWindowSize(g_window, css_w, css_h);
        pending = 1;
    }

    SDL_PumpEvents();
    pending |= MAIN_THREAD_EM_ASM_INT({
        var v = Module['ab3dFullscreenResizePending'] | 0;
        Module['ab3dFullscreenResizePending'] = 0;
        return v;
    });
    int out_w = 0;
    int out_h = 0;
    int changed = 0;
    if (SDL_GetRendererOutputSize(g_sdl_ren, &out_w, &out_h) == 0) {
        if (out_w < 1) out_w = 1;
        if (out_h < 1) out_h = 1;
        changed = (out_w != g_present_width || out_h != g_present_height);
    }
    if (pending || changed) {
        display_handle_resize();
    }
#endif
}

int display_is_fullscreen(void)
{
    if (!g_window) return 0;
    if (g_release_borderless_desktop) return 1;
    return (SDL_GetWindowFlags(g_window) & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_FULLSCREEN_DESKTOP)) != 0;
}

void display_toggle_fullscreen(void)
{
    if (!g_window) return;

#if defined(__EMSCRIPTEN__)
    display_emscripten_canvas_fullscreen_toggle();
    SDL_PumpEvents();
    display_handle_resize();
    return;
#endif

    Uint32 wf = SDL_GetWindowFlags(g_window);
    int sdl_fs = (wf & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_FULLSCREEN_DESKTOP)) != 0;

    if (sdl_fs) {
        SDL_SetWindowFullscreen(g_window, 0);
    } else if (g_release_borderless_desktop) {
        /* display_mode=fullscreen at startup: borderless desktop-sized window (no SDL fullscreen API). */
        g_release_borderless_desktop = 0;
        SDL_SetWindowSize(g_window, 1280, 720);
        SDL_SetWindowPosition(g_window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    } else {
        SDL_SetWindowFullscreen(g_window, SDL_WINDOW_FULLSCREEN_DESKTOP);
    }

    SDL_PumpEvents();
    display_handle_resize();
}

void display_shutdown(void)
{
#ifdef AB3D_DOS
    free(g_dos_map_scratch);g_dos_map_scratch=NULL;g_dos_map_capacity=0;
    for(int i=0;i<2;i++) {
        if(g_dos_frame_surfaces[i])SDL_DestroySurface(g_dos_frame_surfaces[i]);
        g_dos_frame_surfaces[i]=NULL;g_dos_frame_pixels[i]=NULL;
    }
#endif
    display_hud_digits_free();
    display_key_hud_free_textures();
    for (int i = 0; i < 2; i++) {
        if (g_bar_hud_tex[i]) SDL_DestroyTexture(g_bar_hud_tex[i]);
        g_bar_hud_tex[i] = NULL;
    }
    display_ascii_font_free();
    display_title_screen_free();
    display_gl_lines_release_scratch();
    g_key_hud_tex_tag = 0;
    renderer_shutdown();
    display_gl_shutdown_unpack();
    if (g_texture)  SDL_DestroyTexture(g_texture);
    if (g_sdl_ren)  SDL_DestroyRenderer(g_sdl_ren);
    if (g_window)   SDL_DestroyWindow(g_window);
    SDL_Quit();
    printf("[DISPLAY] SDL2 shutdown\n");
}

/* -----------------------------------------------------------------------
 * Screen management (no-ops for SDL2)
 * ----------------------------------------------------------------------- */
void display_alloc_text_screen(void)        { display_clear_text_screen(); }
void display_release_text_screen(void)      { display_clear_text_screen(); display_ascii_font_free(); }
void display_alloc_copper_screen(void)      { }
void display_release_copper_screen(void)    { }
void display_alloc_title_memory(void)       { }
void display_release_title_memory(void)     { display_title_screen_free(); }
void display_alloc_panel_memory(void)       { }
void display_release_panel_memory(void)     { }

void display_setup_title_screen(void)       { display_title_screen_ensure_loaded(); }
void display_load_title_screen(void)        { display_title_screen_ensure_loaded(); }
void display_clear_opt_screen(void)         { display_clear_text_screen(); }
void display_draw_opt_screen(int screen_num) { (void)screen_num; }
void display_fade_up_title(int amount)      { (void)amount; }
void display_fade_down_title(int amount)    { (void)amount; }
void display_clear_title_palette(void)      { }

void display_init_copper_screen(void)       { }

static void display_automap_amiga12_to_rgb(uint16_t cw, Uint8 *r, Uint8 *g, Uint8 *b)
{
    uint16_t c = (uint16_t)(cw & 0xFFFu);
    *r = (Uint8)(((c >> 8) & 0xFu) * 17u);
    *g = (Uint8)(((c >> 4) & 0xFu) * 17u);
    *b = (Uint8)((c & 0xFu) * 17u);
}

static void display_automap_map_pt(int x, int y, int iw, int ih, int *ox, int *oy)
{
    if (iw < 1) iw = 1;
    if (ih < 1) ih = 1;
    *ox = g_present_dst_rect.x + (x * g_present_dst_rect.w) / iw;
    *oy = g_present_dst_rect.y + (y * g_present_dst_rect.h) / ih;
}

static void display_automap_line_stroked_sdl(SDL_Renderer *ren,
                                             int ax0, int ay0, int ax1, int ay1,
                                             Uint8 fr, Uint8 fg, Uint8 fb, Uint8 alpha,
                                             int black_k_max, int fg_k_max)
{
#ifdef AB3D_DOS
    SDL_SetRenderDrawBlendMode(ren,SDL_BLENDMODE_NONE);
    SDL_SetRenderDrawColor(ren,fr*alpha/255,fg*alpha/255,fb*alpha/255,255);
    SDL_RenderDrawLine(ren,ax0,ay0,ax1,ay1);
    return;
#endif
    int dx = ax1 - ax0;
    int dy = ay1 - ay0;
    double len = hypot((double)dx, (double)dy);
    if (len < 1e-6) return;
    double px = -dy / len;
    double py = dx / len;
    if (black_k_max < 0) black_k_max = 0;
    if (fg_k_max < 0) fg_k_max = 0;
    SDL_SetRenderDrawBlendMode(ren, (alpha < 255u) ? SDL_BLENDMODE_BLEND : SDL_BLENDMODE_NONE);
    SDL_SetRenderDrawColor(ren, 0, 0, 0, alpha);
    for (int k = -black_k_max; k <= black_k_max; k++) {
        double oxk = k * px, oyk = k * py;
        int ox = (int)(oxk + (oxk >= 0.0 ? 0.5 : -0.5));
        int oy = (int)(oyk + (oyk >= 0.0 ? 0.5 : -0.5));
        SDL_RenderDrawLine(ren, ax0 + ox, ay0 + oy, ax1 + ox, ay1 + oy);
    }
    SDL_SetRenderDrawColor(ren, fr, fg, fb, alpha);
    for (int k = -fg_k_max; k <= fg_k_max; k++) {
        double oxk = k * px, oyk = k * py;
        int ox = (int)(oxk + (oxk >= 0.0 ? 0.5 : -0.5));
        int oy = (int)(oyk + (oyk >= 0.0 ? 0.5 : -0.5));
        SDL_RenderDrawLine(ren, ax0 + ox, ay0 + oy, ax1 + ox, ay1 + oy);
    }
}

#if SDL_VERSION_ATLEAST(2, 0, 18)
static int display_automap_append_stroke_quad(SDL_Vertex *verts, int max_verts, int *vert_count,
                                              int ax0, int ay0, int ax1, int ay1,
                                              Uint8 r, Uint8 g, Uint8 b, Uint8 a,
                                              float half_width)
{
    if (!verts || !vert_count || half_width <= 0.0f) return 1;
    if (*vert_count > max_verts - 6) return 0;

    float dx = (float)(ax1 - ax0);
    float dy = (float)(ay1 - ay0);
    float len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-6f) return 1;

    float inv_len = 1.0f / SDL_sqrtf(len_sq);
    float ox = -dy * inv_len * half_width;
    float oy = dx * inv_len * half_width;
    SDL_Color color = { r, g, b, a };

    SDL_Vertex *v = &verts[*vert_count];
    v[0].position.x = (float)ax0 + ox; v[0].position.y = (float)ay0 + oy; v[0].color = color; v[0].tex_coord.x = 0.0f; v[0].tex_coord.y = 0.0f;
    v[1].position.x = (float)ax1 + ox; v[1].position.y = (float)ay1 + oy; v[1].color = color; v[1].tex_coord.x = 0.0f; v[1].tex_coord.y = 0.0f;
    v[2].position.x = (float)ax1 - ox; v[2].position.y = (float)ay1 - oy; v[2].color = color; v[2].tex_coord.x = 0.0f; v[2].tex_coord.y = 0.0f;
    v[3].position.x = (float)ax0 + ox; v[3].position.y = (float)ay0 + oy; v[3].color = color; v[3].tex_coord.x = 0.0f; v[3].tex_coord.y = 0.0f;
    v[4].position.x = (float)ax1 - ox; v[4].position.y = (float)ay1 - oy; v[4].color = color; v[4].tex_coord.x = 0.0f; v[4].tex_coord.y = 0.0f;
    v[5].position.x = (float)ax0 - ox; v[5].position.y = (float)ay0 - oy; v[5].color = color; v[5].tex_coord.x = 0.0f; v[5].tex_coord.y = 0.0f;
    *vert_count += 6;
    return 1;
}
#endif

/* Parallel-offset stroke: black_k_max / fg_k_max are inclusive half-widths in pixels. */
static void display_automap_draw_line_outlined(SDL_Renderer *ren,
                                               int ax0, int ay0, int ax1, int ay1,
                                               Uint8 fr, Uint8 fg, Uint8 fb, Uint8 alpha,
                                               int black_k_max, int fg_k_max)
{
    if (g_gl_unpack_ok && g_gl_hud_ok) {
        display_automap_line_stroked_gl(ax0, ay0, ax1, ay1, fr, fg, fb, alpha,
                                        black_k_max, fg_k_max);
        (void)ren;
        return;
    }
    display_automap_line_stroked_sdl(ren, ax0, ay0, ax1, ay1, fr, fg, fb, alpha,
                                     black_k_max, fg_k_max);
}

/* Filled quadrilateral (base, left wing, tip, right wing) under player arrow outline. */
static void display_automap_player_arrow_fill(int axb, int ayb, int axl, int ayl, int axt, int ayt,
                                              int axr, int ayr)
{
#ifdef AB3D_DOS
    SDL_Vertex verts[4]={0};
    int dos_xy[4][2]={{axb,ayb},{axl,ayl},{axt,ayt},{axr,ayr}};
    const int indices[6]={0,1,2,0,2,3};
    for(int i=0;i<4;i++) {
        verts[i].position=(SDL_FPoint){(float)dos_xy[i][0],(float)dos_xy[i][1]};
        verts[i].color=(SDL_FColor){18/255.0f,60/255.0f,82/255.0f,1.0f};
    }
    SDL_SetRenderDrawBlendMode(g_sdl_ren,SDL_BLENDMODE_NONE);
    SDL_RenderGeometry(g_sdl_ren,NULL,verts,4,indices,6);
    return;
#endif
    float xy[12] = {
        (float)axb, (float)ayb, (float)axl, (float)ayl, (float)axt, (float)ayt,
        (float)axb, (float)ayb, (float)axt, (float)ayt, (float)axr, (float)ayr,
    };
    if (g_gl_unpack_ok && g_gl_hud_ok) {
        display_gl_solid_triangles_xy(xy, 6, 32, 110, 150, 140);
        return;
    }
#if SDL_VERSION_ATLEAST(2, 0, 18)
    if (!g_sdl_ren) return;
    SDL_SetRenderDrawBlendMode(g_sdl_ren, SDL_BLENDMODE_BLEND);
    {
        SDL_Vertex v[6];
        SDL_Color c = { 32, 110, 150, 140 };
        int k = 0;
        v[k].position.x = (float)axb; v[k].position.y = (float)ayb; v[k].color = c; v[k].tex_coord.x = 0; v[k].tex_coord.y = 0; k++;
        v[k].position.x = (float)axl; v[k].position.y = (float)ayl; v[k].color = c; v[k].tex_coord.x = 0; v[k].tex_coord.y = 0; k++;
        v[k].position.x = (float)axt; v[k].position.y = (float)ayt; v[k].color = c; v[k].tex_coord.x = 0; v[k].tex_coord.y = 0; k++;
        v[k].position.x = (float)axb; v[k].position.y = (float)ayb; v[k].color = c; v[k].tex_coord.x = 0; v[k].tex_coord.y = 0; k++;
        v[k].position.x = (float)axt; v[k].position.y = (float)ayt; v[k].color = c; v[k].tex_coord.x = 0; v[k].tex_coord.y = 0; k++;
        v[k].position.x = (float)axr; v[k].position.y = (float)ayr; v[k].color = c; v[k].tex_coord.x = 0; v[k].tex_coord.y = 0; k++;
        SDL_RenderGeometry(g_sdl_ren, NULL, v, 6, NULL, 0);
    }
#endif
}

/* Letterbox-relative coords (same space as hud_key_row_layout). */
static void display_key_hud_fill_rect(SDL_Renderer *ren, int lx, int ly, int lw, int lh,
                                      Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    SDL_Rect rr;
    rr.x = g_present_dst_rect.x + lx;
    rr.y = g_present_dst_rect.y + ly;
    rr.w = lw;
    rr.h = lh;
    (void)ren;
    display_overlay_fill_rect_abs(&rr, r, g, b, a);
}

static void display_key_hud_free_textures(void)
{
    for (int i = 0; i < 4; i++) {
        if (g_key_hud_tex[i]) {
            SDL_DestroyTexture(g_key_hud_tex[i]);
            g_key_hud_tex[i] = NULL;
        }
    }
}

/* Static HUD textures need filtered minification when their screen footprint
 * is smaller than the source. SDL handles non-GL renderers via ScaleMode. */
static void display_hud_texture_mipmaps(SDL_Texture *tex, GLint magnification)
{
#if SDL_VERSION_ATLEAST(2, 0, 12)
    if (g_gl_unpack_ok && g_gl_hud_ok) {
        if (!g_gl_generate_mipmap || !g_gl_tex_parameteri)
            display_load_gl_mipmap_procs();
        if (g_gl_generate_mipmap && g_gl_tex_parameteri) {
            float tw = 0.0f, th = 0.0f;
            if (SDL_GL_BindTexture(tex, &tw, &th) == 0) {
                g_gl_tex_parameteri(DGL_TEXTURE_2D, DGL_TEXTURE_MIN_FILTER,
                                    (GLint)DGL_LINEAR_MIPMAP_LINEAR);
                g_gl_tex_parameteri(DGL_TEXTURE_2D, DGL_TEXTURE_MAG_FILTER,
                                    magnification);
                g_gl_tex_parameteri(DGL_TEXTURE_2D, DGL_TEXTURE_WRAP_S,
                                    (GLint)DGL_CLAMP_TO_EDGE);
                g_gl_tex_parameteri(DGL_TEXTURE_2D, DGL_TEXTURE_WRAP_T,
                                    (GLint)DGL_CLAMP_TO_EDGE);
                g_gl_generate_mipmap(DGL_TEXTURE_2D);
                SDL_GL_UnbindTexture(tex);
            }
        }
    }
#else
    (void)tex;
    (void)magnification;
#endif
}

static SDL_Texture *display_key_hud_texture_for_frame(int frame_idx)
{
    if (frame_idx < 0 || frame_idx >= 4 || !g_sdl_ren) return NULL;
    if (g_key_hud_tex[frame_idx]) return g_key_hud_tex[frame_idx];

    uint32_t pixels[32 * 32];
    if (!renderer_key_sprite_rasterize_frame_argb(frame_idx, pixels, 32)) return NULL;

    SDL_Texture *t = SDL_CreateTexture(g_sdl_ren, SDL_PIXELFORMAT_ARGB8888,
                                         SDL_TEXTUREACCESS_STATIC, 32, 32);
    if (!t) return NULL;
    if (SDL_UpdateTexture(t, NULL, pixels, (int)(32 * sizeof(uint32_t))) != 0) {
        SDL_DestroyTexture(t);
        return NULL;
    }
    SDL_SetTextureBlendMode(t, SDL_BLENDMODE_BLEND);
#if SDL_VERSION_ATLEAST(2, 0, 12)
    SDL_SetTextureScaleMode(t, SDL_ScaleModeNearest);
    display_hud_texture_mipmaps(t, GL_NEAREST);
#endif
    g_key_hud_tex[frame_idx] = t;
    return t;
}

static void display_hud_digits_free(void)
{
    for (int i = 0; i < DISPLAY_HUD_DIGIT_KIND_COUNT; i++) {
        if (g_hud_digit_array_tex[i] && g_gl_delete_textures) {
            g_gl_delete_textures(1, &g_hud_digit_array_tex[i]);
        }
        g_hud_digit_array_tex[i] = 0;
        g_hud_digit_array_ok[i] = 0;
        g_hud_digit_array_upload_attempted[i] = 0;
        if (g_hud_digit_tex[i]) {
            SDL_DestroyTexture(g_hud_digit_tex[i]);
            g_hud_digit_tex[i] = NULL;
        }
        g_hud_digit_tex_w[i] = 0;
        g_hud_digit_tex_h[i] = 0;
    }
    g_hud_digits_load_attempted = 0;
}

static void display_hud_digit_upload_array_texture(int kind)
{
#if !defined(AB3D_DOS) && SDL_VERSION_ATLEAST(2, 0, 12)
    if (kind < 0 || kind >= DISPLAY_HUD_DIGIT_KIND_COUNT)
        return;

    int atlas_w = 0, atlas_h = 0;
    unsigned char *atlas = NULL;
    unsigned char *layers = NULL;

    g_hud_digit_array_ok[kind] = 0;
    if (!g_gl_unpack_ok || !g_gl_hud_ok || !g_gl_prog_hud_font)
        return;
    if (!g_gl_gen_textures || !g_gl_bind_texture || !g_gl_tex_image_3d ||
        !g_gl_tex_parameteri2 || !g_gl_delete_textures || !g_gl_pixel_storei)
        return;
    if (g_hud_digit_array_upload_attempted[kind])
        return;
    g_hud_digit_array_upload_attempted[kind] = 1;

    if (!g_gl_generate_mipmap)
        display_load_gl_mipmap_procs();
    if (!g_gl_generate_mipmap)
        return;

    atlas = display_load_png_rgba_asset(g_hud_digit_paths[kind],
                                        &atlas_w, &atlas_h,
                                        "HUD digits");
    if (!atlas)
        return;

    int scale_k = hud_digit_tex_scale_k(atlas_w, atlas_h);
    if (scale_k < 1) {
        stbi_image_free(atlas);
        return;
    }

    int layer_w = DISPLAY_HUD_DIGIT_CELL_W * scale_k;
    int layer_h = DISPLAY_HUD_DIGIT_CELL_H * scale_k;
    int stride = DISPLAY_HUD_DIGIT_STRIDE * scale_k;
    size_t layer_bytes = (size_t)layer_w * (size_t)layer_h * 4u;
    size_t total_bytes = layer_bytes * (size_t)DISPLAY_HUD_DIGIT_COUNT;

    layers = (unsigned char *)malloc(total_bytes);
    if (!layers) {
        stbi_image_free(atlas);
        return;
    }
    memset(layers, 0, total_bytes);

    for (int digit = 0; digit < DISPLAY_HUD_DIGIT_COUNT; digit++) {
        int src_x = digit * stride;
        unsigned char *dst_layer = layers + (size_t)digit * layer_bytes;
        if (src_x + layer_w > atlas_w)
            continue;
        for (int y = 0; y < layer_h; y++) {
            const unsigned char *src =
                atlas + ((size_t)y * (size_t)atlas_w +
                         (size_t)src_x) * 4u;
            unsigned char *dst =
                dst_layer + (size_t)y * (size_t)layer_w * 4u;
            memcpy(dst, src, (size_t)layer_w * 4u);
        }
    }

    if (g_hud_digit_array_tex[kind]) {
        g_gl_delete_textures(1, &g_hud_digit_array_tex[kind]);
        g_hud_digit_array_tex[kind] = 0;
    }

    if (g_gl_get_error) {
        for (int i = 0; i < 8 && g_gl_get_error() != 0; i++) {
        }
    }

    g_gl_gen_textures(1, &g_hud_digit_array_tex[kind]);
    if (!g_hud_digit_array_tex[kind]) {
        free(layers);
        stbi_image_free(atlas);
        return;
    }
    g_gl_bind_texture(DGL_TEXTURE_2D_ARRAY, g_hud_digit_array_tex[kind]);
    display_gl_reset_rgba_texture_unpack(1);
    g_gl_tex_parameteri2(DGL_TEXTURE_2D_ARRAY, DGL_TEXTURE_MIN_FILTER,
                         (GLint)DGL_LINEAR_MIPMAP_LINEAR);
    g_gl_tex_parameteri2(DGL_TEXTURE_2D_ARRAY, DGL_TEXTURE_MAG_FILTER,
                         (GLint)GL_NEAREST);
    g_gl_tex_parameteri2(DGL_TEXTURE_2D_ARRAY, DGL_TEXTURE_WRAP_S,
                         (GLint)DGL_CLAMP_TO_EDGE);
    g_gl_tex_parameteri2(DGL_TEXTURE_2D_ARRAY, DGL_TEXTURE_WRAP_T,
                         (GLint)DGL_CLAMP_TO_EDGE);
    g_gl_tex_image_3d(DGL_TEXTURE_2D_ARRAY, 0, (GLint)GL_RGBA8,
                      layer_w, layer_h, DISPLAY_HUD_DIGIT_COUNT, 0,
                      GL_RGBA, GL_UNSIGNED_BYTE, layers);
    g_gl_generate_mipmap(DGL_TEXTURE_2D_ARRAY);
    display_gl_reset_rgba_texture_unpack(4);
    g_gl_bind_texture(DGL_TEXTURE_2D_ARRAY, 0);

    if (g_gl_get_error && g_gl_get_error() != 0) {
        g_gl_delete_textures(1, &g_hud_digit_array_tex[kind]);
        g_hud_digit_array_tex[kind] = 0;
    } else {
        g_hud_digit_array_ok[kind] = 1;
        printf("[DISPLAY] HUD digits: GL texture array mipmaps active for %s\n",
               g_hud_digit_paths[kind]);
    }

    free(layers);
    stbi_image_free(atlas);
#else
    (void)kind;
#endif
}

static void display_hud_digits_ensure_loaded(void)
{
    if (g_hud_digits_load_attempted || !g_sdl_ren) return;
    g_hud_digits_load_attempted = 1;

    for (int i = 0; i < DISPLAY_HUD_DIGIT_KIND_COUNT; i++) {
        int w=0,h=0;
        SDL_Texture *t=display_load_ui_image_asset(g_hud_digit_paths[i],&w,&h,"HUD digits");
        if(!t)continue;
        g_hud_digit_tex[i]=t;g_hud_digit_tex_w[i]=w;g_hud_digit_tex_h[i]=h;
        display_hud_digit_upload_array_texture(i);
    }
}

/*
 * Bottom HUD: compact cluster bottom-right. Letterbox pixels (g_present_dst_rect);
 * scale ~1/32 of height so size tracks resolution without dominating the screen.
 */
static int hud_key_mark_height(int key_height)
{
    int height=key_height/10;
    return height<2?2:height;
}

static void hud_key_row_layout(int lay_w, int lay_h, int *margin, int *kh, int *gap, int *group_w,
                               int *ix_key0, int *iy)
{
    int m = lay_h / 64;
    if (m < 2) m = 2;
    int k = lay_h / 32;
    if (k < 11) k = 11;
    if (k > DISPLAY_HUD_DIGIT_CELL_H) {
        int whole = k / DISPLAY_HUD_DIGIT_CELL_H;
        if (whole < 1) whole = 1;
        k = whole * DISPLAY_HUD_DIGIT_CELL_H;
    }
    int g = k / 12;
    if (g < 2) g = 2;
#ifdef AB3D_DOS
    m=2;k=32;g=2;
#endif
    int gw = 4 * k + 3 * g;
    int ix = lay_w - m - gw;
    if (ix < 0) ix = 0;
    int y = lay_h - m - k;
#ifdef AB3D_DOS
    y-=hud_key_mark_height(k)+1;
#endif
    *margin = m;
    *kh = k;
    *gap = g;
    *group_w = gw;
    *ix_key0 = ix;
    *iy = y;
}

/*
 * Digit strip layout must match the generator (5x7 glyph, pad=1, spacing=2):
 *   stride = gw + pad*2 + 2 + spacing  (= 11 at scale 1)
 *   cell_w = gw + pad*2 + 2            (= 9)
 *   row_h  = gh + pad*2 + 2            (= 11)
 *   sheet_w = 108 * k, sheet_h = 11 * k for integer scale k (NEAREST upscale).
 * Do not use tex_w/10 — columns are not equal width; the last column is not "remainder".
 */
static int hud_digit_tex_scale_k(int tex_w, int tex_h)
{
    if (tex_h < DISPLAY_HUD_DIGIT_CELL_H ||
        tex_w < DISPLAY_HUD_DIGIT_SHEET_W) {
        return 0;
    }
    if (tex_h % DISPLAY_HUD_DIGIT_CELL_H != 0) return 0;
    int k = tex_h / DISPLAY_HUD_DIGIT_CELL_H;
    if (k < 1 || tex_w != DISPLAY_HUD_DIGIT_SHEET_W * k) return 0;
    return k;
}

static void hud_digit_src_rect(int tex_w, int tex_h, int digit, SDL_Rect *src)
{
    if (digit < 0) digit = 0;
    if (digit >= DISPLAY_HUD_DIGIT_COUNT)
        digit = DISPLAY_HUD_DIGIT_COUNT - 1;

    int k = hud_digit_tex_scale_k(tex_w, tex_h);
    if (k > 0) {
        int stride = DISPLAY_HUD_DIGIT_STRIDE * k;
        int cell_w = DISPLAY_HUD_DIGIT_CELL_W * k;
        int row_h = DISPLAY_HUD_DIGIT_CELL_H * k;
        src->x = digit * stride;
        src->y = 0;
        src->w = cell_w;
        src->h = row_h;
        return;
    }

    /* Fallback: generic 10-column atlas */
    int cell = tex_w / DISPLAY_HUD_DIGIT_COUNT;
    if (cell < 1) cell = 1;
    src->x = digit * cell;
    src->y = 0;
    src->w = (digit == DISPLAY_HUD_DIGIT_COUNT - 1) ?
        (tex_w - digit * cell) : cell;
    src->h = tex_h;
}

/* Widest scaled glyph — fixed column width so 0..9 align in three slots. */
static int hud_digit_crisp_height(int tex_h, int digit_h)
{
#ifdef AB3D_DOS
    return tex_h;
#endif
    if (tex_h < 1 || digit_h < 1)
        return digit_h;
    if (digit_h <= tex_h)
        return digit_h;

    int whole = digit_h / tex_h;
    if (whole < 1) whole = 1;
    return whole * tex_h;
}

static int hud_digit_gl_blit(int kind, int digit, const SDL_Rect *dst)
{
    if (kind < 0 || kind >= DISPLAY_HUD_DIGIT_KIND_COUNT)
        return 0;
    if (digit < 0 || digit >= DISPLAY_HUD_DIGIT_COUNT)
        return 0;
    if (!g_hud_digit_array_ok[kind] && g_hud_digit_tex[kind])
        display_hud_digit_upload_array_texture(kind);
    if (!g_hud_digit_array_ok[kind] || !g_hud_digit_array_tex[kind])
        return 0;

    return display_gl_array_layer_blit(g_hud_digit_array_tex[kind], digit,
                                       DISPLAY_HUD_DIGIT_COUNT,
                                       0.0f, 0.0f, 1.0f, 1.0f, dst, 255);
}

static int hud_max_digit_scaled_width(int tex_w, int tex_h, int digit_h)
{
    int mw = 0;
    for (int d = 0; d < DISPLAY_HUD_DIGIT_COUNT; d++) {
        SDL_Rect src;
        hud_digit_src_rect(tex_w, tex_h, d, &src);
        int dw = (src.w * digit_h + src.h / 2) / src.h;
        if (dw < 1) dw = 1;
        if (dw > mw) mw = dw;
    }
    return mw;
}

static int hud_three_slot_width(int tex_w, int tex_h, int digit_h, int gap)
{
    int sw = hud_max_digit_scaled_width(tex_w, tex_h, digit_h);
    return sw * 3 + gap * 2;
}

/*
 * value 0..max_value as %03d with up to two leading zeros not drawn; slots stay fixed width
 * (health max 100, ammo count max 999).
 * x_left, y_top, digit_h, gap are in letterbox pixels (same space as hud_key_row_layout).
 */
static void hud_draw_three_slot_value(int kind, SDL_Texture *tex, int tex_w, int tex_h,
                                      int digit_h, int value, int max_value,
                                      int x_left, int y_top, int gap)
{
    if (value < 0) value = 0;
    if (value > max_value) value = max_value;
    char b[8];
    snprintf(b, sizeof(b), "%03d", value);

    int slot_w = hud_max_digit_scaled_width(tex_w, tex_h, digit_h);
    int bx = g_present_dst_rect.x;
    int by = g_present_dst_rect.y;
    int sy0 = by + y_top;
    int sy1 = by + y_top + digit_h;
    int sh = sy1 - sy0;
    if (sh < 1) sh = 1;

    int start;
    if (value == 0) {
        start = 2;
    } else {
        start = 0;
        while (start < 2 && b[start] == '0')
            start++;
    }

    for (int s = start; s < 3; s++) {
        int d = b[s] - '0';
        if (d < 0 || d > 9) continue;
        SDL_Rect src;
        hud_digit_src_rect(tex_w, tex_h, d, &src);
        int dw = (src.w * digit_h + src.h / 2) / src.h;
        if (dw < 1) dw = 1;
        int slot_ix = x_left + s * (slot_w + gap);
        int dx = slot_ix + (slot_w - dw) / 2;

        int px0i = bx + dx;
        int px1i = bx + dx + dw;
        SDL_Rect dst;
        dst.x = px0i;
        dst.y = sy0;
        dst.w = px1i - px0i;
        if (dst.w < 1) dst.w = 1;
        dst.h = sh;
        if (!hud_digit_gl_blit(kind, d, &dst)) {
            display_overlay_copy(tex, &src, &dst);
        }
    }
}

/* Top-left FPS (ammo_digits.png atlas); letterbox-relative coords match hud_key_row_layout margins. */
static void display_fps_overlay(const GameState *state)
{
    if (!state || !state->cfg_show_fps || !g_sdl_ren) return;

    int pw = g_present_dst_rect.w;
    int ph = g_present_dst_rect.h;
    if (pw < 8 || ph < 8) return;

    display_hud_digits_ensure_loaded();
    SDL_Texture *tex = g_hud_digit_tex[1];
    if (!tex) return;

    int tex_w = g_hud_digit_tex_w[1];
    int tex_h = g_hud_digit_tex_h[1];
    SDL_SetTextureAlphaMod(tex, 255);

    int margin = ph / 64;
    if (margin < 2) margin = 2;
    int digit_h = ph / 32;
    if (digit_h < 11) digit_h = 11;
    digit_h = hud_digit_crisp_height(tex_h, digit_h);
    int d_gap = digit_h / 16;
    if (d_gap < 1) d_gap = 1;

    int value = (int)state->fps_display;
    if (value < 0) value = 0;
    if (value > 9999) value = 9999;
    char buf[8];
    snprintf(buf, sizeof(buf), "%d", value);
    size_t len = strlen(buf);

    int slot_w = hud_max_digit_scaled_width(tex_w, tex_h, digit_h);
    int bx = g_present_dst_rect.x;
    int by = g_present_dst_rect.y;
    int x_left = margin;
    int y_top = margin;
    int sy0 = by + y_top;
    int sy1 = sy0 + digit_h;
    int sh = sy1 - sy0;
    if (sh < 1) sh = 1;

    for (size_t i = 0; i < len; i++) {
        int d = buf[i] - '0';
        if (d < 0 || d > 9) continue;
        SDL_Rect src;
        hud_digit_src_rect(tex_w, tex_h, d, &src);
        int dw = (src.w * digit_h + src.h / 2) / src.h;
        if (dw < 1) dw = 1;
        int slot_ix = x_left + (int)i * (slot_w + d_gap);
        int dx = slot_ix + (slot_w - dw) / 2;
        int px0i = bx + dx;
        int px1i = bx + dx + dw;
        SDL_Rect dst;
        dst.x = px0i;
        dst.y = sy0;
        dst.w = px1i - px0i;
        if (dst.w < 1) dst.w = 1;
        dst.h = sh;
        if (!hud_digit_gl_blit(1, d, &dst)) {
            display_overlay_copy(tex, &src, &dst);
        }
    }
}

typedef struct {
    Uint8 r, g, b;
} DisplayCrosshairRgb;

/* Ported from TKG modules/draw.s Draw_Crosshair / Draw_CrosshairPens_vb.
 * TKG stores palette pens; the modern overlay draws the same 0..7 slots as RGB. */
static const uint8_t k_tkg_crosshair_pens[8] = {
    0, 255, 254, 190, 25, 250, 133, 69
};

static const DisplayCrosshairRgb k_tkg_crosshair_rgb[8] = {
    {   0,   0,   0 }, /* off */
    {   0, 255,   0 }, /* intense green */
    {   0, 176,   0 }, /* mid green */
    { 255, 255,   0 }, /* intense yellow */
    { 216, 216, 216 }, /* bright grey */
    { 255,   0,   0 }, /* intense red */
    { 128, 224, 255 }, /* ice blue */
    {   0,  64, 255 }  /* intense blue */
};

static int display_round_div_i64(int64_t n, int d)
{
    if (d <= 0) return 0;
    return (int)((n >= 0)
        ? ((n + d / 2) / d)
        : ((n - d / 2) / d));
}

static int display_clamp_int(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void display_crosshair_dot(int cx, int cy, int dot, Uint8 r, Uint8 g, Uint8 b)
{
    SDL_Rect rr;
    if (dot < 1) dot = 1;
    rr.x = cx - dot / 2;
    rr.y = cy - dot / 2;
    rr.w = dot;
    rr.h = dot;
    display_overlay_fill_rect_abs(&rr, r, g, b, 255);
}

static void display_crosshair_sdl_overlay(const GameState *state)
{
    if (!state || !g_sdl_ren) return;

    uint8_t slot = (uint8_t)(state->cfg_crosshair_colour & 7u);
    if (slot == 0 || k_tkg_crosshair_pens[slot] == 0) return;

    int pw = g_present_dst_rect.w;
    int ph = g_present_dst_rect.h;
    if (pw < 16 || ph < 16) return;

    int cx = g_present_dst_rect.x + pw / 2;
    int cy = g_present_dst_rect.y + ph / 2;

    int margin = 5;
    int unit = display_round_div_i64((int64_t)ph, RENDER_DEFAULT_HEIGHT);
#ifdef AB3D_DOS
    unit=1;
#endif
    if (unit < 1) unit = 1;
    if (unit > 4) unit = 4;
    int dot = unit;

    cx = display_clamp_int(cx, g_present_dst_rect.x + margin * unit,
                           g_present_dst_rect.x + pw - 1 - margin * unit);
    cy = display_clamp_int(cy, g_present_dst_rect.y + margin * unit,
                           g_present_dst_rect.y + ph - 1 - margin * unit);

    const DisplayCrosshairRgb rgb = k_tkg_crosshair_rgb[slot];
    static const int8_t offsets[8][2] = {
        { -4, -4 }, {  4, -4 },
        { -2, -2 }, {  2, -2 },
        { -2,  2 }, {  2,  2 },
        { -4,  4 }, {  4,  4 }
    };
    for (int i = 0; i < 8; i++) {
        display_crosshair_dot(cx + offsets[i][0] * unit,
                              cy + offsets[i][1] * unit,
                              dot, rgb.r, rgb.g, rgb.b);
    }
}

static void display_hud_numbers_sdl_overlay(const GameState *state)
{
    if (!state || !g_sdl_ren) return;

    int pw = g_present_dst_rect.w;
    int ph = g_present_dst_rect.h;
    if (pw < 8 || ph < 8) return;

    display_hud_digits_ensure_loaded();
    if (!g_hud_digit_tex[0] && !g_hud_digit_tex[1]) return;

    int margin, kh, gap_keys, group_w, ix_key0, iy;
    hud_key_row_layout(pw, ph, &margin, &kh, &gap_keys, &group_w, &ix_key0, &iy);

    int16_t e = state->energy;
    if (e < 0) e = 0;
    if (e > PLAYER_MAX_ENERGY) e = PLAYER_MAX_ENERGY;
    int hp_pct = (e * 100 + PLAYER_MAX_ENERGY / 2) / PLAYER_MAX_ENERGY;
    if (hp_pct < 0) hp_pct = 0;
    if (hp_pct > 100) hp_pct = 100;

    /* Previous numeric HUD: health percent and displayed ammo count. */
    int ammo_count;
    if (state->infinite_ammo) {
        ammo_count = (int)MAX_AMMO_RAW;
    } else {
        ammo_count = (int)state->ammo;
        if (ammo_count < 0) ammo_count = 0;
        if (ammo_count > (int)MAX_AMMO_RAW) ammo_count = (int)MAX_AMMO_RAW;
    }

    /* Horizontal margin between health | ammo | keys (letterbox pixels). */
    int gap_stat = kh / 4;
    if (gap_stat < 4) gap_stat = 4;
    int d_gap = kh / 16;
    if (d_gap < 1) d_gap = 1;

    int digit_base_h = 0;
    if (g_hud_digit_tex[0]) {
        digit_base_h = g_hud_digit_tex_h[0];
    } else if (g_hud_digit_tex[1]) {
        digit_base_h = g_hud_digit_tex_h[1];
    }

    int digit_h = hud_digit_crisp_height(digit_base_h, kh);
    int health_w = 0, ammo_w = 0;
    if (g_hud_digit_tex[0])
        health_w = hud_three_slot_width(g_hud_digit_tex_w[0], g_hud_digit_tex_h[0], digit_h, d_gap);
    if (g_hud_digit_tex[1])
        ammo_w = hud_three_slot_width(g_hud_digit_tex_w[1], g_hud_digit_tex_h[1], digit_h, d_gap);

#ifndef AB3D_DOS
    /* Shrink digits if stats would extend left of the margin (health | ammo | gap | keys). */
    for (int attempt = 0; attempt < 10; attempt++) {
        int inner = (health_w > 0 && ammo_w > 0) ? gap_stat : 0;
        int stats_total = health_w + inner + ammo_w;
        int leftmost = ix_key0 - gap_stat - stats_total;
        if (leftmost >= margin) break;
        digit_h = hud_digit_crisp_height(digit_base_h, digit_h * 9 / 10);
        if (digit_h < 6) break;
        health_w = g_hud_digit_tex[0] ? hud_three_slot_width(g_hud_digit_tex_w[0], g_hud_digit_tex_h[0], digit_h, d_gap) : 0;
        ammo_w = g_hud_digit_tex[1] ? hud_three_slot_width(g_hud_digit_tex_w[1], g_hud_digit_tex_h[1], digit_h, d_gap) : 0;
    }
#else
    iy+=kh-digit_h;
#endif

    /* Right edge of stat block sits just left of the key row (same baseline iy). */
    int right = ix_key0 - gap_stat;
    if (g_hud_digit_tex[1] && ammo_w > 0) {
        int ammo_x = right - ammo_w;
        SDL_SetTextureAlphaMod(g_hud_digit_tex[1], 255);
        hud_draw_three_slot_value(1, g_hud_digit_tex[1], g_hud_digit_tex_w[1], g_hud_digit_tex_h[1],
                                  digit_h, ammo_count, (int)MAX_AMMO_RAW, ammo_x, iy, d_gap);
        right = ammo_x - gap_stat;
    }
    if (g_hud_digit_tex[0] && health_w > 0) {
        int health_x = right - health_w;
        SDL_SetTextureAlphaMod(g_hud_digit_tex[0], 255);
        hud_draw_three_slot_value(0, g_hud_digit_tex[0], g_hud_digit_tex_w[0], g_hud_digit_tex_h[0],
                                  digit_h, hp_pct, 100, health_x, iy, d_gap);
    }
}

/* AB3DI.s EnergyBar/AmmoBar: 8-pixel strips in attached border sprites,
 * starting at row 24 (25*16 includes the sprite control words). Health is
 * on the left, ammo on the right; both drain downwards from the top.
 * Preserve the original strip silhouettes and ScreenSetup.s HealthPal ramp. */
static SDL_Texture *display_bar_hud_texture(int kind)
{
    if (g_bar_hud_tex[kind]) return g_bar_hud_tex[kind];
    char path[512];
    io_make_data_path(path, sizeof(path), kind ? "gfx/AmmoStrip" : "gfx/HealthStrip");
    size_t size = 0;
    Uint8 *strip = (Uint8 *)SDL_LoadFile(path, &size);
    if (!strip || size != 512) {
        fprintf(stderr, "[DISPLAY] HUD strip %s: expected 512 bytes, got %zu (%s)\n",
                path, size, SDL_GetError());
        SDL_free(strip);
        exit(1);
    }
    io_make_data_path(path, sizeof(path), "pal/HealthPal");
    size_t palette_size = 0;
    Uint8 *palette = (Uint8 *)SDL_LoadFile(path, &palette_size);
    if (!palette || palette_size != 160) {
        fprintf(stderr, "[DISPLAY] HUD palette %s: expected 160 bytes, got %zu (%s)\n",
                path, palette_size, SDL_GetError());
        SDL_free(strip);
        SDL_free(palette);
        exit(1);
    }
    uint32_t pixels[8 * 128];
    for (int y = 0; y < 128; y++) {
        /* ScreenSetup.s advances HealthPal once per two physical rows.
         * The sprite strip begins 24 rows below the view's top edge. */
        int pal_row = (24 + y) / 2;
        uint16_t c12 = (uint16_t)((palette[pal_row * 2] << 8) |
                                  palette[pal_row * 2 + 1]);
        for (int x = 0; x < 8; x++) {
            int colour = 0;
            for (int plane = 0; plane < 4; plane++)
                colour |= ((strip[y * 4 + plane] >> (7 - x)) & 1) << plane;
            /* Keep zero transparent and the original four-plane shading. */
            Uint8 r = (Uint8)(((c12 >> 8) & 15) * 17 * colour / 15);
            Uint8 g = (Uint8)(((c12 >> 4) & 15) * 17 * colour / 15);
            Uint8 b = (Uint8)((c12 & 15) * 17 * colour / 15);
            pixels[y * 8 + x] = colour ? (0x60000000u | ((uint32_t)r << 16) |
                                         ((uint32_t)g << 8) | b) : 0;
        }
    }
    SDL_free(strip);
    SDL_free(palette);
    SDL_Texture *tex = SDL_CreateTexture(g_sdl_ren, SDL_PIXELFORMAT_ARGB8888,
                                         SDL_TEXTUREACCESS_STATIC, 8, 128);
    if (!tex || SDL_UpdateTexture(tex, NULL, pixels, 8 * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "[DISPLAY] Cannot upload HUD bar: %s\n", SDL_GetError());
        exit(1);
    }
    SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_BLEND);
#if SDL_VERSION_ATLEAST(2, 0, 12)
    SDL_SetTextureScaleMode(tex, SDL_ScaleModeLinear);
    display_hud_texture_mipmaps(tex, GL_LINEAR);
#endif
    g_bar_hud_tex[kind] = tex;
    return tex;
}

/* Anchor the fill to the track bottom and round upward: a non-zero amount
 * must remain visible even when one source row is less than one screen pixel. */
static SDL_Rect display_hud_bar_fill_rect(SDL_Rect track, int value, int maximum)
{
    if (value < 0) value = 0;
    if (value > maximum) value = maximum;
    int filled = (value * track.h + maximum - 1) / maximum;
    SDL_Rect fill = { track.x, track.y + track.h - filled, track.w, filled };
    return fill;
}

static void display_hud_stats_sdl_overlay(const GameState *state)
{
    if (!state || !g_sdl_ren) return;
    if (!state->cfg_hud_bars) {
        display_hud_numbers_sdl_overlay(state);
        return;
    }
    const PlayerState *plr = state->mode == MODE_SLAVE ? &state->plr2 : &state->plr1;
    int values[2] = { plr->energy, state->ammo };
    if (state->infinite_ammo) values[1] = 63;
    int pw = g_present_dst_rect.w, ph = g_present_dst_rect.h;
    if (pw < 8 || ph < 8) return;
    /* Compact borderless HUD requested from the Amiga reference: half-view
     * height, narrow strips, centred vertically at the side edges. */
    int bw = ph * 4 / 256;
    if (bw < 4) bw = 4;
    int margin = ph * 8 / 256;
    int full_h = ph * 128 / 256;
#ifdef AB3D_DOS
    bw=8;margin=8;full_h=128;
#endif
    int top = (ph - full_h) / 2;
#ifdef AB3D_DOS
    int key_margin,key_h,key_gap,key_w,key_x,key_y;
    hud_key_row_layout(pw,ph,&key_margin,&key_h,&key_gap,&key_w,&key_x,&key_y);
    /* A 200-line mode needs room above the native key row as well. */
    int max_top=key_y-2-full_h-1;
    if(top>max_top)top=max_top;
#endif
    for (int kind = 0; kind < 2; kind++) {
        int maximum = kind ? 63 : PLAYER_MAX_ENERGY;
        int value = values[kind];
        if (value < 0) value = 0;
        if (value > maximum) value = maximum;
        int x = kind ? pw - margin - bw : margin;
        display_key_hud_fill_rect(g_sdl_ren, x - 1, top - 1, bw + 2, full_h + 2,
                                  12, 12, 12, 24);
        int missing_rows = kind ? (63 - value) * 2 : PLAYER_MAX_ENERGY - value;
        int rows = kind ? value * 2 : value;
        if (rows == 0) continue;
        SDL_Rect src = { 0, missing_rows, 8, rows };
        SDL_Rect track = { g_present_dst_rect.x + x, g_present_dst_rect.y + top,
                           bw, full_h };
        SDL_Rect dst = display_hud_bar_fill_rect(track, value, maximum);
#ifdef AB3D_DOS
        dst.y=track.y+track.h-rows;dst.h=rows;
#endif
        display_overlay_copy(display_bar_hud_texture(kind), &src, &dst);
    }
}

static void display_hit_flash_overlay(const GameState *state)
{
    if (!state || state->infinite_health) return;
    int hit = state->mode == MODE_SLAVE ? state->hitcol2 : state->hitcol;
    if (hit <= 0) return;
    if (hit > 0xF00) hit = 0xF00;
#ifdef AB3D_DOS
    /* Keep damage feedback without blending every world pixel on a 486. */
    SDL_Rect r = g_present_dst_rect;
    int red = hit * 255 / 0xF00;
    SDL_Rect edges[4] = {{r.x,r.y,r.w,2},{r.x,r.y+r.h-2,r.w,2},
                         {r.x,r.y+2,2,r.h-4},{r.x+r.w-2,r.y+2,2,r.h-4}};
    for (int i=0;i<4;i++) display_overlay_fill_rect_abs(&edges[i],red,0,0,255);
#else
    /* Original hitcol red pulse, adapted to a borderless view. */
    display_overlay_fill_rect_abs(&g_present_dst_rect, 255, 0, 0,
                                   (Uint8)(hit * 88 / 0xF00));
#endif
}

static void display_key_hud_sdl_overlay(const GameState *state)
{
    if (!state || !g_sdl_ren) return;

    int pw = g_present_dst_rect.w;
    int ph = g_present_dst_rect.h;
    if (pw < 8 || ph < 8) return;

    uintptr_t tag = renderer_key_sprite_hud_cache_tag(state);
    if (tag != g_key_hud_tex_tag) {
        display_key_hud_free_textures();
        g_key_hud_tex_tag = tag;
    }

    int margin, kh, gap, group_w, ix0, iy;
    hud_key_row_layout(pw, ph, &margin, &kh, &gap, &group_w, &ix0, &iy);

    static const Uint8 fallback_rgb[4][3] = {
        { 255, 210, 32 },
        { 255, 64, 64 },
        { 64, 220, 96 },
        { 64, 140, 255 },
    };

    for (int i = 0; i < 4; i++) {
        uint8_t bit = (uint8_t)(1u << i);
        int collected = (((uint16_t)game_conditions & (uint16_t)bit) != 0);
        Uint8 alpha = collected ? (Uint8)255 : (Uint8)40;
#ifdef AB3D_DOS
        /* The DOS cache selects opaque grey for a disabled key. */
        alpha = collected ? 255 : 128;
#endif

        int frame = renderer_key_sprite_frame_for_condition_bit(state, i);
        if (frame < 0) frame = 0;
        if (frame > 3) frame = 3;

        int ix = ix0 + i * (kh + gap);

        if (collected) {
            /* An underline identifies ownership even against bright scenery. */
            int mark_h = hud_key_mark_height(kh);
            display_key_hud_fill_rect(g_sdl_ren, ix, iy + kh + 1, kh, mark_h,
                                      255, 245, 200, 255);
        }
        SDL_Texture *tex = display_key_hud_texture_for_frame(frame);
        if (tex) {
            SDL_SetTextureAlphaMod(tex, alpha);
            SDL_Rect dst;
            dst.x = g_present_dst_rect.x + ix;
            dst.y = g_present_dst_rect.y + iy;
            dst.w = kh;
            dst.h = kh;
            display_overlay_copy(tex, NULL, &dst);
            continue;
        }

        /* Fallback if key .wad/.ptr not loaded */
        int kw = (kh * 5) / 8;
        if (kw < 10) kw = 10;
        int cx = ix + kh / 2;
        ix = cx - kw / 2;
        uint16_t c12 = renderer_key_condition_bit_color_c12(state, i);
        Uint8 r, g, b;
        if (c12 != 0)
            display_automap_amiga12_to_rgb(c12, &r, &g, &b);
        else {
            r = fallback_rgb[i][0];
            g = fallback_rgb[i][1];
            b = fallback_rgb[i][2];
        }
#ifdef AB3D_DOS
        if (!collected) r = g = b = 112;
#endif
        int bow_w = kw * 2 / 3;
        int bow_h = kh * 5 / 12;
        display_key_hud_fill_rect(g_sdl_ren, ix, iy, bow_w, bow_h, r, g, b, alpha);
        int sw = kw / 3;
        if (sw < 3) sw = 3;
        int sx = ix + kw / 3;
        int sy = iy + kh / 4;
        int sh = kh - kh / 4;
        display_key_hud_fill_rect(g_sdl_ren, sx, sy, sw, sh, r, g, b, alpha);
        int tw = kw / 3;
        int th = kh / 7;
        if (th < 2) th = 2;
        int tx = ix + kw * 2 / 3;
        int ty = iy + kh - th - margin / 2;
        if (ty < sy + sh / 2) ty = sy + sh / 2;
        display_key_hud_fill_rect(g_sdl_ren, tx, ty, tw, th, r, g, b, alpha);
    }
}

#ifdef AB3D_DOS
static void display_automap_sdl_overlay_impl(GameState *state)
#else
static void display_automap_sdl_overlay(GameState *state)
#endif
{
    if (!state || !state->automap_visible || !g_sdl_ren) return;

    /* ~264 KiB: must not live on the stack — Emscripten default stack is 64 KiB and overflow
     * here caused a solid tab freeze when opening the map on web. */
#ifdef AB3D_DOS
    unsigned seen=state->level.automap_seen_count;
    int wanted=seen>DISPLAY_AUTOMAP_MAX_SEGS-3?DISPLAY_AUTOMAP_MAX_SEGS:(int)seen+3;
    if(wanted>g_dos_map_capacity) {
        unsigned char *scratch=realloc(g_dos_map_scratch,(size_t)wanted*(4*sizeof(int)+sizeof(uint16_t)));
        if(!scratch){fprintf(stderr,"Cannot allocate DOS map segments\n");exit(1);}
        g_dos_map_scratch=scratch;g_dos_map_capacity=wanted;
    }
    int *s_ix0=(int *)g_dos_map_scratch,*s_iy0=s_ix0+g_dos_map_capacity;
    int *s_ix1=s_iy0+g_dos_map_capacity,*s_iy1=s_ix1+g_dos_map_capacity;
    uint16_t *s_c12=(uint16_t *)(s_iy1+g_dos_map_capacity);
    int segment_capacity=g_dos_map_capacity;
#else
    static int s_ix0[DISPLAY_AUTOMAP_MAX_SEGS];
    static int s_iy0[DISPLAY_AUTOMAP_MAX_SEGS];
    static int s_ix1[DISPLAY_AUTOMAP_MAX_SEGS];
    static int s_iy1[DISPLAY_AUTOMAP_MAX_SEGS];
    static uint16_t s_c12[DISPLAY_AUTOMAP_MAX_SEGS];
    int segment_capacity=DISPLAY_AUTOMAP_MAX_SEGS;
#endif

    int iw=renderer_get_width(),ih=renderer_get_height();
#ifdef AB3D_DOS
    iw=g_present_dst_rect.w;ih=g_present_dst_rect.h;
#endif
    int n = renderer_automap_collect_line_segments(state, s_ix0, s_iy0, s_ix1, s_iy1, s_c12,
                                                   segment_capacity,iw,ih);
    if (n <= 0) return;

#if SDL_VERSION_ATLEAST(2, 0, 14)
    if (g_gl_unpack_ok && g_gl_hud_ok)
        SDL_RenderFlush(g_sdl_ren);
#endif


    int wall_end = n;
    if (n >= 3 && (s_c12[n - 3] & RENDERER_AUTOMAP_SEGFLAG_PLAYER))
        wall_end = n - 3;

    if (g_gl_unpack_ok && g_gl_hud_ok) {
        enum { DISPLAY_AUTOMAP_GL_MAX_BUCKETS = 16 };
        static float s_black_xy[DISPLAY_AUTOMAP_MAX_SEGS * 28];
        static float s_fg_xy[DISPLAY_AUTOMAP_MAX_SEGS * 12];
        DisplayAutomapGlLineBucket black_buckets[DISPLAY_AUTOMAP_GL_MAX_BUCKETS];
        DisplayAutomapGlLineBucket fg_buckets[DISPLAY_AUTOMAP_GL_MAX_BUCKETS];
        int black_bucket_count = 0;
        int fg_bucket_count = 0;
        int batched = 1;

        memset(black_buckets, 0, sizeof(black_buckets));
        memset(fg_buckets, 0, sizeof(fg_buckets));

        for (int i = 0; i < wall_end; i++) {
            uint16_t seg = s_c12[i];
            Uint8 alpha = (seg & RENDERER_AUTOMAP_SEGFLAG_INTERNAL) ? (Uint8)128 : (Uint8)255;
            Uint8 fr, fg, fb;
            int black_idx;
            int fg_idx;

            display_automap_amiga12_to_rgb(seg, &fr, &fg, &fb);
            black_idx = display_automap_gl_find_bucket(black_buckets, &black_bucket_count,
                                                       DISPLAY_AUTOMAP_GL_MAX_BUCKETS,
                                                       0, 0, 0, alpha);
            fg_idx = display_automap_gl_find_bucket(fg_buckets, &fg_bucket_count,
                                                    DISPLAY_AUTOMAP_GL_MAX_BUCKETS,
                                                    fr, fg, fb, alpha);
            if (black_idx < 0 || fg_idx < 0) {
                batched = 0;
                break;
            }
            black_buckets[black_idx].count += 28;
            fg_buckets[fg_idx].count += 12;
        }

        if (batched) {
            display_automap_gl_prepare_buckets(black_buckets, black_bucket_count);
            display_automap_gl_prepare_buckets(fg_buckets, fg_bucket_count);

            for (int i = 0; i < wall_end; i++) {
                int ax0, ay0, ax1, ay1;
                uint16_t seg = s_c12[i];
                Uint8 alpha = (seg & RENDERER_AUTOMAP_SEGFLAG_INTERNAL) ? (Uint8)128 : (Uint8)255;
                Uint8 fr, fg, fb;
                int black_idx;
                int fg_idx;
                int dx;
                int dy;
                double len;
                double px;
                double py;

                display_automap_map_pt(s_ix0[i], s_iy0[i], iw, ih, &ax0, &ay0);
                display_automap_map_pt(s_ix1[i], s_iy1[i], iw, ih, &ax1, &ay1);
                display_automap_amiga12_to_rgb(seg, &fr, &fg, &fb);
                black_idx = display_automap_gl_find_bucket(black_buckets, &black_bucket_count,
                                                           DISPLAY_AUTOMAP_GL_MAX_BUCKETS,
                                                           0, 0, 0, alpha);
                fg_idx = display_automap_gl_find_bucket(fg_buckets, &fg_bucket_count,
                                                        DISPLAY_AUTOMAP_GL_MAX_BUCKETS,
                                                        fr, fg, fb, alpha);
                if (black_idx < 0 || fg_idx < 0) {
                    batched = 0;
                    break;
                }

                dx = ax1 - ax0;
                dy = ay1 - ay0;
                len = hypot((double)dx, (double)dy);
                if (len < 1e-6) continue;
                px = -dy / len;
                py = dx / len;

                for (int k = -3; k <= 3; k++) {
                    double oxk = k * px;
                    double oyk = k * py;
                    float ox = (float)(oxk + (oxk >= 0.0 ? 0.5 : -0.5));
                    float oy = (float)(oyk + (oyk >= 0.0 ? 0.5 : -0.5));
                    display_automap_gl_append_bucket_line(s_black_xy, (int)(sizeof(s_black_xy) / sizeof(s_black_xy[0])),
                                                          &black_buckets[black_idx],
                                                          (float)ax0 + ox, (float)ay0 + oy,
                                                          (float)ax1 + ox, (float)ay1 + oy);
                }
                for (int k = -1; k <= 1; k++) {
                    double oxk = k * px;
                    double oyk = k * py;
                    float ox = (float)(oxk + (oxk >= 0.0 ? 0.5 : -0.5));
                    float oy = (float)(oyk + (oyk >= 0.0 ? 0.5 : -0.5));
                    display_automap_gl_append_bucket_line(s_fg_xy, (int)(sizeof(s_fg_xy) / sizeof(s_fg_xy[0])),
                                                          &fg_buckets[fg_idx],
                                                          (float)ax0 + ox, (float)ay0 + oy,
                                                          (float)ax1 + ox, (float)ay1 + oy);
                }
            }
        }

        if (batched) {
            for (int i = 0; i < black_bucket_count; i++) {
                if (black_buckets[i].count <= 0) continue;
                display_gl_lines_xy(s_black_xy + black_buckets[i].begin,
                                    black_buckets[i].count / 2,
                                    black_buckets[i].r,
                                    black_buckets[i].g,
                                    black_buckets[i].b,
                                    black_buckets[i].a);
            }
            for (int i = 0; i < fg_bucket_count; i++) {
                if (fg_buckets[i].count <= 0) continue;
                display_gl_lines_xy(s_fg_xy + fg_buckets[i].begin,
                                    fg_buckets[i].count / 2,
                                    fg_buckets[i].r,
                                    fg_buckets[i].g,
                                    fg_buckets[i].b,
                                    fg_buckets[i].a);
            }
        } else {
            for (int i = 0; i < wall_end; i++) {
                int ax0, ay0, ax1, ay1;
                display_automap_map_pt(s_ix0[i], s_iy0[i], iw, ih, &ax0, &ay0);
                display_automap_map_pt(s_ix1[i], s_iy1[i], iw, ih, &ax1, &ay1);
                uint16_t seg = s_c12[i];
                Uint8 alpha = (seg & RENDERER_AUTOMAP_SEGFLAG_INTERNAL) ? (Uint8)128 : (Uint8)255;
                Uint8 fr, fg, fb;
                display_automap_amiga12_to_rgb(seg, &fr, &fg, &fb);
                display_automap_draw_line_outlined(g_sdl_ren, ax0, ay0, ax1, ay1, fr, fg, fb, alpha, 3, 1);
            }
        }
    } else {

#if SDL_VERSION_ATLEAST(2, 0, 18)
        static SDL_Vertex s_black_verts[DISPLAY_AUTOMAP_MAX_SEGS * 6];
        static SDL_Vertex s_fg_verts[DISPLAY_AUTOMAP_MAX_SEGS * 6];
        int black_vert_count = 0;
        int fg_vert_count = 0;
        int batched = 1;

        for (int i = 0; i < wall_end; i++) {
            int ax0, ay0, ax1, ay1;
            display_automap_map_pt(s_ix0[i], s_iy0[i], iw, ih, &ax0, &ay0);
            display_automap_map_pt(s_ix1[i], s_iy1[i], iw, ih, &ax1, &ay1);
            uint16_t seg = s_c12[i];
            Uint8 alpha = (seg & RENDERER_AUTOMAP_SEGFLAG_INTERNAL) ? (Uint8)128 : (Uint8)255;
            Uint8 fr, fg, fb;
            display_automap_amiga12_to_rgb(seg, &fr, &fg, &fb);

            if (!display_automap_append_stroke_quad(s_black_verts, DISPLAY_AUTOMAP_MAX_SEGS * 6,
                                                    &black_vert_count,
                                                    ax0, ay0, ax1, ay1,
                                                    0, 0, 0, alpha, 3.5f) ||
                !display_automap_append_stroke_quad(s_fg_verts, DISPLAY_AUTOMAP_MAX_SEGS * 6,
                                                    &fg_vert_count,
                                                    ax0, ay0, ax1, ay1,
                                                    fr, fg, fb, alpha, 1.5f)) {
                batched = 0;
                break;
            }
        }

        if (batched) {
            SDL_SetRenderDrawBlendMode(g_sdl_ren, SDL_BLENDMODE_BLEND);
            if (black_vert_count > 0)
                SDL_RenderGeometry(g_sdl_ren, NULL, s_black_verts, black_vert_count, NULL, 0);
            if (fg_vert_count > 0)
                SDL_RenderGeometry(g_sdl_ren, NULL, s_fg_verts, fg_vert_count, NULL, 0);
        } else {
            for (int i = 0; i < wall_end; i++) {
                int ax0, ay0, ax1, ay1;
                display_automap_map_pt(s_ix0[i], s_iy0[i], iw, ih, &ax0, &ay0);
                display_automap_map_pt(s_ix1[i], s_iy1[i], iw, ih, &ax1, &ay1);
                uint16_t seg = s_c12[i];
                Uint8 alpha = (seg & RENDERER_AUTOMAP_SEGFLAG_INTERNAL) ? (Uint8)128 : (Uint8)255;
                Uint8 fr, fg, fb;
                display_automap_amiga12_to_rgb(seg, &fr, &fg, &fb);
                display_automap_draw_line_outlined(g_sdl_ren, ax0, ay0, ax1, ay1, fr, fg, fb, alpha, 3, 1);
            }
        }
#endif
    {
        for (int i = 0; i < wall_end; i++) {
            int ax0, ay0, ax1, ay1;
            display_automap_map_pt(s_ix0[i], s_iy0[i], iw, ih, &ax0, &ay0);
            display_automap_map_pt(s_ix1[i], s_iy1[i], iw, ih, &ax1, &ay1);
            uint16_t seg = s_c12[i];
            Uint8 alpha = (seg & RENDERER_AUTOMAP_SEGFLAG_INTERNAL) ? (Uint8)128 : (Uint8)255;
            Uint8 fr, fg, fb;
            display_automap_amiga12_to_rgb(seg, &fr, &fg, &fb);
            display_automap_draw_line_outlined(g_sdl_ren, ax0, ay0, ax1, ay1, fr, fg, fb, alpha, 3, 1);
        }
    }
    }

    if (wall_end < n) {
        int axb, ayb, axt, ayt, axl, ayl, axr, ayr;
        display_automap_map_pt(s_ix0[n - 3], s_iy0[n - 3], iw, ih, &axb, &ayb);
        display_automap_map_pt(s_ix1[n - 3], s_iy1[n - 3], iw, ih, &axt, &ayt);
        display_automap_map_pt(s_ix1[n - 2], s_iy1[n - 2], iw, ih, &axl, &ayl);
        display_automap_map_pt(s_ix1[n - 1], s_iy1[n - 1], iw, ih, &axr, &ayr);
        display_automap_player_arrow_fill(axb, ayb, axl, ayl, axt, ayt, axr, ayr);

        for (int i = wall_end; i < n; i++) {
            int ax0, ay0, ax1, ay1;
            display_automap_map_pt(s_ix0[i], s_iy0[i], iw, ih, &ax0, &ay0);
            display_automap_map_pt(s_ix1[i], s_iy1[i], iw, ih, &ax1, &ay1);
            Uint8 fr = 255, fg = 252, fb = 235;
            display_automap_draw_line_outlined(g_sdl_ren, ax0, ay0, ax1, ay1, fr, fg, fb, 255, 4, 2);
        }
    }
}

/* -----------------------------------------------------------------------
 * Main rendering
 * ----------------------------------------------------------------------- */
#ifdef AB3D_DOS
static void display_automap_sdl_overlay(GameState *state)
{
    if(!state || !state->automap_visible || !g_sdl_ren)return;
    SDL_Rect previous;
    bool clipped=SDL_RenderClipEnabled(g_sdl_ren);
    if(!SDL_GetRenderClipRect(g_sdl_ren,&previous) ||
       !SDL_SetRenderClipRect(g_sdl_ren,&g_present_dst_rect)) {
        fprintf(stderr,"DOS map clip setup failed: %s\n",SDL_GetError());exit(1);
    }
    display_automap_sdl_overlay_impl(state);
    if(!SDL_SetRenderClipRect(g_sdl_ren,clipped?&previous:NULL)) {
        fprintf(stderr,"DOS map clip restore failed: %s\n",SDL_GetError());exit(1);
    }
}
#endif

static void display_present_cw_frame(GameState *state)
{
    if (!g_sdl_ren) return;

    const uint16_t *src = renderer_get_cw_buffer();
    if (!src) return;

    display_sync_present_output_size(NULL, NULL);

    int w = renderer_get_width(), h = renderer_get_height();

    int use_gl_weapon = state && state->cfg_weapon_post_gl && g_gl_unpack_ok && g_gl_hud_ok;
    int use_gl_water_tint = state && state->cfg_post_tint && g_gl_unpack_ok && g_gl_hud_ok;

    if (g_gl_unpack_ok) {
        display_gl_present_cw(src, w, h);
        if (g_gl_hud_ok) {
            display_gl_overlay_begin();
            /* Underwater multiply: cheap GL quad; avoid per-pixel CPU copy of whole buffer. */
            if (use_gl_water_tint)
                display_gl_multiply_tint_rect(renderer_get_last_fill_screen_water());
            if (use_gl_weapon) {
                if (state->cfg_weapon_draw) {
                    int frame_slot, ix, iy, iw, ih;
                    if (renderer_get_gun_draw_info(state, &frame_slot, &ix, &iy, &iw, &ih)) {
                        int rw = renderer_get_width(), rh = renderer_get_height();
                        SDL_Rect r = g_present_dst_rect;
                        /* Map top-left from (ix,iy) and size from (ix+iw,iy+ih) so bottom/right
                         * match the framebuffer edges after integer scale. Using iw*pw/rw and
                         * ih*ph/rh alone can lose 1px vs (exclusive end)*(dim)/dim - (start)*(dim)/dim
                         * when the window is smaller than internal res (gap under the weapon). */
                        int64_t px0 = (int64_t)r.x + (int64_t)ix * (int64_t)r.w / (int64_t)rw;
                        int64_t py0 = (int64_t)r.y + (int64_t)iy * (int64_t)r.h / (int64_t)rh;
                        int64_t px1 = (int64_t)r.x + (int64_t)(ix + iw) * (int64_t)r.w / (int64_t)rw;
                        int64_t py1 = (int64_t)r.y + (int64_t)(iy + ih) * (int64_t)r.h / (int64_t)rh;
                        SDL_Rect gun_dst;
                        gun_dst.x = (int)px0;
                        gun_dst.y = (int)py0;
                        gun_dst.w = (int)(px1 - px0);
                        gun_dst.h = (int)(py1 - py0);
                        if (gun_dst.w < 1) gun_dst.w = 1;
                        if (gun_dst.h < 1) gun_dst.h = 1;
                        display_gl_gun_quad(frame_slot, &gun_dst);
                    }
                }
            }
        } else
            display_sdl_resync_after_raw_gl();
    } else {
#ifdef AB3D_DOS
        SDL_SetRenderDrawColor(g_sdl_ren,0,0,0,255);
        SDL_RenderClear(g_sdl_ren);
        SDL_RenderFlush(g_sdl_ren);
        display_dos_copy_cw(src,w,h,&g_present_dst_rect);
#else
        if (!g_texture) return;
        display_cpu_unpack_cw_to_texture(src, w, h);
#if SDL_VERSION_ATLEAST(2, 0, 12)
        display_regenerate_framebuffer_mipmaps_if_downscaled(w, h);
#endif
        SDL_SetRenderDrawColor(g_sdl_ren, 0, 0, 0, 255);
        SDL_RenderClear(g_sdl_ren);
        SDL_RenderCopy(g_sdl_ren, g_texture, NULL, &g_present_dst_rect);
#endif
    }

#ifdef AB3D_DOS
    SDL_Rect dos_world_rect=g_present_dst_rect;
    g_present_dst_rect=(SDL_Rect){0,0,g_present_width,g_present_height};
#endif
    if (state) {
        display_hit_flash_overlay(state);
        display_hud_stats_sdl_overlay(state);
        display_key_hud_sdl_overlay(state);
        display_crosshair_sdl_overlay(state);
        if (state->automap_visible)
            display_automap_sdl_overlay(state);
        display_fps_overlay(state);
    }
    if (g_screen_tint_enabled && g_screen_tint_a > 0) {
        if (g_gl_unpack_ok && g_gl_hud_ok) {
            SDL_Rect tr = g_present_dst_rect;
            display_gl_solid_rect_fill(&tr, g_screen_tint_r, g_screen_tint_g, g_screen_tint_b,
                                       g_screen_tint_a);
        } else {
#ifdef AB3D_DOS
            SDL_SetRenderDrawBlendMode(g_sdl_ren, SDL_BLENDMODE_NONE);
#else
            SDL_SetRenderDrawBlendMode(g_sdl_ren, SDL_BLENDMODE_BLEND);
#endif
            SDL_SetRenderDrawColor(g_sdl_ren,
                                   g_screen_tint_r,
                                   g_screen_tint_g,
                                   g_screen_tint_b,
                                   g_screen_tint_a);
            SDL_RenderFillRect(g_sdl_ren, &g_present_dst_rect);
        }
    }
    display_text_screen_draw(255);
#ifdef AB3D_DOS
    g_present_dst_rect=dos_world_rect;
#endif

    if (g_gl_unpack_ok && g_gl_hud_ok)
        display_gl_overlay_end();

    if (g_suppress_next_present_count > 0) {
        g_suppress_next_present_count--;
        return;
    }
#ifdef AB3D_DOS
    if(dos_capture_next_frame) {
        dos_capture_next_frame=0;
        dos_capture_frame("FRAME.BMP");
    }
#endif
    SDL_RenderPresent(g_sdl_ren);

    /* Debug: show player position in window title (throttled) */
    if (g_debug_window_title && state && g_window) {
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

static void display_draw_display_impl(GameState *state)
{
    /* Tell renderer to skip CPU work when the GL overlay path will handle it. */
    int use_gl_weapon = state && state->cfg_weapon_post_gl && g_gl_unpack_ok && g_gl_hud_ok;
    int use_gl_water_tint = state && state->cfg_post_tint && g_gl_unpack_ok && g_gl_hud_ok;
    renderer_set_weapon_post_gl_active(use_gl_weapon);
    renderer_set_gl_water_tint_post_active(use_gl_water_tint);
    renderer_draw_display(state);
#ifdef AB3D_DOS
    audio_dos_pump();
#endif
    display_present_cw_frame(state);
}

void display_present_last_frame(GameState *state)
{
    display_present_cw_frame(state);
}

void display_suppress_next_present(void)
{
    if (g_suppress_next_present_count < 16)
        g_suppress_next_present_count++;
}

void display_present_text_screen(void)
{
    display_present_text_screen_alpha(255);
}

void display_present_text_screen_alpha(int alpha)
{
    if (!g_sdl_ren) return;
    if (alpha < 0) alpha = 0;
    if (alpha > 255) alpha = 255;

    int out_w = 0;
    int out_h = 0;
    if (SDL_GetRendererOutputSize(g_sdl_ren, &out_w, &out_h) != 0) {
        out_w = g_present_width;
        out_h = g_present_height;
    }
    if (out_w < 1) out_w = 1;
    if (out_h < 1) out_h = 1;
    if (out_w != g_present_width || out_h != g_present_height) {
        display_update_letterbox(out_w, out_h);
        g_present_width = out_w;
        g_present_height = out_h;
    }

    if (g_gl_unpack_ok && g_gl_hud_ok) {
        display_gl_output_size(&g_gl_overlay_win_w, &g_gl_overlay_win_h);
        g_gl_viewport(0, 0, g_gl_overlay_win_w, g_gl_overlay_win_h);
        g_gl_clear_color(0.0f, 0.0f, 0.0f, 1.0f);
        g_gl_clear(GL_COLOR_BUFFER_BIT);
        display_gl_overlay_begin();
        display_text_screen_draw((Uint8)alpha);
        display_gl_overlay_end();
    } else {
        SDL_SetRenderDrawBlendMode(g_sdl_ren, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(g_sdl_ren, 0, 0, 0, 255);
        SDL_RenderClear(g_sdl_ren);
        display_text_screen_draw((Uint8)alpha);
    }

    SDL_RenderPresent(g_sdl_ren);
}

void display_present_title_screen_alpha(int text_alpha, int dim_alpha)
{
    if (!g_sdl_ren) return;
    if (text_alpha < 0) text_alpha = 0;
    if (text_alpha > 255) text_alpha = 255;
    if (dim_alpha < 0) dim_alpha = 0;
    if (dim_alpha > 255) dim_alpha = 255;
#ifdef AB3D_DOS
    dim_alpha = 0;
#endif

    display_title_screen_ensure_loaded();
    if (!g_title_screen_tex && !g_title_screen_gl_tex) {
        display_present_text_screen_alpha(text_alpha);
        return;
    }

    int out_w = 0;
    int out_h = 0;
    if (SDL_GetRendererOutputSize(g_sdl_ren, &out_w, &out_h) != 0) {
        out_w = g_present_width;
        out_h = g_present_height;
    }
    if (out_w < 1) out_w = 1;
    if (out_h < 1) out_h = 1;
    if (out_w != g_present_width || out_h != g_present_height) {
        display_update_letterbox(out_w, out_h);
        g_present_width = out_w;
        g_present_height = out_h;
    }

    SDL_Rect title_dst = display_fit_rect(out_w, out_h,
                                          DISPLAY_TITLE_W,
                                          DISPLAY_TITLE_H);

    if (g_gl_unpack_ok && g_gl_hud_ok) {
        display_gl_output_size(&g_gl_overlay_win_w, &g_gl_overlay_win_h);
        g_gl_viewport(0, 0, g_gl_overlay_win_w, g_gl_overlay_win_h);
        g_gl_clear_color(0.0f, 0.0f, 0.0f, 1.0f);
        g_gl_clear(GL_COLOR_BUFFER_BIT);
        title_dst = display_fit_rect(g_gl_overlay_win_w, g_gl_overlay_win_h,
                                     DISPLAY_TITLE_W, DISPLAY_TITLE_H);
        display_gl_overlay_begin();
        display_gl_texture_id_blit(g_title_screen_gl_tex, &title_dst, 255);
        if (dim_alpha > 0)
            display_overlay_fill_rect_abs(&title_dst, 0, 0, 0, (Uint8)dim_alpha);
        display_text_screen_draw_in_rect((Uint8)text_alpha, title_dst);
        display_gl_overlay_end();
    } else {
        SDL_SetRenderDrawBlendMode(g_sdl_ren, SDL_BLENDMODE_NONE);
        SDL_SetRenderDrawColor(g_sdl_ren, 0, 0, 0, 255);
        SDL_RenderClear(g_sdl_ren);
        display_overlay_copy(g_title_screen_tex, NULL, &title_dst);
        if (dim_alpha > 0)
            display_overlay_fill_rect_abs(&title_dst, 0, 0, 0, (Uint8)dim_alpha);
        display_text_screen_draw_in_rect((Uint8)text_alpha, title_dst);
    }

    SDL_RenderPresent(g_sdl_ren);
}

void display_swap_buffers(void)
{
    /* Handled in display_draw_display */
}

void display_wait_vblank(void)
{
    /* VSync is handled by SDL_RENDERER_PRESENTVSYNC */
}

void display_set_screen_tint(int r, int g, int b, int alpha)
{
    if (r < 0) r = 0; else if (r > 255) r = 255;
    if (g < 0) g = 0; else if (g > 255) g = 255;
    if (b < 0) b = 0; else if (b > 255) b = 255;
    if (alpha < 0) alpha = 0; else if (alpha > 255) alpha = 255;
#ifdef AB3D_DOS
    if (alpha) alpha = 255;
#endif

    g_screen_tint_r = (Uint8)r;
    g_screen_tint_g = (Uint8)g;
    g_screen_tint_b = (Uint8)b;
    g_screen_tint_a = (Uint8)alpha;
    g_screen_tint_enabled = (alpha > 0) ? 1 : 0;
}

void display_clear_screen_tint(void)
{
    g_screen_tint_enabled = 0;
    g_screen_tint_r = 0;
    g_screen_tint_g = 0;
    g_screen_tint_b = 0;
    g_screen_tint_a = 0;
}

/* -----------------------------------------------------------------------
 * HUD
 * ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
 * Text (minimal for now)
 * ----------------------------------------------------------------------- */
void display_draw_line_of_text(const char *text, int line)
{
    display_draw_line_of_text_alpha(text, line, 255);
}

void display_draw_line_of_text_alpha(const char *text, int line, int alpha)
{
    if (line < 0 || line >= DISPLAY_TEXT_MAX_LINES) return;
    if (!text) text = "";
    if (alpha < 0) alpha = 0;
    if (alpha > 255) alpha = 255;
    snprintf(g_text_lines[line], sizeof(g_text_lines[line]), "%s", text);
    g_text_line_used[line] = g_text_lines[line][0] ? 1u : 0u;
    g_text_line_alpha[line] = (uint8_t)alpha;
}

void display_clear_text_screen(void)
{
    memset(g_text_lines, 0, sizeof(g_text_lines));
    memset(g_text_line_used, 0, sizeof(g_text_line_used));
    memset(g_text_line_alpha, 255, sizeof(g_text_line_alpha));
}

#ifdef AB3D_DOS
void display_dos_check_text_screen(const char *label, int allow_pages)
{
    if(!display_text_has_lines())return;
    int saved_page=g_dos_text_page,saved_pages=g_dos_text_pages,saved_rows=g_dos_text_page_rows;
    unsigned expected=0;
    for(int row=0;row<DISPLAY_TEXT_MAX_LINES;row++)
        for(const char *p=g_text_lines[row];*p;p++)if(*p!=' ' && *p!='\t')expected++;
    for(int height=200;height<=240;height+=40) {
        SDL_Rect r={0,0,320,height};
        unsigned actual=0;
        g_dos_text_page=0;
        do {
            DisplayTextRun runs[DISPLAY_TEXT_LAYOUT_MAX_RUNS];
            int count,scale,line_h,draw_h,top,mx,my;
            if(!display_text_layout_in_rect(r,runs,DISPLAY_TEXT_LAYOUT_MAX_RUNS,
                    &count,&scale,&line_h,&draw_h,&top,&mx,&my) ||
               scale!=DISPLAY_ASCII_SCALE_ONE || (!allow_pages && g_dos_text_pages!=1)) {
                fprintf(stderr,"DOS %s does not fit native 320x%d\n",label,height);exit(1);
            }
            int first=g_dos_text_page*g_dos_text_page_rows;
            int end=first+g_dos_text_page_rows;if(end>count)end=count;
            int bottom=g_dos_text_pages>1?height-my-DISPLAY_ASCII_DRAW_H:height;
            for(int i=first;i<end;i++) {
                int w=runs[i].len*DISPLAY_ASCII_XADVANCE;
                int y=top+(i-first)*line_h;
                if(w>320-2*mx || y<0 || y+draw_h>bottom) {
                    fprintf(stderr,"DOS %s row %d clipped/overlaps prompt at 320x%d\n",
                        label,i,height);exit(1);
                }
                for(int j=0;j<runs[i].len;j++) {
                    char c=runs[i].text[runs[i].start+j];if(c!=' ' && c!='\t')actual++;
                }
            }
            g_dos_text_page++;
        } while(g_dos_text_page<g_dos_text_pages);
        if(actual!=expected) {
            fprintf(stderr,"DOS %s lost glyphs at 320x%d\n",label,height);exit(1);
        }
    }
    g_dos_text_page=saved_page;g_dos_text_pages=saved_pages;g_dos_text_page_rows=saved_rows;
}

void display_dos_check_hud_bounds(GameState *state)
{
    GameState *test=malloc(sizeof(*test));
    if(!test){fprintf(stderr,"DOS HUD diagnostic allocation failed\n");exit(1);}
    memcpy(test,state,sizeof(*test));
    int saved_w=g_present_width,saved_h=g_present_height;
    SDL_Rect saved_rect=g_present_dst_rect;
    int16_t conditions=game_conditions;game_conditions|=15;
    test->cfg_show_fps=1;test->fps_display=9999;test->cfg_crosshair_colour=7;
    test->automap_visible=true;
    test->infinite_ammo=false;test->infinite_health=false;
    test->hitcol=test->hitcol2=0xF00;
    for(int height=200;height<=240 && height<=saved_h;height+=40) {
        g_present_width=320;g_present_height=height;
        for(int bars=0;bars<2;bars++)for(int value=0;value<3;value++) {
            test->cfg_hud_bars=bars;
            test->energy=test->plr1.energy=test->plr2.energy=value==2?PLAYER_MAX_ENERGY:value;
            test->ammo=value==2?MAX_AMMO_RAW:value;
            SDL_SetRenderDrawColor(g_sdl_ren,0,0,0,255);SDL_RenderClear(g_sdl_ren);
            SDL_RenderFlush(g_sdl_ren);
            int w=renderer_get_width(),h=renderer_get_height();
            g_present_dst_rect=(SDL_Rect){(320-w)/2,(height-h)/2,w,h};
            if(w<=320 && h<=height)display_dos_copy_cw(renderer_get_cw_buffer(),w,h,&g_present_dst_rect);
            g_present_dst_rect=(SDL_Rect){0,0,320,height};
            display_hit_flash_overlay(test);
            display_hud_stats_sdl_overlay(test);
            display_key_hud_sdl_overlay(test);
            display_crosshair_sdl_overlay(test);
            display_fps_overlay(test);
            display_automap_sdl_overlay(test);
            SDL_RenderPresent(g_sdl_ren);
            if(value==2) {
                char filename[13];snprintf(filename,sizeof(filename),"%s%d.BMP",bars?"BAR":"HUD",height);
                dos_capture_frame(filename);
            }
        }
    }
    game_conditions=conditions;
    g_present_width=saved_w;g_present_height=saved_h;g_present_dst_rect=saved_rect;
    free(test);
    printf("[DOSCHECK] HUD bars/numbers, four keys/underlines, 9999 FPS, crosshair and map fit 320x200/240\n");
}

void display_dos_check_text_layouts(void)
{
    SDL_Rect r={0,0,g_present_width,g_present_height};
    for(int lev=0;lev<MAX_LEVELS;lev++) {
        display_clear_text_screen();display_text_reset_pages();
        display_draw_line_of_text(level_text_heading[lev],0);
        for(int row=0;row<LEVEL_TEXT_VISIBLE_ROWS;row++)
            display_draw_line_of_text(level_text[lev][row],row+2);
        display_dos_check_text_screen(level_text_heading[lev],1);
        DisplayTextRun runs[DISPLAY_TEXT_LAYOUT_MAX_RUNS];
        int count,scale,line_h,draw_h,top,mx,my;
        if(!display_text_layout_in_rect(r,runs,DISPLAY_TEXT_LAYOUT_MAX_RUNS,
                &count,&scale,&line_h,&draw_h,&top,&mx,&my) || scale!=DISPLAY_ASCII_SCALE_ONE) {
            fprintf(stderr,"DOS level %d native font layout failed\n",lev+1);exit(1);
        }
        unsigned expected=0,actual=0;
        for(int row=0;row<DISPLAY_TEXT_MAX_LINES;row++)
            for(const char *p=g_text_lines[row];*p;p++)if(*p!=' ' && *p!='\t')expected++;
        for(int i=0;i<count;i++) {
            if(runs[i].len*DISPLAY_ASCII_XADVANCE>r.w-2*mx) {
                fprintf(stderr,"DOS level %d text exceeds screen width\n",lev+1);exit(1);
            }
            for(int j=0;j<runs[i].len;j++) {
                char c=runs[i].text[runs[i].start+j];if(c!=' ' && c!='\t')actual++;
            }
        }
        if(actual!=expected){fprintf(stderr,"DOS level %d text lost glyphs\n",lev+1);exit(1);}
        if(lev==0) {
            display_present_text_screen_alpha(255);dos_capture_frame("INTRO1.BMP");
            while(display_text_next_page())display_present_text_screen_alpha(255);
            dos_capture_frame("INTRO2.BMP");
        }
    }
    display_clear_text_screen();display_text_reset_pages();
    printf("[DOSCHECK] all 16 level texts fit native font pages without lost glyphs\n");
}
#endif

void display_draw_display(GameState *state) {
#ifdef AB3D_DOS
    dos_profile_begin(DOS_PRESENT);
    audio_dos_pump();
#endif
    display_draw_display_impl(state);
#ifdef AB3D_DOS
    audio_dos_pump();
    dos_profile_end();
#endif
}

#ifdef AB3D_DOS
void dos_capture_frame(const char *filename) {
    /* DOS drawing is synchronous; stream the resident native surface directly. */
    SDL_Surface *surface=g_sdl_ren?g_sdl_ren->surface:NULL;
    if(!surface) { fprintf(stderr,"DOS frame readback failed: %s\n",SDL_GetError());exit(1); }
    const SDL_PixelFormatDetails *fmt=SDL_GetPixelFormatDetails(surface->format);
    unsigned stride=(surface->w*3+3)&~3u,size=54+stride*surface->h;
    unsigned char header[54]={0};header[0]='B';header[1]='M';header[10]=54;header[14]=40;header[26]=1;header[28]=24;
    for(int i=0;i<4;i++){header[2+i]=(unsigned char)(size>>(8*i));header[18+i]=(unsigned char)(surface->w>>(8*i));header[22+i]=(unsigned char)(surface->h>>(8*i));}
    FILE *f=fopen(filename,"wb");unsigned char *row=calloc(1,stride);
    if(!f || !row || fwrite(header,1,54,f)!=54){fprintf(stderr,"DOS frame file allocation/write failed\n");exit(1);}
    for(int y=surface->h-1;y>=0;y--) {
        for(int x=0;x<surface->w;x++) {
            Uint32 pixel=0;Uint8 r,g,b,a;
            memcpy(&pixel,(Uint8 *)surface->pixels+y*surface->pitch+x*fmt->bytes_per_pixel,fmt->bytes_per_pixel);
            SDL_GetRGBA(pixel,fmt,SDL_GetSurfacePalette(surface),&r,&g,&b,&a);
            row[x*3]=b;row[x*3+1]=g;row[x*3+2]=r;
        }
        if(fwrite(row,1,stride,f)!=stride){fprintf(stderr,"DOS frame write failed\n");exit(1);}
    }
    fclose(f);free(row);
}
#endif
