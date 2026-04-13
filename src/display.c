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
#include "renderer.h"
#include "game_data.h"
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
EM_JS(void, display_emscripten_install_fullscreen_listeners, (void), {
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
    var mark = function() { Module['ab3dFullscreenResizePending'] = 1; };
    document.addEventListener('fullscreenchange', mark, false);
    document.addEventListener('webkitfullscreenchange', mark, false);
    window.addEventListener('keydown', function(e) {
        if (e.code !== 'F11' && e.key !== 'F11' && e.keyCode !== 122) return;
        e.preventDefault();
        e.stopPropagation();
        ab3dToggleCanvasFullscreen();
    }, true);
});

EM_JS(void, display_emscripten_canvas_fullscreen_toggle, (void), {
    if (Module['ab3dToggleCanvasFullscreen']) Module['ab3dToggleCanvasFullscreen']();
});

/* Pick render_width/height from the nearest common aspect to screen.width/screen.height. */
EM_JS(void, display_emscripten_apply_screen_aspect_resolution, (int *out_w, int *out_h), {
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
    var h32 = (typeof HEAP32 !== 'undefined') ? HEAP32 : Module['HEAP32'];
    h32[out_w >> 2] = tw;
    h32[out_h >> 2] = th;
});
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
/* True when g_texture accepts native packed 0x0RGB words directly (XRGB4444/ARGB4444). */
static int g_texture_is_4444_direct = 0;
/* Key HUD: one SDL texture per key sprite frame 0..3 (invalidated when level/assets change). */
static SDL_Texture *g_key_hud_tex[4];
static uintptr_t    g_key_hud_tex_tag;
static void display_key_hud_free_textures(void);

/* HUD digit strips: 0 = health %, 1 = ammo count (fonts, e.g. health_digits.png). */
static SDL_Texture *g_hud_digit_tex[2];
static int          g_hud_digit_tex_w[2];
static int          g_hud_digit_tex_h[2];
static int          g_hud_digits_load_attempted;
static void         display_hud_digits_free(void);
static void         display_hud_digits_ensure_loaded(void);
static int g_gl_unpack_ok; /* OpenGL R16UI + shader path active */
static int g_present_width = 0;
static int g_present_height = 0;
static SDL_Rect g_present_dst_rect;
static int g_internal_w = RENDER_WIDTH;
static int g_internal_h = RENDER_HEIGHT;
static int g_release_borderless_desktop = 0;
static int g_screen_tint_enabled = 0;
static Uint8 g_screen_tint_r = 0;
static Uint8 g_screen_tint_g = 0;
static Uint8 g_screen_tint_b = 0;
static Uint8 g_screen_tint_a = 0;

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

/* HUD overlays: SDL_Renderer GL backend does not reliably mix with raw glUseProgram; draw HUD with
 * SDL_GL_BindTexture + our own GLSL (same context as the 12-bit present shader). */
static int g_gl_hud_ok;
static GLuint g_gl_prog_hud_tex;
static GLuint g_gl_prog_hud_solid;
static GLuint g_gl_hud_vao_tex;
static GLuint g_gl_hud_vao_solid;
static GLuint g_gl_hud_vbo;
static GLint g_gl_hud_loc_tex;
static GLint g_gl_hud_loc_color_tex;
static GLint g_gl_hud_loc_rb_swap;
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
    "in highp vec2 v_uv;\n"
    "out highp vec4 o_col;\n"
    "void main() {\n"
    "  highp vec2 uv = v_uv;\n"
    "  highp uint w = texture(u_cw, uv).r;\n"
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
    "in vec2 v_uv;\n"
    "out vec4 o_col;\n"
    "void main() {\n"
    "  /* Top-down CPU rows match default GL unpack (row 0 -> t=0); do not flip v_uv. */\n"
    "  vec2 uv = v_uv;\n"
    "  uint w = texture(u_cw, uv).r;\n"
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
    if (g_gl_delete_program && g_gl_prog_hud_solid) g_gl_delete_program(g_gl_prog_hud_solid);
    g_gl_prog_hud_tex = 0;
    g_gl_prog_hud_solid = 0;
    g_gl_hud_ok = 0;
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
    printf("[DISPLAY] HUD: GL overlay (SDL_GL_BindTexture + GLSL)\n");
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
    if (g_gl_unpack_ok && g_gl_hud_ok)
        display_gl_texture_blit(tex, src_opt, dst);
    else
        SDL_RenderCopy(g_sdl_ren, tex, src_opt, dst);
}

static void display_overlay_fill_rect_abs(const SDL_Rect *rect, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    if (!g_sdl_ren || !rect) return;
    if (g_gl_unpack_ok && g_gl_hud_ok)
        display_gl_solid_rect_fill(rect, r, g, b, a);
    else {
        SDL_SetRenderDrawBlendMode(g_sdl_ren, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(g_sdl_ren, r, g, b, a);
        SDL_RenderFillRect(g_sdl_ren, rect);
    }
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
    g_gl_unpack_ok = 0;
}

static int display_gl_try_init_unpack(void)
{
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
    if (g_texture_is_4444_direct) {
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
        const uint16_t *srow = src + (size_t)y * (size_t)w;
        for (int x = 0; x < w; x++) {
            uint32_t c = (uint32_t)(srow[x] & 0xFFFu);
            uint32_t r4 = (c >> 8) & 0xFu;
            uint32_t g4 = (c >> 4) & 0xFu;
            uint32_t b4 = c & 0xFu;
            dst_row[x] = 0xFF000000u | (r4 * 0x11u << 16) | (g4 * 0x11u << 8) | (b4 * 0x11u);
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

#if SDL_VERSION_ATLEAST(2, 0, 12)
    /* OpenGL render path: nearest when magnifying the software framebuffer; linear when
     * minifying (pairs with glGenerateMipmap + trilinear in display_regenerate_*). */
    if (g_texture && !g_gl_unpack_ok) {
        SDL_SetTextureScaleMode(g_texture,
            (sc > 1.0) ? SDL_ScaleModeNearest : SDL_ScaleModeLinear);
    }
#endif
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
        g_texture = SDL_CreateTexture(g_sdl_ren,
            SDL_PIXELFORMAT_XRGB4444, SDL_TEXTUREACCESS_STREAMING, g_internal_w, g_internal_h);
        if (g_texture) {
            g_texture_is_4444_direct = 1;
        } else {
            g_texture = SDL_CreateTexture(g_sdl_ren,
                SDL_PIXELFORMAT_ARGB4444, SDL_TEXTUREACCESS_STREAMING, g_internal_w, g_internal_h);
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
               cur.w, cur.h, cur.refresh_rate,
               desk.w, desk.h, desk.refresh_rate);
        return;
    }
    if (cur_ok) {
        printf("[DISPLAY] %s: current=%dx%d@%d desktop=unavailable\n",
               tag, cur.w, cur.h, cur.refresh_rate);
        return;
    }
    if (desk_ok) {
        printf("[DISPLAY] %s: current=unavailable desktop=%dx%d@%d\n",
               tag, desk.w, desk.h, desk.refresh_rate);
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
    if (base_rw < 96) base_rw = 96;
    if (base_rh < 80) base_rh = 80;
    if (supersampling < 1) supersampling = 1;
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

    renderer_init();

    const char *driver_override = SDL_getenv("AB3D_RENDER_DRIVER");
    /* Set AB3D_DISABLE_GL_UNPACK=1 to force D3D + SDL texture (CPU blit path). Otherwise prefer GL_R16UI. */
    const char *disable_gl_unpack = SDL_getenv("AB3D_DISABLE_GL_UNPACK");
    int prefer_gpu_unpack = 1;
    if (disable_gl_unpack && disable_gl_unpack[0] != '\0') {
        if (disable_gl_unpack[0] == '1' || disable_gl_unpack[0] == 'y' || disable_gl_unpack[0] == 'Y')
            prefer_gpu_unpack = 0;
    }
    int window_w = base_rw;
    int window_h = base_rh;
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
            window_flags |= SDL_WINDOW_OPENGL;
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
    if (!g_sdl_ren) return;
    int pending = EM_ASM_INT({
        var v = Module['ab3dFullscreenResizePending'] | 0;
        Module['ab3dFullscreenResizePending'] = 0;
        return v;
    });
    if (pending) {
        SDL_PumpEvents();
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
    display_hud_digits_free();
    display_key_hud_free_textures();
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

#define DISPLAY_AUTOMAP_MAX_SEGS 4096

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
    g_key_hud_tex[frame_idx] = t;
    return t;
}

static void display_hud_digits_free(void)
{
    for (int i = 0; i < 2; i++) {
        if (g_hud_digit_tex[i]) {
            SDL_DestroyTexture(g_hud_digit_tex[i]);
            g_hud_digit_tex[i] = NULL;
        }
        g_hud_digit_tex_w[i] = 0;
        g_hud_digit_tex_h[i] = 0;
    }
    g_hud_digits_load_attempted = 0;
}

static void display_hud_digits_ensure_loaded(void)
{
    if (g_hud_digits_load_attempted || !g_sdl_ren) return;
    g_hud_digits_load_attempted = 1;

    const char *names[2] = { "health_digits.png", "ammo_digits.png" };
    char *base = SDL_GetBasePath();
    char path[512];

    for (int i = 0; i < 2; i++) {
        int w = 0, h = 0, comp = 0;
        unsigned char *rgba = NULL;

        if (base) {
            int plen = snprintf(path, sizeof(path), "%sfonts/%s", base, names[i]);
            if (plen > 0 && plen < (int)sizeof(path))
                rgba = stbi_load(path, &w, &h, &comp, 4);
        }
        if (!rgba) {
            int plen = snprintf(path, sizeof(path), "fonts/%s", names[i]);
            if (plen > 0 && plen < (int)sizeof(path))
                rgba = stbi_load(path, &w, &h, &comp, 4);
        }
        if (!rgba) {
            printf("[DISPLAY] HUD digits: could not load %s (%s)\n", names[i],
                   stbi_failure_reason() ? stbi_failure_reason() : "?");
            continue;
        }

        size_t npix = (size_t)w * (size_t)h;
        uint32_t *argb = (uint32_t *)malloc(npix * sizeof(uint32_t));
        if (!argb) {
            stbi_image_free(rgba);
            continue;
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
            printf("[DISPLAY] HUD digits: SDL texture failed for %s\n", names[i]);
            continue;
        }
        free(argb);
        SDL_SetTextureBlendMode(t, SDL_BLENDMODE_BLEND);
#if SDL_VERSION_ATLEAST(2, 0, 12)
        /* Nearest: linear upscale bleeds between adjacent digits in the strip atlas. */
        SDL_SetTextureScaleMode(t, SDL_ScaleModeNearest);
#endif
        g_hud_digit_tex[i] = t;
        g_hud_digit_tex_w[i] = w;
        g_hud_digit_tex_h[i] = h;
    }

    if (base)
        SDL_free(base);
}

/*
 * Bottom HUD: compact cluster bottom-right. Letterbox pixels (g_present_dst_rect);
 * scale ~1/32 of height so size tracks resolution without dominating the screen.
 */
static void hud_key_row_layout(int lay_w, int lay_h, int *margin, int *kh, int *gap, int *group_w,
                               int *ix_key0, int *iy)
{
    int m = lay_h / 64;
    if (m < 2) m = 2;
    int k = lay_h / 32;
    if (k < 11) k = 11;
    int g = k / 12;
    if (g < 2) g = 2;
    int gw = 4 * k + 3 * g;
    int ix = lay_w - m - gw;
    if (ix < 0) ix = 0;
    int y = lay_h - m - k;
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
    if (tex_h < 11 || tex_w < 108) return 0;
    if (tex_h % 11 != 0) return 0;
    int k = tex_h / 11;
    if (k < 1 || tex_w != 108 * k) return 0;
    return k;
}

static void hud_digit_src_rect(int tex_w, int tex_h, int digit, SDL_Rect *src)
{
    if (digit < 0) digit = 0;
    if (digit > 9) digit = 9;

    int k = hud_digit_tex_scale_k(tex_w, tex_h);
    if (k > 0) {
        int stride = 11 * k;
        int cell_w = 9 * k;
        int row_h = 11 * k;
        src->x = digit * stride;
        src->y = 0;
        src->w = cell_w;
        src->h = row_h;
        return;
    }

    /* Fallback: generic 10-column atlas */
    int cell = tex_w / 10;
    if (cell < 1) cell = 1;
    src->x = digit * cell;
    src->y = 0;
    src->w = (digit == 9) ? (tex_w - digit * cell) : cell;
    src->h = tex_h;
}

/* Widest scaled glyph — fixed column width so 0..9 align in three slots. */
static int hud_max_digit_scaled_width(int tex_w, int tex_h, int digit_h)
{
    int mw = 0;
    for (int d = 0; d <= 9; d++) {
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
static void hud_draw_three_slot_value(SDL_Texture *tex, int tex_w, int tex_h, int digit_h,
                                      int value, int max_value, int x_left, int y_top, int gap)
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
        display_overlay_copy(tex, &src, &dst);
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
        display_overlay_copy(tex, &src, &dst);
    }
}

static void display_hud_stats_sdl_overlay(const GameState *state)
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

    /* Raw ammo count (same scale as display_ammo_bar); cap HUD at MAX_AMMO_RAW. */
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

    int digit_h = kh;
    int health_w = 0, ammo_w = 0;
    if (g_hud_digit_tex[0])
        health_w = hud_three_slot_width(g_hud_digit_tex_w[0], g_hud_digit_tex_h[0], digit_h, d_gap);
    if (g_hud_digit_tex[1])
        ammo_w = hud_three_slot_width(g_hud_digit_tex_w[1], g_hud_digit_tex_h[1], digit_h, d_gap);

    /* Shrink digits if stats would extend left of the margin (health | ammo | gap | keys). */
    for (int attempt = 0; attempt < 10; attempt++) {
        int inner = (health_w > 0 && ammo_w > 0) ? gap_stat : 0;
        int stats_total = health_w + inner + ammo_w;
        int leftmost = ix_key0 - gap_stat - stats_total;
        if (leftmost >= margin) break;
        digit_h = digit_h * 9 / 10;
        if (digit_h < 6) break;
        health_w = g_hud_digit_tex[0] ? hud_three_slot_width(g_hud_digit_tex_w[0], g_hud_digit_tex_h[0], digit_h, d_gap) : 0;
        ammo_w = g_hud_digit_tex[1] ? hud_three_slot_width(g_hud_digit_tex_w[1], g_hud_digit_tex_h[1], digit_h, d_gap) : 0;
    }

    /* Right edge of stat block sits just left of the key row (same baseline iy). */
    int right = ix_key0 - gap_stat;
    if (g_hud_digit_tex[1] && ammo_w > 0) {
        int ammo_x = right - ammo_w;
        SDL_SetTextureAlphaMod(g_hud_digit_tex[1], 255);
        hud_draw_three_slot_value(g_hud_digit_tex[1], g_hud_digit_tex_w[1], g_hud_digit_tex_h[1],
                                  digit_h, ammo_count, (int)MAX_AMMO_RAW, ammo_x, iy, d_gap);
        right = ammo_x - gap_stat;
    }
    if (g_hud_digit_tex[0] && health_w > 0) {
        int health_x = right - health_w;
        SDL_SetTextureAlphaMod(g_hud_digit_tex[0], 255);
        hud_draw_three_slot_value(g_hud_digit_tex[0], g_hud_digit_tex_w[0], g_hud_digit_tex_h[0],
                                  digit_h, hp_pct, 100, health_x, iy, d_gap);
    }
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
        Uint8 alpha = collected ? (Uint8)255 : (Uint8)128;

        int frame = renderer_key_sprite_frame_for_condition_bit(state, i);
        if (frame < 0) frame = 0;
        if (frame > 3) frame = 3;

        int ix = ix0 + i * (kh + gap);

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

static void display_automap_sdl_overlay(GameState *state)
{
    if (!state || !state->automap_visible || !g_sdl_ren) return;

    /* ~264 KiB: must not live on the stack — Emscripten default stack is 64 KiB and overflow
     * here caused a solid tab freeze when opening the map on web. */
    static int s_ix0[DISPLAY_AUTOMAP_MAX_SEGS];
    static int s_iy0[DISPLAY_AUTOMAP_MAX_SEGS];
    static int s_ix1[DISPLAY_AUTOMAP_MAX_SEGS];
    static int s_iy1[DISPLAY_AUTOMAP_MAX_SEGS];
    static uint16_t s_c12[DISPLAY_AUTOMAP_MAX_SEGS];

    int n = renderer_automap_collect_line_segments(state, s_ix0, s_iy0, s_ix1, s_iy1, s_c12,
                                                   DISPLAY_AUTOMAP_MAX_SEGS);
    if (n <= 0) return;

#if SDL_VERSION_ATLEAST(2, 0, 14)
    if (g_gl_unpack_ok && g_gl_hud_ok)
        SDL_RenderFlush(g_sdl_ren);
#endif

    int iw = renderer_get_width();
    int ih = renderer_get_height();

    int wall_end = n;
    if (n >= 3 && (s_c12[n - 3] & RENDERER_AUTOMAP_SEGFLAG_PLAYER))
        wall_end = n - 3;

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
static void display_present_cw_frame(GameState *state)
{
    if (!g_sdl_ren) return;

    const uint16_t *src = renderer_get_cw_buffer();
    if (!src) return;

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
        if (!g_texture) return;
        display_cpu_unpack_cw_to_texture(src, w, h);
#if SDL_VERSION_ATLEAST(2, 0, 12)
        display_regenerate_framebuffer_mipmaps_if_downscaled(w, h);
#endif
        SDL_SetRenderDrawColor(g_sdl_ren, 0, 0, 0, 255);
        SDL_RenderClear(g_sdl_ren);
        SDL_RenderCopy(g_sdl_ren, g_texture, NULL, &g_present_dst_rect);
    }

    if (state) {
        display_hud_stats_sdl_overlay(state);
        display_key_hud_sdl_overlay(state);
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
            SDL_SetRenderDrawBlendMode(g_sdl_ren, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(g_sdl_ren,
                                   g_screen_tint_r,
                                   g_screen_tint_g,
                                   g_screen_tint_b,
                                   g_screen_tint_a);
            SDL_RenderFillRect(g_sdl_ren, &g_present_dst_rect);
        }
    }

    if (g_gl_unpack_ok && g_gl_hud_ok)
        display_gl_overlay_end();

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

void display_draw_display(GameState *state)
{
    /* Tell renderer to skip CPU work when the GL overlay path will handle it. */
    int use_gl_weapon = state && state->cfg_weapon_post_gl && g_gl_unpack_ok && g_gl_hud_ok;
    int use_gl_water_tint = state && state->cfg_post_tint && g_gl_unpack_ok && g_gl_hud_ok;
    renderer_set_weapon_post_gl_active(use_gl_weapon);
    renderer_set_gl_water_tint_post_active(use_gl_water_tint);
    renderer_draw_display(state);
    display_present_cw_frame(state);
}

void display_present_last_frame(GameState *state)
{
    display_present_cw_frame(state);
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
void display_energy_bar(int16_t energy)
{
    int w = g_renderer.width, h = g_renderer.height;
    int bar_y = h - 2;
    int bar_w = (energy > 0) ? ((int)energy * (w - 4) / 127) : 0;
    const uint32_t px = RENDER_RGB_RASTER_PIXEL(0x00CC00u);
    if (renderer_get_rgb_raster_expand()) {
        uint32_t *rgb = renderer_get_active_rgb_target();
        if (!rgb) return;
        for (int x = 2; x < 2 + bar_w && x < w - 2; x++) {
            rgb[bar_y * w + x] = px;
            rgb[(bar_y - 1) * w + x] = px;
        }
    } else {
        uint16_t *cw = renderer_get_active_cw_target();
        if (!cw) return;
        uint16_t c = renderer_argb_to_amiga12(px);
        for (int x = 2; x < 2 + bar_w && x < w - 2; x++) {
            cw[bar_y * w + x] = c;
            cw[(bar_y - 1) * w + x] = c;
        }
    }
}

void display_ammo_bar(int16_t ammo)
{
    int w = g_renderer.width, h = g_renderer.height;
    int bar_y = h - 5;
    int max_ammo = 999;
    int bar_w = (ammo > 0) ? ((int)ammo * (w - 4) / max_ammo) : 0;
    if (bar_w > w - 4) bar_w = w - 4;
    const uint32_t px = RENDER_RGB_RASTER_PIXEL(0xCCCC00u);
    if (renderer_get_rgb_raster_expand()) {
        uint32_t *rgb = renderer_get_active_rgb_target();
        if (!rgb) return;
        for (int x = 2; x < 2 + bar_w && x < w - 2; x++) {
            rgb[bar_y * w + x] = px;
            rgb[(bar_y - 1) * w + x] = px;
        }
    } else {
        uint16_t *cw = renderer_get_active_cw_target();
        if (!cw) return;
        uint16_t c = renderer_argb_to_amiga12(px);
        for (int x = 2; x < 2 + bar_w && x < w - 2; x++) {
            cw[bar_y * w + x] = c;
            cw[(bar_y - 1) * w + x] = c;
        }
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
