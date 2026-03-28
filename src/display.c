/*
 * Alien Breed 3D I - PC Port
 * display.c - SDL2 display backend
 *
 * Creates a window and presents the 12-bit Amiga color-word framebuffer:
 * OpenGL renderer uploads GL_R16UI and unpacks in a fragment shader; other
 * drivers unpack on the CPU to an ARGB8888 SDL texture.
 *
 * Base window size comes from ab3d.ini (render_width/render_height).
 * Internal render size is base size multiplied by supersampling.
 * The final image is letterboxed, centered, aspect preserved.
 */

#include "display.h"
#include "renderer.h"
#include <SDL.h>
#include <SDL_opengl.h>
/* SDL_GL_BindTexture / glGenerateMipmap: framebuffer minification when window < internal size */
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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
static int g_gl_unpack_ok; /* OpenGL R16UI + shader path active */
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
/* Release: run in borderless desktop-sized window (never request display mode switch). */
#endif

/* Non-GL path: g_texture is ARGB8888 after CPU unpack. GL path uses g_gl_tex_cw (R16UI). */

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
 * Fallback: CPU unpack to ARGB8888 SDL texture (D3D / no GL context).
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
typedef GLuint (APIENTRY *DisplayGlCreateProgramFn)(void);
typedef GLuint (APIENTRY *DisplayGlCreateShaderFn)(GLenum type);
typedef void   (APIENTRY *DisplayGlAttachShaderFn)(GLuint program, GLuint shader);
typedef void   (APIENTRY *DisplayGlCompileShaderFn)(GLuint shader);
typedef void   (APIENTRY *DisplayGlGetProgramInfoLogFn)(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);

static GLuint g_gl_tex_cw;
static GLuint g_gl_prog;
static GLuint g_gl_vao;
static GLuint g_gl_vbo;

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
    "  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);\n"
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

static void display_gl_shutdown_unpack(void)
{
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
    printf("[DISPLAY] 12-bit unpack: GPU (GL_R16UI + shader)\n");
    return 1;
}

static void display_gl_resize_cw_texture(int w, int h)
{
    if (!g_gl_unpack_ok || w < 1 || h < 1) return;
    g_gl_pixel_storei(0x0CF5 /* GL_UNPACK_ALIGNMENT */, 2);
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
    SDL_GL_GetDrawableSize(g_window, &win_w, &win_h);
    if (win_w < 1) win_w = 1;
    if (win_h < 1) win_h = 1;

    g_gl_pixel_storei(0x0CF5 /* GL_UNPACK_ALIGNMENT */, 2);
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

    SDL_RenderPresent(g_sdl_ren);
}

static void display_cpu_unpack_cw_to_texture(const uint16_t *src, int w, int h)
{
    void *pixels;
    int pitch;
    if (!g_texture || SDL_LockTexture(g_texture, NULL, &pixels, &pitch) < 0) return;
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
    if (g_gl_unpack_ok) {
        display_gl_resize_cw_texture(w, h);
    } else {
        g_texture = SDL_CreateTexture(g_sdl_ren,
            SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
            g_internal_w, g_internal_h);
        if (g_texture) {
            SDL_SetTextureScaleMode(g_texture, SDL_ScaleModeLinear);
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

/* -----------------------------------------------------------------------
 * Lifecycle
 * ----------------------------------------------------------------------- */
void display_init(GameState *state)
{
    int base_rw = 1280;
    int base_rh = 720;
    int supersampling = 1;
    int rw, rh;
    if (state) {
        base_rw = (int)state->cfg_render_width;
        base_rh = (int)state->cfg_render_height;
        supersampling = (int)state->cfg_supersampling;
    }
    if (base_rw < 96) base_rw = 96;
    if (base_rh < 80) base_rh = 80;
    if (supersampling < 1) supersampling = 1;
    if (supersampling > 4) supersampling = 4;

    {
        /* Keep aspect ratio if a requested supersampled target exceeds renderer max size. */
        const double req_w = (double)base_rw * (double)supersampling;
        const double req_h = (double)base_rh * (double)supersampling;
        double fit = 1.0;
        if (req_w > 4096.0 || req_h > 4096.0) {
            const double fit_w = 4096.0 / req_w;
            const double fit_h = 4096.0 / req_h;
            fit = (fit_w < fit_h) ? fit_w : fit_h;
            if (fit < 0.0) fit = 0.0;
        }
        rw = (int)(req_w * fit + 0.5);
        rh = (int)(req_h * fit + 0.5);
        if (rw < 96) rw = 96;
        if (rh < 80) rh = 80;
        if (rw > 4096) rw = 4096;
        if (rh > 4096) rh = 4096;
    }

    g_internal_w = rw;
    g_internal_h = rh;

    if (!SDL_WasInit(SDL_INIT_VIDEO)) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            printf("[DISPLAY] SDL_Init failed: %s\n", SDL_GetError());
            return;
        }
    }
    display_log_display_mode_snapshot("startup-before-window");

    /* Request GL 3.0+ so integer textures (GL_R16UI) work when using the OpenGL render driver. */
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

#ifdef AB3D_RELEASE
    printf("[DISPLAY] SDL2 init (window=desktop, base=%dx%d, internal render=%dx%d, supersampling=%d)\n",
           base_rw, base_rh, g_internal_w, g_internal_h, supersampling);
#else
    printf("[DISPLAY] SDL2 init (window %dx%d, internal render %dx%d, supersampling=%d; resize to letterbox)\n",
           base_rw, base_rh, g_internal_w, g_internal_h, supersampling);
#endif

    renderer_init();

    const char *driver_override = SDL_getenv("AB3D_RENDER_DRIVER");
    int window_w = base_rw;
    int window_h = base_rh;
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
    window_flags = SDL_WINDOW_SHOWN | SDL_WINDOW_BORDERLESS;
    if (driver_override && strcmp(driver_override, "opengl") == 0) {
        /* OpenGL path keeps desktop-sized borderless mode but requests GL-capable window. */
        window_flags |= SDL_WINDOW_OPENGL;
        printf("[DISPLAY] Release OpenGL path: borderless desktop window + SDL_WINDOW_OPENGL\n");
    }
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

    {
        const char *driver_try[4];
        int driver_try_count = 0;
        const char *selected_hint = "";

        if (driver_override && driver_override[0] != '\0' && strcmp(driver_override, "auto") != 0) {
            driver_try[driver_try_count++] = driver_override;
            driver_try[driver_try_count++] = "";
        } else {
#ifdef AB3D_RELEASE
            /* Release: avoid forced OpenGL to reduce fullscreen mode-switch regressions. */
            driver_try[driver_try_count++] = "direct3d11";
            driver_try[driver_try_count++] = "direct3d";
            driver_try[driver_try_count++] = "";
#else
            /* Dev/debug: keep OpenGL preference for mip-based minification quality testing. */
            driver_try[driver_try_count++] = "opengl";
            driver_try[driver_try_count++] = "direct3d11";
            driver_try[driver_try_count++] = "direct3d";
            driver_try[driver_try_count++] = "";
#endif
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
#ifdef AB3D_RELEASE
            if (ri.name && strcmp(ri.name, "opengl") == 0) {
                printf("[DISPLAY] warning: Release selected OpenGL driver; set AB3D_RENDER_DRIVER=direct3d11 to force D3D\n");
            }
#endif
        }
    }

    display_gl_try_init_unpack();
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

/* -----------------------------------------------------------------------
 * Main rendering
 * ----------------------------------------------------------------------- */
void display_draw_display(GameState *state)
{
    /* 1. Software-render the 3D scene into the rgb buffer */
    renderer_draw_display(state);

    if (!g_sdl_ren) return;

    const uint16_t *src = renderer_get_cw_buffer();
    if (!src) return;

    int w = renderer_get_width(), h = renderer_get_height();

    if (g_gl_unpack_ok) {
        display_gl_present_cw(src, w, h);
    } else {
        if (!g_texture) return;
        display_cpu_unpack_cw_to_texture(src, w, h);
#if SDL_VERSION_ATLEAST(2, 0, 12)
        display_regenerate_framebuffer_mipmaps_if_downscaled(w, h);
#endif
        SDL_SetRenderDrawColor(g_sdl_ren, 0, 0, 0, 255);
        SDL_RenderClear(g_sdl_ren);
        SDL_RenderCopy(g_sdl_ren, g_texture, NULL, &g_present_dst_rect);
        SDL_RenderPresent(g_sdl_ren);
    }

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
