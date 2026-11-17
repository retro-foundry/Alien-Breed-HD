/* Exercise the actual SDL/GL title composition and its menu hit rectangle. */
#include <SDL.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t *captured;
static int captured_w, captured_h;
static SDL_Window *hidden_window(const char *title, int x, int y, int w, int h, Uint32 flags)
{
    return SDL_CreateWindow(title, x, y, w, h, (flags & ~SDL_WINDOW_SHOWN) | SDL_WINDOW_HIDDEN);
}
static void capture_present(SDL_Renderer *renderer)
{
    assert(SDL_GetRendererOutputSize(renderer, &captured_w, &captured_h) == 0);
    free(captured);
    captured = malloc((size_t)captured_w * captured_h * sizeof(*captured));
    assert(captured);
    assert(SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_ARGB8888,
                                captured, captured_w * 4) == 0);
    SDL_RenderPresent(renderer);
}
#define SDL_CreateWindow hidden_window
#define SDL_RenderPresent capture_present
#include "../src/display.c"
#undef SDL_CreateWindow
#undef SDL_RenderPresent

static void check_hit_rect(int source_w, int source_h)
{
    SDL_Rect expected = display_fit_rect(captured_w, captured_h, source_w, source_h);
    DisplayTextRun runs[DISPLAY_TEXT_LAYOUT_MAX_RUNS];
    int count, scale, line_h, draw_h, top, margin;
    display_clear_text_screen();
    display_draw_line_of_text("NEW GAME", 2);
    display_draw_line_of_text("OPTIONS", 4);
    display_draw_line_of_text("EXIT", 6);
    assert(display_text_layout_in_rect(expected, runs, DISPLAY_TEXT_LAYOUT_MAX_RUNS,
                                      &count, &scale, &line_h, &draw_h, &top, &margin, NULL));
    int window_w, window_h, checked = 0;
    SDL_GetWindowSize(g_window, &window_w, &window_h);
    for (int i = 0; i < count; i++) {
        if (!runs[i].len) continue;
        int x = (expected.x + expected.w / 2) * window_w / captured_w;
        int y = (top + i * line_h + draw_h / 2) * window_h / captured_h;
        int line = -1;
        assert(display_text_line_at_point(x, y, 1, &line));
        assert(line == runs[i].source_line);
        checked++;
    }
    assert(checked == 3);
    display_present_title_screen_alpha(255, 90);
}

int main(int argc, char **argv)
{
    int gl = argc > 1 && !strcmp(argv[1], "--gl");
    SDL_SetMainReady();
    assert(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) == 0);
    SDL_setenv("AB3D_RENDER_DRIVER", gl ? "opengl" : "software", 1);
    SDL_setenv("AB3D_DISABLE_GL_UNPACK", gl ? "0" : "1", 1);
    game_state_init(&g_state);
    g_state.cfg_display_mode = 0;
    g_state.cfg_render_width = 1672;
    g_state.cfg_render_height = 941;
    g_state.cfg_supersampling = 1;
    display_init(&g_state);
    assert(g_window && g_sdl_ren);
    assert(!gl || (g_gl_unpack_ok && g_gl_hud_ok));
    display_load_title_screen();
    int source_w, source_h;
    unsigned char *source = display_load_png_rgba_asset(DISPLAY_TITLE_PNG_PATH,
                                                       &source_w, &source_h, "title test");
    assert(source && source_w == 1672 && source_h == 941);
    assert(g_title_screen_width == source_w && g_title_screen_height == source_h);
    display_present_title_screen_alpha(0, 0);
    assert(captured_w == source_w && captured_h == source_h);
    for (int i = 0; i < captured_w * captured_h; i++) {
        for (int c = 0; c < 3; c++) {
            int actual = (captured[i] >> (16 - c * 8)) & 255;
            assert(abs(actual - source[i * 4 + c]) <= (gl ? 1 : 0));
        }
    }
    if (argc > 2) {
        SDL_Surface *image = SDL_CreateRGBSurfaceWithFormatFrom(captured, captured_w,
            captured_h, 32, captured_w * 4, SDL_PIXELFORMAT_ARGB8888);
        assert(image && SDL_SaveBMP(image, argv[2]) == 0);
        SDL_FreeSurface(image);
    }
    const int sizes[][2] = {{960, 540}, {640, 480}, {480, 800}};
    for (unsigned s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        SDL_SetWindowSize(g_window, sizes[s][0], sizes[s][1]);
        SDL_PumpEvents();
        display_clear_text_screen();
        display_present_title_screen_alpha(0, 0);
        SDL_Rect expected = display_fit_rect(captured_w, captured_h, source_w, source_h);
        assert(expected.x >= 0 && expected.y >= 0 && expected.w <= captured_w && expected.h <= captured_h);
        if (expected.y > 0) assert((captured[captured_w / 2] & 0xffffff) == 0);
        if (expected.x > 0) assert((captured[(captured_h / 2) * captured_w] & 0xffffff) == 0);
        check_hit_rect(source_w, source_h);
    }
    display_release_title_memory();
    display_load_title_screen();
    assert(g_title_screen_width == source_w && g_title_screen_height == source_h);
    stbi_image_free(source);
    free(captured);
    display_shutdown();
    renderer_shutdown();
    SDL_Quit();
    printf("PASS: %s supplied title pixels, widescreen/4:3/portrait menu hit targets and reload\n",
           gl ? "OpenGL" : "SDL software");
    return 0;
}
