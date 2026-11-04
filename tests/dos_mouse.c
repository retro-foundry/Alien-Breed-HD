/* Run the real SDL DOS backend and game's input/INI paths under INT 33h.
 * Event injection checks the game's capture gate without moving the host's
 * pointer. Rendering and audio are outside this input-only fixture. */
#include "../src/input.c"
#include "../src/settings.c"
#include <assert.h>

GameState g_state;
void display_handle_resize(void) {}
void audio_dos_pump(void) {}
void dos_profile_begin(int bucket) { (void)bucket; }
void dos_profile_end(void) {}

static void push_motion(float dx, float dy)
{
    SDL_Event event = {0};
    event.type = SDL_MOUSEMOTION;
    event.motion.windowID = SDL_GetWindowID(SDL_GetKeyboardFocus());
    event.motion.xrel = dx;
    event.motion.yrel = dy;
    assert(SDL_PushEvent((void *)&event));
}

static void push_button(SDL_EventType type)
{
    SDL_Event event = {0};
    event.type = type;
    event.button.windowID = SDL_GetWindowID(SDL_GetKeyboardFocus());
    event.button.button = SDL_BUTTON_LEFT;
    event.button.down = type == SDL_MOUSEBUTTONDOWN;
    assert(SDL_PushEvent((void *)&event));
}

static void check_capture(void)
{
    uint8_t keys[256] = {0}, last = 0;
    MouseState mouse;
    input_init();
    input_update(keys, &last);
    assert(!SDL_GetRelativeMouseMode());
    push_motion(17, -9);
    input_update(keys, &last);
    input_read_mouse(&mouse);
    assert(mouse.dx == 0 && mouse.dy == 0);

    /* Left click must enable the actual driver's relative mode, but must
     * not fire. This assertion fails with the missing SDL callback. */
    push_button(SDL_MOUSEBUTTONDOWN);
    input_update(keys, &last);
    assert(SDL_GetRelativeMouseMode());
    assert(!keys[AMIGA_KEY_RALT]);
    for (int i = 0; i < 8; i++) {
        assert(SDL_SetRelativeMouseMode(true) == 0);
        push_motion(17, -9);
        input_update(keys, &last);
        input_read_mouse(&mouse);
        assert(mouse.dx == 17 && mouse.dy == -9);
        input_consume_mouse_deltas();
    }
    push_button(SDL_MOUSEBUTTONDOWN);
    input_update(keys, &last);
    assert(keys[AMIGA_KEY_RALT]);
    push_button(SDL_MOUSEBUTTONUP);
    input_update(keys, &last);
    assert(!keys[AMIGA_KEY_RALT]);
    push_motion(2, 3);
    input_update(keys, &last);
    input_set_menu_mouse_active(true, keys);
    assert(!SDL_GetRelativeMouseMode());
    input_read_mouse(&mouse);
    assert(mouse.dx == 0 && mouse.dy == 0 && !mouse.left_button);
    push_motion(20, 30);
    input_update(keys, &last);
    input_read_mouse(&mouse);
    assert(mouse.dx == 0 && mouse.dy == 0);
    input_set_menu_mouse_active(false, keys);
    push_button(SDL_MOUSEBUTTONDOWN);
    input_update(keys, &last);
    assert(SDL_GetRelativeMouseMode());
    push_motion(-11, 7);
    input_update(keys, &last);
    input_read_mouse(&mouse);
    assert(mouse.dx == -11 && mouse.dy == 7);
    input_shutdown();
    assert(!SDL_GetRelativeMouseMode());
    puts("PASS: DOS relative capture, motion, buttons, menu release and recapture");
}

static void check_settings(void)
{
    char path[64], tiny[3];
    assert(settings_path_with_suffix(path, sizeof(path), "c:/ab3d.ini", ".tmp"));
    assert(!strcmp(path, "c:/ab3d.tmp"));
    assert(settings_path_with_suffix(path, sizeof(path), "c:\\a.b\\test.ini", ".tmp"));
    assert(!strcmp(path, "c:\\a.b\\test.tmp"));
    assert(settings_path_with_suffix(path, sizeof(path), "c:/a.b/test", ".tmp"));
    assert(!strcmp(path, "c:/a.b/test.tmp"));
    assert(!settings_path_with_suffix(tiny, sizeof(tiny), "test.ini", ".tmp"));
    strcpy(g_menu_options_save_path, "c:/test.ini");
    FILE *file = fopen("c:/test.ini", "wb");
    assert(file);
    fputs("# Keep custom settings\nstart_level=8\nmouse_look=0\n", file);
    fclose(file);
    for (int enabled = 0; enabled < 2; enabled++) {
        g_state.cfg_mouse_look = enabled;
        g_state.cfg_mouse_look_invert_y = !enabled;
        g_state.cfg_run_speed = 100;
        g_state.cfg_gamepad_look_speed = 200;
        settings_save_menu_options(&g_state);
        GameState loaded = {0};
        parse_file(&loaded, "c:/test.ini");
        assert(loaded.cfg_mouse_look == enabled);
        assert(loaded.cfg_mouse_look_invert_y == !enabled);
        assert(loaded.cfg_start_level == 7);
        file = fopen("c:/test.tmp", "rb");
        assert(!file);
    }
    puts("PASS: DOS 8.3 INI replacement persists mouse look and inversion");
}

static void check_driver_motion(void)
{
    uint8_t keys[256] = {0}, last = 0;
    unsigned directions = 0;
    input_init();
    input_apply_relative_mouse(SDL_TRUE, keys);
    assert(SDL_GetRelativeMouseMode());
    puts("READY: native DOS driver motion");
    fflush(NULL);
    Uint64 deadline = SDL_GetTicks() + 10000;
    while (directions != 15 && SDL_GetTicks() < deadline) {
        MouseState mouse;
        input_update(keys, &last);
        input_read_mouse(&mouse);
        if (mouse.dx > 0) directions |= 1;
        if (mouse.dx < 0) directions |= 2;
        if (mouse.dy > 0) directions |= 4;
        if (mouse.dy < 0) directions |= 8;
        input_consume_mouse_deltas();
        SDL_Delay(10);
    }
    input_shutdown();
    assert(directions == 15);
    puts("PASS: host motion reaches both game mouse axes through INT 33h");
}

int main(void)
{
    SDL_SetHint("SDL_DOS_SKIP_EVENT_YIELD", "1");
    SDL_SetHint("SDL_DOS_SKIP_MOUSE_PUMP", "0");
    assert(SDL_Init(SDL_INIT_VIDEO) == 0);
    SDL_Window *window = SDL_CreateWindow("DOS mouse regression", 0, 0, 320, 240, 0);
    assert(window);
    if (getenv("AB3D_TEST_NO_MOUSE")) {
        assert(SDL_SetRelativeMouseMode(true) < 0);
        assert(!SDL_GetRelativeMouseMode());
        puts("PASS: absent DOS mouse driver rejects capture");
    } else if (getenv("AB3D_TEST_LIVE_MOUSE")) {
        check_driver_motion();
    } else {
        check_capture();
    }
    check_settings();
    /* DOSBox Staging keeps its shell open if a program exits within 1s. */
    SDL_Delay(1500);
    SDL_DestroyWindow(window);
    SDL_Quit();
    puts("DOS mouse regression passed");
    return 0;
}
