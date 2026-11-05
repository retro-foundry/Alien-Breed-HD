/* Run the real SDL DOS backend and game's input/INI paths under INT 33h.
 * Event injection checks the game's capture gate without moving the host's
 * pointer. Rendering and audio are outside this input-only fixture. */
#include "../src/input.c"
#include "../src/settings.c"
#include <assert.h>
#include <dpmi.h>

GameState g_state;
void display_handle_resize(void) {}
void audio_dos_pump(void) {}
void dos_profile_begin(int bucket) { (void)bucket; }
void dos_profile_end(void) {}

/* Compile the installed SDL button helper unchanged, with a scripted INT 33h
 * boundary. SDL's real button state/event queue and game input remain live. */
typedef struct { SDL_Window *focus; } TestDosMouse;
static unsigned driver_buttons,driver_presses[3];
static int scripted_int33(int interrupt,__dpmi_regs *regs)
{
    assert(interrupt==0x33 && regs->x.ax==5 && regs->x.bx<3);
    unsigned button=regs->x.bx;
    regs->x.ax=(unsigned short)driver_buttons;
    regs->x.bx=(unsigned short)driver_presses[button];
    driver_presses[button]=0;
    return 0;
}
extern void SDL_SendMouseButton(Uint64,SDL_Window *,SDL_MouseID,Uint8,bool);
#define SDL_Mouse TestDosMouse
#define __dpmi_int scripted_int33
#include "dos_mouse_buttons.h"
#undef __dpmi_int
#undef SDL_Mouse

static void check_driver_buttons(SDL_Window *window)
{
    uint8_t keys[256]={0},last=0;
    TestDosMouse mouse={window};
    const Uint8 buttons[]={SDL_BUTTON_LEFT,SDL_BUTTON_RIGHT};
    const unsigned keycodes[]={AMIGA_KEY_RALT,AMIGA_KEY_SPACE};
    input_init();
    input_apply_relative_mouse(SDL_TRUE,keys);
    SDL_SetHint("SDL_DOS_SKIP_MOUSE_PUMP","1");
    for(unsigned b=0;b<2;b++) {
        unsigned mask=1u<<b,key=keycodes[b];
        /* A released tap must survive both the driver boundary and repeated
         * frame polls, then disappear after one control consumption. */
        for(unsigned count=1;count<=3;count++) {
            driver_presses[b]=count;driver_buttons=0;
            DOSVESA_SendButton(&mouse,b,buttons[b]);
            input_update(keys,&last);assert(keys[key]);
            input_update(keys,&last);assert(keys[key]);
            input_consume_mouse_deltas();
            DOSVESA_SendButton(&mouse,b,buttons[b]);
            input_update(keys,&last);assert(!keys[key]);
        }
        /* Holding remains held after consuming the edge; releasing restores
         * the physical state without adding another pressed frame. */
        driver_presses[b]=1;driver_buttons=mask;
        DOSVESA_SendButton(&mouse,b,buttons[b]);
        input_update(keys,&last);assert(keys[key]);input_consume_mouse_deltas();
        DOSVESA_SendButton(&mouse,b,buttons[b]);
        input_update(keys,&last);assert(keys[key]);
        driver_buttons=0;DOSVESA_SendButton(&mouse,b,buttons[b]);
        input_update(keys,&last);assert(!keys[key]);
    }
    driver_presses[0]=1;DOSVESA_SendButton(&mouse,0,SDL_BUTTON_LEFT);
    input_update(keys,&last);assert(keys[AMIGA_KEY_RALT]);
    input_set_menu_mouse_active(true,keys);
    assert(!keys[AMIGA_KEY_RALT]);
    driver_presses[0]=1;DOSVESA_SendButton(&mouse,0,SDL_BUTTON_LEFT);
    input_update(keys,&last);assert(!keys[AMIGA_KEY_RALT]);
    MenuMouseState menu;input_read_menu_mouse(&menu);assert(menu.left_pressed);
    input_set_menu_mouse_active(false,keys);
    input_apply_relative_mouse(SDL_TRUE,keys);
    input_update(keys,&last);assert(!keys[AMIGA_KEY_RALT]);
    SDL_SetHint("SDL_DOS_SKIP_MOUSE_PUMP","0");
    input_shutdown();
    puts("PASS: installed SDL button-counter helper preserves released taps, holds, releases and menu/capture boundaries");
}

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
    input_consume_mouse_deltas();
    push_button(SDL_MOUSEBUTTONUP);
    input_update(keys, &last);
    assert(!keys[AMIGA_KEY_RALT]);
    /* Both edges can arrive while a slow frame draws an enemy. Preserve the
     * press until control consumes it, even if more polls precede that tick. */
    push_button(SDL_MOUSEBUTTONDOWN);
    push_button(SDL_MOUSEBUTTONUP);
    input_update(keys, &last);
    assert(keys[AMIGA_KEY_RALT]);
    input_read_mouse(&mouse);
    assert(mouse.left_button);
    input_update(keys, &last);
    assert(keys[AMIGA_KEY_RALT]);
    input_consume_mouse_deltas();
    input_update(keys, &last);
    assert(!keys[AMIGA_KEY_RALT]);
    input_read_mouse(&mouse);
    assert(!mouse.left_button);
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
    assert(directions == 15);
    puts("PASS: host motion reaches both game mouse axes through INT 33h");
    input_update(keys, &last);
    input_consume_mouse_deltas();
    puts("READY: native DOS button gap");
    fflush(NULL);
    /* No event polling while a real host press/release occurs, as during a
     * long enemy draw. The DOS driver's press counter must preserve it. */
    SDL_Delay(1500);
    input_update(keys, &last);
    printf("[MOUSE] released host tap reached control=%d\n", keys[AMIGA_KEY_RALT]);
    fflush(NULL);
    assert(keys[AMIGA_KEY_RALT] && !g_mouse.left_button);
    input_consume_mouse_deltas();
    input_update(keys, &last);
    assert(!keys[AMIGA_KEY_RALT]);
    puts("PASS: released host click survives a 1500ms gap without extra held fire");
    input_shutdown();
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
        check_driver_buttons(window);
    }
    check_settings();
    /* DOSBox Staging keeps its shell open if a program exits within 1s. */
    SDL_Delay(1500);
    SDL_DestroyWindow(window);
    SDL_Quit();
    puts("DOS mouse regression passed");
    return 0;
}
