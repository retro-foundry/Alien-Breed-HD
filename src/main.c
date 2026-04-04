/*
 * Alien Breed 3D I - PC Port
 * main.c - Entry point
 *
 * Translated from: AB3DI.s Main (~line 66)
 *
 * Original flow:
 *   Main:
 *     SAVEREGS
 *     jsr OSFriendlyStartup     ; Take over Amiga hardware from OS
 *     jsr SetupGame             ; Init hardware, open libraries, alloc memory
 *     jsr PlayGame              ; Outer game loop (menu + levels)
 *     jsr TearDownGame          ; Cleanup: stop audio, free memory, save passwords
 *     jsr OSFriendlyExit        ; Restore Amiga OS
 *     move.l #0,d0
 *     rts
 *
 * On PC:
 *   - OSFriendlyStartup/Exit are no-ops (no Amiga OS to save/restore)
 *   - SetupGame initializes subsystems instead of hardware registers
 *   - PlayGame is the same outer loop
 *   - TearDownGame shuts down subsystems
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "logging.h"
#define printf ab3d_log_printf

/* Define SDL_MAIN_HANDLED before including SDL.h so SDL doesn't hijack main(). */
#define SDL_MAIN_HANDLED
#include <SDL.h>
#include "game_types.h"
#include "game_state.h"
#include "math_tables.h"
#include "control_loop.h"
#include "display.h"
#include "input.h"
#include "audio.h"
#include "io.h"
#include "renderer_3dobj.h"
#include "settings.h"

/*
 * setup_game - Initialize all subsystems
 *
 * Translated from AB3DI.s SetupGame (~line 84):
 *   - Set serial port baud rate (for multiplayer)   -> stubbed
 *   - Init control mode flags                       -> game_state_init
 *   - Open dos.library, lowlevel.library            -> stubbed (io_init)
 *   - Allocate text screen, level data memory       -> stubbed (display)
 *   - Setup text screen copper                      -> stubbed
 *   - Add copper interrupt, keyboard interrupt      -> stubbed (input)
 *   - Load passwords, prefs                         -> stubbed (io)
 *   - Setup mouse sensitivity                       -> stubbed
 */
static void setup_game(GameState *state)
{
    printf("=== Alien Breed 3D I - PC Port ===\n");
    printf("SetupGame\n");

    math_tables_init();
    game_state_init(state);

    io_init();
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("[DISPLAY] SDL_Init failed: %s\n", SDL_GetError());
        exit(1);
    }
    settings_load(state);
    fflush(stdout);

    display_init(state);
    input_init();
    audio_init();
    audio_set_master_volume((int)state->cfg_volume);

    display_alloc_text_screen();

    io_load_passwords();
    io_load_prefs(state->prefs_file, sizeof(state->prefs_file));

    printf("SetupGame complete\n\n");
}

/*
 * tear_down_game - Cleanup all subsystems
 *
 * Translated from AB3DI.s TearDownGame (~line 163):
 *   - mt_end (stop audio)
 *   - Set null copper list
 *   - Remove copper/keyboard interrupts
 *   - SavePasswords
 *   - Close libraries
 *   - Release all allocated memory
 */
static void tear_down_game(GameState *state)
{
    printf("\nTearDownGame\n");

    audio_mt_end();

    io_save_passwords();

    io_release_level_memory(&state->level);

    display_release_text_screen();
    display_release_copper_screen();
    display_release_title_memory();

    audio_shutdown();
    input_shutdown();
    display_shutdown();
    io_shutdown();

    printf("TearDownGame complete\n");
}

/*
 * main - Entry point
 */
int main(int argc, char *argv[])
{
    int log_init_ok;
    int enable_3dobj_anim = 1;

    /* OSFriendlyStartup is a no-op on PC
     * (no Amiga system state to save/restore) */
    SDL_SetMainReady();

    log_init_ok = ab3d_log_init_file();
    if (!log_init_ok) {
        fprintf(stderr, "[LOG] Failed to open log file (expected: %s)\n", ab3d_log_path());
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--3dobj-anim") == 0) {
            enable_3dobj_anim = 1;
        } else if (strcmp(argv[i], "--amiga-3dobj-static") == 0) {
            enable_3dobj_anim = 0;
        }
    }
    poly_obj_set_use_object_frame(enable_3dobj_anim);
    if (enable_3dobj_anim) {
        printf("[3DOBJ] Animation mode: object frame enabled (default; --3dobj-anim)\n");
    } else {
        printf("[3DOBJ] Animation mode: Amiga static frame 0 (--amiga-3dobj-static)\n");
    }

    setup_game(&g_state);
    play_game(&g_state);
    tear_down_game(&g_state);

    /* OSFriendlyExit is a no-op on PC */

    printf("\n=== Exit (code 0) ===\n");
    ab3d_log_shutdown();
    return 0;
}
