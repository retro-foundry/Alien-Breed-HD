/*
 * Alien Breed 3D I - PC Port
 * control_loop.c - Outer control loop
 *
 * Translated from: ControlLoop.s, AB3DI.s (PlayTheGame)
 *
 * Flow (from ControlLoop.s comments):
 *   1. Load Title Music
 *   2. Load title screen
 *   3. Fade up title screen
 *   4. Add 'loading' message
 *   5. Load samples and walls
 *   6. LOOP START
 *   7. Option select screens
 *   8. Free music mem, allocate level mem
 *   9. Load level
 *  10. Play level with options selected
 *  11. Reload title music
 *  12. Reload title screen
 *  13. goto 6
 */

#include "control_loop.h"
#include "game_loop.h"
#include "game_data.h"
#include "level.h"
#include "objects.h"
#include "player.h"
#include "display.h"
#include "renderer.h"
#include "input.h"
#include "audio.h"
#ifdef AB3D_DOS
#include "platform_dos.h"
#endif
#include "io.h"
#include "settings.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "logging.h"
#include <SDL.h>
#define printf ab3d_log_printf

#define LEVEL_TEXT_FADE_STEPS_NATIVE 8
#define LEVEL_TEXT_FADE_FRAME_MS     20
#define LEVEL_TEXT_MIN_DISMISS_MS    750

static void control_game_over_fade_tick(float progress_0_to_1, void *userdata);
static void control_level_complete_fade_tick(float progress_0_to_1, void *userdata);
static bool s_autosave_next_level_start = false;
static bool s_show_transition_level_text = false;
static bool s_front_menu_requested = true;

/* -----------------------------------------------------------------------
 * Password system
 *
 * Translated from ControlLoop.s CalcPassword / PassLineToGame / GetStats.
 *
 * The password encodes: energy, max_level, which guns are visible,
 * ammo for each gun. It uses parity bits and a checksum for validation,
 * then interleaves/mixes the bits and converts to A-P characters.
 * ----------------------------------------------------------------------- */

/* GetParity: set bit 7 as parity of bits 0-6 */
static uint8_t get_parity(uint8_t val)
{
    uint8_t result = val;
    for (int i = 6; i >= 0; i--) {
        if (result & (1 << i)) {
            result ^= 0x80;
        }
    }
    return result;
}

/* CheckParity: returns true if parity is valid */
static bool check_parity(uint8_t val)
{
    uint8_t computed = 0;
    for (int i = 6; i >= 0; i--) {
        if (val & (1 << i)) {
            computed ^= 0x80;
        }
    }
    return (val & 0x80) == (computed & 0x80);
}

/* Mix two bytes by interleaving their bits */
static uint16_t mix_bytes(uint8_t a, uint8_t b)
{
    uint16_t result = 0;
    b = ~b; /* NOT b before mixing */
    for (int i = 0; i < 8; i++) {
        result <<= 1;
        if (a & 1) result |= 1;
        a >>= 1;
        result <<= 1;
        if (b & 1) result |= 1;
        b >>= 1;
    }
    return result;
}

static int16_t control_read_be16(const uint8_t *p)
{
    return (int16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}

/* Rebuild per-level condition bits from loaded level state.
 * Prevents switch/key conditions from leaking between levels (e.g. level 2 exit switch puzzle). */
static void game_rebuild_level_conditions(GameState *state)
{
    if (!state) {
        game_conditions = 0;
        return;
    }

    uint16_t rebuilt = 0;

    /* Preserve key bits already collected in this loaded state (key object removed from world). */
    if (state->level.object_data) {
        const uint8_t *obj = state->level.object_data;
        for (int i = 0; i < MAX_OBJECTS; i++, obj += OBJECT_SIZE) {
            int16_t cid = control_read_be16(obj + 0);
            if (cid < 0) break;

            int8_t obj_type = (int8_t)obj[16];
            int16_t zone = control_read_be16(obj + 12);
            if (obj_type == OBJ_NBR_KEY && zone < 0) {
                rebuilt = (uint16_t)(rebuilt | (uint8_t)obj[17]);
            }
        }
    }

    /* Mirror switch on/off state bytes into condition bits.
     * Amiga SwitchRoutine iterates 8 entries (bits 4..11). */
    if (state->level.switch_data) {
        const uint8_t *sw = state->level.switch_data;
        for (int switch_index = 0; switch_index < 8; switch_index++, sw += 14) {
            unsigned bit_num = 4u + (unsigned)switch_index;
            uint16_t bit_mask = (uint16_t)(1u << bit_num);
            if ((int8_t)sw[10] != 0)
                rebuilt = (uint16_t)(rebuilt | bit_mask);
        }
    }

    game_conditions = (int16_t)rebuilt;
}

/* Unmix a word into two bytes */
static void unmix_word(uint16_t word, uint8_t *a, uint8_t *b)
{
    uint8_t ra = 0, rb = 0;
    for (int i = 0; i < 8; i++) {
        rb |= (uint8_t)((word & 1) << i);
        word >>= 1;
        ra |= (uint8_t)((word & 1) << i);
        word >>= 1;
    }
    *a = ra;
    *b = (uint8_t)(~rb); /* invert back */
}

void calc_password(GameState *state)
{
    uint8_t passbuf[8];
    memset(passbuf, 0, sizeof(passbuf));

    /* Byte 0: energy with parity */
    passbuf[0] = get_parity((uint8_t)(state->plr1.energy & 0x7F));

    /* Byte 1: gun visibility flags (bits 7-4) + max_level (bits 3-0) */
    uint8_t guns = 0;
    if (state->plr1.gun_data[1].visible) guns |= 0x80;
    if (state->plr1.gun_data[2].visible) guns |= 0x40;
    if (state->plr1.gun_data[4].visible) guns |= 0x20;
    if (state->plr1.gun_data[7].visible) guns |= 0x10;
    passbuf[1] = (uint8_t)(guns | (state->max_level & 0x0F));

    /* Byte 7: checksum of byte 1 */
    passbuf[7] = (uint8_t)((uint8_t)(passbuf[1] ^ 0xB5) * (uint8_t)(-1) + 50);

    /* Bytes 2-6: ammo for guns 0,1,2,4,7 with parity */
    passbuf[2] = get_parity((uint8_t)((state->plr1.gun_data[0].ammo >> 3) & 0x7F));
    passbuf[3] = get_parity((uint8_t)((state->plr1.gun_data[1].ammo >> 3) & 0x7F));
    passbuf[4] = get_parity((uint8_t)((state->plr1.gun_data[2].ammo >> 3) & 0x7F));
    passbuf[5] = get_parity((uint8_t)((state->plr1.gun_data[4].ammo >> 3) & 0x7F));
    passbuf[6] = get_parity((uint8_t)((state->plr1.gun_data[7].ammo >> 3) & 0x7F));

    /* Mix bytes: interleave passbuf[0..3] with passbuf[7..4] */
    uint16_t pass[4];
    pass[0] = mix_bytes(passbuf[0], passbuf[7]);
    pass[1] = mix_bytes(passbuf[1], passbuf[6]);
    pass[2] = mix_bytes(passbuf[2], passbuf[5]);
    pass[3] = mix_bytes(passbuf[3], passbuf[4]);

    /* Convert to A-P characters (4 bits -> 'A'+nibble) */
    char password_str[17];
    int pos = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t *pb = (uint8_t*)&pass[i];
        password_str[pos++] = 'A' + (pb[1] & 0x0F);
        password_str[pos++] = 'A' + ((pb[1] >> 4) & 0x0F);
        password_str[pos++] = 'A' + (pb[0] & 0x0F);
        password_str[pos++] = 'A' + ((pb[0] >> 4) & 0x0F);
    }
    password_str[16] = '\0';

    /* Store in password storage */
    if (state->max_level >= 0 && state->max_level < MAX_LEVELS) {
        memcpy(state->password_storage + state->max_level * (PASSWORD_LENGTH + 1),
               password_str, PASSWORD_LENGTH + 1);
    }

    printf("[PASSWORD] Generated: %s (level %d)\n", password_str, state->max_level);
}

int pass_line_to_game(GameState *state, const char *password)
{
    /* Convert A-P characters back to 4 mixed words */
    uint16_t pass[4];
    for (int i = 0; i < 4; i++) {
        uint8_t lo = (uint8_t)((password[i*4+1] - 'A') << 4) | (password[i*4] - 'A');
        uint8_t hi = (uint8_t)((password[i*4+3] - 'A') << 4) | (password[i*4+2] - 'A');
        pass[i] = (uint16_t)((hi << 8) | lo);
    }

    /* Unmix */
    uint8_t passbuf[8];
    unmix_word(pass[0], &passbuf[0], &passbuf[7]);
    unmix_word(pass[1], &passbuf[1], &passbuf[6]);
    unmix_word(pass[2], &passbuf[2], &passbuf[5]);
    unmix_word(pass[3], &passbuf[3], &passbuf[4]);

    /* Validate parity on bytes 0,2,3,4,5,6 */
    if (!check_parity(passbuf[0])) return -1;
    if (!check_parity(passbuf[2])) return -1;
    if (!check_parity(passbuf[3])) return -1;
    if (!check_parity(passbuf[4])) return -1;
    if (!check_parity(passbuf[5])) return -1;
    if (!check_parity(passbuf[6])) return -1;

    /* Validate checksum */
    uint8_t expected = (uint8_t)((uint8_t)(passbuf[1] ^ 0xB5) * (uint8_t)(-1) + 50);
    if (expected != passbuf[7]) return -1;

    /* Decode into game state (GetStats) */
    state->plr1.energy = (int16_t)(passbuf[0] & 0x7F);
    state->max_level = (int16_t)(passbuf[1] & 0x0F);
    state->plr1.gun_data[1].visible = (passbuf[1] & 0x80) ? -1 : 0;
    state->plr1.gun_data[2].visible = (passbuf[1] & 0x40) ? -1 : 0;
    state->plr1.gun_data[4].visible = (passbuf[1] & 0x20) ? -1 : 0;
    state->plr1.gun_data[7].visible = (passbuf[1] & 0x10) ? -1 : 0;
    state->plr1.gun_data[0].ammo = (int16_t)((passbuf[2] & 0x7F) << 3);
    state->plr1.gun_data[1].ammo = (int16_t)((passbuf[3] & 0x7F) << 3);
    state->plr1.gun_data[2].ammo = (int16_t)((passbuf[4] & 0x7F) << 3);
    state->plr1.gun_data[4].ammo = (int16_t)((passbuf[5] & 0x7F) << 3);
    state->plr1.gun_data[7].ammo = (int16_t)((passbuf[6] & 0x7F) << 3);

    printf("[PASSWORD] Decoded: energy=%d level=%d\n",
           state->plr1.energy, state->max_level);
    return 0;
}

int play_the_game_should_show_level_text(const GameState *state)
{
    return s_show_transition_level_text &&
           state &&
           state->mode == MODE_SINGLE &&
           !state->f9_pending_apply_save &&
           state->current_level >= 0 &&
           state->current_level < MAX_LEVELS;
}

int play_the_game_level_text_fade_steps(void)
{
    return LEVEL_TEXT_FADE_STEPS_NATIVE;
}

int play_the_game_level_text_min_dismiss_ms(void)
{
    return LEVEL_TEXT_MIN_DISMISS_MS;
}

int play_the_game_level_text_alpha_for_step(int step)
{
    int steps = play_the_game_level_text_fade_steps();
    if (steps < 1) steps = 1;
    if (step < 0) step = 0;
    if (step >= steps) step = steps - 1;

    return ((step + 1) * 255 + steps - 1) / steps;
}

void play_the_game_present_level_text(GameState *state, int alpha)
{
    if (!play_the_game_should_show_level_text(state)) return;

    display_clear_text_screen();
    int lev = state->current_level;
    int heading_row = 0;
    for (int row = 0; row < LEVEL_TEXT_VISIBLE_ROWS; row++) {
        const char *line = level_text[lev][row];
        if (line && line[0]) {
            heading_row = (row > 1) ? row - 2 : 0;
            break;
        }
    }
    for (int row = 0; row < LEVEL_TEXT_VISIBLE_ROWS; row++) {
        const char *line = level_text[lev][row];
        display_draw_line_of_text((line && line[0]) ? line : " ", row);
    }
    display_draw_line_of_text(level_text_heading[lev], heading_row);
    display_present_text_screen_alpha(alpha);
}

int play_the_game_poll_level_text_input(GameState *state)
{
    (void)state;

    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
        switch (ev.type) {
        case SDL_QUIT:
            return -1;

        case SDL_KEYDOWN:
            if (!ev.key.repeat) {
#ifdef AB3D_DOS
                if(display_text_next_page())return 0;
#endif
                return 1;
            }
            break;

        case SDL_MOUSEBUTTONDOWN:
        case SDL_CONTROLLERBUTTONDOWN:
        case SDL_JOYBUTTONDOWN:
        case SDL_FINGERDOWN:
#ifdef AB3D_DOS
            if(display_text_next_page())return 0;
#endif
            return 1;

        case SDL_WINDOWEVENT:
            switch (ev.window.event) {
            case SDL_WINDOWEVENT_RESIZED:
            case SDL_WINDOWEVENT_SIZE_CHANGED:
                display_handle_resize();
                play_the_game_present_level_text(state, 255);
                break;
            default:
                break;
            }
            break;

        default:
            break;
        }
    }

    if (SDL_GetMouseState(NULL, NULL) &
        (SDL_BUTTON(SDL_BUTTON_LEFT) |
         SDL_BUTTON(SDL_BUTTON_MIDDLE) |
         SDL_BUTTON(SDL_BUTTON_RIGHT))) {
        return 1;
    }

    return 0;
}

int play_the_game_drain_level_text_input(GameState *state)
{
    (void)state;

    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
        switch (ev.type) {
        case SDL_QUIT:
            return -1;

        case SDL_WINDOWEVENT:
            switch (ev.window.event) {
            case SDL_WINDOWEVENT_RESIZED:
            case SDL_WINDOWEVENT_SIZE_CHANGED:
                display_handle_resize();
                play_the_game_present_level_text(state, 255);
                break;
            default:
                break;
            }
            break;

        default:
            break;
        }
    }

    return 0;
}

void play_the_game_clear_level_text(GameState *state)
{
#ifdef AB3D_DOS
    display_text_reset_pages();
#endif
    if (state) input_clear_keyboard(state->key_map);
    display_clear_text_screen();
}

#if !defined(__EMSCRIPTEN__)
static void control_level_text_fade_in(GameState *state)
{
#ifdef AB3D_DOS
    display_text_reset_pages();
#endif
    int steps = play_the_game_level_text_fade_steps();
    for (int step = 0; step < steps; step++) {
        play_the_game_present_level_text(state,
            play_the_game_level_text_alpha_for_step(step));
        SDL_Delay(LEVEL_TEXT_FADE_FRAME_MS);
    }
}

static int control_level_text_wait_and_fade_out(GameState *state)
{
    if (play_the_game_drain_level_text_input(state) < 0) {
        state->running = false;
        state->finished_level = 0;
        return 0;
    }

    Uint32 wait_started_ms = SDL_GetTicks();
    Uint32 min_dismiss_ms =
        (Uint32)play_the_game_level_text_min_dismiss_ms();

    for (;;) {
        play_the_game_present_level_text(state, 255);
        if (min_dismiss_ms > 0 &&
            (Uint32)(SDL_GetTicks() - wait_started_ms) < min_dismiss_ms) {
            if (play_the_game_drain_level_text_input(state) < 0) {
                state->running = false;
                state->finished_level = 0;
                return 0;
            }
            SDL_Delay(16);
            continue;
        }

        int input = play_the_game_poll_level_text_input(state);
        if (input < 0) {
            state->running = false;
            state->finished_level = 0;
            return 0;
        }
        if (input > 0) {
            break;
        }
        SDL_Delay(16);
    }

    for (int step = play_the_game_level_text_fade_steps() - 1; step >= 0; step--) {
        play_the_game_present_level_text(state,
            play_the_game_level_text_alpha_for_step(step));
        SDL_Delay(LEVEL_TEXT_FADE_FRAME_MS);
    }
    play_the_game_clear_level_text(state);
    display_present_text_screen_alpha(0);
    return 1;
}

#endif

/*
 * play_the_game - Runs a single level from start to death/completion
 *
 * Translated from AB3DI.s PlayTheGame (~line 484 onwards).
 *
 * Original flow:
 *   - Clear/draw text screen
 *   - Allocate copper screen memory
 *   - Allocate level memory
 *   - Setup players (send level number in MP)
 *   - Load level data, graphics, clips (via dos.library + UnLHA)
 *   - Parse level data (blag:) - resolve pointers, assign clips
 *   - Setup control mode from prefs
 *   - Initialize player positions
 *   - Setup audio channels
 *   - Build scale table
 *   - Set copper/DMA/interrupt registers
 *   - Enter main game loop (mainLoop)
 */
void play_the_game_prepare_level(GameState *state, bool *copper_screen_ready)
{
    bool applying_pending_save = state->f9_pending_apply_save;

    s_show_transition_level_text = false;
    state->running = true;

    printf("[GAME] === PlayTheGame: level %d ===\n", state->current_level);

    /* Level intro text is prepared/faded by caller, matching AB3DI.s
     * DrawLevelText before level load. */

    if (!*copper_screen_ready) {
        display_alloc_copper_screen();
        display_init_copper_screen();
        *copper_screen_ready = true;
    }

    /* ---- Load level data ---- */
#ifdef AB3D_DOS
    /* Reload the front-end picture on demand; it is not resident in a level. */
    display_release_title_memory();
#endif
    io_load_level_data(&state->level, state->current_level);
    io_load_level_graphics(&state->level, state->current_level);
    io_load_level_clips(&state->level, state->current_level);

    /* ---- Parse level data (blag:) ----
         * Original resolves offsets in level data to absolute pointers:
         *   DoorData, LiftData, SwitchData, ZoneGraphAdds, zoneAdds,
         *   Points, FloorLines, ObjectData, PlayerShotData, NastyShotData,
         *   ObjectPoints, PLR1_Obj, PLR2_Obj
         * And assigns clip data to zone graph lists.
         */
    /* Parse raw level data only once; level_parse resolves zone_adds/points. */
    if (state->level.data && state->level.graphics && !state->level.zone_adds) {
        level_parse(&state->level);

        /* Assign clip data to zone graph lists */
        if (state->level.clips && state->level.num_zones > 0) {
            level_assign_clips(&state->level, state->level.num_zones);
        }

        /* ListOfGraphRooms is now derived per-frame from the player's
         * current zone data (at offset 48 = ToListOfGraph).  It is set
         * in player.c (player_init_from_level and player_physics_and_collision).
         * No global allocation needed here. */

        /* Allocate workspace (zone visibility bitmask) */
        int zone_slots = level_zone_slot_count(&state->level);
        if (zone_slots > 0 && !state->level.workspace) {
            state->level.workspace = (uint8_t *)calloc(1,
                (size_t)(zone_slots + 1));
        }

        /* Initialize brightness animation state (Amiga brightAnimTable indices) */
        memset(state->level.bright_anim_indices, 0, sizeof(state->level.bright_anim_indices));
        state->level.bright_anim_values[0] = 0;
        state->level.bright_anim_values[1] = 0;
        state->level.bright_anim_values[2] = 0;

        printf("[GAME] Level parsed: %d zones\n", state->level.num_zones);
    } else if (state->level.points) {
        printf("[GAME] Level pointers already resolved by loader\n");
    } else {
        printf("[GAME] No level data loaded\n");
    }

    /* Apply one-time level-specific data fixes at load time.
     * Keep this outside the parse-only branch so it still runs when
     * pointers were already resolved by the loader path. */
    level_apply_level_specific_fixes(&state->level, state->current_level);

    printf("[GAME] Shot pools: player=%d nasty=%d object_points=%d\n",
           PLAYER_SHOT_SLOT_COUNT,
           NASTY_SHOT_SLOT_COUNT,
           (int)state->level.num_object_points);
    /* Ensure each object has world size in its record (Amiga style), for file and test levels */
    if (state->level.object_data && state->level.num_object_points > 0)
        object_init_world_sizes_from_types(&state->level);
    if (!state->f9_pending_apply_save) {
        renderer_build_level_sky_cache(&state->level);
    }

    /* ---- Setup control mode from prefs ---- */
    /* Original checks Prefsfile[0] for 'k','m','n','j','p' */
    printf("[GAME] Control mode: mouse+kbd (default)\n");

    /* ---- Init player positions from level header (skip when F9 will apply save after load) ---- */
    if (!state->f9_pending_apply_save) {
        player_init_from_level(state);
        game_state_reset_enemy_burning(state);
    }

    state->num_explosions = 0;
    state->num_pending_blasts = 0;

    /* ---- Audio setup ---- */
    audio_mt_init();

    /* ---- Clear keyboard ---- */
    input_clear_keyboard(state->key_map);

    /* ---- Set initial state ---- */
    state->hitcol = 0;
    state->hitcol2 = 0;
    state->xdiff1 = 0;
    state->zdiff1 = 0;
    state->xdiff2 = 0;
    state->zdiff2 = 0;
    state->master_quitting = false;
    state->slave_quitting = (state->mode == MODE_SINGLE);
    state->do_anything = true;

    if (state->mode != MODE_SINGLE) {
        state->plr1.energy = PLAYER_MAX_ENERGY;
    }
    state->plr2.energy = PLAYER_MAX_ENERGY;

    /* F9 cross-level: apply position/orientation only after level + objects are ready */
    if (state->f9_pending_apply_save) {
        state->f9_pending_apply_save = false;
        player_apply_save_payload_after_level_load(state);
        renderer_build_level_sky_cache(&state->level);
        printf("[PLAYER] load: save restored (level %d)\n",
               (int)state->current_level);
    }

    renderer_automap_preallocate_for_level(&state->level);

    game_rebuild_level_conditions(state);

    if (s_autosave_next_level_start && !applying_pending_save &&
        state->mode == MODE_SINGLE) {
        player_save_autosave(state);
    }
    s_autosave_next_level_start = false;

    /* AB3DI.s fades TextCop down, then switches to BigFieldCop. Clear the
     * modern double-buffer history at the same boundary so the first new-level
     * frame cannot sample pixels from the previous level. */
    renderer_clear_frame_history();

    printf("[GAME] Entering main loop...\n");
}

int play_the_game_after_game_loop(GameState *state)
{
    if (state->debug_f9_need_level_reload) {
        state->debug_f9_need_level_reload = false;
        state->f9_pending_apply_save = true;
        printf("[GAME] Reloading level %d and applying save state\n",
               (int)state->current_level);
        audio_mt_end();
        io_release_level_memory(&state->level);
        return 1;
    }

    printf("[GAME] === Level ended ===\n");

    /* ---- quitGame equivalent (AB3DI.s line ~4628-4731) ---- */
    {
        /* Update energy bar one last time */
        int16_t final_energy;
        if (state->mode == MODE_SLAVE) {
            final_energy = state->plr2.energy;
        } else {
            final_energy = state->plr1.energy;
        }
        state->energy = final_energy;

        /* Stop background music */
        audio_mt_end();

        /* Win/loss follows game_loop (end zone, death, ESC). Do not treat ESC as victory. */
        if (state->finished_level == 1 && final_energy > 0) {
            printf("[GAME] Level completed successfully!\n");
            if (state->mode == MODE_SINGLE) {
                if (state->max_level < MAX_LEVELS) {
                    state->max_level = (int16_t)(state->current_level + 1);
                }
            }
            if (state->current_level >= MAX_LEVELS - 1) {
                printf("[GAME] Final level complete! %s\n", end_game_text);
                state->max_level = MAX_LEVELS - 1;
            }
        } else {
            state->finished_level = 0;
            if (final_energy <= 0) {
                printf("[GAME] Player died.\n");
            }
        }
    }
    return 0;
}

void play_the_game_finalize_session(GameState *state)
{
    /* ---- Cleanup for main menu (AB3DI.s CleanupForMainMenu ~4774) ---- */
    audio_mt_end();
    display_release_copper_screen();

    io_release_level_memory(&state->level);

    state->master_pause = false;
    state->slave_pause = false;
    state->master_quitting = false;
    state->slave_quitting = false;
    state->do_anything = false;
}

#if !defined(__EMSCRIPTEN__)
void play_the_game(GameState *state)
{
    /* Fresh session: do not carry F9 reload state from a prior play_the_game call.
     * Keep f9_pending_apply_save: startup/death autosave loads set it before
     * entering this function so prepare_level can apply the pending full save. */
    state->debug_f9_need_level_reload = false;

    bool copper_screen_ready = false;

    for (;;) {
        int show_level_text = play_the_game_should_show_level_text(state);
        if (show_level_text) {
            control_level_text_fade_in(state);
        }
        play_the_game_prepare_level(state, &copper_screen_ready);
        if (show_level_text && !control_level_text_wait_and_fade_out(state)) {
            break;
        }
        game_loop_with_hidden_start_frames(state, show_level_text ? 2 : 0);
        if (play_the_game_after_game_loop(state)) continue;
        break;
    }

    play_the_game_finalize_session(state);
}
#endif

static void control_game_over_fade_tick(float progress_0_to_1, void *userdata)
{
    (void)userdata;

    if (progress_0_to_1 < 0.0f) progress_0_to_1 = 0.0f;
    if (progress_0_to_1 > 1.0f) progress_0_to_1 = 1.0f;

    /* Fade into a strong red tint while Game Over music plays. */
    int alpha = (int)(progress_0_to_1 * 220.0f + 0.5f);
    if (alpha < 0) alpha = 0;
    if (alpha > 255) alpha = 255;
    display_set_screen_tint(255, 0, 0, alpha);
    display_present_last_frame(NULL);
}

static void control_level_complete_fade_tick(float progress_0_to_1, void *userdata)
{
    (void)userdata;

    if (progress_0_to_1 < 0.0f) progress_0_to_1 = 0.0f;
    if (progress_0_to_1 > 1.0f) progress_0_to_1 = 1.0f;

    /* Fade into bright white while level-complete music plays. */
    int alpha = (int)(progress_0_to_1 * 220.0f + 0.5f);
    if (alpha < 0) alpha = 0;
    if (alpha > 255) alpha = 255;
    display_set_screen_tint(255, 255, 255, alpha);
    display_present_last_frame(NULL);
}

static void control_setup_new_game_state(GameState *state)
{
    display_clear_screen_tint();
    game_state_setup_default(state);
    s_show_transition_level_text = false;

    state->plr1.gun_selected = 0;
    state->plr2.gun_selected = 0;

    if (state->cfg_all_weapons) {
        for (int g = 0; g < MAX_GUNS; g++) {
            state->plr1.gun_data[g].visible = -1;
            state->plr1.gun_data[g].ammo = 999 * 8;
            state->plr2.gun_data[g].visible = -1;
            state->plr2.gun_data[g].ammo = 999 * 8;
        }
    }
    player_apply_weapon_config(state);

    /* New Game starts from the configured level after the title menu.
     * INI start_level=0 means level 1; 1..MAX_LEVELS are 1-based overrides. */
    state->current_level = 0;
    state->max_level = 16;
    state->finished_level = 0;
    state->f9_pending_apply_save = false;
    state->debug_f9_need_level_reload = false;
    s_autosave_next_level_start = false;
    state->nasty = true;
    state->plr1.angpos = 0;
    state->plr2.angpos = 0;

    if (state->cfg_start_level >= 0 && state->cfg_start_level < MAX_LEVELS) {
        state->current_level = state->cfg_start_level;
        state->max_level = state->cfg_start_level;
        printf("[CONTROL] start_level=%d: starting configured level\n",
               (int)state->current_level + 1);
    } else {
        printf("[CONTROL] start_level=0: new game starts at level 1\n");
    }
}

static void control_setup_new_game_from_menu(GameState *state)
{
    control_setup_new_game_state(state);
    s_show_transition_level_text = true;
}

int play_game_front_menu_requested(void)
{
    return s_front_menu_requested ? 1 : 0;
}

int play_game_apply_front_menu_result(GameState *state, int menu_result)
{
    if (!state) return 0;

    switch (menu_result) {
    case GAME_LOOP_FRONT_MENU_NEW_GAME:
        s_front_menu_requested = false;
        control_setup_new_game_from_menu(state);
        return 1;

    case GAME_LOOP_FRONT_MENU_LOAD_AUTOSAVE:
        s_front_menu_requested = false;
        s_show_transition_level_text = false;
        s_autosave_next_level_start = false;
        display_clear_screen_tint();
        state->running = true;
        state->finished_level = 0;
        state->restart_game_requested = false;
        state->debug_f9_need_level_reload = false;
        return 1;

    case GAME_LOOP_FRONT_MENU_EXIT:
    default:
        s_front_menu_requested = false;
        return 0;
    }
}

void play_game_load_shared_assets(GameState *state)
{
    state->mode = MODE_SINGLE;

    printf("[CONTROL] PlayGame starting\n");
    /* Re-read ab3d.ini so prefs match the file on disk (exe dir / fallbacks). */
    settings_load(state);
    settings_log_recap(state);

    /* ---- Load shared assets ---- */
    io_load_walls();
#ifdef AB3D_DOS
    dos_memory_report("walls");
#endif
    io_load_floor();
#ifdef AB3D_DOS
    dos_memory_report("floor");
#endif
    io_load_sky();
    io_load_gun_graphics();
    display_upload_gun_gl_textures();
    io_load_objects();
#ifdef AB3D_DOS
    dos_memory_report("sprites");
#endif
    io_load_vec_objects();
    io_load_sfx();

    display_alloc_title_memory();
    display_setup_title_screen();
    display_load_title_screen();
    control_setup_new_game_state(state);
    s_front_menu_requested = true;

    io_load_panel();
}

#if !defined(__EMSCRIPTEN__)
int play_game_outer_should_continue(GameState *state)
{
    if (state->restart_game_requested) {
        control_setup_new_game_from_menu(state);
        printf("[CONTROL] Restarting new game from in-game menu\n");
        return 1;
    }

    if (!state->finished_level) {
        if (state->energy <= 0) {
            audio_play_module_blocking_once_with_tick("sounds/mt/GameOver.mt",
                                                      control_game_over_fade_tick,
                                                      state);
            display_clear_screen_tint();
            printf("[MUSIC] outcome: game over\n");
            s_front_menu_requested = true;
            s_show_transition_level_text = false;
            s_autosave_next_level_start = false;
            printf("[CONTROL] Returning to title menu after death\n");
            return 1;
        }
        return 0;
    }

    if (state->current_level >= MAX_LEVELS - 1) {
        audio_play_module_blocking_once_with_tick("sounds/mt/EndGame.mt",
                                                  control_level_complete_fade_tick,
                                                  state);
        display_clear_screen_tint();
        printf("[MUSIC] outcome: end game\n");
        return 0;
    }

    audio_play_module_blocking_once_with_tick("sounds/mt/WellDone.mt",
                                              control_level_complete_fade_tick,
                                              state);
    display_clear_screen_tint();
    printf("[MUSIC] outcome: well done\n");

    state->current_level++;
    s_autosave_next_level_start = true;
    s_show_transition_level_text = true;
    printf("[CONTROL] Loading next level %d (player state preserved)\n",
           (int)state->current_level);
    return 1;
}
#else
typedef enum {
    EM_OUTER_BR_NEW_GAME,
    EM_OUTER_BR_GAMEOVER,
    EM_OUTER_BR_ENDGAME,
    EM_OUTER_BR_WELLDONE,
    EM_OUTER_BR_QUIT_NO_LEVEL
} EmOuterBranch;

static EmOuterBranch s_em_outer_br;
static Uint32 s_em_fade_t0;

int play_game_outer_emscripten_begin(GameState *state)
{
    if (state->restart_game_requested) {
        s_em_outer_br = EM_OUTER_BR_NEW_GAME;
        return 0;
    }

    if (!state->finished_level) {
        if (state->energy <= 0) {
            s_em_outer_br = EM_OUTER_BR_GAMEOVER;
            if (!audio_start_one_shot_module("sounds/mt/GameOver.mt"))
                return 0;
            control_game_over_fade_tick(0.0f, state);
            s_em_fade_t0 = SDL_GetTicks();
            return 1;
        }
        s_em_outer_br = EM_OUTER_BR_QUIT_NO_LEVEL;
        return 0;
    }

    if (state->current_level >= MAX_LEVELS - 1) {
        s_em_outer_br = EM_OUTER_BR_ENDGAME;
        if (!audio_start_one_shot_module("sounds/mt/EndGame.mt"))
            return 0;
        control_level_complete_fade_tick(0.0f, state);
        s_em_fade_t0 = SDL_GetTicks();
        return 1;
    }

    s_em_outer_br = EM_OUTER_BR_WELLDONE;
    if (!audio_start_one_shot_module("sounds/mt/WellDone.mt"))
        return 0;
    control_level_complete_fade_tick(0.0f, state);
    s_em_fade_t0 = SDL_GetTicks();
    return 1;
}

int play_game_outer_emscripten_fade_frame(GameState *state)
{
    unsigned int duration_ms = audio_music_duration_ms();
    Uint32 elapsed = SDL_GetTicks() - s_em_fade_t0;
    if (elapsed >= duration_ms) {
        switch (s_em_outer_br) {
        case EM_OUTER_BR_GAMEOVER:
            control_game_over_fade_tick(1.0f, state);
            break;
        case EM_OUTER_BR_ENDGAME:
        case EM_OUTER_BR_WELLDONE:
            control_level_complete_fade_tick(1.0f, state);
            break;
        default:
            break;
        }
        audio_stop_one_shot_module();
        /* Match desktop blocking tail: one more tick(1) after playback ends */
        switch (s_em_outer_br) {
        case EM_OUTER_BR_GAMEOVER:
            control_game_over_fade_tick(1.0f, state);
            break;
        case EM_OUTER_BR_ENDGAME:
        case EM_OUTER_BR_WELLDONE:
            control_level_complete_fade_tick(1.0f, state);
            break;
        default:
            break;
        }
        return 1;
    }

    float progress = (float)elapsed / (float)duration_ms;
    if (progress > 1.0f) progress = 1.0f;
    switch (s_em_outer_br) {
    case EM_OUTER_BR_GAMEOVER:
        control_game_over_fade_tick(progress, state);
        break;
    case EM_OUTER_BR_ENDGAME:
    case EM_OUTER_BR_WELLDONE:
        control_level_complete_fade_tick(progress, state);
        break;
    default:
        break;
    }
    SDL_PumpEvents();
    return 0;
}

int play_game_outer_emscripten_finish(GameState *state)
{
    switch (s_em_outer_br) {
    case EM_OUTER_BR_NEW_GAME:
        control_setup_new_game_from_menu(state);
        printf("[CONTROL] Restarting new game from in-game menu\n");
        return 1;
    case EM_OUTER_BR_GAMEOVER:
        display_clear_screen_tint();
        printf("[MUSIC] outcome: game over\n");
        s_front_menu_requested = true;
        s_show_transition_level_text = false;
        s_autosave_next_level_start = false;
        printf("[CONTROL] Returning to title menu after death\n");
        return 1;
    case EM_OUTER_BR_ENDGAME:
        display_clear_screen_tint();
        printf("[MUSIC] outcome: end game\n");
        /* A browser cannot return to the desktop after the final fade.
         * Reuse the title-menu path, keeping the animation loop alive. */
        s_front_menu_requested = true;
        s_show_transition_level_text = false;
        s_autosave_next_level_start = false;
        printf("[CONTROL] Returning to title menu after ending\n");
        return 1;
    case EM_OUTER_BR_WELLDONE:
        display_clear_screen_tint();
        printf("[MUSIC] outcome: well done\n");
        state->current_level++;
        s_autosave_next_level_start = true;
        s_show_transition_level_text = true;
        printf("[CONTROL] Loading next level %d (player state preserved)\n",
               (int)state->current_level);
        return 1;
    case EM_OUTER_BR_QUIT_NO_LEVEL:
    default:
        return 0;
    }
}
#endif

#if !defined(__EMSCRIPTEN__)
static int control_run_front_menu(GameState *state)
{
    GameLoopCtx ctx;

    game_loop_front_menu_init(&ctx, state);
    for (;;) {
        int result = game_loop_front_menu_tick(state, &ctx);
        if (result != GAME_LOOP_FRONT_MENU_NONE)
            return result;
        SDL_Delay(16);
    }
}
#endif

/*
 * play_game - The outermost game loop
 *
 * Translated from ControlLoop.s PlayGame (~line 142).
 */
void play_game(GameState *state)
{
#ifndef __EMSCRIPTEN__
    play_game_load_shared_assets(state);

    for (;;) {
        if (s_front_menu_requested) {
            int menu_result = control_run_front_menu(state);
            if (!play_game_apply_front_menu_result(state, menu_result))
                break;
        }

        play_the_game(state);

        if (!play_game_outer_should_continue(state)) break;
    }

    display_release_panel_memory();

    printf("[CONTROL] PlayGame finished\n");
#endif
}
