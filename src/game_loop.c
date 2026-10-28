/*
 * Alien Breed 3D I - PC Port
 * game_loop.c - Main per-frame game loop
 *
 * Translated from: AB3DI.s mainLoop (~line 1296-2211)
 *
 * This is the innermost loop that runs every frame during gameplay.
 * The original code is one massive block of assembly; here we break
 * it into clearly labeled phases matching the original flow.
 *
 * Frame-rate decoupling:
 *   The Amiga game loop runs as fast as it can, measuring elapsed
 *   VBlanks (50Hz) via an interrupt counter (frames_to_draw / temp_frames).
 *   On a typical Amiga, the loop runs at ~17-25fps with temp_frames=2-3.
 *   We reproduce this by accumulating 50Hz VBlanks and only running
 *   game logic when at least GAME_TICK_VBLANKS have passed.
 *   Rendering (display_draw_display) runs every display frame (60Hz VSync)
 *   for smooth visuals.
 */

#include "game_loop.h"
#include "level.h"
#include "player.h"
#include "objects.h"
#include "game_data.h"
#include "game_types.h"
#include "visibility.h"
#include "renderer.h"
#include "display.h"
#include "input.h"
#include "audio.h"
#include "settings.h"
#ifdef AB3D_DOS
#include "platform_dos.h"
#endif
#include <SDL.h>
#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Amiga key codes used in the original */
#define KEY_PAUSE   0x19
#define KEY_RETURN  0x44
#define KEY_ESC     0x45
#define KEY_SPACE   0x40
#define KEY_UP      0x4C
#define KEY_DOWN    0x4D
#define KEY_RIGHT   0x4E
#define KEY_LEFT    0x4F

#define INGAME_MENU_SCREEN_MAIN     0
#define INGAME_MENU_SCREEN_NEW_GAME 1
#define INGAME_MENU_SCREEN_AUTOSAVES 2
#define INGAME_MENU_SCREEN_LOAD_AUTOSAVE 3
#define INGAME_MENU_SCREEN_OPTIONS  4

#define INGAME_MENU_CONTINUE    0
#define INGAME_MENU_NEW_GAME    1
#define INGAME_MENU_AUTOSAVES   2
#define INGAME_MENU_OPTIONS     3
#define INGAME_MENU_EXIT        4
#define INGAME_MENU_COUNT       5

#define INGAME_MENU_AUTOSAVE_BACK PLAYER_AUTOSAVE_SLOT_COUNT

enum {
    INGAME_OPTIONS_MENU_MOUSE_LOOK,
    INGAME_OPTIONS_MENU_INVERT_MOUSE,
    INGAME_OPTIONS_MENU_CROSSHAIR,
    INGAME_OPTIONS_MENU_GAMEPAD_TURN,
    INGAME_OPTIONS_MENU_RUN_DEFAULT,
    INGAME_OPTIONS_MENU_RUN_SPEED,
    INGAME_OPTIONS_MENU_FPS_COUNTER,
    INGAME_OPTIONS_MENU_VOLUME,
    INGAME_OPTIONS_MENU_HUD_STYLE,
    INGAME_OPTIONS_MENU_BACK,
    INGAME_OPTIONS_MENU_COUNT
};

#define INGAME_MENU_CONFIRM_YES   0
#define INGAME_MENU_CONFIRM_NO    1
#define INGAME_MENU_CONFIRM_COUNT 2

#define INGAME_MENU_DIM_ALPHA 130
#define INGAME_MENU_TITLE_DIM_ALPHA 90
#define INGAME_MENU_DISABLED_ALPHA 96
/* ControlLoop.s INTROTUNENAME -> disk/sounds/abreed3d.med;
 * AB3DI.s enables title music with ENABLETITLEMUSIC EQU 1. */
#define FRONT_MENU_TITLE_MODULE "sounds/med/ABreed3D.med"
/* Original CheckMenu is silent; these are deliberate PC menu polish using
 * original LoadFromDisk.s SFX slots: 10 switch, 11 switch1.sfx. */
#define INGAME_MENU_SFX_MOVE   10
#define INGAME_MENU_SFX_ACCEPT 11
#define INGAME_MENU_SFX_VOLUME 50
#define INGAME_MENU_VOLUME_STEP 10
#define INGAME_MENU_GAMEPAD_TURN_STEP 25
#define INGAME_MENU_GAMEPAD_TURN_MIN 25
#define INGAME_MENU_GAMEPAD_TURN_MAX 800

/* Maximum frame count before clamping */
#define MAX_TEMP_FRAMES 15

/* Minimum VBlanks to accumulate before running a game logic tick.
 * 1 VBlank = 20ms → 50Hz logic, matching the original game's simulation
 * rate. Running logic at 50Hz avoids large per-tick jumps that make the
 * player feel sluggish and can skip/tunnel through collision edges. */
#define GAME_TICK_VBLANKS 1

static int g_zone_log_initialized = 0;
static int g_zone_log_env_initialized = 0;
static int g_zone_log_enabled = 0;
static int16_t g_zone_log_last_level = -1;
static int16_t g_zone_log_last_zone[2] = { -32768, -32768 };
static int32_t g_zone_log_last_roompt[2] = { -1, -1 };

static int game_zone_log_enabled(void)
{
    if (!g_zone_log_env_initialized) {
        const char *env = getenv("AB3D_ZONE_ENTER_LOG");
        g_zone_log_enabled = (env && *env && atoi(env) > 0) ? 1 : 0;
        g_zone_log_env_initialized = 1;
    }
    return g_zone_log_enabled;
}

static void game_log_zone_changes(const GameState *state)
{
    const PlayerState *plrs[2];
    int view_player_index = 0;

    if (!state) return;
    if (!game_zone_log_enabled()) return;

    plrs[0] = &state->plr1;
    plrs[1] = &state->plr2;
    view_player_index = (state->mode == MODE_SLAVE) ? 1 : 0;

    if (!g_zone_log_initialized || state->current_level != g_zone_log_last_level) {
        g_zone_log_last_level = state->current_level;
        g_zone_log_last_zone[0] = (int16_t)-32768;
        g_zone_log_last_zone[1] = (int16_t)-32768;
        g_zone_log_last_roompt[0] = -1;
        g_zone_log_last_roompt[1] = -1;
        g_zone_log_initialized = 1;
    }

    for (int i = 0; i < 2; i++) {
        const PlayerState *plr = plrs[i];
        if (!plr) continue;
        if (plr->zone == g_zone_log_last_zone[i] && plr->roompt == g_zone_log_last_roompt[i]) continue;

        printf("[ZONEENTER] plr%d level=%d from=%d to=%d top=%d pos=(%d,%d) roompt=%ld lgr=%ld reason=%s\n",
               i + 1,
               (int)state->current_level,
               (int)g_zone_log_last_zone[i],
               (int)plr->zone,
               (int)plr->stood_in_top,
               (int)(plr->xoff >> 16),
               (int)(plr->zoff >> 16),
               (long)plr->roompt,
               (long)plr->list_of_graph_rooms,
               (g_zone_log_last_zone[i] == (int16_t)-32768) ? "init" : "change");

        if (i == view_player_index) {
            renderer_request_zone_trace();
        }

        g_zone_log_last_zone[i] = plr->zone;
        g_zone_log_last_roompt[i] = plr->roompt;
    }
}

static bool mark_visible_zones_from_graph_list(uint8_t vis_zones[256],
                                               const LevelState *level,
                                               int32_t list_offset)
{
    if (!vis_zones || !level || !level->data || list_offset < 0) {
        return false;
    }

    const uint8_t *list = level->data + list_offset;
    const uint8_t *data_end = NULL;
    if (level->data_byte_count > 0) {
        data_end = level->data + level->data_byte_count;
    }

    bool any = false;
    for (int i = 0; i < 1024; i++) {
        if (data_end && (list + 1) >= data_end) {
            break;
        }

        int16_t zone = (int16_t)((list[0] << 8) | list[1]);
        if (zone < 0) {
            break;
        }
        if (zone >= 0 && zone < 256) {
            vis_zones[(uint8_t)zone] = 1;
            any = true;
        }

        if (data_end && (list + 8) > data_end) {
            break;
        }
        list += 8;
    }

    return any;
}

/* Level-specific zone-order tweak:
 * Level 8 (1-indexed), standing in zone 49, can occasionally order zone 3
 * in front of zone 16 from a problematic viewpoint. Force 16 in front of 3.
 * Renderer draws zone_order in reverse, so lower index is closer/front. */
static void apply_zone_order_workaround_level8_zone49(GameState *state, const PlayerState *view_plr)
{
    if (!state || !view_plr) return;
    if (state->current_level != 7) return; /* level 8 (1-indexed) */
    if (view_plr->zone != 49) return;

    int count = state->zone_order_count;
    if (count < 0) return;
    if (count > 256) count = 256;

    int idx16 = -1;
    int idx3 = -1;
    for (int i = 0; i < count; i++) {
        if (state->zone_order_zones[i] == 16) idx16 = i;
        else if (state->zone_order_zones[i] == 3) idx3 = i;
    }
    if (idx16 < 0 || idx3 < 0) return;

    if (idx16 > idx3) {
        int16_t t = state->zone_order_zones[idx16];
        state->zone_order_zones[idx16] = state->zone_order_zones[idx3];
        state->zone_order_zones[idx3] = t;
    }
}

#if defined(__EMSCRIPTEN__)
EM_JS(int, game_loop_web_front_menu_audio_ready, (void), {
    if (typeof document !== 'undefined') {
        if (document.hidden) return 0;
        if (document.hasFocus && !document.hasFocus()) return 0;
    }
    if (typeof Module !== 'undefined' && Module['SDL2'] &&
        Module['SDL2'].audioContext &&
        Module['SDL2'].audioContext.state !== 'running') {
        return 0;
    }
    return 1;
});
#endif

static void game_loop_prepare_zone_order(GameState *state)
{
    if (!state) return;

    {
        PlayerState *view_plr = (state->mode == MODE_SLAVE) ?
                                &state->plr2 : &state->plr1;

        /* Amiga: ListOfGraphRooms = current room's list (roompt + 48). */
        const uint8_t *lgr_ptr = NULL;
        if (state->level.data) {
            int32_t lgr_off = -1;
            if (view_plr->roompt >= 0) {
                lgr_off = view_plr->roompt + 48;  /* ToListOfGraph */
            } else if (view_plr->list_of_graph_rooms > 0) {
                lgr_off = view_plr->list_of_graph_rooms;
            }
            if (lgr_off > 0) {
                lgr_ptr = state->level.data + lgr_off;
            }
        }

        /* Amiga uses high 16 bits of xoff/zoff for OrderZones side test
         * (move.w xoff,d2 = first word of long). */
        int32_t view_x = (int32_t)(int16_t)(view_plr->xoff >> 16);
        int32_t view_z = (int32_t)(int16_t)(view_plr->zoff >> 16);
        int32_t move_dx = (state->mode == MODE_SLAVE) ?
                          (int32_t)state->xdiff2 : (int32_t)state->xdiff1;
        int32_t move_dz = (state->mode == MODE_SLAVE) ?
                          (int32_t)state->zdiff2 : (int32_t)state->zdiff1;
        ZoneOrder zo;
        order_zones(&zo, &state->level,
                    view_x, view_z, move_dx, move_dz,
                    (int)(view_plr->angpos & 0x3FFF),
                    lgr_ptr);
        memcpy(state->zone_order_zones, zo.zones,
               (size_t)(zo.count < 256 ? zo.count : 256) * sizeof(int16_t));
        state->zone_order_count = zo.count;
        apply_zone_order_workaround_level8_zone49(state, view_plr);
        state->view_list_of_graph_rooms = lgr_ptr;
    }
}

void game_loop_ctx_init(GameLoopCtx *ctx, GameState *state)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->last_ticks = SDL_GetTicks();
    ctx->fps_sample_start_counter = SDL_GetPerformanceCounter();
    g_zone_log_initialized = 0;
    g_zone_log_last_level = -1;
    g_zone_log_last_zone[0] = (int16_t)-32768;
    g_zone_log_last_zone[1] = (int16_t)-32768;
    g_zone_log_last_roompt[0] = -1;
    g_zone_log_last_roompt[1] = -1;

    /* Allocate the previous-tick snapshot buffer used for render interpolation.
     * Initialise it to the current positions so the first frame blends cleanly. */
    if (state->level.num_object_points > 0 && !state->level.prev_object_points) {
        state->level.prev_object_points =
            (uint8_t *)calloc((size_t)state->level.num_object_points, 8u);
        if (state->level.prev_object_points && state->level.object_points) {
            memcpy(state->level.prev_object_points,
                   state->level.object_points,
                   (size_t)state->level.num_object_points * 8u);
        }
    }
    state->obj_interp_alpha = 0.0f;
}

static void game_loop_pause_timing(GameState *state, GameLoopCtx *ctx)
{
    ctx->last_ticks = SDL_GetTicks();
    ctx->pending_vblanks = 0;
    ctx->vblank_remainder_ms = 0;
    ctx->fps_sample_start_counter = SDL_GetPerformanceCounter();
    ctx->fps_frames_in_sample = 0;
    state->obj_interp_alpha = 0.0f;
}

static void game_loop_clear_queued_actions(void)
{
    (void)input_f2_pick_log_requested();
    (void)input_f5_save_requested();
    (void)input_f9_load_requested();
    (void)input_f6_gouraud_visualize_requested();
    (void)input_f7_spill_visualize_requested();
    (void)input_automap_toggle_requested();
    (void)input_automap_pgup_requested();
    (void)input_automap_pgdn_requested();
}

static void game_loop_refresh_autosave_info(GameLoopCtx *ctx)
{
    if (!ctx) return;
    ctx->ingame_menu_autosave_count = 0;
    for (int slot = 0; slot < PLAYER_AUTOSAVE_SLOT_COUNT; slot++) {
        PlayerAutosaveInfo info;
        memset(&info, 0, sizeof(info));
        ctx->ingame_menu_autosave_available[slot] = 0;
        ctx->ingame_menu_autosave_level[slot] = -1;
        ctx->ingame_menu_autosave_timestamp[slot][0] = '\0';
        if (player_read_autosave_info(slot, &info)) {
            ctx->ingame_menu_autosave_available[slot] = 1;
            ctx->ingame_menu_autosave_level[slot] = info.level;
            snprintf(ctx->ingame_menu_autosave_timestamp[slot],
                     sizeof(ctx->ingame_menu_autosave_timestamp[slot]), "%s",
                     info.timestamp);
            ctx->ingame_menu_autosave_count++;
        }
    }
}

static int game_loop_main_menu_item_visible(const GameLoopCtx *ctx, int item)
{
    if (item == INGAME_MENU_CONTINUE && ctx && ctx->ingame_menu_frontend) {
        return ctx->ingame_menu_autosave_count > 0;
    }
#if defined(__EMSCRIPTEN__)
    if (item == INGAME_MENU_EXIT) {
        return 0;
    }
#endif
    return 1;
}

static int game_loop_main_menu_item_enabled(const GameLoopCtx *ctx, int item)
{
    if (!game_loop_main_menu_item_visible(ctx, item))
        return 0;
    if (item == INGAME_MENU_AUTOSAVES) {
        return ctx && ctx->ingame_menu_autosave_count > 0;
    }
    return 1;
}

static int game_loop_latest_autosave_slot(const GameLoopCtx *ctx)
{
    if (!ctx) return -1;
    for (int slot = 0; slot < PLAYER_AUTOSAVE_SLOT_COUNT; slot++) {
        if (ctx->ingame_menu_autosave_available[slot])
            return slot;
    }
    return -1;
}

static void game_loop_set_main_menu_selection(GameLoopCtx *ctx, int preferred)
{
    int fallback;
    if (!ctx) return;
    if (preferred >= 0 && preferred < INGAME_MENU_COUNT &&
        game_loop_main_menu_item_enabled(ctx, preferred)) {
        ctx->ingame_menu_selected = preferred;
        return;
    }

    fallback = (ctx->ingame_menu_frontend &&
                !game_loop_main_menu_item_enabled(ctx, INGAME_MENU_CONTINUE)) ?
               INGAME_MENU_NEW_GAME : INGAME_MENU_CONTINUE;
    if (game_loop_main_menu_item_enabled(ctx, fallback)) {
        ctx->ingame_menu_selected = fallback;
        return;
    }

    for (int item = 0; item < INGAME_MENU_COUNT; item++) {
        if (game_loop_main_menu_item_enabled(ctx, item)) {
            ctx->ingame_menu_selected = item;
            return;
        }
    }
}

static int game_loop_move_main_menu_selection(GameLoopCtx *ctx, int step)
{
    if (!ctx || step == 0) return 0;
    int selected = ctx->ingame_menu_selected;
    for (int i = 0; i < INGAME_MENU_COUNT; i++) {
        selected = (selected + step + INGAME_MENU_COUNT) % INGAME_MENU_COUNT;
        if (game_loop_main_menu_item_enabled(ctx, selected)) {
            if (ctx->ingame_menu_selected == selected)
                return 0;
            ctx->ingame_menu_selected = selected;
            return 1;
        }
    }
    return 0;
}

static int game_loop_autosave_menu_item_enabled(const GameLoopCtx *ctx, int item)
{
    if (item == INGAME_MENU_AUTOSAVE_BACK) return 1;
    if (!ctx || item < 0 || item >= PLAYER_AUTOSAVE_SLOT_COUNT) return 0;
    return ctx->ingame_menu_autosave_available[item];
}

static void game_loop_set_autosave_menu_selection(GameLoopCtx *ctx, int preferred)
{
    if (!ctx) return;
    if (game_loop_autosave_menu_item_enabled(ctx, preferred)) {
        ctx->ingame_menu_selected = preferred;
        return;
    }
    for (int slot = 0; slot < PLAYER_AUTOSAVE_SLOT_COUNT; slot++) {
        if (game_loop_autosave_menu_item_enabled(ctx, slot)) {
            ctx->ingame_menu_selected = slot;
            return;
        }
    }
    ctx->ingame_menu_selected = INGAME_MENU_AUTOSAVE_BACK;
}

static int game_loop_move_autosave_menu_selection(GameLoopCtx *ctx, int step)
{
    int item_count = PLAYER_AUTOSAVE_SLOT_COUNT + 1;
    int selected;
    if (!ctx || step == 0) return 0;
    selected = ctx->ingame_menu_selected;
    for (int i = 0; i < item_count; i++) {
        selected = (selected + step + item_count) % item_count;
        if (game_loop_autosave_menu_item_enabled(ctx, selected)) {
            if (ctx->ingame_menu_selected == selected)
                return 0;
            ctx->ingame_menu_selected = selected;
            return 1;
        }
    }
    return 0;
}

static int game_loop_main_menu_item_at_line(const GameLoopCtx *ctx, int line)
{
    int row = 2;

    if (game_loop_main_menu_item_visible(ctx, INGAME_MENU_CONTINUE)) {
        if (line == row) return INGAME_MENU_CONTINUE;
        row++;
    }

    if (line == row) return INGAME_MENU_NEW_GAME;
    row++;

    if (line == row) {
        return game_loop_main_menu_item_enabled(ctx, INGAME_MENU_AUTOSAVES) ?
               INGAME_MENU_AUTOSAVES : -1;
    }
    row++;

    if (line == row) return INGAME_MENU_OPTIONS;
    row++;

    if (game_loop_main_menu_item_visible(ctx, INGAME_MENU_EXIT) &&
        line == row)
        return INGAME_MENU_EXIT;
    return -1;
}

static int game_loop_options_menu_item_enabled(int item)
{
    return item >= 0 && item < INGAME_OPTIONS_MENU_COUNT;
}

static void game_loop_set_options_menu_selection(GameLoopCtx *ctx, int preferred)
{
    if (!ctx) return;
    if (game_loop_options_menu_item_enabled(preferred)) {
        ctx->ingame_menu_selected = preferred;
        return;
    }
    ctx->ingame_menu_selected = INGAME_OPTIONS_MENU_MOUSE_LOOK;
}

static int game_loop_move_options_menu_selection(GameLoopCtx *ctx, int step)
{
    int selected;
    if (!ctx || step == 0) return 0;
    selected = ctx->ingame_menu_selected;
    for (int i = 0; i < INGAME_OPTIONS_MENU_COUNT; i++) {
        selected = (selected + step + INGAME_OPTIONS_MENU_COUNT) %
                   INGAME_OPTIONS_MENU_COUNT;
        if (game_loop_options_menu_item_enabled(selected)) {
            if (ctx->ingame_menu_selected == selected)
                return 0;
            ctx->ingame_menu_selected = selected;
            return 1;
        }
    }
    return 0;
}

static int game_loop_options_menu_item_at_line(int line)
{
    int item = line - 2;
    return game_loop_options_menu_item_enabled(item) ? item : -1;
}

static int game_loop_menu_item_at_line(const GameLoopCtx *ctx, int line)
{
    if (!ctx) return -1;

    if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_MAIN)
        return game_loop_main_menu_item_at_line(ctx, line);

    if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_OPTIONS)
        return game_loop_options_menu_item_at_line(line);

    if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_AUTOSAVES) {
        if (line >= 2 && line < 2 + PLAYER_AUTOSAVE_SLOT_COUNT) {
            int slot = line - 2;
            return game_loop_autosave_menu_item_enabled(ctx, slot) ? slot : -1;
        }
        if (line == 7) return INGAME_MENU_AUTOSAVE_BACK;
        return -1;
    }

    if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_NEW_GAME ||
        ctx->ingame_menu_screen == INGAME_MENU_SCREEN_LOAD_AUTOSAVE) {
        if (line == 6) return INGAME_MENU_CONFIRM_YES;
        if (line == 7) return INGAME_MENU_CONFIRM_NO;
    }

    return -1;
}

static int game_loop_move_menu_selection(GameLoopCtx *ctx, int step)
{
    int old_selected;
    if (!ctx || step == 0) return 0;
    old_selected = ctx->ingame_menu_selected;
    if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_MAIN) {
        return game_loop_move_main_menu_selection(ctx, step);
    } else if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_OPTIONS) {
        return game_loop_move_options_menu_selection(ctx, step);
    } else if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_AUTOSAVES) {
        return game_loop_move_autosave_menu_selection(ctx, step);
    } else {
        if (step < 0) {
            ctx->ingame_menu_selected =
                (ctx->ingame_menu_selected + INGAME_MENU_CONFIRM_COUNT - 1) %
                INGAME_MENU_CONFIRM_COUNT;
        } else {
            ctx->ingame_menu_selected =
                (ctx->ingame_menu_selected + 1) % INGAME_MENU_CONFIRM_COUNT;
        }
        return ctx->ingame_menu_selected != old_selected;
    }
}

static int game_loop_front_menu_music_ready(void)
{
#if defined(__EMSCRIPTEN__)
    return game_loop_web_front_menu_audio_ready();
#else
    return 1;
#endif
}

static void game_loop_start_front_menu_music(GameState *state, GameLoopCtx *ctx)
{
    (void)state;
    if (!ctx || ctx->front_menu_music_started)
        return;
    audio_load_module(FRONT_MENU_TITLE_MODULE);
    audio_init_module();
    audio_play_module();
    ctx->front_menu_music_started = 1;
}

static void game_loop_maybe_start_front_menu_music(GameState *state,
                                                   GameLoopCtx *ctx)
{
    if (game_loop_front_menu_music_ready())
        game_loop_start_front_menu_music(state, ctx);
}

static void game_loop_stop_front_menu_music(GameLoopCtx *ctx)
{
    if (!ctx)
        return;
    audio_stop_player();
    audio_unload_module();
    ctx->front_menu_music_started = 0;
}

static void game_loop_play_menu_sample(GameState *state, int sample_id)
{
    (void)state;
    /* Menu overlays can run while gameplay logic is paused, so deliberate UI
     * sounds need their own SFX frame marker instead of waiting for logic. */
    audio_begin_frame();
    audio_play_sample(sample_id, INGAME_MENU_SFX_VOLUME);
}

static void game_loop_play_menu_move_sound(GameState *state)
{
    game_loop_play_menu_sample(state, INGAME_MENU_SFX_MOVE);
}

static void game_loop_play_menu_accept_sound(GameState *state)
{
    game_loop_play_menu_sample(state, INGAME_MENU_SFX_ACCEPT);
}

static void game_loop_draw_ingame_menu(GameState *state, const GameLoopCtx *ctx);

static void game_loop_apply_volume(GameState *state)
{
    if (!state) return;
    if (state->cfg_volume < 0)
        state->cfg_volume = 0;
    else if (state->cfg_volume > 100)
        state->cfg_volume = 100;
    audio_set_master_volume((int)state->cfg_volume);
    settings_save_menu_options(state);
}

static void game_loop_adjust_volume(GameState *state,
                                    GameLoopCtx *ctx,
                                    int delta)
{
    int volume;
    if (!state || !ctx) return;
    volume = (int)state->cfg_volume + delta;
    if (volume < 0) volume = 0;
    if (volume > 100) volume = 100;
    if (volume == state->cfg_volume) return;
    state->cfg_volume = (int16_t)volume;
    game_loop_apply_volume(state);
    game_loop_play_menu_accept_sound(state);
    game_loop_pause_timing(state, ctx);
    game_loop_draw_ingame_menu(state, ctx);
}

static const char *game_loop_crosshair_label(uint8_t colour)
{
    switch (colour & 7u) {
    case 1:  return "GREEN";
    case 2:  return "MID GREEN";
    case 3:  return "YELLOW";
    case 4:  return "GREY";
    case 5:  return "RED";
    case 6:  return "ICE BLUE";
    case 7:  return "BLUE";
    case 0:
    default: return "OFF";
    }
}

static void game_loop_adjust_crosshair(GameState *state,
                                       GameLoopCtx *ctx,
                                       int delta)
{
    int colour;
    if (!state || !ctx || delta == 0) return;
    colour = (int)(state->cfg_crosshair_colour & 7u) + delta;
    while (colour < 0) colour += 8;
    while (colour > 7) colour -= 8;
    if (colour == (int)(state->cfg_crosshair_colour & 7u)) return;
    state->cfg_crosshair_colour = (uint8_t)colour;
    settings_save_menu_options(state);
    game_loop_play_menu_accept_sound(state);
    game_loop_pause_timing(state, ctx);
    game_loop_draw_ingame_menu(state, ctx);
}

static void game_loop_adjust_gamepad_turn_speed(GameState *state,
                                                GameLoopCtx *ctx,
                                                int delta,
                                                int wrap)
{
    int speed;
    if (!state || !ctx || delta == 0) return;
    speed = (int)state->cfg_gamepad_look_speed + delta;
    if (wrap) {
        if (speed > INGAME_MENU_GAMEPAD_TURN_MAX)
            speed = INGAME_MENU_GAMEPAD_TURN_MIN;
        else if (speed < INGAME_MENU_GAMEPAD_TURN_MIN)
            speed = INGAME_MENU_GAMEPAD_TURN_MAX;
    } else {
        if (speed < INGAME_MENU_GAMEPAD_TURN_MIN)
            speed = INGAME_MENU_GAMEPAD_TURN_MIN;
        if (speed > INGAME_MENU_GAMEPAD_TURN_MAX)
            speed = INGAME_MENU_GAMEPAD_TURN_MAX;
    }
    if (speed == (int)state->cfg_gamepad_look_speed) return;
    state->cfg_gamepad_look_speed = (int16_t)speed;
    settings_save_menu_options(state);
    game_loop_play_menu_accept_sound(state);
    game_loop_pause_timing(state, ctx);
    game_loop_draw_ingame_menu(state, ctx);
}

static void game_loop_adjust_run_speed(GameState *state,
                                       GameLoopCtx *ctx,
                                       int delta,
                                       int wrap)
{
    int speed;
    if (!state || !ctx || delta == 0) return;
    speed = (int)state->cfg_run_speed + delta;
    if (wrap) {
        if (speed > RUN_SPEED_PERCENT_MAX)
            speed = RUN_SPEED_PERCENT_MIN;
        else if (speed < RUN_SPEED_PERCENT_MIN)
            speed = RUN_SPEED_PERCENT_MAX;
    } else {
        if (speed < RUN_SPEED_PERCENT_MIN)
            speed = RUN_SPEED_PERCENT_MIN;
        if (speed > RUN_SPEED_PERCENT_MAX)
            speed = RUN_SPEED_PERCENT_MAX;
    }
    if (speed == (int)state->cfg_run_speed) return;
    state->cfg_run_speed = (int16_t)speed;
    settings_save_menu_options(state);
    game_loop_play_menu_accept_sound(state);
    game_loop_pause_timing(state, ctx);
    game_loop_draw_ingame_menu(state, ctx);
}

static void game_loop_present_menu(GameState *state, const GameLoopCtx *ctx)
{
    if (ctx && ctx->ingame_menu_frontend) {
        display_present_title_screen_alpha(255, INGAME_MENU_TITLE_DIM_ALPHA);
        return;
    }

    display_set_screen_tint(0, 0, 0, INGAME_MENU_DIM_ALPHA);
    display_present_last_frame(state);
    display_clear_screen_tint();
}

static void game_loop_draw_ingame_menu(GameState *state, const GameLoopCtx *ctx)
{
    char line[80];

    display_clear_text_screen();
    display_draw_line_of_text("ALIEN BREED 3D", 0);
    display_draw_line_of_text(" ", 1);

    if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_AUTOSAVES) {
        display_draw_line_of_text("AUTOSAVES", 1);
        for (int slot = 0; slot < PLAYER_AUTOSAVE_SLOT_COUNT; slot++) {
            int row = 2 + slot;
            if (ctx->ingame_menu_autosave_available[slot]) {
                snprintf(line, sizeof(line), "%c %d  LEVEL %d  %s",
                         (ctx->ingame_menu_selected == slot) ? '>' : ' ',
                         slot + 1,
                         (int)ctx->ingame_menu_autosave_level[slot] + 1,
                         ctx->ingame_menu_autosave_timestamp[slot]);
                display_draw_line_of_text(line, row);
            } else {
                snprintf(line, sizeof(line), "  %d  EMPTY", slot + 1);
                display_draw_line_of_text_alpha(line, row,
                                                INGAME_MENU_DISABLED_ALPHA);
            }
        }
        display_draw_line_of_text(
            (ctx->ingame_menu_selected == INGAME_MENU_AUTOSAVE_BACK) ?
            "> BACK" : "  BACK", 7);
        game_loop_present_menu(state, ctx);
        return;
    }

    if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_NEW_GAME ||
        ctx->ingame_menu_screen == INGAME_MENU_SCREEN_LOAD_AUTOSAVE) {
        if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_LOAD_AUTOSAVE) {
            int slot = ctx->ingame_menu_autosave_selected_slot;
            if (game_loop_autosave_menu_item_enabled(ctx, slot)) {
                snprintf(line, sizeof(line), "LOAD AUTOSAVE %d  LEVEL %d  %s",
                         slot + 1,
                         (int)ctx->ingame_menu_autosave_level[slot] + 1,
                         ctx->ingame_menu_autosave_timestamp[slot]);
                display_draw_line_of_text(line, 3);
            } else {
                display_draw_line_of_text_alpha("LOAD AUTOSAVE  NO AUTOSAVE", 3,
                                                INGAME_MENU_DISABLED_ALPHA);
            }
        } else {
            display_draw_line_of_text("NEW GAME", 2);
            display_draw_line_of_text(" ", 3);
        }
        display_draw_line_of_text(" ", 4);
        display_draw_line_of_text("ARE YOU SURE?", 5);
        display_draw_line_of_text((ctx->ingame_menu_selected == INGAME_MENU_CONFIRM_YES) ?
                                  "> YES" : "  YES", 6);
        display_draw_line_of_text((ctx->ingame_menu_selected == INGAME_MENU_CONFIRM_NO) ?
                                  "> NO" : "  NO", 7);
        game_loop_present_menu(state, ctx);
        return;
    }

    if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_OPTIONS) {
        display_draw_line_of_text("OPTIONS", 1);

        snprintf(line, sizeof(line), "%c MOUSE LOOK   : %s",
                 (ctx->ingame_menu_selected == INGAME_OPTIONS_MENU_MOUSE_LOOK) ?
                 '>' : ' ',
                 (state && state->cfg_mouse_look) ? "ON" : "OFF");
        display_draw_line_of_text(line, 2);

        snprintf(line, sizeof(line), "%c INVERT MOUSE : %s",
                 (ctx->ingame_menu_selected == INGAME_OPTIONS_MENU_INVERT_MOUSE) ?
                 '>' : ' ',
                 (state && state->cfg_mouse_look_invert_y) ? "ON" : "OFF");
        display_draw_line_of_text(line, 3);

        snprintf(line, sizeof(line), "%c CROSSHAIR    : %s",
                 (ctx->ingame_menu_selected == INGAME_OPTIONS_MENU_CROSSHAIR) ?
                 '>' : ' ',
                 state ? game_loop_crosshair_label(state->cfg_crosshair_colour) :
                 "OFF");
        display_draw_line_of_text(line, 4);

        snprintf(line, sizeof(line), "%c GAMEPAD TURN : %3d%%",
                 (ctx->ingame_menu_selected == INGAME_OPTIONS_MENU_GAMEPAD_TURN) ?
                 '>' : ' ',
                 state ? (int)state->cfg_gamepad_look_speed : 100);
        display_draw_line_of_text(line, 5);

        snprintf(line, sizeof(line), "%c RUN DEFAULT  : %s",
                 (ctx->ingame_menu_selected == INGAME_OPTIONS_MENU_RUN_DEFAULT) ?
                 '>' : ' ',
                 (state && state->cfg_run_default) ? "ON" : "OFF");
        display_draw_line_of_text(line, 6);

        snprintf(line, sizeof(line), "%c RUN SPEED    : %3d%%",
                 (ctx->ingame_menu_selected == INGAME_OPTIONS_MENU_RUN_SPEED) ?
                 '>' : ' ',
                 state ? (int)state->cfg_run_speed : RUN_SPEED_PERCENT_DEFAULT);
        display_draw_line_of_text(line, 7);

        snprintf(line, sizeof(line), "%c FPS COUNTER  : %s",
                 (ctx->ingame_menu_selected == INGAME_OPTIONS_MENU_FPS_COUNTER) ?
                 '>' : ' ',
                 (state && state->cfg_show_fps) ? "ON" : "OFF");
        display_draw_line_of_text(line, 8);

        snprintf(line, sizeof(line), "%c VOLUME       : %3d%%",
                 (ctx->ingame_menu_selected == INGAME_OPTIONS_MENU_VOLUME) ?
                 '>' : ' ',
                 state ? (int)state->cfg_volume : 100);
        display_draw_line_of_text(line, 9);

        snprintf(line, sizeof(line), "%c HUD          : %s",
                 (ctx->ingame_menu_selected == INGAME_OPTIONS_MENU_HUD_STYLE) ?
                 '>' : ' ',
                 (!state || state->cfg_hud_bars) ? "BARS" : "NUMBERS");
        display_draw_line_of_text(line, 10);

        display_draw_line_of_text(
            (ctx->ingame_menu_selected == INGAME_OPTIONS_MENU_BACK) ?
            "> BACK" : "  BACK", 11);
        game_loop_present_menu(state, ctx);
        return;
    }

    int row = 2;

    if (game_loop_main_menu_item_visible(ctx, INGAME_MENU_CONTINUE)) {
        display_draw_line_of_text((ctx->ingame_menu_selected == INGAME_MENU_CONTINUE) ?
                                  "> CONTINUE" : "  CONTINUE", row);
        row++;
    }

    display_draw_line_of_text((ctx->ingame_menu_selected == INGAME_MENU_NEW_GAME) ?
                              "> NEW GAME" : "  NEW GAME", row);
    row++;

    if (ctx->ingame_menu_autosave_count > 0) {
        snprintf(line, sizeof(line), "%c AUTOSAVES",
                 (ctx->ingame_menu_selected == INGAME_MENU_AUTOSAVES) ?
                 '>' : ' ');
        display_draw_line_of_text(line, row);
    } else {
        display_draw_line_of_text_alpha("  AUTOSAVES  NONE", row,
                                        INGAME_MENU_DISABLED_ALPHA);
    }
    row++;

    display_draw_line_of_text((ctx->ingame_menu_selected == INGAME_MENU_OPTIONS) ?
                              "> OPTIONS" : "  OPTIONS", row);
    row++;

    if (game_loop_main_menu_item_visible(ctx, INGAME_MENU_EXIT)) {
        display_draw_line_of_text((ctx->ingame_menu_selected == INGAME_MENU_EXIT) ?
                                  "> EXIT GAME" : "  EXIT GAME", row);
    }
    game_loop_present_menu(state, ctx);
}

static void game_loop_return_to_ingame_main_menu(GameState *state, GameLoopCtx *ctx,
                                                 int preferred)
{
    ctx->ingame_menu_screen = INGAME_MENU_SCREEN_MAIN;
    game_loop_refresh_autosave_info(ctx);
    game_loop_set_main_menu_selection(ctx, preferred);
    input_clear_keyboard(state->key_map);
    game_loop_pause_timing(state, ctx);
    game_loop_draw_ingame_menu(state, ctx);
}

static void game_loop_return_to_options_menu(GameState *state, GameLoopCtx *ctx,
                                             int preferred)
{
    ctx->ingame_menu_screen = INGAME_MENU_SCREEN_OPTIONS;
    game_loop_set_options_menu_selection(ctx, preferred);
    input_clear_keyboard(state->key_map);
    game_loop_pause_timing(state, ctx);
    game_loop_draw_ingame_menu(state, ctx);
}

static void game_loop_return_to_autosaves_menu(GameState *state, GameLoopCtx *ctx,
                                               int preferred_slot)
{
    ctx->ingame_menu_screen = INGAME_MENU_SCREEN_AUTOSAVES;
    game_loop_refresh_autosave_info(ctx);
    game_loop_set_autosave_menu_selection(ctx, preferred_slot);
    input_clear_keyboard(state->key_map);
    game_loop_pause_timing(state, ctx);
    game_loop_draw_ingame_menu(state, ctx);
}

static void game_loop_close_ingame_menu(GameState *state, GameLoopCtx *ctx)
{
    ctx->ingame_menu_open = 0;
    ctx->ingame_menu_frontend = 0;
    ctx->ingame_menu_screen = INGAME_MENU_SCREEN_MAIN;
    input_set_menu_mouse_active(false, state->key_map);
    display_clear_text_screen();
    input_clear_keyboard(state->key_map);
    game_loop_pause_timing(state, ctx);
}

static void game_loop_open_ingame_menu(GameState *state, GameLoopCtx *ctx)
{
    ctx->ingame_menu_open = 1;
    ctx->ingame_menu_frontend = 0;
    ctx->ingame_menu_result = GAME_LOOP_FRONT_MENU_NONE;
    ctx->ingame_menu_screen = INGAME_MENU_SCREEN_MAIN;
    game_loop_refresh_autosave_info(ctx);
    ctx->ingame_menu_autosave_selected_slot = 0;
    game_loop_set_main_menu_selection(ctx, INGAME_MENU_CONTINUE);
    input_clear_keyboard(state->key_map);
    input_set_menu_mouse_active(true, state->key_map);
    game_loop_clear_queued_actions();
    game_loop_pause_timing(state, ctx);
    game_loop_draw_ingame_menu(state, ctx);
}

static void game_loop_load_autosave(GameState *state, GameLoopCtx *ctx)
{
    int slot = ctx->ingame_menu_autosave_selected_slot;
    if (!game_loop_autosave_menu_item_enabled(ctx, slot)) {
        game_loop_return_to_autosaves_menu(state, ctx, slot);
        return;
    }

    switch (player_load_autosave_from_file(state, slot)) {
    case PLAYER_SAVE_LOAD_APPLIED:
        if (ctx->ingame_menu_frontend) {
            input_set_menu_mouse_active(false, state->key_map);
            display_clear_text_screen();
            input_clear_keyboard(state->key_map);
            ctx->ingame_menu_open = 0;
            ctx->ingame_menu_result = GAME_LOOP_FRONT_MENU_LOAD_AUTOSAVE;
        } else {
            game_loop_close_ingame_menu(state, ctx);
        }
        return;

    case PLAYER_SAVE_LOAD_NEED_LEVEL_RELOAD:
        input_set_menu_mouse_active(false, state->key_map);
        display_clear_text_screen();
        input_clear_keyboard(state->key_map);
        state->f9_pending_apply_save = true;
        state->debug_f9_need_level_reload = ctx->ingame_menu_frontend ? false : true;
        if (ctx->ingame_menu_frontend) {
            ctx->ingame_menu_open = 0;
            ctx->ingame_menu_result = GAME_LOOP_FRONT_MENU_LOAD_AUTOSAVE;
        } else {
            state->running = false;
        }
        return;

    case PLAYER_SAVE_LOAD_FAILED:
    default:
        printf("[PLAYER] autosave load: slot %d load failed\n", slot + 1);
        game_loop_return_to_autosaves_menu(state, ctx, slot);
        return;
    }
}

static void game_loop_select_confirm_item(GameState *state, GameLoopCtx *ctx)
{
    switch (ctx->ingame_menu_selected) {
    case INGAME_MENU_CONFIRM_YES:
        if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_LOAD_AUTOSAVE) {
            game_loop_load_autosave(state, ctx);
        } else {
            input_set_menu_mouse_active(false, state->key_map);
            display_clear_text_screen();
            input_clear_keyboard(state->key_map);
            if (ctx->ingame_menu_frontend) {
                ctx->ingame_menu_open = 0;
                ctx->ingame_menu_result = GAME_LOOP_FRONT_MENU_NEW_GAME;
            } else {
                state->restart_game_requested = true;
                state->finished_level = 0;
                state->running = false;
            }
        }
        return;

    case INGAME_MENU_CONFIRM_NO:
    default:
        if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_LOAD_AUTOSAVE) {
            game_loop_return_to_autosaves_menu(
                state, ctx, ctx->ingame_menu_autosave_selected_slot);
        } else {
            game_loop_return_to_ingame_main_menu(
                state, ctx, INGAME_MENU_NEW_GAME);
        }
        return;
    }
}

static void game_loop_select_options_menu_item(GameState *state, GameLoopCtx *ctx)
{
    switch (ctx->ingame_menu_selected) {
    case INGAME_OPTIONS_MENU_MOUSE_LOOK:
        state->cfg_mouse_look = !state->cfg_mouse_look;
        if (!state->cfg_mouse_look) {
            player_clear_mouse_look_aim_state(state);
        }
        settings_save_menu_options(state);
        break;

    case INGAME_OPTIONS_MENU_INVERT_MOUSE:
        state->cfg_mouse_look_invert_y = !state->cfg_mouse_look_invert_y;
        settings_save_menu_options(state);
        break;

    case INGAME_OPTIONS_MENU_CROSSHAIR:
        state->cfg_crosshair_colour =
            (uint8_t)(((int)(state->cfg_crosshair_colour & 7u) + 1) & 7);
        settings_save_menu_options(state);
        break;

    case INGAME_OPTIONS_MENU_GAMEPAD_TURN:
        {
            int speed = (int)state->cfg_gamepad_look_speed +
                        INGAME_MENU_GAMEPAD_TURN_STEP;
            if (speed > INGAME_MENU_GAMEPAD_TURN_MAX)
                speed = INGAME_MENU_GAMEPAD_TURN_MIN;
            state->cfg_gamepad_look_speed = (int16_t)speed;
        }
        settings_save_menu_options(state);
        break;

    case INGAME_OPTIONS_MENU_RUN_SPEED:
        game_loop_adjust_run_speed(state, ctx, RUN_SPEED_PERCENT_STEP, 1);
        return;

    case INGAME_OPTIONS_MENU_RUN_DEFAULT:
        state->cfg_run_default = !state->cfg_run_default;
        settings_save_menu_options(state);
        break;

    case INGAME_OPTIONS_MENU_HUD_STYLE:
        state->cfg_hud_bars = !state->cfg_hud_bars;
        settings_save_menu_options(state);
        break;

    case INGAME_OPTIONS_MENU_FPS_COUNTER:
        state->cfg_show_fps = !state->cfg_show_fps;
        settings_save_menu_options(state);
        break;

    case INGAME_OPTIONS_MENU_VOLUME:
        {
            int volume = (int)state->cfg_volume + INGAME_MENU_VOLUME_STEP;
            state->cfg_volume = (int16_t)((volume > 100) ? 0 : volume);
        }
        game_loop_apply_volume(state);
        break;

    case INGAME_OPTIONS_MENU_BACK:
        game_loop_return_to_ingame_main_menu(state, ctx, INGAME_MENU_OPTIONS);
        return;

    default:
        break;
    }

    game_loop_pause_timing(state, ctx);
    game_loop_draw_ingame_menu(state, ctx);
}

static void game_loop_select_ingame_menu_item(GameState *state, GameLoopCtx *ctx)
{
    if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_NEW_GAME ||
        ctx->ingame_menu_screen == INGAME_MENU_SCREEN_LOAD_AUTOSAVE) {
        game_loop_select_confirm_item(state, ctx);
        return;
    }
    if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_OPTIONS) {
        game_loop_select_options_menu_item(state, ctx);
        return;
    }
    if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_AUTOSAVES) {
        if (ctx->ingame_menu_selected == INGAME_MENU_AUTOSAVE_BACK) {
            game_loop_return_to_ingame_main_menu(state, ctx,
                                                INGAME_MENU_AUTOSAVES);
            return;
        }
        if (game_loop_autosave_menu_item_enabled(ctx,
                                                 ctx->ingame_menu_selected)) {
            ctx->ingame_menu_autosave_selected_slot =
                ctx->ingame_menu_selected;
            ctx->ingame_menu_screen = INGAME_MENU_SCREEN_LOAD_AUTOSAVE;
            ctx->ingame_menu_selected = INGAME_MENU_CONFIRM_NO;
        }
        game_loop_pause_timing(state, ctx);
        game_loop_draw_ingame_menu(state, ctx);
        return;
    }

    switch (ctx->ingame_menu_selected) {
    case INGAME_MENU_CONTINUE:
        if (ctx->ingame_menu_frontend) {
            int slot = game_loop_latest_autosave_slot(ctx);
            if (slot < 0) break;
            ctx->ingame_menu_autosave_selected_slot = slot;
            game_loop_load_autosave(state, ctx);
            return;
        }
        game_loop_close_ingame_menu(state, ctx);
        return;

    case INGAME_MENU_AUTOSAVES:
        if (ctx->ingame_menu_autosave_count <= 0) {
            break;
        }
        ctx->ingame_menu_screen = INGAME_MENU_SCREEN_AUTOSAVES;
        game_loop_set_autosave_menu_selection(ctx, 0);
        break;

    case INGAME_MENU_NEW_GAME:
        ctx->ingame_menu_screen = INGAME_MENU_SCREEN_NEW_GAME;
        ctx->ingame_menu_selected = INGAME_MENU_CONFIRM_NO;
        break;

    case INGAME_MENU_OPTIONS:
        game_loop_return_to_options_menu(state, ctx,
                                         INGAME_OPTIONS_MENU_MOUSE_LOOK);
        return;

    case INGAME_MENU_EXIT:
        input_set_menu_mouse_active(false, state->key_map);
        display_clear_text_screen();
        input_clear_keyboard(state->key_map);
        if (ctx->ingame_menu_frontend) {
            ctx->ingame_menu_open = 0;
            ctx->ingame_menu_result = GAME_LOOP_FRONT_MENU_EXIT;
        } else {
            state->finished_level = 0;
            state->running = false;
        }
        return;

    default:
        break;
    }

    game_loop_pause_timing(state, ctx);
    game_loop_draw_ingame_menu(state, ctx);
}

static int game_loop_update_ingame_menu_mouse(GameState *state, GameLoopCtx *ctx)
{
    MenuMouseState mouse;
    int text_line = -1;
    int item = -1;
    int have_mouse_event;

    input_read_menu_mouse(&mouse);
    have_mouse_event = mouse.moved || mouse.left_pressed ||
                       mouse.right_pressed || mouse.wheel_y != 0;
    if (!have_mouse_event)
        return 0;

    if (mouse.wheel_y > 0) {
        if (game_loop_move_menu_selection(ctx, -1))
            game_loop_play_menu_move_sound(state);
    } else if (mouse.wheel_y < 0) {
        if (game_loop_move_menu_selection(ctx, 1))
            game_loop_play_menu_move_sound(state);
    }

    if ((mouse.moved || mouse.left_pressed) && mouse.valid &&
        display_text_line_at_point(mouse.x, mouse.y,
                                   ctx->ingame_menu_frontend, &text_line)) {
        item = game_loop_menu_item_at_line(ctx, text_line);
        if (item >= 0) {
            if (ctx->ingame_menu_selected != item) {
                ctx->ingame_menu_selected = item;
                game_loop_play_menu_move_sound(state);
            }
            if (mouse.left_pressed) {
                game_loop_play_menu_accept_sound(state);
                input_consume_menu_mouse_events();
                game_loop_clear_queued_actions();
                game_loop_select_ingame_menu_item(state, ctx);
                return 1;
            }
        }
    }

    input_consume_menu_mouse_events();
    return 0;
}

static int game_loop_update_ingame_menu(GameState *state, GameLoopCtx *ctx)
{
    if (input_consume_key_press(KEY_ESC)) {
        int play_back_sound = 1;
        game_loop_clear_queued_actions();
        if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_LOAD_AUTOSAVE) {
            game_loop_return_to_autosaves_menu(
                state, ctx, ctx->ingame_menu_autosave_selected_slot);
        } else if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_AUTOSAVES) {
            game_loop_return_to_ingame_main_menu(state, ctx,
                                                INGAME_MENU_AUTOSAVES);
        } else if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_OPTIONS) {
            game_loop_return_to_ingame_main_menu(state, ctx,
                                                INGAME_MENU_OPTIONS);
        } else if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_NEW_GAME) {
            game_loop_return_to_ingame_main_menu(state, ctx,
                                                INGAME_MENU_NEW_GAME);
        } else if (ctx->ingame_menu_frontend) {
            game_loop_draw_ingame_menu(state, ctx);
            play_back_sound = 0;
        } else {
            game_loop_close_ingame_menu(state, ctx);
        }
        if (play_back_sound)
            game_loop_play_menu_accept_sound(state);
        return 1;
    }
    if (input_key_pressed(state->key_map, KEY_ESC)) {
        int play_back_sound = 1;
        game_loop_clear_queued_actions();
        if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_LOAD_AUTOSAVE) {
            game_loop_return_to_autosaves_menu(
                state, ctx, ctx->ingame_menu_autosave_selected_slot);
        } else if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_AUTOSAVES) {
            game_loop_return_to_ingame_main_menu(state, ctx,
                                                INGAME_MENU_AUTOSAVES);
        } else if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_OPTIONS) {
            game_loop_return_to_ingame_main_menu(state, ctx,
                                                INGAME_MENU_OPTIONS);
        } else if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_NEW_GAME) {
            game_loop_return_to_ingame_main_menu(state, ctx,
                                                INGAME_MENU_NEW_GAME);
        } else if (ctx->ingame_menu_frontend) {
            game_loop_draw_ingame_menu(state, ctx);
            play_back_sound = 0;
        } else {
            input_set_menu_mouse_active(false, state->key_map);
            display_clear_text_screen();
            input_clear_keyboard(state->key_map);
            state->finished_level = 0;
            state->running = false;
        }
        if (play_back_sound)
            game_loop_play_menu_accept_sound(state);
        return 1;
    }

    bool menu_up = input_consume_key_press(KEY_UP);
    bool menu_down = input_consume_key_press(KEY_DOWN);
    if (menu_up && game_loop_move_menu_selection(ctx, -1))
        game_loop_play_menu_move_sound(state);
    if (menu_down && game_loop_move_menu_selection(ctx, 1))
        game_loop_play_menu_move_sound(state);

    bool menu_left = input_consume_key_press(KEY_LEFT);
    bool menu_right = input_consume_key_press(KEY_RIGHT);
    if (ctx->ingame_menu_screen == INGAME_MENU_SCREEN_OPTIONS &&
        (menu_left || menu_right)) {
        int delta = 0;
        switch (ctx->ingame_menu_selected) {
        case INGAME_OPTIONS_MENU_HUD_STYLE:
            game_loop_select_options_menu_item(state, ctx);
            game_loop_clear_queued_actions();
            return 1;

        case INGAME_OPTIONS_MENU_CROSSHAIR:
            if (menu_left) delta -= 1;
            if (menu_right) delta += 1;
            if (delta != 0)
                game_loop_adjust_crosshair(state, ctx, delta);
            game_loop_clear_queued_actions();
            return 1;

        case INGAME_OPTIONS_MENU_RUN_SPEED:
            if (menu_left) delta -= RUN_SPEED_PERCENT_STEP;
            if (menu_right) delta += RUN_SPEED_PERCENT_STEP;
            if (delta != 0)
                game_loop_adjust_run_speed(state, ctx, delta, 0);
            game_loop_clear_queued_actions();
            return 1;

        case INGAME_OPTIONS_MENU_GAMEPAD_TURN:
            if (menu_left) delta -= INGAME_MENU_GAMEPAD_TURN_STEP;
            if (menu_right) delta += INGAME_MENU_GAMEPAD_TURN_STEP;
            if (delta != 0)
                game_loop_adjust_gamepad_turn_speed(state, ctx, delta, 0);
            game_loop_clear_queued_actions();
            return 1;

        case INGAME_OPTIONS_MENU_VOLUME:
            if (menu_left) delta -= INGAME_MENU_VOLUME_STEP;
            if (menu_right) delta += INGAME_MENU_VOLUME_STEP;
            if (delta != 0)
                game_loop_adjust_volume(state, ctx, delta);
            game_loop_clear_queued_actions();
            return 1;

        default:
            break;
        }

        game_loop_clear_queued_actions();
    }

    if (game_loop_update_ingame_menu_mouse(state, ctx))
        return 1;

    bool menu_accept = input_consume_key_press(KEY_RETURN);
    menu_accept = input_consume_key_press(KEY_SPACE) || menu_accept;
    if (menu_accept) {
        game_loop_play_menu_accept_sound(state);
        game_loop_clear_queued_actions();
        game_loop_select_ingame_menu_item(state, ctx);
        return 1;
    }

    game_loop_clear_queued_actions();
    game_loop_pause_timing(state, ctx);
    game_loop_draw_ingame_menu(state, ctx);
    return 1;
}

void game_loop_front_menu_init(GameLoopCtx *ctx, GameState *state)
{
    if (!ctx || !state) return;

    memset(ctx, 0, sizeof(*ctx));
    ctx->last_ticks = SDL_GetTicks();
    ctx->fps_sample_start_counter = SDL_GetPerformanceCounter();
    ctx->ingame_menu_open = 1;
    ctx->ingame_menu_frontend = 1;
    ctx->ingame_menu_result = GAME_LOOP_FRONT_MENU_NONE;
    ctx->ingame_menu_screen = INGAME_MENU_SCREEN_MAIN;
    ctx->ingame_menu_autosave_selected_slot = 0;

    display_setup_title_screen();
    game_loop_refresh_autosave_info(ctx);
    game_loop_set_main_menu_selection(
        ctx,
        (ctx->ingame_menu_autosave_count > 0) ?
        INGAME_MENU_CONTINUE : INGAME_MENU_NEW_GAME);
    input_clear_keyboard(state->key_map);
    game_loop_clear_queued_actions();
    game_loop_pause_timing(state, ctx);
    input_set_menu_mouse_active(true, state->key_map);
    game_loop_maybe_start_front_menu_music(state, ctx);
    game_loop_draw_ingame_menu(state, ctx);
}

int game_loop_front_menu_tick(GameState *state, GameLoopCtx *ctx)
{
    if (!state || !ctx) return GAME_LOOP_FRONT_MENU_EXIT;
    if (ctx->ingame_menu_result != GAME_LOOP_FRONT_MENU_NONE)
        return ctx->ingame_menu_result;

    input_update(state->key_map, &state->last_pressed);
#if defined(__EMSCRIPTEN__)
    display_emscripten_frame_resize_poll();
#endif
    if (input_fullscreen_toggle_requested())
        display_toggle_fullscreen();
    game_loop_maybe_start_front_menu_music(state, ctx);

    if (input_quit_requested()) {
        input_set_menu_mouse_active(false, state->key_map);
        ctx->ingame_menu_open = 0;
        ctx->ingame_menu_result = GAME_LOOP_FRONT_MENU_EXIT;
        game_loop_stop_front_menu_music(ctx);
        return ctx->ingame_menu_result;
    }

    if (ctx->ingame_menu_open)
        game_loop_update_ingame_menu(state, ctx);

    if (ctx->ingame_menu_result != GAME_LOOP_FRONT_MENU_NONE)
        game_loop_stop_front_menu_music(ctx);

    return ctx->ingame_menu_result;
}

/*
 * game_loop_tick - One display frame of the main game loop (Emscripten calls this
 * from the browser main loop; native game_loop wraps it in a while).
 */
void game_loop_tick(GameState *state, GameLoopCtx *ctx)
{
    if (!state->running) return;

    game_log_zone_changes(state);

    /* ================================================================
     * Always: Poll input every display frame for responsiveness
     * ================================================================ */
        input_update(state->key_map, &state->last_pressed);
#if defined(__EMSCRIPTEN__)
        display_emscripten_frame_resize_poll();
#endif
        if (input_fullscreen_toggle_requested())
            display_toggle_fullscreen();

        if (ctx->ingame_menu_open) {
            game_loop_update_ingame_menu(state, ctx);
            return;
        }
        if (input_consume_key_press(KEY_ESC)) {
            game_loop_open_ingame_menu(state, ctx);
            return;
        }

        bool f2_pick_log_requested = input_f2_pick_log_requested();
        bool f5_save_requested = input_f5_save_requested();
        bool f9_load_requested = input_f9_load_requested();
        if (state->cfg_quicksave_load && f5_save_requested)
            player_save_position(state);
        if (state->cfg_quicksave_load && f9_load_requested) {
            switch (player_load_save_from_file(state)) {
            case PLAYER_SAVE_LOAD_APPLIED:
                break;
            case PLAYER_SAVE_LOAD_NEED_LEVEL_RELOAD:
                state->debug_f9_need_level_reload = true;
                state->running = false;
                break;
            case PLAYER_SAVE_LOAD_FAILED:
                printf("[PLAYER] F9: no savegame.bin or load failed\n");
                break;
            }
        }
        if (input_f6_gouraud_visualize_requested())
            renderer_toggle_floor_gouraud_debug_view();
        if (input_f7_spill_visualize_requested())
            renderer_toggle_spill_visualize_debug_view();
        if (input_automap_toggle_requested())
            state->automap_visible = !state->automap_visible;
        if (input_automap_pgup_requested())
            renderer_automap_adjust_scale(-1); /* zoom in: fewer world units per pixel */
        if (input_automap_pgdn_requested())
            renderer_automap_adjust_scale(1);  /* zoom out */

        /* ================================================================
         * Frame timing: accumulate 50Hz VBlanks from real elapsed time
         * ================================================================ */
        {
            Uint32 now = SDL_GetTicks();
#ifdef AB3D_DOS
            if(dos_fixed_test_ticks)now=ctx->last_ticks+20;
#endif
            state->current_ticks_ms = (uint32_t)now;
            Uint32 elapsed = now - ctx->last_ticks;
            ctx->last_ticks = now;

            /* Clamp to prevent spiral-of-death after alt-tab etc. */
            if (elapsed > 200) elapsed = 200;

            /* Accumulate VBlanks at 50Hz (20ms per VBlank) */
            ctx->vblank_remainder_ms += elapsed;
            ctx->pending_vblanks += (int)(ctx->vblank_remainder_ms / 20);
            ctx->vblank_remainder_ms %= 20;
            if (ctx->hidden_present_frames > 0 &&
                ctx->pending_vblanks < GAME_TICK_VBLANKS) {
                ctx->pending_vblanks = GAME_TICK_VBLANKS;
                ctx->vblank_remainder_ms = 0;
            }
        }

        /* Track how far we are into the current 50Hz tick (0=just ticked, 1=about to tick).
         * Used by the renderer to interpolate enemy positions between logic ticks. */
        state->obj_interp_alpha = (float)ctx->vblank_remainder_ms / 20.0f;
        if (state->obj_interp_alpha > 1.0f) state->obj_interp_alpha = 1.0f;

        /* ================================================================
         * Game logic: Amiga-style cadence.
         *
         * FramesToDraw is accumulated by VBlank interrupt, then TempFrames
         * is latched once per main-loop iteration (clamped) and a single
         * ObjMoveAnim-style logic pass runs with that TempFrames value.
         * ================================================================ */
        if (ctx->pending_vblanks >= GAME_TICK_VBLANKS) {
            int ticks = ctx->pending_vblanks;
            if (ticks > MAX_TEMP_FRAMES) ticks = MAX_TEMP_FRAMES;
            ctx->pending_vblanks = 0;

            state->frames_to_draw = (int16_t)ticks;
            state->temp_frames = (int16_t)ticks;
            audio_begin_frame();

            /* Water waves are an Amiga draw-loop visual animation. Keep them
             * on latched vblank cadence, not display-refresh cadence. */
            renderer_step_water_anim_vblanks(ticks);

            /* ---- Phase 1: Pause handling ---- */
            if (state->mode == MODE_SINGLE) {
                if (input_key_pressed(state->key_map, KEY_PAUSE)) {
                    state->do_anything = false;
                    state->do_anything = true;
                }
            }

            /* ---- Phase 2: Set READCONTROLS flag ---- */
            state->read_controls = true;

            /* ---- Phase 3: Hit flash fadedown ---- */
            if (state->hitcol > 0) {
                state->hitcol -= ticks * 0x100;
                if (state->hitcol < 0) state->hitcol = 0;
            }
            if (state->hitcol2 > 0) {
                state->hitcol2 -= ticks * 0x100;
                if (state->hitcol2 < 0) state->hitcol2 = 0;
            }

            /* ---- Phase 3b: Gun animation frame countdown ---- */
            if (state->plr1.gun_frame > 0) {
                state->plr1.gun_frame -= state->temp_frames;
                if (state->plr1.gun_frame < 0) state->plr1.gun_frame = 0;
            }
            if (state->plr2.gun_frame > 0) {
                state->plr2.gun_frame -= state->temp_frames;
                if (state->plr2.gun_frame < 0) state->plr2.gun_frame = 0;
            }

            /* ---- Phase 6: Gun selection / ammo ---- */
            {
                PlayerState *gun_plr;
                if (state->mode == MODE_SLAVE) {
                    gun_plr = &state->plr2;
                } else {
                    gun_plr = &state->plr1;
                }

                int gun_idx = gun_plr->gun_selected;
                if (gun_idx >= 0 && gun_idx < MAX_GUNS) {
                    state->ammo = *player_weapon_ammo_ptr(gun_plr, gun_idx) >> 3;
                }
            }

            /* ---- Phase 7: Save old positions ---- */
            int32_t old_x1 = state->plr1.xoff;
            int32_t old_z1 = state->plr1.zoff;
            int32_t old_x2 = state->plr2.xoff;
            int32_t old_z2 = state->plr2.zoff;

            /* ---- Phase 9: Player control ----
             * Snapshotting now happens inside player*_control after
             * simulation/fall and before full collision resolution. */
            if (state->mode == MODE_SINGLE) {
                state->energy = state->plr1.energy;
                player1_control(state);
            } else if (state->mode == MODE_MASTER) {
                state->energy = state->plr1.energy;
                player1_control(state);
                player2_control(state);
            } else if (state->mode == MODE_SLAVE) {
                state->energy = state->plr2.energy;
                player1_control(state);
                player2_control(state);
            }
            input_consume_mouse_deltas();

            /* ---- Phase 10: Zone brightness animation (bright_anim_values updated; rendering reads from zone data) ---- */
            bright_anim_handler(state);

            /* ---- Phase 11: Original multiplayer CanItBeSeen path is absent in this single-view runtime. ---- */

            /* ---- Phase 12: Fire holddown tracking ---- */
            {
                int16_t tf = state->temp_frames;

                state->plr1.p_holddown += tf;
                if (state->plr1.p_holddown > 30) state->plr1.p_holddown = 30;
                if (!state->plr1.p_fire) {
                    state->plr1.p_holddown -= tf;
                    if (state->plr1.p_holddown < 0) state->plr1.p_holddown = 0;
                }

                state->plr2.p_holddown += tf;
                if (state->plr2.p_holddown > 30) state->plr2.p_holddown = 30;
                if (!state->plr2.p_fire) {
                    state->plr2.p_holddown -= tf;
                    if (state->plr2.p_holddown < 0) state->plr2.p_holddown = 0;
                }
            }

            /* ---- Phase 13: Position diff ---- */
            {
                int16_t tf = state->temp_frames;
                if (tf < 1) tf = 1;

                int16_t dx1 = (int16_t)((state->plr1.xoff >> 16) - (old_x1 >> 16));
                int16_t dz1 = (int16_t)((state->plr1.zoff >> 16) - (old_z1 >> 16));
                int16_t dx2 = (int16_t)((state->plr2.xoff >> 16) - (old_x2 >> 16));
                int16_t dz2 = (int16_t)((state->plr2.zoff >> 16) - (old_z2 >> 16));
                state->xdiff1 = (int16_t)(((int32_t)dx1 << 4) / tf);
                state->zdiff1 = (int16_t)(((int32_t)dz1 << 4) / tf);
                state->xdiff2 = (int16_t)(((int32_t)dx2 << 4) / tf);
                state->zdiff2 = (int16_t)(((int32_t)dz2 << 4) / tf);
            }

            /* ---- Phase 14: Shooting, objects ---- */
            player1_shoot(state);
            if (state->mode != MODE_SINGLE) {
                player2_shoot(state);
            }

            use_player1(state);
            if (state->mode != MODE_SINGLE) {
                use_player2(state);
            }

            /* Zone ordering for rendering */
            game_loop_prepare_zone_order(state);

            game_log_zone_changes(state);

            /* Snapshot object positions before they are mutated by objects_update.
             * The renderer uses prev_object_points + obj_interp_alpha to draw smooth
             * movement at display frame rate even though logic runs at 50 Hz. */
            if (state->level.prev_object_points && state->level.object_points &&
                state->level.num_object_points > 0) {
                memcpy(state->level.prev_object_points,
                       state->level.object_points,
                       (size_t)state->level.num_object_points * 8u);
            }

            objects_update(state);
            explosion_advance(state);

            /* ---- Phase 15: Object worry flags ---- */
            if (state->level.object_data) {
                uint8_t vis_zones[256];
                memset(vis_zones, 0, sizeof(vis_zones));
                bool has_visible = false;

                /* Amiga builds WorkSpace from each player's ToListOfGraph list,
                 * then wakes objects whose objZone bit is set in WorkSpace. */
                if (state->mode != MODE_SINGLE) {
                    int32_t lgr2 = state->plr2.roompt;
                    if (lgr2 >= 0) lgr2 += 48;  /* ToListOfGraph */
                    else lgr2 = state->plr2.list_of_graph_rooms;
                    has_visible |= mark_visible_zones_from_graph_list(
                        vis_zones, &state->level, lgr2);
                }
                int32_t lgr1 = state->plr1.roompt;
                if (lgr1 >= 0) lgr1 += 48;  /* ToListOfGraph */
                else lgr1 = state->plr1.list_of_graph_rooms;
                has_visible |= mark_visible_zones_from_graph_list(
                    vis_zones, &state->level, lgr1);

                if (!has_visible) {
                    /* Match Amiga behavior: no visible-zone bits -> no wake this tick. */
                }

                int obj_idx = 0;
                while (1) {
                    GameObject *wobj = (GameObject*)(state->level.object_data +
                                        obj_idx * OBJECT_SIZE);
                    if (OBJ_CID(wobj) < 0) break;

                    if (OBJ_ZONE(wobj) >= 0 && OBJ_ZONE(wobj) < 256) {
                        if (vis_zones[OBJ_ZONE(wobj)]) {
                            /* Amiga: or.b #127,objWorry(a0) */
                            wobj->obj.worry |= 127;
                        }
                    }
                    obj_idx++;
                }
            }

            if (state->infinite_health) {
                state->plr1.energy = PLAYER_MAX_ENERGY;
                state->plr2.energy = PLAYER_MAX_ENERGY;
            }

            ctx->logic_count++;

        } /* end if (run_logic) */

        /* ================================================================
         * Always: Update sprite rotation frames every display frame so
         * sprites animate (face the camera) even when the player hasn't moved.
         * ================================================================ */
        objects_update_sprite_frames(state);

        /* ================================================================
         * Always: Render every display frame (60Hz VSync) for smooth output
         * ================================================================ */
        if (f2_pick_log_requested) {
            renderer_request_center_pick_capture();
        }
        bool hidden_start_frame = (ctx->hidden_present_frames > 0);
        if (hidden_start_frame) {
            display_suppress_next_present();
            ctx->hidden_present_frames--;
        }
        display_draw_display(state);
        audio_set_underwater(renderer_get_last_fill_screen_water() != 0);
        if (hidden_start_frame && ctx->hidden_present_frames <= 0) {
            game_loop_pause_timing(state, ctx);
        }
        if (f2_pick_log_requested) {
            PlayerState *view_plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
            int16_t looking_zone = -1;
            renderer_get_center_pick(&looking_zone, NULL);
            renderer_log_f2_pick_debug(state, view_plr->zone, looking_zone);
            level_log_zones(&state->level);
            level_log_player_zone_full(state);
            level_log_zone_full(&state->level, looking_zone, "looking");
        }

        ctx->fps_frames_in_sample++;
        {
            Uint64 fps_now = SDL_GetPerformanceCounter();
            Uint64 fps_freq = SDL_GetPerformanceFrequency();
            Uint64 fps_dt = fps_now - ctx->fps_sample_start_counter;
            Uint64 fps_update_dt = fps_freq / 4u; /* 250ms updates with high-res timing. */
            if (fps_update_dt == 0) fps_update_dt = 1;
            if (fps_freq > 0 && fps_dt >= fps_update_dt) {
                double sec = (double)fps_dt / (double)fps_freq;
                double fps = (double)ctx->fps_frames_in_sample / sec;
                int rounded = (int)(fps + 0.5);
                if (rounded < 0) rounded = 0;
                if (rounded > 9999) rounded = 9999;
                state->fps_display = (uint16_t)rounded;
                ctx->fps_sample_start_counter = fps_now;
                ctx->fps_frames_in_sample = 0;
            }
        }

        /* ================================================================
         * Always: Quit / death / level-end checks
         * ================================================================ */
        if (input_key_pressed(state->key_map, KEY_ESC)) {
            if (state->mode == MODE_SINGLE) {
                printf("[LOOP] ESC pressed - exiting to menu\n");
                state->finished_level = 0;
                state->running = false;
            } else if (state->mode == MODE_SLAVE) {
                state->slave_quitting = true;
            } else if (state->mode == MODE_MASTER) {
                state->master_quitting = true;
            }
        }

        if (state->master_quitting && state->slave_quitting) {
            state->running = false;
        }

        if (state->mode == MODE_SINGLE || state->mp_mode == 1) {
            int16_t end_zone = end_zones[state->current_level & 0x0F];
            if (state->plr1.zone == end_zone) {
                printf("[LOOP] Player 1 reached end zone %d - level complete!\n",
                       end_zone);
                state->finished_level = 1;
                if (state->current_level < MAX_LEVELS - 1) {
                    state->max_level = state->current_level + 1;
                }
                state->running = false;
            }
        }

        if (state->plr1.energy <= 0) {
            state->finished_level = 0;
            state->running = false;
        }
        if (state->plr2.energy <= 0 && state->mode != MODE_SINGLE) {
            state->finished_level = 0;
            state->running = false;
        }

        /* ================================================================
         * Frame counter
         * ================================================================ */
        ctx->frame_count++;

}

/*
 * game_loop - The main per-frame game loop
 *
 * Runs until one of:
 *   - ESC pressed (quit to menu)
 *   - Player energy <= 0 (death)
 *   - Player reaches end zone (level complete)
 *   - Both players quitting (multiplayer)
 */
void game_loop_with_hidden_start_frames(GameState *state, int hidden_frames)
{
    GameLoopCtx ctx;
    game_loop_ctx_init(&ctx, state);
    if (hidden_frames > 0)
        ctx.hidden_present_frames = hidden_frames;
    while (state->running) {
        game_loop_tick(state, &ctx);
    }

    printf("[LOOP] Exited: %d display frames, %d logic ticks (avg temp_frames=%d)\n",
           ctx.frame_count, ctx.logic_count,
           ctx.logic_count > 0 ? (ctx.frame_count * 5 / 6) / ctx.logic_count : 0);
}

void game_loop(GameState *state)
{
    game_loop_with_hidden_start_frames(state, 0);
}
