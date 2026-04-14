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
#include <SDL.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Amiga key codes used in the original */
#define KEY_PAUSE   0x19
#define KEY_ESC     0x45

/* Maximum frame count before clamping */
#define MAX_TEMP_FRAMES 15

/* Minimum VBlanks to accumulate before running a game logic tick.
 * 1 VBlank = 20ms → 50Hz logic, matching the original game's simulation
 * rate. Running logic at 50Hz avoids large per-tick jumps that make the
 * player feel sluggish and can skip/tunnel through collision edges. */
#define GAME_TICK_VBLANKS 1

/* Number of frames to run in stub mode before auto-exiting */
/* Max frames before auto-exit (0 = disabled, relies on ESC key) */
#define STUB_MAX_FRAMES 0

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

void game_loop_ctx_init(GameLoopCtx *ctx, GameState *state)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->last_ticks = SDL_GetTicks();
    ctx->fps_log_start_ms = ctx->last_ticks;
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
        bool f2_pick_log_requested = input_f2_pick_log_requested();
        if (input_f5_save_requested())
            player_save_position(state);
        if (input_f9_load_requested()) {
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
        if (input_fullscreen_toggle_requested())
            display_toggle_fullscreen();

        /* ================================================================
         * Frame timing: accumulate 50Hz VBlanks from real elapsed time
         * ================================================================ */
        {
            Uint32 now = SDL_GetTicks();
            state->current_ticks_ms = (uint32_t)now;
            Uint32 elapsed = now - ctx->last_ticks;
            ctx->last_ticks = now;

            /* Clamp to prevent spiral-of-death after alt-tab etc. */
            if (elapsed > 200) elapsed = 200;

            /* Accumulate VBlanks at 50Hz (20ms per VBlank) */
            ctx->vblank_remainder_ms += elapsed;
            ctx->pending_vblanks += (int)(ctx->vblank_remainder_ms / 20);
            ctx->vblank_remainder_ms %= 20;

            /* Keep water phase moving every display frame; average speed stays 50Hz. */
            renderer_step_water_anim_ms(elapsed);
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
                state->hitcol -= 0x100;
                state->hitcol2 = state->hitcol;
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
                    state->ammo = gun_plr->gun_data[gun_idx].ammo >> 3;
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

            /* ---- Phase 10: Zone brightness animation (bright_anim_values updated; rendering reads from zone data) ---- */
            bright_anim_handler(state);

            /* ---- Phase 11: Visibility checks (multiplayer) ---- */
            if (state->mode != MODE_SINGLE) {
                if (state->level.zone_adds && state->level.data) {
                    /* CanItBeSeen stub */
                }
            }

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

                /* Amiga uses high 16 bits of xoff/zoff for OrderZones side test (move.w xoff,d2 = first word of long). */
                int32_t view_x = (int32_t)(int16_t)(view_plr->xoff >> 16);
                int32_t view_z = (int32_t)(int16_t)(view_plr->zoff >> 16);
                int32_t move_dx = (state->mode == MODE_SLAVE) ? (int32_t)state->xdiff2 : (int32_t)state->xdiff1;
                int32_t move_dz = (state->mode == MODE_SLAVE) ? (int32_t)state->zdiff2 : (int32_t)state->zdiff1;
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

            /* Match Amiga frame timing: in-line target vectors are refreshed
             * after object movement/AI for use by the next shot solve. */
            calc_plr1_in_line(state);
            if (state->mode != MODE_SINGLE) {
                calc_plr2_in_line(state);
            }

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
        display_energy_bar(state->energy);
        display_ammo_bar(state->ammo);
        display_draw_display(state);
        if (f2_pick_log_requested) {
            PlayerState *view_plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
            int16_t looking_zone = -1;
            renderer_get_center_pick(&looking_zone, NULL);
            printf("[DEBUG][F2] standing_zone=%d looking_zone=%d\n",
                   (int)view_plr->zone, (int)looking_zone);
            level_log_zones(&state->level);
            level_log_player_zone_full(state);
            level_log_zone_full(&state->level, looking_zone, "looking");
        }

        ctx->fps_frames_in_period++;
        {
            Uint32 fps_now = SDL_GetTicks();
            Uint32 fps_dt = fps_now - ctx->fps_log_start_ms;
            if (fps_dt >= 1000u) {
                double sec = fps_dt / 1000.0;
                double fps = (double)ctx->fps_frames_in_period / sec;
                int rounded = (int)(fps + 0.5);
                if (rounded < 0) rounded = 0;
                if (rounded > 9999) rounded = 9999;
                state->fps_display = (uint16_t)rounded;
                ctx->fps_log_start_ms = fps_now;
                ctx->fps_frames_in_period = 0;
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

        /* Auto-exit (only if STUB_MAX_FRAMES > 0) */
#if STUB_MAX_FRAMES > 0
        if (ctx->frame_count >= STUB_MAX_FRAMES) {
            printf("[LOOP] Auto-exit after %d frames\n", STUB_MAX_FRAMES);
            state->running = false;
        }
#endif
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
void game_loop(GameState *state)
{
    GameLoopCtx ctx;
    game_loop_ctx_init(&ctx, state);
    while (state->running) {
        game_loop_tick(state, &ctx);
    }

    printf("[LOOP] Exited: %d display frames, %d logic ticks (avg temp_frames=%d)\n",
           ctx.frame_count, ctx.logic_count,
           ctx.logic_count > 0 ? (ctx.frame_count * 5 / 6) / ctx.logic_count : 0);
}
