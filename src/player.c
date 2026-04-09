/*
 * Alien Breed 3D I - PC Port
 * player.c - Player control and movement physics
 *
 * Translated from: Plr1Control.s, Plr2Control.s, PlayerShoot.s, LevelData2.s,
 *                  AB3DI.s (PLR1_Control, PLR2_Control)
 */

#include "player.h"
#include "game_data.h"
#include "level.h"
#include "movement.h"
#include "objects.h"
#include "visibility.h"
#include "math_tables.h"
#include "input.h"
#include "audio.h"
#include "io.h"
#include "renderer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* -----------------------------------------------------------------------
 * Constants from the original ASM
 * ----------------------------------------------------------------------- */
#define WALK_SPEED          2
#define RUN_SPEED           3
#define MAX_TURN_WALK       35
#define MAX_TURN_RUN        60
#define TURN_STEP           10
#define MOUSE_CLAMP         50
#define HEIGHT_STEP         1024
#define BOBBLE_MASK         ANGLE_MASK
#define CLUMP_MASK          4095
#define CLUMP_THRESHOLD     (-4096)  /* and.w #-4096,d1 checks bit 12+ */
/* Amiga step-up: same scale as zone floor heights. game_data uses 40*256 for marines;
 * movement.c default is 40*256. Step-UP blocked when ledge is higher than this.
 * Step-DOWN always passable (unlimited). */
#define STEP_UP_NORMAL      (40 * 256)   /* 10240 - match Amiga ObjectMove step-up */
#define STEP_UP_DUCKED      (10 * 256)   /* 2560 - AB3DI.s ducked step-up */
#define STEP_DOWN_DEFAULT   0x1000000    /* AB3DI.s step-down */
#define INSTANT_TRACE_MAX_ITERS 1024
/* PlayerShoot.s applies projectile bulyspd clamping in PLR1FIREBULLET. */

/* Gun selection key -> gun index mapping (from GUNVALS in Plr1Control.s):
 * key1 pistol, key2 shotgun, key3 plasma, key4 grenade, key5 rocket, key6 flamethrower. */
static const int8_t gun_key_map[6] = { 0, 7, 1, 4, 2, 3 };
#define GUN_KEY_COUNT ((int)(sizeof(gun_key_map) / sizeof(gun_key_map[0])))

static int16_t player_room_zone_word(const uint8_t *room)
{
    if (!room) return -1;
    return (int16_t)((room[0] << 8) | room[1]);
}

static int player_weapon_slot_from_gun(int16_t gun_idx)
{
    for (int i = 0; i < GUN_KEY_COUNT; i++) {
        if (gun_key_map[i] == gun_idx) return i;
    }
    return 0;
}

static bool player_weapon_is_selectable(const PlayerState *plr, int gun_idx)
{
    if (!plr) return false;
    if (gun_idx < 0 || gun_idx >= MAX_GUNS) return false;
    return plr->gun_data[gun_idx].visible != 0;
}

static void player_ensure_valid_weapon_selection(PlayerState *plr)
{
    if (!plr) return;
    if (player_weapon_is_selectable(plr, plr->gun_selected)) return;

    for (int i = 0; i < GUN_KEY_COUNT; i++) {
        int gun_idx = gun_key_map[i];
        if (player_weapon_is_selectable(plr, gun_idx)) {
            plr->gun_selected = (int16_t)gun_idx;
            return;
        }
    }
}

static void player_cycle_weapon(PlayerState *plr, int direction)
{
    if (!plr || GUN_KEY_COUNT <= 0) return;

    player_ensure_valid_weapon_selection(plr);

    int slot = player_weapon_slot_from_gun(plr->gun_selected);
    int step_dir = (direction >= 0) ? 1 : -1;

    for (int tries = 0; tries < GUN_KEY_COUNT; tries++) {
        slot = (slot + step_dir + GUN_KEY_COUNT) % GUN_KEY_COUNT;
        int gun_idx = gun_key_map[slot];
        if (player_weapon_is_selectable(plr, gun_idx)) {
            plr->gun_selected = (int16_t)gun_idx;
            return;
        }
    }
}

static GameObject *find_free_player_shot_slot(uint8_t *shots, int16_t *saved_cid)
{
    if (!shots) return NULL;
    for (int i = 0; i < PLAYER_SHOT_SLOT_COUNT; i++) {
        GameObject *candidate = (GameObject *)(shots + i * OBJECT_SIZE);
        if (OBJ_ZONE(candidate) < 0) {
            if (saved_cid) *saved_cid = OBJ_CID(candidate);
            return candidate;
        }
    }
    return NULL;
}

static uint8_t *resolve_player_room_ptr(GameState *state, const PlayerState *plr)
{
    if (!state || !state->level.data || !plr) return NULL;
    if (plr->roompt >= 0) {
        size_t data_len = state->level.data_byte_count;
        if (data_len == 0 || ((size_t)plr->roompt + 2u) <= data_len)
            return state->level.data + plr->roompt;
    }
    if (state->level.zone_adds) {
        int zi = level_connect_to_zone_index(&state->level, plr->zone);
        if (zi < 0) {
            int zone_slots = level_zone_slot_count(&state->level);
            if (plr->zone >= 0 && plr->zone < zone_slots) zi = plr->zone;
        }
        if (zi >= 0) return level_get_zone_data_ptr(&state->level, (int16_t)zi);
    }
    return NULL;
}

static int16_t zone_from_room_or_fallback(LevelState *level, const uint8_t *room, int16_t fallback)
{
    if (room) {
        int zi = level_zone_index_from_room_ptr(level, room);
        if (zi >= 0) return (int16_t)zi;
        {
            int16_t room_zone = (int16_t)((room[0] << 8) | room[1]);
            zi = level_connect_to_zone_index(level, room_zone);
            if (zi >= 0) return (int16_t)zi;
            return room_zone;
        }
    }
    return fallback;
}

/* Matches PlayerShoot.s MISSINSTANT path:
 * repeated MoveObject steps until hitwall, then use impact coords for pop sprite. */
static bool trace_instant_miss_path(GameState *state, const PlayerState *plr,
                                    int16_t init_newx, int16_t init_newz, int32_t init_newy,
                                    bool use_wall_hit_y,
                                    int16_t *out_x, int16_t *out_z, int32_t *out_y,
                                    uint8_t **out_room, int8_t *out_in_top)
{
    if (!state || !plr || !out_x || !out_z || !out_y || !out_room || !out_in_top) return false;
    if (!state->level.data || !state->level.floor_lines) return false;

    MoveContext ctx;
    move_context_init(&ctx);
    ctx.exitfirst = 1;
    ctx.wallbounce = 0;
    ctx.extlen = 0;
    ctx.awayfromwall = -1;
    ctx.wall_flags = 0x0400;
    ctx.step_up_val = 0;
    ctx.step_down_val = 0x1000000;
    ctx.thing_height = 0;
    ctx.stood_in_top = plr->stood_in_top;

    uint8_t *room = resolve_player_room_ptr(state, plr);
    if (!room) return false;

    int16_t oldx = (int16_t)(plr->xoff >> 16);
    int16_t oldz = (int16_t)(plr->zoff >> 16);
    int16_t newx = init_newx;
    int16_t newz = init_newz;
    int32_t oldy = plr->yoff + 20 * 128;
    int32_t newy = init_newy;
    int8_t in_top = plr->stood_in_top;

    for (int iter = 0; iter < INSTANT_TRACE_MAX_ITERS; iter++) {
        ctx.oldx = oldx;
        ctx.oldz = oldz;
        ctx.newx = newx;
        ctx.newz = newz;
        ctx.oldy = oldy;
        ctx.newy = newy;
        ctx.objroom = room;
        ctx.stood_in_top = in_top;

        move_object(&ctx, &state->level);
        room = ctx.objroom;
        in_top = ctx.stood_in_top;

        if (ctx.hitwall) {
            *out_x = (int16_t)ctx.newx;
            *out_z = (int16_t)ctx.newz;
            *out_y = use_wall_hit_y ? ctx.wall_hit_y : newy;
            *out_room = room;
            *out_in_top = in_top;
            return true;
        }

        {
            int16_t dx = (int16_t)((int16_t)ctx.newx - (int16_t)ctx.oldx);
            int16_t dz = (int16_t)((int16_t)ctx.newz - (int16_t)ctx.oldz);
            oldx = (int16_t)(oldx + dx);
            newx = (int16_t)(newx + dx);
            oldz = (int16_t)(oldz + dz);
            newz = (int16_t)(newz + dz);
        }

        {
            int32_t dy = newy - oldy;
            oldy += dy;
            newy += dy;
        }
    }
    return false;
}

static void init_instant_pop_slot(GameObject *shot, int gun_idx, int16_t zone,
                                  int8_t in_top, int32_t accypos)
{
    shot->obj.number = OBJ_NBR_BULLET;
    OBJ_SET_ZONE(shot, zone);
    shot->obj.in_top = in_top;
    SHOT_STATUS(*shot) = 1;
    SHOT_SET_GRAV(*shot, 0);
    SHOT_SIZE(*shot) = (int8_t)gun_idx;
    SHOT_ANIM(*shot) = 0;
    SHOT_SET_ACCYPOS(*shot, accypos);
    obj_sw(shot->raw + 4, (int16_t)(accypos >> 7));
    shot->obj.worry = 127;
}

/* PlayerShoot.s PLR1HITINSTANT/PLR2HITINSTANT behavior:
 * find free slot, spawn pop at target, then apply damage+impact push. */
static void spawn_instant_hit_effect(GameState *state, const PlayerState *plr, int gun_idx,
                                     GameObject *target, int8_t shot_power,
                                     int16_t dir_x, int16_t dir_z)
{
    if (!state || !plr || !target) return;
    if (gun_idx < 0 || gun_idx >= MAX_BULLET_ANIM_IDX) return;

    uint8_t *shot_pool = state->level.player_shot_data;
    if (!shot_pool) return;

    int16_t saved_cid = -1;
    GameObject *impact = find_free_player_shot_slot(shot_pool, &saved_cid);
    if (!impact) return;

    obj_sw(impact->raw, saved_cid);

    {
        int16_t target_cid = OBJ_CID(target);
        if (saved_cid >= 0 && target_cid >= 0 &&
            state->level.object_points &&
            saved_cid < state->level.num_object_points &&
            target_cid < state->level.num_object_points) {
            const uint8_t *src = state->level.object_points + (uint32_t)(uint16_t)target_cid * 8u;
            uint8_t *dst = state->level.object_points + (uint32_t)(uint16_t)saved_cid * 8u;
            dst[0] = src[0]; dst[1] = src[1];
            dst[4] = src[4]; dst[5] = src[5];
        }
    }

    {
        int16_t target_y = obj_w(target->raw + 4);
        init_instant_pop_slot(impact, gun_idx, OBJ_ZONE(target), target->obj.in_top,
                              ((int32_t)target_y) << 7);
    }

    {
        target->raw[19] = (uint8_t)(target->raw[19] + (uint8_t)shot_power);
    }

    {
        int16_t impact_x = (int16_t)(((int32_t)dir_x << 3) >> 16);
        int16_t impact_z = (int16_t)(((int32_t)dir_z << 3) >> 16);
        NASTY_SET_IMPACTX(target, impact_x);
        NASTY_SET_IMPACTZ(target, impact_z);
    }
}

/* PlayerShoot.s PLR1MISSINSTANT/PLR2MISSINSTANT behavior (miss with a target):
 * trace toward midpoint to target, continue along ray until first wall hit, spawn pop there. */
static void spawn_instant_miss_effect_target(GameState *state, const PlayerState *plr,
                                             int gun_idx, const GameObject *target)
{
    if (!state || !plr || !target) return;
    if (gun_idx < 0 || gun_idx >= MAX_BULLET_ANIM_IDX) return;
    if (!state->level.player_shot_data || !state->level.object_points) return;

    int16_t target_cid = OBJ_CID(target);
    if (target_cid < 0 || target_cid >= state->level.num_object_points) return;

    int16_t oldx = (int16_t)(plr->xoff >> 16);
    int16_t oldz = (int16_t)(plr->zoff >> 16);
    const uint8_t *target_pt = state->level.object_points + (uint32_t)(uint16_t)target_cid * 8u;
    int16_t tx = obj_w(target_pt + 0);
    int16_t tz = obj_w(target_pt + 4);
    int16_t newx = (int16_t)(oldx + ((int16_t)(tx - oldx) >> 1));
    int16_t newz = (int16_t)(oldz + ((int16_t)(tz - oldz) >> 1));
    int32_t newy = ((int32_t)obj_w(target->raw + 4)) << 7;

    int16_t hit_x = 0, hit_z = 0;
    int32_t hit_y = 0;
    uint8_t *hit_room = NULL;
    int8_t hit_in_top = plr->stood_in_top;
    if (!trace_instant_miss_path(state, plr, newx, newz, newy, false,
                                 &hit_x, &hit_z, &hit_y, &hit_room, &hit_in_top))
        return;

    int16_t saved_cid = -1;
    GameObject *impact = find_free_player_shot_slot(state->level.player_shot_data, &saved_cid);
    if (!impact) return;
    obj_sw(impact->raw, saved_cid);

    if (saved_cid >= 0 && saved_cid < state->level.num_object_points) {
        uint8_t *pt = state->level.object_points + (uint32_t)(uint16_t)saved_cid * 8u;
        obj_sw(pt, hit_x);
        obj_sw(pt + 4, hit_z);
    }

    init_instant_pop_slot(impact, gun_idx,
                          zone_from_room_or_fallback(&state->level, hit_room, plr->zone),
                          hit_in_top, hit_y);
}

/* PlayerShoot.s nothingtoshoot path for instant weapons:
 * fire forward, random vertical offset, trace until wall hit, spawn pop at wallhitheight. */
static void spawn_instant_miss_effect_no_target(GameState *state, const PlayerState *plr,
                                                int gun_idx, int16_t sin_val, int16_t cos_val)
{
    if (!state || !plr) return;
    if (gun_idx < 0 || gun_idx >= MAX_BULLET_ANIM_IDX) return;
    if (!state->level.player_shot_data) return;

    int16_t oldx = (int16_t)(plr->xoff >> 16);
    int16_t oldz = (int16_t)(plr->zoff >> 16);
    int16_t newx = (int16_t)(oldx + (sin_val >> 7));
    int16_t newz = (int16_t)(oldz + (cos_val >> 7));
    int32_t oldy = plr->yoff + 20 * 128;
    int32_t newy = oldy + (int32_t)((rand() & 0x0FFF) - 0x800);

    int16_t hit_x = 0, hit_z = 0;
    int32_t hit_y = 0;
    uint8_t *hit_room = NULL;
    int8_t hit_in_top = plr->stood_in_top;
    if (!trace_instant_miss_path(state, plr, newx, newz, newy, true,
                                 &hit_x, &hit_z, &hit_y, &hit_room, &hit_in_top))
        return;

    int16_t saved_cid = -1;
    GameObject *impact = find_free_player_shot_slot(state->level.player_shot_data, &saved_cid);
    if (!impact) return;
    obj_sw(impact->raw, saved_cid);

    if (saved_cid >= 0 && state->level.object_points &&
        saved_cid < state->level.num_object_points) {
        uint8_t *pt = state->level.object_points + (uint32_t)(uint16_t)saved_cid * 8u;
        obj_sw(pt, hit_x);
        obj_sw(pt + 4, hit_z);
    }

    init_instant_pop_slot(impact, gun_idx,
                          zone_from_room_or_fallback(&state->level, hit_room, plr->zone),
                          hit_in_top, hit_y);
}

/* -----------------------------------------------------------------------
 * Friction helper
 *
 * The original code computes a per-tick friction term from velocity,
 * then applies it as:
 *   velocity = velocity - friction_term
 *
 * The friction term uses this sequence:
 *   neg.l d6; ble .nobug; asr #N,d6; add #1,d6; bra .done
 *   .nobug: asr #N,d6
 *
 * Control mode shifts:
 *   Mouse:    #1
 *   Keyboard: #3
 *   Mouse+KBD (port tuning): #2
 * ----------------------------------------------------------------------- */
static int32_t apply_friction(int32_t val, int shift)
{
    int32_t neg = -val;
    if (neg > 0) {
        return -((neg >> shift) + 1);
    } else {
        return -(neg >> shift);
    }
}

/* -----------------------------------------------------------------------
 * PLR1_alwayskeys - Keys checked regardless of control mode
 *
 * Translated from Plr1Control.s PLR1_alwayskeys (~line 187-315)
 * ----------------------------------------------------------------------- */
static void player_always_keys(PlayerState *plr, uint8_t *key_map,
                                const KeyBindings *keys, uint8_t *old_space,
                                const GameState *state)
{
    player_ensure_valid_weapon_selection(plr);

    /* Operate/Space - tap detection */
    uint8_t space_state = key_map[keys->operate];
    if (space_state && !(*old_space)) {
        plr->spctap = -1;  /* st = set to $FF */
    }
    *old_space = space_state;

    /* Duck toggle */
    if (key_map[keys->duck]) {
        /* Amiga PLR*_alwayskeys clears duck key after reading to make it a toggle. */
        key_map[keys->duck] = 0;
        plr->s_targheight = PLAYER_HEIGHT;
        plr->ducked = !plr->ducked;
        if (plr->ducked) {
            plr->s_targheight = PLAYER_CROUCHED;
        }
    }

    /* Amiga parity: force crouch when room is too low to stand.
     * Source: Plr1Control.s / Plr2Control.s PLR*_alwayskeys:
     *   room_h = floor - roof (or upper floor/roof if stood_in_top)
     *   if room_h <= playerheight + 3*1024: ducked=1, targheight=playercrouched
     */
    if (state && state->level.data && plr->roompt >= 0) {
        const uint8_t *room = state->level.data + plr->roompt;
        int off = plr->stood_in_top ? 8 : 0;
        int32_t floor_h = (int32_t)(
            (room[2 + off] << 24) |
            (room[3 + off] << 16) |
            (room[4 + off] <<  8) |
             room[5 + off]);
        int32_t roof_h = (int32_t)(
            (room[6 + off] << 24) |
            (room[7 + off] << 16) |
            (room[8 + off] <<  8) |
             room[9 + off]);
        int32_t room_h = floor_h - roof_h;
        if (room_h <= PLAYER_HEIGHT + 3 * 1024) {
            plr->ducked = 1;
            plr->s_targheight = PLAYER_CROUCHED;
        }
    }

    /* Height interpolation toward target */
    int32_t h = plr->s_height;
    int32_t target = plr->s_targheight;
    if (h < target) {
        h += HEIGHT_STEP;
        if (h > target) h = target;
    } else if (h > target) {
        h -= HEIGHT_STEP;
        if (h < target) h = target;
    }
    plr->s_height = h;

    /* Weapon selection (number keys) */
    for (int i = 0; i < GUN_KEY_COUNT; i++) {
        if (key_map[i + 1]) {
            int gun_idx = gun_key_map[i];
            if (player_weapon_is_selectable(plr, gun_idx)) {
                plr->gun_selected = (int16_t)gun_idx;
            }
        }
    }

    /* Mouse wheel cycles through the mapped weapon slots. */
    {
        MouseState mouse;
        input_read_mouse(&mouse);
        int wheel_steps = (int)mouse.wheel_y;
        if (wheel_steps != 0) {
            int direction = (wheel_steps > 0) ? -1 : 1;
            if (wheel_steps < 0) wheel_steps = -wheel_steps;
            while (wheel_steps-- > 0) {
                player_cycle_weapon(plr, direction);
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Keyboard control
 *
 * Translated from Plr1Control.s PLR1_keyboard_control (~line 337-547)
 * ----------------------------------------------------------------------- */
static void player_keyboard_control(PlayerState *plr, const uint8_t *key_map,
                                     const KeyBindings *keys, int16_t temp_frames)
{
    int16_t max_turn = MAX_TURN_WALK;
    int16_t move_speed = WALK_SPEED;

    /* Running */
    if (key_map[keys->run]) {
        max_turn = MAX_TURN_RUN;
        move_speed = RUN_SPEED;
    }

    /* Ducked halves speed */
    if (plr->ducked) {
        move_speed >>= 1;
    }

    /* Scale by temp_frames: on the Amiga, speed = temp_frames * walk_speed.
     * This gives larger per-tick deltas so the collision system works. */
    move_speed *= temp_frames;

    /* ---- Turning ---- */
    int16_t angspd = plr->s_angspd;

    /* Friction on turn speed: d3 = d3*3/4 */
    angspd = (angspd * 3) / 4;

    /* Force sidestep mode swaps turn keys with strafe keys */
    uint8_t left_key = keys->turn_left;
    uint8_t right_key = keys->turn_right;
    uint8_t sl_key = keys->sidestep_left;
    uint8_t sr_key = keys->sidestep_right;

    if (key_map[keys->force_sidestep]) {
        sl_key = keys->turn_left;
        sr_key = keys->turn_right;
        left_key = 0xFF;  /* disable turning */
        right_key = 0xFF;
    }

    if (left_key < 128 && key_map[left_key]) {
        angspd -= TURN_STEP;
    }
    if (right_key < 128 && key_map[right_key]) {
        angspd += TURN_STEP;
    }

    /* Clamp turn speed */
    if (angspd > max_turn) angspd = max_turn;
    if (angspd < -max_turn) angspd = -max_turn;

    /* Apply turn to angle (doubled in original: add.w d3,d0; add.w d3,d0) */
    int16_t angpos = plr->s_angpos;
    angpos += angspd * 2;
    plr->s_angspd = angspd;

    /* ---- Strafe ---- */
    int16_t strafe = 0;
    if (sl_key < 128 && key_map[sl_key]) {
        strafe = (move_speed * 2 + 1) / 2;   /* add+add+asr = *1.5 rounded */
    }
    if (sr_key < 128 && key_map[sr_key]) {
        strafe = -((move_speed * 2 + 1) / 2);
    }

    /* ---- Angle and sin/cos lookup ---- */
    angpos &= ANGLE_MASK;
    plr->s_angpos = angpos;
    plr->s_sinval = sin_lookup(angpos);
    plr->s_cosval = cos_lookup(angpos);

    /* ---- Velocity friction (keyboard: ~1/8 removed per tick) ---- */
    int32_t xspd = plr->s_xspdval - apply_friction(plr->s_xspdval, 3);
    int32_t zspd = plr->s_zspdval - apply_friction(plr->s_zspdval, 3);

    /* ---- Forward/backward ---- */
    int16_t fwd = 0;
    if (key_map[keys->forward]) {
        fwd = -move_speed;
    }
    if (key_map[keys->backward]) {
        fwd = move_speed;
    }

    /* ---- Bobble ---- */
    int16_t bob_delta = fwd << 6;  /* asl.w #6,d2 */
    plr->bobble = (plr->bobble + bob_delta) & BOBBLE_MASK;

    /* ---- Clump (footstep sound trigger) ---- */
    /* clumptime accumulates movement; when bit 12+ overflows, trigger sound */
    /* TODO: actual clump timing once audio works */

    /* ---- Calculate movement vector ---- */
    /* Forward: x -= sin*fwd, z -= cos*fwd */
    int32_t sin_val = plr->s_sinval;
    int32_t cos_val = plr->s_cosval;

    xspd -= sin_val * fwd;
    zspd -= cos_val * fwd;

    /* Strafe: x -= cos*strafe, z += sin*strafe */
    xspd -= cos_val * strafe;
    zspd += sin_val * strafe;

    /* ---- Apply velocity to position ---- */
    plr->s_xspdval = xspd;
    plr->s_zspdval = zspd;
    plr->s_xoff += xspd;
    plr->s_zoff += zspd;

    /* ---- Fire handling ---- */
    if (plr->fire) {
        /* Fire was pressed last frame */
        if (key_map[keys->fire]) {
            plr->fire = -1;  /* still held */
        } else {
            plr->fire = 0;   /* released */
        }
    } else {
        /* Fire was not pressed last frame */
        if (key_map[keys->fire]) {
            plr->clicked = -1;  /* new click */
            plr->fire = -1;
        }
    }
}

/* -----------------------------------------------------------------------
 * Mouse control
 *
 * Translated from Plr1Control.s PLR1_mouse_control (~line 13-138)
 * ----------------------------------------------------------------------- */
static void player_mouse_control(PlayerState *plr, const uint8_t *key_map,
                                  const KeyBindings *keys, int16_t temp_frames)
{
    MouseState mouse;
    input_read_mouse(&mouse);

    /* Angle from mouse X - mouse delta self-scales (accumulates between ticks) */
    int16_t angpos = plr->s_angpos;
    angpos += mouse.dx;

    /* Forward/backward from mouse Y - also self-scaling */
    int16_t fwd_delta = mouse.dy;

    /* Clamp */
    if (fwd_delta > MOUSE_CLAMP) fwd_delta = MOUSE_CLAMP;
    if (fwd_delta < -MOUSE_CLAMP) fwd_delta = -MOUSE_CLAMP;

    /* Halve when ducked */
    if (plr->ducked) {
        fwd_delta >>= 1;
    }

    int16_t move_mag = fwd_delta << 4;

    /* Angle and sin/cos */
    angpos &= ANGLE_MASK;
    plr->s_angpos = angpos;
    plr->s_sinval = sin_lookup(angpos);
    plr->s_cosval = cos_lookup(angpos);

    /* Velocity friction (mouse: ~1/2 removed per tick) */
    int32_t xspd = plr->s_xspdval - apply_friction(plr->s_xspdval, 1);
    int32_t zspd = plr->s_zspdval - apply_friction(plr->s_zspdval, 1);

    /* Strafe from keyboard (lrs variable in original) - scale by temp_frames */
    int16_t lrs = 0;
    if (key_map[keys->sidestep_left])  lrs += 2 * temp_frames;
    if (key_map[keys->sidestep_right]) lrs -= 2 * temp_frames;

    /* Calculate movement */
    int32_t sin_val = plr->s_sinval;
    int32_t cos_val = plr->s_cosval;

    /* Strafe component */
    int32_t sx = -cos_val * lrs;
    int32_t sz = sin_val * lrs;

    /* Forward component */
    int32_t fx = sin_val * fwd_delta;
    int32_t fz = cos_val * fwd_delta;

    xspd = xspd - fx - sx;
    zspd = zspd - fz + sz;

    /* Apply */
    plr->s_xspdval = xspd;
    plr->s_zspdval = zspd;
    plr->s_xoff += xspd;
    plr->s_zoff += zspd;

    /* Bobble */
    plr->bobble = (plr->bobble + move_mag) & BOBBLE_MASK;

    /* Fire from mouse button */
    if (plr->fire) {
        if (mouse.left_button) {
            plr->fire = -1;
        } else {
            plr->fire = 0;
        }
    } else {
        if (mouse.left_button) {
            plr->clicked = -1;
            plr->fire = -1;
        }
    }

    (void)keys; /* some keys used above directly */
}

/* -----------------------------------------------------------------------
 * Mouse + Keyboard control (modern FPS style)
 *
 * Mouse X controls turning (horizontal look).
 * WASD / arrow keys control movement and strafing.
 * Mouse left button or fire key fires weapon.
 * Shift to run.
 * ----------------------------------------------------------------------- */
#define MOUSE_SENSITIVITY 3  /* angle units per SDL pixel; tune to taste */

static void player_mouse_kbd_control(PlayerState *plr, const uint8_t *key_map,
                                      const KeyBindings *keys, int16_t temp_frames)
{
    MouseState mouse;
    input_read_mouse(&mouse);

    int16_t move_speed = WALK_SPEED;

    /* Running (shift) */
    if (key_map[keys->run]) {
        move_speed = RUN_SPEED;
    }

    /* Ducked halves speed */
    if (plr->ducked) {
        move_speed >>= 1;
    }

    /* Scale by temp_frames: on the Amiga, speed = temp_frames * walk_speed.
     * This gives larger per-tick deltas so the collision system works. */
    move_speed *= temp_frames;

    /* ---- Turning from mouse X ---- */
    /* Mouse delta self-scales (accumulates between game ticks) */
    int16_t angpos = plr->s_angpos;
    angpos += (int16_t)(mouse.dx * MOUSE_SENSITIVITY);
    angpos &= ANGLE_MASK;
    plr->s_angpos = angpos;
    plr->s_sinval = sin_lookup(angpos);
    plr->s_cosval = cos_lookup(angpos);

    /* ---- Velocity friction (~1/8 removed per tick).
     * Plr1WasdControl.s uses asr.l #3 for mouse+kbd, same as keyboard mode. */
    int32_t xspd = plr->s_xspdval - apply_friction(plr->s_xspdval, 3);
    int32_t zspd = plr->s_zspdval - apply_friction(plr->s_zspdval, 3);

    /* ---- Forward / backward (W/S or arrow up/down) ---- */
    int16_t fwd = 0;
    if (key_map[keys->forward]) {
        fwd = -move_speed;
    }
    if (key_map[keys->backward]) {
        fwd = move_speed;
    }

    /* ---- Strafe (A/D or . / keys) ---- */
    int16_t strafe = 0;
    if (key_map[keys->sidestep_left]) {
        strafe = (move_speed * 2 + 1) / 2;   /* match original 1.5x strafe */
    }
    if (key_map[keys->sidestep_right]) {
        strafe = -((move_speed * 2 + 1) / 2);
    }

    /* Also allow arrow left/right for turning when mouse isn't moved */
    if (key_map[keys->turn_left]) {
        int16_t ang = plr->s_angpos;
        ang -= TURN_STEP * 2 * temp_frames;
        ang &= ANGLE_MASK;
        plr->s_angpos = ang;
        plr->s_sinval = sin_lookup(ang);
        plr->s_cosval = cos_lookup(ang);
    }
    if (key_map[keys->turn_right]) {
        int16_t ang = plr->s_angpos;
        ang += TURN_STEP * 2 * temp_frames;
        ang &= ANGLE_MASK;
        plr->s_angpos = ang;
        plr->s_sinval = sin_lookup(ang);
        plr->s_cosval = cos_lookup(ang);
    }

    /* ---- Bobble ---- */
    int16_t bob_delta = fwd << 6;
    plr->bobble = (plr->bobble + bob_delta) & BOBBLE_MASK;

    /* ---- Movement vector ---- */
    int32_t sin_val = plr->s_sinval;
    int32_t cos_val = plr->s_cosval;

    /* Forward: x -= sin*fwd, z -= cos*fwd */
    xspd -= sin_val * fwd;
    zspd -= cos_val * fwd;

    /* Strafe: x -= cos*strafe, z += sin*strafe */
    xspd -= cos_val * strafe;
    zspd += sin_val * strafe;

    /* ---- Apply velocity ---- */
    plr->s_xspdval = xspd;
    plr->s_zspdval = zspd;
    plr->s_xoff += xspd;
    plr->s_zoff += zspd;

    /* ---- Fire from mouse button or keyboard fire key ---- */
    bool fire_pressed = mouse.left_button || key_map[keys->fire];
    if (plr->fire) {
        if (fire_pressed) {
            plr->fire = -1;  /* still held */
        } else {
            plr->fire = 0;   /* released */
        }
    } else {
        if (fire_pressed) {
            plr->clicked = -1;  /* new click */
            plr->fire = -1;
        }
    }
}

/* -----------------------------------------------------------------------
 * Top-level player 1/2 simulation control
 *
 * This runs the input -> movement -> fall pipeline.
 * The PLR1_Control in AB3DI.s then takes the simulation results
 * and applies collision/room-transition. That second stage is below.
 * ----------------------------------------------------------------------- */
static void player_sim_control(PlayerState *plr, const ControlMode *ctrl,
                                uint8_t *key_map, const KeyBindings *keys,
                                const GameState *state,
                                int16_t temp_frames)
{
    static uint8_t old_space = 0;

    /* Always-active keys */
    player_always_keys(plr, key_map, keys, &old_space, state);

    /* Dispatch based on control mode */
    if (ctrl->mouse_kbd) {
        player_mouse_kbd_control(plr, key_map, keys, temp_frames);
    } else if (ctrl->mouse) {
        player_mouse_control(plr, key_map, keys, temp_frames);
    } else if (ctrl->keys) {
        player_keyboard_control(plr, key_map, keys, temp_frames);
    } else if (ctrl->joy) {
        player_keyboard_control(plr, key_map, keys, temp_frames);
    } else {
        /* Default: mouse + keyboard */
        player_mouse_kbd_control(plr, key_map, keys, temp_frames);
    }
}

/* -----------------------------------------------------------------------
 * PLR1_Control / PLR2_Control - Full control + collision + room update
 *
 * Translated from AB3DI.s PLR1_Control (~line 2972-3180)
 *
 * 1. Snapshot old/new positions
 * 2. Calculate sin/cos and bobble
 * 3. Check teleport
 * 4. Object-to-object collision
 * 5. MoveObject (room transitions, wall collision)
 * 6. Update room data, floor heights, graph rooms
 * ----------------------------------------------------------------------- */
static void player_full_control(PlayerState *plr, GameState *state, int plr_num)
{
    MoveContext ctx;
    move_context_init(&ctx);
    const int zone_slots = level_zone_slot_count(&state->level);

    /* 1. Snapshot positions (16.16 fixed-point) */
    plr->oldxoff = plr->xoff;
    plr->oldzoff = plr->zoff;

    /* Copy simulation position to actual (both in 16.16 fixed-point).
     * On the Amiga (AB3DI.s PLR1_Control ~line 2984):
     *   move.l PLR1s_xoff,PLR1_xoff
     * PLR1s_xoff is updated by PLR1s_Control (the sim) before this point. */
    plr->xoff = plr->s_xoff;
    plr->zoff = plr->s_zoff;

    /* For collision_check (object-to-object): use integer coordinates.
     * The Amiga does this via .w operations on big-endian 32-bit values. */
    ctx.oldx = plr->oldxoff >> 16;
    ctx.oldz = plr->oldzoff >> 16;
    ctx.newx = plr->xoff >> 16;
    ctx.newz = plr->zoff >> 16;

    plr->height = plr->p_height;

    /* Movement delta (integer for collision_check) */
    ctx.xdiff = ctx.newx - ctx.oldx;
    ctx.zdiff = ctx.newz - ctx.oldz;

    /* 2. Angle and sin/cos */
    int16_t angpos = plr->p_angpos;
    plr->angpos = angpos;
    plr->sinval = sin_lookup(angpos);
    plr->cosval = cos_lookup(angpos);

    /* 3. Bobble -> Y offset adjustment */
    int16_t bob_sin = sin_lookup(plr->p_bobble);
    int16_t bob_val = bob_sin;
    if (bob_val > 0) bob_val = -bob_val;
    bob_val += 16384;
    bob_val >>= 4;
    if (!plr->ducked) {
        bob_val *= 2;
    }

    int32_t yoff = plr->p_yoff;
    yoff += bob_val;
    plr->yoff = yoff;
    ctx.newy = yoff;
    ctx.oldy = yoff;

    ctx.thing_height = plr->height - bob_val;

    /* Step values */
    ctx.step_up_val = plr->ducked ? STEP_UP_DUCKED : STEP_UP_NORMAL;
    ctx.step_down_val = STEP_DOWN_DEFAULT;

    /* 4a. Set objroom from current zone (required for move_object wall checks).
     *     Prefer zone so that after transitioning we immediately use the new
     *     zone's walls; keep roompt in sync so zone is updated from it. */
    ctx.objroom = NULL;
    if (state->level.data && state->level.zone_adds && plr->zone >= 0 &&
        plr->zone < zone_slots) {
        int32_t zone_off = (int32_t)(
            (state->level.zone_adds[plr->zone * 4    ] << 24) |
            (state->level.zone_adds[plr->zone * 4 + 1] << 16) |
            (state->level.zone_adds[plr->zone * 4 + 2] <<  8) |
             state->level.zone_adds[plr->zone * 4 + 3]);
        ctx.objroom = state->level.data + zone_off;
    } else if (state->level.data && plr->roompt >= 0) {
        ctx.objroom = state->level.data + plr->roompt;
    }

    /* 4b. Teleport check (uses integer ctx) */
    if (plr->zone >= 0) {
        if (check_teleport(&ctx, &state->level, plr->zone)) {
            /* Teleport destination is in integer coords; sync to 16.16. */
            plr->xoff = (int32_t)ctx.newx << 16;
            plr->zoff = (int32_t)ctx.newz << 16;
            /* Keep old/new in sync so MoveObject does not sweep between
             * pre-teleport and post-teleport positions this frame. */
            plr->oldxoff = plr->xoff;
            plr->oldzoff = plr->zoff;
            plr->s_xoff = plr->xoff;
            plr->s_zoff = plr->zoff;
            /* check_teleport adjusts ctx.newy by source/destination floor
             * delta; carry that back into sim Y (without bob offset). */
            plr->s_yoff = ctx.newy - bob_val;
            plr->yoff = ctx.newy;
            audio_play_sample(26, 64);
        }
    }

    /* 5. Object collision check.
     * coll_id must be set to the player's OWN object CID so that
     * collision_check skips the player object.  Without this, the player
     * collides with themselves every frame and can never move. */
    {
        uint8_t *plr_obj_ptr = (plr_num == 1) ?
            state->level.plr1_obj : state->level.plr2_obj;
        if (plr_obj_ptr) {
            GameObject *plr_obj = (GameObject*)plr_obj_ptr;
            ctx.coll_id = OBJ_CID(plr_obj);
        } else {
            ctx.coll_id = -1;
        }
    }
    /* CollideFlags from AB3DI.s PLR1_Control / PLR2_Control:
     *   P1: %1011111110111000001 = 0x5FDC1  (excludes PLR1 bit 5)
     *   P2: %1011111010111100001 = 0x5F5E1  (excludes PLR2 bit 11) */
    ctx.collide_flags = (plr_num == 1) ? 0x5FDC1 : 0x5F5E1;
    ctx.wall_flags = (plr_num == 1) ? 0x100 : 0x800;
    ctx.stood_in_top = plr->stood_in_top;

    collision_check(&ctx, &state->level);

    /* On object collision, revert to old position.
     * The Amiga (AB3DI.s ~line 3119-3123):
     *   move.w oldx,PLR1_xoff        ; Writes integer part only (high word)
     *   move.w oldz,PLR1_zoff        ; Low word (sim fraction) is PRESERVED
     *   move.l PLR1_xoff,PLR1s_xoff  ; Full 32-bit sync
     *   bra.b  .cantmove             ; SKIP MoveObject!
     *
     * Note: the Amiga uses move.w which only overwrites the high word
     * (integer part) while preserving the low word (fractional part from
     * the sim).  This means sub-pixel velocity accumulates even when
     * blocked by an object.
     *
     * Also note: when Collision hits, the Amiga SKIPS MoveObject entirely
     * (branches to .cantmove).  In our code, we still call move_object
     * but xdiff=zdiff=0 causes it to return immediately, which is equivalent. */
    if (ctx.hitwall) {
        /* Revert integer part only, preserve sim fraction (Amiga move.w semantics) */
        plr->xoff = (plr->oldxoff & (int32_t)0xFFFF0000) | (plr->xoff & 0x0000FFFF);
        plr->zoff = (plr->oldzoff & (int32_t)0xFFFF0000) | (plr->zoff & 0x0000FFFF);
        plr->s_xoff = plr->xoff;
        plr->s_zoff = plr->zoff;
    }

    /* 6. MoveObject - room transitions and wall sliding.
     *
     * CRITICAL: The Amiga MoveObject uses move.w (16-bit word) reads on
     * the big-endian 32-bit position values, extracting only the INTEGER
     * part (high word).  All cross products, wall slides, and distance
     * calculations happen in integer space.
     *
     * Using full 16.16 fixed-point (pos_shift=16) causes sub-pixel wall
     * crossings that the original game would never detect, resulting in
     * the player getting stuck against walls.  We match the Amiga by
     * passing integer coordinates (pos_shift=0). */
    ctx.pos_shift = 0;
    ctx.oldx = plr->oldxoff >> 16;   /* Integer part (Amiga: move.w oldx) */
    ctx.oldz = plr->oldzoff >> 16;
    ctx.newx = plr->xoff >> 16;      /* Integer part (Amiga: move.w newx) */
    ctx.newz = plr->zoff >> 16;
    ctx.xdiff = ctx.newx - ctx.oldx;
    ctx.zdiff = ctx.newz - ctx.oldz;
    ctx.extlen = 40;
    ctx.awayfromwall = 0;  /* AB3DI.s PLR1_Control: move.b #0,awayfromwall */
    ctx.exitfirst = 0;
    ctx.wallbounce = 0;

    /* Run wall/room collision in small integer sub-steps to avoid tunneling
     * through thin walls/corners when per-tick velocity gets high. */
    {
        int32_t start_x = ctx.oldx;
        int32_t start_z = ctx.oldz;
        int32_t target_x = ctx.newx;
        int32_t target_z = ctx.newz;
        int32_t dx = target_x - start_x;
        int32_t dz = target_z - start_z;
        int32_t adx = (dx < 0) ? -dx : dx;
        int32_t adz = (dz < 0) ? -dz : dz;
        int32_t maxd = (adx > adz) ? adx : adz;

        /* 4 world units per sub-step so we don't tunnel through walls. */
        int steps = (int)(maxd / 4) + 1;
        if (steps > 32) steps = 32;

        /* Stairs: for one frame after a zone change, don't allow stepping back into the previous zone. */
        ctx.no_transition_back = (plr->no_transition_back_roompt >= 0 && state->level.data) ?
            (state->level.data + plr->no_transition_back_roompt) : NULL;
        plr->no_transition_back_roompt = -1;  /* use once then clear */

        int32_t cur_x = start_x;
        int32_t cur_z = start_z;
        uint8_t *cur_room = ctx.objroom;
        int8_t cur_top = ctx.stood_in_top;
        int8_t any_hit = 0;

        for (int i = 1; i <= steps; i++) {
            uint8_t *room_before = cur_room;
            MoveContext step = ctx;
            step.objroom = cur_room;
            step.stood_in_top = cur_top;
            step.oldx = cur_x;
            step.oldz = cur_z;
            step.newx = start_x + (dx * i) / steps;
            step.newz = start_z + (dz * i) / steps;
            step.xdiff = step.newx - step.oldx;
            step.zdiff = step.newz - step.oldz;

            move_object(&step, &state->level);

            if (step.hitwall) any_hit = 1;
            cur_x = step.newx;
            cur_z = step.newz;
            cur_room = step.objroom;
            cur_top = step.stood_in_top;
            if (step.hitwall) any_hit = 1;

            /* After transitioning, validate position in the new zone so we don't end up
             * inside a wall (wall clipping) or trigger phantom walls next frame.
             * Do not stop the sub-step sweep here: Amiga MoveObject can continue
             * evaluating further transitions within the same move tick. */
            if (cur_room != room_before) {
                /* Run move_object again in the new zone so we check the new zone's
                 * walls and slide/revert if we stepped into a wall. */
                MoveContext step2 = ctx;
                step2.objroom = cur_room;
                step2.stood_in_top = cur_top;
                step2.oldx = cur_x;
                step2.oldz = cur_z;
                step2.newx = start_x + (dx * i) / steps;
                step2.newz = start_z + (dz * i) / steps;
                step2.xdiff = step2.newx - step2.oldx;
                step2.zdiff = step2.newz - step2.oldz;
                move_object(&step2, &state->level);
                cur_x = step2.newx;
                cur_z = step2.newz;
                cur_room = step2.objroom;
                cur_top = step2.stood_in_top;
                if (step2.hitwall) any_hit = 1;
                if (state->level.data && room_before) {
                    int32_t no_back_off = (int32_t)(room_before - state->level.data);
                    plr->no_transition_back_roompt = no_back_off;
                    ctx.no_transition_back = state->level.data + no_back_off;
                }
            }
        }

        ctx.newx = cur_x;
        ctx.newz = cur_z;
        ctx.objroom = cur_room;
        ctx.stood_in_top = cur_top;
        ctx.hitwall = any_hit;
    }

    plr->stood_in_top = ctx.stood_in_top;

    /* Remember zone before we apply move_object result (detect zone transition) */
    int16_t prev_zone = plr->zone;

    /* Update roompt and zone from move_object result.
     * ctx.objroom may have changed if the player crossed into a new zone.
     * We use zone to set objroom next frame, so zone must stay in sync. */
    if (ctx.objroom && state->level.data) {
        int16_t room_zone_word = player_room_zone_word(ctx.objroom);
        int room_zone_from_ptr = level_zone_index_from_room_ptr(&state->level, ctx.objroom);
        int16_t new_zone = -1;

        plr->roompt = (int32_t)(ctx.objroom - state->level.data);

        /* Room pointer is authoritative. Some levels have incorrect zone words at room+0. */
        if (room_zone_from_ptr >= 0) {
            new_zone = (int16_t)room_zone_from_ptr;
        } else {
            int mapped_room_word = level_connect_to_zone_index(&state->level, room_zone_word);
            if (mapped_room_word >= 0) new_zone = (int16_t)mapped_room_word;
        }

        /* Recovery path: if room pointer/word mapping both fail, infer by point-in-zone. */
        if (new_zone < 0) {
            int guessed_from_point = level_find_zone_for_point(
                &state->level, ctx.newx, ctx.newz, prev_zone);
            if (guessed_from_point >= 0) new_zone = (int16_t)guessed_from_point;
        }

        if (new_zone >= 0 && new_zone < zone_slots)
            plr->zone = new_zone;
    }

    if (plr->zone < 0 || plr->zone >= zone_slots) {
        int recovered_zone = level_find_zone_for_point(
            &state->level, ctx.newx, ctx.newz, prev_zone);
        if (recovered_zone >= 0 && recovered_zone < zone_slots)
            plr->zone = (int16_t)recovered_zone;
    }

    /* Write MoveObject's integer result back to the 16.16 position.
     * The Amiga (AB3DI.s ~line 3137-3140):
     *   move.w newx,PLR1_xoff       ; Write integer to HIGH WORD only
     *   move.w newz,PLR1_zoff       ; Low word (sim fraction) preserved
     *   move.l PLR1_xoff,PLR1s_xoff ; Full 32-bit sync
     *
     * This preserves the fractional part from the simulation while
     * updating only the integer part from MoveObject's wall-slid result. */
    plr->xoff = ((int32_t)ctx.newx << 16) | (plr->xoff & 0x0000FFFF);
    plr->zoff = ((int32_t)ctx.newz << 16) | (plr->zoff & 0x0000FFFF);
    plr->s_xoff = plr->xoff;
    plr->s_zoff = plr->zoff;

    /* 7. Update target Y from room floor height
     * AND set up PointsToRotatePtr / ListOfGraphRooms for this zone.
     *
     * Translated from AB3DI.s .cantmove (~line 3145-3178):
     *   move.l PLR1_Roompt,a0
     *   move.l ToZoneFloor(a0),d0       ; floor height
     *   ...
     *   adda.w #ToZonePts,a0            ; a0 = zone_data + 34
     *   move.w (a0)+,d1                 ; d1 = relative offset to points list
     *   ext.l  d1
     *   add.l  PLR1_Roompt,d1
     *   move.l d1,PLR1_PointsToRotatePtr
     *   tst.w  (a0)+                    ; backdrop flag at offset 36
     *   ...
     *   adda.w #10,a0                   ; a0 = zone_data + 48
     *   move.l a0,PLR1_ListOfGraphRooms
     */
    if (state->level.zone_adds && state->level.data && plr->zone >= 0 &&
        plr->zone < zone_slots) {
        const uint8_t *za = state->level.zone_adds;
        int z = plr->zone;
        int32_t zoff = (int32_t)((za[z*4]<<24)|(za[z*4+1]<<16)|
                       (za[z*4+2]<<8)|za[z*4+3]);
        const uint8_t *zd = state->level.data + zoff;
        int zd_off = plr->stood_in_top ? 8 : 0;
        int32_t floor_h = (int32_t)((zd[2+zd_off]<<24)|(zd[3+zd_off]<<16)|
                          (zd[4+zd_off]<<8)|zd[5+zd_off]);
        plr->s_tyoff = floor_h - plr->s_height;

        /* When we just transitioned zones (e.g. stepped onto stairs), snap Y to new floor
         * immediately so we stand on the step instead of smoothing through it. */
        if (plr->zone != prev_zone && plr->s_tyoff < plr->s_yoff) {
            plr->s_yoff = plr->s_tyoff;
            plr->s_yvel = 0;
            /* Keep shoot/collision-facing PLR_yoff in sync with the snapped sim Y
             * for this same frame (important for auto-aim on stairs). */
            plr->yoff = plr->s_yoff + bob_val;
        }

        /* PointsToRotatePtr = roompt + (int16_t)(zone_data[34..35])
         * ToZonePts (offset 34) contains a 16-bit relative offset. */
        int16_t pts_rel = (int16_t)((zd[34] << 8) | zd[35]);
        plr->points_to_rotate_ptr = zoff + (int32_t)pts_rel;

        /* ListOfGraphRooms = zone_data + 48 (ToListOfGraph) */
        plr->list_of_graph_rooms = zoff + 48;
    }
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

/* Default key bindings (matching original ControlLoop.s CONTROLBUFFER) */
static const KeyBindings default_keys = {
    .turn_left      = 0x4F,  /* Cursor Left */
    .turn_right     = 0x4E,  /* Cursor Right */
    .forward        = 0x4C,  /* Cursor Up */
    .backward       = 0x4D,  /* Cursor Down */
    .fire           = 0x65,  /* Right Alt */
    .operate        = 0x40,  /* Space */
    .run            = 0x61,  /* Right Shift */
    .force_sidestep = 0x67,  /* Right Amiga */
    .sidestep_left  = 0x39,  /* . > */
    .sidestep_right = 0x3A,  /* / ? */
    .duck           = 0x22,  /* D */
    .look_behind    = 0x28,  /* L */
};

void player1_snapshot(GameState *state)
{
    PlayerState *p = &state->plr1;
    /* Convert 16.16 fixed-point to integer for object/AI code.
     * On the Amiga, .w reads the high word of the 32-bit position. */
    p->p_xoff        = p->s_xoff >> 16;
    p->p_zoff        = p->s_zoff >> 16;
    p->p_yoff        = p->s_yoff;
    p->p_height      = p->s_height;
    p->p_angpos      = p->s_angpos;
    p->p_bobble      = p->bobble;
    p->p_clicked     = p->clicked;
    p->clicked       = 0;
    p->p_fire        = p->fire;
    p->p_spctap      = p->spctap;
    p->spctap        = 0;
    p->p_ducked      = p->ducked;
    p->p_gunselected = (int8_t)p->gun_selected;
}

void player2_snapshot(GameState *state)
{
    PlayerState *p = &state->plr2;
    /* Convert 16.16 fixed-point to integer for object/AI code. */
    p->p_xoff        = p->s_xoff >> 16;
    p->p_zoff        = p->s_zoff >> 16;
    p->p_yoff        = p->s_yoff;
    p->p_height      = p->s_height;
    p->p_angpos      = p->s_angpos;
    p->p_bobble      = p->bobble;
    p->p_clicked     = p->clicked;
    p->clicked       = 0;
    p->p_fire        = p->fire;
    p->p_spctap      = p->spctap;
    p->spctap        = 0;
    p->p_ducked      = p->ducked;
    p->p_gunselected = (int8_t)p->gun_selected;
}

void player1_control(GameState *state)
{
    /* Phase 1: Simulation (input -> velocity -> position) */
    player_sim_control(&state->plr1, &state->plr1_control,
                       state->key_map, &default_keys, state, state->temp_frames);

    /* Phase 2: Falling (with water level from zone data) */
    {
        int32_t water_level = 0;
        bool in_water = false;
        int zone_slots = level_zone_slot_count(&state->level);
        if (state->level.zone_adds && state->level.data &&
            state->plr1.zone >= 0 && state->plr1.zone < zone_slots) {
            const uint8_t *za = state->level.zone_adds;
            int z = state->plr1.zone;
            int32_t zoff = (int32_t)((za[z*4]<<24)|(za[z*4+1]<<16)|
                           (za[z*4+2]<<8)|za[z*4+3]);
            const uint8_t *zd = state->level.data + zoff;
            water_level = (int32_t)((zd[18]<<24)|(zd[19]<<16)|(zd[20]<<8)|zd[21]);
            in_water = (water_level != 0);
        }
        player_fall(&state->plr1.s_yoff, &state->plr1.s_yvel,
                    state->plr1.s_tyoff, water_level, in_water);
    }

    /* Phase 3: Snapshot current sim state for full control + shooting.
     * In the original engine this copy happens after control/fall and
     * before PLR1_Control consumes p1_* values. */
    player1_snapshot(state);

    /* Phase 4: Collision + room update */
    player_full_control(&state->plr1, state, 1);
}

void player2_control(GameState *state)
{
    player_sim_control(&state->plr2, &state->plr2_control,
                       state->key_map, &default_keys, state, state->temp_frames);

    {
        int32_t water_level = 0;
        bool in_water = false;
        int zone_slots = level_zone_slot_count(&state->level);
        if (state->level.zone_adds && state->level.data &&
            state->plr2.zone >= 0 && state->plr2.zone < zone_slots) {
            const uint8_t *za = state->level.zone_adds;
            int z = state->plr2.zone;
            int32_t zoff = (int32_t)((za[z*4]<<24)|(za[z*4+1]<<16)|
                           (za[z*4+2]<<8)|za[z*4+3]);
            const uint8_t *zd = state->level.data + zoff;
            water_level = (int32_t)((zd[18]<<24)|(zd[19]<<16)|(zd[20]<<8)|zd[21]);
            in_water = (water_level != 0);
        }
        player_fall(&state->plr2.s_yoff, &state->plr2.s_yvel,
                    state->plr2.s_tyoff, water_level, in_water);
    }

    /* Match original ordering: snapshot after sim/fall, before full control. */
    player2_snapshot(state);

    player_full_control(&state->plr2, state, 2);
}

/* Save file: savegame.bin beside the executable.
 * New format (magic "AB3S") stores full game + level runtime state.
 * Legacy format (magic "AB3D") remains readable for old position-only saves. */
#define SAVE_MAGIC_LEGACY "AB3D"
#define SAVE_MAGIC_FULL   "AB3S"
#define SAVE_VERSION_FULL 3u
#define SAVE_FILE_SUBPATH "savegame.bin"
#define SAVE_MAX_TABLE_ENTRIES 4096
#define SAVE_MAX_CHUNK_BYTES (256u * 1024u * 1024u)
#define SAVE_NASTY_SLOT_SCRATCH_BYTES 64u

typedef struct {
    char     magic[4];
    uint32_t version;
    uint32_t game_state_size;
    uint32_t level_data_bytes;
    uint32_t level_graphics_bytes;
    uint32_t level_clips_bytes;
    uint32_t player_shot_bytes;
    uint32_t nasty_shot_bytes;
    uint32_t object_points_bytes;
    uint32_t door_data_bytes;
    uint32_t switch_data_bytes;
    uint32_t lift_data_bytes;
    uint32_t zone_adds_bytes;
    uint32_t door_wall_list_bytes;
    uint32_t door_wall_offsets_bytes;
    uint32_t lift_wall_list_bytes;
    uint32_t lift_wall_offsets_bytes;
    uint32_t workspace_bytes;
    uint32_t automap_seen_bytes;
    int16_t  current_level;
    int16_t  num_object_points;
    int16_t  num_zones;
    int16_t  num_zone_slots;
    int32_t  num_floor_lines;
    int32_t  num_doors;
    int32_t  num_lifts;
    uint8_t  zone_brightness_le;
    uint8_t  door_data_owned;
    uint8_t  switch_data_owned;
    uint8_t  lift_data_owned;
    uint8_t  zone_adds_owned;
    uint8_t  door_wall_list_owned;
    uint8_t  lift_wall_list_owned;
    uint8_t  reserved0;
    int16_t  bright_anim_values[3];
    uint32_t bright_anim_indices[3];
} FullSaveHeader;

typedef struct {
    uint32_t gfx_off;
    int16_t  x1, z1;
    int16_t  x2, z2;
    uint16_t flags; /* bit0: is_door, bits8..15: door_key_id */
} SaveAutomapSeenWallDisk;

typedef struct {
    bool     valid;
    FullSaveHeader header;
    GameState game_state;
    uint8_t *level_data;
    uint8_t *level_graphics;
    uint8_t *level_clips;
    uint8_t *player_shot_data;
    uint8_t *nasty_shot_data;
    uint8_t *object_points;
    uint8_t *door_data;
    uint8_t *switch_data;
    uint8_t *lift_data;
    uint8_t *zone_adds;
    uint8_t *door_wall_list;
    uint8_t *door_wall_offsets;
    uint8_t *lift_wall_list;
    uint8_t *lift_wall_offsets;
    uint8_t *workspace;
    uint8_t *automap_seen;
} FullSavePending;

static FullSavePending g_full_save_pending;

static int16_t player_save_read_word_be(const uint8_t *p)
{
    return (int16_t)((p[0] << 8) | p[1]);
}

static bool player_save_size_to_u32(size_t value, uint32_t *out)
{
    if (!out) return false;
    if (value > 0xFFFFFFFFu) return false;
    *out = (uint32_t)value;
    return true;
}

static bool player_save_write_exact(FILE *f, const void *data, size_t bytes)
{
    if (bytes == 0) return true;
    return f && data && fwrite(data, 1, bytes, f) == bytes;
}

static bool player_save_read_exact(FILE *f, void *data, size_t bytes)
{
    if (bytes == 0) return true;
    return f && data && fread(data, 1, bytes, f) == bytes;
}

static size_t player_save_table_size_with_sentinel(const uint8_t *table,
                                                    size_t stride)
{
    if (!table || stride == 0) return 0;
    for (int i = 0; i < SAVE_MAX_TABLE_ENTRIES; i++) {
        const uint8_t *entry = table + (size_t)i * stride;
        if (player_save_read_word_be(entry) < 0)
            return (size_t)(i + 1) * stride;
    }
    return 0;
}

static size_t player_save_zone_adds_size(const LevelState *level)
{
    int slots = level_zone_slot_count(level);
    if (!level || !level->zone_adds || slots <= 0) return 0;
    return (size_t)slots * 4u;
}

static size_t player_save_workspace_size(const LevelState *level)
{
    int slots = level_zone_slot_count(level);
    if (!level || !level->workspace || slots <= 0) return 0;
    return (size_t)(slots + 1);
}

static size_t player_save_door_wall_list_size(const LevelState *level)
{
    if (!level || !level->door_wall_list || !level->door_wall_list_offsets ||
        level->num_doors < 0 || level->num_doors > SAVE_MAX_TABLE_ENTRIES)
        return 0;
    return (size_t)level->door_wall_list_offsets[level->num_doors] * 10u;
}

static size_t player_save_lift_wall_list_size(const LevelState *level)
{
    if (!level || !level->lift_wall_list || !level->lift_wall_list_offsets ||
        level->num_lifts < 0 || level->num_lifts > SAVE_MAX_TABLE_ENTRIES)
        return 0;
    return (size_t)level->lift_wall_list_offsets[level->num_lifts] * 10u;
}

static size_t player_save_door_wall_offsets_size(const LevelState *level)
{
    if (!level || !level->door_wall_list_offsets ||
        level->num_doors < 0 || level->num_doors > SAVE_MAX_TABLE_ENTRIES)
        return 0;
    return (size_t)(level->num_doors + 1) * sizeof(uint32_t);
}

static size_t player_save_lift_wall_offsets_size(const LevelState *level)
{
    if (!level || !level->lift_wall_list_offsets ||
        level->num_lifts < 0 || level->num_lifts > SAVE_MAX_TABLE_ENTRIES)
        return 0;
    return (size_t)(level->num_lifts + 1) * sizeof(uint32_t);
}

static void player_save_clear_pending_full_save(void)
{
    free(g_full_save_pending.level_data);
    free(g_full_save_pending.level_graphics);
    free(g_full_save_pending.level_clips);
    free(g_full_save_pending.player_shot_data);
    free(g_full_save_pending.nasty_shot_data);
    free(g_full_save_pending.object_points);
    free(g_full_save_pending.door_data);
    free(g_full_save_pending.switch_data);
    free(g_full_save_pending.lift_data);
    free(g_full_save_pending.zone_adds);
    free(g_full_save_pending.door_wall_list);
    free(g_full_save_pending.door_wall_offsets);
    free(g_full_save_pending.lift_wall_list);
    free(g_full_save_pending.lift_wall_offsets);
    free(g_full_save_pending.workspace);
    free(g_full_save_pending.automap_seen);
    memset(&g_full_save_pending, 0, sizeof(g_full_save_pending));
}

static bool player_save_alloc_read_chunk(FILE *f, uint32_t size, uint8_t **out)
{
    uint8_t *buf = NULL;
    if (!out) return false;
    *out = NULL;
    if (size == 0) return true;
    if (size > SAVE_MAX_CHUNK_BYTES) return false;
    buf = (uint8_t *)malloc((size_t)size);
    if (!buf) return false;
    if (!player_save_read_exact(f, buf, (size_t)size)) {
        free(buf);
        return false;
    }
    *out = buf;
    return true;
}

static bool player_save_apply_chunk(const char *label, void *dst, size_t dst_size,
                                     const uint8_t *src, uint32_t src_size)
{
    if (src_size == 0) return true;
    if (!src || !dst) {
        printf("[PLAYER] load: missing destination for %s (%u bytes)\n",
               label ? label : "chunk", (unsigned)src_size);
        return false;
    }
    if (dst_size < (size_t)src_size) {
        printf("[PLAYER] load: %s too small (%zu < %u)\n",
               label ? label : "chunk", dst_size, (unsigned)src_size);
        return false;
    }
    memcpy(dst, src, (size_t)src_size);
    return true;
}

static bool player_save_read_full_save(FILE *f, GameState *state, const char *path)
{
    FullSaveHeader hdr;
    if (!player_save_read_exact(f, &hdr, sizeof(hdr))) {
        printf("[PLAYER] load: full save header read failed (%s)\n", path);
        return false;
    }
    if (memcmp(hdr.magic, SAVE_MAGIC_FULL, 4) != 0 ||
        hdr.version < 2u || hdr.version > SAVE_VERSION_FULL) {
        printf("[PLAYER] load: unsupported full save format in %s\n", path);
        return false;
    }
    if (hdr.game_state_size != (uint32_t)sizeof(GameState)) {
        printf("[PLAYER] load: save game state size mismatch (%u != %u)\n",
               (unsigned)hdr.game_state_size, (unsigned)sizeof(GameState));
        return false;
    }

    player_save_clear_pending_full_save();
    g_full_save_pending.header = hdr;

    if (!player_save_read_exact(f, &g_full_save_pending.game_state, sizeof(GameState)))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.level_data_bytes, &g_full_save_pending.level_data))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.level_graphics_bytes, &g_full_save_pending.level_graphics))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.level_clips_bytes, &g_full_save_pending.level_clips))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.player_shot_bytes, &g_full_save_pending.player_shot_data))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.nasty_shot_bytes, &g_full_save_pending.nasty_shot_data))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.object_points_bytes, &g_full_save_pending.object_points))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.door_data_bytes, &g_full_save_pending.door_data))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.switch_data_bytes, &g_full_save_pending.switch_data))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.lift_data_bytes, &g_full_save_pending.lift_data))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.zone_adds_bytes, &g_full_save_pending.zone_adds))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.door_wall_list_bytes, &g_full_save_pending.door_wall_list))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.door_wall_offsets_bytes, &g_full_save_pending.door_wall_offsets))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.lift_wall_list_bytes, &g_full_save_pending.lift_wall_list))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.lift_wall_offsets_bytes, &g_full_save_pending.lift_wall_offsets))
        goto fail;
    if (!player_save_alloc_read_chunk(f, hdr.workspace_bytes, &g_full_save_pending.workspace))
        goto fail;
    if (hdr.version >= 3u) {
        if (!player_save_alloc_read_chunk(f, hdr.automap_seen_bytes, &g_full_save_pending.automap_seen))
            goto fail;
    }

    g_full_save_pending.valid = true;
    state->current_level = hdr.current_level;
    printf("[PLAYER] load: full save read (level %d), reloading level to restore state\n",
           (int)hdr.current_level);
    return true;

fail:
    player_save_clear_pending_full_save();
    printf("[PLAYER] load: full save read failed (truncated %s?)\n", path);
    return false;
}

static bool player_apply_pending_full_save_after_level_load(GameState *state)
{
    const FullSaveHeader *hdr;
    LevelState live_level;
    size_t door_data_dst;
    size_t switch_data_dst;
    size_t lift_data_dst;
    size_t zone_adds_dst;
    size_t door_wall_list_dst;
    size_t door_wall_offsets_dst;
    size_t lift_wall_list_dst;
    size_t lift_wall_offsets_dst;
    size_t workspace_dst;

    if (!g_full_save_pending.valid) return false;

    hdr = &g_full_save_pending.header;
    live_level = state->level;
    /* Preserve ab3d.ini prefs from the running session (saved GameState may predate new fields). */
    int16_t ini_cfg_start_level = state->cfg_start_level;
    bool    ini_infinite_health = state->infinite_health;
    bool    ini_infinite_ammo = state->infinite_ammo;
    bool    ini_cfg_all_weapons = state->cfg_all_weapons;
    bool    ini_cfg_all_keys = state->cfg_all_keys;
    int16_t ini_cfg_render_width = state->cfg_render_width;
    int16_t ini_cfg_render_height = state->cfg_render_height;
    int16_t ini_cfg_supersampling = state->cfg_supersampling;
    bool    ini_cfg_render_threads = state->cfg_render_threads;
    int16_t ini_cfg_render_threads_max = state->cfg_render_threads_max;
    int16_t ini_cfg_volume = state->cfg_volume;
    int16_t ini_cfg_y_proj_scale = state->cfg_y_proj_scale;
    bool    ini_cfg_billboard_sprite_rendering_enhancement = state->cfg_billboard_sprite_rendering_enhancement;
    bool    ini_cfg_weapon_draw = state->cfg_weapon_draw;
    bool    ini_cfg_post_tint = state->cfg_post_tint;
    bool    ini_cfg_weapon_post_gl = state->cfg_weapon_post_gl;
    bool    ini_cfg_show_fps = state->cfg_show_fps;

    *state = g_full_save_pending.game_state;
    state->level = live_level;
    state->view_list_of_graph_rooms = NULL;

    state->cfg_start_level = ini_cfg_start_level;
    state->infinite_health = ini_infinite_health;
    state->infinite_ammo = ini_infinite_ammo;
    state->cfg_all_weapons = ini_cfg_all_weapons;
    state->cfg_all_keys = ini_cfg_all_keys;
    state->cfg_render_width = ini_cfg_render_width;
    state->cfg_render_height = ini_cfg_render_height;
    state->cfg_supersampling = ini_cfg_supersampling;
    state->cfg_render_threads = ini_cfg_render_threads;
    state->cfg_render_threads_max = ini_cfg_render_threads_max;
    state->cfg_volume = ini_cfg_volume;
    state->cfg_y_proj_scale = ini_cfg_y_proj_scale;
    state->cfg_billboard_sprite_rendering_enhancement = ini_cfg_billboard_sprite_rendering_enhancement;
    state->cfg_weapon_draw = ini_cfg_weapon_draw;
    state->cfg_post_tint = ini_cfg_post_tint;
    state->cfg_weapon_post_gl = ini_cfg_weapon_post_gl;
    state->cfg_show_fps = ini_cfg_show_fps;

    door_data_dst = player_save_table_size_with_sentinel(state->level.door_data, 22u);
    switch_data_dst = player_save_table_size_with_sentinel(state->level.switch_data, 14u);
    lift_data_dst = player_save_table_size_with_sentinel(state->level.lift_data, 20u);
    zone_adds_dst = player_save_zone_adds_size(&state->level);
    door_wall_list_dst = player_save_door_wall_list_size(&state->level);
    door_wall_offsets_dst = player_save_door_wall_offsets_size(&state->level);
    lift_wall_list_dst = player_save_lift_wall_list_size(&state->level);
    lift_wall_offsets_dst = player_save_lift_wall_offsets_size(&state->level);
    workspace_dst = player_save_workspace_size(&state->level);

    if (!player_save_apply_chunk("level data",
            state->level.data, state->level.data_byte_count,
            g_full_save_pending.level_data, hdr->level_data_bytes))
        goto fail;
    if (!player_save_apply_chunk("level graphics",
            state->level.graphics, state->level.graphics_byte_count,
            g_full_save_pending.level_graphics, hdr->level_graphics_bytes))
        goto fail;
    if (!player_save_apply_chunk("level clips",
            state->level.clips, state->level.clips_byte_count,
            g_full_save_pending.level_clips, hdr->level_clips_bytes))
        goto fail;
    if (!player_save_apply_chunk("player shots",
            state->level.player_shot_data, (size_t)PLAYER_SHOT_SLOT_COUNT * OBJECT_SIZE,
            g_full_save_pending.player_shot_data, hdr->player_shot_bytes))
        goto fail;
    if (!player_save_apply_chunk("nasty shots",
            state->level.nasty_shot_data,
            (size_t)NASTY_SHOT_SLOT_COUNT * (OBJECT_SIZE + SAVE_NASTY_SLOT_SCRATCH_BYTES),
            g_full_save_pending.nasty_shot_data, hdr->nasty_shot_bytes))
        goto fail;
    if (!player_save_apply_chunk("object points",
            state->level.object_points,
            (state->level.num_object_points > 0) ? (size_t)state->level.num_object_points * 8u : 0u,
            g_full_save_pending.object_points, hdr->object_points_bytes))
        goto fail;
    if (!player_save_apply_chunk("door table",
            state->level.door_data, door_data_dst,
            g_full_save_pending.door_data, hdr->door_data_bytes))
        goto fail;
    if (!player_save_apply_chunk("switch table",
            state->level.switch_data, switch_data_dst,
            g_full_save_pending.switch_data, hdr->switch_data_bytes))
        goto fail;
    if (!player_save_apply_chunk("lift table",
            state->level.lift_data, lift_data_dst,
            g_full_save_pending.lift_data, hdr->lift_data_bytes))
        goto fail;
    if (!player_save_apply_chunk("zone adds",
            state->level.zone_adds, zone_adds_dst,
            g_full_save_pending.zone_adds, hdr->zone_adds_bytes))
        goto fail;
    if (!player_save_apply_chunk("door wall list",
            state->level.door_wall_list, door_wall_list_dst,
            g_full_save_pending.door_wall_list, hdr->door_wall_list_bytes))
        goto fail;
    if (!player_save_apply_chunk("door wall offsets",
            state->level.door_wall_list_offsets, door_wall_offsets_dst,
            g_full_save_pending.door_wall_offsets, hdr->door_wall_offsets_bytes))
        goto fail;
    if (!player_save_apply_chunk("lift wall list",
            state->level.lift_wall_list, lift_wall_list_dst,
            g_full_save_pending.lift_wall_list, hdr->lift_wall_list_bytes))
        goto fail;
    if (!player_save_apply_chunk("lift wall offsets",
            state->level.lift_wall_list_offsets, lift_wall_offsets_dst,
            g_full_save_pending.lift_wall_offsets, hdr->lift_wall_offsets_bytes))
        goto fail;
    if (!player_save_apply_chunk("workspace",
            state->level.workspace, workspace_dst,
            g_full_save_pending.workspace, hdr->workspace_bytes))
        goto fail;

    /* Automap (v3+): restore seen wall list. */
    if (hdr->version >= 3u && hdr->automap_seen_bytes > 0) {
        uint32_t bytes = hdr->automap_seen_bytes;
        if (bytes % (uint32_t)sizeof(SaveAutomapSeenWallDisk) != 0u) {
            printf("[PLAYER] load: automap chunk size invalid (%u)\n", (unsigned)bytes);
            goto fail;
        }
        uint32_t count = bytes / (uint32_t)sizeof(SaveAutomapSeenWallDisk);
        if (count > 200000u) {
            printf("[PLAYER] load: automap chunk too large (%u)\n", (unsigned)count);
            goto fail;
        }

        renderer_automap_lock();
        free(state->level.automap_seen_walls);
        free(state->level.automap_seen_hash);
        state->level.automap_seen_walls = NULL;
        state->level.automap_seen_hash = NULL;
        state->level.automap_seen_count = 0;
        state->level.automap_seen_cap = 0;
        state->level.automap_seen_hash_cap = 0;

        if (count > 0) {
            state->level.automap_seen_walls = (AutomapSeenWall *)malloc((size_t)count * sizeof(AutomapSeenWall));
            if (!state->level.automap_seen_walls) {
                renderer_automap_unlock();
                goto fail;
            }
            state->level.automap_seen_cap = count;

            const SaveAutomapSeenWallDisk *disk = (const SaveAutomapSeenWallDisk *)g_full_save_pending.automap_seen;
            for (uint32_t i = 0; i < count; i++) {
                AutomapSeenWall *w = &state->level.automap_seen_walls[i];
                w->gfx_off = disk[i].gfx_off;
                w->x1 = disk[i].x1; w->z1 = disk[i].z1;
                w->x2 = disk[i].x2; w->z2 = disk[i].z2;
                w->is_door = (uint8_t)((disk[i].flags & 0x0001u) ? 1u : 0u);
                w->door_key_id = (uint8_t)((disk[i].flags >> 8) & 0x00FFu);
                w->reserved = 0;
            }
            state->level.automap_seen_count = count;

            uint32_t want = 1024u;
            while (want < count * 2u) want <<= 1u;
            state->level.automap_seen_hash = (uint32_t *)calloc((size_t)want, sizeof(uint32_t));
            if (!state->level.automap_seen_hash) {
                renderer_automap_unlock();
                goto fail;
            }
            state->level.automap_seen_hash_cap = want;

            /* Same seen-wall key as renderer.c. */
            for (uint32_t i = 0; i < count; i++) {
                const AutomapSeenWall *w = &state->level.automap_seen_walls[i];
                uint32_t keyp1 = renderer_automap_seen_key_plus1(
                    w->gfx_off, w->x1, w->z1, w->x2, w->z2);
                uint32_t mask = want - 1u;
                uint32_t h = keyp1;
                h ^= h >> 16;
                h *= 0x7FEB352Du;
                h ^= h >> 15;
                h *= 0x846CA68Bu;
                h ^= h >> 16;
                h &= mask;
                while (state->level.automap_seen_hash[h] != 0u) h = (h + 1u) & mask;
                state->level.automap_seen_hash[h] = keyp1;
            }
        }
        renderer_automap_unlock();
    }

    state->level.num_object_points = hdr->num_object_points;
    state->level.num_zones = hdr->num_zones;
    state->level.num_zone_slots = hdr->num_zone_slots;
    state->level.num_floor_lines = hdr->num_floor_lines;
    state->level.num_doors = hdr->num_doors;
    state->level.num_lifts = hdr->num_lifts;
    state->level.zone_brightness_le = hdr->zone_brightness_le ? true : false;
    state->level.door_data_owned = hdr->door_data_owned ? true : false;
    state->level.switch_data_owned = hdr->switch_data_owned ? true : false;
    state->level.lift_data_owned = hdr->lift_data_owned ? true : false;
    state->level.zone_adds_owned = hdr->zone_adds_owned ? true : false;
    state->level.door_wall_list_owned = hdr->door_wall_list_owned ? true : false;
    state->level.lift_wall_list_owned = hdr->lift_wall_list_owned ? true : false;
    memcpy(state->level.bright_anim_values, hdr->bright_anim_values,
           sizeof(state->level.bright_anim_values));
    memcpy(state->level.bright_anim_indices, hdr->bright_anim_indices,
           sizeof(state->level.bright_anim_indices));

    if (state->level.nasty_shot_data) {
        state->level.other_nasty_data =
            state->level.nasty_shot_data + (size_t)NASTY_SHOT_SLOT_COUNT * OBJECT_SIZE;
    } else {
        state->level.other_nasty_data = NULL;
    }

    state->debug_f9_need_level_reload = false;
    state->f9_pending_apply_save = false;

    player_save_clear_pending_full_save();
    printf("[PLAYER] load: full game + level state restored\n");
    return true;

fail:
    player_save_clear_pending_full_save();
    printf("[PLAYER] load: full save apply failed\n");
    return false;
}

void player_save_position(GameState *state)
{
    char path[512];
    FILE *f = NULL;
    FullSaveHeader hdr;
    GameState game_snapshot;
    uint32_t level_data_bytes = 0, level_graphics_bytes = 0, level_clips_bytes = 0;
    uint32_t player_shot_bytes = 0, nasty_shot_bytes = 0, object_points_bytes = 0;
    uint32_t door_data_bytes = 0, switch_data_bytes = 0, lift_data_bytes = 0;
    uint32_t zone_adds_bytes = 0;
    uint32_t door_wall_list_bytes = 0, door_wall_offsets_bytes = 0;
    uint32_t lift_wall_list_bytes = 0, lift_wall_offsets_bytes = 0;
    uint32_t workspace_bytes = 0;
    uint32_t automap_seen_bytes = 0;
    size_t door_data_sz, switch_data_sz, lift_data_sz;

    if (!state) return;

    if (state->level.data && state->level.data_byte_count == 0) {
        printf("[PLAYER] save: level data size unknown; aborting save\n");
        return;
    }
    if (state->level.graphics && state->level.graphics_byte_count == 0) {
        printf("[PLAYER] save: level graphics size unknown; aborting save\n");
        return;
    }
    if (state->level.clips && state->level.clips_byte_count == 0) {
        printf("[PLAYER] save: level clips size unknown; aborting save\n");
        return;
    }

    door_data_sz = player_save_table_size_with_sentinel(state->level.door_data, 22u);
    switch_data_sz = player_save_table_size_with_sentinel(state->level.switch_data, 14u);
    lift_data_sz = player_save_table_size_with_sentinel(state->level.lift_data, 20u);
    if (state->level.door_data && door_data_sz == 0) {
        printf("[PLAYER] save: door table missing terminator; aborting save\n");
        return;
    }
    if (state->level.switch_data && switch_data_sz == 0) {
        printf("[PLAYER] save: switch table missing terminator; aborting save\n");
        return;
    }
    if (state->level.lift_data && lift_data_sz == 0) {
        printf("[PLAYER] save: lift table missing terminator; aborting save\n");
        return;
    }

    if (!player_save_size_to_u32(state->level.data ? state->level.data_byte_count : 0, &level_data_bytes) ||
        !player_save_size_to_u32(state->level.graphics ? state->level.graphics_byte_count : 0, &level_graphics_bytes) ||
        !player_save_size_to_u32(state->level.clips ? state->level.clips_byte_count : 0, &level_clips_bytes) ||
        !player_save_size_to_u32(state->level.player_shot_data ? (size_t)PLAYER_SHOT_SLOT_COUNT * OBJECT_SIZE : 0, &player_shot_bytes) ||
        !player_save_size_to_u32(state->level.nasty_shot_data ? (size_t)NASTY_SHOT_SLOT_COUNT * (OBJECT_SIZE + SAVE_NASTY_SLOT_SCRATCH_BYTES) : 0, &nasty_shot_bytes) ||
        !player_save_size_to_u32((state->level.object_points && state->level.num_object_points > 0) ? (size_t)state->level.num_object_points * 8u : 0u, &object_points_bytes) ||
        !player_save_size_to_u32(door_data_sz, &door_data_bytes) ||
        !player_save_size_to_u32(switch_data_sz, &switch_data_bytes) ||
        !player_save_size_to_u32(lift_data_sz, &lift_data_bytes) ||
        !player_save_size_to_u32(player_save_zone_adds_size(&state->level), &zone_adds_bytes) ||
        !player_save_size_to_u32(player_save_door_wall_list_size(&state->level), &door_wall_list_bytes) ||
        !player_save_size_to_u32(player_save_door_wall_offsets_size(&state->level), &door_wall_offsets_bytes) ||
        !player_save_size_to_u32(player_save_lift_wall_list_size(&state->level), &lift_wall_list_bytes) ||
        !player_save_size_to_u32(player_save_lift_wall_offsets_size(&state->level), &lift_wall_offsets_bytes) ||
        !player_save_size_to_u32(player_save_workspace_size(&state->level), &workspace_bytes)) {
        printf("[PLAYER] save: payload too large; aborting save\n");
        return;
    }

    {
        size_t count = (size_t)state->level.automap_seen_count;
        size_t bytes = count * sizeof(SaveAutomapSeenWallDisk);
        if (!player_save_size_to_u32(bytes, &automap_seen_bytes)) {
            printf("[PLAYER] save: automap payload too large; aborting save\n");
            return;
        }
    }

    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, SAVE_MAGIC_FULL, 4);
    hdr.version = SAVE_VERSION_FULL;
    hdr.game_state_size = (uint32_t)sizeof(GameState);
    hdr.level_data_bytes = level_data_bytes;
    hdr.level_graphics_bytes = level_graphics_bytes;
    hdr.level_clips_bytes = level_clips_bytes;
    hdr.player_shot_bytes = player_shot_bytes;
    hdr.nasty_shot_bytes = nasty_shot_bytes;
    hdr.object_points_bytes = object_points_bytes;
    hdr.door_data_bytes = door_data_bytes;
    hdr.switch_data_bytes = switch_data_bytes;
    hdr.lift_data_bytes = lift_data_bytes;
    hdr.zone_adds_bytes = zone_adds_bytes;
    hdr.door_wall_list_bytes = door_wall_list_bytes;
    hdr.door_wall_offsets_bytes = door_wall_offsets_bytes;
    hdr.lift_wall_list_bytes = lift_wall_list_bytes;
    hdr.lift_wall_offsets_bytes = lift_wall_offsets_bytes;
    hdr.workspace_bytes = workspace_bytes;
    hdr.automap_seen_bytes = automap_seen_bytes;
    hdr.current_level = state->current_level;
    hdr.num_object_points = state->level.num_object_points;
    hdr.num_zones = state->level.num_zones;
    hdr.num_zone_slots = state->level.num_zone_slots;
    hdr.num_floor_lines = state->level.num_floor_lines;
    hdr.num_doors = state->level.num_doors;
    hdr.num_lifts = state->level.num_lifts;
    hdr.zone_brightness_le = state->level.zone_brightness_le ? 1u : 0u;
    hdr.door_data_owned = state->level.door_data_owned ? 1u : 0u;
    hdr.switch_data_owned = state->level.switch_data_owned ? 1u : 0u;
    hdr.lift_data_owned = state->level.lift_data_owned ? 1u : 0u;
    hdr.zone_adds_owned = state->level.zone_adds_owned ? 1u : 0u;
    hdr.door_wall_list_owned = state->level.door_wall_list_owned ? 1u : 0u;
    hdr.lift_wall_list_owned = state->level.lift_wall_list_owned ? 1u : 0u;
    memcpy(hdr.bright_anim_values, state->level.bright_anim_values, sizeof(hdr.bright_anim_values));
    memcpy(hdr.bright_anim_indices, state->level.bright_anim_indices, sizeof(hdr.bright_anim_indices));

    io_make_exe_path(path, sizeof(path), SAVE_FILE_SUBPATH);
    f = fopen(path, "wb");
    if (!f) {
        printf("[PLAYER] save: could not open %s for write\n", path);
        return;
    }

    game_snapshot = *state;
    if (!player_save_write_exact(f, &hdr, sizeof(hdr))) goto fail;
    if (!player_save_write_exact(f, &game_snapshot, sizeof(game_snapshot))) goto fail;
    if (!player_save_write_exact(f, state->level.data, hdr.level_data_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.graphics, hdr.level_graphics_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.clips, hdr.level_clips_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.player_shot_data, hdr.player_shot_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.nasty_shot_data, hdr.nasty_shot_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.object_points, hdr.object_points_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.door_data, hdr.door_data_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.switch_data, hdr.switch_data_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.lift_data, hdr.lift_data_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.zone_adds, hdr.zone_adds_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.door_wall_list, hdr.door_wall_list_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.door_wall_list_offsets, hdr.door_wall_offsets_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.lift_wall_list, hdr.lift_wall_list_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.lift_wall_list_offsets, hdr.lift_wall_offsets_bytes)) goto fail;
    if (!player_save_write_exact(f, state->level.workspace, hdr.workspace_bytes)) goto fail;
    if (hdr.automap_seen_bytes > 0) {
        size_t count = (size_t)state->level.automap_seen_count;
        SaveAutomapSeenWallDisk *tmp = (SaveAutomapSeenWallDisk *)malloc(count * sizeof(SaveAutomapSeenWallDisk));
        if (!tmp) goto fail;
        for (size_t i = 0; i < count; i++) {
            const AutomapSeenWall *w = &state->level.automap_seen_walls[i];
            tmp[i].gfx_off = w->gfx_off;
            tmp[i].x1 = w->x1; tmp[i].z1 = w->z1;
            tmp[i].x2 = w->x2; tmp[i].z2 = w->z2;
            tmp[i].flags = (uint16_t)((w->is_door ? 1u : 0u) | ((uint16_t)w->door_key_id << 8));
        }
        bool ok = player_save_write_exact(f, tmp, count * sizeof(SaveAutomapSeenWallDisk));
        free(tmp);
        if (!ok) goto fail;
    }

    fclose(f);
    printf("[PLAYER] save: full game + level state (level %d) written to %s\n",
           (int)state->current_level, path);
    return;

fail:
    fclose(f);
    printf("[PLAYER] save: write failed\n");
}

static void player_sync_loaded_player(GameState *state, PlayerState *plr, int plr_num)
{
    int zone_slots = level_zone_slot_count(&state->level);
    int16_t saved_zone = plr->zone;

    int resolved_zone = level_find_zone_for_point(
        &state->level, plr->xoff >> 16, plr->zoff >> 16, saved_zone);
    if (resolved_zone < 0)
        resolved_zone = level_connect_to_zone_index(&state->level, saved_zone);
    if (resolved_zone < 0 && saved_zone >= 0 && saved_zone < zone_slots)
        resolved_zone = saved_zone;

    if (!state->level.zone_adds || !state->level.data ||
        resolved_zone < 0 || resolved_zone >= zone_slots) {
        printf("[PLAYER] load: plr%d unresolved zone %d at (%d,%d)\n",
               plr_num, (int)saved_zone,
               (int)(plr->xoff >> 16), (int)(plr->zoff >> 16));
        return;
    }

    if (resolved_zone != saved_zone) {
        printf("[PLAYER] load: plr%d remapped zone %d -> %d at (%d,%d)\n",
               plr_num, (int)saved_zone, resolved_zone,
               (int)(plr->xoff >> 16), (int)(plr->zoff >> 16));
    }

    plr->zone = (int16_t)resolved_zone;
    plr->s_zone = plr->zone;

    {
        const uint8_t *za = state->level.zone_adds;
        int z = plr->zone;
        int32_t zoff = (int32_t)((za[z*4]<<24)|(za[z*4+1]<<16)|(za[z*4+2]<<8)|za[z*4+3]);
        const uint8_t *zd = state->level.data + zoff;
        int32_t roof_h = (int32_t)((zd[6]<<24)|(zd[7]<<16)|(zd[8]<<8)|zd[9]);
        int zd_off;
        int32_t floor_h;

        plr->stood_in_top = (plr->yoff < roof_h) ? 1 : 0;
        zd_off = plr->stood_in_top ? 8 : 0;
        floor_h = (int32_t)((zd[2+zd_off]<<24)|(zd[3+zd_off]<<16)|
                            (zd[4+zd_off]<<8)|zd[5+zd_off]);

        plr->s_tyoff = floor_h - PLAYER_HEIGHT;
        plr->tyoff = plr->s_tyoff;
        plr->roompt = zoff;
        plr->old_roompt = zoff;
        plr->s_roompt = zoff;
        plr->s_old_roompt = zoff;

        {
            int16_t pts = (int16_t)((zd[34] << 8) | zd[35]);
            int32_t pts_ptr = zoff + (int32_t)pts;
            int32_t graph_ptr = zoff + 48;
            plr->points_to_rotate_ptr = pts_ptr;
            plr->s_points_to_rotate_ptr = pts_ptr;
            plr->list_of_graph_rooms = graph_ptr;
            plr->s_list_of_graph_rooms = graph_ptr;
        }
    }
}

static void player_seed_facing_and_snapshots(GameState *state)
{
    PlayerState *plrs[2] = { &state->plr1, &state->plr2 };
    for (int i = 0; i < 2; i++) {
        PlayerState *plr = plrs[i];
        int16_t ang = (int16_t)(plr->angpos & ANGLE_MASK);
        plr->angpos = ang;
        plr->s_angpos = ang;
        plr->sinval = sin_lookup(ang);
        plr->cosval = cos_lookup(ang);
        plr->s_sinval = plr->sinval;
        plr->s_cosval = plr->cosval;
        plr->oldxoff = plr->xoff;
        plr->oldzoff = plr->zoff;
        plr->s_oldxoff = plr->s_xoff;
        plr->s_oldzoff = plr->s_zoff;
    }
    player1_snapshot(state);
    player2_snapshot(state);
}

void player_apply_save_payload_after_level_load(GameState *state)
{
    if (player_apply_pending_full_save_after_level_load(state))
        return;

    state->plr1.s_xoff = state->plr1.xoff;
    state->plr1.s_zoff = state->plr1.zoff;
    state->plr1.s_yoff = state->plr1.yoff;
    state->plr1.s_angpos = state->plr1.angpos;
    state->plr2.s_xoff = state->plr2.xoff;
    state->plr2.s_zoff = state->plr2.zoff;
    state->plr2.s_yoff = state->plr2.yoff;
    state->plr2.s_angpos = state->plr2.angpos;
    player_sync_loaded_player(state, &state->plr1, 1);
    player_sync_loaded_player(state, &state->plr2, 2);
    /* View/collision use angpos + sin/cos; first frame may render before player_control */
    player_seed_facing_and_snapshots(state);
}

PlayerSaveLoadResult player_load_save_from_file(GameState *state)
{
    char path[512];
    int16_t file_level = -1;
    bool has_level = false;
    char magic[4];

    player_save_clear_pending_full_save();
    io_make_exe_path(path, sizeof(path), SAVE_FILE_SUBPATH);
    FILE *f = fopen(path, "rb");
    if (!f) return PLAYER_SAVE_LOAD_FAILED;

    if (fread(magic, 1, 4, f) != 4) {
        fclose(f);
        printf("[PLAYER] load: read failed (truncated %s?)\n", path);
        return PLAYER_SAVE_LOAD_FAILED;
    }

    if (memcmp(magic, SAVE_MAGIC_FULL, 4) == 0) {
        rewind(f);
        if (!player_save_read_full_save(f, state, path)) {
            fclose(f);
            return PLAYER_SAVE_LOAD_FAILED;
        }
        fclose(f);
        return PLAYER_SAVE_LOAD_NEED_LEVEL_RELOAD;
    }

    if (memcmp(magic, SAVE_MAGIC_LEGACY, 4) != 0) {
        fclose(f);
        printf("[PLAYER] load: invalid or missing magic in %s\n", path);
        return PLAYER_SAVE_LOAD_FAILED;
    }
    if (fread(&state->plr1.xoff, sizeof(state->plr1.xoff), 1, f) != 1) goto load_fail;
    if (fread(&state->plr1.zoff, sizeof(state->plr1.zoff), 1, f) != 1) goto load_fail;
    if (fread(&state->plr1.zone, sizeof(state->plr1.zone), 1, f) != 1) goto load_fail;
    if (fread(&state->plr1.angpos, sizeof(state->plr1.angpos), 1, f) != 1) goto load_fail;
    if (fread(&state->plr1.yoff, sizeof(state->plr1.yoff), 1, f) != 1) goto load_fail;
    if (fread(&state->plr2.xoff, sizeof(state->plr2.xoff), 1, f) != 1) goto load_fail;
    if (fread(&state->plr2.zoff, sizeof(state->plr2.zoff), 1, f) != 1) goto load_fail;
    if (fread(&state->plr2.zone, sizeof(state->plr2.zone), 1, f) != 1) goto load_fail;
    if (fread(&state->plr2.angpos, sizeof(state->plr2.angpos), 1, f) != 1) goto load_fail;
    if (fread(&state->plr2.yoff, sizeof(state->plr2.yoff), 1, f) != 1) goto load_fail;

    has_level = (fread(&file_level, sizeof(file_level), 1, f) == 1);
    fclose(f);

    if (has_level && file_level >= 0 && file_level < MAX_LEVELS &&
        file_level != state->current_level) {
        state->current_level = file_level;
        printf("[PLAYER] load: save is level %d; reloading level, then applying position\n",
               (int)file_level);
        return PLAYER_SAVE_LOAD_NEED_LEVEL_RELOAD;
    }

    /* Same level, or old save without level tail: apply immediately */
    player_apply_save_payload_after_level_load(state);
    printf("[PLAYER] load: position/orientation restored from %s\n", path);
    return PLAYER_SAVE_LOAD_APPLIED;

load_fail:
    fclose(f);
    printf("[PLAYER] load: read failed (truncated %s?)\n", path);
    return PLAYER_SAVE_LOAD_FAILED;
}

void player_init_from_level(GameState *state)
{
    /* Default values */
    state->plr1.xoff = 0;
    state->plr1.zoff = 0;
    state->plr1.yoff = -PLAYER_HEIGHT;
    state->plr1.s_xoff = 0;
    state->plr1.s_zoff = 0;
    state->plr1.s_yoff = -PLAYER_HEIGHT;
    state->plr1.s_height = PLAYER_HEIGHT;
    state->plr1.s_targheight = PLAYER_HEIGHT;

    state->plr2.xoff = 0;
    state->plr2.zoff = 0;
    state->plr2.yoff = -PLAYER_HEIGHT;
    state->plr2.s_xoff = 0;
    state->plr2.s_zoff = 0;
    state->plr2.s_yoff = -PLAYER_HEIGHT;
    state->plr2.s_height = PLAYER_HEIGHT;
    state->plr2.s_targheight = PLAYER_HEIGHT;

    if (state->level.data) {
        /* Parse start positions from level data header (Amiga format).
         * Real Amiga header layout (from AB3DI.s):
         *   Byte  0: PLR1 start X (word)
         *   Byte  2: PLR1 start Z (word)
         *   Byte  4: PLR1 start zone (word)
         *   Byte  6: PLR2 start X (word)
         *   Byte  8: PLR2 start Z (word)
         *   Byte 10: PLR2 start zone (word)
         *   Byte 12: num control pts (word, unused)
         *   Byte 14: num points (word)
         * Note: Player angle is NOT in the header; initialized to 0.
         */
        const uint8_t *hdr = state->level.data;

        /* Positions are stored in 16.16 fixed-point internally,
         * matching the Amiga convention (move.w to high word of dc.l).
         * Upper 16 bits = world coordinate, lower 16 bits = sub-pixel. */
        state->plr1.xoff = (int32_t)(int16_t)((hdr[0] << 8) | hdr[1]) << 16;
        state->plr1.zoff = (int32_t)(int16_t)((hdr[2] << 8) | hdr[3]) << 16;
        state->plr1.zone = (int16_t)((hdr[4] << 8) | hdr[5]);
        state->plr1.angpos = 0;  /* Not stored in real Amiga header */
        state->plr1.s_xoff = state->plr1.xoff;
        state->plr1.s_zoff = state->plr1.zoff;

        state->plr2.xoff = (int32_t)(int16_t)((hdr[6] << 8) | hdr[7]) << 16;
        state->plr2.zoff = (int32_t)(int16_t)((hdr[8] << 8) | hdr[9]) << 16;
        state->plr2.zone = (int16_t)((hdr[10] << 8) | hdr[11]);
        state->plr2.angpos = 0;  /* Not stored in real Amiga header */
        state->plr2.s_xoff = state->plr2.xoff;
        state->plr2.s_zoff = state->plr2.zoff;

        /* Set initial Y from zone floor height + initialize roompt.
         * Translated from AB3DI.s: Initial PLR1_Roompt / PLR2_Roompt setup */
        int zone_slots = level_zone_slot_count(&state->level);
        if (state->level.zone_adds && state->plr1.zone >= 0 &&
            state->plr1.zone < zone_slots) {
            const uint8_t *za = state->level.zone_adds;
            int z = state->plr1.zone;
            int32_t zoff = (int32_t)((za[z*4]<<24)|(za[z*4+1]<<16)|
                           (za[z*4+2]<<8)|za[z*4+3]);
            const uint8_t *zd = state->level.data + zoff;
            int32_t floor_h = (int32_t)((zd[2]<<24)|(zd[3]<<16)|(zd[4]<<8)|zd[5]);
            state->plr1.s_tyoff = floor_h - PLAYER_HEIGHT;
            state->plr1.yoff = floor_h - PLAYER_HEIGHT;
            state->plr1.s_yoff = state->plr1.yoff;
            state->plr1.roompt = zoff;
            /* PointsToRotatePtr = roompt + (int16) zone_data[34..35] */
            int16_t pts1 = (int16_t)((zd[34] << 8) | zd[35]);
            state->plr1.points_to_rotate_ptr = zoff + (int32_t)pts1;
            /* ListOfGraphRooms = zone_data + 48 (ToListOfGraph) */
            state->plr1.list_of_graph_rooms = zoff + 48;
        }
        if (state->level.zone_adds && state->plr2.zone >= 0 &&
            state->plr2.zone < zone_slots) {
            const uint8_t *za = state->level.zone_adds;
            int z = state->plr2.zone;
            int32_t zoff = (int32_t)((za[z*4]<<24)|(za[z*4+1]<<16)|
                           (za[z*4+2]<<8)|za[z*4+3]);
            const uint8_t *zd = state->level.data + zoff;
            int32_t floor_h = (int32_t)((zd[2]<<24)|(zd[3]<<16)|(zd[4]<<8)|zd[5]);
            state->plr2.s_tyoff = floor_h - PLAYER_HEIGHT;
            state->plr2.yoff = floor_h - PLAYER_HEIGHT;
            state->plr2.s_yoff = state->plr2.yoff;
            state->plr2.roompt = zoff;
            /* PointsToRotatePtr = roompt + (int16) zone_data[34..35] */
            int16_t pts2 = (int16_t)((zd[34] << 8) | zd[35]);
            state->plr2.points_to_rotate_ptr = zoff + (int32_t)pts2;
            /* ListOfGraphRooms = zone_data + 48 (ToListOfGraph) */
            state->plr2.list_of_graph_rooms = zoff + 48;
        }

        printf("[PLAYER] init_from_level: PLR1 at (%d,%d) zone %d yoff=%d tyoff=%d, "
               "PLR2 at (%d,%d) zone %d\n",
               (int)(state->plr1.xoff >> 16), (int)(state->plr1.zoff >> 16), state->plr1.zone,
               state->plr1.s_yoff, state->plr1.s_tyoff,
               (int)(state->plr2.xoff >> 16), (int)(state->plr2.zoff >> 16), state->plr2.zone);

        /* Zone dump removed - see player_init log for initial values */
    } else {
        printf("[PLAYER] init_from_level (no level data - players at origin)\n");
    }
}

/* -----------------------------------------------------------------------
 * Player shooting (full implementation)
 *
 * Translated from PlayerShoot.s Player1Shot (line ~23-340).
 *
 * Flow:
 * 1. Decrement fire rate timer
 * 2. Check if player clicked/held fire
 * 3. Check ammo
 * 4. For instant weapons: auto-aim, trace, apply damage
 * 5. For projectile weapons: create bullet in PlayerShotData
 * 6. Play sound, animate gun
 * ----------------------------------------------------------------------- */
static void player_shoot_internal(GameState *state, PlayerState *plr,
                                  int plr_num, const GunDataEntry *guns)
{
    int gun_idx = plr->gun_selected;
    if (gun_idx < 0 || gun_idx >= MAX_GUNS) return;
    if (!player_weapon_is_selectable(plr, gun_idx)) return;

    const GunDataEntry *gun = &guns[gun_idx];

    /* 1. Decrement fire rate timer (exact PlayerShoot.s cadence).
     * Amiga only enters fire path when timer is already zero at function start.
     * If timer is non-zero, it is decremented and this tick exits regardless,
     * even when the decrement reaches/passes zero. */
    if (plr->time_to_shoot != 0) {
        plr->time_to_shoot -= state->temp_frames;
        if (plr->time_to_shoot >= 0) return;
        plr->time_to_shoot = 0;
        return;
    }

    /* 2. Check fire state */
    bool fire = false;
    if (gun->click_or_hold) {
        /* Hold-down weapon: fire while held */
        fire = (plr->p_fire != 0);
    } else {
        /* Click weapon: fire on new click */
        fire = (plr->p_clicked != 0);
        plr->p_clicked = 0;
    }

    if (!fire) return;

    /* 3. Check ammo */
    int16_t ammo = plr->gun_data[gun_idx].ammo;
    if (!state->infinite_ammo) {
        if (ammo < gun->ammo_per_shot) {
            /* Click sound (empty) */
            audio_play_sample(12, 300);
            plr->time_to_shoot = 10; /* prevent spam */
            return;
        }
        plr->gun_data[gun_idx].ammo -= gun->ammo_per_shot;
    }

    /* Set fire rate delay */
    plr->time_to_shoot = gun->fire_delay;

    /* Play gun sound */
    audio_play_sample(gun->gun_sample, 64);

    /* Set gun animation frame (PLR_GunFrame = max frame, counts down) */
    if (gun_idx < 8 && gun_anims[gun_idx].num_frames > 0) {
        plr->gun_frame = (int16_t)gun_anims[gun_idx].num_frames;
    }

    /* 4. Calculate firing direction */
    int16_t sin_val = plr->sinval;
    int16_t cos_val = plr->cosval;

    /* Forward direction matches Amiga: (sin, cos) not negated */
    int32_t dir_x = sin_val;
    int32_t dir_z = cos_val;

    /* 5. Auto-aim system */
    int16_t bulyspd = 0; /* bullet Y velocity for vertical auto-aim */

    int8_t *obs_in_line = (plr_num == 1) ? plr1_obs_in_line : plr2_obs_in_line;
    int16_t *obj_dists = (plr_num == 1) ? plr1_obj_dists : plr2_obj_dists;

    /* Match PlayerShoot.s bitmasks exactly:
     * PLR1: %1111111111110111000001
     * PLR2: %1111111111010111100001 */
    uint32_t enemy_flags = (plr_num == 1) ? 0x003FFDC1u : 0x003FF5E1u;
    uint8_t player_can_see_bit = (plr_num == 1) ? 0x01u : 0x02u;

    /* Find closest target in line of fire (PlayerShoot.s findclosestinline). */
    int closest_idx = -1;
    int32_t closest_dist = 32767;
    int32_t closest_target_ydiff = 0;
    bool has_target = false;

    if (state->level.object_data) {
        for (int i = 0; i < MAX_OBJECTS; i++) {
            GameObject *obj = (GameObject*)(state->level.object_data + i * OBJECT_SIZE);
            int16_t obj_cid = OBJ_CID(obj);
            if (obj_cid < 0) break;

            uint8_t obj_type = (uint8_t)obj->obj.number;
            if (!obs_in_line[i]) continue;
            if ((((uint8_t)obj->obj.can_see) & player_can_see_bit) == 0u) continue;
            if (OBJ_ZONE(obj) < 0) continue;
            if (!(enemy_flags & (1u << (obj_type & 31u)))) continue;
            if (NASTY_LIVES(*obj) == 0) continue;
            if (obj_cid < 0 || obj_cid >= MAX_OBJECTS) continue;

            int32_t dist = obj_dists[obj_cid];
            if (dist <= 0) continue;

            /* PlayerShoot.s vertical gate: abs((objY<<7)-PLR_yoff) / 44 <= forward_dist. */
            int16_t obj_y = obj_w(obj->raw + 4);
            int32_t ydiff = ((int32_t)obj_y << 7) - plr->yoff;
            int32_t abs_ydiff = (ydiff < 0) ? -ydiff : ydiff;
            if ((abs_ydiff / 44) > dist) continue;

            if (dist <= closest_dist) {
                closest_dist = dist;
                closest_idx = i;
                closest_target_ydiff = ydiff;
            }
        }
    }
    has_target = (closest_idx >= 0 && state->level.object_data);
    /* Calculate vertical aim toward target (PlayerShoot.s lines 99-139). */
    {
        if (has_target) {
            int32_t target_ydiff = closest_target_ydiff;
            int32_t aim_dist = closest_dist;

            target_ydiff -= plr->height;
            target_ydiff += 18 * 256;

            int shift = gun->bullet_speed;
            if (shift < 0) shift = 0;
            if (shift > 15) shift = 15;
            int32_t dist_shifted = aim_dist >> shift;
            if (dist_shifted < 1) dist_shifted = 1;

            bulyspd = (int16_t)(target_ydiff / dist_shifted);
        }
    }

    /* 6. Fire the weapon */
    if (gun->fire_bullet != 0) {
        /* Instant-hit weapon (pistol, shotgun) */
        int num_pellets = gun->bullet_count;
        if (num_pellets < 1) num_pellets = 1;
        if (closest_idx < 0 || !state->level.object_data || !state->level.object_points) {
            /* Amiga nothingtoshoot instant branch: one miss effect straight ahead. */
            spawn_instant_miss_effect_no_target(state, plr, gun_idx, sin_val, cos_val);
            return;
        }

        GameObject *target = (GameObject *)(state->level.object_data + closest_idx * OBJECT_SIZE);
        int16_t target_cid = OBJ_CID(target);
        if (target_cid < 0 || target_cid >= state->level.num_object_points) {
            spawn_instant_miss_effect_no_target(state, plr, gun_idx, sin_val, cos_val);
            return;
        }

        {
            const uint8_t *pt = state->level.object_points + (uint32_t)(uint16_t)target_cid * 8u;
            int16_t tgt_ox = obj_w(pt + 0);
            int16_t tgt_oz = obj_w(pt + 4);
            int16_t plr_x = (int16_t)(plr->xoff >> 16);
            int16_t plr_z = (int16_t)(plr->zoff >> 16);
            int16_t adx = (int16_t)(tgt_ox - plr_x);
            int16_t adz = (int16_t)(tgt_oz - plr_z);
            int32_t scaled_dist_sq = (((int32_t)adx * adx) + ((int32_t)adz * adz)) >> 6;

            for (int p = 0; p < num_pellets; p++) {
                int32_t rand_val = (int32_t)(rand() & 0x7FFF);
                bool hit = ((rand_val << 1) > scaled_dist_sq);

                if (hit) {
                    spawn_instant_hit_effect(state, plr, gun_idx, target,
                                             gun->shot_power, (int16_t)dir_x, (int16_t)dir_z);
                } else {
                    spawn_instant_miss_effect_target(state, plr, gun_idx, target);
                }
            }
        }
    } else {
        /* Projectile weapon: Amiga uses PlayerShotData for player projectiles. */
        if (!state->level.player_shot_data) return;
        /* PlayerShoot.s nothingtoshoot path zeros bulyspd before PLR1FIREBULLET.
         * All projectile guns share this; per-gun arc differs via gun data
         * (bullet_y_offset, gravity, flags). */
        if (!has_target) {
            bulyspd = 0;
        }

        /* Find free slot in player_shot_data */
        uint8_t *shots = state->level.player_shot_data;
        GameObject *bullet = NULL;
        for (int i = 0; i < PLAYER_SHOT_SLOT_COUNT; i++) {
            GameObject *candidate = (GameObject*)(shots + i * OBJECT_SIZE);
            if (OBJ_ZONE(candidate) < 0) {
                bullet = candidate;
                break;
            }
        }

        if (!bullet) return; /* No free slot */

        /* Save CID before memset (it's baked into the slot by level data) */
        int16_t saved_cid = OBJ_CID(bullet);

        /* Set up bullet */
        memset(bullet, 0, OBJECT_SIZE);

        /* Amiga spawns projectiles at the player's current world position. */
        int16_t spawn_x = (int16_t)(plr->xoff >> 16);
        int16_t spawn_z = (int16_t)(plr->zoff >> 16);

        /* Restore CID and write spawn position into that object point */
        obj_sw(bullet->raw, saved_cid);
        if (saved_cid >= 0 && state->level.object_points) {
            uint8_t *pt = state->level.object_points + (int)saved_cid * 8;
            obj_sl(pt,     (int32_t)spawn_x << 16);
            obj_sl(pt + 4, (int32_t)spawn_z << 16);
        }

        OBJ_SET_ZONE(bullet, plr->zone);
        bullet->obj.number = OBJ_NBR_BULLET;
        bullet->obj.in_top = plr->stood_in_top;

        /* Initialise sprite from first frame of the bullet animation table.
         * object_handle_bullet will advance SHOT_ANIM each tick. */
        if (gun_idx >= 0 && gun_idx < 8 && bullet_anim_tables[gun_idx]) {
            const BulletAnimFrame *f = &bullet_anim_tables[gun_idx][0];
            if (f->width >= 0) {
                bullet->raw[6] = (uint8_t)f->width;
                bullet->raw[7] = (uint8_t)f->height;
                obj_sw(bullet->raw + 8,  f->vect_num);
                obj_sw(bullet->raw + 10, f->frame_num);
            }
        }
        bullet->raw[14] = bullet_fly_src_cols[gun_idx]; /* src cols (from BulletSizes) */
        bullet->raw[15] = bullet_fly_src_rows[gun_idx]; /* src rows */
        SHOT_ANIM(*bullet) = 0;

        int shift = gun->bullet_speed;
        if (shift < 0) shift = 0;
        if (shift > 15) shift = 15;
        /* Full-scale parity for projectile launch speed. */
        int32_t xvel = ((int32_t)sin_val) << (shift + 1);
        int32_t zvel = ((int32_t)cos_val) << (shift + 1);
        SHOT_SET_XVEL(*bullet, xvel);
        SHOT_SET_ZVEL(*bullet, zvel);
        /* PlayerShoot.s PLR1FIREBULLET clamp order:
         *   if (bulyspd >= 20)  bulyspd = 20;
         *   if (bulyspd >= -20) bulyspd = -20;
         *   bulyspd += bullet_y_offset; */
        int16_t final_yvel = bulyspd;
        if (final_yvel >= 20) final_yvel = 20;
        if (final_yvel >= -20) final_yvel = -20;
        final_yvel = (int16_t)((int32_t)final_yvel + (int32_t)gun->bullet_y_offset);
        SHOT_SET_YVEL(*bullet, final_yvel);
        SHOT_POWER(*bullet) = gun->shot_power;
        SHOT_STATUS(*bullet) = 0;
        SHOT_SET_LIFE(*bullet, 0);
        SHOT_SET_GRAV(*bullet, gun->shot_gravity);
        SHOT_SET_FLAGS(*bullet, gun->shot_flags);
        SHOT_SET_ACCYPOS(*bullet, plr->yoff + 20 * 128);
        NASTY_SET_EFLAGS(*bullet, enemy_flags);
        SHOT_SIZE(*bullet) = (int8_t)gun_idx;
        bullet->obj.worry = 127;
    }
}

void player1_shoot(GameState *state)
{
    player_shoot_internal(state, &state->plr1, 1, default_plr1_guns);
}

void player2_shoot(GameState *state)
{
    player_shoot_internal(state, &state->plr2, 2, default_plr2_guns);
}
