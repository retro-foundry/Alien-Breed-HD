/*
 * Alien Breed 3D I - PC Port
 * objects.c - Object system (full implementation)
 *
 * Translated from: Anims.s, ObjectMove.s, AI.s, NormalAlien.s,
 *                  Robot.s, BigRedThing.s, HalfWorm.s, FlameMarine.s,
 *                  ToughMarine.s, MutantMarine.s, BigUglyAlien.s,
 *                  BigClaws.s, FlyingScalyBall.s, Tree.s
 *
 * The main entry point is objects_update(), called once per frame.
 */

#include "objects.h"
#include "ai.h"
#include "game_data.h"
#include "game_types.h"
#include "level.h"
#include "movement.h"
#include "math_tables.h"
#include "audio.h"
#include "visibility.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define LIFT_ENTRY_SIZE 20

/* Big claws (small red): minimum distance before they stop rushing in (CalcDist-style units). */
#define SMALL_RED_STANDOFF_DIST 128
/* Hard minimum spacing from players for red enemy families. */
#define RED_ALIEN_MIN_PLAYER_DIST      112
#define HUGE_RED_MIN_PLAYER_DIST       224
#define SMALL_RED_THING_MIN_PLAYER_DIST 160

/* Big-endian read/write helpers (Amiga data is big-endian) */
static inline int16_t be16(const uint8_t *p) {
    return (int16_t)((p[0] << 8) | p[1]);
}
static inline int32_t be32(const uint8_t *p) {
    return (int32_t)(((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                     ((uint32_t)p[2] << 8) | (uint32_t)p[3]);
}
static inline void write_be16(uint8_t *p, int16_t w) {
    p[0] = (uint8_t)((unsigned)w >> 8);
    p[1] = (uint8_t)(unsigned)w;
}
static inline void wbe16(uint8_t *p, int16_t v) {
    p[0] = (uint8_t)((uint16_t)v >> 8); p[1] = (uint8_t)v;
}
static inline void wbe32(uint8_t *p, int32_t v) {
    p[0] = (uint8_t)((uint32_t)v >> 24); p[1] = (uint8_t)((uint32_t)v >> 16);
    p[2] = (uint8_t)((uint32_t)v >> 8);  p[3] = (uint8_t)v;
}

/* -----------------------------------------------------------------------
 * Visibility arrays (from AB3DI.s CalcPLR1InLine, CalcPLR2InLine)
 * ----------------------------------------------------------------------- */
int8_t  plr1_obs_in_line[MAX_OBJECTS];
int8_t  plr2_obs_in_line[MAX_OBJECTS];
int16_t plr1_obj_dists[MAX_OBJECTS];
int16_t plr2_obj_dists[MAX_OBJECTS];
static uint8_t explosion_damage_flag[MAX_OBJECTS];
static inline uint8_t damage_accumulate_u8(uint8_t current, int32_t add);

/* Animation timer (from Anims.s) */
static int16_t anim_timer = 2;

/* Walk-cycle counter (Amiga: alan[] table / alanptr, advances each game tick).
 * Range 0-31 (wraps at 32). Each frame step spans 8 ticks → step = counter >> 3, range 0-3.
 * Amiga: alan[] has 8 copies of each value 0-3 → 32 entries, cycles at game tick rate. */
static uint8_t walk_cycle = 0;

/* Gib floor/wall splat (sample 13): only one per objects_update — many gibs can time out together. */
static bool gib_impact_splat_sound_this_update;

enum {
    ENEMY_OBJ_TIMER_OFF    = 16, /* ObjTimer   (raw+34) */
    ENEMY_SEC_TIMER_OFF    = 18, /* SecTimer   (raw+36) */
    ENEMY_OBJ_YVEL_OFF     = 30, /* objyvel    (raw+48) */
    ENEMY_TURN_SPEED_OFF   = 32, /* TurnSpeed  (raw+50) */
    ENEMY_THIRD_TIMER_OFF  = 34, /* ThirdTimer (raw+52) */
    ENEMY_FOURTH_TIMER_OFF = 36  /* FourthTimer(raw+54) */
};

/* Explosion visual pacing in logic-vblank units.
 * 2 means one animation frame every 2 logic-vblank units. */
#define EXPLOSION_FRAME_STEP_VBLANKS 2

static int32_t enemy_move_y_for_context(const GameObject *obj,
                                        const EnemyParams *params,
                                        const GameState *state,
                                        int zone_slots);
static void enemy_update_flying_vertical(GameObject *obj, const GameState *state);

/* -----------------------------------------------------------------------
 * enemy_viewpoint - Amiga AlienControl.s ViewpointToDraw
 * Returns 0=towards, 1=right, 2=away, 3=left based on angle between
 * enemy facing direction and enemy-to-player direction vector.
 *
 * Exact translation of Amiga assembly:
 *   d0 = dx*sin + dz*cos  (forward: > 0 means player is ahead)
 *   d4 = dx*cos - dz*sin  (lateral: > 0 means player is to enemy's right)
 * ----------------------------------------------------------------------- */
static int enemy_viewpoint(GameObject *obj, int16_t plr_x, int16_t plr_z,
                            const LevelState *level)
{
    int cid = (int)OBJ_CID(obj);
    int16_t ox = 0, oz = 0;
    if (level->object_points && cid >= 0 && cid < level->num_object_points) {
        const uint8_t *p = level->object_points + cid * 8;
        ox = be16(p);
        oz = be16(p + 4);
    }

    /* dx, dz = direction from enemy toward player (as in HeadTowards) */
    int32_t dx = (int32_t)plr_x - ox;
    int32_t dz = (int32_t)plr_z - oz;

    int16_t facing = NASTY_FACING(*obj);
    int32_t sf = sin_lookup(facing);  /* Amiga: d2 = SineTable[facing] */
    int32_t cf = cos_lookup(facing);  /* Amiga: d3 = SineTable[facing+1024] */

    /* Amiga: d0 = dx*sin + dz*cos (forward), d4 = dx*cos - dz*sin (lateral) */
    int32_t fwd  = dx * sf + dz * cf;   /* positive = player ahead of enemy */
    int32_t lat  = dx * cf - dz * sf;   /* positive = player to enemy's right */

    /* Amiga quadrant logic (tst/cmp/bgt chain):
     *   FacingTowardsPlayer (fwd > 0):
     *     lat > 0 && lat > fwd  → RIGHT(1)
     *     lat > 0 && lat <= fwd → TOWARDS(0)
     *     lat <= 0 && -lat > fwd → LEFT(3)
     *     lat <= 0 && -lat <= fwd → TOWARDS(0)
     *   FacingAway (fwd <= 0):
     *     lat > 0 && lat > -fwd → RIGHT(1)
     *     lat > 0 && lat <= -fwd → AWAY(2)
     *     lat <= 0 && -lat > -fwd → LEFT(3)
     *     lat <= 0 && -lat <= -fwd → AWAY(2)
     */
    int q;
    if (fwd > 0) {
        if (lat > 0)
            q = (lat > fwd)  ? 1 : 0;  /* RIGHT or TOWARDS */
        else
            q = (-lat > fwd) ? 3 : 0;  /* LEFT  or TOWARDS */
    } else {
        int32_t afwd = -fwd;
        if (lat > 0)
            q = (lat > afwd)  ? 1 : 2; /* RIGHT or AWAY */
        else
            q = (-lat > afwd) ? 3 : 2; /* LEFT  or AWAY */
    }

    /* Sprite directional groups are rotated 180 degrees relative to the
     * Amiga quadrant constants in this port's frame tables. */
    return (q + 2) & 3;
}

/* Update only the facing quadrant in the frame index and preserve the
 * low 2-bit walk/attack sub-frame (Amiga layout: frame = angle*4 + step). */
static void enemy_update_display_facing_frame(GameObject *obj, const GameState *state,
                                              int16_t view_x, int16_t view_z)
{
    int16_t old_frame = OBJ_DEADL(obj);
    /* Flying variants use special attack/hit frames (16+) that are not angle*4+step.
     * Preserve those until logic code assigns a new frame. */
    if ((obj->obj.number == OBJ_NBR_FLYING_NASTY || obj->obj.number == OBJ_NBR_EYEBALL) &&
        old_frame >= 16) {
        return;
    }
    int16_t step = (int16_t)(old_frame & 3);
    int16_t angle = (int16_t)(enemy_viewpoint(obj, view_x, view_z, &state->level) & 3);
    OBJ_SET_DEADL(obj, (int16_t)((angle << 2) | step));
}

/* -----------------------------------------------------------------------
 * enemy_update_anim - set raw[8..11] (OBJ_DEADH|OBJ_DEADL) each tick.
 * Translates Amiga: asl.l #2,d0; add.l alframe,d0 [; add.l #vectnum_high,d0]; move.l d0,8(a0)
 * vect_num is the sprite WAD index for this enemy type (high word of 8(a0)).
 * frame = angle*4 + walk_step (0-15, low word of 8(a0)).
 * ----------------------------------------------------------------------- */
static void enemy_update_anim_with_step(GameObject *obj, GameState *state,
                                        int16_t vect_num, int walk_step);
static int marine_pick_target_player(GameObject *obj, GameState *state);
static int32_t marine_track_target(GameObject *obj, const EnemyParams *params,
                                   GameState *state, int player_num,
                                   bool apply_translation);
static bool enemy_is_facing_player_cone(GameObject *obj, GameState *state,
                                        int player_num, int16_t min_cos_q14);

static void enemy_update_anim(GameObject *obj, GameState *state, int16_t vect_num)
{
    enemy_update_anim_with_step(obj, state, vect_num, (walk_cycle >> 3) & 3);
}

/* Marine attack stance uses the same facing frame logic but with a fixed
 * walk sub-frame (alframe disabled) while firing. */
static void enemy_update_anim_with_step(GameObject *obj, GameState *state,
                                        int16_t vect_num, int walk_step)
{
    if (walk_step < 0) walk_step = 0;
    if (walk_step > 3) walk_step = 3;

    /* Which player to use for viewpoint (use player 1, or 2 if 1 has invalid zone) */
    int16_t plr_x, plr_z;
    if (state->plr1.zone >= 0) {
        plr_x = (int16_t)state->plr1.p_xoff;
        plr_z = (int16_t)state->plr1.p_zoff;
    } else {
        plr_x = (int16_t)state->plr2.p_xoff;
        plr_z = (int16_t)state->plr2.p_zoff;
    }

    int angle = enemy_viewpoint(obj, plr_x, plr_z, &state->level);
    int16_t frame = (int16_t)(angle * 4 + walk_step);

    /* Store as big-endian long at raw[8..11]:
     * raw[8..9] = vect_num (high word), raw[10..11] = frame (low word) */
    wbe16(obj->raw + 8,  vect_num);
    wbe16(obj->raw + 10, frame);
}

/* -----------------------------------------------------------------------
 * Set world width/height in each object record from type defaults when unset (Amiga: each type sets 6(a0)).
 * ----------------------------------------------------------------------- */
void object_init_world_sizes_from_types(LevelState *level)
{
    if (!level->object_data) return;
    /* Object table is CID-terminated; do not use num_object_points (different pool size). */
    for (int i = 0; i < 256; i++) {
        uint8_t *raw = level->object_data + i * OBJECT_SIZE;
        int16_t cid = be16(raw);
        if (cid < 0) break;
        int t = (int8_t)raw[16];
        if (t < 0 || t > 20) continue;
        if (raw[6] == 0 || raw[7] == 0) {
            raw[6] = (uint8_t)default_object_world_size[t].w;
            raw[7] = (uint8_t)default_object_world_size[t].h;
        }
    }
}

/* -----------------------------------------------------------------------
 * Object iteration helpers
 * ----------------------------------------------------------------------- */
static GameObject *get_object(LevelState *level, int index)
{
    if (!level->object_data) return NULL;
    return (GameObject *)(level->object_data + index * OBJECT_SIZE);
}

/* Get object X/Z position from ObjectPoints array (big-endian) */
static void get_object_pos(const LevelState *level, int index,
                           int16_t *x, int16_t *z)
{
    if (level->object_points) {
        const uint8_t *p = level->object_points + index * 8;
        *x = obj_w(p);
        *z = obj_w(p + 4);
    } else {
        *x = 0;
        *z = 0;
    }
}

static GameObject *find_free_shot_slot(uint8_t *shots, int16_t *saved_cid)
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

static int count_used_shot_slots(const uint8_t *shots, int shot_slots)
{
    int used = 0;
    if (!shots || shot_slots <= 0) return 0;
    for (int i = 0; i < shot_slots; i++) {
        const GameObject *candidate = (const GameObject *)(shots + (size_t)i * OBJECT_SIZE);
        if (OBJ_ZONE(candidate) >= 0) used++;
    }
    return used;
}

/* Amiga Noisevol (sfx importance / channel mix) → PC mixer 0–255 (see SoundPlayer.s). */
static inline int amiga_noisevol_to_pc(int noisevol)
{
    if (noisevol < 1) noisevol = 1;
    int v = (noisevol * 255 + 400) / 800;
    if (v > 255) v = 255;
    return v;
}

static inline bool enemy_is_human_marine(int8_t obj_type)
{
    return obj_type == OBJ_NBR_FLAME_MARINE
        || obj_type == OBJ_NBR_TOUGH_MARINE
        || obj_type == OBJ_NBR_MARINE;
}

/* Amiga ExplodeIntoBits call sites usually pass d2 derived from the
 * killing blow; marine/alien/worm classes shift it down before spawning bits. */
static int16_t enemy_gib_level_from_damage(const GameObject *obj,
                                           const EnemyParams *params,
                                           int32_t damage)
{
    int32_t d2 = damage;
    if (params &&
        (params->damage_audio_class == ENEMY_DMG_AUDIO_ALIEN ||
         params->damage_audio_class == ENEMY_DMG_AUDIO_WORM ||
         enemy_is_human_marine(obj->obj.number))) {
        d2 >>= 2;
    }
    if (d2 < 1) d2 = 1;
    return (int16_t)d2;
}

/* Queue player damage through damagetaken (raw[19]) so USEPLR1/2 apply
 * hit flash + pain audio exactly like the Amiga path. */
static void player_add_damage(GameState *state, int player_num, int damage)
{
    if (damage <= 0) return;
    if (player_num != 1 && player_num != 2) return;

    GameObject *plr_obj = (player_num == 1)
        ? (GameObject *)state->level.plr1_obj
        : (GameObject *)state->level.plr2_obj;

    if (!plr_obj) {
        PlayerState *plr = (player_num == 1) ? &state->plr1 : &state->plr2;
        plr->energy -= (int16_t)damage;
        return;
    }

    int cur = NASTY_DAMAGE(*plr_obj);
    if (cur < 0) cur = 0;
    int next = cur + damage;
    if (next > 127) next = 127;
    NASTY_SET_DAMAGE(plr_obj, next);
}

static void enemy_apply_death_outcome(GameObject *obj, const EnemyParams *params, bool instant_kill)
{
    if (!instant_kill && params->death_frames[0] >= 0) {
        int8_t original_type = obj->obj.number;
        obj->obj.type_data[0] = 0;
        obj->obj.type_data[1] = original_type;
        OBJ_SET_DEADH(obj, params->death_frames[0]);
        OBJ_SET_DEADL(obj, 0);
        obj->obj.number = OBJ_NBR_DEAD;
    } else {
        OBJ_SET_ZONE(obj, -1);
    }
}

/* Floating enemy soft death (FlyingScalyBall/EyeBall):
 * keep the corpse in-world and settle it toward floor height, advancing
 * frame 18->19 while falling, then resting at frame 20. */
static void enemy_update_flying_soft_dead(GameObject *obj,
                                          const EnemyParams *params,
                                          GameState *state)
{
    if (!obj || !params || !state) return;
    if (OBJ_ZONE(obj) < 0) return;

    NASTY_LIVES(*obj) = 0;
    NASTY_DAMAGE(*obj) = 0;

    int zone_slots = level_zone_slot_count(&state->level);
    int zone_idx = level_connect_to_zone_index(&state->level, OBJ_ZONE(obj));
    if (zone_idx < 0 && OBJ_ZONE(obj) >= 0 && OBJ_ZONE(obj) < zone_slots)
        zone_idx = OBJ_ZONE(obj);
    if (zone_idx < 0 || zone_idx >= zone_slots ||
        !state->level.zone_adds || !state->level.data) {
        return;
    }

    int32_t zoff = (int32_t)be32(state->level.zone_adds + (uint32_t)zone_idx * 4u);
    const uint8_t *zd = state->level.data + zoff;
    int32_t floor_h = be32(zd + (obj->obj.in_top ? ZONE_OFF_UPPER_FLOOR : ZONE_OFF_FLOOR));
    int16_t floor_y = (int16_t)((floor_h >> 7) - params->nas_height);

    int16_t y = obj_w(obj->raw + 4);
    if (y < floor_y) {
        int16_t dy = (int16_t)(state->temp_frames << 4);
        y = (int16_t)(y + dy);
        if (y > floor_y) y = floor_y;
        obj_sw(obj->raw + 4, y);

        int16_t fourth = OBJ_TD_W(obj, ENEMY_FOURTH_TIMER_OFF);
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, (int16_t)(fourth + dy));

        int16_t third = OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF);
        third = (int16_t)(third - state->temp_frames);
        if (third < 0) {
            third = 20;
            int16_t frame = OBJ_DEADL(obj);
            if (frame < 19) OBJ_SET_DEADL(obj, (int16_t)(frame + 1));
        }
        OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, third);
    } else {
        obj_sw(obj->raw + 4, floor_y);
        /* Amiga FlyingScalyBall soft death: on first floor contact, burst into bits,
         * then switch to final floor-corpse frame (20). */
        if (OBJ_DEADL(obj) < 20) {
            int splatv = (params->gib_splat_noisevol > 0) ? params->gib_splat_noisevol : 400;
            int16_t gib_level = (int16_t)((OBJ_TD_W(obj, ENEMY_FOURTH_TIMER_OFF) >> 4) + 1);
            audio_play_sample(14, amiga_noisevol_to_pc(splatv));
            explode_into_bits(obj, state, false, gib_level);
            OBJ_SET_DEADL(obj, 20);
        }
    }
}

/* NormalAlien.s / FlameMarine.s: splat #14@400 if damage>1, scream if death anim, instant if huge hit. */
static void enemy_death_marine_like(GameObject *obj, const EnemyParams *params,
                                    GameState *state, int32_t damage,
                                    bool instant_kill, int scream_sample,
                                    bool explosion_damage)
{
    if (damage > 1 || explosion_damage) {
        int16_t gib_level = enemy_gib_level_from_damage(obj, params, damage);
        audio_play_sample(14, amiga_noisevol_to_pc(400));
        explode_into_bits(obj, state, explosion_damage, gib_level);
    }
    if ((!instant_kill || explosion_damage) && params->death_frames[0] >= 0) {
        audio_play_sample(scream_sample, amiga_noisevol_to_pc(200));
    }
    enemy_apply_death_outcome(obj, params, instant_kill);
}

/* -----------------------------------------------------------------------
 * Enemy common: check damage and handle death
 *
 * Returns: true if enemy is dead (caller should return early)
 * ----------------------------------------------------------------------- */
static bool enemy_check_damage(GameObject *obj, const EnemyParams *params, GameState *state)
{
    uint8_t raw_damage = obj->raw[19];
    if (raw_damage == 0) return false;

    obj->raw[19] = 0;
    int32_t damage = (int32_t)raw_damage;

    int obj_idx = -1;
    if (state->level.object_data) {
        obj_idx = (int)(((uint8_t*)obj - state->level.object_data) / OBJECT_SIZE);
    }
    bool explosion_damage = false;
    if (obj_idx >= 0 && obj_idx < MAX_OBJECTS) {
        explosion_damage = explosion_damage_flag[obj_idx];
        explosion_damage_flag[obj_idx] = 0;
    }

    /* Apply damage reduction */
    if (params->damage_shift > 0) {
        damage >>= params->damage_shift;
        if (damage < 1) {
            if (params->min_damage_after_shift) {
                damage = 1;
            } else {
                return false; /* Armor absorbed this hit (Amiga beq noscream path). */
            }
        }
    }

    int32_t lives = (int32_t)NASTY_LIVES(*obj);
    lives -= damage;

    if (lives <= 0) {
        /* Instant-kill parity:
         * - Flying/Eyeball compare raw blow directly (> 40).
         * - NormalAlien + human marines + HalfWorm compare AFTER an extra asr #2
         *   in their kill branches, so the effective threshold is much higher. */
        bool instant_kill = false;
        if (params->explode_threshold > 0) {
            int32_t instant_cmp = damage;

            if (params->damage_audio_class == ENEMY_DMG_AUDIO_ALIEN ||
                params->damage_audio_class == ENEMY_DMG_AUDIO_WORM ||
                enemy_is_human_marine(obj->obj.number)) {
                instant_cmp >>= 2;
            }

            if (params->damage_audio_class == ENEMY_DMG_AUDIO_FLYING) {
                instant_kill = (instant_cmp > params->explode_threshold);
            } else {
                instant_kill = (instant_cmp >= params->explode_threshold);
            }
        }

        /* Human marines: FlameMarine.s / ToughMarine.s / MutantMarine.s */
        if (enemy_is_human_marine(obj->obj.number)) {
            enemy_death_marine_like(obj, params, state, damage, instant_kill, 0, explosion_damage);
            return true;
        }

        switch (params->damage_audio_class) {
        case ENEMY_DMG_AUDIO_ALIEN:
            enemy_death_marine_like(obj, params, state, damage, instant_kill, 0, explosion_damage);
            return true;

        case ENEMY_DMG_AUDIO_ROBOT:
            /* Robot.s: boom #15 @400, ComputeBlast — no splat #14, no ExplodeIntoBits */
            audio_play_sample(15, amiga_noisevol_to_pc(400));
            enemy_apply_death_outcome(obj, params, instant_kill);
            return true;

        case ENEMY_DMG_AUDIO_BIG_GIB:
            /* BigRedThing.s / BigClaws.s / Tree.s: #14@400 + gibs, no screamsound on kill */
            audio_play_sample(14, amiga_noisevol_to_pc(400));
            explode_into_bits(obj, state, explosion_damage, 7);
            enemy_apply_death_outcome(obj, params, instant_kill);
            return true;

        case ENEMY_DMG_AUDIO_WORM: {
            /* HalfWorm.s: splat #14 @300 if damage>1; scream #27@200 for anim; instant if damage>=80 */
            int splatv = (params->gib_splat_noisevol > 0) ? params->gib_splat_noisevol : 400;
            if (damage > 1 || explosion_damage) {
                int16_t gib_level = enemy_gib_level_from_damage(obj, params, damage);
                audio_play_sample(14, amiga_noisevol_to_pc(splatv));
                explode_into_bits(obj, state, explosion_damage, gib_level);
            }
            if ((!instant_kill || explosion_damage) && params->death_frames[0] >= 0) {
                audio_play_sample(params->scream_sound, amiga_noisevol_to_pc(200));
            }
            enemy_apply_death_outcome(obj, params, instant_kill);
            return true;
        }

        case ENEMY_DMG_AUDIO_FLYING:
            /* FlyingScalyBall.s / EyeBall.s: killing blow >40 → #14@400+gib; else #8@200 */
            if (instant_kill) {
                audio_play_sample(14, amiga_noisevol_to_pc(400));
                explode_into_bits(obj, state, explosion_damage, 9);
                if (explosion_damage && params->scream_sound >= 0)
                    audio_play_sample(params->scream_sound, amiga_noisevol_to_pc(200));
                OBJ_SET_ZONE(obj, -1);
            } else {
                if (params->scream_sound >= 0)
                    audio_play_sample(params->scream_sound, amiga_noisevol_to_pc(200));
                if (obj->obj.number == OBJ_NBR_EYEBALL ||
                    obj->obj.number == OBJ_NBR_FLYING_NASTY) {
                    /* Flying/Eyeball soft kill: keep corpse and hand over to dead-state update. */
                    NASTY_LIVES(*obj) = 0;
                    OBJ_SET_DEADL(obj, 18);
                    OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 30);
                    OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, 0);
                } else {
                    enemy_apply_death_outcome(obj, params, false);
                }
            }
            return true;

        case ENEMY_DMG_AUDIO_BIGUGLY:
            /* BigUglyAlien.s: lowscream #8 @200 only, no gibs on kill */
            if (params->scream_sound >= 0)
                audio_play_sample(params->scream_sound, amiga_noisevol_to_pc(200));
            enemy_apply_death_outcome(obj, params, instant_kill);
            return true;

        case ENEMY_DMG_AUDIO_GENERIC:
        default:
            if (params->death_sound >= 0) {
                audio_play_sample(params->death_sound, amiga_noisevol_to_pc(200));
            }
            if (damage > 1 || explosion_damage) {
                int16_t gib_level = enemy_gib_level_from_damage(obj, params, damage);
                explode_into_bits(obj, state, explosion_damage, gib_level);
            }
            enemy_apply_death_outcome(obj, params, instant_kill);
            return true;
        }
    }

    NASTY_LIVES(*obj) = (int8_t)lives;

    /* Hurt: screamsound @ Noisevol 200 (all nasties) */
    if (params->scream_sound >= 0) {
        audio_play_sample(params->scream_sound, amiga_noisevol_to_pc(200));
    }

    return false;
}

static int16_t enemy_min_player_separation_for_type(int8_t obj_type)
{
    switch (obj_type) {
    case OBJ_NBR_ALIEN:
        return RED_ALIEN_MIN_PLAYER_DIST;
    case OBJ_NBR_HUGE_RED_THING:
        return HUGE_RED_MIN_PLAYER_DIST;
    case OBJ_NBR_SMALL_RED_THING:
        return SMALL_RED_THING_MIN_PLAYER_DIST;
    default:
        return 0;
    }
}

static void enemy_enforce_min_distance_from_player(const GameObject *obj,
                                                   const PlayerState *plr,
                                                   int16_t min_dist,
                                                   int16_t *x, int16_t *z)
{
    if (!obj || !plr || !x || !z || min_dist <= 0) return;

    int32_t dx = (int32_t)(*x) - (int32_t)plr->p_xoff;
    int32_t dz = (int32_t)(*z) - (int32_t)plr->p_zoff;
    int32_t dist = calc_dist_approx(dx, dz);
    if (dist >= min_dist) return;

    if (dist <= 0) {
        /* Degenerate overlap: push out along current facing so we still enforce spacing. */
        int32_t sx = (int32_t)sin_lookup(NASTY_FACING(*obj));
        int32_t sz = (int32_t)cos_lookup(NASTY_FACING(*obj));
        dx = (sx == 0) ? 1 : sx;
        dz = (sz == 0) ? 0 : sz;
        dist = calc_dist_approx(dx, dz);
        if (dist <= 0) {
            dx = 1;
            dz = 0;
            dist = 1;
        }
    }

    int32_t nx = (int32_t)plr->p_xoff + (dx * (int32_t)min_dist) / dist;
    int32_t nz = (int32_t)plr->p_zoff + (dz * (int32_t)min_dist) / dist;
    if (nx < -32768) nx = -32768;
    if (nx >  32767) nx =  32767;
    if (nz < -32768) nz = -32768;
    if (nz >  32767) nz =  32767;
    *x = (int16_t)nx;
    *z = (int16_t)nz;
}

static void enemy_enforce_player_separation(const GameObject *obj, const GameState *state,
                                            int16_t *x, int16_t *z)
{
    if (!obj || !state || !x || !z) return;
    int16_t min_dist = enemy_min_player_separation_for_type(obj->obj.number);
    if (min_dist <= 0) return;

    enemy_enforce_min_distance_from_player(obj, &state->plr1, min_dist, x, z);
    if (state->mode != MODE_SINGLE) {
        enemy_enforce_min_distance_from_player(obj, &state->plr2, min_dist, x, z);
    }
}

/* -----------------------------------------------------------------------
 * Enemy common: wander behavior using Amiga-style ObjTimer ranges.
 * ----------------------------------------------------------------------- */
static void enemy_wander_with_timer(GameObject *obj, const EnemyParams *params,
                                    GameState *state, int16_t timer_base,
                                    int16_t timer_mask)
{
    bool flying_hover = (obj->obj.number == OBJ_NBR_FLYING_NASTY ||
                         obj->obj.number == OBJ_NBR_EYEBALL);
    int16_t timer = NASTY_TIMER(*obj);
    timer -= state->temp_frames;

    if (timer <= 0) {
        timer = timer_base;
        if (timer_mask > 0) {
            timer = (int16_t)(timer + (rand() & timer_mask));
        }

        if (flying_hover) {
            /* Amiga EyeBall/FlyingScalyBall: ObjTimer rollover updates
             * TurnSpeed + objyvel (not an immediate random Facing snap). */
            int16_t turn_speed = (int16_t)(((rand() >> 4) & 255) - 128);
            turn_speed = (int16_t)(turn_speed * 2);
            OBJ_SET_TD_W(obj, ENEMY_TURN_SPEED_OFF, turn_speed);

            int16_t yv = (int16_t)(((rand() >> 4) & 7) - 3);
            yv = (int16_t)(yv - ((rand() >> 5) & 1));
            OBJ_SET_TD_W(obj, ENEMY_OBJ_YVEL_OFF, yv);
        } else {
            int16_t new_facing = (int16_t)(rand() & 8190);
            NASTY_SET_FACING(*obj, new_facing);
        }
    }

    NASTY_SET_TIMER(*obj, timer);

    int16_t facing = NASTY_FACING(*obj);
    int16_t speed = NASTY_MAXSPD(*obj);
    if (speed == 0) speed = 4;

    int16_t s = sin_lookup(facing);
    int16_t c = cos_lookup(facing);

    int16_t obj_x, obj_z;
    int cid = (int)OBJ_CID(obj);
    if (cid < 0) return;
    get_object_pos(&state->level, cid, &obj_x, &obj_z);

    /* Amiga enemy scripts use type-specific object-collision masks and
     * some types skip pre-move Collision entirely. */
    uint32_t collide_mask = 0x3FDE1;
    bool use_pre_collision = true;
    switch (obj->obj.number) {
    case OBJ_NBR_ALIEN:
        collide_mask = 0x3FDE1;
        break;
    case OBJ_NBR_WORM:
        collide_mask = 0x7FDE1;
        break;
    case OBJ_NBR_TREE:
        collide_mask = 0xDFDE1;
        break;
    case OBJ_NBR_HUGE_RED_THING:
        collide_mask = 0x0DE1;
        break;
    case OBJ_NBR_SMALL_RED_THING:
    case OBJ_NBR_FLAME_MARINE:
    case OBJ_NBR_TOUGH_MARINE:
    case OBJ_NBR_MARINE:
    case OBJ_NBR_FLYING_NASTY:
    case OBJ_NBR_EYEBALL:
        collide_mask = 0xFFDE1;
        break;
    case OBJ_NBR_ROBOT:
    case OBJ_NBR_BIG_NASTY:
        use_pre_collision = false;
        break;
    default:
        break;
    }

    MoveContext ctx;
    move_context_init(&ctx);
    ctx.oldx = obj_x;
    ctx.oldz = obj_z;
    ctx.newx = obj_x - ((int32_t)s * speed * state->temp_frames) / 16384;
    ctx.newz = obj_z - ((int32_t)c * speed * state->temp_frames) / 16384;
    ctx.thing_height = params->thing_height;
    int zone_slots = level_zone_slot_count(&state->level);
    {
        int32_t move_y = enemy_move_y_for_context(obj, params, state, zone_slots);
        ctx.oldy = move_y;
        ctx.newy = move_y;
    }
    ctx.step_up_val = params->step_up;
    ctx.step_down_val = params->step_down;
    ctx.extlen = params->extlen;
    ctx.awayfromwall = params->awayfromwall;
    ctx.collide_flags = collide_mask;
    ctx.coll_id = OBJ_CID(obj);
    ctx.pos_shift = 0;
    ctx.stood_in_top = obj->obj.in_top;
    if (OBJ_ZONE(obj) >= 0 && state->level.zone_adds && state->level.data) {
        int src_zone = level_connect_to_zone_index(&state->level, OBJ_ZONE(obj));
        if (src_zone < 0 && OBJ_ZONE(obj) < zone_slots)
            src_zone = OBJ_ZONE(obj);
        if (src_zone >= 0 && src_zone < zone_slots) {
            int32_t zo = (int32_t)be32(state->level.zone_adds + (uint32_t)src_zone * 4u);
            ctx.objroom = (uint8_t *)(state->level.data + zo);
        }
    }

    /* Amiga enemy flow: many types do object Collision first; some (Robot/BigUgly) don't. */
    if (use_pre_collision) {
        collision_check(&ctx, &state->level);
    }
    if (ctx.hitwall) {
        ctx.newx = ctx.oldx;
        ctx.newz = ctx.oldz;
    } else {
        /* Amiga enemy movement uses a single MoveObject pass per tick. */
        move_object(&ctx, &state->level);
    }

    /* Gameplay guardrail: red enemies must keep minimum spacing from players. */
    {
        int16_t sep_x = (int16_t)ctx.newx;
        int16_t sep_z = (int16_t)ctx.newz;
        enemy_enforce_player_separation(obj, state, &sep_x, &sep_z);
        ctx.newx = sep_x;
        ctx.newz = sep_z;
    }

    if (state->level.object_points) {
        uint8_t *pts = state->level.object_points + cid * 8;
        obj_sw(pts, (int16_t)ctx.newx);
        obj_sw(pts + 4, (int16_t)ctx.newz);
    }
    if (ctx.objroom && state->level.data) {
        int new_zone = level_zone_index_from_room_ptr(&state->level, ctx.objroom);
        if (new_zone < 0) {
            int16_t room_zone_word = (int16_t)((ctx.objroom[0] << 8) | ctx.objroom[1]);
            new_zone = level_connect_to_zone_index(&state->level, room_zone_word);
        }
        if (new_zone >= 0 && new_zone < zone_slots)
            OBJ_SET_ZONE(obj, (int16_t)new_zone);
        obj->obj.in_top = ctx.stood_in_top;
    }

    if (ctx.hitwall) {
        NASTY_SET_TIMER(*obj, -1);
    }
}

/* -----------------------------------------------------------------------
 * Map ObjNumber -> enemy_params index
 * ----------------------------------------------------------------------- */
static int obj_type_to_enemy_index(int8_t obj_type)
{
    switch (obj_type) {
    case OBJ_NBR_ALIEN:          return 0;
    case OBJ_NBR_ROBOT:          return 1;
    case OBJ_NBR_HUGE_RED_THING: return 2;
    case OBJ_NBR_WORM:           return 3;
    case OBJ_NBR_FLAME_MARINE:   return 4;
    case OBJ_NBR_TOUGH_MARINE:   return 5;
    case OBJ_NBR_MARINE:         return 6;  /* Mutant Marine */
    case OBJ_NBR_BIG_NASTY:      return 7;
    case OBJ_NBR_SMALL_RED_THING:return 8;  /* BigClaws uses same slot */
    case OBJ_NBR_FLYING_NASTY:   return 9;
    case OBJ_NBR_EYEBALL:        return 9;  /* same params as FlyingNasty */
    case OBJ_NBR_TREE:           return 10;
    default:                     return -1;
    }
}

/* -----------------------------------------------------------------------
 * Update obj->obj.can_see via CanItBeSeen for both players.
 * Bit 0 = can see player 1, bit 1 = can see player 2.
 * Called by every enemy handler each tick (Amiga: each NormalAlien / MutantMarine
 * calls CanItBeSeen itself in its prowl/attack routine).
 * ----------------------------------------------------------------------- */
static void enemy_update_can_see(GameObject *obj, GameState *state)
{
    int16_t enemy_zone = OBJ_ZONE(obj);
    if (enemy_zone < 0) return;
    const uint8_t *from_room = level_get_zone_data_ptr(&state->level, enemy_zone);
    if (!from_room) return;

    int enemy_cid = (int)OBJ_CID(obj);
    int16_t enemy_x = 0, enemy_z = 0;
    get_object_pos(&state->level, enemy_cid, &enemy_x, &enemy_z);
    int16_t viewer_y = (int16_t)obj_w(obj->raw + 4);

    obj->obj.can_see = 0;

    /* Player 1 -> bit 0 */
    {
        int16_t plr_zone = state->plr1.zone;
        const uint8_t *to_room = level_get_zone_data_ptr(&state->level, plr_zone);
        if (to_room) {
            int16_t plr_x    = (int16_t)state->plr1.p_xoff;
            int16_t plr_z    = (int16_t)state->plr1.p_zoff;
            int16_t target_y = (int16_t)(state->plr1.p_yoff >> 7);
            uint8_t vis = can_it_be_seen(&state->level,
                                         from_room, to_room, plr_zone,
                                         enemy_x, enemy_z, viewer_y,
                                         plr_x, plr_z, target_y,
                                         obj->obj.in_top,
                                         state->plr1.stood_in_top, 0);
            if (vis) obj->obj.can_see |= 0x01;
        }
    }
    /* Player 2 -> bit 1 */
    {
        int16_t plr_zone = state->plr2.zone;
        const uint8_t *to_room = level_get_zone_data_ptr(&state->level, plr_zone);
        if (to_room) {
            int16_t plr_x    = (int16_t)state->plr2.p_xoff;
            int16_t plr_z    = (int16_t)state->plr2.p_zoff;
            int16_t target_y = (int16_t)(state->plr2.p_yoff >> 7);
            uint8_t vis = can_it_be_seen(&state->level,
                                         from_room, to_room, plr_zone,
                                         enemy_x, enemy_z, viewer_y,
                                         plr_x, plr_z, target_y,
                                         obj->obj.in_top,
                                         state->plr2.stood_in_top, 0);
            if (vis) obj->obj.can_see |= 0x02;
        }
    }
}

/* -----------------------------------------------------------------------
 * Enemy helper timers and class-specific support
 * ----------------------------------------------------------------------- */
static int16_t enemy_tick_third_timer(GameObject *obj, GameState *state,
                                      int can_see, int16_t no_los_base,
                                      int16_t no_los_mask)
{
    int16_t third_timer = OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF);
    if (can_see) {
        third_timer -= state->temp_frames;
        if (third_timer < 0) third_timer = 0;
    } else {
        third_timer = no_los_base;
        if (no_los_mask > 0) {
            third_timer = (int16_t)(third_timer + (rand() & no_los_mask));
        }
    }
    OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, third_timer);
    return third_timer;
}

/* Amiga SecTimer: occasional howl/hiss (NormalAlien.s lines 278–303, 582–608;
 * FlyingScalyBall.s 352–374, 662–684). Sample = howl1/howl2 (17–18) or newhiss (16). */
static void enemy_tick_sec_timer_vocal(GameObject *obj, GameState *state,
                                       const EnemyParams *params, int attacking)
{
    if (params->periodic_vocal1 < 0)
        return;
    if (params->hiss_timer_min <= 0 && params->hiss_timer_range <= 0)
        return;

    int16_t sec = OBJ_TD_W(obj, ENEMY_SEC_TIMER_OFF);
    sec = (int16_t)(sec - state->temp_frames);
    if (sec < 0) {
        int samp = params->periodic_vocal1;
        if (params->periodic_vocal2 >= params->periodic_vocal1) {
            int vspan = params->periodic_vocal2 - params->periodic_vocal1 + 1;
            samp = params->periodic_vocal1 + (int)(rand() % vspan);
        }
        int noisevol = attacking ? params->periodic_vol_attack : params->periodic_vol_idle;
        if (noisevol < 1) noisevol = 1;
        int vol = (noisevol * 255 + 400) / 800;
        if (vol < 12) vol = 12;
        if (vol > 255) vol = 255;
        audio_play_sample(samp, vol);

        int tspan = (int)params->hiss_timer_range + 1;
        if (tspan < 1) tspan = 1;
        sec = (int16_t)(params->hiss_timer_min + (rand() % tspan));
    }
    OBJ_SET_TD_W(obj, ENEMY_SEC_TIMER_OFF, sec);
}

/* Amiga ObjectDataHandler only runs enemy logic when objWorry != 0. */
static bool object_type_uses_worry_gate(int8_t obj_type)
{
    switch (obj_type) {
    case OBJ_NBR_ALIEN:
    case OBJ_NBR_ROBOT:
    case OBJ_NBR_BIG_NASTY:
    case OBJ_NBR_FLYING_NASTY:
    case OBJ_NBR_EYEBALL:
    case OBJ_NBR_MARINE:
    case OBJ_NBR_WORM:
    case OBJ_NBR_HUGE_RED_THING:
    case OBJ_NBR_SMALL_RED_THING:
    case OBJ_NBR_TREE:
    case OBJ_NBR_TOUGH_MARINE:
    case OBJ_NBR_FLAME_MARINE:
        return true;
    default:
        return false;
    }
}

/* Amiga handlers set objY (4(a0)) from floor using per-type constants for most enemies,
 * not raw object height byte 7(a0). Matching these offsets avoids "floating" enemies. */
static int object_floor_render_offset_units(const GameObject *obj, int8_t obj_type)
{
    switch (obj_type) {
    case OBJ_NBR_ALIEN:           return 40;   /* NormalAlien.s: sub.w #40,d0 */
    case OBJ_NBR_ROBOT:           return 120;  /* Robot.s: sub.w #120,d0 */
    case OBJ_NBR_BIG_NASTY:       return 70;   /* BigUglyAlien.s: sub.w #70,d0 */
    case OBJ_NBR_MARINE:
    case OBJ_NBR_TOUGH_MARINE:
    case OBJ_NBR_FLAME_MARINE:    return 64;   /* *Marine.s: sub.w #64,d0 */
    case OBJ_NBR_WORM:            return 100;  /* HalfWorm.s: sub.w #100,d0 */
    case OBJ_NBR_HUGE_RED_THING:  return 256;  /* BigRedThing.s: sub.w #256,d0 */
    case OBJ_NBR_SMALL_RED_THING: return 128;  /* BigClaws.s: sub.w #128,d0 */
    case OBJ_NBR_TREE:            return 100;  /* Tree.s: sub.w #100,d0 */
    default: {
        int world_h = (int)(uint8_t)obj->raw[7];
        if (world_h == 0) world_h = 32;
        return world_h;
    }
    }
}

/* Amiga handlers (most enemies): decay low 7 bits, preserve bit 7 latch. */
static void enemy_decay_worry_latched(GameObject *obj)
{
    uint8_t w = (uint8_t)obj->obj.worry;
    uint8_t hi = (uint8_t)(w & 0x80u);
    uint8_t lo = (uint8_t)(w & 0x7Fu);
    if (lo > 0)
        lo--;
    obj->obj.worry = (int8_t)(hi | lo);
}

/* Amiga Robot.s uses "sub.b #1,objWorry(a0)" (full byte decay). */
static void enemy_decay_worry_full_byte(GameObject *obj)
{
    uint8_t w = (uint8_t)obj->obj.worry;
    w = (uint8_t)(w - 1u);
    obj->obj.worry = (int8_t)w;
}

static bool enemy_type_slot_reusable_for_tree_spawn(int8_t obj_type)
{
    switch (obj_type) {
    case OBJ_NBR_DEAD:
    case OBJ_NBR_ALIEN:
    case OBJ_NBR_ROBOT:
    case OBJ_NBR_BIG_NASTY:
    case OBJ_NBR_FLYING_NASTY:
    case OBJ_NBR_MARINE:
    case OBJ_NBR_WORM:
    case OBJ_NBR_HUGE_RED_THING:
    case OBJ_NBR_SMALL_RED_THING:
    case OBJ_NBR_TREE:
    case OBJ_NBR_EYEBALL:
    case OBJ_NBR_TOUGH_MARINE:
    case OBJ_NBR_FLAME_MARINE:
    case OBJ_NBR_GAS_PIPE:
        return true;
    default:
        return false;
    }
}

static bool enemy_spawn_tree_eyeball(GameObject *tree, GameState *state)
{
    if (!state->level.object_data || !state->level.object_points) return false;

    int16_t tree_x = 0, tree_z = 0;
    get_object_pos(&state->level, (int)OBJ_CID(tree), &tree_x, &tree_z);

    int obj_index = 0;
    while (1) {
        GameObject *obj = get_object(&state->level, obj_index);
        if (!obj) break;
        if (OBJ_CID(obj) < 0) break;
        if (obj == tree) {
            obj_index++;
            continue;
        }
        if (OBJ_ZONE(obj) >= 0) {
            obj_index++;
            continue;
        }
        if (!enemy_type_slot_reusable_for_tree_spawn(obj->obj.number)) {
            obj_index++;
            continue;
        }

        {
            int16_t spawn_cid = OBJ_CID(obj);
            memset(obj->raw + 2, 0, OBJECT_SIZE - 2);
            OBJ_SET_CID(obj, spawn_cid);
        }

        obj->obj.number = OBJ_NBR_EYEBALL;
        obj->obj.can_see = 0;
        obj->obj.worry = 1;
        obj->obj.in_top = tree->obj.in_top;
        OBJ_SET_ZONE(obj, OBJ_ZONE(tree));
        OBJ_SET_GROOM(obj, OBJ_GROOM(tree));
        obj_sw(obj->raw + 4, obj_w(tree->raw + 4));
        obj->raw[6] = (uint8_t)default_object_world_size[OBJ_NBR_EYEBALL].w;
        obj->raw[7] = (uint8_t)default_object_world_size[OBJ_NBR_EYEBALL].h;

        NASTY_LIVES(*obj) = 10;
        NASTY_DAMAGE(*obj) = 0;
        NASTY_SET_MAXSPD(*obj, 5);
        NASTY_SET_CURRSPD(*obj, 0);
        NASTY_SET_FACING(*obj, NASTY_FACING(*tree));
        {
            int16_t turn_speed = (int16_t)(((rand() >> 4) & 255) - 128);
            turn_speed = (int16_t)(turn_speed * 2);
            OBJ_SET_TD_W(obj, ENEMY_TURN_SPEED_OFF, turn_speed);

            int16_t yv = (int16_t)(((rand() >> 4) & 7) - 3);
            yv = (int16_t)(yv - ((rand() >> 5) & 1));
            OBJ_SET_TD_W(obj, ENEMY_OBJ_YVEL_OFF, yv);
        }
        NASTY_SET_TIMER(*obj, 100);
        OBJ_SET_TD_W(obj, ENEMY_SEC_TIMER_OFF, 100);
        OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 100);
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, 100);
        OBJ_SET_DEADH(obj, 0);
        OBJ_SET_DEADL(obj, 0);

        {
            int cid = (int)OBJ_CID(obj);
            if (cid >= 0 && cid < state->level.num_object_points) {
                uint8_t *pt = state->level.object_points + cid * 8;
                obj_sw(pt, tree_x);
                obj_sw(pt + 4, tree_z);
            }
        }
        return true;
    }

    return false;
}

static int16_t enemy_anim_vect_for_type(int8_t obj_type)
{
    switch (obj_type) {
    case OBJ_NBR_ALIEN:           return 0;
    case OBJ_NBR_ROBOT:           return 5;
    case OBJ_NBR_BIG_NASTY:       return 3;
    case OBJ_NBR_FLYING_NASTY:    return 4;
    case OBJ_NBR_EYEBALL:         return 0;
    case OBJ_NBR_MARINE:          return 10;
    case OBJ_NBR_WORM:            return 13;
    case OBJ_NBR_HUGE_RED_THING:
    case OBJ_NBR_SMALL_RED_THING: return 14;
    case OBJ_NBR_TREE:            return 15;
    case OBJ_NBR_TOUGH_MARINE:    return 16;
    case OBJ_NBR_FLAME_MARINE:    return 17;
    default:                      return -1;
    }
}

static int32_t enemy_move_y_for_context(const GameObject *obj,
                                        const EnemyParams *params,
                                        const GameState *state,
                                        int zone_slots)
{
    /* Default Amiga pattern in handlers:
     *   move.w 4(a0),d0
     *   sub.w #(thingheight>>8),d0
     *   asl.l #7,d0 -> newy
     */
    int32_t move_y = (((int32_t)obj_w(obj->raw + 4) -
                       (int32_t)(params->thing_height >> 8)) << 7);

    if (OBJ_ZONE(obj) >= 0 && state->level.zone_adds && state->level.data) {
        int src_zone = level_connect_to_zone_index(&state->level, OBJ_ZONE(obj));
        if (src_zone < 0 && OBJ_ZONE(obj) < zone_slots)
            src_zone = OBJ_ZONE(obj);
        if (src_zone >= 0 && src_zone < zone_slots) {
            int32_t zo = (int32_t)be32(state->level.zone_adds + (uint32_t)src_zone * 4u);
            const uint8_t *zd = state->level.data + zo;
            int32_t floor_h = be32(zd + (obj->obj.in_top ? ZONE_OFF_UPPER_FLOOR : ZONE_OFF_FLOOR));

            /* Most ground enemies effectively use floor-thingheight for MoveObject.
             * Big Ugly is a documented outlier in BigUglyAlien.s:
             * newy = (objY<<7), while objY is maintained as floor>>7 - 70. */
            switch (obj->obj.number) {
            case OBJ_NBR_FLYING_NASTY:
            case OBJ_NBR_EYEBALL:
                /* Keep floating variants on their scripted objY-based path. */
                break;
            case OBJ_NBR_BIG_NASTY:
                move_y = floor_h - (70 * 128);
                break;
            default:
                move_y = floor_h - params->thing_height;
                break;
            }
        }
    }
    return move_y;
}

/* Amiga EyeBall.s / FlyingScalyBall.s vertical bob:
 * add objyvel to objY, then clamp between floor/roof bands and flip direction. */
static void enemy_update_flying_vertical(GameObject *obj, const GameState *state)
{
    if (!state || !state->level.zone_adds || !state->level.data) return;

    int zone_slots = level_zone_slot_count(&state->level);
    int16_t zid = OBJ_ZONE(obj);
    if (zid < 0) return;

    int src_zone = level_connect_to_zone_index(&state->level, zid);
    if (src_zone < 0 && zid < zone_slots) src_zone = zid;
    if (src_zone < 0 || src_zone >= zone_slots) return;

    int32_t zo = (int32_t)be32(state->level.zone_adds + (uint32_t)src_zone * 4u);
    if (zo <= 0) return;
    const uint8_t *room = state->level.data + zo;

    int32_t floor_h = be32(room + ZONE_OFF_FLOOR);
    int32_t roof_h  = be32(room + ZONE_OFF_ROOF);
    if (obj->obj.in_top) {
        floor_h = be32(room + ZONE_OFF_UPPER_FLOOR);
        roof_h  = be32(room + ZONE_OFF_UPPER_ROOF);
    }

    int16_t yvel = OBJ_TD_W(obj, ENEMY_OBJ_YVEL_OFF);
    int16_t y = obj_w(obj->raw + 4);
    y = (int16_t)(y + yvel);

    {
        int32_t y_fp = ((int32_t)y) << 7;
        int32_t d2 = y_fp + (48 * 256);
        int32_t d3 = y_fp - (48 * 256);

        if (d2 >= floor_h) {
            d2 = floor_h;
            d3 = d2;
            yvel = (int16_t)-yvel;
            d3 -= (96 * 256);
        }

        if (d3 <= roof_h) {
            d3 = roof_h;
            yvel = (int16_t)-yvel;
        }

        d3 += (48 * 256);
        y = (int16_t)(d3 >> 7);
    }

    OBJ_SET_TD_W(obj, ENEMY_OBJ_YVEL_OFF, yvel);
    obj_sw(obj->raw + 4, y);
}

/* ab3d.ini all_keys: OR key bits from every key object slot (same as object_handle_key pickup). */
static void game_apply_all_keys_from_level(const GameState *state)
{
    if (!state->cfg_all_keys || !state->level.object_data)
        return;
    const uint8_t *obj = state->level.object_data;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if ((int8_t)obj[16] == OBJ_NBR_KEY)
            game_conditions |= (int8_t)obj[17];
        obj += OBJECT_SIZE;
    }
}

/* -----------------------------------------------------------------------
 * objects_update - Main per-frame object processing
 *
 * Translated from Anims.s ObjMoveAnim.
 * ----------------------------------------------------------------------- */
void objects_update(GameState *state)
{
    const int zone_slots = level_zone_slot_count(&state->level);

    gib_impact_splat_sound_this_update = false;

    /* 1. Update player zones (from room pointer if available) */
    /* Zone is already maintained by player_full_control -> MoveObject */

    /* 2. Player shooting - called from game_loop */

    /* 3. Level mechanics */
    game_apply_all_keys_from_level(state);
    switch_routine(state);
    door_routine(state);

    /* Set stood_on_lift from current zone and Y (same frame: use zone floor from previous frame). */
    if (state->level.lift_data && state->level.zone_adds && state->level.data) {
        for (int p = 0; p < 2; p++) {
            PlayerState *plr = (p == 0) ? &state->plr1 : &state->plr2;
            plr->stood_on_lift = 0;
            int16_t zid = plr->zone;
            if (zid < 0 || zid >= zone_slots) continue;
            int32_t zone_off = (int32_t)be32(state->level.zone_adds + (uint32_t)zid * 4u);
            if (zone_off < 0) continue;
            const uint8_t *zd = state->level.data + zone_off;
            int32_t zone_floor = be32(zd + ZONE_OFF_FLOOR);
            int32_t floor_y = zone_floor - plr->s_height;
            int32_t tol = 512;
            if (plr->s_yoff >= floor_y - tol && plr->s_yoff <= floor_y + tol) {
                const uint8_t *lift = state->level.lift_data;
                while (1) {
                    int16_t lz = be16(lift);
                    if (lz < 0) break;
                    if (lz == zid) {
                        plr->stood_on_lift = 1;
                        break;
                    }
                    lift += LIFT_ENTRY_SIZE;
                }
            }
        }
    }

    lift_routine(state);

    /* Water animations */
    do_water_anims(state);

    /* 4. Iterate all objects */
    if (!state->level.object_data) {
        return;
    }

    /* Animation timer */
    anim_timer -= state->temp_frames;
    if (anim_timer <= 0) {
        anim_timer = 2;
        /* Swap RipTear/otherrip buffers (for rendering) */
    }

    /* Advance walk-cycle counter (Amiga: alanptr steps through alan[] table each game tick).
     * walk_cycle 0-31 wraps; walk_step = walk_cycle >> 3 gives 0-3 (8 ticks per step). */
    walk_cycle = (uint8_t)((walk_cycle + (uint8_t)state->temp_frames) & 31u);

    int obj_index = 0;
    while (1) {
        GameObject *obj = get_object(&state->level, obj_index);
        if (!obj) break;
        if (OBJ_CID(obj) < 0) break;
        if (OBJ_ZONE(obj) < 0) {
            obj_index++;
            continue;
        }

        /* Dispatch by object type (needed before Y refresh — Amiga gates on objNumber). */
        int8_t obj_type = obj->obj.number;

        /* Update rendering Y from live zone floor (lift_routine / doors write ToZoneFloor each tick).
         * ObjectDataHandler (Anims.s): objNumber < 0 skips ItsA* — we still skip for decorative 3D
         * (obj[6]==OBJ_3D_SPRITE) so level-authored obj[4] is preserved (e.g. exit sign).
         * Billboard corpses (OBJ_NBR_DEAD, not 3D) must refresh like pickups or they stay fixed
         * while a moving lift changes the floor under them. */
        {
            int corpse_on_floor = (obj_type == OBJ_NBR_DEAD &&
                                   (uint8_t)obj->raw[6] != (uint8_t)OBJ_3D_SPRITE);
            int flying_hover = (obj_type == OBJ_NBR_FLYING_NASTY ||
                                obj_type == OBJ_NBR_EYEBALL);
            if ((obj_type >= 0 || corpse_on_floor) && !flying_hover) {
                int16_t obj_zone = OBJ_ZONE(obj);
                if (obj_zone >= 0 && obj_zone < zone_slots &&
                    state->level.zone_adds && state->level.data) {
                    int32_t zo = be32(state->level.zone_adds + obj_zone * 4);
                    if (zo > 0) {
                        const uint8_t *zd = state->level.data + zo;
                        int32_t floor_h = be32(zd + 2);  /* ToZoneFloor */

                        int32_t upper_floor = be32(zd + 10);
                        if (upper_floor != 0 && obj->obj.in_top)
                            floor_h = upper_floor;

                        int world_h = object_floor_render_offset_units(obj, obj_type);

                        int16_t render_y = (int16_t)((floor_h >> 7) - world_h);
                        obj_sw(obj->raw + 4, render_y);
                    }
                }
            }
        }

        int param_idx;

        /* Match Amiga wake behavior: dormant enemies (worry==0) do not run AI. */
        if (object_type_uses_worry_gate(obj_type) && obj->obj.worry == 0) {
            obj_index++;
            continue;
        }

        switch (obj_type) {
        case OBJ_NBR_ALIEN:
            object_handle_alien(obj, state);
            break;
        case OBJ_NBR_MEDIKIT:
            object_handle_medikit(obj, state);
            break;
        case OBJ_NBR_BULLET:
            object_handle_bullet(obj, state);
            break;
        case OBJ_NBR_BIG_GUN:
            object_handle_big_gun(obj, state);
            break;
        case OBJ_NBR_KEY:
            object_handle_key(obj, state);
            break;
        case OBJ_NBR_PLR1:
        case OBJ_NBR_PLR2:
            break;
        case OBJ_NBR_ROBOT:
            object_handle_robot(obj, state);
            break;
        case OBJ_NBR_BIG_NASTY:
            object_handle_big_nasty(obj, state);
            break;
        case OBJ_NBR_FLYING_NASTY:
        case OBJ_NBR_EYEBALL:
            object_handle_flying_nasty(obj, state);
            break;
        case OBJ_NBR_AMMO:
            object_handle_ammo(obj, state);
            break;
        case OBJ_NBR_BARREL:
            object_handle_barrel(obj, state);
            break;
        case OBJ_NBR_MARINE:
        case OBJ_NBR_TOUGH_MARINE:
        case OBJ_NBR_FLAME_MARINE:
            object_handle_marine(obj, state);
            break;
        case OBJ_NBR_WORM:
            object_handle_worm(obj, state);
            break;
        case OBJ_NBR_HUGE_RED_THING:
        case OBJ_NBR_SMALL_RED_THING:
            object_handle_huge_red(obj, state);
            break;
        case OBJ_NBR_TREE:
            object_handle_tree(obj, state);
            break;
        case OBJ_NBR_GAS_PIPE:
            object_handle_gas_pipe(obj, state);
            break;
        case OBJ_NBR_DEAD:
            /* Decorative vector objects in level data also use obj.number == -1.
             * Amiga ObjectDataHandler skips all negative objNumber entries; do not run
             * corpse animation logic on 3D vector props (e.g. EXIT sign), or their
             * objVectNumber/frame fields at +8/+10 get clobbered. */
            if ((uint8_t)obj->raw[6] == (uint8_t)OBJ_3D_SPRITE)
                break;

            /* Advance death animation; original type stored in type_data[1] when we died.
             * Amiga: ThirdTimer counts down from 25, one step per ObjMoveAnim (~0.5 s at 50 Hz).
             * Advance every tick (dead_l >= 1) to match. */
            {
                int8_t original_type = obj->obj.type_data[1];
                param_idx = obj_type_to_enemy_index(original_type);
                if (param_idx >= 0 && param_idx < num_enemy_types) {
                    const EnemyParams *ep = &enemy_params[param_idx];
                    int8_t death_index = obj->obj.type_data[0];
                    int16_t dead_l = OBJ_DEADL(obj);
                    dead_l += state->temp_frames;
                    if (dead_l >= 1) {
                        dead_l = 0;
                        /* Only advance if next frame is valid; otherwise keep showing last frame (don't remove) */
                        if (death_index + 1 < 30 && ep->death_frames[death_index + 1] >= 0) {
                            death_index++;
                            obj->obj.type_data[0] = death_index;
                            OBJ_SET_DEADH(obj, ep->death_frames[death_index]);
                        }
                    }
                    OBJ_SET_DEADL(obj, dead_l);
                }
            }
            break;
        default:
            break;
        }

        /* Refresh display-facing for the current camera while preserving the
         * walk/attack sub-frame in the low 2 bits (Amiga: frame=angle*4+step). */
        if (NASTY_LIVES(*obj) > 0) {
            switch (obj_type) {
            case OBJ_NBR_ALIEN: case OBJ_NBR_ROBOT: case OBJ_NBR_BIG_NASTY:
            case OBJ_NBR_FLYING_NASTY: case OBJ_NBR_EYEBALL:
            case OBJ_NBR_MARINE: case OBJ_NBR_TOUGH_MARINE: case OBJ_NBR_FLAME_MARINE:
            case OBJ_NBR_WORM: case OBJ_NBR_HUGE_RED_THING: case OBJ_NBR_SMALL_RED_THING:
            case OBJ_NBR_TREE:
                {
                    const PlayerState *view_plr = (state->mode == MODE_SLAVE)
                        ? &state->plr2 : &state->plr1;
                    int16_t view_x = (int16_t)(view_plr->xoff >> 16);
                    int16_t view_z = (int16_t)(view_plr->zoff >> 16);
                    enemy_update_display_facing_frame(obj, state, view_x, view_z);
                }
                break;
            default:
                break;
            }
        }

        obj_index++;
    }

    /* Process nasty_shot_data bullets and gibs (not in object_data list) */
    if (state->level.nasty_shot_data) {
        uint8_t *shots = state->level.nasty_shot_data;
        for (int j = 0; j < NASTY_SHOT_SLOT_COUNT; j++) {
            GameObject *bullet = (GameObject *)(shots + j * OBJECT_SIZE);
            if (OBJ_ZONE(bullet) < 0) continue;
            if (bullet->obj.number != OBJ_NBR_BULLET) continue;
            object_handle_bullet(bullet, state);
        }
    }
    if (state->level.player_shot_data) {
        uint8_t *shots = state->level.player_shot_data;
        for (int j = 0; j < PLAYER_SHOT_SLOT_COUNT; j++) {
            GameObject *bullet = (GameObject *)(shots + j * OBJECT_SIZE);
            if (OBJ_ZONE(bullet) < 0) continue;
            if (bullet->obj.number != OBJ_NBR_BULLET) continue;
            object_handle_bullet(bullet, state);
        }
    }
}

/* -----------------------------------------------------------------------
 * Update sprite rotation frames for visible enemies (call every display frame).
 * ----------------------------------------------------------------------- */
void objects_update_sprite_frames(GameState *state)
{
    if (!state->level.object_data || !state->level.zone_adds) return;

    uint8_t vis_zones[256];
    memset(vis_zones, 0, sizeof(vis_zones));
    if (state->zone_order_count > 0) {
        for (int i = 0; i < state->zone_order_count && i < 256; i++) {
            int16_t z = state->zone_order_zones[i];
            if (z >= 0 && z < 256) vis_zones[z] = 1;
        }
    } else {
        for (int z = 0; z < 256; z++) vis_zones[z] = 1;
    }

    const PlayerState *view_plr = (state->mode == MODE_SLAVE)
        ? &state->plr2 : &state->plr1;
    int16_t view_x = (int16_t)(view_plr->xoff >> 16);
    int16_t view_z = (int16_t)(view_plr->zoff >> 16);

    int obj_index = 0;
    while (1) {
        GameObject *obj = get_object(&state->level, obj_index);
        if (!obj) break;
        if (OBJ_CID(obj) < 0) break;
        int16_t obj_zone = OBJ_ZONE(obj);
        if (obj_zone < 0 || obj_zone >= 256 || !vis_zones[obj_zone]) {
            obj_index++;
            continue;
        }
        if (NASTY_LIVES(*obj) <= 0) {
            obj_index++;
            continue;
        }

        int8_t obj_type = obj->obj.number;
        switch (obj_type) {
        case OBJ_NBR_ALIEN: case OBJ_NBR_ROBOT: case OBJ_NBR_BIG_NASTY:
        case OBJ_NBR_FLYING_NASTY: case OBJ_NBR_EYEBALL:
        case OBJ_NBR_MARINE: case OBJ_NBR_TOUGH_MARINE: case OBJ_NBR_FLAME_MARINE:
        case OBJ_NBR_WORM: case OBJ_NBR_HUGE_RED_THING: case OBJ_NBR_SMALL_RED_THING:
        case OBJ_NBR_TREE:
            {
                enemy_update_display_facing_frame(obj, state, view_x, view_z);
            }
            break;
        default:
            break;
        }
        obj_index++;
    }
}

/* -----------------------------------------------------------------------
 * Enemy handlers (per-class state loops)
 * ----------------------------------------------------------------------- */

static void enemy_handle_big_red_variant(GameObject *obj, GameState *state, bool big_claws)
{
    if (!state->nasty) return;
    enemy_decay_worry_latched(obj);

    int param_idx = big_claws ? 8 : 2;
    const EnemyParams *params = &enemy_params[param_idx];
    if (enemy_check_damage(obj, params, state)) return;
    if (NASTY_LIVES(*obj) <= 0) return;

    enemy_update_can_see(obj, state);
    int can_see = obj->obj.can_see & 0x03;
    int16_t third_timer = OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF);
    bool attacking = false;
    int target_player = 0;

    if (can_see && third_timer <= 0) {
        target_player = marine_pick_target_player(obj, state);
        attacking = (target_player != 0);
    }

    if (attacking) {
        (void)marine_track_target(obj, params, state, target_player, false);

        int16_t fourth_timer = OBJ_TD_W(obj, ENEMY_FOURTH_TIMER_OFF);
        fourth_timer -= state->temp_frames;
        if (fourth_timer <= 0) {
            OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 50);
        }

        if (fourth_timer < 20) {
            fourth_timer = 30;
            OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, (int16_t)(100 + (rand() & 0x7F)));
            audio_play_sample(9, 100);
            if (big_claws) {
                enemy_fire_at_player(obj, state, target_player, 2, 10, 64, 6);
            } else {
                enemy_fire_at_player(obj, state, target_player, 2, 10, 32, 4);
            }
        }
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, fourth_timer);
    } else {
        (void)enemy_tick_third_timer(obj, state, can_see, 20, 63);
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, 70);
        enemy_wander_with_timer(obj, params, state, 150, 0);
    }

    enemy_tick_sec_timer_vocal(obj, state, params, attacking);

    {
        int16_t vn = enemy_anim_vect_for_type(obj->obj.number);
        if (vn >= 0) {
            int walk_step = attacking ? 0 : ((walk_cycle >> 3) & 3);
            enemy_update_anim_with_step(obj, state, vn, walk_step);
        }
    }
}

void object_handle_alien(GameObject *obj, GameState *state)
{
    if (!state->nasty) return;
    enemy_decay_worry_latched(obj);

    const EnemyParams *params = &enemy_params[0];
    if (enemy_check_damage(obj, params, state)) return;
    if (NASTY_LIVES(*obj) <= 0) return;

    enemy_update_can_see(obj, state);
    int can_see = obj->obj.can_see & 0x03;
    int16_t third_timer = OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF);
    bool attacking = false;
    int target_player = 0;

    if (can_see && third_timer <= 0) {
        target_player = marine_pick_target_player(obj, state);
        attacking = (target_player != 0);
    }

    if (attacking) {
        int32_t dist = marine_track_target(obj, params, state, target_player, true);
        int16_t fourth_timer = OBJ_TD_W(obj, ENEMY_FOURTH_TIMER_OFF);

        if (dist <= params->melee_range) {
            fourth_timer -= state->temp_frames;
            if (fourth_timer <= 0) {
                player_add_damage(state, target_player, params->melee_damage);
                fourth_timer = 20;
            }
        }
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, fourth_timer);
    } else {
        (void)enemy_tick_third_timer(obj, state, can_see, 20, 63);
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, 25);
        enemy_wander_with_timer(obj, params, state, 50, 0);
    }

    enemy_tick_sec_timer_vocal(obj, state, params, attacking);

    enemy_update_anim(obj, state, 0);
}

void object_handle_robot(GameObject *obj, GameState *state)
{
    if (!state->nasty) return;
    enemy_decay_worry_full_byte(obj);

    const EnemyParams *params = &enemy_params[1];
    if (enemy_check_damage(obj, params, state)) return;
    if (NASTY_LIVES(*obj) <= 0) return;

    enemy_update_can_see(obj, state);
    int can_see = obj->obj.can_see & 0x03;
    int16_t third_timer = OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF);
    bool attacking = false;
    int target_player = 0;

    if (can_see && third_timer <= 0) {
        target_player = marine_pick_target_player(obj, state);
        attacking = (target_player != 0);
    }

    if (attacking) {
        (void)marine_track_target(obj, params, state, target_player, true);

        int16_t fourth_timer = OBJ_TD_W(obj, ENEMY_FOURTH_TIMER_OFF);
        fourth_timer -= state->temp_frames;
        if (fourth_timer <= 0) {
            OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 50);
        }

        if (fourth_timer < 20) {
            /* Robot.s gates shooting with canshootgun (must be facing target enough). */
            if (enemy_is_facing_player_cone(obj, state, target_player, 8192)) { /* cos >= 0.5 */
                fourth_timer = 50;
                OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, (int16_t)(150 + (rand() & 0x7F)));
                audio_play_sample(9, 100);
                enemy_fire_at_player(obj, state, target_player, 4, 10, 16, 3);
            }
        }
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, fourth_timer);
    } else {
        (void)enemy_tick_third_timer(obj, state, can_see, 20, 63);
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, 30);
        enemy_wander_with_timer(obj, params, state, 100, 63);
    }

    enemy_update_anim(obj, state, 5);
}

void object_handle_huge_red(GameObject *obj, GameState *state)
{
    enemy_handle_big_red_variant(obj, state, obj->obj.number == OBJ_NBR_SMALL_RED_THING);
}

void object_handle_worm(GameObject *obj, GameState *state)
{
    if (!state->nasty) return;
    enemy_decay_worry_latched(obj);

    const EnemyParams *params = &enemy_params[3];
    if (enemy_check_damage(obj, params, state)) return;
    if (NASTY_LIVES(*obj) <= 0) return;

    enemy_update_can_see(obj, state);
    int can_see = obj->obj.can_see & 0x03;
    int16_t third_timer = OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF);
    bool attacking = false;
    int target_player = 0;

    if (can_see && third_timer <= 0) {
        target_player = marine_pick_target_player(obj, state);
        attacking = (target_player != 0);
    }

    if (attacking) {
        (void)marine_track_target(obj, params, state, target_player, false);

        int16_t fourth_timer = OBJ_TD_W(obj, ENEMY_FOURTH_TIMER_OFF);
        fourth_timer -= state->temp_frames;
        if (fourth_timer <= 0) {
            OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 50);
        }

        if (fourth_timer < 20) {
            fourth_timer = 30;
            OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, (int16_t)(100 + (rand() & 0xFF)));
            audio_play_sample(9, 100);
            enemy_fire_at_player(obj, state, target_player, 5, 10, 16, 3);
        }
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, fourth_timer);
    } else {
        (void)enemy_tick_third_timer(obj, state, can_see, 20, 63);
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, 30);
        enemy_wander_with_timer(obj, params, state, 50, 0);
    }

    enemy_tick_sec_timer_vocal(obj, state, params, attacking);

    {
        int walk_step = attacking ? 0 : ((walk_cycle >> 3) & 3);
        enemy_update_anim_with_step(obj, state, 13, walk_step);
    }
}

static int marine_pick_target_player(GameObject *obj, GameState *state)
{
    int can_see = obj->obj.can_see & 0x03;
    if (can_see == 0x01) return 1;
    if (can_see == 0x02) return 2;
    if (can_see != 0x03) return 0;

    int16_t ox = 0, oz = 0;
    get_object_pos(&state->level, (int)OBJ_CID(obj), &ox, &oz);

    int32_t dx1 = (int32_t)state->plr1.p_xoff - (int32_t)ox;
    int32_t dz1 = (int32_t)state->plr1.p_zoff - (int32_t)oz;
    int32_t dx2 = (int32_t)state->plr2.p_xoff - (int32_t)ox;
    int32_t dz2 = (int32_t)state->plr2.p_zoff - (int32_t)oz;
    int32_t d1 = dx1 * dx1 + dz1 * dz1;
    int32_t d2 = dx2 * dx2 + dz2 * dz2;
    return (d1 <= d2) ? 1 : 2;
}

static int32_t marine_track_target(GameObject *obj, const EnemyParams *params,
                                   GameState *state, int player_num,
                                   bool apply_translation)
{
    PlayerState *plr = (player_num == 1) ? &state->plr1 : &state->plr2;

    int16_t obj_x = 0, obj_z = 0;
    get_object_pos(&state->level, (int)OBJ_CID(obj), &obj_x, &obj_z);

    int32_t target_x = (int32_t)plr->p_xoff;
    int32_t target_z = (int32_t)plr->p_zoff;
    int32_t dx = target_x - obj_x;
    int32_t dz = target_z - obj_z;
    int32_t dist = calc_dist_approx(dx, dz);

    if (obj->obj.number == OBJ_NBR_SMALL_RED_THING && dist > 0
        && dist < SMALL_RED_STANDOFF_DIST) {
        /* Mirror chase target across the enemy so head_towards_angle walks backward,
         * keeping them at a hittable range instead of under the barrel. */
        target_x = obj_x + (obj_x - target_x);
        target_z = obj_z + (obj_z - target_z);
    }

    int16_t speed = NASTY_MAXSPD(*obj);
    if (speed == 0) speed = 6;

    uint32_t collide_mask = 0xFFDE1;
    bool use_pre_collision = true;
    switch (obj->obj.number) {
    case OBJ_NBR_ALIEN:
        collide_mask = 0xFFDC1;
        break;
    case OBJ_NBR_WORM:
        collide_mask = 0x7FDE1;
        break;
    case OBJ_NBR_TREE:
        collide_mask = 0xDFDE1;
        break;
    case OBJ_NBR_HUGE_RED_THING:
        collide_mask = 0x0DE1;
        break;
    case OBJ_NBR_SMALL_RED_THING:
    case OBJ_NBR_FLAME_MARINE:
    case OBJ_NBR_TOUGH_MARINE:
    case OBJ_NBR_MARINE:
    case OBJ_NBR_FLYING_NASTY:
    case OBJ_NBR_EYEBALL:
        collide_mask = 0xFFDE1;
        break;
    case OBJ_NBR_ROBOT:
    case OBJ_NBR_BIG_NASTY:
        use_pre_collision = false;
        break;
    default:
        break;
    }

    MoveContext ctx;
    move_context_init(&ctx);
    ctx.oldx = obj_x;
    ctx.oldz = obj_z;
    ctx.thing_height = params->thing_height;
    int zone_slots = level_zone_slot_count(&state->level);
    {
        int32_t move_y = enemy_move_y_for_context(obj, params, state, zone_slots);
        ctx.oldy = move_y;
        ctx.newy = move_y;
    }
    ctx.step_up_val = params->step_up;
    ctx.step_down_val = params->step_down;
    ctx.extlen = params->extlen;
    ctx.awayfromwall = params->awayfromwall;
    ctx.collide_flags = collide_mask;
    ctx.coll_id = OBJ_CID(obj);
    ctx.pos_shift = 0;
    ctx.stood_in_top = obj->obj.in_top;

    if (OBJ_ZONE(obj) >= 0 && state->level.zone_adds && state->level.data) {
        int src_zone = level_connect_to_zone_index(&state->level, OBJ_ZONE(obj));
        if (src_zone < 0 && OBJ_ZONE(obj) < zone_slots)
            src_zone = OBJ_ZONE(obj);
        if (src_zone >= 0 && src_zone < zone_slots) {
            int32_t zo = (int32_t)be32(state->level.zone_adds + (uint32_t)src_zone * 4u);
            ctx.objroom = (uint8_t *)(state->level.data + zo);
        }
    }

    int16_t facing = NASTY_FACING(*obj);
    head_towards_angle(&ctx, &facing, target_x, target_z,
                       speed * state->temp_frames, 120);
    NASTY_SET_FACING(*obj, facing);

    if (use_pre_collision) {
        collision_check(&ctx, &state->level);
    }
    if (ctx.hitwall) {
        ctx.newx = ctx.oldx;
        ctx.newz = ctx.oldz;
    } else {
        /* Match original enemy chase movement (single MoveObject call). */
        move_object(&ctx, &state->level);
    }

    if (!apply_translation) {
        ctx.newx = ctx.oldx;
        ctx.newz = ctx.oldz;
    }

    /* Gameplay guardrail: red enemies must keep minimum spacing from players. */
    {
        int16_t sep_x = (int16_t)ctx.newx;
        int16_t sep_z = (int16_t)ctx.newz;
        enemy_enforce_player_separation(obj, state, &sep_x, &sep_z);
        ctx.newx = sep_x;
        ctx.newz = sep_z;
    }

    if (state->level.object_points) {
        int cid = (int)OBJ_CID(obj);
        uint8_t *pts = state->level.object_points + cid * 8;
        obj_sw(pts, (int16_t)ctx.newx);
        obj_sw(pts + 4, (int16_t)ctx.newz);
    }

    if (ctx.objroom && state->level.data) {
        int new_zone = level_zone_index_from_room_ptr(&state->level, ctx.objroom);
        if (new_zone < 0) {
            int16_t room_zone_word = (int16_t)((ctx.objroom[0] << 8) | ctx.objroom[1]);
            new_zone = level_connect_to_zone_index(&state->level, room_zone_word);
        }
        if (new_zone >= 0 && new_zone < zone_slots)
            OBJ_SET_ZONE(obj, (int16_t)new_zone);
        obj->obj.in_top = ctx.stood_in_top;
    }

    return dist;
}

/* True when the enemy is facing within a cone toward the target player.
 * min_cos_q14 uses the same scale as sin/cos lookup (16384 = 1.0). */
static bool enemy_is_facing_player_cone(GameObject *obj, GameState *state,
                                        int player_num, int16_t min_cos_q14)
{
    PlayerState *plr = (player_num == 1) ? &state->plr1 : &state->plr2;
    int16_t obj_x = 0, obj_z = 0;
    get_object_pos(&state->level, (int)OBJ_CID(obj), &obj_x, &obj_z);

    int32_t dx = (int32_t)plr->p_xoff - (int32_t)obj_x;
    int32_t dz = (int32_t)plr->p_zoff - (int32_t)obj_z;
    int32_t dist = calc_dist_approx(dx, dz);
    if (dist <= 0) return true;

    int16_t facing = NASTY_FACING(*obj);
    int32_t sf = sin_lookup(facing);
    int32_t cf = cos_lookup(facing);
    int64_t fwd = (int64_t)dx * sf + (int64_t)dz * cf;
    if (fwd <= 0) return false;

    return fwd >= (int64_t)dist * (int64_t)min_cos_q14;
}

static void marine_hitscan_burst(GameObject *obj, GameState *state,
                                 int player_num, int pellets, int damage)
{
    int16_t obj_x = 0, obj_z = 0;
    get_object_pos(&state->level, (int)OBJ_CID(obj), &obj_x, &obj_z);

    PlayerState *plr = (player_num == 1) ? &state->plr1 : &state->plr2;
    int32_t dx = (int32_t)plr->p_xoff - (int32_t)obj_x;
    int32_t dz = (int32_t)plr->p_zoff - (int32_t)obj_z;
    int32_t dist_sq = dx * dx + dz * dz;
    int32_t hit_threshold = dist_sq >> 6; /* Amiga compares against rand<<2 */

    for (int i = 0; i < pellets; i++) {
        int32_t r = (int32_t)(rand() & 0x7FFF) << 2;
        if (r > hit_threshold) {
            player_add_damage(state, player_num, damage);
        }
    }
}

void object_handle_marine(GameObject *obj, GameState *state)
{
    if (!state->nasty) return;
    enemy_decay_worry_latched(obj);

    int8_t type = obj->obj.number;

    int param_idx;
    if (type == OBJ_NBR_FLAME_MARINE)     param_idx = 4;
    else if (type == OBJ_NBR_TOUGH_MARINE) param_idx = 5;
    else                                   param_idx = 6;

    const EnemyParams *params = &enemy_params[param_idx];
    if (enemy_check_damage(obj, params, state)) return;

    int8_t lives = NASTY_LIVES(*obj);
    if (lives <= 0) return;

    enemy_update_can_see(obj, state);

    int can_see = obj->obj.can_see & 0x03;
    int16_t third_timer = OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF);
    int16_t fourth_reset = (type == OBJ_NBR_FLAME_MARINE) ? 30 : 25;
    bool attacking = false;
    int target_player = 0;

    if (can_see && third_timer <= 0) {
        target_player = marine_pick_target_player(obj, state);
        attacking = (target_player != 0);
    }

    if (attacking) {
        bool apply_translation = (type == OBJ_NBR_MARINE);
        (void)marine_track_target(obj, params, state, target_player, apply_translation);

        int16_t fourth_timer = OBJ_TD_W(obj, ENEMY_FOURTH_TIMER_OFF);
        fourth_timer -= state->temp_frames;
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, fourth_timer);
        if (fourth_timer <= 0) {
            OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 50);
        }

        if ((type == OBJ_NBR_MARINE && fourth_timer <= 20) ||
            (type != OBJ_NBR_MARINE && fourth_timer < 20)) {
            if (type == OBJ_NBR_MARINE) {
                OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 50 + (int16_t)(rand() & 0xFF));
                if (params->attack_sound >= 0) audio_play_sample(params->attack_sound, 100);
                marine_hitscan_burst(obj, state, target_player, 1, 4);
            } else if (type == OBJ_NBR_TOUGH_MARINE) {
                OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 50 + (int16_t)(rand() & 0x1F));
                if (params->attack_sound >= 0) audio_play_sample(params->attack_sound, 100);
                enemy_fire_at_player(obj, state, target_player, 6, 7, 32, 4);
            } else if (type == OBJ_NBR_FLAME_MARINE) {
                fourth_timer = 25;
                if (params->attack_sound >= 0) audio_play_sample(params->attack_sound, 100);
                enemy_fire_at_player(obj, state, target_player, 3, 7, 8, 2);
            } else {
                OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 200 + (int16_t)(rand() & 0xFF));
                if (params->attack_sound >= 0) audio_play_sample(params->attack_sound, 100);
                marine_hitscan_burst(obj, state, target_player, 5, 2);
            }
            OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, fourth_timer);
        }
    } else {
        (void)enemy_tick_third_timer(obj, state, can_see, 20, 63);
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, fourth_reset);
        enemy_wander_with_timer(obj, params, state, 50, 0);
    }

    enemy_tick_sec_timer_vocal(obj, state, params, attacking);

    {
        int16_t vn = (type == OBJ_NBR_TOUGH_MARINE) ? 16 :
                     (type == OBJ_NBR_FLAME_MARINE) ? 17 : 10;
        int walk_step = attacking ? 0 : ((walk_cycle >> 3) & 3);
        enemy_update_anim_with_step(obj, state, vn, walk_step);
    }
}

void object_handle_big_nasty(GameObject *obj, GameState *state)
{
    if (!state->nasty) return;

    const EnemyParams *params = &enemy_params[7];
    if (enemy_check_damage(obj, params, state)) return;
    if (NASTY_LIVES(*obj) <= 0) return;

    enemy_update_can_see(obj, state);
    int can_see = obj->obj.can_see & 0x03;
    bool attacking = false;
    int target_player = 0;

    if (can_see && NASTY_TIMER(*obj) <= 0 && (rand() & 0xFF) <= 250) {
        target_player = marine_pick_target_player(obj, state);
        attacking = (target_player != 0);
    }

    if (attacking) {
        int32_t dist = marine_track_target(obj, params, state, target_player, true);
        if (dist <= 80) {
            PlayerState *plr = (target_player == 1) ? &state->plr1 : &state->plr2;
            plr->energy -= (int16_t)state->temp_frames;
            audio_play_sample(2, amiga_noisevol_to_pc(50)); /* BigUglyAlien.s GotThere */
        }

        /* BigUglyAlien.s: baddiegun (#9) at Noisevol 200; SecTimer = 50+(rand&7) after shot */
        int16_t sec_timer = OBJ_TD_W(obj, ENEMY_SEC_TIMER_OFF);
        sec_timer = (int16_t)(sec_timer - state->temp_frames);
        if (sec_timer < 0) {
            sec_timer = (int16_t)(50 + (rand() & 7));
            int vol = (200 * 255 + 400) / 800;
            if (vol > 255) vol = 255;
            audio_play_sample(9, vol);
            enemy_fire_at_player(obj, state, target_player, 1, 10, 16, 4);
        }
        OBJ_SET_TD_W(obj, ENEMY_SEC_TIMER_OFF, sec_timer);
    } else {
        enemy_wander_with_timer(obj, params, state, 20, 15);
    }

    enemy_update_anim(obj, state, 3);
}

void object_handle_big_claws(GameObject *obj, GameState *state)
{
    enemy_handle_big_red_variant(obj, state, true);
}

/* -----------------------------------------------------------------------
 * Flying Nasty / Eyeball
 *
 * Translated from FlyingScalyBall.s ItsAFlyingNasty.
 *
 * Unique behavior: continuous rotation, vertical bobbing.
 * ----------------------------------------------------------------------- */
void object_handle_flying_nasty(GameObject *obj, GameState *state)
{
    if (!state->nasty) return;
    enemy_decay_worry_latched(obj);

    /* Amiga scripts force per-type world size every tick. */
    if (obj->obj.number == OBJ_NBR_EYEBALL) {
        obj->raw[6] = 0x10;
        obj->raw[7] = 0x20;
    } else {
        obj->raw[6] = 0x60;
        obj->raw[7] = 0x60;
    }

    const EnemyParams *params = &enemy_params[9];

    if (enemy_check_damage(obj, params, state)) return;

    int8_t lives = NASTY_LIVES(*obj);
    if (lives <= 0) {
        if (obj->obj.number == OBJ_NBR_EYEBALL ||
            obj->obj.number == OBJ_NBR_FLYING_NASTY) {
            enemy_update_flying_soft_dead(obj, params, state);
        }
        return;
    }

    enemy_update_can_see(obj, state);
    int can_see = obj->obj.can_see & 0x03;
    int16_t third_timer = OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF);
    bool attacking = false;
    bool fired_this_tick = false;
    int target_player = 0;

    if (can_see && third_timer <= 0) {
        target_player = marine_pick_target_player(obj, state);
        attacking = (target_player != 0);
    }

    if (attacking) {
        (void)marine_track_target(obj, params, state, target_player, false);

        int16_t fourth_timer = OBJ_TD_W(obj, ENEMY_FOURTH_TIMER_OFF);
        fourth_timer -= state->temp_frames;
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, fourth_timer);
        if (fourth_timer <= 0) {
            OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 50);
        }

        if (fourth_timer < 20) {
            OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 50);
            audio_play_sample(20, 100);
            enemy_fire_at_player(obj, state, target_player, 0, 5, 16, 3);
            fired_this_tick = true;
        }
    } else {
        /* Continuous rotation while prowling */
        int16_t facing = NASTY_FACING(*obj);
        int16_t turn_speed = OBJ_TD_W(obj, ENEMY_TURN_SPEED_OFF);
        facing = (facing + turn_speed * state->temp_frames) & ANGLE_MASK;
        NASTY_SET_FACING(*obj, facing);

        (void)enemy_tick_third_timer(obj, state, can_see, 10, 31);
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, 30);

        enemy_wander_with_timer(obj, params, state, 50, 0);
    }

    enemy_update_flying_vertical(obj, state);

    enemy_tick_sec_timer_vocal(obj, state, params, attacking);

    int16_t vn_fly = (obj->obj.number == OBJ_NBR_EYEBALL) ? 0 : 4;
    int walk_step = (attacking && obj->obj.number != OBJ_NBR_FLYING_NASTY)
        ? 0 : ((walk_cycle >> 3) & 3);
    enemy_update_anim_with_step(obj, state, vn_fly, walk_step);

    if (obj->obj.number == OBJ_NBR_FLYING_NASTY && attacking) {
        int16_t plr_x, plr_z;
        if (state->plr1.zone >= 0) {
            plr_x = (int16_t)state->plr1.p_xoff;
            plr_z = (int16_t)state->plr1.p_zoff;
        } else {
            plr_x = (int16_t)state->plr2.p_xoff;
            plr_z = (int16_t)state->plr2.p_zoff;
        }

        int angle = enemy_viewpoint(obj, plr_x, plr_z, &state->level);
        if (fired_this_tick) {
            wbe16(obj->raw + 10, 17);  /* FlyingScalyBall.s fire frame */
        } else if (angle == 0) {
            wbe16(obj->raw + 10, 16);  /* Front-facing attack frame */
        }
    }
}

/* -----------------------------------------------------------------------
 * Tree enemy
 *
 * Translated from Tree.s ItsATree.
 * ----------------------------------------------------------------------- */
void object_handle_tree(GameObject *obj, GameState *state)
{
    if (!state->nasty) return;
    enemy_decay_worry_latched(obj);

    const EnemyParams *params = &enemy_params[10];
    if (enemy_check_damage(obj, params, state)) return;
    if (NASTY_LIVES(*obj) <= 0) return;

    enemy_update_can_see(obj, state);
    int can_see = obj->obj.can_see & 0x03;
    int16_t third_timer = OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF);
    bool attacking = false;
    int target_player = 0;

    if (can_see && third_timer <= 0) {
        target_player = marine_pick_target_player(obj, state);
        attacking = (target_player != 0);
    }

    if (attacking) {
        (void)marine_track_target(obj, params, state, target_player, false);

        int16_t fourth_timer = OBJ_TD_W(obj, ENEMY_FOURTH_TIMER_OFF);
        fourth_timer -= state->temp_frames;
        if (fourth_timer <= 0) {
            OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 50);
        }

        if (fourth_timer < 20) {
            fourth_timer = 30;
            OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, (int16_t)(300 + (rand() & 0x7F)));
            audio_play_sample(16, 100);
            (void)enemy_spawn_tree_eyeball(obj, state);
        }
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, fourth_timer);
    } else {
        (void)enemy_tick_third_timer(obj, state, can_see, 20, 63);
        OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, 30);
        enemy_wander_with_timer(obj, params, state, 50, 0);
    }

    enemy_tick_sec_timer_vocal(obj, state, params, attacking);

    enemy_update_anim_with_step(obj, state, 15, attacking ? 0 : ((walk_cycle >> 3) & 3));
}

/* -----------------------------------------------------------------------
 * Gas pipe flame emitter
 *
 * Translated from Anims.s ItsAGasPipe (line ~1230-1325).
 * Periodically spawns flame projectiles in its facing direction.
 * ----------------------------------------------------------------------- */
void object_handle_gas_pipe(GameObject *obj, GameState *state)
{
    obj->obj.worry = 0;

    int16_t tf = state->temp_frames;

    /* ThirdTimer = delay before starting a burst */
    int16_t third = NASTY_TIMER(*obj);
    if (third > 0) {
        NASTY_SET_TIMER(*obj, (int16_t)(third - tf));
        OBJ_SET_TD_W(obj, 6, 5);   /* SecTimer */
        OBJ_SET_TD_W(obj, 10, 10);  /* FourthTimer */
        return;
    }

    /* FourthTimer = interval between shots in burst */
    int16_t fourth = OBJ_TD_W(obj, 10);
    fourth -= tf;
    if (fourth > 0) {
        OBJ_SET_TD_W(obj, 10, fourth);
        return;
    }
    OBJ_SET_TD_W(obj, 10, 10);

    int16_t sec = OBJ_TD_W(obj, 6);
    sec--;
    OBJ_SET_TD_W(obj, 6, sec);
    if (sec <= 0) {
        NASTY_SET_TIMER(*obj, OBJ_TD_W(obj, 14));
    }
    if (sec == 4) audio_play_sample(22, 200);

    /* Spawn flame projectile */
    if (!state->level.nasty_shot_data) return;
    uint8_t *shots = state->level.nasty_shot_data;
    GameObject *bullet = NULL;
    for (int i = 0; i < NASTY_SHOT_SLOT_COUNT; i++) {
        GameObject *c = (GameObject*)(shots + i * OBJECT_SIZE);
        if (OBJ_ZONE(c) < 0) { bullet = c; break; }
    }
    if (!bullet) return;

    SHOT_ANIM(*bullet) = 0;
    bullet->obj.number = OBJ_NBR_BULLET;
    OBJ_SET_ZONE(bullet, OBJ_ZONE(obj));
    int16_t src_y = (int16_t)((obj->raw[4] << 8) | obj->raw[5]);
    src_y -= 80;
    bullet->raw[4] = (uint8_t)(src_y >> 8);
    bullet->raw[5] = (uint8_t)(src_y);
    SHOT_SET_ACCYPOS(*bullet, (int32_t)src_y << 7);
    SHOT_STATUS(*bullet) = 0;
    SHOT_SET_YVEL(*bullet, 0);
    SHOT_SIZE(*bullet) = 3;
    SHOT_SET_FLAGS(*bullet, 0);
    SHOT_SET_GRAV(*bullet, 0);
    SHOT_POWER(*bullet) = 7;
    SHOT_SET_LIFE(*bullet, 0);

    /* Copy position from gas pipe to bullet in ObjectPoints */
    if (state->level.object_points && state->level.object_data) {
        int self_idx = (int)(((uint8_t*)obj - state->level.object_data) / OBJECT_SIZE);
        int bul_idx = (int)OBJ_CID(bullet);
        if (bul_idx >= 0) {
            uint8_t *sp = state->level.object_points + self_idx * 8;
            uint8_t *dp = state->level.object_points + bul_idx * 8;
            memcpy(dp, sp, 8);
        }
    }

    int16_t facing = NASTY_FACING(*obj);
    int16_t s = sin_lookup(facing);
    int16_t c = cos_lookup(facing);
    int32_t xvel = (((int32_t)s << 4) >> 16);
    int32_t zvel = (((int32_t)c << 4) >> 16);
    SHOT_SET_XVEL(*bullet, xvel << 16);
    SHOT_SET_ZVEL(*bullet, zvel << 16);
    NASTY_SET_EFLAGS(*bullet, 0x00100020);
    bullet->obj.worry = 127;
}

/* -----------------------------------------------------------------------
 * Barrel
 *
 * Translated from Anims.s ItsABarrel.
 * Barrels explode on enough damage, dealing area damage.
 * ----------------------------------------------------------------------- */
void object_handle_barrel(GameObject *obj, GameState *state)
{
    obj->obj.worry = 0;
    /* Keep visibility bits updated so PlayerShoot's can_see gate can
     * treat barrels consistently with other targetable objects. */
    enemy_update_can_see(obj, state);

    /* Anims.s ItsABarrel: exploding state is vect=8, frame increments 0..7 then remove. */
    if (obj_w(obj->raw + 8) == 8) {
        int16_t tf = state->temp_frames;
        if (tf < 1) tf = 1;

        int16_t accum = OBJ_TD_W(obj, ENEMY_OBJ_TIMER_OFF);
        if (accum < 0) accum = 0;
        accum = (int16_t)(accum + tf);
        int steps = accum / EXPLOSION_FRAME_STEP_VBLANKS;
        accum = (int16_t)(accum % EXPLOSION_FRAME_STEP_VBLANKS);
        OBJ_SET_TD_W(obj, ENEMY_OBJ_TIMER_OFF, accum);

        if (steps <= 0) return;

        int16_t frame = obj_w(obj->raw + 10);
        while (steps-- > 0) {
            obj->raw[6] = (uint8_t)((uint8_t)obj->raw[6] + 4u);
            obj->raw[7] = (uint8_t)((uint8_t)obj->raw[7] + 4u);
            frame = (int16_t)(frame + 1);
            if (frame == 8) {
                OBJ_SET_ZONE(obj, -1);
                return;
            }
        }
        obj_sw(obj->raw + 10, frame);
        return;
    }

    /* Keep barrel anchored to floor each tick (y = floor>>7 - 60). */
    {
        int zone_slots = level_zone_slot_count(&state->level);
        int src_zone = level_connect_to_zone_index(&state->level, OBJ_ZONE(obj));
        if (src_zone < 0 && OBJ_ZONE(obj) >= 0 && OBJ_ZONE(obj) < zone_slots)
            src_zone = OBJ_ZONE(obj);
        if (src_zone >= 0 && src_zone < zone_slots && state->level.zone_adds && state->level.data) {
            int32_t zo = (int32_t)be32(state->level.zone_adds + (uint32_t)src_zone * 4u);
            const uint8_t *zd = state->level.data + zo;
            int32_t floor_h = be32(zd + (obj->obj.in_top ? ZONE_OFF_UPPER_FLOOR : ZONE_OFF_FLOOR));
            obj_sw(obj->raw + 4, (int16_t)((floor_h >> 7) - 60));
        }
    }

    uint8_t damage = obj->raw[19];
    if (damage == 0) return;

    obj->raw[19] = 0;
    int32_t lives = (int32_t)NASTY_LIVES(*obj);
    lives -= (int32_t)damage;
    if (lives > 0) {
        NASTY_LIVES(*obj) = (int8_t)lives;
        return;
    }

    NASTY_LIVES(*obj) = 0;

    int16_t bx, bz;
    get_object_pos(&state->level, (int)OBJ_CID(obj), &bx, &bz);

    /* Amiga ItsABarrel: move.w #40,d0 ; jsr ComputeBlast */
    compute_blast(state, bx, bz, ((int32_t)obj_w(obj->raw + 4)) << 7,
                  40, OBJ_ZONE(obj), obj->obj.in_top);
    audio_play_sample(15, 300);

    obj_sw(obj->raw + 8, 8);     /* vect = explosion */
    obj_sw(obj->raw + 10, 0);    /* frame = 0 */
    OBJ_SET_TD_W(obj, ENEMY_OBJ_TIMER_OFF, 0);
    obj->raw[14] = 0x20;         /* src cols = 32 */
    obj->raw[15] = 0x20;         /* src rows = 32 */
    obj_sw(obj->raw + 2, -30);   /* objVectBright */
}

/* -----------------------------------------------------------------------
 * Pickup: Medikit
 *
 * Translated from Anims.s ItsAMediKit (line ~1463-1591).
 * ----------------------------------------------------------------------- */
void object_handle_medikit(GameObject *obj, GameState *state)
{
    /* Check distance to each player */
    if (pickup_distance_check(obj, state, 1)) {
        PlayerState *plr = &state->plr1;
        if (plr->energy < PLAYER_MAX_ENERGY) {
            plr->energy += HEAL_FACTOR;
            if (plr->energy > PLAYER_MAX_ENERGY)
                plr->energy = PLAYER_MAX_ENERGY;
            printf("[PICKUP] player 1 picked up medikit\n");
            OBJ_SET_ZONE(obj, -1); /* Remove pickup */
            audio_play_sample(4, 50);
        }
    }
    if (state->mode != MODE_SINGLE && pickup_distance_check(obj, state, 2)) {
        PlayerState *plr = &state->plr2;
        if (plr->energy < PLAYER_MAX_ENERGY) {
            plr->energy += HEAL_FACTOR;
            if (plr->energy > PLAYER_MAX_ENERGY)
                plr->energy = PLAYER_MAX_ENERGY;
            printf("[PICKUP] player 2 picked up medikit\n");
            OBJ_SET_ZONE(obj, -1);
            audio_play_sample(4, 50);
        }
    }
}

/* -----------------------------------------------------------------------
 * Pickup: Ammo
 *
 * Translated from Anims.s ItsAnAmmoClip (line ~1601-1744).
 * ----------------------------------------------------------------------- */
void object_handle_ammo(GameObject *obj, GameState *state)
{
    /* Amiga ItsAnAmmoClip: set vect=PICKUPS(1) and frame from AMGR table every tick */
    int16_t ammo_gun_type = OBJ_TD_W(obj, 0);  /* AmmoType EQU 18 */
    bool ammo_type_valid = (ammo_gun_type >= 0 && ammo_gun_type < MAX_GUNS);
    if (ammo_type_valid) {
        obj_sw(obj->raw + 8,  1);  /* objVectNumber  = PICKUPS vect */
        obj_sw(obj->raw + 10, (int16_t)ammo_graphic_table[(int)ammo_gun_type]);
    }

    if (pickup_distance_check(obj, state, 1)) {
        if (ammo_type_valid) {
            PlayerState *plr = &state->plr1;
            int gun_idx = ammo_gun_type;
            int16_t ammo = plr->gun_data[gun_idx].ammo;
            ammo += AMMO_PER_CLIP * 8;
            if (ammo > MAX_AMMO_DISPLAY) ammo = (int16_t)MAX_AMMO_DISPLAY;
            plr->gun_data[gun_idx].ammo = ammo;
            printf("[PICKUP] player 1 picked up ammo (gun %d)\n", gun_idx);
            OBJ_SET_ZONE(obj, -1);
            audio_play_sample(11, 50);
        }
    }
    if (state->mode != MODE_SINGLE && pickup_distance_check(obj, state, 2)) {
        if (ammo_type_valid) {
            PlayerState *plr = &state->plr2;
            int gun_idx = ammo_gun_type;
            int16_t ammo = plr->gun_data[gun_idx].ammo;
            ammo += AMMO_PER_CLIP * 8;
            if (ammo > MAX_AMMO_DISPLAY) ammo = (int16_t)MAX_AMMO_DISPLAY;
            plr->gun_data[gun_idx].ammo = ammo;
            printf("[PICKUP] player 2 picked up ammo (gun %d)\n", gun_idx);
            OBJ_SET_ZONE(obj, -1);
            audio_play_sample(11, 50);
        }
    }
}

/* -----------------------------------------------------------------------
 * Pickup: Key
 *
 * Translated from Anims.s ItsAKey (line ~1905-2010).
 * Keys set condition bits that unlock doors.
 * ----------------------------------------------------------------------- */
void object_handle_key(GameObject *obj, GameState *state)
{
    if (pickup_distance_check(obj, state, 1)) {
        /* Determine which key from shared "can_see" */
        game_conditions |= obj->obj.can_see;
        printf("[PICKUP] player 1 picked up key (key_id %d)\n", obj->obj.can_see);
        OBJ_SET_ZONE(obj, -1);
        audio_play_sample(4, 50);
    }
    if (state->mode != MODE_SINGLE && pickup_distance_check(obj, state, 2)) {
        game_conditions |= obj->obj.can_see;
        printf("[PICKUP] player 2 picked up key (key_id %d)\n", obj->obj.can_see);
        OBJ_SET_ZONE(obj, -1);
        audio_play_sample(4, 50);
    }
}

/* -----------------------------------------------------------------------
 * Pickup: Big Gun (weapon pickup)
 *
 * Translated from Anims.s ItsABigGun (line ~1748-1890).
 * ----------------------------------------------------------------------- */
void object_handle_big_gun(GameObject *obj, GameState *state)
{
    if (pickup_distance_check(obj, state, 1)) {
        int pickup_idx = (int)(uint8_t)obj->obj.can_see; /* Amiga ItsABigGun byte 17 */
        int gun_idx = -1;
        int ammo_idx = -1;
        /* Amiga: a1 = PLR1_GunData+32 then index by pickup id (0..6) => gun slot 1..7.
         * Keep pickup id 7 as compatibility fallback to gun slot 7. */
        if (pickup_idx >= 0 && pickup_idx < (MAX_GUNS - 1)) {
            gun_idx = pickup_idx + 1;
            ammo_idx = pickup_idx;
        } else if (pickup_idx == (MAX_GUNS - 1)) {
            gun_idx = pickup_idx;
            ammo_idx = pickup_idx;
        }
        if (gun_idx >= 0 && gun_idx < MAX_GUNS && ammo_idx >= 0 && ammo_idx < MAX_GUNS) {
            PlayerState *plr = &state->plr1;
            plr->gun_data[gun_idx].visible = -1; /* Mark as acquired */
            /* Add some ammo */
            int16_t ammo_add = ammo_in_guns[ammo_idx] * 8;
            plr->gun_data[gun_idx].ammo += ammo_add;
            printf("[PICKUP] player 1 picked up big gun (pickup %d -> gun %d)\n",
                   pickup_idx, gun_idx);
            OBJ_SET_ZONE(obj, -1);
            audio_play_sample(4, 50);
        }
    }
    if (state->mode != MODE_SINGLE && pickup_distance_check(obj, state, 2)) {
        int pickup_idx = (int)(uint8_t)obj->obj.can_see;
        int gun_idx = -1;
        int ammo_idx = -1;
        if (pickup_idx >= 0 && pickup_idx < (MAX_GUNS - 1)) {
            gun_idx = pickup_idx + 1;
            ammo_idx = pickup_idx;
        } else if (pickup_idx == (MAX_GUNS - 1)) {
            gun_idx = pickup_idx;
            ammo_idx = pickup_idx;
        }
        if (gun_idx >= 0 && gun_idx < MAX_GUNS && ammo_idx >= 0 && ammo_idx < MAX_GUNS) {
            PlayerState *plr = &state->plr2;
            plr->gun_data[gun_idx].visible = -1;
            int16_t ammo_add = ammo_in_guns[ammo_idx] * 8;
            plr->gun_data[gun_idx].ammo += ammo_add;
            printf("[PICKUP] player 2 picked up big gun (pickup %d -> gun %d)\n",
                   pickup_idx, gun_idx);
            OBJ_SET_ZONE(obj, -1);
            audio_play_sample(4, 50);
        }
    }
}

/* -----------------------------------------------------------------------
 * Bullet processing
 *
 * Translated from Anims.s ItsABullet (line ~2774-3384).
 * ----------------------------------------------------------------------- */
void object_handle_bullet(GameObject *obj, GameState *state)
{
    int32_t xvel = SHOT_XVEL(*obj);
    int32_t zvel = SHOT_ZVEL(*obj);
    int16_t yvel = SHOT_YVEL(*obj);
    int16_t grav = SHOT_GRAV(*obj);
    int16_t life = SHOT_LIFE(*obj);
    int16_t flags = SHOT_FLAGS(*obj);
    int8_t  shot_status = SHOT_STATUS(*obj);
    int8_t  shot_size = SHOT_SIZE(*obj);
    int zone_slots = level_zone_slot_count(&state->level);
    int anim_ticks = (int)state->temp_frames;
    if (anim_ticks < 1) anim_ticks = 1;
    bool    timed_out = false;

    /* Popping path (Amiga ItsABullet shotstatus!=0): advance pop sequence only. */
    if (shot_status != 0) {
        if (shot_size < 0 || shot_size >= MAX_BULLET_ANIM_IDX ||
            !bullet_pop_tables[shot_size]) {
            OBJ_SET_ZONE(obj, -1);
            SHOT_STATUS(*obj) = 0;
            SHOT_ANIM(*obj) = 0;
            return;
        }

        const BulletAnimFrame *table = bullet_pop_tables[shot_size];
        int pop_step_vblanks = 1;
        if (shot_size == 2 || shot_size == 4) {
            /* Keep RockPop-derived sequences at their intended slower cadence. */
            pop_step_vblanks = EXPLOSION_FRAME_STEP_VBLANKS;
        }
        int16_t pop_accum = SHOT_LIFE(*obj);
        if (pop_accum < 0) pop_accum = 0;
        pop_accum = (int16_t)(pop_accum + anim_ticks);
        int anim_steps = pop_accum / pop_step_vblanks;
        pop_accum = (int16_t)(pop_accum % pop_step_vblanks);
        SHOT_SET_LIFE(*obj, pop_accum);
        if (anim_steps < 1) return;

        uint8_t anim_idx = SHOT_ANIM(*obj);
        int32_t acc = SHOT_ACCYPOS(*obj);
        const BulletAnimFrame *f = NULL;
        while (anim_steps > 0) {
            f = &table[anim_idx];
            if (f->width < 0) {
                OBJ_SET_ZONE(obj, -1);
                SHOT_STATUS(*obj) = 0;
                SHOT_ANIM(*obj) = 0;
                return;
            }
            acc += ((int32_t)f->y_offset << 7);
            anim_idx = (uint8_t)(anim_idx + 1);
            anim_steps--;
        }

        SHOT_ANIM(*obj) = anim_idx;
        obj->raw[6] = (uint8_t)f->width;
        obj->raw[7] = (uint8_t)f->height;
        obj_sw(obj->raw + 8,  f->vect_num);
        obj_sw(obj->raw + 10, f->frame_num);

        if (shot_size >= 50) {
            obj->raw[14] = 8;
            obj->raw[15] = 8;
        } else {
            obj->raw[14] = bullet_pop_src_cols[shot_size];
            obj->raw[15] = bullet_pop_src_rows[shot_size];
        }

        SHOT_SET_ACCYPOS(*obj, acc);
        obj_sw(obj->raw + 4, (int16_t)(acc >> 7));
        return;
    }

    /* Check lifetime against gun data */
    if (shot_size >= 0 && shot_size < 8) {
        int16_t max_life = default_plr1_guns[shot_size].bullet_lifetime;
        /* Amiga ItsABullet times out only once shotlife exceeds max life
         * (cmp.w shotlife,maxlife ; bge notdone). */
        if (max_life >= 0 && life > max_life) {
            timed_out = true;
        }
    }
    /* Increment life for regular bullets.
     * Gibs (50-53) reuse SHOT_LIFE as an animation cadence accumulator. */
    if (shot_size < 50) {
        life += state->temp_frames;
        SHOT_SET_LIFE(*obj, life);
    }

    /* Advance bullet animation (Amiga ItsABullet notpopping path).
     * BulletTypes[shot_size].anim_ptr drives size/vect/frame each tick. */
    if (shot_status == 0 && shot_size >= 0 && shot_size < MAX_BULLET_ANIM_IDX && bullet_anim_tables[shot_size]) {
        int anim_steps = anim_ticks;
        if (shot_size >= 50) {
            int16_t accum = SHOT_LIFE(*obj);
            if (accum < 0) accum = 0;
            accum = (int16_t)(accum + anim_ticks);
            anim_steps = accum / EXPLOSION_FRAME_STEP_VBLANKS;
            accum = (int16_t)(accum % EXPLOSION_FRAME_STEP_VBLANKS);
            SHOT_SET_LIFE(*obj, accum);
        }

        uint8_t anim_idx = SHOT_ANIM(*obj);
        const BulletAnimFrame *f = &bullet_anim_tables[shot_size][anim_idx];
        /* Wrap at end-of-sequence sentinel (width == -1) */
        if (f->width < 0) {
            anim_idx = 0;
            f = &bullet_anim_tables[shot_size][0];
        }
        /* Write size to obj[6:7], vect to obj[8:9], frame to obj[10:11] */
        obj->raw[6] = (uint8_t)f->width;
        obj->raw[7] = (uint8_t)f->height;
        obj_sw(obj->raw + 8,  f->vect_num);
        obj_sw(obj->raw + 10, f->frame_num);
        /* Update src_cols/rows (obj[14:15]).
         * Gibs (50-53): each frame is 16×16 in the alien sheet. eff = src*2, so src=8 → eff=16. */
        if (shot_size >= 50) {
            obj->raw[14] = 8;
            obj->raw[15] = 8;
        } else {
            obj->raw[14] = bullet_fly_src_cols[(uint8_t)shot_size < MAX_BULLET_ANIM_IDX ? (uint8_t)shot_size : 0];
            obj->raw[15] = bullet_fly_src_rows[(uint8_t)shot_size < MAX_BULLET_ANIM_IDX ? (uint8_t)shot_size : 0];
        }
        /* ItsABullet notpopping: add anim y_offset to accypos (Anims.s; flame table uses 0). */
        {
            int32_t acc = SHOT_ACCYPOS(*obj) + ((int32_t)f->y_offset << 7);
            SHOT_SET_ACCYPOS(*obj, acc);
            obj_sw(obj->raw + 4, (int16_t)(acc >> 7));
        }
        /* Advance for next tick(s). Gibs use paced cadence to match Amiga feel. */
        while (anim_steps > 0) {
            anim_idx = (uint8_t)(anim_idx + 1);
            if (bullet_anim_tables[shot_size][anim_idx].width < 0)
                anim_idx = 0;
            anim_steps--;
        }
        SHOT_ANIM(*obj) = anim_idx;
    }

    /* Position is in object_points at OBJ_CID (works for both object_data and nasty_shot_data bullets). */
    int idx = (int)OBJ_CID(obj);
    if (idx < 0 || (state->level.object_points && idx >= state->level.num_object_points)) {
        OBJ_SET_ZONE(obj, -1);
        return;
    }
    uint8_t *bullet_pts = state->level.object_points + (uint32_t)idx * 8u;
    int32_t bx_fp = obj_l(bullet_pts);
    int32_t bz_fp = obj_l(bullet_pts + 4);
    int16_t bx = (int16_t)(bx_fp >> 16);
    int16_t bz = (int16_t)(bz_fp >> 16);
    int32_t accypos = SHOT_ACCYPOS(*obj);

    int bullet_zone_idx = -1;
    if (OBJ_ZONE(obj) >= 0 && state->level.zone_adds && state->level.data) {
        bullet_zone_idx = level_connect_to_zone_index(&state->level, OBJ_ZONE(obj));
        if (bullet_zone_idx < 0 && OBJ_ZONE(obj) < zone_slots)
            bullet_zone_idx = OBJ_ZONE(obj);
    }

    /* Floor/roof collision (from Anims.s ItsABullet lines 2882-3015) */
    if (state->level.zone_adds && state->level.data &&
        bullet_zone_idx >= 0 && bullet_zone_idx < zone_slots) {
        const uint8_t *za = state->level.zone_adds;
        int16_t zone = (int16_t)bullet_zone_idx;
        int32_t zone_off = (int32_t)((za[zone*4]<<24)|(za[zone*4+1]<<16)|
                           (za[zone*4+2]<<8)|za[zone*4+3]);
        const uint8_t *zd = state->level.data + zone_off;
        int zd_off = obj->obj.in_top ? 8 : 0;

        /* Roof check (Amiga: blt .nohitroof → hit when roof - accypos >= 10*128) */
        int32_t roof = (int32_t)((zd[6+zd_off]<<24)|(zd[7+zd_off]<<16)|
                       (zd[8+zd_off]<<8)|zd[9+zd_off]);
        if (roof - accypos >= 10 * 128) {
            if (flags & 1) {
                /* Bounce off roof */
                yvel = (int16_t)(-yvel);
                SHOT_SET_YVEL(*obj, yvel);
                accypos = roof + 10 * 128;
                SHOT_SET_ACCYPOS(*obj, accypos);
                if (flags & 2) {
                    xvel >>= 1;
                    zvel >>= 1;
                    SHOT_SET_XVEL(*obj, xvel);
                    SHOT_SET_ZVEL(*obj, zvel);
                }
            } else {
                timed_out = true; /* Impact on roof */
            }
        }

        /* Floor check (Amiga: bgt .nohitfloor → hit when floor - accypos <= 10*128) */
        int32_t floor_h = (int32_t)((zd[2+zd_off]<<24)|(zd[3+zd_off]<<16)|
                          (zd[4+zd_off]<<8)|zd[5+zd_off]);
        if (floor_h - accypos <= 10 * 128) {
            if (flags & 1) {
                /* Bounce off floor */
                if (yvel > 0) {
                    yvel = (int16_t)(-(yvel >> 1));
                    SHOT_SET_YVEL(*obj, yvel);
                    accypos = floor_h - 10 * 128;
                    SHOT_SET_ACCYPOS(*obj, accypos);
                    if (flags & 2) {
                        xvel >>= 1;
                        zvel >>= 1;
                        SHOT_SET_XVEL(*obj, xvel);
                        SHOT_SET_ZVEL(*obj, zvel);
                    }
                }
            } else {
                timed_out = true; /* Impact on floor */
            }
        }
    }
    int32_t new_bx_fp = bx_fp + xvel * state->temp_frames;
    int32_t new_bz_fp = bz_fp + zvel * state->temp_frames;
    int16_t new_bx = (int16_t)(new_bx_fp >> 16);
    int16_t new_bz = (int16_t)(new_bz_fp >> 16);

    /* Y integration order matches Amiga ItsABullet: oldy=accypos, then apply yvel and gravity. */
    int32_t old_accypos = accypos;
    int32_t y_delta = (int32_t)yvel * state->temp_frames;
    if (grav != 0) {
        int32_t grav_delta = (int32_t)grav * state->temp_frames;
        y_delta += grav_delta;
        int32_t new_yvel = (int32_t)yvel + grav_delta;
        if (new_yvel > 10 * 256) new_yvel = 10 * 256;
        if (new_yvel < -10 * 256) new_yvel = -10 * 256;
        yvel = (int16_t)new_yvel;
        SHOT_SET_YVEL(*obj, yvel);
    }
    accypos += y_delta;
    SHOT_SET_ACCYPOS(*obj, accypos);

    /* MoveObject for wall collision */
    MoveContext ctx;
    move_context_init(&ctx);
    ctx.oldx = bx;
    ctx.oldz = bz;
    ctx.newx = new_bx;
    ctx.newz = new_bz;
    ctx.oldy = old_accypos;
    ctx.newy = accypos - 5 * 128;
    ctx.thing_height = 10 * 128;
    ctx.step_up_val = 0;
    ctx.step_down_val = 0x1000000;
    ctx.extlen = 0;
    ctx.awayfromwall = -1;
    ctx.exitfirst = (flags & 1) ? 0 : 1;
    ctx.wallbounce = (int8_t)(flags & 1);
    ctx.stood_in_top = obj->obj.in_top;
    ctx.wall_flags = 0x0400;
    /* Set objroom from bullet's current zone so move_object uses zone-based collision and updates zone on transition */
    if (bullet_zone_idx >= 0 && state->level.zone_adds && state->level.data &&
        bullet_zone_idx < zone_slots) {
        int32_t zo = (int32_t)be32(state->level.zone_adds + (uint32_t)bullet_zone_idx * 4u);
        ctx.objroom = (uint8_t *)(state->level.data + zo);
    }

    if (new_bx != bx || new_bz != bz) {
        move_object_substepped(&ctx, &state->level);
    }

    int32_t final_bx_fp = new_bx_fp;
    int32_t final_bz_fp = new_bz_fp;
    if (ctx.newx != new_bx) final_bx_fp = ((int32_t)ctx.newx << 16);
    if (ctx.newz != new_bz) final_bz_fp = ((int32_t)ctx.newz << 16);

    obj->obj.in_top = ctx.stood_in_top;

    /* Wall bounce physics (Anims.s lines 3098-3140) */
    if (ctx.wallbounce && ctx.hitwall) {
        int16_t wall_len = ctx.wall_length;
        if (wall_len != 0) {
            int16_t vx = (int16_t)(xvel >> 16);
            int16_t vz = (int16_t)(zvel >> 16);
            int32_t d0 = (int32_t)vz * (int32_t)ctx.wall_xsize -
                         (int32_t)vx * (int32_t)ctx.wall_zsize;
            d0 /= wall_len;

            vx = (int16_t)(vx + (int16_t)(((int32_t)(ctx.wall_zsize * 2) * d0) / wall_len));
            vz = (int16_t)(vz - (int16_t)(((int32_t)(ctx.wall_xsize * 2) * d0) / wall_len));

            xvel = (int32_t)vx << 16;
            zvel = (int32_t)vz << 16;
            SHOT_SET_XVEL(*obj, xvel);
            SHOT_SET_ZVEL(*obj, zvel);
        } else {
            SHOT_SET_XVEL(*obj, -xvel);
            SHOT_SET_ZVEL(*obj, -zvel);
        }
        if (flags & 2) {
            /* Friction on bounce */
            SHOT_SET_XVEL(*obj, SHOT_XVEL(*obj) >> 1);
            SHOT_SET_ZVEL(*obj, SHOT_ZVEL(*obj) >> 1);
        }
    } else if (!ctx.wallbounce && ctx.hitwall) {
        timed_out = true; /* Non-bouncing bullet hit a wall */
    }

    /* Impact handling */
    if (timed_out) {
        SHOT_STATUS(*obj) = 1;
        SHOT_ANIM(*obj) = 0;
        SHOT_SET_LIFE(*obj, 0);

        /* Hit sound */
        if (shot_size >= 0 && shot_size < 8 &&
            bullet_types[shot_size].hit_noise >= 0) {
            audio_play_sample(bullet_types[shot_size].hit_noise,
                              bullet_types[shot_size].hit_volume);
        }
        /* Gibs: splatter sound when they hit floor/roof/wall (one per logic tick max). */
        if (shot_size >= 50 && !gib_impact_splat_sound_this_update) {
            gib_impact_splat_sound_this_update = true;
            audio_play_sample(13, 64);  /* splotch */
        }

        /* Explosive force splash + blast particles */
        if (shot_size >= 0 && shot_size < 8 &&
            bullet_types[shot_size].explosive_force > 0) {
            int16_t blast_zone = OBJ_ZONE(obj);
            int8_t blast_top = obj->obj.in_top;
            if (ctx.objroom && state->level.data) {
                int zi = level_zone_index_from_room_ptr(&state->level, ctx.objroom);
                if (zi < 0) {
                    int16_t room_zone_word = (int16_t)((ctx.objroom[0] << 8) | ctx.objroom[1]);
                    zi = level_connect_to_zone_index(&state->level, room_zone_word);
                }
                if (zi >= 0 && zi < zone_slots)
                    blast_zone = (int16_t)zi;
                blast_top = ctx.stood_in_top;
            }
            compute_blast(state, ctx.newx, ctx.newz, accypos,
                          bullet_types[shot_size].explosive_force,
                          blast_zone, blast_top);
        }

        /* Pop in place at impact position. */
        obj_sl(bullet_pts, (int32_t)ctx.newx << 16);
        obj_sl(bullet_pts + 4, (int32_t)ctx.newz << 16);
        obj_sw(obj->raw + 4, (int16_t)(accypos >> 7));
        return;
    }

    /* Update position in ObjectPoints */
    obj_sl(bullet_pts, final_bx_fp);
    obj_sl(bullet_pts + 4, final_bz_fp);

    /* Amiga ItsABullet: move.w (accypos>>7),4(a0). */
    obj_sw(obj->raw + 4, (int16_t)(accypos >> 7));

    /* Update zone from room (room pointer is authoritative). */
    if (ctx.objroom && state->level.data) {
        int new_zone = level_zone_index_from_room_ptr(&state->level, ctx.objroom);
        if (new_zone < 0) {
            int16_t room_zone_word = (int16_t)((ctx.objroom[0] << 8) | ctx.objroom[1]);
            new_zone = level_connect_to_zone_index(&state->level, room_zone_word);
        }
        if (new_zone >= 0 && new_zone < zone_slots)
            OBJ_SET_ZONE(obj, (int16_t)new_zone);
    }

    /* ---- Object-to-object hit detection (ObjectMove.s Collision) ---- */
    uint32_t enemy_flags = NASTY_EFLAGS(*obj);
    if (enemy_flags == 0) return;

    if (!state->level.object_data) return;

    int16_t xdiff = (int16_t)(ctx.newx - bx);
    int16_t zdiff = (int16_t)(ctx.newz - bz);
    int32_t diff_sq = (int32_t)xdiff * xdiff + (int32_t)zdiff * zdiff;
    int16_t range = 1;
    if (diff_sq > 0) {
        int32_t r = calc_dist_euclidean((int32_t)xdiff, (int32_t)zdiff);
        if (r < 1) r = 1;
        if (r > 32767) r = 32767;
        range = (int16_t)r;
    }
    int check_idx = 0;
    while (1) {
        GameObject *target = (GameObject*)(state->level.object_data +
                             check_idx * OBJECT_SIZE);
        if (OBJ_CID(target) < 0) break;

        if (OBJ_ZONE(target) < 0 || target == obj) {
            check_idx++;
            continue;
        }

        /* Check enemy flags bitmask */
        int tgt_type = target->obj.number;
        if (tgt_type < 0 || tgt_type > 20 || !(enemy_flags & (1u << tgt_type))) {
            check_idx++;
            continue;
        }

        /* Dead enemies (death animation or removed) do not receive shot hits */
        if (tgt_type == OBJ_NBR_DEAD) {
            check_idx++;
            continue;
        }
        /* Check lives (barrels with 0 lives can still be hit to trigger explosion) */
        if (NASTY_LIVES(*target) <= 0 && tgt_type != OBJ_NBR_BARREL) {
            check_idx++;
            continue;
        }

        /* Height check (Anims.s: abs(target_y - bullet_y) <= half_height). */
        const CollisionBox *box = &col_box_table[tgt_type];
        int32_t sqr_edge = (int32_t)range + 40;
        int64_t sqrnum = (int64_t)sqr_edge * (int64_t)sqr_edge;
        int16_t ydiff = (int16_t)(obj_w(obj->raw + 4) - obj_w(target->raw + 4));
        if (ydiff < 0) ydiff = (int16_t)(-ydiff);
        if (ydiff > box->half_height) {
            check_idx++;
            continue;
        }

        /* Position (Amiga: (a1,d0.w*8) = target point index) */
        int16_t tx, tz;
        get_object_pos(&state->level, OBJ_CID(target), &tx, &tz);

        int32_t old_dx = (int32_t)tx - (int32_t)bx;
        int32_t old_dz = (int32_t)tz - (int32_t)bz;
        int32_t new_dx = (int32_t)tx - (int32_t)ctx.newx;
        int32_t new_dz = (int32_t)tz - (int32_t)ctx.newz;

        int64_t cross = (int64_t)old_dx * (int64_t)zdiff - (int64_t)old_dz * (int64_t)xdiff;
        if (cross < 0) cross = -cross;
        int32_t perp_dist = (int32_t)(cross / range);
        if (perp_dist > box->width) {
            check_idx++;
            continue;
        }

        int64_t dist_old_sq = (int64_t)old_dx * old_dx + (int64_t)old_dz * old_dz;
        if (dist_old_sq > sqrnum) {
            check_idx++;
            continue;
        }
        int64_t dist_new_sq = (int64_t)new_dx * new_dx + (int64_t)new_dz * new_dz;
        if (dist_new_sq > sqrnum) {
            check_idx++;
            continue;
        }

        bool explosive_projectile_hit = (shot_size >= 0 && shot_size < 8 &&
                                         bullet_types[shot_size].explosive_force > 0);

        /* HIT! Apply damage */
        target->raw[19] = damage_accumulate_u8(target->raw[19], (int32_t)(uint8_t)SHOT_POWER(*obj));
        NASTY_SET_IMPACTX(target, SHOT_XVEL(*obj) >> 16);
        NASTY_SET_IMPACTZ(target, SHOT_ZVEL(*obj) >> 16);
        if (explosive_projectile_hit && check_idx >= 0 && check_idx < MAX_OBJECTS) {
            /* Direct rocket/grenade impacts should always count as explosion kills
             * even if the subsequent splash visibility test excludes the target. */
            explosion_damage_flag[check_idx] = 1;
        }

        /* Set bullet to popping */
        SHOT_STATUS(*obj) = 1;
        SHOT_ANIM(*obj) = 0;
        SHOT_SET_LIFE(*obj, 0);

        /* Hit sound + explosive splash */
        if (shot_size >= 0 && shot_size < 8 &&
            bullet_types[shot_size].hit_noise >= 0) {
            audio_play_sample(bullet_types[shot_size].hit_noise,
                              bullet_types[shot_size].hit_volume);
        }
        if (explosive_projectile_hit) {
            int16_t blast_zone = OBJ_ZONE(obj);
            int8_t blast_top = obj->obj.in_top;
            if (ctx.objroom && state->level.data) {
                int zi = level_zone_index_from_room_ptr(&state->level, ctx.objroom);
                if (zi < 0) {
                    int16_t room_zone_word = (int16_t)((ctx.objroom[0] << 8) | ctx.objroom[1]);
                    zi = level_connect_to_zone_index(&state->level, room_zone_word);
                }
                if (zi >= 0 && zi < zone_slots)
                    blast_zone = (int16_t)zi;
                blast_top = ctx.stood_in_top;
            }
            compute_blast(state, ctx.newx, ctx.newz, accypos,
                          bullet_types[shot_size].explosive_force,
                          blast_zone, blast_top);
        }
        return;
    }
}

/* Door/lift wall list: 10 bytes per entry:
 *   +0 fline (be16), +2 ptr_to_wall_rec (be32), +6 gfx_base (be32). */
#define DOOR_WALL_ENT_SIZE   10
#define DOOR_FLINE_SIZE      16
#define DOOR_FLINE_X         0
#define DOOR_FLINE_Z         2
#define DOOR_FLINE_XLEN      4
#define DOOR_FLINE_ZLEN      6
#define DOOR_NEAR_THRESH     24  /* forgiving door press distance from the controlling floor line */
#define LIFT_NEAR_THRESH     32  /* give lifts a little extra interaction room near their trigger lines */

static bool player_near_floor_line(GameState *state, int16_t fline_idx, int plr_num, int16_t near_thresh)
{
    if (!state || !state->level.floor_lines || near_thresh < 0) return false;
    if (fline_idx < 0 || (int32_t)fline_idx >= state->level.num_floor_lines) return false;

    const PlayerState *plr = (plr_num == 0) ? &state->plr1 : &state->plr2;
    int32_t px = plr->p_xoff;
    int32_t pz = plr->p_zoff;

    const uint8_t *fl = state->level.floor_lines + (uint32_t)(int16_t)fline_idx * DOOR_FLINE_SIZE;
    int32_t ax = (int32_t)be16(fl + DOOR_FLINE_X);
    int32_t az = (int32_t)be16(fl + DOOR_FLINE_Z);
    int32_t bx = ax + (int32_t)be16(fl + DOOR_FLINE_XLEN);
    int32_t bz = az + (int32_t)be16(fl + DOOR_FLINE_ZLEN);

    int64_t vx = (int64_t)bx - (int64_t)ax;
    int64_t vz = (int64_t)bz - (int64_t)az;
    int64_t wx = (int64_t)px - (int64_t)ax;
    int64_t wz = (int64_t)pz - (int64_t)az;
    int64_t max_dist_sq = (int64_t)near_thresh * (int64_t)near_thresh;
    int64_t line_len_sq = vx * vx + vz * vz;

    if (line_len_sq <= 0) {
        int64_t point_dist_sq = wx * wx + wz * wz;
        return point_dist_sq <= max_dist_sq;
    }

    int64_t dot = wx * vx + wz * vz;
    if (dot <= 0) {
        int64_t start_dist_sq = wx * wx + wz * wz;
        return start_dist_sq <= max_dist_sq;
    }
    if (dot >= line_len_sq) {
        int64_t ex = (int64_t)px - (int64_t)bx;
        int64_t ez = (int64_t)pz - (int64_t)bz;
        int64_t end_dist_sq = ex * ex + ez * ez;
        return end_dist_sq <= max_dist_sq;
    }

    /* For projections that land on the segment: compare perpendicular distance
     * without dividing to keep this fast and stable in integer math. */
    int64_t cross = wx * vz - wz * vx;
    if (cross < 0) cross = -cross;
    return (cross * cross) <= (max_dist_sq * line_len_sq);
}

/* Amiga DoorRoutine consumes floorline word +14 wall-touch bits set by MoveObject:
 * PLR1 uses 0x0100, PLR2 uses 0x0800.  We mirror that behavior here. */
static bool player_at_door_zone(GameState *state, int16_t door_zone_id, int16_t player_zone, int door_idx, int plr_num)
{
    (void)door_zone_id;
    if (player_zone < 0) return false;

    if (!state->level.door_wall_list || !state->level.door_wall_list_offsets || !state->level.floor_lines) return false;
    if (door_idx < 0 || door_idx >= state->level.num_doors) return false;
    uint16_t player_touch_flag = (plr_num == 0) ? 0x0100u : 0x0800u;

    uint32_t start = state->level.door_wall_list_offsets[door_idx];
    uint32_t end   = state->level.door_wall_list_offsets[door_idx + 1];
    for (uint32_t j = start; j < end; j++) {
        const uint8_t *ent = state->level.door_wall_list + j * DOOR_WALL_ENT_SIZE;
        int16_t fi = be16(ent);
        if (fi < 0 || (int32_t)fi >= state->level.num_floor_lines) continue;
        const uint8_t *fl = state->level.floor_lines + (uint32_t)(int16_t)fi * DOOR_FLINE_SIZE;
        uint16_t touched = (uint16_t)be16(fl + 14);
        if ((touched & player_touch_flag) != 0u) return true;
        if (player_near_floor_line(state, fi, plr_num, DOOR_NEAR_THRESH)) return true;
    }
    return false;
}

/* Amiga LiftRoutine sets PLR*_stoodonlift from zone equality only:
 *   cmp.w (PLR*_Roompt),lift_zone ; seq PLR*_stoodonlift
 * There is no geometric "near lift wall" test in this step. */
static bool player_at_lift_zone(GameState *state, int16_t lift_zone_id, int16_t player_zone, int lift_idx, int plr_num)
{
    if ((player_zone >= 0) && (player_zone == lift_zone_id)) return true;

    if (!state || !state->level.lift_wall_list || !state->level.lift_wall_list_offsets ||
        !state->level.floor_lines)
        return false;
    if (lift_idx < 0 || lift_idx >= state->level.num_lifts) return false;

    uint32_t start = state->level.lift_wall_list_offsets[lift_idx];
    uint32_t end   = state->level.lift_wall_list_offsets[lift_idx + 1];
    for (uint32_t j = start; j < end; j++) {
        const uint8_t *ent = state->level.lift_wall_list + j * DOOR_WALL_ENT_SIZE;
        int16_t fi = be16(ent);
        if (player_near_floor_line(state, fi, plr_num, LIFT_NEAR_THRESH)) return true;
    }
    return false;
}

/* -----------------------------------------------------------------------
 * Door routine
 *
 * Translated from Anims.s DoorRoutine (line ~642-796).
 *
 * Door data format (per door, 16 bytes in DoorData array):
 *   0: zone index (word)
 *   2: door type (word) - 0=player+space, 1=condition, 2=condition2, etc.
 *   4: door position (long) - current Y (*256), 10: top (open), 14: bot (closed). 22 bytes per entry.
 *   8: door velocity (word) - current speed
 *  10: door max (word) - maximum opening height
 *  12: timer (word) - close delay
 *  14: flags (word) - for type 0: 0 = player-only (space at door); else condition mask (switch bit)
 * ----------------------------------------------------------------------- */
void door_routine(GameState *state)
{
    if (!state->level.door_data) return;
    const int zone_slots = level_zone_slot_count(&state->level);

    uint8_t *door = state->level.door_data;
    int door_idx = 0;

    /* Iterate door entries (22 bytes each, terminated by -1). Same layout as lift: pos/top/bot (world *256). */
    while (1) {
        int16_t zone_id = be16(door);
        if (zone_id < 0) break;

        int16_t door_type = be16(door + 2); /* high byte=open mode, low byte=close mode (Amiga bytes 16/17) */
        int32_t door_pos = be32(door + 4);
        int16_t door_vel = be16(door + 8);
        int32_t door_top = be32(door + 10);  /* open position (more negative) */
        int32_t door_bot = be32(door + 14);  /* closed position (more positive) */
        int16_t timer = be16(door + 18);
        uint16_t door_flags = (uint16_t)be16(door + 20);
        uint8_t door_open_mode = (uint8_t)((uint16_t)door_type >> 8);
        uint8_t door_close_mode = (uint8_t)((uint16_t)door_type & 0xFFu);

        /* Amiga state flow:
         * 1) Advance with current velocity.
         * 2) Clamp at top/bottom and zero velocity at limits.
         * 3) Only at limits, evaluate trigger mask and optionally set new velocity.
         * This avoids mid-travel flip-flopping that causes visible stutter. */
        int16_t prev_vel = door_vel;
        door_pos += (int32_t)door_vel * state->temp_frames * 64;

        bool door_closed = false;
        bool door_open = false;
        if (door_pos >= door_bot) {
            door_closed = true;
            if (door_pos > door_bot) door_pos = door_bot;
            if (door_vel > 0) door_vel = 0;
        }
        if (door_pos <= door_top) {
            door_open = true;
            if (door_pos < door_top) door_pos = door_top;
            if (door_vel < 0) door_vel = 0;
        }

        int16_t trigger_vel = door_vel;
        uint16_t trigger_mask = 0;
        bool clear_touch_flags = false; /* Amiga simplecheck writes 0 to floorline+14 when conditions not met. */
        bool plr1_door_press_near = false;
        bool plr2_door_press_near = false;

        /* Amiga NotGoBackUp (Anims.s DoorRoutine ~707–718): if player 1 is in the door zone,
         * door not fully open, and velocity >= 0, jump to backfromtst and skip the Conditions
         * check. That incorrectly opens switch-gated doors (e.g. level 2 exit needing 6 switches)
         * as soon as you enter the zone. Only take this shortcut when door_flags==0 (no mask). */
        if (zone_id == state->plr1.zone && !door_open && door_vel >= 0 && door_flags == 0) {
            trigger_vel = -16;
            trigger_mask = (uint16_t)0x8000;
        } else {
            bool conditions_met = (((uint16_t)game_conditions & door_flags) == door_flags);
            if (!conditions_met) {
                clear_touch_flags = true;
            } else if (door_open) {
                if (door_close_mode == 0) {
                    trigger_vel = 4;
                    trigger_mask = (uint16_t)0x8000;
                }
            } else if (door_closed) {
                trigger_vel = -16;
                switch (door_open_mode) {
                    case 0: {
                        uint16_t m = 0;
                        if (state->plr1.p_spctap) {
                            if (player_at_door_zone(state, zone_id, state->plr1.zone, door_idx, 0)) {
                                plr1_door_press_near = true;
                                m = (uint16_t)0x8000;
                            } else {
                                m |= (uint16_t)0x0100;
                            }
                        }
                        if (m != (uint16_t)0x8000 && state->plr2.p_spctap) {
                            if (player_at_door_zone(state, zone_id, state->plr2.zone, door_idx, 1)) {
                                plr2_door_press_near = true;
                                m = (uint16_t)0x8000;
                            } else {
                                m |= (uint16_t)0x0800;
                            }
                        }
                        trigger_mask = m;
                        break;
                    }
                    case 1: trigger_mask = (uint16_t)0x0900; break;
                    case 2: trigger_mask = (uint16_t)0x0400; break;
                    case 3: trigger_mask = (uint16_t)0x0200; break;
                    case 4: trigger_mask = (uint16_t)0x8000; break;
                    case 5: trigger_mask = 0; break;
                    default: trigger_mask = 0; break;
                }
            }
        }

        /* Write back (big-endian) */
        wbe32(door + 4, door_pos);
        /* door_vel may be updated below from trigger bits. */
        wbe16(door + 18, timer);

        /* Update zone data: write door position directly to zone roof (same as Amiga). */
        if (zone_id >= 0 && zone_id < zone_slots)
            level_set_zone_roof(&state->level, zone_id, door_pos);

        /* Amiga-style: patch floor line 14 and graphics wall record for each door wall (when data was loaded). */
        if (state->level.door_wall_list && state->level.door_wall_list_offsets &&
            state->level.graphics && door_idx < state->level.num_doors) {
            bool triggered = false;
            uint32_t start = state->level.door_wall_list_offsets[door_idx];
            uint32_t end   = state->level.door_wall_list_offsets[door_idx + 1];
            for (uint32_t j = start; j < end; j++) {
                const uint8_t *ent = state->level.door_wall_list + j * DOOR_WALL_ENT_SIZE;
                int16_t fline = be16(ent);
                int32_t gfx_off = (int32_t)be32(ent + 2);
                int32_t gfx_base = (int32_t)be32(ent + 6);
                if (state->level.floor_lines && fline >= 0 && (int32_t)fline < state->level.num_floor_lines) {
                    uint8_t *fl = state->level.floor_lines + (uint32_t)(int16_t)fline * 16u;
                    uint16_t old_flags = (uint16_t)be16(fl + 14);
                    if (trigger_mask != 0 && (old_flags & trigger_mask) != 0)
                        triggered = true;
                    wbe16(fl + 14, clear_touch_flags ? (int16_t)0 : (int16_t)(uint16_t)0x8000);
                }
                if (gfx_off >= 0) {
                    uint8_t *wall_rec = state->level.graphics + (uint32_t)gfx_off;
                    wbe32(wall_rec + 24, door_pos);   /* Amiga: move.l d3,24(a1) = door height for this wall */
                    /* Amiga DoorRoutine:
                     *   d0 = -(curr >> 2) & 255  => with curr stored as door_pos/64, this is -(door_pos>>8).
                     *   a2 = gfx_base + d0 ; move.l a2,10(a1)  (writes fromtile+totalyoff together). */
                    /* Align scroll relative to this door's closed position.
                     * Doors whose closed Y is not 0 need this per-door phase offset. */
                    int16_t door_phase = (int16_t)(door_pos >> 8);
                    int16_t door_closed_phase = (int16_t)(door_bot >> 8);
                    int16_t rel_phase = (int16_t)(door_phase - door_closed_phase);
                    uint32_t tex_scroll = (uint32_t)((-(int16_t)rel_phase) & 0x00FF);
                    uint32_t tex_ptr = (uint32_t)((int32_t)gfx_base + (int32_t)tex_scroll); /* Amiga: adda.w d0,a2 */
                    wbe32(wall_rec + 10, (int32_t)tex_ptr);  /* Amiga: move.l a2,10(a1) */
                }
            }
            if (!triggered && (plr1_door_press_near || plr2_door_press_near))
                triggered = true;
            if (triggered)
                door_vel = trigger_vel;
        }

        wbe16(door + 8, door_vel);
        if (door_vel != 0 && prev_vel == 0)
            audio_play_sample(5, 64);  /* newdoor: play when door starts opening or closing */

        door_idx++;
        door += 22;
    }
}

/* -----------------------------------------------------------------------
 * Lift routine
 *
 * Translated from Anims.s LiftRoutine (line ~377-627).
 *
 * Lift entry layout (20 bytes): zone(w), type(w), pos(l), vel(w), top(l), bot(l), flags(w).
 *   type: high byte = behaviour when at top (0=space/on lift, 1=player on lift, 2=auto, 3=no move),
 *         low byte = behaviour when at bottom (same).
 *   flags: 0 = always active; else same as door (satisfied when (game_conditions & flags) == flags).
 * ----------------------------------------------------------------------- */
void lift_routine(GameState *state)
{
    if (!state->level.lift_data) return;
    const int zone_slots = level_zone_slot_count(&state->level);

    uint8_t *lift = state->level.lift_data;
    int lift_idx = 0;
    /* Per-zone max floor (×64 scale to match doors). Lifts store pos in ×256; we convert when writing to zone/wall. */
    //int32_t zone_max_floor[256];
    //uint8_t zone_lift_seen[256];
    //memset(zone_lift_seen, 0, sizeof(zone_lift_seen));

    /* Iterate lift entries (terminated by -1) */
    while (1) {
        int16_t zone_id = be16(lift);
        if (zone_id < 0) break;

        int16_t lift_type = be16(lift + 2);
        int32_t lift_pos = be32(lift + 4);
        int16_t lift_vel = be16(lift + 8);
        int16_t prev_lift_vel = lift_vel;
        int32_t lift_top = be32(lift + 10);
        int32_t lift_bot = be32(lift + 14);
        uint16_t lift_flags = (uint16_t)be16(lift + 18);

        int plr1_at = player_at_lift_zone(state, zone_id, state->plr1.zone, lift_idx, 0);
        int plr2_at = player_at_lift_zone(state, zone_id, state->plr2.zone, lift_idx, 1);
        int plr1_on = state->plr1.stood_on_lift && plr1_at;
        int plr2_on = state->plr2.stood_on_lift && plr2_at;

        int32_t old_pos = lift_pos;
        lift_pos += (int32_t)lift_vel * state->temp_frames * 64;

        bool at_top = false;
        bool at_bot = false;
        if (lift_pos >= lift_bot) {
            at_bot = true;
            if (lift_pos > lift_bot) lift_pos = lift_bot;
            if (lift_vel > 0) lift_vel = 0;
        }
        if (lift_pos <= lift_top) {
            at_top = true;
            if (lift_pos < lift_top) lift_pos = lift_top;
            if (lift_vel < 0) lift_vel = 0;
        }

        /* Amiga: mode word bytes at +16/+17 in source data.
         * At top: use low byte (d5) to decide lowering behavior.
         * At bottom: use high byte (d4) to decide raising behavior. */
        uint8_t mode_raise_at_bottom = (uint8_t)((uint16_t)lift_type >> 8);
        uint8_t mode_lower_at_top    = (uint8_t)((uint16_t)lift_type & 0xFFu);

        int16_t trigger_vel = lift_vel;
        uint16_t trigger_mask = 0;
        bool clear_touch_flags = false;

        /* Conditions gate all behavior; when not satisfied Amiga simplecheck clears 14(a4). */
        bool conditions_met = (((uint16_t)game_conditions & lift_flags) == lift_flags);
        if (!conditions_met) {
            clear_touch_flags = true;
        } else if (at_top) {
            switch (mode_lower_at_top) {
                case 0: {
                    trigger_vel = 4;
                    uint16_t m = 0;
                    if (state->plr1.p_spctap) {
                        m |= (uint16_t)0x0100;
                        if (plr1_at) m = (uint16_t)0x8000;
                    }
                    if (m != (uint16_t)0x8000 && state->plr2.p_spctap) {
                        m |= (uint16_t)0x0800;
                        if (plr2_at) m = (uint16_t)0x8000;
                    }
                    trigger_mask = m;
                    break;
                }
                case 1:
                    trigger_vel = 4;
                    trigger_mask = (plr1_at || plr2_at) ? (uint16_t)0x8000 : (uint16_t)0x0900;
                    break;
                case 2:
                    trigger_vel = 4;
                    trigger_mask = (uint16_t)0x8000;
                    break;
                case 3:
                default:
                    trigger_mask = 0;
                    break;
            }
        } else if (at_bot) {
            switch (mode_raise_at_bottom) {
                case 0: {
                    trigger_vel = -4;
                    uint16_t m = 0;
                    if (state->plr1.p_spctap) {
                        m |= (uint16_t)0x0100;
                        if (plr1_at) m = (uint16_t)0x8000;
                    }
                    if (m != (uint16_t)0x8000 && state->plr2.p_spctap) {
                        m |= (uint16_t)0x0800;
                        if (plr2_at) m = (uint16_t)0x8000;
                    }
                    trigger_mask = m;
                    break;
                }
                case 1:
                    trigger_vel = -4;
                    trigger_mask = (plr1_at || plr2_at) ? (uint16_t)0x8000 : (uint16_t)0x0900;
                    break;
                case 2:
                    trigger_vel = -4;
                    trigger_mask = (uint16_t)0x8000;
                    break;
                case 3:
                default:
                    trigger_mask = 0;
                    break;
            }
        }

        int32_t lift_delta = lift_pos - old_pos;

        wbe32(lift + 4, lift_pos);

        if (zone_id >= 0 && zone_id < zone_slots)
        {
            level_set_zone_floor(&state->level, (int16_t)zone_id, lift_pos);
        }

        /* Amiga: move players with the lift.
         * lift_pos and s_yoff share the same coordinate system (both come from/write to zone
         * data in the same scale), so adding lift_delta keeps the player glued to the platform. */
        if (lift_delta != 0) {
            if (plr1_on) {
                state->plr1.s_yoff  += lift_delta;
                state->plr1.s_tyoff += lift_delta;
                state->plr1.s_yvel   = 0;
            }
            if (plr2_on) {
                state->plr2.s_yoff  += lift_delta;
                state->plr2.s_tyoff += lift_delta;
                state->plr2.s_yvel   = 0;
            }
        }

        /* Amiga-style: patch floor line 14 and graphics wall record. */
        if (state->level.lift_wall_list && state->level.lift_wall_list_offsets &&
            state->level.graphics && lift_idx < state->level.num_lifts) {
            bool triggered = false;
            uint32_t start = state->level.lift_wall_list_offsets[lift_idx];
            uint32_t end   = state->level.lift_wall_list_offsets[lift_idx + 1];
            for (uint32_t j = start; j < end; j++) {
                const uint8_t *ent = state->level.lift_wall_list + j * DOOR_WALL_ENT_SIZE;
                int16_t fline = be16(ent);
                int32_t gfx_off = (int32_t)be32(ent + 2);
                int32_t gfx_base = (int32_t)be32(ent + 6);
                if (state->level.floor_lines && fline >= 0 && (int32_t)fline < state->level.num_floor_lines) {
                    uint8_t *fl = state->level.floor_lines + (uint32_t)(int16_t)fline * 16u;
                    uint16_t old_flags = (uint16_t)be16(fl + 14);
                    if (trigger_mask != 0 && (old_flags & trigger_mask) != 0u)
                        triggered = true;
                    wbe16(fl + 14, clear_touch_flags ? (int16_t)0 : (int16_t)(uint16_t)0x8000);
                }
                if (gfx_off >= 0) {
                    uint8_t *wall_rec = state->level.graphics + (uint32_t)gfx_off;
                    /* Amiga LiftRoutine does move.l a2,10(a1) (packed ptr into +10/+12), but this port's
                     * wall drawer reads +10 as totalyoff (word) and +12 as tex_id (word). Writing a
                     * full long here clobbers tex_id and breaks lift wall texture/scrolling (regression
                     * from "Water Rendering Fixes"). Scroll/clipping is driven via wall_rec+20 and the
                     * renderer lift-zone path (door_yoff_add). Do not patch wall_rec+10 for lifts. */
                    (void)gfx_base;
                    wbe32(wall_rec + 20, lift_pos);
                }
            }
            if (triggered)
                lift_vel = trigger_vel;
        }

        wbe16(lift + 8, lift_vel);
        if (lift_vel != 0 && prev_lift_vel == 0)
            audio_play_sample(5, 64);  /* newdoor: play when lift starts moving (same as door) */

        lift_idx++;
        lift += LIFT_ENTRY_SIZE;
    }
}


/* -----------------------------------------------------------------------
 * Switch routine
 *
 * Translated from Anims.s SwitchRoutine (line ~868-1034).
 * ----------------------------------------------------------------------- */
void switch_routine(GameState *state)
{
    if (!state->level.switch_data) return;

    /* Real-time auto-switch pacing:
     * Convert elapsed wall-clock ms into virtual 50Hz ticks so auto-reset
     * timing is stable even when render/frame pacing changes. */
    static const uint8_t *s_last_switch_data = NULL;
    static uint32_t s_last_switch_tick_ms = 0;
    static uint32_t s_switch_vblank_remainder_ms = 0;
    uint32_t now_ms = state->current_ticks_ms;
    if (state->level.switch_data != s_last_switch_data) {
        s_last_switch_data = state->level.switch_data;
        s_last_switch_tick_ms = now_ms;
        s_switch_vblank_remainder_ms = 0;
    }
    uint32_t elapsed_ms = now_ms - s_last_switch_tick_ms;
    s_last_switch_tick_ms = now_ms;
    if (elapsed_ms > 200u) elapsed_ms = 200u;
    s_switch_vblank_remainder_ms += elapsed_ms;
    int auto_vblanks = (int)(s_switch_vblank_remainder_ms / 20u);
    s_switch_vblank_remainder_ms %= 20u;
    /* Keep prior gameplay tweak: 50% slower auto-reset than original (2 units per 50Hz tick). */
    int8_t auto_dec = (int8_t)(auto_vblanks * 2);

    /* Make switch pressing a little less strict than original 60-unit radius. */
    const int32_t switch_dist_sq = 80 * 80;
    uint8_t *sw = state->level.switch_data;
    int switch_index = 0;

    while (1) {
        int16_t zone_id = be16(sw);
        if (zone_id < 0) break;

        /* Amiga: condition bit from switch index (d0=7..0, bit = 4 + (7-d0)).
         * With forward index this is simply bit 4 + index. */
        unsigned int bit_num = 4 + (switch_index % 8);
        uint16_t bit_mask = (uint16_t)(1u << bit_num);
        int32_t gfx_off = (int32_t)be32(sw + 6);

        /* Auto-reset branch from Amiga backtoend/nobutt:
         * original logic uses temp_frames*4; here we drive the same byte countdown
         * from wall-clock-based virtual 50Hz ticks for uncapped-frame stability. */
        if ((int8_t)sw[2] != 0 && (int8_t)sw[10] != 0) {
            if (auto_dec != 0) {
                sw[3] = (uint8_t)((int8_t)sw[3] - auto_dec);
                if ((int8_t)sw[3] == 0) {
                    sw[10] = 0;
                    if (state->level.graphics && gfx_off >= 0) {
                        uint8_t *wall_ptr = state->level.graphics + (uint32_t)gfx_off;
                        write_be16(wall_ptr + 4, 11);
                        int16_t w = be16(wall_ptr);
                        w = (int16_t)(w & 0x007C);
                        write_be16(wall_ptr, w);
                    }
                    game_conditions = (int16_t)((uint16_t)game_conditions & (uint16_t)~bit_mask);
                    audio_play_sample(10, 50);
                }
            }
        }

        /* p1/p2 SpaceIsPressed path. Amiga uses switch point index (word at +4)
         * and checks distance to midpoint of two consecutive points:
         *  cx=(x0+x1)/2, cz=(z0+z1)/2, dist^2 < 60^2.
         * No facing test and no explicit zone match in original code. */
        {
            int16_t pidx = be16(sw + 4);
            bool near_plr1 = false;
            bool near_plr2 = false;

            if (state->level.points && pidx >= 0) {
                const uint8_t *p0 = state->level.points + (uint32_t)(uint16_t)pidx * 4u;
                int16_t x0 = be16(p0 + 0);
                int16_t z0 = be16(p0 + 2);
                int16_t x1 = be16(p0 + 4);
                int16_t z1 = be16(p0 + 6);
                int32_t cx = ((int32_t)x0 + (int32_t)x1) >> 1;
                int32_t cz = ((int32_t)z0 + (int32_t)z1) >> 1;

                if (state->plr1.p_spctap) {
                    int32_t dx = cx - state->plr1.p_xoff;
                    int32_t dz = cz - state->plr1.p_zoff;
                    near_plr1 = (dx * dx + dz * dz) < switch_dist_sq;
                }
                if (state->plr2.p_spctap) {
                    int32_t dx = cx - state->plr2.p_xoff;
                    int32_t dz = cz - state->plr2.p_zoff;
                    near_plr2 = (dx * dx + dz * dz) < switch_dist_sq;
                }
            }

            if (near_plr1 || near_plr2) {
                sw[10] = (uint8_t)(~sw[10]); /* not.b 10(a0) */
                if (state->level.graphics && gfx_off >= 0) {
                    uint8_t *wall_ptr = state->level.graphics + (uint32_t)gfx_off;
                    write_be16(wall_ptr + 4, 11);
                    int16_t w = be16(wall_ptr);
                    w = (int16_t)(w & 0x007C);
                    if ((int8_t)sw[10] != 0) w = (int16_t)(w | 2);
                    write_be16(wall_ptr, w);
                }
                game_conditions ^= bit_mask;
                sw[3] = 0;  /* move.b #0,3(a0) */
                audio_play_sample(10, 50);
            }
        }

        /* Keep wall state synced to current on/off and condition bits.
         * This helps after save/load and mirrors what the Amiga writes during toggles. */
        if (state->level.graphics && gfx_off >= 0) {
            uint8_t *wall_ptr = state->level.graphics + gfx_off;
            write_be16(wall_ptr + 4, 11);
            int16_t w = be16(wall_ptr);
            w = (int16_t)(w & 0x007C);
            if ((int8_t)sw[10] != 0) w = (int16_t)(w | 2);
            write_be16(wall_ptr, w);
        }

        sw += 14;
        switch_index++;
    }
}

/* -----------------------------------------------------------------------
 * Water animations
 *
 * Translated from Anims.s DoWaterAnims (line ~322-373).
 * ----------------------------------------------------------------------- */
void do_water_anims(GameState *state)
{
    /* Raw Amiga WaterList lives after LiftData's 999 terminator.
     * Format per anim slot (Anims.s DoWaterAnims):
     *   long top, long bot, long current, word velocity,
     *   then repeating pairs: word zone_id, long water_poly_gfx_off, ... , word -1.
     * There are 21 slots (d0 starts at 20, dbra loop). */
    if (!state->level.water_list) return;
    const int zone_slots = level_zone_slot_count(&state->level);

    uint8_t *wl = state->level.water_list;
    for (int slot = 0; slot <= 20; slot++) {
        int32_t top_level = be32(wl + 0);
        int32_t bot_level = be32(wl + 4);
        int32_t cur_level = be32(wl + 8);
        int16_t vel = be16(wl + 12);

        cur_level += (int32_t)vel * state->temp_frames;
        if (cur_level <= top_level) {
            cur_level = top_level;
            vel = 128;
        }
        if (cur_level >= bot_level) {
            cur_level = bot_level;
            vel = -128;
        }

        wbe32(wl + 8, cur_level);
        wbe16(wl + 12, vel);

        wl += 14;

        int safety = 128;
        while (safety-- > 0) {
            int16_t zone_id = be16(wl);
            wl += 2;
            if (zone_id < 0) break;

            int32_t gfx_off = be32(wl);
            wl += 4;

            /* Amiga writes water ypos word at +2 in the floor entry pointed by graphics offset. */
            if (state->level.graphics && gfx_off >= 0) {
                uint8_t *poly = state->level.graphics + (uint32_t)gfx_off;
                wbe16(poly + 2, (int16_t)(cur_level >> 6));
            }

            if (zone_id >= 0 && zone_id < zone_slots)
                level_set_zone_water(&state->level, zone_id, cur_level);
        }
    }
}

#define BRIGHT_ANIM_SENTINEL 999
/* Advance anim tables every logic tick (50 Hz), matching Amiga vblank. */
#define BRIGHT_ANIM_TICK_DIVIDER 1

/* Advance one brightness animation; returns current value, updates index. */
static int16_t bright_anim_advance(const int16_t *table, unsigned int *idx)
{
    int16_t val = table[*idx];
    if (val == BRIGHT_ANIM_SENTINEL) {
        *idx = 0;
        val = table[0];
        (*idx)++;
    } else {
        (*idx)++;
    }
    return val;
}

/* -----------------------------------------------------------------------
 * Brightness animation handler
 *
 * Advances the three global anims (pulse, flicker, fire) each tick.
 * Render time reads these via level_get_zone_brightness() and level_get_point_brightness().
 * Anim selection is encoded in zone words and pointBrights entries (Amiga format).
 * ----------------------------------------------------------------------- */
void bright_anim_handler(GameState *state)
{
    LevelState *lev = &state->level;
    static unsigned int bright_anim_tick_count = 0;

    bright_anim_tick_count++;
    if (bright_anim_tick_count % BRIGHT_ANIM_TICK_DIVIDER == 0) {
        lev->bright_anim_values[0] = bright_anim_advance(pulse_anim, &lev->bright_anim_indices[0]);
        lev->bright_anim_values[1] = bright_anim_advance(flicker_anim, &lev->bright_anim_indices[1]);
        lev->bright_anim_values[2] = bright_anim_advance(fire_flicker_anim, &lev->bright_anim_indices[2]);
    }
}

/* -----------------------------------------------------------------------
 * Utility: fire a projectile from an enemy
 *
 * Translated from the common FireAtPlayer pattern in enemy .s files.
 * Creates a bullet in NastyShotData.
 * ----------------------------------------------------------------------- */
void enemy_fire_at_player(GameObject *obj, GameState *state,
                          int player_num, int shot_type, int shot_power,
                          int shot_speed, int shot_shift)
{
    if (!state->level.nasty_shot_data) return;

    PlayerState *plr = (player_num == 1) ? &state->plr1 : &state->plr2;

    /* Find free slot in NastyShotData */
    uint8_t *shots = state->level.nasty_shot_data;
    GameObject *bullet = NULL;
    for (int i = 0; i < NASTY_SHOT_SLOT_COUNT; i++) {
        GameObject *candidate = (GameObject*)(shots + i * OBJECT_SIZE);
        if (OBJ_ZONE(candidate) < 0) {
            bullet = candidate;
            break;
        }
    }
    if (!bullet) return;

    /* Calculate direction to player (AlienControl.s FireAtPlayer1 lines 360-411) */
    int idx = (int)(((uint8_t*)obj - state->level.object_data) / OBJECT_SIZE);
    int16_t obj_x, obj_z;
    get_object_pos(&state->level, idx, &obj_x, &obj_z);

    int32_t plr_x = plr->p_xoff;
    int32_t plr_z = plr->p_zoff;

    /* Lead prediction: offset target by player velocity * (dist/speed) */
    int32_t dx = plr_x - obj_x;
    int32_t dz = plr_z - obj_z;
    int32_t dist = calc_dist_approx(dx, dz);
    if (dist == 0) dist = 1;

    /* Apply lead if speed > 0 */
    int16_t target_xdiff = (player_num == 1) ? state->xdiff1 : state->xdiff2;
    int16_t target_zdiff = (player_num == 1) ? state->zdiff1 : state->zdiff2;
    if (shot_speed > 0 && (target_xdiff != 0 || target_zdiff != 0)) {
        int16_t lead_x = (int16_t)((target_xdiff * dist) / (shot_speed * 16));
        int16_t lead_z = (int16_t)((target_zdiff * dist) / (shot_speed * 16));
        plr_x += lead_x;
        plr_z += lead_z;
    }

    /* Set up bullet (preserve slot CID, which indexes ObjectPoints). */
    int16_t saved_cid = OBJ_CID(bullet);
    memset(bullet, 0, OBJECT_SIZE);
    OBJ_SET_CID(bullet, saved_cid);
    OBJ_SET_ZONE(bullet, OBJ_ZONE(obj));
    bullet->obj.number = OBJ_NBR_BULLET;

    /* Use HeadTowards to calculate velocity toward (potentially led) target */
    MoveContext hctx;
    move_context_init(&hctx);
    hctx.oldx = obj_x;
    hctx.oldz = obj_z;
    head_towards(&hctx, (int32_t)plr_x, (int32_t)plr_z, (int16_t)shot_speed);

    int32_t xvel = hctx.newx - obj_x;
    int32_t zvel = hctx.newz - obj_z;

    /* Copy bullet position to ObjectPoints */
    int bul_idx = (int)OBJ_CID(bullet);
    if (state->level.object_points && bul_idx >= 0) {
        uint8_t *pts = state->level.object_points + bul_idx * 8;
        obj_sl(pts, (int32_t)hctx.newx << 16);
        obj_sl(pts + 4, (int32_t)hctx.newz << 16);
    }

    SHOT_SET_XVEL(*bullet, xvel << 16);
    SHOT_SET_ZVEL(*bullet, zvel << 16);
    SHOT_POWER(*bullet) = (int8_t)shot_power;
    SHOT_SIZE(*bullet) = (int8_t)shot_type;
    SHOT_SET_LIFE(*bullet, 0);

    /* EnemyFlags = both players (bits 5 and 11) */
    NASTY_SET_EFLAGS(*bullet, 0x00100020);

    /* Y position and vertical aim (AlienControl.s lines 415-439) */
    int16_t obj_y = (int16_t)((obj->raw[4] << 8) | obj->raw[5]);
    int32_t acc_y = ((int32_t)obj_y << 7);
    SHOT_SET_ACCYPOS(*bullet, acc_y);
    bullet->raw[4] = obj->raw[4];
    bullet->raw[5] = obj->raw[5];
    bullet->obj.in_top = obj->obj.in_top;

    /* Vertical aim toward player */
    uint8_t *plr_obj_raw = (player_num == 1) ? state->level.plr1_obj : state->level.plr2_obj;
    if (plr_obj_raw) {
        int16_t plr_y = (int16_t)((plr_obj_raw[4] << 8) | plr_obj_raw[5]);
        int32_t y_diff = ((int32_t)(plr_y - 20) << 7) - acc_y;
        y_diff += y_diff; /* *2 */
        int32_t dist_shifted = dist;
        if (shot_shift > 0) dist_shifted >>= shot_shift;
        if (dist_shifted < 1) dist_shifted = 1;
        SHOT_SET_YVEL(*bullet, (int16_t)(y_diff / dist_shifted));
    }

    /* Gravity and flags from gun data (for shot_type) */
    if (shot_type >= 0 && shot_type < 8) {
        SHOT_SET_GRAV(*bullet, default_plr1_guns[shot_type].shot_gravity);
        SHOT_SET_FLAGS(*bullet, default_plr1_guns[shot_type].shot_flags);
    }

    bullet->obj.worry = 127;

    /* Play firing sound */
    audio_play_sample(3, 100);
}

static int16_t blast_rand_xz_delta(int16_t spread)
{
    /* Anims.s ComputeBlast:
     *   jsr GetRand
     *   ext.w d0        ; byte -> word
     *   muls d5,d0
     *   asr.w #1,d0
     * So X/Z spread uses signed 8-bit randomness. */
    int16_t r = (int16_t)(int8_t)(rand() & 0xFF);
    int32_t prod = (int32_t)r * (int32_t)spread;
    int16_t d = (int16_t)prod;               /* asr.w operates on low word */
    d = (int16_t)(d >> 1);
    if (d == 0) d = 2;
    return d;
}

static int16_t blast_rand_s16(void)
{
    /* Amiga ComputeBlast Y jitter uses GetRand low word with MULS (signed word).
     * Compose a full 16-bit value so sign bit is not lost on CRT rand() impls
     * that only return 15 random bits. */
    uint16_t lo = (uint16_t)(rand() & 0xFF);
    uint16_t hi = (uint16_t)(rand() & 0xFF);
    return (int16_t)(lo | (uint16_t)(hi << 8));
}

static void spawn_blast_particles(GameState *state, int32_t x, int32_t z, int32_t y,
                                  int16_t zone, int8_t in_top)
{
    if (!state) return;

    uint8_t *shot_pool = state->level.player_shot_data;
    if (!shot_pool) return; /* blast particles only use player_shot_data; nasty_shot_data is reserved for gibs */

    int zone_slots = level_zone_slot_count(&state->level);
    int src_zone = level_connect_to_zone_index(&state->level, zone);
    if (src_zone < 0 && zone >= 0 && zone < zone_slots) src_zone = zone;

    uint8_t *src_room = NULL;
    if (src_zone >= 0 && src_zone < zone_slots &&
        state->level.zone_adds && state->level.data) {
        int32_t zo = (int32_t)be32(state->level.zone_adds + (uint32_t)src_zone * 4u);
        src_room = state->level.data + zo;
    }

    for (int ring = 0; ring < 4; ring++) {
        int16_t spread = (int16_t)(2 + ring);          /* Amiga d5: 2..5 */
        int8_t start_anim = (int8_t)(5 - spread);      /* Amiga: shotanim=5-d5 */

        for (int n = 0; n < 3; n++) {
            int16_t saved_cid = -1;
            GameObject *part = find_free_shot_slot(shot_pool, &saved_cid);
            if (!part) return;

            part->obj.number = OBJ_NBR_BULLET;

            MoveContext ctx;
            move_context_init(&ctx);
            ctx.oldx = x;
            ctx.oldz = z;
            ctx.newx = x + blast_rand_xz_delta(spread);
            ctx.newz = z + blast_rand_xz_delta(spread);
            {
                int16_t ry = blast_rand_s16();
                ctx.newy = y + (((int32_t)ry * (int32_t)spread) >> 3); /* asr.l #3 */
            }
            ctx.objroom = src_room;
            ctx.extlen = 80;
            ctx.awayfromwall = 1;
            ctx.exitfirst = 0;
            ctx.wallbounce = 0;
            ctx.stood_in_top = in_top;
            ctx.thing_height = 10 * 128;
            ctx.step_up_val = 0;
            ctx.step_down_val = 0x1000000;
            ctx.wall_flags = 0;

            if (ctx.objroom && (ctx.newx != ctx.oldx || ctx.newz != ctx.oldz))
                move_object(&ctx, &state->level);

            int16_t new_zone = zone;
            if (ctx.objroom && state->level.data) {
                int zi = level_zone_index_from_room_ptr(&state->level, ctx.objroom);
                if (zi < 0) {
                    int16_t room_zone_word = (int16_t)((ctx.objroom[0] << 8) | ctx.objroom[1]);
                    zi = level_connect_to_zone_index(&state->level, room_zone_word);
                }
                if (zi >= 0 && zi < zone_slots) new_zone = (int16_t)zi;
            }

            int32_t py = ctx.newy;
            if (ctx.objroom) {
                int off = ctx.stood_in_top ? 8 : 0;
                int32_t floor_h = be32(ctx.objroom + 2 + off);
                int32_t roof_h  = be32(ctx.objroom + 6 + off);
                if (py > floor_h) py = floor_h;
                if (py < roof_h)  py = roof_h;
            }

            OBJ_SET_ZONE(part, new_zone);
            part->obj.in_top = ctx.stood_in_top;
            SHOT_SET_ACCYPOS(*part, py);
            obj_sw(part->raw + 4, (int16_t)(py >> 7));
            SHOT_STATUS(*part) = 1;
            SHOT_SIZE(*part) = 2;                /* RockPop */
            SHOT_ANIM(*part) = (uint8_t)start_anim;
            SHOT_SET_LIFE(*part, 0);
            part->raw[14] = 0x20;
            part->raw[15] = 0x20;
            if (bullet_pop_tables[2]) {
                const BulletAnimFrame *pf = &bullet_pop_tables[2][(uint8_t)start_anim];
                if (pf->width >= 0) {
                    part->raw[6] = (uint8_t)pf->width;
                    part->raw[7] = (uint8_t)pf->height;
                    obj_sw(part->raw + 8, pf->vect_num);
                    obj_sw(part->raw + 10, pf->frame_num);
                }
            }
            part->obj.worry = 127;

            if (saved_cid >= 0 && state->level.object_points &&
                saved_cid < state->level.num_object_points) {
                uint8_t *pt = state->level.object_points + (uint32_t)saved_cid * 8u;
                obj_sl(pt,     (int32_t)ctx.newx << 16);
                obj_sl(pt + 4, (int32_t)ctx.newz << 16);
            }
        }
    }
}

static bool compute_blast_can_hit_obj_type(int8_t obj_type)
{
    switch (obj_type) {
    case OBJ_NBR_ALIEN:
    case OBJ_NBR_PLR1:
    case OBJ_NBR_ROBOT:
    case OBJ_NBR_BIG_NASTY:
    case OBJ_NBR_FLYING_NASTY:
    case OBJ_NBR_BARREL:
    case OBJ_NBR_PLR2:
    case OBJ_NBR_MARINE:
    case OBJ_NBR_WORM:
    case OBJ_NBR_HUGE_RED_THING:
    case OBJ_NBR_SMALL_RED_THING:
    case OBJ_NBR_TREE:
    case OBJ_NBR_EYEBALL:
    case OBJ_NBR_TOUGH_MARINE:
    case OBJ_NBR_FLAME_MARINE:
    case OBJ_NBR_GAS_PIPE:
        return true;
    default:
        return false;
    }
}

/* Amiga ComputeBlast distance approximation:
 * sqrt(dx*dx + dz*dz) via 3 Newton refinements from 2^(highest_bit/2). */
static int16_t compute_blast_dist_sqrt(int16_t dx, int16_t dz)
{
    int32_t sx = dx;
    int32_t sz = dz;
    int32_t sum_sq = sx * sx + sz * sz;
    if (sum_sq <= 0) return 0;

    int32_t guess = 1;
    for (int bit = 31; bit >= 0; bit--) {
        if (((uint32_t)sum_sq & (1u << bit)) != 0u) {
            guess = 1 << (bit >> 1);
            break;
        }
    }

    for (int i = 0; i < 3; i++) {
        int64_t num = (((int64_t)guess * (int64_t)guess) - (int64_t)sum_sq) >> 1;
        int32_t corr = (int32_t)(num / guess);
        guess -= corr;
        if (guess <= 0) guess = 1;
    }

    if (guess > 32767) guess = 32767;
    return (int16_t)guess;
}

#define BLAST_IMPACT_BASE_MAG 36
#define BLAST_IMPACT_MAX_ABS  64
#define BLAST_LOS_BYPASS_DIST 24

static inline int16_t clamp_blast_impact_component(int32_t v)
{
    if (v > BLAST_IMPACT_MAX_ABS) return (int16_t)BLAST_IMPACT_MAX_ABS;
    if (v < -BLAST_IMPACT_MAX_ABS) return (int16_t)-BLAST_IMPACT_MAX_ABS;
    return (int16_t)v;
}

static inline uint8_t damage_accumulate_u8(uint8_t current, int32_t add)
{
    if (add <= 0) return current;
    {
        int32_t sum = (int32_t)current + add;
        if (sum > 255) sum = 255;
        return (uint8_t)sum;
    }
}

/* -----------------------------------------------------------------------
 * Utility: compute blast damage + visual blast particles
 *
 * Translated from Anims.s ComputeBlast.
 * ----------------------------------------------------------------------- */
void compute_blast(GameState *state, int32_t x, int32_t z, int32_t y,
                   int16_t max_damage, int16_t zone, int8_t in_top)
{
    if (!state || !state->level.object_data) return;
    if (max_damage <= 0) return;

    int zone_slots = level_zone_slot_count(&state->level);
    int src_zone = level_connect_to_zone_index(&state->level, zone);
    if (src_zone < 0 && zone >= 0 && zone < zone_slots) src_zone = zone;
    if (src_zone < 0 || src_zone >= zone_slots ||
        !state->level.zone_adds || !state->level.data) {
        spawn_blast_particles(state, (int16_t)x, (int16_t)z, ((int32_t)((int16_t)(y >> 7))) << 7, zone, in_top);
        return;
    }

    int16_t viewer_x = (int16_t)x;
    int16_t viewer_z = (int16_t)z;
    int16_t viewer_y = (int16_t)(y >> 7);
    int32_t src_off = (int32_t)be32(state->level.zone_adds + (uint32_t)src_zone * 4u);
    if (src_off < 0) {
        spawn_blast_particles(state, viewer_x, viewer_z, ((int32_t)viewer_y) << 7, zone, in_top);
        return;
    }
    const uint8_t *from_room = state->level.data + src_off;

    for (int obj_index = 0;; obj_index++) {
        GameObject *obj = get_object(&state->level, obj_index);
        if (!obj || OBJ_CID(obj) < 0) break;
        if (OBJ_ZONE(obj) < 0) continue;
        if (!compute_blast_can_hit_obj_type(obj->obj.number)) continue;

        int tgt_zone = level_connect_to_zone_index(&state->level, OBJ_ZONE(obj));
        if (tgt_zone < 0 && OBJ_ZONE(obj) >= 0 && OBJ_ZONE(obj) < zone_slots)
            tgt_zone = OBJ_ZONE(obj);
        if (tgt_zone < 0 || tgt_zone >= zone_slots) continue;

        int32_t tgt_off = (int32_t)be32(state->level.zone_adds + (uint32_t)tgt_zone * 4u);
        if (tgt_off < 0) continue;
        const uint8_t *to_room = state->level.data + tgt_off;

        int16_t target_x, target_z;
        get_object_pos(&state->level, (int)OBJ_CID(obj), &target_x, &target_z);
        int16_t target_y = obj_w(obj->raw + 4);

        int16_t dx = (int16_t)(target_x - viewer_x);
        int16_t dz = (int16_t)(target_z - viewer_z);
        int16_t dist = compute_blast_dist_sqrt(dx, dz);

        int16_t target_radius = 40;
        if (obj->obj.number >= 0 && obj->obj.number < 21)
            target_radius = col_box_table[(int)obj->obj.number].width;
        if (target_radius < 0) target_radius = 0;

        /* Treat the target as a cylinder: blast reaches the collision edge, not just center point. */
        int16_t effective_dist = (int16_t)(dist - target_radius);
        if (effective_dist < 0) effective_dist = 0;

        int16_t to_zone_id = (int16_t)((to_room[0] << 8) | to_room[1]);
        /* Very close overlaps are allowed even when LOS is flaky at portal seams/edge cases. */
        if (effective_dist > BLAST_LOS_BYPASS_DIST) {
            uint8_t vis = can_it_be_seen(&state->level,
                                         from_room, to_room, to_zone_id,
                                         viewer_x, viewer_z, viewer_y,
                                         target_x, target_z, target_y,
                                         in_top, obj->obj.in_top, 1);
            if (!vis) continue;
        }

        int16_t dist_bucket = (int16_t)(effective_dist >> 3);
        dist_bucket = (int16_t)(dist_bucket - 4);
        if (dist_bucket < 0) dist_bucket = 0;
        if (dist_bucket > 31) continue;

        int16_t atten = (int16_t)(32 - dist_bucket);
        int32_t damage = ((int32_t)max_damage * (int32_t)atten) >> 5;
        if (damage > max_damage) damage = max_damage;
        if (damage < 1) damage = 1;

        obj->raw[19] = damage_accumulate_u8(obj->raw[19], damage);
        if (obj_index >= 0 && obj_index < MAX_OBJECTS) {
            explosion_damage_flag[obj_index] = 1;
        }

        /* Keep blast knockback in a sane range:
         * directional unit vector scaled by attenuated magnitude.
         * This keeps explosion gib force around "gunshot + 150%" territory
         * instead of huge distance-driven spikes. */
        int32_t push_mag = ((int32_t)atten * BLAST_IMPACT_BASE_MAG + 16) >> 5;
        if (push_mag < 1) push_mag = 1;
        int32_t impact_x = 0;
        int32_t impact_z = 0;
        if (dist > 0) {
            impact_x = ((int32_t)dx * push_mag) / (int32_t)dist;
            impact_z = ((int32_t)dz * push_mag) / (int32_t)dist;
        }
        int16_t impact_x_clamped = clamp_blast_impact_component(impact_x);
        int16_t impact_z_clamped = clamp_blast_impact_component(impact_z);
        if ((int32_t)impact_x_clamped != impact_x || (int32_t)impact_z_clamped != impact_z) {
            printf("[BLAST-IMPACT] clamp obj=%d type=%d raw=(%ld,%ld) clamped=(%d,%d) dist=%d atten=%d\n",
                   obj_index, (int)obj->obj.number,
                   (long)impact_x, (long)impact_z,
                   (int)impact_x_clamped, (int)impact_z_clamped,
                   (int)dist, (int)atten);
        }
        NASTY_SET_IMPACTX(obj, impact_x_clamped);
        NASTY_SET_IMPACTZ(obj, impact_z_clamped);
    }

    spawn_blast_particles(state, viewer_x, viewer_z, ((int32_t)viewer_y) << 7, zone, in_top);
}

/* -----------------------------------------------------------------------
 * Explosion animation (visual only; damage is compute_blast).
 * Progress in TempFrames (50 Hz logic-vblank units) so speed stays stable
 * when a slow frame batches multiple logic ticks.
 * ----------------------------------------------------------------------- */
void explosion_spawn(GameState *state, int16_t x, int16_t z, int16_t zone, int8_t in_top, int32_t y_floor,
                    int8_t size_scale, int8_t anim_rate)
{
    if (state->num_explosions >= MAX_EXPLOSIONS) return;
    if (size_scale <= 0) size_scale = 100;
    if (anim_rate <= 0) anim_rate = 100;
    int i = state->num_explosions++;
    state->explosions[i].x = x;
    state->explosions[i].z = z;
    state->explosions[i].zone = zone;
    state->explosions[i].in_top = in_top;
    state->explosions[i].y_floor = y_floor;
    state->explosions[i].frame = 0;
    state->explosions[i].frame_frac = 0;
    state->explosions[i].size_scale = size_scale;
    state->explosions[i].anim_rate = anim_rate;
    /* 0..3 tick delay so particles don't all animate in lockstep */
    state->explosions[i].start_delay = (int8_t)(rand() & 3);
}

/* anim_rate is percentage of one frame step per logic-vblank (100 = 1 frame / 20ms). */
void explosion_advance(GameState *state)
{
    int n = state->num_explosions;
    int ticks = (int)state->temp_frames;
    if (ticks < 1) ticks = 1;
    for (int i = 0; i < n; i++) {
        int ticks_left = ticks;
        if (state->explosions[i].start_delay > 0) {
            int delay = (int)state->explosions[i].start_delay;
            if (ticks_left >= delay) {
                ticks_left -= delay;
                state->explosions[i].start_delay = 0;
            } else {
                state->explosions[i].start_delay = (int8_t)(delay - ticks_left);
                ticks_left = 0;
            }
        }
        if (ticks_left > 0) {
            int rate = (int)state->explosions[i].anim_rate;
            if (rate <= 0) rate = 100;
            int frac = (int)state->explosions[i].frame_frac + (rate * ticks_left);
            while (frac >= 100) {
                state->explosions[i].frame = (int8_t)(state->explosions[i].frame + 1);
                frac -= 100;
            }
            state->explosions[i].frame_frac = (int8_t)frac;
        }
        if ((int)state->explosions[i].frame >= 9) {
            /* Remove: shift down */
            n--;
            for (int j = i; j < n; j++)
                state->explosions[j] = state->explosions[j + 1];
            state->num_explosions = n;
            i--;
        }
    }
}

/* -----------------------------------------------------------------------
 * Utility: pickup distance check
 *
 * Returns 1 if player is close enough to pick up the object.
 * ----------------------------------------------------------------------- */
int pickup_distance_check(GameObject *obj, GameState *state, int player_num)
{
    PlayerState *plr = (player_num == 1) ? &state->plr1 : &state->plr2;

    int16_t obj_zone = OBJ_ZONE(obj);
    if (obj_zone < 0 || plr->zone != obj_zone) {
        return 0;
    }
    if ((plr->stood_in_top != 0) != (obj->obj.in_top != 0)) {
        return 0;
    }

    int16_t cid = OBJ_CID(obj);
    if (cid < 0 || !state->level.object_points ||
        cid >= state->level.num_object_points) {
        return 0;
    }

    int16_t ox, oz;
    get_object_pos(&state->level, (int)cid, &ox, &oz);

    int32_t dx = (plr->xoff >> 16) - ox;
    int32_t dz = (plr->zoff >> 16) - oz;
    int32_t dist_sq = dx * dx + dz * dz;
    if (dist_sq >= PICKUP_DISTANCE_SQ) {
        return 0;
    }

    int16_t pickup_half_height = 20;
    int obj_type = (int8_t)obj->obj.number;
    if (obj_type >= 0 && obj_type < 21) {
        pickup_half_height = col_box_table[obj_type].half_height;
    }

    int16_t obj_y = obj_w(obj->raw + 4);
    int16_t obj_bot = obj_y - pickup_half_height;
    int16_t obj_top = obj_y + pickup_half_height;

    int16_t plr_bot = (int16_t)(plr->yoff >> 7);
    int16_t plr_top = (int16_t)((plr->yoff + plr->height) >> 7);
    if (plr_top < plr_bot) {
        int16_t tmp = plr_top;
        plr_top = plr_bot;
        plr_bot = tmp;
    }

    if (plr_top < obj_bot || plr_bot > obj_top) {
        return 0;
    }

    return 1;
}

/* -----------------------------------------------------------------------
 * USEPLR1 / USEPLR2 - Update player object data for rendering
 *
 * Translated from AB3DI.s USEPLR1 (line ~2302-2537).
 * Copies player state into the player's GameObject in ObjectData.
 * ----------------------------------------------------------------------- */
void use_player1(GameState *state)
{
    if (!state->level.plr1_obj) return;

    GameObject *plr_obj = (GameObject*)state->level.plr1_obj;

    /* Update position in ObjectPoints */
    int idx = 0;
    if (state->level.object_points) {
        idx = (int)(state->level.plr1_obj - state->level.object_data) / OBJECT_SIZE;
        uint8_t *pts = state->level.object_points + idx * 8;
        obj_sw(pts, (int16_t)(state->plr1.xoff >> 16));
        obj_sw(pts + 4, (int16_t)(state->plr1.zoff >> 16));
    }

    /* Update zone */
    OBJ_SET_ZONE(plr_obj, state->plr1.zone);

    /* Update objInTop */
    plr_obj->obj.in_top = state->plr1.stood_in_top;

    /* Damage flash + pain sound (AB3DI.s lines 2323-2348) */
    int8_t damage = NASTY_DAMAGE(*plr_obj);
    if (damage > 0) {
        state->plr1.energy -= damage;
        NASTY_DAMAGE(*plr_obj) = 0;
        state->hitcol = 0xF00; /* Red flash */
        audio_play_sample(19, amiga_noisevol_to_pc(100)); /* AB3DI.s: pant @ Noisevol 100 */
    }

    /* Update numlives from energy */
    NASTY_LIVES(*plr_obj) = (int8_t)(state->plr1.energy + 1);

    /* Zone brightness (AB3DI.s lines 2358-2366) */
    if (state->plr1.zone >= 0) {
        int16_t zb = level_get_zone_brightness(&state->level, state->plr1.zone,
                                                state->plr1.stood_in_top ? 1 : 0);
        plr_obj->raw[2] = (uint8_t)(zb >> 8);
        plr_obj->raw[3] = (uint8_t)(zb);
    }

    /* Y position: (yoff + height/2) >> 7 (AB3DI.s line 2368) */
    int32_t plr_y = (state->plr1.p_yoff + state->plr1.s_height / 2) >> 7;
    plr_obj->raw[4] = (uint8_t)(plr_y >> 8);
    plr_obj->raw[5] = (uint8_t)(plr_y);

    /* ViewpointToDraw for PLR2 looking at PLR1 (AB3DI.s line 2370) */
    /* This sets the animation frame for how PLR1 looks from PLR2's perspective.
     * Rendering-only but the angle calc is game logic. */
    int16_t frame = viewpoint_to_draw(
        (int16_t)(state->plr2.xoff >> 16), (int16_t)(state->plr2.zoff >> 16),
        (int16_t)(state->plr1.xoff >> 16), (int16_t)(state->plr1.zoff >> 16),
        (int16_t)(state->plr1.angpos * 2));

    /* Facing (AB3DI.s lines 2407-2416) */
    plr_obj->raw[8] = (uint8_t)(state->plr1.angpos >> 8);
    plr_obj->raw[9] = (uint8_t)(state->plr1.angpos);

    /* Animation frame with head bob + viewpoint (AB3DI.s line 2418) */
    int16_t anim = (int16_t)(frame + state->plr1.bob_frame);
    plr_obj->raw[10] = (uint8_t)(anim >> 8);
    plr_obj->raw[11] = (uint8_t)(anim);

    /* Graphic room = -1 (AB3DI.s line 2420) */
    OBJ_SET_GROOM(plr_obj, -1);
}

void use_player2(GameState *state)
{
    if (!state->level.plr2_obj) return;

    GameObject *plr_obj = (GameObject*)state->level.plr2_obj;

    int idx = 0;
    if (state->level.object_points) {
        idx = (int)(state->level.plr2_obj - state->level.object_data) / OBJECT_SIZE;
        uint8_t *pts = state->level.object_points + idx * 8;
        obj_sw(pts, (int16_t)(state->plr2.xoff >> 16));
        obj_sw(pts + 4, (int16_t)(state->plr2.zoff >> 16));
    }

    OBJ_SET_ZONE(plr_obj, state->plr2.zone);
    plr_obj->obj.in_top = state->plr2.stood_in_top;

    int8_t damage = NASTY_DAMAGE(*plr_obj);
    if (damage > 0) {
        state->plr2.energy -= damage;
        NASTY_DAMAGE(*plr_obj) = 0;
        state->hitcol2 = 0xF00;
        audio_play_sample(19, amiga_noisevol_to_pc(100));
    }

    NASTY_LIVES(*plr_obj) = (int8_t)(state->plr2.energy + 1);

    if (state->plr2.zone >= 0) {
        int16_t zb = level_get_zone_brightness(&state->level, state->plr2.zone,
                                                state->plr2.stood_in_top ? 1 : 0);
        plr_obj->raw[2] = (uint8_t)(zb >> 8);
        plr_obj->raw[3] = (uint8_t)(zb);
    }

    int32_t plr_y = (state->plr2.p_yoff + state->plr2.s_height / 2) >> 7;
    plr_obj->raw[4] = (uint8_t)(plr_y >> 8);
    plr_obj->raw[5] = (uint8_t)(plr_y);

    int16_t frame = viewpoint_to_draw(
        (int16_t)(state->plr1.xoff >> 16), (int16_t)(state->plr1.zoff >> 16),
        (int16_t)(state->plr2.xoff >> 16), (int16_t)(state->plr2.zoff >> 16),
        (int16_t)(state->plr2.angpos * 2));

    plr_obj->raw[8] = (uint8_t)(state->plr2.angpos >> 8);
    plr_obj->raw[9] = (uint8_t)(state->plr2.angpos);

    int16_t anim = (int16_t)(frame + state->plr2.bob_frame);
    plr_obj->raw[10] = (uint8_t)(anim >> 8);
    plr_obj->raw[11] = (uint8_t)(anim);

    OBJ_SET_GROOM(plr_obj, -1);
}

/* -----------------------------------------------------------------------
 * CalcPLR1InLine / CalcPLR2InLine
 *
 * Translated from AB3DI.s CalcPLR1InLine (line ~4171-4236).
 *
 * For each object, determine if it's in the player's field of view
 * and calculate its distance. Used by auto-aim and rendering.
 * ----------------------------------------------------------------------- */
static void calc_plr_in_line_for(LevelState *level, const PlayerState *plr,
                                 int8_t *obs_out, int16_t *dists_out)
{
    int16_t sin_val = plr->sinval;
    int16_t cos_val = plr->cosval;
    int16_t plr_x = (int16_t)(plr->xoff >> 16);
    int16_t plr_z = (int16_t)(plr->zoff >> 16);

    int point_count = level->num_object_points;
    if (point_count < 0) point_count = 0;
    if (point_count > MAX_OBJECTS) point_count = MAX_OBJECTS;

    for (int i = 0; i < point_count; i++) {
        GameObject *obj = get_object(level, i);
        if (!obj) break;
        if (OBJ_CID(obj) < 0) break;

        if (OBJ_ZONE(obj) < 0) {
            obs_out[i] = 0;
            dists_out[i] = 0;
            continue;
        }

        /* Amiga CalcPLR1InLine walks ObjectPoints and ObjectData in slot order. */
        const uint8_t *pt = level->object_points + (size_t)i * 8u;
        int16_t ox = obj_w(pt + 0);
        int16_t oz = obj_w(pt + 4);
        int16_t dx = (int16_t)(ox - plr_x);
        int16_t dz = (int16_t)(oz - plr_z);

        int obj_type = (uint8_t)obj->obj.number;
        int16_t box_width = 40;
        if (obj_type >= 0 && obj_type <= 20) {
            box_width = col_box_table[obj_type].width;
        }

        int32_t cross = (int32_t)(((int64_t)dx * cos_val) - ((int64_t)dz * sin_val));
        cross = (int32_t)((uint32_t)cross << 1);
        if (cross <= 0) cross = -cross;
        int16_t perp = (int16_t)(((uint32_t)cross) >> 16);

        int32_t dot = (int32_t)(((int64_t)dx * sin_val) + ((int64_t)dz * cos_val));
        dot = (int32_t)((uint32_t)dot << 2);
        int16_t fwd = (int16_t)(((uint32_t)dot) >> 16);

        int8_t in_line = 0;
        if (fwd > 0) {
            int16_t half_perp = (int16_t)(perp >> 1);
            if (half_perp <= box_width) in_line = -1;
        }

        obs_out[i] = in_line;
        dists_out[i] = fwd;

        /* PlayerShoot reads ObjDists by CID; keep that mapping synced too. */
        {
            int16_t cid = OBJ_CID(obj);
            if (cid >= 0 && cid < MAX_OBJECTS) dists_out[cid] = fwd;
        }
    }
}

void calc_plr1_in_line(GameState *state)
{
    if (!state->level.object_data || !state->level.object_points) return;
    memset(plr1_obs_in_line, 0, sizeof(plr1_obs_in_line));
    memset(plr1_obj_dists, 0, sizeof(plr1_obj_dists));
    calc_plr_in_line_for(&state->level, &state->plr1, plr1_obs_in_line, plr1_obj_dists);
}

void calc_plr2_in_line(GameState *state)
{
    if (!state->level.object_data || !state->level.object_points) return;
    memset(plr2_obs_in_line, 0, sizeof(plr2_obs_in_line));
    memset(plr2_obj_dists, 0, sizeof(plr2_obj_dists));
    calc_plr_in_line_for(&state->level, &state->plr2, plr2_obs_in_line, plr2_obj_dists);
}

