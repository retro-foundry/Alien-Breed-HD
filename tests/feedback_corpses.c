/* Requested corpse gibbing through ComputeBlast / Anims.s ExplodeIntoBits. */
#include "../src/objects.c"
#include "sb_decompress.h"
#include <assert.h>

GameState g_state;
static GameState state;
static LevelState geometry;
static GameObject objects[2], shots[NASTY_SHOT_SLOT_COUNT];
static uint8_t points[(NASTY_SHOT_SLOT_COUNT + 2) * 8];
static uint8_t previous_points[sizeof(points)];
static int noises, last_noise;
void audio_play_sample(int sample, int volume)
{
    (void)volume; noises++; last_noise = sample;
}

static GameObject *setup(int type)
{
    memset(&state, 0, sizeof(state));
    memset(objects, 0, sizeof(objects));
    memset(shots, 0, sizeof(shots));
    memset(points, 0, sizeof(points));
    memset(previous_points, 0, sizeof(previous_points));
    memset(explosion_damage_flag, 0, sizeof(explosion_damage_flag));
    state.level = geometry;
    state.level.object_data = (uint8_t *)objects;
    state.level.object_points = points;
    state.level.prev_object_points = previous_points;
    state.level.num_object_points = NASTY_SHOT_SLOT_COUNT + 2;
    state.level.nasty_shot_data = (uint8_t *)shots;
    state.level.player_shot_data = NULL;
    state.level.plr1_obj = state.level.plr2_obj = NULL;
    state.plr1.zone = state.plr2.zone = -1;
    state.nasty = 1; state.temp_frames = 1;
    for (int i = 0; i < NASTY_SHOT_SLOT_COUNT; i++) {
        OBJ_SET_CID(&shots[i], i + 1); OBJ_SET_ZONE(&shots[i], -1);
    }
    GameObject *obj = &objects[0];
    OBJ_SET_CID(obj, 0); OBJ_SET_ZONE(obj, 1);
    OBJ_SET_CID(&objects[1], -1);
    obj->obj.number = type;
    obj->raw[6] = default_object_world_size[type].w;
    obj->raw[7] = default_object_world_size[type].h;
    int floor = obj_l(level_get_zone_data_ptr(&state.level, 1) + ZONE_OFF_FLOOR);
    obj_sw(obj->raw + 4, floor / 128 - 64);
    obj_sw(points, -736); obj_sw(points + 4, 328);
    memcpy(previous_points, points, sizeof(points));
    noises = 0; srand(1234);
    return obj;
}

static int gib_count(void)
{
    int count = 0;
    for (int i = 0; i < NASTY_SHOT_SLOT_COUNT; i++) {
        GameObject *bit = &shots[i];
        if (OBJ_ZONE(bit) < 0 || SHOT_SIZE(*bit) < 50 || SHOT_SIZE(*bit) > 53) continue;
        assert(SHOT_POWER(*bit) == 0 && SHOT_GRAV(*bit) == 40);
        assert(bit->obj.number == OBJ_NBR_BULLET);
        assert(obj_w(points + OBJ_CID(bit) * 8) == -736);
        assert(obj_w(points + OBJ_CID(bit) * 8 + 4) == 328);
        assert(!memcmp(points + OBJ_CID(bit) * 8, previous_points + OBJ_CID(bit) * 8, 8));
        count++;
    }
    return count;
}

static void blast(int damage, int distance)
{
    compute_blast(&state, -736 + distance, 328,
                  obj_w(objects[0].raw + 4) * 128, damage, 1, 0);
}

static void check_floater_report(void)
{
    GameObject *obj = setup(OBJ_NBR_FLYING_NASTY);
    NASTY_LIVES(*obj) = 0; OBJ_SET_DEADL(obj, 20);
    blast(100, 0);
    object_handle_flying_nasty(obj, &state);
    if (noises) fprintf(stderr, "Corpse explosion made %d sounds (last sample=%d)\n", noises, last_noise);
    assert(noises == 0);
    assert(OBJ_ZONE(obj) < 0 && gib_count() > 0);
    puts("PASS: exploding the resting floater corpse emits gibs without pain/death audio");
}

static void check_all_corpse_frames(void)
{
    const int types[] = {OBJ_NBR_ALIEN, OBJ_NBR_WORM, OBJ_NBR_MARINE,
        OBJ_NBR_TOUGH_MARINE, OBJ_NBR_FLAME_MARINE, OBJ_NBR_SMALL_RED_THING,
        OBJ_NBR_FLYING_NASTY};
    const int damages[] = {1, 100, 400};
    int cases = 0;
    for (unsigned t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        int type = types[t], index = obj_type_to_enemy_index(type);
        const EnemyParams *params = &enemy_params[index];
        int frames = 3;
        if (type != OBJ_NBR_FLYING_NASTY) {
            for (frames = 0; frames < 30 && params->death_frames[frames] >= 0; frames++) {}
            assert(frames > 0);
        }
        for (int frame = 0; frame < frames; frame++) {
            for (unsigned d = 0; d < sizeof(damages) / sizeof(damages[0]); d++) {
                GameObject *obj = setup(type);
                if (type == OBJ_NBR_FLYING_NASTY) {
                    NASTY_LIVES(*obj) = 0; OBJ_SET_DEADL(obj, 18 + frame);
                } else {
                    enemy_apply_death_outcome(obj, params, false);
                    obj->obj.type_data[0] = frame;
                    OBJ_SET_DEADH(obj, params->death_frames[frame]);
                }
                /* Dormant and still-burning bodies must not need another AI tick. */
                obj->obj.worry = 0;
                state.enemy_burn_time_left[0] = 50;
                blast(damages[d], 0);
                assert(OBJ_ZONE(obj) < 0 && noises == 0);
                int count = gib_count();
                assert(count > 0);
                assert(state.enemy_burn_time_left[0] == 0 && explosion_damage_flag[0] == 0);
                blast(damages[d], 0);
                assert(gib_count() == count && noises == 0);
                cases++;
            }
        }
    }
    printf("PASS: %d corpse-frame/blast-strength cases: seven retained corpse types, falling/settled floaters, dormant/burning bodies, no repeat gibbing\n", cases);
}

static void check_boundaries(void)
{
    GameObject *obj = setup(OBJ_NBR_ALIEN);
    enemy_apply_death_outcome(obj, &enemy_params[0], false);
    GameObject before = *obj;
    blast(0, 0); blast(100, 1000);
    assert(!memcmp(&before, obj, sizeof(before)) && gib_count() == 0 && noises == 0);

    /* Negative object numbers also identify decorative props. */
    for (int vector = 0; vector < 2; vector++) {
        obj = setup(OBJ_NBR_ALIEN);
        enemy_apply_death_outcome(obj, &enemy_params[0], false);
        if (vector) obj->raw[6] = OBJ_3D_SPRITE;
        else OBJ_SET_DEADH(obj, 0); /* Authored billboard, not a death frame. */
        before = *obj;
        blast(100, 0);
        assert(!memcmp(&before, obj, sizeof(before)) && gib_count() == 0 && noises == 0);
    }
    obj = setup(OBJ_NBR_KEY); obj->obj.can_see = 8;
    before = *obj; blast(400, 0);
    assert(!memcmp(&before, obj, sizeof(before)) && gib_count() == 0);

    /* Living enemies keep their normal damage and pain response. */
    obj = setup(OBJ_NBR_FLYING_NASTY); NASTY_LIVES(*obj) = 50;
    blast(1, 0);
    assert(NASTY_LIVES(*obj) == 50 && gib_count() == 0);
    assert(!enemy_check_damage(obj, &enemy_params[9], &state));
    assert(NASTY_LIVES(*obj) == 49 && noises == 1 && last_noise == 8);

    obj = setup(OBJ_NBR_FLYING_NASTY); NASTY_LIVES(*obj) = 0; OBJ_SET_DEADL(obj, 20);
    obj->raw[19] = 100;
    object_handle_flying_nasty(obj, &state);
    assert(noises == 0 && OBJ_ZONE(obj) == 1 && gib_count() == 0);
    obj->raw[19] = 100; explosion_damage_flag[0] = 1;
    object_handle_flying_nasty(obj, &state);
    assert(noises == 0 && OBJ_ZONE(obj) < 0 && gib_count() > 0);

    obj = setup(OBJ_NBR_FLYING_NASTY); NASTY_LIVES(*obj) = 0; OBJ_SET_DEADL(obj, 20);
    for (int i = 0; i < NASTY_SHOT_SLOT_COUNT; i++) OBJ_SET_ZONE(&shots[i], 1);
    blast(400, 0);
    assert(noises == 0 && OBJ_ZONE(obj) < 0 && gib_count() == 0);
    puts("PASS: range/zero blast, decorative props, blue key, living pain, stale corpse damage and full debris pool");
}

static void check_explosive_weapons(void)
{
    const int types[] = {OBJ_NBR_ALIEN, OBJ_NBR_FLYING_NASTY};
    const int guns[] = {AB3D_GUN_ROCKET, AB3D_GUN_GRENADE_LAUNCHER};
    for (int type = 0; type < 2; type++) {
        for (int gun = 0; gun < 2; gun++) {
            GameObject *corpse = setup(types[type]);
            if (types[type] == OBJ_NBR_FLYING_NASTY) {
                NASTY_LIVES(*corpse) = 0; OBJ_SET_DEADL(corpse, 20);
            } else {
                enemy_apply_death_outcome(corpse, &enemy_params[0], false);
            }
            GameObject projectile = {0};
            int cid = NASTY_SHOT_SLOT_COUNT + 1;
            OBJ_SET_CID(&projectile, cid); OBJ_SET_ZONE(&projectile, 1);
            projectile.obj.number = OBJ_NBR_BULLET;
            SHOT_SIZE(projectile) = guns[gun];
            const GunDataEntry *data = &default_plr1_guns[guns[gun]];
            SHOT_SET_FLAGS(projectile, data->shot_flags);
            SHOT_SET_GRAV(projectile, data->shot_gravity);
            SHOT_POWER(projectile) = data->shot_power;
            int y = obj_w(corpse->raw + 4) * 128;
            if (guns[gun] == AB3D_GUN_ROCKET) {
                /* Rockets have no fuse: detonate against the real room floor. */
                y = obj_l(level_get_zone_data_ptr(&state.level, 1) + ZONE_OFF_FLOOR) - 5 * 128;
            } else {
                SHOT_SET_LIFE(projectile, data->bullet_lifetime + 1);
            }
            SHOT_SET_ACCYPOS(projectile, y);
            obj_sw(projectile.raw + 4, y / 128);
            obj_sw(points + cid * 8, -736); obj_sw(points + cid * 8 + 4, 328);
            object_handle_bullet(&projectile, &state);
            assert(OBJ_ZONE(corpse) < 0 && gib_count() > 0);
            assert(noises == 1 && last_noise == 15); /* Weapon explosion only. */
        }
    }
    GameObject *corpse = setup(OBJ_NBR_FLYING_NASTY);
    NASTY_LIVES(*corpse) = 0; OBJ_SET_DEADL(corpse, 20);
    GameObject barrel = {0};
    int cid = NASTY_SHOT_SLOT_COUNT + 1;
    OBJ_SET_CID(&barrel, cid); OBJ_SET_ZONE(&barrel, 1);
    barrel.obj.number = OBJ_NBR_BARREL;
    NASTY_LIVES(barrel) = 1; NASTY_DAMAGE(barrel) = 1;
    obj_sw(points + cid * 8, -736); obj_sw(points + cid * 8 + 4, 328);
    object_handle_barrel(&barrel, &state);
    assert(OBJ_ZONE(corpse) < 0 && gib_count() > 0);
    assert(noises == 1 && last_noise == 15);
    puts("PASS: actual rocket floor impact, grenade fuse and barrel detonation gib corpses; only the weapon explosion sound plays");
}

int main(void)
{
    math_tables_init(); size_t bytes;
    assert(!sb_load_file("amiga/data/levels/level_h/twolev.bin", &geometry.data, &bytes));
    geometry.data_byte_count = bytes;
    assert(!sb_load_file("amiga/data/levels/level_h/twolev.graph.bin", &geometry.graphics, &bytes));
    geometry.graphics_byte_count = bytes;
    assert(!level_parse(&geometry));
    check_floater_report();
    check_all_corpse_frames();
    check_boundaries();
    check_explosive_weapons();
    return 0;
}
