/* Robot.s ItsARobot / ROBAttack_Player behavior and level_h escape routes. */
#include "../src/objects.c"
#include "sb_decompress.h"
#include <assert.h>

GameState g_state;
static int attacks;
void audio_play_sample(int sample, int volume)
{
    (void)volume;
    if (sample == 9) attacks++;
}

static GameState state;
static GameObject objects[3];
static uint8_t points[(NASTY_SHOT_SLOT_COUNT + 3) * 8];
static GameObject shots[NASTY_SHOT_SLOT_COUNT];

static GameObject *setup(void)
{
    memset(&state, 0, sizeof(state));
    memset(objects, 0, sizeof(objects));
    memset(points, 0, sizeof(points));
    memset(shots, 0, sizeof(shots));
    for (int i = 0; i < NASTY_SHOT_SLOT_COUNT; i++) {
        OBJ_SET_CID(&shots[i], i + 3);
        OBJ_SET_ZONE(&shots[i], -1);
    }
    memset(g_enemy_motion_stamp, 0, sizeof(g_enemy_motion_stamp));
    state.level.object_data = (uint8_t *)objects;
    state.level.object_points = points;
    state.level.num_object_points = NASTY_SHOT_SLOT_COUNT + 3;
    state.level.nasty_shot_data = (uint8_t *)shots;
    state.nasty = 1;
    state.temp_frames = 1;
    state.plr1.zone = state.plr2.zone = -1;
    state.plr1.p_zoff = 1000;
    state.plr1.zoff = state.plr2.zoff = 1000 * 65536;
    OBJ_SET_CID(&objects[0], 0);
    OBJ_SET_ZONE(&objects[0], 0);
    OBJ_SET_CID(&objects[1], -1);
    objects[0].obj.number = OBJ_NBR_ROBOT;
    objects[0].obj.can_see = 1;
    NASTY_LIVES(objects[0]) = 100;
    NASTY_SET_MAXSPD(objects[0], 10);
    NASTY_SET_FACING(objects[0], ANGLE_90);
    NASTY_SET_TIMER(objects[0], 100);
    OBJ_SET_TD_W(&objects[0], ENEMY_FOURTH_TIMER_OFF, 30);
    return &objects[0];
}

static void check_turning(void)
{
    for (int player = 1; player <= 2; player++) {
        for (int side = -1; side <= 1; side += 2) {
            GameObject *obj = setup();
            PlayerState *p = player == 1 ? &state.plr1 : &state.plr2;
            p->p_xoff = side * 10; p->p_zoff = 1000;
            p->xoff = p->p_xoff * 65536; p->zoff = 1000 * 65536;
            NASTY_SET_FACING(*obj, 0); /* Internal Robot.s facing. */
            bool can_shoot;
            robot_track_target(obj, &enemy_params[1], &state, player, &can_shoot);
            assert(can_shoot);
            assert(NASTY_FACING(*obj) == 0); /* Inside the original turn deadband. */
        }
    }
    for (int player = 1; player <= 2; player++) {
        GameObject *obj = setup();
        PlayerState *p = player == 1 ? &state.plr1 : &state.plr2;
        p->p_xoff = 1000; p->p_zoff = 0; /* Sim position before collision. */
        p->xoff = 0; p->zoff = 1000 * 65536; /* Current resolved position. */
        NASTY_SET_FACING(*obj, 0);
        bool can_shoot;
        robot_track_target(obj, &enemy_params[1], &state, player, &can_shoot);
        assert(can_shoot && NASTY_FACING(*obj) == 0);
        assert(obj_w(points) == 0 && obj_w(points + 4) == 10);
    }
    puts("PASS: robot aim remains stable inside the original turn deadband for both targets");
}

static void check_prowl_and_attack(void)
{
    GameObject *obj = setup();
    obj->obj.can_see = 0;
    obj_sw(obj->raw + 28, 1024); /* CurrCPt: wander toward 45 degrees. */
    object_handle_robot(obj, &state);
    assert(NASTY_FACING(*obj) == ANGLE_90 + 120);
    assert(obj_w(points + 4) > 0);
    assert(obj_w(obj->raw + 28) == 1024);

    obj = setup();
    OBJ_SET_TD_W(obj, ENEMY_THIRD_TIMER_OFF, 150);
    NASTY_SET_TIMER(*obj, 1);
    object_handle_robot(obj, &state);
    assert(OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF) == 149);
    assert(OBJ_TD_W(obj, ENEMY_FOURTH_TIMER_OFF) == 29);
    assert(NASTY_TIMER(*obj) == 0); /* Expired ObjTimer resumes attack now. */

    for (int player = 1; player <= 2; player++) {
        for (int batch = 1; batch <= 4; batch += 3) {
            obj = setup();
            obj->obj.can_see = 1 << (player - 1);
            state.plr2.p_zoff = 1000;
            state.temp_frames = batch;
            int before = attacks;
            for (int tick = 0; tick < 40 && attacks == before; tick++)
                object_handle_robot(obj, &state);
            assert(attacks == before + 1);
            assert(OBJ_ZONE(&shots[0]) == OBJ_ZONE(obj) && SHOT_POWER(shots[0]) == 10);
            assert(OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF) == -1);
            assert(OBJ_TD_W(obj, ENEMY_FOURTH_TIMER_OFF) == 50);
            for (int tick = 0; tick < 40 && attacks == before + 1; tick++)
                object_handle_robot(obj, &state);
            assert(attacks == before + 2);
            assert(OBJ_ZONE(&shots[1]) == OBJ_ZONE(obj) && SHOT_POWER(shots[1]) == 10);
            int third = OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF);
            assert(player == 1 ? (third >= 150 && third <= 277) : third == -2);
            if (player == 2) {
                for (int tick = 0; tick < 40 && attacks == before + 2; tick++)
                    object_handle_robot(obj, &state);
                assert(attacks == before + 3);
                third = OBJ_TD_W(obj, ENEMY_THIRD_TIMER_OFF);
                assert(third >= 100 && third <= 227);
            }
        }
    }
    obj = setup();
    OBJ_SET_TD_W(obj, ENEMY_FOURTH_TIMER_OFF, 19);
    obj->raw[19] = 16;
    int before = attacks;
    object_handle_robot(obj, &state);
    assert(NASTY_LIVES(*obj) == 99 && attacks == before);
    object_handle_robot(obj, &state);
    assert(attacks == before + 1);
    puts("PASS: robot prowl turns gradually, timer expiry resumes attack, original shot bursts survive tick batching");
}

static void check_level8_routes(void)
{
    size_t bytes;
    static GameState original;
    assert(!sb_load_file("amiga/data/levels/level_h/twolev.bin", &original.level.data, &bytes));
    original.level.data_byte_count = bytes;
    assert(!sb_load_file("amiga/data/levels/level_h/twolev.graph.bin", &original.level.graphics, &bytes));
    original.level.graphics_byte_count = bytes;
    assert(!level_parse(&original.level));
    GameObject *boss = NULL;
    for (int i = 0; ; i++) {
        GameObject *candidate = get_object(&original.level, i);
        if (OBJ_CID(candidate) < 0) break;
        if (candidate->obj.number == OBJ_NBR_ROBOT) { boss = candidate; break; }
    }
    assert(boss);
    GameObject authored = *boss;
    uint8_t *position = original.level.object_points + OBJ_CID(boss) * 8;
    int start_x = obj_w(position), start_z = obj_w(position + 4), start_zone = OBJ_ZONE(boss);
    const int destinations[][3] = {{400,328,29},{-2200,328,49},{-736,1440,19},{-736,-1176,39}};
    int routes = 0;
    for (int player = 1; player <= 2; player++) {
        for (int direction = 0; direction < 4; direction++) {
            memset(&state, 0, sizeof(state)); state.level = original.level;
            *boss = authored;
            obj_sw(position, start_x); obj_sw(position + 4, start_z);
            state.nasty = 1; state.temp_frames = 1;
            state.plr1.zone = state.plr2.zone = -1;
            PlayerState *p = player == 1 ? &state.plr1 : &state.plr2;
            p->p_xoff = destinations[direction][0]; p->p_zoff = destinations[direction][1];
            p->zone = destinations[direction][2];
            p->xoff = p->p_xoff * 65536; p->zoff = p->p_zoff * 65536;
            uint8_t *room = level_get_zone_data_ptr(&state.level, p->zone);
            p->p_yoff = obj_l(room + ZONE_OFF_FLOOR) - PLAYER_HEIGHT;
            int escaped = 0, before = attacks;
            srand(1234);
            for (int tick = 0; tick < 500; tick++) {
                object_handle_robot(boss, &state);
                if (OBJ_ZONE(boss) != start_zone) { escaped = tick + 1; break; }
            }
            printf("Robot level 8 route: target=%d direction=%d escape=%d attacks=%d position=%d,%d zone=%d\n",
                   player, direction, escaped, attacks-before, obj_w(position), obj_w(position+4), OBJ_ZONE(boss));
            assert(escaped);
            routes++;
        }
    }
    printf("PASS: robot leaves its authored level 8 starting area on %d chase routes\n", routes);

    /* Check the four 192-unit outer exits with the original collision size.
     * Gradual turning is checked separately above: diagonal approaches may
     * legitimately clip a jamb and restart the original wander timer. */
    const int exits[][5] = {{4,29,112,352,2048}, {0,49,-1584,352,6144},
                            {2,19,-736,1200,0}, {3,39,-736,-496,4096}};
    for (int direction = 0; direction < 4; direction++) {
        memset(&state, 0, sizeof(state)); state.level = original.level;
        *boss = authored; OBJ_SET_ZONE(boss, exits[direction][0]);
        obj_sw(position, exits[direction][2]); obj_sw(position + 4, exits[direction][3]);
        state.nasty = 1; state.temp_frames = 1;
        state.plr1.zone = state.plr2.zone = -1;
        boss->obj.can_see = 0;
        NASTY_SET_TIMER(*boss, 100);
        obj_sw(boss->raw + 28, exits[direction][4]);
        NASTY_SET_FACING(*boss, (exits[direction][4] + ANGLE_90) & ANGLE_MASK);
        int escaped = 0;
        for (int tick = 0; tick < 20; tick++) {
            object_handle_robot(boss, &state);
            if (OBJ_ZONE(boss) == exits[direction][1]) { escaped = tick + 1; break; }
        }
        printf("Robot level 8 prowl exit: %d->%d ticks=%d\n", exits[direction][0], exits[direction][1], escaped);
        assert(escaped);
    }
    puts("PASS: original-size robot crosses all four outer exits while prowling");

    for (int attacking = 0; attacking <= 1; attacking++) {
        GameObject *obj = setup();
        LevelState level = state.level;
        state.level = original.level;
        state.level.object_data = level.object_data;
        state.level.object_points = level.object_points;
        state.level.num_object_points = level.num_object_points;
        state.level.nasty_shot_data = level.nasty_shot_data;
        state.level.player_shot_data = NULL;
        state.level.plr1_obj = state.level.plr2_obj = NULL;
        OBJ_SET_ZONE(obj, start_zone);
        obj_sw(points, start_x); obj_sw(points + 4, start_z);
        obj_sw(obj->raw + 4, obj_w(authored.raw + 4));
        NASTY_LIVES(*obj) = 1; obj->raw[19] = 16;
        obj->obj.can_see = attacking;
        obj_sw(obj->raw + 28, ANGLE_180); /* Prowl stops while turning. */
        state.plr1.zone = start_zone; state.plr1.energy = 1000;
        state.plr1.p_xoff = start_x + 200; state.plr1.p_zoff = start_z;
        state.plr1.xoff = state.plr1.p_xoff * 65536;
        state.plr1.zoff = state.plr1.p_zoff * 65536;
        state.plr1.s_height = PLAYER_HEIGHT;
        state.plr1.p_yoff = obj_l(level_get_zone_data_ptr(&state.level, start_zone) + ZONE_OFF_FLOOR) - PLAYER_HEIGHT;
        int before = attacks;
        object_handle_robot(obj, &state);
        int bucket = ((200 - col_box_table[OBJ_NBR_PLR1].width) >> 3) - 4;
        int expected = ((attacking ? 400 : 120) * (32 - bucket)) >> 5;
        assert(1000 - state.plr1.energy == expected);
        assert(obj->obj.number == OBJ_NBR_KEY && obj->obj.can_see == 8);
        assert(OBJ_ZONE(obj) == start_zone); /* Its own blast must leave the key collectible. */
        assert(attacks == before);
        for (int i = 0; i < NASTY_SHOT_SLOT_COUNT; i++) assert(OBJ_ZONE(&shots[i]) == -1);
    }
    puts("PASS: original 120/400 death blast strengths, blue key, and no post-death projectile volley");
}

int main(void)
{
    math_tables_init();
    check_turning();
    check_prowl_and_attack();
    check_level8_routes();
    return 0;
}
