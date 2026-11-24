/* Exercise PLR1/2_Control with authored transporter rooms. */
#include "../src/player.c"
#include "sb_decompress.h"
#include <assert.h>
#undef printf

GameState g_state;
RendererState g_renderer;
int renderer_get_width(void) { return 320; }
int renderer_get_height(void) { return 200; }
void audio_play_sample(int id, int volume) { (void)id; (void)volume; }
bool input_gamepad_fire_held(void) { return false; }
bool input_consume_key_press(uint8_t key) { (void)key; return false; }
bool input_gamepad_duck_toggle_requested(void) { return false; }
int16_t input_consume_gamepad_weapon_cycle_steps(void) { return 0; }
void input_read_mouse(MouseState *mouse) { memset(mouse, 0, sizeof(*mouse)); }

typedef enum {
    CLEAR_EXIT,
    ALIEN_AT_LANDING_HEIGHT,
    ALIEN_AT_SOURCE_HEIGHT,
    TOUGH_MARINE,
    FLAME_MARINE,
    ALIEN_KILLED_ON_RETRY,
} ArrivalOccupant;

static void load_teleport_level(LevelState *level, int map)
{
    size_t bytes;
    char path[128];
    snprintf(path, sizeof(path), "amiga/data/levels/level_%c/twolev.bin", 'a' + map);
    assert(!sb_load_file(path, &level->data, &bytes));
    level->data_byte_count = bytes;
    snprintf(path, sizeof(path), "amiga/data/levels/level_%c/twolev.graph.bin", 'a' + map);
    assert(!sb_load_file(path, &level->graphics, &bytes));
    level->graphics_byte_count = bytes;
    assert(!level_parse(level));
}

static int check_level(int map)
{
    static GameState state;
    memset(&state, 0, sizeof(state));
    load_teleport_level(&state.level, map);
    uint8_t *authored_objects = state.level.object_data;
    uint8_t *authored_points = state.level.object_points;
    int pads = 0, cases = 0;
    for (int z = 0; z < level_zone_slot_count(&state.level); z++) {
        uint8_t *room = level_get_zone_data_ptr(&state.level, z);
        int dest = obj_w(room + ZONE_OFF_TEL_ZONE);
        if (dest < 0) continue;
        pads++;
        int sum_x = 0, sum_z = 0, count = 0;
        uint8_t *list = room + obj_w(room + ZONE_OFF_EXIT_LIST);
        for (int i = 0; i < 128; i++) {
            int id = obj_w(list + i * 2);
            if (id < 0) break;
            uint8_t *line = state.level.floor_lines + id * 16;
            sum_x += obj_w(line); sum_z += obj_w(line + 2); count++;
        }
        assert(count > 0);
        int x = sum_x / count, zz = sum_z / count;
        int dest_x = obj_w(room + ZONE_OFF_TEL_X);
        int dest_z = obj_w(room + ZONE_OFF_TEL_Z);
        uint8_t *dest_room = level_get_zone_data_ptr(&state.level, dest);
        int32_t source_floor = obj_l(room + ZONE_OFF_FLOOR);
        int32_t dest_floor = obj_l(dest_room + ZONE_OFF_FLOOR);
        /* These level 9 pads land on opposite sides of the source height.
         * An alien standing at either landing overlaps the arrival height
         * but is outside the original player's probe at the source height. */
        bool check_landing_height = map == 8 && (z == 42 || z == 134);
        for (int player = 1; player <= 2; player++) {
            for (int crouch = 0; crouch < 2; crouch++) {
                for (int delta = -8; delta <= 8; delta += 8) {
                    for (ArrivalOccupant occupant = CLEAR_EXIT; occupant <= ALIEN_KILLED_ON_RETRY; occupant++) {
                        if (occupant == ALIEN_AT_LANDING_HEIGHT && !check_landing_height) continue;
                        PlayerState *p = player == 1 ? &state.plr1 : &state.plr2;
                        memset(p, 0, sizeof(*p));
                        p->zone = z; p->roompt = room - state.level.data;
                        p->xoff = x * 65536;
                        p->s_xoff = (x + delta) * 65536 + 137;
                        p->zoff = p->s_zoff = zz * 65536 + 412;
                        p->s_height = p->p_height = crouch ? PLAYER_CROUCHED : PLAYER_HEIGHT;
                        p->p_yoff = p->s_yoff = source_floor - p->s_height;
                        p->ducked = crouch;
                        GameObject blockers[2] = {0};
                        uint8_t points[8] = {0};
                        if (occupant != CLEAR_EXIT) {
                            /* AB3DI.s player teleport mask has bits 0..17;
                             * ObjectMove.s enemy CheckTeleport has bits 0..18. */
                            OBJ_SET_CID(&blockers[0], 0); OBJ_SET_ZONE(&blockers[0], dest);
                            blockers[0].obj.number = occupant == TOUGH_MARINE ? OBJ_NBR_TOUGH_MARINE :
                                occupant == FLAME_MARINE ? OBJ_NBR_FLAME_MARINE : OBJ_NBR_ALIEN;
                            NASTY_LIVES(blockers[0]) = 12;
                            int32_t blocker_floor = occupant == TOUGH_MARINE || occupant == ALIEN_AT_LANDING_HEIGHT
                                ? dest_floor : source_floor;
                            obj_sw(blockers[0].raw + 4, blocker_floor / 128 - 40);
                            OBJ_SET_CID(&blockers[1], -1);
                            obj_sw(points, dest_x); obj_sw(points + 4, dest_z);
                            state.level.object_data = (uint8_t *)blockers;
                            state.level.object_points = points;
                        }
                        if (occupant == TOUGH_MARINE || occupant == ALIEN_AT_LANDING_HEIGHT) {
                            /* Enemy CheckTeleport probes the landing height with its wider mask. */
                            MoveContext enemy;
                            move_context_init(&enemy);
                            enemy.oldx = x; enemy.oldz = zz;
                            enemy.newx = x + delta; enemy.newz = zz;
                            enemy.newy = p->p_yoff;
                            enemy.thing_height = p->p_height;
                            enemy.objroom = room;
                            enemy.collide_flags = 0x55AA;
                            assert(!check_teleport(&enemy, &state.level, z, TELEPORT_ENEMY));
                            assert(enemy.collide_flags == 0x55AA);
                            assert(enemy.newx == x + delta && enemy.newz == zz);
                            assert(enemy.objroom == room && enemy.newy == p->p_yoff);
                            if (occupant == TOUGH_MARINE) {
                                /* Reuse the pad after an enemy has traveled through it. */
                                uint8_t teleport_record[6];
                                memcpy(teleport_record, room + ZONE_OFF_TEL_ZONE, 6);
                                OBJ_SET_ZONE(&blockers[0], z);
                                obj_sw(points, x); obj_sw(points + 4, zz);
                                enemy.coll_id = 0;
                                assert(check_teleport(&enemy, &state.level, z, TELEPORT_ENEMY));
                                assert(enemy.newx == dest_x && enemy.newz == dest_z);
                                assert(enemy.objroom == dest_room && enemy.collide_flags == 0x55AA);
                                assert(!memcmp(teleport_record, room + ZONE_OFF_TEL_ZONE, 6));
                                OBJ_SET_ZONE(&blockers[0], dest);
                                obj_sw(points, enemy.newx); obj_sw(points + 4, enemy.newz);
                                /* Check the player mask with a blocker at the player's probe height. */
                                obj_sw(blockers[0].raw + 4, source_floor / 128 - 40);
                            }
                        }
                        player_full_control(p, &state, player);
                        if (occupant == ALIEN_AT_SOURCE_HEIGHT || occupant == ALIEN_KILLED_ON_RETRY) {
                            assert(p->zone == z);
                            assert((p->xoff >> 16) == x + delta && (p->zoff >> 16) == zz);
                            /* Continued attempts must not corrupt the pad. */
                            for (int retry = 0; retry < 3; retry++) {
                                player_full_control(p, &state, player);
                                assert(p->zone == z);
                            }
                            if (occupant == ALIEN_KILLED_ON_RETRY) NASTY_LIVES(blockers[0]) = 0;
                            else {
                                obj_sw(points, dest_x + 500);
                                obj_sw(points + 4, dest_z + 500);
                            }
                            player_full_control(p, &state, player);
                        }
                        if (p->zone != dest) {
                            fprintf(stderr, "Teleport failure: level=%d pad=%d dest=%d player=%d occupant=%d actual=%d\n",
                                    map + 1, z, dest, player, occupant, p->zone);
                            abort();
                        }
                        assert(p->xoff == dest_x * 65536 + 137 && p->zoff == dest_z * 65536 + 412);
                        assert(p->oldxoff == p->xoff && p->oldzoff == p->zoff);
                        assert(p->s_xoff == p->xoff && p->s_zoff == p->zoff);
                        assert(p->s_yoff == dest_floor - p->s_height);
                        assert(p->s_tyoff == dest_floor - p->s_height);
                        assert(p->roompt == dest_room - state.level.data);
                        assert(p->list_of_graph_rooms == p->roompt + ZONE_OFF_LIST_OF_GRAPH);
                        state.level.object_data = authored_objects;
                        state.level.object_points = authored_points;
                        cases++;
                    }
                }
            }
        }
    }
    assert(pads == 4);
    printf("PASS: %d level %d teleporter cases: four authored pads, both players, crouching, motion, height and mask differences, blocked retries\n", cases, map + 1);
    return cases;
}

static void check_level_three_routes(void)
{
    static const struct {
        int pad, landing, approach, start_x, start_z, angle, enemy_cid;
    } routes[] = {
        /* Level 3's marine and flying alien already inhabit these landing rooms. */
        { 67, 69, 65, 1600, -4992, ANGLE_180, 35 },
        { 67, 69, 66, 1408, -5184, ANGLE_90, 35 },
        { 67, 69, 68, 1792, -5184, ANGLE_270, 35 },
        { 159, 127, 158, 2688, -832, 0, 60 },
    };
    static GameState state;
    load_teleport_level(&state.level, 2);
    state.plr1_control.keys = state.plr2_control.keys = true;
    int cases = 0;

    for (size_t route = 0; route < sizeof(routes) / sizeof(routes[0]); route++) {
        int pad = routes[route].pad, landing = routes[route].landing;
        int approach = routes[route].approach;
        int start_x = routes[route].start_x, start_z = routes[route].start_z;
        const uint8_t *source = level_get_zone_data_ptr(&state.level, pad);
        const uint8_t *entrance = level_get_zone_data_ptr(&state.level, approach);
        assert(obj_w(source + ZONE_OFF_TEL_ZONE) == landing);
        assert(level_find_zone_for_point(&state.level, start_x, start_z, approach) == approach);
        int x = obj_w(source + ZONE_OFF_TEL_X), z = obj_w(source + ZONE_OFF_TEL_Z);
        int cid = routes[route].enemy_cid;
        const GameObject *enemy = (const GameObject *)state.level.object_data;
        while (OBJ_CID(enemy) >= 0 && OBJ_CID(enemy) != cid) enemy++;
        assert(OBJ_CID(enemy) == cid && OBJ_ZONE(enemy) == landing && NASTY_LIVES(*enemy) > 0);
        uint8_t *point = state.level.object_points + cid * 8;
        int16_t authored_x = obj_w(point), authored_z = obj_w(point + 4);

        for (int occupied = 0; occupied < 2; occupied++) {
            obj_sw(point, occupied ? x : authored_x);
            obj_sw(point + 4, occupied ? z : authored_z);
            for (int player = 1; player <= 2; player++) {
                PlayerState *p = player == 1 ? &state.plr1 : &state.plr2;
                void (*update)(GameState *) = player == 1 ? player1_control : player2_control;
                for (int crouched = 0; crouched < 2; crouched++) {
                    for (int frames = 1; frames <= 15; frames++) {
                        memset(p, 0, sizeof(*p));
                        p->zone = approach;
                        p->roompt = (int32_t)(entrance - state.level.data);
                        p->no_transition_back_roompt = -1;
                        p->s_angpos = routes[route].angle;
                        p->xoff = p->s_xoff = start_x * 65536;
                        p->zoff = p->s_zoff = start_z * 65536;
                        p->ducked = crouched;
                        p->s_targheight = p->s_height = crouched ? PLAYER_CROUCHED : PLAYER_HEIGHT;
                        p->s_tyoff = p->s_yoff = obj_l(entrance + ZONE_OFF_FLOOR) - p->s_height;
                        state.temp_frames = frames;
                        state.key_map[default_keys.forward] = 1;
                        for (int tick = 0; tick < 512 && p->zone == approach; tick++) update(&state);
                        assert(p->zone == pad);
                        update(&state);
                        if (p->zone != landing) {
                            fprintf(stderr, "Level 3: %d -> %d -> %d stopped in %d (player %d, crouched %d, frames %d, occupied %d)\n",
                                    approach, pad, landing, p->zone, player, crouched, frames, occupied);
                            abort();
                        }
                        assert((p->xoff >> 16) == x && (p->zoff >> 16) == z);
                        cases++;
                    }
                }
            }
        }
        obj_sw(point, authored_x);
        obj_sw(point + 4, authored_z);
    }
    assert(cases == 480);
    printf("PASS: %d level 3 walk-on cases: both pads, four entrances, authored enemies, 1-15 VBlanks\n", cases);
}

int main(void)
{
    math_tables_init();
    check_level_three_routes();
    assert(check_level(8) == 264);
    assert(check_level(14) == 240);
    return 0;
}
