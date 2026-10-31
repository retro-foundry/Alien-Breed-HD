/* Exercise PLR1/2_Control with level 9/15's authored teleporter records. */
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

static int check_level(int map)
{
    static GameState state;
    memset(&state, 0, sizeof(state));
    size_t bytes;
    char path[128];
    snprintf(path, sizeof(path), "amiga/data/levels/level_%c/twolev.bin", 'a' + map);
    assert(!sb_load_file(path, &state.level.data, &bytes));
    state.level.data_byte_count = bytes;
    snprintf(path, sizeof(path), "amiga/data/levels/level_%c/twolev.graph.bin", 'a' + map);
    assert(!sb_load_file(path, &state.level.graphics, &bytes));
    state.level.graphics_byte_count = bytes;
    assert(!level_parse(&state.level));
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
        for (int player = 1; player <= 2; player++) {
            for (int crouch = 0; crouch < 2; crouch++) {
                for (int delta = -8; delta <= 8; delta += 8) {
                    for (int occupant = 0; occupant < 5; occupant++) {
                        PlayerState *p = player == 1 ? &state.plr1 : &state.plr2;
                        memset(p, 0, sizeof(*p));
                        p->zone = z; p->roompt = room - state.level.data;
                        p->xoff = x * 65536; p->s_xoff = (x + delta) * 65536;
                        p->zoff = p->s_zoff = zz * 65536;
                        p->s_height = p->p_height = (crouch ? 48 : 80) * 256;
                        p->p_yoff = p->s_yoff = obj_l(room + ZONE_OFF_FLOOR) - p->s_height;
                        p->ducked = crouch;
                        GameObject blockers[2] = {0};
                        uint8_t points[8] = {0};
                        if (occupant) {
                            /* AB3DI.s player teleport mask has bits 0..17;
                             * ObjectMove.s enemy CheckTeleport has bits 0..18. */
                            OBJ_SET_CID(&blockers[0], 0); OBJ_SET_ZONE(&blockers[0], dest);
                            blockers[0].obj.number = occupant == 2 ? OBJ_NBR_TOUGH_MARINE :
                                occupant == 3 ? OBJ_NBR_FLAME_MARINE : OBJ_NBR_ALIEN;
                            NASTY_LIVES(blockers[0]) = 12;
                            obj_sw(blockers[0].raw + 4, obj_l(dest_room + ZONE_OFF_FLOOR) / 128 - 40);
                            OBJ_SET_CID(&blockers[1], -1);
                            obj_sw(points, dest_x); obj_sw(points + 4, dest_z);
                            state.level.object_data = (uint8_t *)blockers;
                            state.level.object_points = points;
                        }
                        if (occupant == 2) {
                            /* Enemy CheckTeleport must retain its wider mask. */
                            MoveContext enemy;
                            move_context_init(&enemy);
                            enemy.oldx = x; enemy.oldz = zz;
                            enemy.newx = x + delta; enemy.newz = zz;
                            enemy.newy = p->p_yoff;
                            enemy.thing_height = p->p_height;
                            enemy.objroom = room;
                            enemy.collide_flags = 0x55AA;
                            assert(!check_teleport(&enemy, &state.level, z, 0x7FFFFu));
                            assert(enemy.collide_flags == 0x55AA);
                            assert(enemy.newx == x + delta && enemy.newz == zz);
                            assert(enemy.objroom == room && enemy.newy == p->p_yoff);
                            /* An enemy using this same pad must not change its
                             * record or disable the player's subsequent use. */
                            uint8_t teleport_record[6];
                            memcpy(teleport_record, room + ZONE_OFF_TEL_ZONE, 6);
                            OBJ_SET_ZONE(&blockers[0], z);
                            obj_sw(points, x); obj_sw(points + 4, zz);
                            enemy.coll_id = 0;
                            assert(check_teleport(&enemy, &state.level, z, 0x7FFFFu));
                            assert(enemy.newx == dest_x && enemy.newz == dest_z);
                            assert(enemy.objroom == dest_room && enemy.collide_flags == 0x55AA);
                            assert(!memcmp(teleport_record, room + ZONE_OFF_TEL_ZONE, 6));
                            OBJ_SET_ZONE(&blockers[0], dest);
                            obj_sw(points, enemy.newx); obj_sw(points + 4, enemy.newz);
                        }
                        player_full_control(p, &state, player);
                        if (occupant == 1 || occupant == 4) {
                            assert(p->zone == z);
                            assert((p->xoff >> 16) == x + delta && (p->zoff >> 16) == zz);
                            /* Continued attempts must not corrupt the pad. */
                            for (int retry = 0; retry < 3; retry++) {
                                player_full_control(p, &state, player);
                                assert(p->zone == z);
                            }
                            if (occupant == 4) NASTY_LIVES(blockers[0]) = 0;
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
                        assert((p->xoff >> 16) == dest_x && (p->zoff >> 16) == dest_z);
                        assert(p->s_tyoff == obj_l(dest_room + ZONE_OFF_FLOOR) - p->s_height);
                        state.level.object_data = authored_objects;
                        state.level.object_points = authored_points;
                        cases++;
                    }
                }
            }
        }
    }
    assert(pads == 4);
    printf("PASS: %d level %d teleporter cases: four authored pads, both players, crouching, motion, occupied exits, live/dead retry, original player masks\n", cases, map + 1);
    return cases;
}

int main(void)
{
    math_tables_init();
    assert(check_level(8) == 240);
    assert(check_level(14) == 240);
    return 0;
}
