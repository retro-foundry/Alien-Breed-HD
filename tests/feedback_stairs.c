/* Replay authored stairs, including level_g's flame-spout stairway.
 * Run through tools/check_feedback.py; movement and hazard code are production C. */
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

typedef struct { int from, to, line; } Stair;
static const Stair stairs[] = {
    {123, 124, 554}, {124, 125, 558}, {125, 126, 562},
    {126, 127, 567}, {127, 128, 570},
    {132, 131, 592}, {131, 130, 588}, {130, 129, 584}
};

static void cross_stair(GameState *state, const Stair *stair, int player,
                        int crouch, int bobble, int pass)
{
    int from = stair->from, to = stair->to;
    uint8_t *room = level_get_zone_data_ptr(&state->level, from);
    uint8_t *dest = level_get_zone_data_ptr(&state->level, to);
    uint8_t *line = state->level.floor_lines + stair->line * 16;
    assert(level_connect_to_zone_index(&state->level, obj_w(line + 8)) == stair->to);
    int dx = obj_w(line + 4), dz = obj_w(line + 6), len = obj_w(line + 10);
    int mx = obj_w(line) + dx / 2, mz = obj_w(line + 2) + dz / 2;
    int x = mx + dz * 12 / len, z = mz - dx * 12 / len;
    int tx = mx - dz * 4 / len, tz = mz + dx * 4 / len;
    assert(level_find_zone_for_point(&state->level, x, z, from) == from);
    PlayerState *p = player == 1 ? &state->plr1 : &state->plr2;
    memset(p, 0, sizeof(*p));
    p->zone = from; p->roompt = (int32_t)(room - state->level.data);
    p->no_transition_back_roompt = -1;
    p->xoff = x * 65536; p->zoff = z * 65536;
    p->s_xoff = tx * 65536; p->s_zoff = tz * 65536;
    p->s_height = p->p_height = crouch ? PLAYER_CROUCHED : PLAYER_HEIGHT;
    p->s_yoff = p->p_yoff = obj_l(room + ZONE_OFF_FLOOR) - p->s_height;
    p->ducked = crouch; p->p_bobble = bobble;
    player_full_control(p, state, player);
    if (p->zone != (pass ? to : from)) {
        fprintf(stderr, "Level 7 stair failure: player=%d crouch=%d bob=%d zone=%d->%d got=%d pass=%d\n",
                player, crouch, bobble, from, to, p->zone, pass);
        abort();
    }
    assert(p->ducked == crouch);
    assert(p->s_tyoff == obj_l((pass ? dest : room) + ZONE_OFF_FLOOR) - p->s_height);
    if (pass) assert(p->s_yoff == p->s_tyoff);
}

static int check_limits(GameState *state, const Stair *stair)
{
    uint8_t *room = level_get_zone_data_ptr(&state->level, stair->from);
    uint8_t *dest = level_get_zone_data_ptr(&state->level, stair->to);
    assert(!level_zone_has_upper_layer(&state->level, stair->to));
    int32_t floor = obj_l(dest + ZONE_OFF_FLOOR), roof = obj_l(dest + ZONE_OFF_ROOF);
    int checked = 0;
    for (int player = 1; player <= 2; player++) {
        for (int crouch = 0; crouch < 2; crouch++) {
            for (int rise = 39; rise <= 41; rise++) {
                obj_sl(dest + ZONE_OFF_FLOOR, obj_l(room + ZONE_OFF_FLOOR) - rise * 256);
                obj_sl(dest + ZONE_OFF_ROOF, obj_l(dest + ZONE_OFF_FLOOR) - 100 * 256);
                cross_stair(state, stair, player, crouch, 0, rise < 40);
                checked++;
            }
        }
        obj_sl(dest + ZONE_OFF_FLOOR, floor);
        /* A gap between standing/crouched heights admits only a crouched player. */
        obj_sl(dest + ZONE_OFF_ROOF, floor - (PLAYER_HEIGHT + PLAYER_CROUCHED) / 2);
        cross_stair(state, stair, player, 0, 0, 0);
        cross_stair(state, stair, player, 1, 0, 1);
        /* A gap smaller than the crouched body still blocks movement. */
        obj_sl(dest + ZONE_OFF_ROOF, floor - PLAYER_CROUCHED + 2048);
        cross_stair(state, stair, player, 1, 0, 0);
        checked += 3;
    }
    obj_sl(dest + ZONE_OFF_FLOOR, floor); obj_sl(dest + ZONE_OFF_ROOF, roof);
    return checked;
}

static int check_flames(GameState *state, uint8_t *authored_objects)
{
    int checked = 0;
    for (int i = 0; ; i++) {
        GameObject original = *(GameObject *)(authored_objects + i * OBJECT_SIZE);
        if (OBJ_CID(&original) < 0) break;
        int zone = OBJ_ZONE(&original);
        if (original.obj.number != OBJ_NBR_GAS_PIPE || zone < 123 || zone > 132) continue;
        uint8_t *room = level_get_zone_data_ptr(&state->level, zone);
        int32_t floor = obj_l(room + ZONE_OFF_FLOOR);
        for (int player = 1; player <= 2; player++) {
            for (int crouch = 0; crouch < 2; crouch++) {
                GameObject emitter = original, targets[2] = {0};
                GameObject *shot = (GameObject *)state->level.nasty_shot_data;
                assert(OBJ_CID(&emitter) != 0);
                memcpy(state->level.object_points,
                       state->level.object_points + OBJ_CID(&emitter) * 8, 8);
                OBJ_SET_CID(&targets[0], 0); OBJ_SET_CID(&targets[1], -1);
                OBJ_SET_ZONE(&targets[0], zone);
                targets[0].obj.number = player == 1 ? OBJ_NBR_PLR1 : OBJ_NBR_PLR2;
                NASTY_LIVES(targets[0]) = PLAYER_MAX_ENERGY;
                int height = crouch ? PLAYER_CROUCHED : PLAYER_HEIGHT;
                obj_sw(targets[0].raw + 4, (int16_t)((floor - height / 2) >> 7));
                state->level.object_data = (uint8_t *)targets;
                for (int s = 0; s < NASTY_SHOT_SLOT_COUNT; s++)
                    OBJ_SET_ZONE((GameObject *)(state->level.nasty_shot_data + s * OBJECT_SIZE), -1);
                /* Defs.i ThirdTimer/FourthTimer are td+34/td+36. */
                OBJ_SET_TD_W(&emitter, 34, 0);
                OBJ_SET_TD_W(&emitter, 36, 0);
                OBJ_SET_TD_W(&emitter, 22, 5);
                state->temp_frames = 1;
                object_handle_gas_pipe(&emitter, state);
                assert(OBJ_ZONE(shot) == zone && SHOT_SIZE(*shot) == 3);
                /* Anims.s ItsAGasPipe: authored emitter Y minus 80. */
                assert(obj_w(shot->raw + 4) == obj_w(original.raw + 4) - 80);
                object_handle_bullet(shot, state);
                assert((NASTY_DAMAGE(targets[0]) != 0) == !crouch);
                checked++;
            }
        }
    }
    state->level.object_data = NULL;
    assert(checked > 0);
    return checked;
}

int main(void)
{
    math_tables_init();
    int crossings = 0, limits = 0, flames = 0;
    static GameState state;
    state.mode = MODE_SINGLE;
    size_t bytes;
    assert(!sb_load_file("amiga/data/levels/level_g/twolev.bin", &state.level.data, &bytes));
    state.level.data_byte_count = bytes;
    assert(!sb_load_file("amiga/data/levels/level_g/twolev.graph.bin", &state.level.graphics, &bytes));
    state.level.graphics_byte_count = bytes;
    assert(!level_parse(&state.level));
    uint8_t *authored_objects = state.level.object_data;
    state.level.object_data = NULL;
    for (unsigned i = 0; i < sizeof(stairs) / sizeof(stairs[0]); i++) {
        for (int player = 1; player <= 2; player++)
        for (int crouch = 0; crouch < 2; crouch++)
        for (int bob = 0; bob < 3; bob++) {
            cross_stair(&state, &stairs[i], player, crouch, bob * ANGLE_90, 1);
            crossings++;
        }
    }
    limits = check_limits(&state, &stairs[0]);
    flames = check_flames(&state, authored_objects);
    printf("PASS: %d authored stair crossings, %d step/headroom boundaries, %d flame-spout height cases; both players and stances\n",
           crossings, limits, flames);
    return 0;
}
