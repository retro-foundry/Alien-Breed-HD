#include "objects_dos.h"
#include "level.h"
#include <string.h>

enum { ROOM_COUNT = 256, OBJECT_COUNT = 256, MASK_WORDS = OBJECT_COUNT / 32 };
static uint32_t room_objects[ROOM_COUNT][MASK_WORDS];
static int16_t billboard_heads[ROOM_COUNT], billboard_next[OBJECT_COUNT];
static uint8_t all_objects[OBJECT_COUNT];
static unsigned all_count;
static int indexed_rooms;
static uint8_t shots[2][NASTY_SHOT_SLOT_COUNT + PLAYER_SHOT_SLOT_COUNT];
static unsigned shot_count[2];

static int read_word(const uint8_t *p) { return (int16_t)((p[0] << 8) | p[1]); }
static int draw_group(const LevelState *level, int zone)
{ return zone >= 0 && zone < level->draw_group_count ? level->draw_groups[zone] : zone; }

void dos_objects_prepare(const LevelState *level, const ObjRotatedPoint *points, int32_t near_z)
{
    memset(room_objects, 0, sizeof(room_objects));
    memset(billboard_heads, 0xff, sizeof(billboard_heads));
    all_count = 0;
    indexed_rooms = 1;
    int point_count = level->num_object_points;
    if (point_count < 0) point_count = 0;
    if (point_count > MAX_OBJ_POINTS) point_count = MAX_OBJ_POINTS;
    if (!level->object_points) point_count = 0;
    for (unsigned i = 0; level->object_data && i < OBJECT_COUNT; i++) {
        const uint8_t *obj = level->object_data + i * OBJECT_SIZE;
        int point = read_word(obj), zone = read_word(obj + 12);
        if (point < 0) break;
        if ((unsigned)point >= (unsigned)point_count || zone < 0) continue;
        int polygon = obj[6] == OBJ_3D_SPRITE;
        if (points[point].z <= (polygon ? 0 : near_z)) continue;
        all_objects[all_count++] = (uint8_t)i;
        int group = draw_group(level, zone);
        if ((unsigned)zone >= ROOM_COUNT || (unsigned)group >= ROOM_COUNT) {
            indexed_rooms = 0;
            continue;
        }
        room_objects[group][i / 32] |= 1u << (i % 32);
        if (!polygon) {
            billboard_next[i] = billboard_heads[zone];
            billboard_heads[zone] = (int16_t)i;
        }
    }
    for (unsigned pool = 0; pool < 2; pool++) {
        const uint8_t *data = pool ? level->player_shot_data : level->nasty_shot_data;
        unsigned slots = pool ? PLAYER_SHOT_SLOT_COUNT : NASTY_SHOT_SLOT_COUNT;
        shot_count[pool] = 0;
        for (unsigned i = 0; data && i < slots; i++) {
            const uint8_t *obj = data + i * OBJECT_SIZE;
            int point = read_word(obj);
            if (read_word(obj + 12) < 0 || (int8_t)obj[16] != OBJ_NBR_BULLET ||
                point < 0 || (unsigned)point >= (unsigned)point_count || points[point].z <= near_z) continue;
            shots[pool][shot_count[pool]++] = (uint8_t)i;
        }
    }
}

unsigned dos_objects_for_room(const LevelState *level, int zone,
                               const int16_t *adjacent, unsigned adjacent_count, uint8_t out[256])
{
    int group = draw_group(level, zone);
    if (!indexed_rooms || (unsigned)group >= ROOM_COUNT) {
        memcpy(out, all_objects, all_count);
        return all_count;
    }
    uint32_t selected[MASK_WORDS];
    memcpy(selected, room_objects[group], sizeof(selected));
    for (unsigned i = 0; i < adjacent_count; i++) {
        if ((unsigned)adjacent[i] >= ROOM_COUNT) continue;
        for (int slot = billboard_heads[adjacent[i]]; slot >= 0; slot = billboard_next[slot])
            selected[slot / 32] |= 1u << (slot % 32);
    }
    unsigned count = 0;
    for (unsigned word = 0; word < MASK_WORDS; word++) {
        uint32_t bits = selected[word];
        while (bits) {
            out[count++] = (uint8_t)(word * 32 + __builtin_ctz(bits));
            bits &= bits - 1;
        }
    }
    return count;
}

const uint8_t *dos_objects_shots(unsigned pool, unsigned *count)
{
    *count = shot_count[pool];
    return shots[pool];
}
