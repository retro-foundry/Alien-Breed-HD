#include "renderer_objects.h"
#include "level.h"
#include <string.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

static unsigned first_set_bit(uint32_t bits)
{
#ifdef _MSC_VER
    unsigned long index;
    _BitScanForward(&index, bits);
    return (unsigned)index;
#else
    return (unsigned)__builtin_ctz(bits);
#endif
}

enum { ROOM_COUNT = 256, OBJECT_COUNT = 256, MASK_WORDS = OBJECT_COUNT / 32,
       ROOM_WORDS = (RENDERER_OBJECT_SLOTS + 31) / 32 };
static uint32_t room_objects[ROOM_COUNT][ROOM_WORDS];
static int16_t billboard_heads[ROOM_COUNT], billboard_next[OBJECT_COUNT];
static uint8_t all_objects[OBJECT_COUNT];
static unsigned all_count;
static int indexed_rooms;
enum { SHOT_COUNT = NASTY_SHOT_SLOT_COUNT + PLAYER_SHOT_SLOT_COUNT };
_Static_assert(SHOT_COUNT <= 128, "DOS signed-byte shot links");
static uint8_t shots[SHOT_COUNT];
static unsigned shot_count[2];
static int8_t shot_group_heads[ROOM_COUNT], shot_zone_heads[ROOM_COUNT];
static int8_t shot_group_next[SHOT_COUNT], shot_zone_next[SHOT_COUNT];
static int indexed_shots;

static int read_word(const uint8_t *p) { return (int16_t)((p[0] << 8) | p[1]); }
static int draw_group(const LevelState *level, int zone)
{ return zone >= 0 && zone < level->draw_group_count ? level->draw_groups[zone] : zone; }

void renderer_objects_add_room(const LevelState *level, int zone, unsigned slot)
{
    int group = draw_group(level, zone);
    if ((unsigned)group < ROOM_COUNT && slot < RENDERER_OBJECT_SLOTS)
        room_objects[group][slot / 32] |= 1u << (slot % 32);
}

int renderer_objects_in_room(const LevelState *level, int zone, unsigned slot)
{
    int group = draw_group(level, zone);
    return (unsigned)group < ROOM_COUNT && slot < RENDERER_OBJECT_SLOTS &&
        ((room_objects[group][slot / 32] >> (slot % 32)) & 1u);
}

int renderer_objects_room_is_exact(const LevelState *level, int zone)
{
    return indexed_rooms && indexed_shots && (unsigned)draw_group(level, zone) < ROOM_COUNT;
}

void renderer_objects_prepare(const LevelState *level, const ObjRotatedPoint *points, int32_t near_z)
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
    memset(shot_group_heads, -1, sizeof(shot_group_heads));
    memset(shot_zone_heads, -1, sizeof(shot_zone_heads));
    indexed_shots = 1;
    for (unsigned pool = 0; pool < 2; pool++) {
        const uint8_t *data = pool ? level->player_shot_data : level->nasty_shot_data;
        unsigned slots = pool ? PLAYER_SHOT_SLOT_COUNT : NASTY_SHOT_SLOT_COUNT;
        shot_count[pool] = 0;
        for (unsigned i = 0; data && i < slots; i++) {
            const uint8_t *obj = data + i * OBJECT_SIZE;
            int point = read_word(obj), zone = read_word(obj + 12);
            if (zone < 0 || (int8_t)obj[16] != OBJ_NBR_BULLET ||
                point < 0 || (unsigned)point >= (unsigned)point_count || points[point].z <= near_z) continue;
            unsigned base = pool ? NASTY_SHOT_SLOT_COUNT : 0;
            shots[base + shot_count[pool]++] = (uint8_t)i;
            int group = draw_group(level, zone);
            if ((unsigned)zone >= ROOM_COUNT || (unsigned)group >= ROOM_COUNT) {
                indexed_shots = 0;
                continue;
            }
            unsigned slot = base + i;
            shot_group_next[slot] = shot_group_heads[group];
            shot_group_heads[group] = (int8_t)slot;
            shot_zone_next[slot] = shot_zone_heads[zone];
            shot_zone_heads[zone] = (int8_t)slot;
        }
    }
}

unsigned renderer_objects_for_room(const LevelState *level, int zone,
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
            out[count++] = (uint8_t)(word * 32 + first_set_bit(bits));
            bits &= bits - 1;
        }
    }
    return count;
}

const uint8_t *renderer_objects_shots(unsigned pool, unsigned *count)
{
    *count = shot_count[pool];
    return shots + (pool ? NASTY_SHOT_SLOT_COUNT : 0);
}

void renderer_objects_shots_for_room(const LevelState *level, int zone,
                                 const int16_t *adjacent, unsigned adjacent_count,
                                 unsigned counts[2], uint8_t out[NASTY_SHOT_SLOT_COUNT + PLAYER_SHOT_SLOT_COUNT])
{
    if (!(shot_count[0] | shot_count[1])) {
        counts[0] = counts[1] = 0;
        return;
    }
    int group = draw_group(level, zone);
    if (!indexed_shots || (unsigned)group >= ROOM_COUNT) {
        counts[0] = shot_count[0]; counts[1] = shot_count[1];
        memcpy(out, shots, counts[0]);
        memcpy(out + counts[0], shots + NASTY_SHOT_SLOT_COUNT, counts[1]);
        return;
    }
    /* Own-room aliases and raw adjacent-zone membership are different tests.
     * Union their lists, then enumerate slots in original ObjDraw order. */
    uint32_t selected[(SHOT_COUNT + 31) / 32] = {0};
    memcpy(selected, room_objects[group] + MASK_WORDS, sizeof(selected));
    if (SHOT_COUNT % 32)
        selected[SHOT_COUNT / 32] &= (1u << (SHOT_COUNT % 32)) - 1u;
    for (int slot = shot_group_heads[group]; slot >= 0; slot = shot_group_next[slot])
        selected[slot / 32] |= 1u << (slot % 32);
    for (unsigned i = 0; i < adjacent_count; i++) {
        if ((unsigned)adjacent[i] >= ROOM_COUNT) continue;
        for (int slot = shot_zone_heads[adjacent[i]]; slot >= 0; slot = shot_zone_next[slot])
            selected[slot / 32] |= 1u << (slot % 32);
    }
    unsigned count = 0; counts[0] = counts[1] = 0;
    for (unsigned word = 0; word < (SHOT_COUNT + 31) / 32; word++) {
        uint32_t bits = selected[word];
        while (bits) {
            unsigned slot = word * 32 + first_set_bit(bits);
            unsigned pool = slot >= NASTY_SHOT_SLOT_COUNT;
            out[count++] = (uint8_t)(slot - (pool ? NASTY_SHOT_SLOT_COUNT : 0));
            counts[pool]++; bits &= bits - 1;
        }
    }
}
