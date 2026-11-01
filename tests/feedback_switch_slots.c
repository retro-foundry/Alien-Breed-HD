/* Anims.s SwitchRoutine: inactive off slots have no wall write. Native slot
 * condition bits, active toggles and the backtoend timer path remain intact. */
#include "objects.h"
#include "game_data.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int16_t game_conditions;
static uint8_t slots[8 * 14], points[8];
static struct { uint8_t before[8], data[16], after[8]; } arena;
static GameState state;
static unsigned int clock_ms;
static int sounds;

void audio_play_sample(int sample, int volume)
{
    assert(sample == 10 && volume == 50);
    sounds++;
}

static void word(uint8_t *bytes, int value) { bytes[0] = (uint8_t)(value >> 8); bytes[1] = (uint8_t)value; }
static int read_word(const uint8_t *bytes) { return (bytes[0] << 8) | bytes[1]; }
static void long_word(uint8_t *bytes, int32_t value)
{
    uint32_t bits = (uint32_t)value;
    bytes[0] = (uint8_t)(bits >> 24); bytes[1] = (uint8_t)(bits >> 16);
    bytes[2] = (uint8_t)(bits >> 8); bytes[3] = (uint8_t)bits;
}
static void reset(void)
{
    memset(&state, 0, sizeof(state)); memset(slots, 0, sizeof(slots)); memset(points, 0, sizeof(points));
    memset(&arena, 0xa5, sizeof(arena)); sounds = 0; game_conditions = 3;
    for (int index = 0; index < 8; index++) word(slots + index * 14, -1);
    state.level.switch_data = slots; state.level.graphics = arena.data;
    state.level.graphics_byte_count = sizeof(arena.data); state.level.points = points;
}
static void step(void)
{
    state.current_ticks_ms = clock_ms += 20; switch_routine(&state);
    for (size_t i = 0; i < sizeof(arena.before); i++) assert(arena.before[i] == 0xa5 && arena.after[i] == 0xa5);
}

static void inactive_slots(void)
{
    const int32_t offsets[] = {0, -1, INT32_MAX, 11, 16};
    for (size_t index = 0; index < sizeof(offsets) / sizeof(offsets[0]); index++) {
        reset();
        for (int slot = 0; slot < 8; slot++) long_word(slots + slot * 14 + 6, offsets[index]);
        uint8_t before[sizeof(arena.data)], records[sizeof(slots)];
        memcpy(before, arena.data, sizeof(before)); memcpy(records, slots, sizeof(records));
        step();
        assert(memcmp(arena.data, before, sizeof(before)) == 0);
        assert(memcmp(slots, records, sizeof(records)) == 0);
        assert(game_conditions == 3 && sounds == 0);
    }
}
static void active_slot(int32_t offset, int valid)
{
    reset();
    const int slot = 3, mask = 1 << (slot + 4);
    uint8_t *record = slots + slot * 14;
    word(record, 0); long_word(record + 6, offset);
    uint8_t before[sizeof(arena.data)]; memcpy(before, arena.data, sizeof(before));
    state.plr1.p_spctap = 1; step();
    assert((uint16_t)game_conditions == (3 | mask)); assert(record[10] == 255 && sounds == 1);
    if (valid) {
        assert(read_word(arena.data + offset) == ((0xa5a5 & 0x7c) | 2));
        assert(read_word(arena.data + offset + 4) == 11);
        assert(memcmp(arena.data, before, (size_t)offset) == 0);
    } else assert(memcmp(arena.data, before, sizeof(before)) == 0);
    state.plr1.p_spctap = 0; step(); assert(sounds == 1);
    state.plr1.p_spctap = 1; step();
    assert(game_conditions == 3 && record[10] == 0 && sounds == 2);
}
static void timed_reset(int active, int32_t offset, int valid)
{
    reset(); uint8_t *record = slots + 14;
    word(record, active ? 0 : -1); long_word(record + 6, offset);
    record[2] = 1; record[3] = 2; record[10] = 255; game_conditions = 3 | (1 << 5);
    uint8_t before[sizeof(arena.data)]; memcpy(before, arena.data, sizeof(before));
    step(); assert(record[3] == 0 && record[10] == 0 && game_conditions == 3 && sounds == 1);
    if (valid) assert(read_word(arena.data + offset) == (0xa5a5 & 0x7c));
    else assert(memcmp(arena.data, before, sizeof(before)) == 0);
}
static void short_buffers(void)
{
    for (size_t size = 0; size < 6; size++) {
        reset(); word(slots, 0); state.level.graphics_byte_count = size;
        state.plr1.p_spctap = 1;
        uint8_t before[sizeof(arena.data)]; memcpy(before, arena.data, sizeof(before));
        step();
        assert(memcmp(arena.data, before, sizeof(before)) == 0);
        assert(game_conditions == (3 | (1 << 4)) && slots[10] == 255 && sounds == 1);
    }
}
int main(void)
{
    inactive_slots();
    active_slot(10, 1); active_slot(11, 0); active_slot(16, 0); active_slot(-1, 0); active_slot(INT32_MAX, 0);
    timed_reset(1, 10, 1); timed_reset(0, 10, 1); timed_reset(0, INT32_MAX, 0);
    short_buffers();
    puts("Native switch slots, bounded writes, condition bits and timer paths passed.");
    return 0;
}
