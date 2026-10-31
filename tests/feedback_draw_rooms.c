/* Exercise the actual setup decoder and renderer membership contract. */
#include "level.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void word(uint8_t *p, unsigned value) { p[0] = value >> 8; p[1] = value; }

int main(void)
{
    LevelState level = {0};
    uint8_t data[256 * 48] = {0}, adds[256 * 4] = {0}, setup[523] = {'A','B','E','X',0,2,255,254};
    level.data = data; level.data_byte_count = sizeof(data); level.zone_adds = adds;
    for (int i = 0; i < 256; i++) { unsigned offset = i * 48; adds[i * 4 + 2] = offset >> 8; adds[i * 4 + 3] = offset; }
    assert(level_same_draw_room(&level, 1, 1)); assert(!level_same_draw_room(&level, 0, 1));
    assert(!level_same_draw_room(&level, -1, -1));
    level.num_zone_slots = 2; word(setup + 8, 2); word(setup + 10, 0); word(setup + 12, 0);
    int16_t exit_zone = -9;
    assert(!level_apply_setup(&level, setup, 14, 1, &exit_zone));
    assert(exit_zone == 1); assert(level.draw_group_count == 2); assert(level_same_draw_room(&level, 0, 1));
    LevelState before = level;
    word(setup + 10, 1); /* cycle 0->1->0 */
    assert(level_apply_setup(&level, setup, 14, 1, &exit_zone)); assert(!memcmp(&level, &before, sizeof(level)));
    word(setup + 10, 0); word(setup + 12, 2);
    assert(level_apply_setup(&level, setup, 14, 1, &exit_zone));
    word(setup + 12, 0); assert(level_apply_setup(&level, setup, 13, 1, &exit_zone));
    assert(level_apply_setup(&level, setup, 15, 1, &exit_zone));
    word(setup + 6, 2); assert(level_apply_setup(&level, setup, 14, 1, &exit_zone));
    word(setup + 6, 65535); setup[5] = 1;
    assert(!level_apply_setup(&level, setup, 8, 1, &exit_zone)); assert(exit_zone == -1); assert(level.draw_group_count == 0);
    assert(!level_same_draw_room(&level, 0, 1));
    setup[5] = 2; level.num_zone_slots = 256; word(setup + 8, 256);
    for (unsigned i = 0; i < 256; i++) word(setup + 10 + i * 2, i);
    assert(!level_apply_setup(&level, setup, 522, 1, &exit_zone)); assert(level.draw_group_count == 256);
    assert(level_same_draw_room(&level, 255, 255)); assert(!level_same_draw_room(&level, 254, 255));
    assert(level_apply_setup(&level, setup, 523, 1, &exit_zone));
    puts("PASS: native draw-room setup, membership, limits, legacy reset and atomic rejection");
    return 0;
}
