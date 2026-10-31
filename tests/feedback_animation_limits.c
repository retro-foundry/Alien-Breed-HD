/* Anims.s DoWaterAnims walks every entry through the negative terminator.
 * A full group must not leave the cursor inside the next group's header. */
#include "../src/level.c"
#include "objects.h"
#include <assert.h>

static uint8_t water[4096], graphics[256], data[64], zone_adds[4];
static GameState state;

static void check_groups(int entries)
{
    memset(water, 255, sizeof(water)); memset(graphics, 0, sizeof(graphics));
    memset(&state, 0, sizeof(state));
    size_t headers[21], positions[21], cursor = 0;
    for (int group = 0; group < 21; group++) {
        headers[group] = cursor;
        write_long_be(water + cursor, 0);
        write_long_be(water + cursor + 4, 65536);
        write_long_be(water + cursor + 8, group * 1024 + 512);
        write_word_be(water + cursor + 12, 64);
        cursor += 14; positions[group] = cursor;
        int count = group ? 1 : entries;
        for (int i = 0; i < count; i++) {
            write_word_be(water + cursor, 200); /* file room ID -> runtime 0 */
            write_long_be(water + cursor + 2, 32 + group * 8);
            cursor += 6;
        }
        write_word_be(water + cursor, -1); cursor += 2;
    }
    assert(cursor <= sizeof(water));
    write_long_be(zone_adds, 54); write_word_be(data + 54, 200);
    normalize_amiga_water_list(water, data, zone_adds, 1);
    for (int group = 0; group < 21; group++) {
        int count = group ? 1 : entries;
        for (int i = 0; i < count; i++) {
            assert(read_word(water + positions[group] + i * 6) == 0);
            assert(read_long(water + positions[group] + i * 6 + 2) == 32 + group * 8);
        }
        assert(read_word(water + positions[group] + count * 6) == -1);
    }
    state.level.water_list = water; state.level.graphics = graphics;
    state.temp_frames = 1;
    do_water_anims(&state);
    for (int group = 0; group < 21; group++) {
        int expected = group * 1024 + 576;
        assert(read_long(water + headers[group] + 8) == expected);
        if (group || entries) assert(read_word(graphics + 34 + group * 8) == expected / 64);
    }
}

int main(void)
{
    check_groups(0); check_groups(1); check_groups(127);
    check_groups(128); check_groups(256);
    puts("Water group normalization, animation and following-group boundaries passed.");
    return 0;
}
