/* Exercise the native decoder with original level data and a controller
 * assigned to the final existing room slot. The data header stores its index,
 * while the graphics prologue describes the full table capacity. */
#include "../src/level.c"
#include "sb_decompress.h"
#include <assert.h>

static int moving_count(const uint8_t *graphics, int32_t offset)
{
    if (offset <= 16) return 0;
    int count = 0;
    const uint8_t *record = graphics + offset;
    while (read_word(record) != 999) { count++; record = skip_amiga_wall_list(record + 18); }
    return count;
}

static void check_original(int map)
{
    LevelState level = {0}; char path[128];
    snprintf(path, sizeof(path), "amiga/data/levels/level_%c/twolev.bin", 'a' + map - 1);
    assert(!sb_load_file(path, &level.data, &level.data_byte_count));
    snprintf(path, sizeof(path), "amiga/data/levels/level_%c/twolev.graph.bin", 'a' + map - 1);
    assert(!sb_load_file(path, &level.graphics, &level.graphics_byte_count));
    const int doors = moving_count(level.graphics, read_long(level.graphics));
    const int lifts = moving_count(level.graphics, read_long(level.graphics + 4));
    assert(!level_parse(&level));
    assert(level.num_doors == doors); assert(level.num_lifts == lifts);
}

static void check_final_slot(int alias)
{
    LevelState level = {0};
    assert(!sb_load_file("amiga/data/levels/level_a/twolev.bin", &level.data, &level.data_byte_count));
    assert(!sb_load_file("amiga/data/levels/level_a/twolev.graph.bin", &level.graphics, &level.graphics_byte_count));
    const int count = (read_long(level.graphics + 12) - 16) / 4;
    const int last = count - 1;
    assert(read_word(level.data + 16) == last);
    const int32_t door = read_long(level.graphics), lift = read_long(level.graphics + 4);
    assert(read_word(level.graphics + door) != 999);
    assert(read_word(level.graphics + lift) != 999);
    write_word_be(level.graphics + door + 12, (int16_t)last);
    write_word_be(level.graphics + lift + 12, (int16_t)(alias ? 300 : last));
    if (alias) write_word_be(level.data + read_long(level.graphics + 16 + last * 4), 300);
    assert(!level_parse(&level));
    assert(level_zone_slot_count(&level) == count);
    assert(level.door_data && read_word(level.door_data) == last);
    assert(level.lift_data && read_word(level.lift_data) == last);
}

int main(void)
{
    uint8_t header[18] = {0};
    assert(parsed_zone_slot_count(header, 20) == 1);
    assert(parsed_zone_slot_count(header, 22) == 0);
    write_word_be(header + 16, 255);
    assert(parsed_zone_slot_count(header, 1040) == 256);
    assert(parsed_zone_slot_count(header, 20) == 0);
    assert(parsed_zone_slot_count(header, 1044) == 0);
    assert(parsed_zone_slot_count(header, 0) == 256);
    write_word_be(header + 16, -1);
    assert(parsed_zone_slot_count(header, 0) == 0);
    assert(parsed_zone_slot_count(header, 20) == 0);
    for (int map = 1; map <= 16; map++) check_original(map);
    check_final_slot(0); check_final_slot(1);
    puts("Original controller counts and final-slot direct/aliased mechanisms passed.");
    return 0;
}
