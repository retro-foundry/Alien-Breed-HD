/* Replay the stock level_m visibility omission exposed by looking up from
 * room 124: the authored roof in room 121 must reach OrderZones/DrawDisplay. */
#include "level.h"
#include "visibility.h"
#include "sb_decompress.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static int word(const uint8_t *p) { return (int16_t)((p[0] << 8) | p[1]); }

int main(void)
{
    LevelState level = {0};
    size_t bytes;
    assert(!sb_load_file("amiga/data/levels/level_m/twolev.bin", &level.data, &bytes));
    level.data_byte_count = bytes;
    assert(!sb_load_file("amiga/data/levels/level_m/twolev.graph.bin", &level.graphics, &bytes));
    level.graphics_byte_count = bytes;
    assert(!level_parse(&level));
    const uint8_t *authored = level_get_zone_data_ptr(&level, 124) + 48;
    uint8_t scratch[MAX_ORDER_ENTRIES * 8 + 2], original[sizeof(scratch)];
    size_t count = 0;
    while (word(authored + count * 8) >= 0) {
        assert(word(authored + count * 8) != 121);
        assert(++count < MAX_ORDER_ENTRIES);
    }
    memcpy(original, authored, count * 8 + 2);
    const uint8_t *fixed = level_view_graph_rooms(&level, 12, 124, authored, scratch, sizeof(scratch));
    assert(fixed == scratch);
    assert(!memcmp(fixed, original, count * 8));
    assert(!memcmp(fixed + count * 8, level_get_zone_data_ptr(&level, 121) + 48, 8));
    assert(word(fixed + count * 8) == 121);
    assert(word(fixed + (count + 1) * 8) == -1);
    assert(!memcmp(authored, original, count * 8 + 2));

    ZoneOrder order;
    order_zones(&order, &level, -1778, 1093, 0, 0, 3588, fixed);
    int seen = 0;
    for (int i = 0; i < order.count; i++) seen += order.zones[i] == 121;
    assert(seen == 1 && order.count == (int)count + 1);
    assert(level_view_graph_rooms(&level, 12, 124, fixed, scratch, sizeof(scratch)) == fixed);
    assert(level_view_graph_rooms(&level, 11, 124, authored, scratch, sizeof(scratch)) == authored);
    for (int zone = 107; zone <= 130; zone++) {
        if (zone == 124) continue;
        const uint8_t *list = level_get_zone_data_ptr(&level, (int16_t)zone) + 48;
        assert(level_view_graph_rooms(&level, 12, (int16_t)zone, list, scratch, sizeof(scratch)) == list);
    }
    /* Replacement geometry must retain its own authored visibility. */
    unsigned graphics_offset = (level.zone_graph_adds[121 * 8] << 24) |
        (level.zone_graph_adds[121 * 8 + 1] << 16) |
        (level.zone_graph_adds[121 * 8 + 2] << 8) | level.zone_graph_adds[121 * 8 + 3];
    level.graphics[graphics_offset + 24] = 0xff;
    assert(level_view_graph_rooms(&level, 12, 124, authored, scratch, sizeof(scratch)) == authored);
    puts("Stock ceiling visibility, original list preservation, ordering and replacement geometry checks passed.");
    return 0;
}
