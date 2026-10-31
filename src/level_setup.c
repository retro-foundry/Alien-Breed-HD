/* Port setup extension. Original geometry/graphics formats remain unchanged.
 * Version 1 stores an exit; version 2 also identifies authored draw rooms. */
#include "level.h"
#include <string.h>

static unsigned setup_word(const uint8_t *bytes) { return ((unsigned)bytes[0] << 8) | bytes[1]; }

int level_apply_setup(LevelState *level, const uint8_t *bytes, size_t length,
                      int16_t original_exit, int16_t *out_exit)
{
    if (!level || !bytes || !out_exit || length < 8 || memcmp(bytes, "ABEX", 4)) return -1;
    unsigned version = setup_word(bytes + 4), count = 0;
    int16_t groups[256];
    if (version == 1) { if (length != 8) return -1; }
    else if (version == 2) {
        if (length < 10) return -1;
        count = setup_word(bytes + 8);
        if (!count || count > 256 || length != 10 + count * 2 || count != (unsigned)level_zone_slot_count(level)) return -1;
        for (unsigned i = 0; i < count; i++) {
            unsigned group = setup_word(bytes + 10 + i * 2);
            if (group >= count || !level_get_zone_data_ptr(level, (int16_t)group) || setup_word(bytes + 10 + group * 2) != group) return -1;
            groups[i] = (int16_t)group;
        }
    } else return -1;
    int16_t exit_zone = (int16_t)setup_word(bytes + 6);
    if (exit_zone < -2 || (exit_zone >= 0 && !level_get_zone_data_ptr(level, exit_zone))) return -1;
    if (count) memcpy(level->draw_groups, groups, count * sizeof(*groups));
    level->draw_group_count = (uint16_t)count;
    *out_exit = exit_zone == -2 ? original_exit : exit_zone;
    return 0;
}
