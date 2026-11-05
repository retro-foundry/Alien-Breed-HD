/* AB3DI.s blag: pointBrights = Points + 4 + header[14]*4.
 * Generated rooms may use the final point instead of reserving switch slots. */
#include "level.h"
#include <assert.h>
#include <stdio.h>

int main(void)
{
    uint8_t header[16] = {0}, brightness[16] = {0};
    LevelState level = {0};
    level.data = header;
    level.data_byte_count = sizeof(header);
    level.point_brights = brightness;
    header[15] = 3;
    brightness[12] = 255; brightness[13] = 233; /* final lower = -23 */
    brightness[15] = 19;                      /* final upper = 19 */
    assert(level_point_count(&level) == 4);
    assert(level_get_point_brightness(&level, 3, 0) == -23);
    assert(level_get_point_brightness(&level, 3, 1) == 19);
    assert(level_get_point_brightness(&level, 4, 0) == 0);
    assert(level_get_point_brightness(&level, -1, 0) == 0);
    header[15] = 0;
    brightness[1] = 11;
    assert(level_point_count(&level) == 1);
    assert(level_get_point_brightness(&level, 0, 0) == 11);
    assert(level_get_point_brightness(&level, 1, 0) == 0);
    /* AB3DI.s currentPointBrights + Anims.s BrightAnimHandler: only table
     * entries 1..3 advance; 4..15 remain zero. Test native ASR rounding,
     * all editor blend values, both words and signed-byte bypass. */
    level.bright_anim_values[0] = -10;
    level.bright_anim_values[1] = 10;
    level.bright_anim_values[2] = -6;
    for (int selector = 1; selector <= 15; ++selector) {
        int target = selector <= 3 ? level.bright_anim_values[selector - 1] : 0;
        for (int strength = 1; strength <= 16; ++strength) {
            for (int base = 0; base <= 127; ++base) {
                int high = (strength - 1) * 16 + selector;
                int delta = (target - base) * strength;
                int scaled = delta >= 0 ? delta / 16 : -((-delta + 15) / 16);
                int expected = (int8_t)(base + scaled);
                brightness[0] = brightness[2] = (uint8_t)high;
                brightness[1] = brightness[3] = (uint8_t)base;
                assert(level_get_point_brightness(&level, 0, 0) == expected);
                assert(level_get_point_brightness(&level, 0, 1) == expected);
            }
        }
    }
    brightness[0] = 0xf5; brightness[1] = 0xf1;
    assert(level_get_point_brightness(&level, 0, 0) == -15);
    header[14] = header[15] = 255;
    assert(level_point_count(&level) == 0);
    level.data_byte_count = 15;
    assert(level_point_count(&level) == 0);
    assert(level_point_count(NULL) == 0);
    puts("Point-index bounds and 61,440 native selector/blend evaluations passed.");
    return 0;
}
