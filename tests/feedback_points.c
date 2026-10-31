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
    header[14] = header[15] = 255;
    assert(level_point_count(&level) == 0);
    level.data_byte_count = 15;
    assert(level_point_count(&level) == 0);
    assert(level_point_count(NULL) == 0);
    puts("Point-index count and final-point brightness checks passed.");
    return 0;
}
