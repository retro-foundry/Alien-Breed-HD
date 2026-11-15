/* Desktop pixel tags are diagnostic output. Capturing them must not change
 * the actual frame, and normal rendering must leave this byte buffer alone. */
#include "../src/renderer.c"
#include "sb_decompress.h"
#include <assert.h>

GameState g_state;
enum { W = 192, H = 160 };
static RendererPixel expected_cw[W * H];
static uint32_t expected_rgb[W * H];
static uint8_t expected_tags[W * H];

static void draw_frame(int capture)
{
    renderer_clear_frame_history();
    memset(g_renderer.buffer, 0xa5, W * H);
    memset(g_renderer.back_buffer, 0xa5, W * H);
    if (!renderer_get_rgb_raster_expand()) for (int i = 0; i < W * H; i++) {
        g_renderer.rgb_buffer[i] = 0x12345678;
        g_renderer.rgb_back_buffer[i] = 0x87654321;
    }
    if (capture) renderer_request_center_pick_capture();
    renderer_draw_display(&g_state);
}

int main(void)
{
    size_t bytes;
    LevelState *level = &g_state.level;
    assert(!sb_load_file("amiga/data/levels/level_m/twolev.bin", &level->data, &bytes));
    level->data_byte_count = bytes;
    assert(!sb_load_file("amiga/data/levels/level_m/twolev.graph.bin", &level->graphics, &bytes));
    level->graphics_byte_count = bytes;
    assert(!sb_load_file("amiga/data/levels/level_m/twolev.clips", &level->clips, &bytes));
    level->clips_byte_count = bytes;
    assert(!level_parse(level));
    level_assign_clips(level, level->num_zones);
    level->object_data = NULL;
    math_tables_init();
    renderer_init(W, H);
    renderer_set_rgb_raster_expand(1);
    uint8_t *backfile;
    assert(!sb_load_file("amiga/data/gfx/BackFile", &backfile, &bytes));
    renderer_set_sky_assets(backfile, 432, 38, bytes, NULL);
    renderer_build_level_sky_cache(level);
    g_state.cfg_y_proj_scale = 100;
    g_state.cfg_mouse_look = 1;
    g_state.obj_interp_alpha = 1;
    PlayerState *player = &g_state.plr1;
    player->zone = 88;
    player->roompt = (int32_t)(level_get_zone_data_ptr(level, 88) - level->data);
    player->zoff = player->oldzoff = 4576 * 65536;
    player->yoff = player->oldyoff = -12288;
    player->angpos = player->oldangpos = 2048;
    player->look_y = player->oldlook_y = -180;
    g_state.view_list_of_graph_rooms = level->data + player->roompt + 48;
    const uint8_t *entry = g_state.view_list_of_graph_rooms;
    while (rd16(entry) >= 0) {
        assert(g_state.zone_order_count < MAX_VISIBLE_ROOMS);
        g_state.zone_order_zones[g_state.zone_order_count++] = rd16(entry);
        entry += 8;
    }
    draw_frame(1);
    memcpy(expected_cw, renderer_get_cw_buffer(), sizeof(expected_cw));
    memcpy(expected_rgb, renderer_get_rgb_buffer(), sizeof(expected_rgb));
    memcpy(expected_tags, renderer_get_buffer(), sizeof(expected_tags));
    int geometry = 0;
    for (int i = 0; i < W * H; i++) geometry += expected_tags[i] == 1 || expected_tags[i] == 2;
    assert(geometry > 100);
    draw_frame(0);
    assert(!memcmp(expected_cw, renderer_get_cw_buffer(), sizeof(expected_cw)));
    assert(!memcmp(expected_rgb, renderer_get_rgb_buffer(), sizeof(expected_rgb)));
    for (int i = 0; i < W * H; i++) assert(renderer_get_buffer()[i] == 0xa5);
    /* GPU color-word presentation has no ARGB raster consumer. Preserve CW
     * history and actual output without copying the unused 32-bit history. */
    renderer_set_rgb_raster_expand(0);
    draw_frame(0);
    assert(!memcmp(expected_cw, renderer_get_cw_buffer(), sizeof(expected_cw)));
    for (int i = 0; i < W * H; i++) assert(renderer_get_rgb_buffer()[i] == 0x12345678);
    renderer_set_rgb_raster_expand(1);
    draw_frame(1);
    assert(!memcmp(expected_cw, renderer_get_cw_buffer(), sizeof(expected_cw)));
    assert(!memcmp(expected_rgb, renderer_get_rgb_buffer(), sizeof(expected_rgb)));
    assert(!memcmp(expected_tags, renderer_get_buffer(), sizeof(expected_tags)));
    puts("Desktop colors and diagnostics preserved; normal frames skip tag writes and GPU presentation skips unused ARGB history.");
    return 0;
}
