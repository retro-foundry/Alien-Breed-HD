/* Rasterize authored water polygons at the fractional heights crossed by bob.
 * AB3DI.s itsafloordraw/texturedwater are the source of the surface and tint. */
#include "../src/renderer.c"
#include "sb_decompress.h"
#include <assert.h>

GameState g_state;

enum { TEST_W = 96, MAX_H = 200 };
static uint8_t tags[TEST_W * MAX_H];
static RendererPixel pixels[TEST_W * MAX_H], reference[TEST_W * MAX_H];
static RendererPixel near_surface[TEST_W * MAX_H];
static RenderSliceContext slice;

static void clear_view(int height)
{
    memset(tags, 0, sizeof(tags));
    for (int y = 0; y < height; y++) for (int x = 0; x < TEST_W; x++) {
        uint16_t colour = (uint16_t)(0x800 | (((x + y) & 15) << 4) | ((y / 3) & 15));
        renderer_cw_store_xy(pixels, x, y, TEST_W, height, renderer_pixel_from_c12(colour));
    }
}

static int draw_water(GameState *state, int zone, int32_t plane, int rel, int centre, int strips)
{
    int height = g_renderer.height;
    g_renderer.yoff = plane - rel;
    g_renderer.flooryoff = (int16_t)((g_renderer.yoff >> 8) * 4);
    g_renderer.view_center_y = centre;
    clear_view(height);
    int fill = 0;
    for (int left = 0; left < TEST_W; left += strips) {
        int right = left + strips;
        if (right > TEST_W) right = TEST_W;
        render_slice_context_init(&slice, (int16_t)left, (int16_t)right, 0, (int16_t)(height - 1));
        renderer_draw_zone_ctx(&slice, state, (int16_t)zone, 0);
        if (left) assert(fill == slice.fill_screen_water);
        fill = slice.fill_screen_water;
    }
    return fill;
}

static int check_water(int map, int zone)
{
    static GameState state;
    memset(&state, 0, sizeof(state));
    size_t bytes;
    char path[128];
    snprintf(path, sizeof(path), "amiga/data/levels/level_%c/twolev.bin", 'a' + map - 1);
    assert(!sb_load_file(path, &state.level.data, &bytes)); state.level.data_byte_count = bytes;
    snprintf(path, sizeof(path), "amiga/data/levels/level_%c/twolev.graph.bin", 'a' + map - 1);
    assert(!sb_load_file(path, &state.level.graphics, &bytes)); state.level.graphics_byte_count = bytes;
    assert(!level_parse(&state.level));
    LevelState *level = &state.level;
    uint8_t *original_graphics = level->graphics;
    uint8_t *room = level_get_zone_data_ptr(level, zone);
    state.plr1.zone = (int16_t)zone;
    state.plr1.roompt = (int32_t)(room - level->data);

    /* Keep the real polygon, heights and points; isolate it from walls/objects
     * so a missing water span cannot be concealed by another primitive. */
    const uint8_t *entry = level->graphics + rd32(level->zone_graph_adds + zone * 8) + 2;
    while (rd16(entry) != 7) {
        assert(rd16(entry) >= 0);
        entry += 2 + zone_gfx_entry_data_skip(rd16(entry), entry + 2);
    }
    size_t entry_bytes = 2 + zone_gfx_entry_data_skip(7, entry + 2);
    uint8_t graphics[256] = {0};
    assert(entry_bytes + 6 <= sizeof(graphics));
    obj_sw(graphics + 2, (int16_t)zone);
    memcpy(graphics + 4, entry, entry_bytes);
    obj_sw(graphics + 4 + entry_bytes, -1);
    int32_t plane = (int32_t)rd16(entry + 2) * 64;
    int count = rd16(entry + 4) + 1, sum_x = 0, sum_z = 0;
    int16_t indices[100]; assert(count >= 3 && count <= 100);
    for (int i = 0; i < count; i++) {
        indices[i] = rd16(entry + 6 + i * 2);
        sum_x += rd16(level->points + indices[i] * 4);
        sum_z += rd16(level->points + indices[i] * 4 + 2);
    }
    g_renderer.xoff = (int16_t)(sum_x / count);
    g_renderer.zoff = (int16_t)(sum_z / count);
    g_renderer.xoff34 = (int16_t)(g_renderer.xoff * 3 / 4);
    g_renderer.zoff34 = (int16_t)(g_renderer.zoff * 3 / 4);
    uint8_t *adds = calloc((size_t)level_zone_slot_count(level), 8); assert(adds);
    obj_sl(adds + zone * 8, 2);
    level->graphics = graphics; level->graphics_byte_count = entry_bytes + 6;
    level->zone_graph_adds = adds;

    int cases = 0;
    const int heights[] = {80, 200};
    const int projection_percents[] = {50, 100, 200};
    const int relative_heights[] = {512, 257, 256, 255, 128, 1, 0, -1, -128, -256};
    for (int view = 0; view < 6; view++) {
        int height = heights[view % 2];
        g_renderer.height = height;
        g_renderer.proj_y_scale = PROJ_Y_SCALE * height / 640 * projection_percents[view / 2] / 100;
        for (int angle = 0; angle < 8192; angle += 2048) {
            g_renderer.sinval = sin_lookup((int16_t)angle);
            g_renderer.cosval = cos_lookup((int16_t)angle);
            for (int i = 0; i < count; i++) rotate_one_point(&g_renderer, level->points, indices[i]);
            for (int pitch = 0; pitch <= 4; pitch++) {
                int centre = height * pitch / 4;
                for (unsigned ri = 0; ri < sizeof(relative_heights) / sizeof(relative_heights[0]); ri++) {
                    int rel = relative_heights[ri];
                    int fill = draw_water(&state, zone, plane, rel, centre, TEST_W);
                    assert(fill == (rel < 0 ? 15 : rel == 0 ? -1 : 0));
                    memcpy(reference, pixels, sizeof(reference));
                    assert(draw_water(&state, zone, plane, rel, centre, 17) == fill);
                    assert(!memcmp(reference, pixels, (size_t)TEST_W * height * sizeof(*pixels)));
                    for (int y = 0; y < height; y++) {
                        if (rel < 0 ? y >= centre : y < centre)
                            for (int x = 0; x < TEST_W; x++) assert(tags[y * TEST_W + x] == 0);
                    }
                    if (rel == 1 || rel == -1 || rel == 0) {
                        int first = rel < 0 ? 0 : centre + 4;
                        int last = rel < 0 ? centre - 4 : height;
                        for (int y = first; y < last; y++) for (int x = 0; x < TEST_W; x++) {
                            if (tags[y * TEST_W + x] != 4) {
                                fprintf(stderr, "Water hole: level=%d zone=%d rel=%d centre=%d angle=%d pixel=%d,%d\n",
                                        map, zone, rel, centre, angle, x, y);
                                assert(0);
                            }
                        }
                    }
                    if (rel == 1) memcpy(near_surface, pixels, sizeof(near_surface));
                    if (rel == 0) {
                        for (int y = centre + 4; y < height; y++) for (int x = 0; x < TEST_W; x++) {
                            size_t at = renderer_cw_index_xy(x, y, TEST_W, height);
                            assert(pixels[at] == near_surface[at]);
                        }
                        renderer_apply_underwater_tint(-1);
                        assert(!memcmp(reference, pixels, (size_t)TEST_W * height * sizeof(*pixels)));
                    }
                    cases++;
                }
            }
        }
    }
    /* A remote coplanar polygon must not tint the viewer's foreground. */
    g_renderer.xoff = INT16_MAX;
    for (int i = 0; i < count; i++) rotate_one_point(&g_renderer, level->points, indices[i]);
    draw_water(&state, zone, plane, 0, g_renderer.height / 2, TEST_W);
    for (int i = 0; i < TEST_W * g_renderer.height; i++) assert(tags[i] == 0);
    free(adds); free(original_graphics); free(level->data);
    return cases;
}

int main(void)
{
    math_tables_init();
    g_renderer.width = TEST_W; g_renderer.view_center_x = TEST_W / 2;
    g_renderer.buffer = tags; g_renderer.cw_buffer = pixels;
    g_renderer.floor_uv_dist_near = 32000; g_renderer.floor_uv_dist_max = 30000;
    uint8_t *water, *brighten; size_t water_bytes, brighten_bytes;
    assert(!sb_load_file("amiga/data/helper/WaterFile", &water, &water_bytes));
    assert(!sb_load_file("amiga/data/helper/OldBrightenFile", &brighten, &brighten_bytes));
    renderer_set_water_assets(water, water_bytes, brighten, brighten_bytes);
    int cases = check_water(1, 85) + check_water(14, 93) + check_water(15, 31);
    clear_view(g_renderer.height);
    renderer_apply_underwater_tint(15);
    for (int i = 0; i < TEST_W * g_renderer.height; i++)
        assert((renderer_pixel_to_c12(pixels[i]) & 0xF00) == 0);
    free(water); free(brighten);
    printf("PASS: waterline raster (%d authored polygon/height/pitch/angle cases, whole and sliced, full tint)\n", cases);
    return 0;
}
