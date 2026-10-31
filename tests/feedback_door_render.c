/* Render real door records: the texture is fixed to the moving panel, so a
 * material point above its lower edge must retain its texel while it moves. */
#include "../src/renderer.c"
#include "objects.h"
#include "sb_decompress.h"
#include <assert.h>

GameState g_state;
void audio_play_sample(int sample, int volume) { (void)sample; (void)volume; }

enum { TEST_W = 4, TEST_H = 2048, TEST_CENTRE = TEST_H / 2 };
static uint8_t tags[TEST_W * TEST_H], palette[4096], texture[512];
static RendererPixel pixels[TEST_W * TEST_H];
static uint16_t rendered[2][TEST_H];
static RenderSliceContext slice;

static void raster_door(const LevelState *level, uint32_t offset, int32_t camera_y)
{
    const uint8_t *wall = level->graphics + offset;
    int16_t top = (int16_t)((obj_l(wall + 20) - camera_y) >> 8);
    int16_t bot = (int16_t)((obj_l(wall + 24) - camera_y) >> 8);
    int32_t span = 0, phase = 0;
    assert(renderer_door_wall_texture_info_for_gfx_off(level, offset, wall[17], &span, &phase));
    for (int pack = 0; pack < 2; pack++) {
        memset(pixels, 0, sizeof(pixels));
        render_slice_context_init(&slice, 0, 0, 0, TEST_H - 1);
        slice.cur_wall_pal = palette;
        slice.update_column_clip = 0;
        draw_wall_rasterize_segment(&slice, 0, 0, 0, 1,
            (1LL << 24) / RENDER_SCALE, 0, 0, 0, 0, 0,
            top, bot, texture, wall[16], wall[17], 0,
            phase, pack, obj_w(wall + 14), span, 64);
        for (int y = 0; y < TEST_H; y++) {
            uint16_t colour = renderer_pixel_to_c12(pixels[renderer_cw_index_xy(0, y, TEST_W, TEST_H)]);
            rendered[pack][y] = colour ? colour - 1 : UINT16_MAX;
        }
    }
}

static int check_level(int map, int *doors, int *faces)
{
    static GameState state;
    memset(&state, 0, sizeof(state));
    state.plr1.zone = state.plr2.zone = -1; state.temp_frames = 1;
    char path[128]; size_t bytes;
    snprintf(path, sizeof(path), "amiga/data/levels/level_%c/twolev.bin", 'a' + map);
    assert(!sb_load_file(path, &state.level.data, &bytes)); state.level.data_byte_count = bytes;
    snprintf(path, sizeof(path), "amiga/data/levels/level_%c/twolev.graph.bin", 'a' + map);
    assert(!sb_load_file(path, &state.level.graphics, &bytes)); state.level.graphics_byte_count = bytes;
    assert(!level_parse(&state.level));
    LevelState *level = &state.level;
    int samples = 0;
    for (int step = 0; step <= 32; step++) {
        int progress = step <= 16 ? step : 32 - step;
        for (int di = 0; di < level->num_doors; di++) {
            uint8_t *door = level->door_data + di * 22;
            int32_t closed = obj_l(door + 14), open = obj_l(door + 10);
            obj_sl(door + 4, closed + (int32_t)(((int64_t)open - closed) * progress / 16));
            obj_sw(door + 8, 0);
        }
        door_routine(&state);
        for (int di = 0; di < level->num_doors; di++) {
            uint8_t *door = level->door_data + di * 22;
            for (unsigned j = level->door_wall_list_offsets[di]; j < level->door_wall_list_offsets[di + 1]; j++) {
                uint32_t offset = (uint32_t)obj_l(level->door_wall_list + j * 10 + 2);
                uint8_t *wall = level->graphics + offset;
                int32_t top = obj_l(wall + 20), bottom = obj_l(wall + 24);
                if (bottom - top < 4 * 256) continue; /* Hidden/vanishing panel. */
                int32_t height = level->door_wall_texture_heights[j] * 256;
                int32_t closed = obj_l(door + 14);
                int32_t camera = top + (closed - top) / 2;
                int rows = 1 << wall[17]; assert(rows <= 256);
                for (int row = 0; row < rows; row++) {
                    uint16_t packed = (uint16_t)((row & 31) | (((row >> 5) & 31) << 5));
                    obj_sw(texture + row * 2, packed);
                }
                raster_door(level, offset, camera);
                int screen_top = ((top - camera) >> 8) + TEST_CENTRE;
                int screen_bottom = ((bottom - camera) >> 8) + TEST_CENTRE;
                for (int fraction = 1; fraction <= 3; fraction++) {
                    int y = screen_top + (screen_bottom - screen_top) * fraction / 4;
                    assert(y >= 0 && y < TEST_H);
                    assert(rendered[0][y] != UINT16_MAX && rendered[1][y] != UINT16_MAX);
                    int actual = rendered[0][y] | (rendered[1][y] << 5);
                    /* Independent physical reference: a full panel of authored
                     * height travels with its lower edge; only the exposed part
                     * is drawn. No absolute room-height phase belongs here. */
                    int64_t point_y = (int64_t)(y - TEST_CENTRE) * 256 + camera;
                    int64_t distance_from_panel_top = point_y - bottom + height;
                    int expected = (int)(distance_from_panel_top * rows / height) & wall[16];
                    int error = (actual - expected) & wall[16];
                    if (error > 1 && error < wall[16]) {
                        fprintf(stderr, "Rendered door slip: level=%d zone=%d gfx=%u step=%d y=%d actual-row=%d expected-row=%d\n",
                            map + 1, obj_w(door), offset, step, y, actual, expected);
                        return -1;
                    }
                    samples++;
                }
            }
        }
    }
    *doors += level->num_doors;
    *faces += level->door_wall_list_offsets[level->num_doors];
    return samples;
}

int main(void)
{
    g_renderer.width = TEST_W; g_renderer.height = TEST_H;
    g_renderer.proj_y_scale = 1; g_renderer.view_center_y = TEST_CENTRE;
    g_renderer.buffer = tags; g_renderer.cw_buffer = pixels;
    for (unsigned i = 0; i < sizeof(palette) / 2; i++) obj_sw(palette + i * 2, (i % 32) + 1);
    int doors = 0, faces = 0, samples = 0;
    for (int map = 0; map < 16; map++) {
        int result = check_level(map, &doors, &faces);
        if (result < 0) return 1;
        samples += result;
    }
    assert(doors == 123 && faces == 245);
    printf("PASS: %d rendered texel samples, %d doors / %d faces across all 16 levels: closed alignment and opening/closing material motion\n", samples, doors, faces);
    return 0;
}
