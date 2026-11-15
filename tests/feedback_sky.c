/* Port-generated sky must continue above the original outdoor lift roof.
 * Replay the actual level_m footprint and BackFile, retaining roof pixels. */
#include "../src/renderer.c"
#include "sb_decompress.h"
#include <assert.h>

GameState g_state;
enum { W = 192, H = 160 };
static uint8_t tags[W * H], geometry_tags[W * H], gap[W * H];
static RendererPixel pixels[W * H], sky[W * H], geometry[W * H];

static void clear_view(void)
{
    memset(tags, 255, sizeof(tags));
    for (int i = 0; i < W * H; i++) pixels[i] = renderer_pixel_from_c12(0xf0f);
}

static void draw_lift(int strip, int background)
{
    clear_view();
    for (int left = 0; left < W; left += strip) {
        int right = left + strip;
        if (right > W) right = W;
        RenderSliceContext ctx;
        render_slice_context_init(&ctx, (int16_t)left, (int16_t)right, 0, H - 1);
        if (background) renderer_draw_zone_backdrop_section_ctx(&ctx, &g_state, 74, 0);
        renderer_draw_zone_ctx(&ctx, &g_state, 74, 0);
    }
}

int main(void)
{
    LevelState *level = &g_state.level;
    size_t bytes;
    assert(!sb_load_file("amiga/data/levels/level_m/twolev.bin", &level->data, &bytes));
    level->data_byte_count = bytes;
    assert(!sb_load_file("amiga/data/levels/level_m/twolev.graph.bin", &level->graphics, &bytes));
    level->graphics_byte_count = bytes;
    assert(!level_parse(level));
    uint8_t *graphics = malloc(bytes);
    assert(graphics);
    memcpy(graphics, level->graphics, bytes);
    renderer_build_level_sky_cache(level);
    assert(!memcmp(graphics, level->graphics, bytes));
    RendererSkyCacheBucket bucket = g_level_sky_cache_buckets[74 * 2];
    assert(bucket.count == 1);
    const RendererSkyCachePoly *poly = &g_level_sky_cache_polys[bucket.start];
    assert(poly->roofed && poly->sides == 4);
    int32_t roof = rd32(level_get_zone_data_ptr(level, 74) + ZONE_OFF_ROOF);
    int32_t outdoor_sky = rd32(level_get_zone_data_ptr(level, 88) + ZONE_OFF_ROOF);
    assert(poly->plane_y == outdoor_sky && poly->plane_y < roof);
    static const int16_t footprint[] = {170, 171, 166, 167};
    assert(!memcmp(poly->pt_indices, footprint, sizeof(footprint)));

    /* Isolate the real map surfaces from gameplay objects. */
    level->object_data = NULL;
    uint8_t *backfile;
    assert(!sb_load_file("amiga/data/gfx/BackFile", &backfile, &bytes));
    assert(bytes == 32832);
    renderer_set_sky_assets(backfile, 432, 38, bytes, NULL);
    math_tables_init();
    build_amiga12_lut();
    g_renderer.width = W; g_renderer.height = H;
    g_renderer.buffer = tags; g_renderer.cw_buffer = pixels;
    g_renderer.view_center_x = W / 2; g_renderer.view_center_y = H * 5 / 6;
    g_renderer.proj_y_scale = PROJ_Y_SCALE * H / 640;
    g_renderer.xoff = 0; g_renderer.zoff = 4576; g_renderer.yoff = -12288;
    g_renderer.sinval = sin_lookup(2048); g_renderer.cosval = cos_lookup(2048);
    g_renderer.sky_frame_angpos = 2048;
    g_renderer.floor_uv_dist_near = 32000; g_renderer.floor_uv_dist_max = 30000;
    g_renderer.rotate_stamp = 1;
    for (int i = 0; i < level_point_count(level); i++) rotate_one_point(&g_renderer, level->points, i);
    RenderSliceContext ctx;
    render_slice_context_init(&ctx, 0, W, 0, H - 1);
    for (int y = 0; y < H; y++) renderer_draw_sky_ceiling_span_ctx(&ctx, (int16_t)y, 0, W - 1);
    memcpy(sky, pixels, sizeof(sky));
    draw_lift(W, 0);
    memcpy(geometry, pixels, sizeof(geometry));
    memcpy(geometry_tags, tags, sizeof(tags));
    draw_lift(W, 1);
    int filled = 0, roof_pixels = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int at = y * W + x;
        size_t pixel = renderer_cw_index_xy(x, y, W, H);
        if (geometry_tags[at] == 1 || geometry_tags[at] == 2) {
            assert(tags[at] == geometry_tags[at] && pixels[pixel] == geometry[pixel]);
            roof_pixels += geometry_tags[at] == 1 && y < g_renderer.view_center_y;
        }
        gap[at] = geometry_tags[at] == 255 && tags[at] == 0;
        if (gap[at]) { assert(pixels[pixel] == sky[pixel]); filled++; }
    }
    assert(filled > 100 && roof_pixels > 100);
    draw_lift(17, 1);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (gap[y * W + x]) {
        size_t pixel = renderer_cw_index_xy(x, y, W, H);
        assert(tags[y * W + x] == 0 && pixels[pixel] == sky[pixel]);
    }
    printf("Sky mesh: %d missing background pixels filled, %d roof pixels retained; sliced sky and original graphics preserved.\n", filled, roof_pixels);
    return 0;
}
