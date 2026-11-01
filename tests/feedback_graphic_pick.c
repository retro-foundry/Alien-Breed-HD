/* Native capture-buffer contract, including the same wall/floor span writers
 * used by the game rasterizer. No SDL window or substitute renderer. */
#include <assert.h>
#define SDL_MAIN_HANDLED
#include "../src/renderer.c"

int main(void)
{
    uint16_t zones[12], zones_back[12];
    uint8_t players[12], players_back[12];
    g_renderer.width = 4; g_renderer.height = 3;
    g_pick_zone_buffer = zones; g_pick_zone_back_buffer = zones_back;
    g_pick_player_buffer = players; g_pick_player_back_buffer = players_back;
    assert(g_pick_graphic_buffer == NULL && g_pick_graphic_back_buffer == NULL);
    assert(renderer_set_graphic_picking(1));
    assert(renderer_set_graphic_picking(1)); /* Idempotent; no reallocation. */
    g_pick_capture_active = 1; g_pick_last_frame_valid = 1;
    renderer_pick_clear_active_buffers();
    RenderSliceContext ctx = {0};
    ctx.pick_zone_id = 7; ctx.pick_graphic = (1234u << 1) | 1u;
    renderer_pick_mark_row_span(&ctx, 1, -10, 2, 7);
    renderer_swap();
    int32_t offset; int upper; int16_t zone;
    for (int x = 0; x <= 2; x++) {
        assert(renderer_get_graphic_pick(x, 1, &offset, &upper));
        assert(offset == 1234 && upper == 1);
        renderer_get_pixel_pick(x, 1, &zone, NULL); assert(zone == 7);
    }
    assert(!renderer_get_graphic_pick(3, 1, &offset, &upper) && offset == -1 && upper == 0);
    assert(!renderer_get_graphic_pick(-1, 0, &offset, &upper));
    assert(!renderer_get_graphic_pick(4, 0, &offset, &upper));
    assert(!renderer_get_graphic_pick(0, -1, &offset, &upper));
    assert(!renderer_get_graphic_pick(0, 3, &offset, &upper));
    g_pick_last_frame_valid = 0;
    assert(!renderer_get_graphic_pick(1, 1, &offset, &upper));
    g_pick_last_frame_valid = 1;
    renderer_pick_clear_active_buffers();
    ctx.pick_graphic = 800u << 1;
    renderer_pick_mark_wall_column(&ctx, 1, 1, 1, 1, 1, 1, 1);
    renderer_swap();
    assert(renderer_get_graphic_pick(0, 1, &offset, &upper) && offset == 800 && upper == 0);
    assert(renderer_get_graphic_pick(2, 1, &offset, &upper));
    assert(renderer_get_graphic_pick(1, 0, &offset, &upper));
    assert(renderer_get_graphic_pick(1, 2, &offset, &upper));
    assert(!renderer_get_graphic_pick(3, 1, &offset, &upper));
    assert(renderer_get_object_pick(1, 1) == -1); /* A surface is not an object. */
    assert(renderer_encode_object_pick(-1) == UINT32_MAX);
    assert(renderer_encode_object_pick(INT_MAX) == UINT32_MAX);
    g_pick_graphic_back_buffer[5] = renderer_encode_object_pick(0);
    assert(renderer_get_object_pick(1, 1) == 0); /* ID zero remains selectable. */
    assert(!renderer_get_graphic_pick(1, 1, &offset, &upper) && offset == -1);
    g_pick_graphic_back_buffer[5] = renderer_encode_object_pick(249);
    assert(renderer_get_object_pick(1, 1) == 249);
    assert(renderer_get_object_pick(-1, 1) == -1);
    assert(renderer_get_object_pick(4, 1) == -1);
    assert(renderer_get_object_pick(1, -1) == -1);
    assert(renderer_get_object_pick(1, 3) == -1);
    g_pick_last_frame_valid = 0;
    assert(renderer_get_object_pick(1, 1) == -1);
    g_pick_last_frame_valid = 1;
    g_pick_capture_active = 0; assert(renderer_get_graphic_pick_target() == NULL);
    assert(renderer_set_graphic_picking(0));
    assert(g_pick_graphic_buffer == NULL && g_pick_graphic_back_buffer == NULL);
    assert(!renderer_get_graphic_pick(1, 1, &offset, &upper));
    assert(renderer_get_object_pick(1, 1) == -1);
    puts("Native graphics picking buffer contracts passed");
    return 0;
}
