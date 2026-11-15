/* A fixed explored area and zoom must cover the same fraction of the map at
 * DOS, desktop and supersampled resolutions. The arrow scales with the view. */
#include "../src/renderer.c"
#include <assert.h>

GameState g_state;

int main(void)
{
    math_tables_init();
    AutomapSeenWall walls[4] = {
        {.x1=-256, .z1=-3328, .x2=2304, .z2=-3328},
        {.x1=2304, .z1=-3328, .x2=2304, .z2=-768},
        {.x1=2304, .z1=-768, .x2=-256, .z2=-768},
        {.x1=-256, .z1=-768, .x2=-256, .z2=-3328}
    };
    g_state.level.automap_seen_walls = walls;
    g_state.level.automap_seen_count = 4;
    g_state.plr1.xoff = 1024 * 65536;
    g_state.plr1.zoff = -2048 * 65536;
    const int sizes[][2] = {
        {320,200}, {320,240}, {640,480}, {768,640},
        {1280,720}, {1920,1080}, {3840,2160}, {7680,4320}
    };
    int x0[7], y0[7], x1[7], y1[7]; uint16_t colors[7];
    for (unsigned view = 0; view < sizeof(sizes)/sizeof(sizes[0]); view++) {
        int w=sizes[view][0], h=sizes[view][1];
        int cx=w-1-w/2, cy=h/2;
        for (int zoom = 2; zoom <= 128; zoom *= 2) {
            g_automap_units_per_px = zoom;
            for (int angle = 0; angle < 8192; angle += 1024) {
                g_state.plr1.angpos = (int16_t)angle;
                assert(renderer_automap_collect_line_segments(&g_state,x0,y0,x1,y1,colors,7,w,h)==7);
                double radius = 1280.0 / zoom * h / RENDER_DEFAULT_HEIGHT;
                assert(fabs((x0[0]-cx)-radius) < 1);
                assert(fabs((y0[0]-cy)+radius) < 1);
                assert(fabs((x1[0]-cx)+radius) < 1);
                assert(fabs((y1[0]-cy)+radius) < 1);
                assert(x0[4]==cx && y0[4]==cy);
                double tipx=-(double)sin_lookup((int16_t)angle)*40/32768*h/RENDER_DEFAULT_HEIGHT;
                double tipy=(double)cos_lookup((int16_t)angle)*40/32768*h/RENDER_DEFAULT_HEIGHT;
                assert(fabs((x1[4]-cx)-tipx)<1 && fabs((y1[4]-cy)-tipy)<1);
                assert(colors[4] & RENDERER_AUTOMAP_SEGFLAG_PLAYER);
            }
        }
    }
    g_automap_units_per_px=8;
    renderer_automap_adjust_scale(-1000); assert(g_automap_units_per_px==2);
    renderer_automap_adjust_scale(1000); assert(g_automap_units_per_px==128);
    puts("Automap coverage and player marker scale across 8 resolutions, 7 zoom levels and 8 headings; zoom limits retained.");
    return 0;
}
