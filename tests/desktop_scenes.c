/* Replay authored rooms without display pacing to measure CPU render cost.
 * This deliberately freezes gameplay; it does not measure combat or GPU time. */
#ifndef SDL_MAIN_HANDLED
#define SDL_MAIN_HANDLED
#endif
#include <SDL.h>
#include "game_state.h"
#include "math_tables.h"
#include "control_loop.h"
#include "renderer.h"
#include "visibility.h"
#include "level.h"
#include "io.h"
#include "settings.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int word(const uint8_t *p)
{
    return (int16_t)(((unsigned)p[0] << 8) | p[1]);
}

static int number(const char *text, int low, int high)
{
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || !*text || *end || value < low || value > high) {
        fprintf(stderr, "Invalid benchmark argument: %s (expected %d..%d)\n", text, low, high);
        exit(2);
    }
    return (int)value;
}

static int compare_times(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int position_in_room(int zone)
{
    LevelState *level = &g_state.level;
    ZoneInfo info;
    const uint8_t *room = level_get_zone_data_ptr(level, (int16_t)zone);
    if (!room || level_get_zone_info(level, (int16_t)zone, &info) ||
        info.floor_y - info.roof_y < PLAYER_HEIGHT) return 0;

    const uint8_t *list = room + word(room + ZONE_OFF_EXIT_LIST);
    int x = 0, z = 0, count = 0;
    for (int i = 0; i < 256; i++) {
        int line = word(list + i * 2);
        if (line == -2) break;
        if (line < 0 || line >= level->num_floor_lines) continue;
        const uint8_t *edge = level->floor_lines + line * 16;
        x += word(edge);
        z += word(edge + 2);
        count++;
    }
    if (!count) return 0;
    x /= count;
    z /= count;
    if (level_find_zone_for_point(level, x, z, (int16_t)zone) != zone) return 0;

    PlayerState *p = &g_state.plr1;
    p->zone = (int16_t)zone;
    p->roompt = (int32_t)(room - level->data);
    p->xoff = p->oldxoff = x * 65536;
    p->zoff = p->oldzoff = z * 65536;
    p->yoff = p->oldyoff = info.floor_y - PLAYER_HEIGHT;
    p->look_y = p->oldlook_y = 0;
    p->stood_in_top = 0;
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 6 && argc != 7) {
        fprintf(stderr, "Usage: desktop_scenes level width height frames workers [zone]\n"
                        "Levels are 1..16; workers=0 uses automatic selection.\n"
                        "Omit zone to sweep lower rooms with a valid interior center.\n");
        return 2;
    }
    int level = number(argv[1], 1, 16);
    int width = number(argv[2], 16, 8192), height = number(argv[3], 16, 8192);
    int frames = number(argv[4], 1, 10000), workers = number(argv[5], 0, 64);
    int zone = argc == 7 ? number(argv[6], 0, INT16_MAX) : -1;
    double *samples = malloc((size_t)frames * sizeof(*samples));
    if (!samples) { fputs("Cannot allocate benchmark samples\n", stderr); return 1; }

    SDL_SetMainReady();
    if (SDL_Init(SDL_INIT_TIMER) < 0) {
        fprintf(stderr, "Cannot initialize SDL timer: %s\n", SDL_GetError());
        free(samples);
        return 1;
    }
    math_tables_init();
    game_state_init(&g_state);
    io_init();
    settings_load(&g_state);
    renderer_init(width, height);
    renderer_set_rgb_raster_expand(0);
    renderer_set_weapon_post_gl_active(1);
    renderer_set_gl_water_tint_post_active(1);
    io_load_walls(); io_load_floor(); io_load_sky();
    io_load_gun_graphics(); io_load_objects(); io_load_vec_objects();
    g_state.current_level = (int16_t)(level - 1);
    bool ready = true;
    play_the_game_prepare_level(&g_state, &ready);
    g_state.obj_interp_alpha = 1.0f;
    g_state.cfg_render_threads = true;
    g_state.cfg_render_threads_max = (int16_t)workers;

    int first = zone >= 0 ? zone : 0;
    int last = zone >= 0 ? zone : level_zone_slot_count(&g_state.level) - 1;
    unsigned views = 0;
    uint8_t repaired[MAX_ORDER_ENTRIES * 8 + 2];
    double frequency = (double)SDL_GetPerformanceFrequency();
    for (int room = first; room <= last; room++) {
        if (!position_in_room(room)) continue;
        PlayerState *p = &g_state.plr1;
        for (int angle = 0; angle < 8192; angle += 2048) {
            p->angpos = p->oldangpos = (int16_t)angle;
            const uint8_t *list = g_state.level.data + p->roompt + ZONE_OFF_LIST_OF_GRAPH;
            list = level_view_graph_rooms(&g_state.level, g_state.current_level, p->zone,
                                         list, repaired, sizeof(repaired));
            ZoneOrder order;
            order_zones(&order, &g_state.level, (int16_t)(p->xoff >> 16),
                        (int16_t)(p->zoff >> 16), 0, 0, p->angpos, list);
            memcpy(g_state.zone_order_zones, order.zones, order.count * sizeof(int16_t));
            g_state.zone_order_count = order.count;
            g_state.view_list_of_graph_rooms = list;
            renderer_clear_frame_history();
            for (int i = 0; i < 12; i++) renderer_draw_display(&g_state);
            double total = 0;
            for (int i = 0; i < frames; i++) {
                Uint64 start = SDL_GetPerformanceCounter();
                renderer_draw_display(&g_state);
                samples[i] = 1000.0 * (SDL_GetPerformanceCounter() - start) / frequency;
                total += samples[i];
            }
            qsort(samples, (size_t)frames, sizeof(*samples), compare_times);
            printf("SCENE level=%d zone=%d angle=%d size=%dx%d workers=%d frames=%d "
                   "avg_ms=%.3f median_ms=%.3f p95_ms=%.3f max_ms=%.3f\n",
                   level, room, angle, width, height, workers, frames, total / frames,
                   samples[frames / 2], samples[(frames - 1) * 95 / 100], samples[frames - 1]);
            fflush(stdout);
            views++;
        }
    }
    if (!views) fputs("No valid benchmark view in the requested room(s)\n", stderr);
    renderer_shutdown();
    io_shutdown();
    SDL_Quit();
    free(samples);
    return views ? 0 : 1;
}
