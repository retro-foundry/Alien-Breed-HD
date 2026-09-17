/* Headless integration check against real levels. No window, audio, saves or
 * gameplay changes: compare frozen views across renderer worker counts. */
#define SDL_MAIN_HANDLED
#include "renderer.h"
#include "io.h"
#include "level.h"
#include "player.h"
#include "objects.h"
#include "visibility.h"
#include "math_tables.h"
#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require(int condition, const char *message)
{
    if (!condition) { fprintf(stderr, "%s\n", message); exit(1); }
}

static uint16_t *render(GameState *state, int voxels, int threads)
{
    state->cfg_voxel_objects = voxels != 0;
    state->cfg_render_threads = threads != 0;
    state->cfg_render_threads_max = (int16_t)threads;
    /* Water samples the previous frame. Two identical frames settle it. */
    renderer_draw_display(state);
    renderer_draw_display(state);
    return g_renderer.cw_back_buffer;
}

static void check_effects(GameState *state)
{
    enum { PIXELS = 640 * 480 };
    /* Reuse the first room and a real shot pool record, restoring it afterwards.
     * This checks both draw-list paths without advancing AI or weapon logic. */
    uint8_t *shot = state->level.player_shot_data;
    require(shot != NULL, "No player shot pool");
    uint8_t saved_shot[OBJECT_SIZE], saved_point[8];
    memcpy(saved_shot, shot, sizeof(saved_shot));
    int cid = obj_w(shot);
    require(cid >= 0 && cid < state->level.num_object_points, "Bad shot point");
    uint8_t *point = state->level.object_points + cid * 8;
    memcpy(saved_point, point, sizeof(saved_point));
    PlayerState *p = &state->plr1;
    int x = p->xoff >> 16, z = (p->zoff >> 16) + 240;
    obj_sw(point, (int16_t)x); obj_sw(point + 4, (int16_t)z);
    obj_sw(shot + 4, (int16_t)(p->yoff / 128));
    shot[6] = shot[7] = 48; shot[14] = shot[15] = 16;
    shot[16] = OBJ_NBR_BULLET; shot[30] = 0; shot[63] = 0;
    uint16_t *blank = malloc(PIXELS * sizeof(*blank));
    uint16_t *reference = malloc(PIXELS * sizeof(*reference));
    require(blank && reference, "Effect test allocation failed");
    static const int frames[][2] = {{2,0},{6,0},{8,3},{0,16},{1,21}};
    for (int i = 0; i < 6; ++i) {
        obj_sw(shot + 12, -1); state->num_explosions = 0;
        memcpy(blank, render(state, 1, 0), PIXELS * sizeof(*blank));
        if (i < 5) {
            obj_sw(shot + 12, p->zone);
            obj_sw(shot + 8, (int16_t)frames[i][0]);
            obj_sw(shot + 10, (int16_t)frames[i][1]);
        } else {
            state->num_explosions = 1;
            state->explosions[0].x = (int16_t)x; state->explosions[0].z = (int16_t)z;
            state->explosions[0].zone = p->zone; state->explosions[0].in_top = 0;
            state->explosions[0].y_floor = p->yoff;
            state->explosions[0].frame = 3;
            state->explosions[0].size_scale = 100;
        }
        memcpy(reference, render(state, 1, 0), PIXELS * sizeof(*reference));
        int changed = 0;
        for (int j = 0; j < PIXELS; ++j) if (blank[j] != reference[j]) ++changed;
        printf("Voxel effect path %d: %d visible pixels\n", i, changed);
        require(changed > 0, "Effect not rendered as a volume");
        uint16_t *actual = render(state, 1, 4);
        require(memcmp(actual, reference, PIXELS * sizeof(*reference)) == 0, "Effect strip mismatch");
    }
    state->num_explosions = 0;
    memcpy(shot, saved_shot, sizeof(saved_shot));
    memcpy(point, saved_point, sizeof(saved_point));
    free(blank); free(reference);
}

int main(int argc, char **argv)
{
    enum { W = 640, H = 480 };
    SDL_SetMainReady();
    require(SDL_Init(SDL_INIT_TIMER) == 0, SDL_GetError());
    math_tables_init(); game_state_init(&g_state);
    GameState *state = &g_state;
    state->cfg_render_width = W; state->cfg_render_height = H;
    state->cfg_weapon_draw = false; state->cfg_post_tint = false;
    state->cfg_mouse_look = true;
    io_init(); renderer_init(); renderer_resize(W, H);
    io_load_objects(); io_load_walls(); io_load_floor(); io_load_sky(); io_load_vec_objects();
    uint16_t *reference = malloc(W * H * sizeof(*reference));
    uint16_t *sprites = malloc(W * H * sizeof(*sprites));
    uint16_t *thread_sprites = malloc(W * H * sizeof(*sprites));
    require(reference && sprites && thread_sprites, "Frame allocation failed");
    int changed_views = 0;
    for (int level = 0; level < 3; ++level) {
        state->current_level = (int16_t)level;
        io_load_level_data(&state->level, level);
        io_load_level_graphics(&state->level, level);
        io_load_level_clips(&state->level, level);
        require(level_parse(&state->level) == 0, "level_parse failed");
        level_assign_clips(&state->level, state->level.num_zones);
        level_apply_level_specific_fixes(&state->level, level);
        object_init_world_sizes_from_types(&state->level);
        player_init_from_level(state);
        PlayerState *p = &state->plr1;
        const uint8_t *rooms = state->level.data + p->roompt + 48;
        for (int view = 0; view < 8; ++view) {
            p->angpos = p->oldangpos = (int16_t)(view * 1024);
            p->oldxoff = p->xoff; p->oldzoff = p->zoff; p->oldyoff = p->yoff;
            ZoneOrder order;
            order_zones(&order, &state->level, p->xoff >> 16, p->zoff >> 16, 0, 0, p->angpos, rooms);
            memcpy(state->zone_order_zones, order.zones, (size_t)order.count * sizeof(int16_t));
            state->zone_order_count = order.count; state->view_list_of_graph_rooms = rooms;
            objects_update_sprite_frames(state);
            memcpy(sprites, render(state, 0, 0), W * H * sizeof(*sprites));
            memcpy(reference, render(state, 1, 0), W * H * sizeof(*reference));
            int changed = 0;
            for (int i = 0; i < W * H; ++i) if (sprites[i] != reference[i]) ++changed;
            if (changed) ++changed_views;
            for (int threads = 2; threads <= 4; threads += 2) {
                memcpy(thread_sprites, render(state, 0, threads), W * H * sizeof(*thread_sprites));
                uint16_t *actual = render(state, 1, threads);
                int differences = 0, background_differences = 0;
                for (int i = 0; i < W * H; ++i) if (actual[i] != reference[i]) {
                    /* Existing floor/water strip rounding can differ. Only
                     * excuse a pixel if neither mode's voxel path touched it. */
                    if (actual[i] == thread_sprites[i] && reference[i] == sprites[i] && sprites[i] != thread_sprites[i])
                        ++background_differences;
                    else ++differences;
                }
                printf("Level %d view %d, voxels changed %d pixels, %d strips: voxel differences %d, existing background differences %d\n", level, view, changed, threads, differences, background_differences);
                require(differences == 0, "Threaded render mismatch");
            }
            if (argc > 1 && changed) {
                char name[1024];
                snprintf(name, sizeof(name), "%s/voxel-level%d-view%d.ppm", argv[1], level, view);
                FILE *out = fopen(name, "wb"); require(out != NULL, "Screenshot open failed");
                fprintf(out, "P6\n%d %d\n255\n", W, H);
                for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
                    size_t offset = AB3D_CW_COL_MAJOR ? (size_t)x * H + y : (size_t)y * W + x;
                    unsigned c = reference[offset];
                    uint8_t rgb[3] = {(uint8_t)(((c >> 8) & 15) * 17), (uint8_t)(((c >> 4) & 15) * 17), (uint8_t)((c & 15) * 17)};
                    require(fwrite(rgb, 1, 3, out) == 3, "Screenshot write failed");
                }
                require(fclose(out) == 0, "Screenshot close failed");
            }
            if (level == 0 && view == 0) check_effects(state);
        }
        io_release_level_memory(&state->level);
    }
    require(changed_views > 0, "Voxel integration produced no visible changes");
    printf("Validated 24 real level views, %d with visible voxel objects\n", changed_views);
    free(reference); free(sprites); free(thread_sprites);
    renderer_shutdown(); io_shutdown(); SDL_Quit();
    return 0;
}
