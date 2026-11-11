/* Replay real barrel/debris updates and capture every rendered frame.
 * tools/check_dos_models.py compares the same engine at 4 and 16 MB with
 * 32 KiB of extra residency to exercise the former late scratch failure. */
#include "../src/game_loop.h"
#include "../src/level.h"
#include "../src/objects.h"
static void model_scene_tick(GameState *, GameLoopCtx *);
#define game_loop_tick model_scene_tick
#include "../src/main.c"
#undef game_loop_tick
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

static volatile uint8_t scene_residency[32 * 1024];

static GameObject *scene_nearest_barrel(const GameState *state)
{
    GameObject *nearest = NULL;
    int64_t best = INT64_MAX;
    for (int i = 0; state->level.object_data; i++) {
        GameObject *obj = (GameObject *)(state->level.object_data + i * OBJECT_SIZE);
        if (OBJ_CID(obj) < 0) break;
        if (OBJ_ZONE(obj) < 0 || obj->obj.number != OBJ_NBR_BARREL) continue;
        const uint8_t *point = state->level.object_points + OBJ_CID(obj) * 8;
        int64_t dx = obj_w(point) - (state->plr1.xoff >> 16);
        int64_t dz = obj_w(point + 4) - (state->plr1.zoff >> 16);
        int64_t distance = dx * dx + dz * dz;
        if (distance < best) { best = distance; nearest = obj; }
    }
    if (!nearest) { fprintf(stderr, "Model scene requires an authored barrel\n"); exit(4); }
    return nearest;
}

static unsigned scene_hash(unsigned hash, const void *data, size_t bytes)
{
    const uint8_t *p = data;
    while (bytes--) hash = (hash ^ *p++) * 16777619u;
    return hash;
}

static void model_scene_tick(GameState *state, GameLoopCtx *ctx)
{
    static int frame;
    if (frame == 0) {
        for (size_t i = 0; i < sizeof(scene_residency); i++) scene_residency[i] = 0;
        int x = -1352, z = 1720, zone = 27;
        if (state->current_level != 0) {
            const GameObject *barrel = scene_nearest_barrel(state);
            const uint8_t *point = state->level.object_points + OBJ_CID(barrel) * 8;
            x = obj_w(point); z = obj_w(point + 4) - 128; zone = OBJ_ZONE(barrel);
        }
        int index = level_connect_to_zone_index(&state->level, zone);
        if (index < 0) exit(4);
        PlayerState *p = &state->plr1;
        p->xoff = p->s_xoff = p->oldxoff = p->p_xoff = x * 65536;
        p->zoff = p->s_zoff = p->oldzoff = p->p_zoff = z * 65536;
        p->zone = p->s_zone = (int16_t)index;
        int32_t room = obj_l(state->level.zone_adds + index * 4);
        p->roompt = p->s_roompt = p->old_roompt = p->s_old_roompt = room;
        p->yoff = p->s_yoff = p->oldyoff = p->p_yoff =
            p->tyoff = p->s_tyoff = obj_l(state->level.data + room + 2) - PLAYER_HEIGHT;
        p->angpos = p->s_angpos = 0;
        state->infinite_health = true;
        ctx->last_ticks = 1000;
    }
    if (frame == 8) {
        GameObject *barrel = scene_nearest_barrel(state);
        barrel->raw[19] = (uint8_t)(NASTY_LIVES(*barrel) > 0 ? NASTY_LIVES(*barrel) : 1);
        object_handle_barrel(barrel, state);
    }
    game_loop_tick(state, ctx);
    unsigned hash = 2166136261u;
    for (int i = 0; state->level.object_data; i++) {
        const GameObject *obj = (const GameObject *)(state->level.object_data + i * OBJECT_SIZE);
        if (OBJ_CID(obj) < 0) break;
        hash = scene_hash(hash, obj, OBJECT_SIZE);
    }
    if (state->level.player_shot_data)
        hash = scene_hash(hash, state->level.player_shot_data, PLAYER_SHOT_SLOT_COUNT * OBJECT_SIZE);
    if (state->level.nasty_shot_data)
        hash = scene_hash(hash, state->level.nasty_shot_data, NASTY_SHOT_SLOT_COUNT * OBJECT_SIZE);
    hash = scene_hash(hash, state->level.object_points, state->level.num_object_points * 8u);
    printf("[MODELCHECK] frame=%d hash=%08x\n", frame, hash);
    char name[13];
    snprintf(name, sizeof(name), "SC%02d.RGB", frame++);
    int fd = open(name, O_CREAT | O_TRUNC | O_WRONLY | O_BINARY, S_IRUSR | S_IWUSR);
    if (fd < 0) exit(4);
    const uint8_t *pixels = (const uint8_t *)renderer_get_cw_buffer();
    size_t bytes = (size_t)g_renderer.width * g_renderer.height * sizeof(RendererPixel);
    while (bytes) {
        int count = write(fd, pixels, bytes);
        if (count <= 0) exit(4);
        pixels += count; bytes -= count;
    }
    if (close(fd)) exit(4);
}
