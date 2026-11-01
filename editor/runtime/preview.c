/* Editor host for the unmodified game simulation and renderer. The normal
 * executable still enters through src/main.c; this host skips menus/intro text. */
#include <emscripten.h>
#define main ab3d_original_main
#include "../../src/main.c"
#undef main
#include "game_loop.h"
#include "level.h"
#include "renderer.h"
#include "visibility.h"
#include "game_data.h"

static GameLoopCtx preview_loop;
static bool copper_ready;
static int preview_loaded;
static int preview_playing;

EM_JS(void, preview_report, (int kind, int zone, int value), {
    Module['editorEvent'](kind, zone, value);
});

static void preview_draw(void)
{
    PlayerState *p = &g_state.plr1;
    ZoneOrder order;
    g_state.view_list_of_graph_rooms = g_state.level.data + p->roompt + ZONE_OFF_LIST_OF_GRAPH;
    order_zones(&order, &g_state.level, p->xoff >> 16, p->zoff >> 16,
                0, 0, p->angpos, g_state.view_list_of_graph_rooms);
    memcpy(g_state.zone_order_zones, order.zones, order.count * sizeof(int16_t));
    g_state.zone_order_count = order.count;
    renderer_request_center_pick_capture();
    display_draw_display(&g_state);
}

EMSCRIPTEN_KEEPALIVE int editor_load_level(int slot)
{
    if (slot < 0 || slot >= MAX_LEVELS) return 0;
    preview_playing = 0;
    preview_loaded = 0;
    if (g_state.level.data) io_release_level_memory(&g_state.level);
    /* cfg_start_level is zero based internally (settings_load converts it). */
    g_state.cfg_start_level = slot;
    play_game_apply_front_menu_result(&g_state, GAME_LOOP_FRONT_MENU_NEW_GAME);
    play_the_game_prepare_level(&g_state, &copper_ready);
    display_clear_text_screen();
    g_state.cfg_show_fps = false;
    g_state.cfg_weapon_draw = false;
    g_state.cfg_mouse_look = true;
    preview_loaded = 1;
    game_loop_ctx_init(&preview_loop, &g_state);
    /* Context initialization resets interpolation to the previous tick. A
     * paused editor has no tick to advance it: show the loaded spawn instead
     * of the previous level's camera (especially visible in authored rooms). */
    g_state.obj_interp_alpha = 1.0f;
    preview_draw();
    preview_report(1, g_state.plr1.zone, level_zone_slot_count(&g_state.level));
    return 1;
}

EMSCRIPTEN_KEEPALIVE void editor_reload_wall_assets(void)
{
    preview_playing = 0;
    io_load_walls();
}

EMSCRIPTEN_KEEPALIVE int editor_camera(int zone, int x, int z, int upper, int angle)
{
    if (!preview_loaded || preview_playing) return 0;
    LevelState *level = &g_state.level;
    uint8_t *room = level_get_zone_data_ptr(level, zone);
    /* Inspection cameras may use original spawn/object metadata directly.
     * Collision's extra corner-wall lists are not the room's polygon and
     * must not reject a camera at a valid original start (e.g. level 3). */
    if (!room || x < -32768 || x > 32767 || z < -32768 || z > 32767) return 0;
    PlayerState *p = &g_state.plr1;
    p->zone = p->s_zone = zone;
    p->roompt = p->s_roompt = p->old_roompt = room - level->data;
    p->no_transition_back_roompt = -1;
    p->xoff = p->s_xoff = p->oldxoff = (int32_t)((int64_t)x * 65536);
    p->zoff = p->s_zoff = p->oldzoff = (int32_t)((int64_t)z * 65536);
    int32_t floor = obj_l(room + (upper ? ZONE_OFF_UPPER_FLOOR : ZONE_OFF_FLOOR));
    int32_t roof = obj_l(room + (upper ? ZONE_OFF_UPPER_ROOF : ZONE_OFF_ROOF));
    int32_t height = floor - roof;
    int32_t eye = floor - (height > PLAYER_HEIGHT ? PLAYER_HEIGHT : height / 2);
    p->yoff = p->s_yoff = p->oldyoff = p->p_yoff = eye;
    p->height = p->s_height = p->p_height = p->s_targheight = PLAYER_HEIGHT;
    p->angpos = p->s_angpos = p->oldangpos = p->p_angpos = angle & ANGLE_MASK;
    p->look_y = p->s_look_y = p->oldlook_y = 0;
    p->stood_in_top = upper ? 1 : 0;
    p->p_xoff = x; p->p_zoff = z;
    p->points_to_rotate_ptr = p->s_points_to_rotate_ptr = p->roompt + obj_w(room + ZONE_OFF_PTS);
    p->list_of_graph_rooms = p->s_list_of_graph_rooms = p->roompt + ZONE_OFF_LIST_OF_GRAPH;
    preview_draw();
    return 1;
}

EMSCRIPTEN_KEEPALIVE void editor_turn(int delta, int look)
{
    if (!preview_loaded || preview_playing) return;
    PlayerState *p = &g_state.plr1;
    p->angpos = p->s_angpos = p->oldangpos = (p->angpos + delta) & ANGLE_MASK;
    int pitch = p->look_y + look;
    if (pitch < -100) pitch = -100;
    if (pitch > 100) pitch = 100;
    p->look_y = p->s_look_y = p->oldlook_y = pitch;
    preview_draw();
}

EMSCRIPTEN_KEEPALIVE int editor_pick(void)
{
    int16_t zone = -1;
    renderer_get_center_pick(&zone, NULL);
    return zone;
}

EMSCRIPTEN_KEEPALIVE int editor_nonblack_pixels(void)
{
    const RendererPixel *pixels = renderer_get_cw_buffer();
    if (!pixels) return 0;
    int count = 0;
    int size = renderer_get_width() * renderer_get_height();
    for (int i = 0; i < size; i++) if (pixels[i] != 0) count++;
    return count;
}

EMSCRIPTEN_KEEPALIVE int editor_current_slot(void) { return g_state.current_level; }
EMSCRIPTEN_KEEPALIVE int editor_current_exit_zone(void) { return g_state.end_zones[g_state.current_level & 0x0F]; }

EMSCRIPTEN_KEEPALIVE unsigned editor_capture(int zone, int x, int z, int upper, int angle)
{
    if (!preview_loaded || preview_playing) return 0;
    renderer_clear_frame_history();
    if (!editor_camera(zone, x, z, upper, angle)) return 0;
    const RendererPixel *pixels = renderer_get_cw_buffer();
    int size = renderer_get_width() * renderer_get_height();
    unsigned hash = 2166136261u;
    for (int i = 0; i < size; i++) { hash ^= (unsigned)pixels[i]; hash *= 16777619u; }
    hash ^= (unsigned)renderer_get_width(); hash *= 16777619u;
    hash ^= (unsigned)renderer_get_height();
    return hash;
}

EMSCRIPTEN_KEEPALIVE void editor_play(int playing)
{
    if (!preview_loaded) return;
    preview_playing = playing != 0;
    g_state.cfg_weapon_draw = preview_playing != 0;
    input_clear_keyboard(g_state.key_map);
    game_loop_ctx_init(&preview_loop, &g_state);
    if (!preview_playing) {
        g_state.obj_interp_alpha = 1.0f;
        preview_draw();
    }
}

/* Read-only playtest inspection: report the actual simulation position,
 * including teleports, rather than the imported header's spawn metadata. */
EMSCRIPTEN_KEEPALIVE int editor_player_zone(void) { return preview_loaded ? g_state.plr1.zone : -1; }
EMSCRIPTEN_KEEPALIVE int editor_conditions(void) { return (uint16_t)game_conditions; }
EMSCRIPTEN_KEEPALIVE int editor_ammunition(int gun) { return preview_loaded && gun >= 0 && gun < MAX_GUNS ? g_state.plr1.gun_data[gun].ammo : -1; }
EMSCRIPTEN_KEEPALIVE int editor_weapon_owned(int gun) { return preview_loaded && gun >= 0 && gun < MAX_GUNS ? g_state.plr1.gun_data[gun].visible != 0 : 0; }
/* Read-only native moving-room state, after level.c converts quarter-unit
 * source words into runtime 24.8 positions. */
EMSCRIPTEN_KEEPALIVE int editor_mechanism_value(int lift, int index, int field)
{
    if (!preview_loaded || index < 0 || index >= (lift ? g_state.level.num_lifts : g_state.level.num_doors)) return field == 0 ? -1 : 0;
    const uint8_t *table = lift ? g_state.level.lift_data : g_state.level.door_data;
    if (!table) return field == 0 ? -1 : 0;
    const uint8_t *record = table + index * (lift ? 20 : 22);
    switch (field) {
    case 0: return obj_w(record);
    case 1: return obj_l(record + 4);
    case 2: return obj_l(record + 10);
    case 3: return obj_l(record + 14);
    case 4: case 6: {
        const uint32_t *offsets = lift ? g_state.level.lift_wall_list_offsets : g_state.level.door_wall_list_offsets;
        const uint8_t *walls = lift ? g_state.level.lift_wall_list : g_state.level.door_wall_list;
        if (!offsets || !walls) return field == 4 ? 0 : -1;
        uint32_t start = offsets[index], end = offsets[index + 1];
        if (field == 4) return (int)(end - start);
        int mismatches = 0;
        for (uint32_t wall = start; wall < end; wall++) {
            int32_t graphic = obj_l(walls + wall * 10u + 2u);
            if (!g_state.level.graphics || graphic < 0 || g_state.level.graphics_byte_count < 28u ||
                (size_t)graphic > g_state.level.graphics_byte_count - 28u ||
                obj_l(g_state.level.graphics + graphic + (lift ? 20 : 24)) != obj_l(record + 4)) mismatches++;
        }
        return mismatches;
    }
    case 5: {
        uint8_t *zone = level_get_zone_data_ptr(&g_state.level, obj_w(record));
        return zone ? obj_l(zone + (lift ? ZONE_OFF_FLOOR : ZONE_OFF_ROOF)) : 0;
    }
    default: return 0;
    }
}
EMSCRIPTEN_KEEPALIVE int editor_player_x(void) { return preview_loaded ? g_state.plr1.xoff >> 16 : 0; }
EMSCRIPTEN_KEEPALIVE int editor_player_z(void) { return preview_loaded ? g_state.plr1.zoff >> 16 : 0; }
/* player.c updates the collision target from floor - s_height. Rendering Y
 * includes the view offset/bob, so report the simulation feet for validation. */
EMSCRIPTEN_KEEPALIVE int editor_player_feet_fixed(void) { return preview_loaded ? g_state.plr1.s_yoff + g_state.plr1.s_height : 0; }

static uint8_t *preview_object(int index)
{
    if (!preview_loaded || index < 0 || index >= MAX_OBJECTS || !g_state.level.object_data) return NULL;
    for (int slot = 0; slot <= index; slot++)
        if (obj_w(g_state.level.object_data + slot * OBJECT_SIZE) < 0) return NULL;
    return g_state.level.object_data + index * OBJECT_SIZE;
}

/* Read-only record inspection, including positions after native shot-pool
 * expansion. Field 0 is the CID; negative means no such object. */
EMSCRIPTEN_KEEPALIVE int editor_object_value(int index, int field)
{
    uint8_t *object = preview_object(index);
    if (!object) return field == 0 ? -1 : 0;
    int cid = obj_w(object);
    switch (field) {
    case 0: return cid;
    case 1: return obj_w(object + 12);
    case 2: return (int8_t)object[16];
    case 3: case 4:
        return g_state.level.object_points && cid >= 0 && cid < g_state.level.num_object_points
            ? obj_l(g_state.level.object_points + cid * 8 + (field == 4 ? 4 : 0)) : 0;
    case 5: return (int32_t)obj_w(object + 4) * 128;
    case 6: return object[18];
    default: return 0;
    }
}

static void preview_frame(void)
{
    if (!preview_loaded) return;
    if (preview_playing) {
        game_loop_tick(&g_state, &preview_loop);
        if (!g_state.running || g_state.finished_level || g_state.restart_game_requested) {
            preview_playing = 0;
            preview_report(2, g_state.plr1.zone, g_state.finished_level);
        }
    } else {
        SDL_Event event;
        while (SDL_PollEvent(&event)) { /* The host owns editor-mode input. */ }
        display_emscripten_frame_resize_poll();
        /* Resize can recreate SDL's presentation surface. Render the paused
         * world again so changing split/full view never leaves a black view. */
        preview_draw();
    }
}

int main(void)
{
    SDL_SetMainReady();
    setup_game(&g_state);
    play_game_load_shared_assets(&g_state);
    emscripten_set_main_loop(preview_frame, 0, 0);
    return 0;
}
