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
    g_state.obj_interp_alpha = 1.0f;
    preview_loaded = 1;
    game_loop_ctx_init(&preview_loop, &g_state);
    preview_draw();
    preview_report(1, g_state.plr1.zone, level_zone_slot_count(&g_state.level));
    return 1;
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
    if (!preview_playing) preview_draw();
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
