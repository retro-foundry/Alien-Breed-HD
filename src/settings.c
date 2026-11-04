/*
 * settings.c - Load ab3d.ini beside the executable.
 */

#include "settings.h"
#include "renderer.h"
#include "game_types.h"
#include <SDL.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif
#include "logging.h"
#define printf ab3d_log_printf

#define SETTINGS_PATH_MAX 1024

static char g_menu_options_save_path[SETTINGS_PATH_MAX] = "";
static char g_menu_options_seed_path[SETTINGS_PATH_MAX] = "";

static void apply_runtime_constraints(GameState *state);
static void log_effective_settings(const GameState *state,
                                   const char *source_label);

#if defined(__EMSCRIPTEN__)
#define SETTINGS_WEB_MOUSE_LOOK_KEY          "ab3d1.settings.mouse_look"
#define SETTINGS_WEB_INVERT_MOUSE_KEY        "ab3d1.settings.mouse_look_invert_y"
#define SETTINGS_WEB_SHOW_FPS_KEY            "ab3d1.settings.show_fps"
#define SETTINGS_WEB_VOLUME_KEY              "ab3d1.settings.volume"
#define SETTINGS_WEB_CROSSHAIR_KEY           "ab3d1.settings.crosshair"
#define SETTINGS_WEB_GAMEPAD_LOOK_SPEED_KEY  "ab3d1.settings.gamepad_look_speed"
#define SETTINGS_WEB_HUD_BARS_KEY            "ab3d1.settings.hud_bars"
#define SETTINGS_WEB_RUN_SPEED_KEY           "ab3d1.settings.run_speed"
#define SETTINGS_WEB_RUN_DEFAULT_KEY         "ab3d1.settings.run_default"

EM_JS(int, settings_web_local_storage_get_int,
      (const char *key_ptr, int default_value), {
    var key = UTF8ToString(key_ptr);
    try {
        if (typeof localStorage === 'undefined') return default_value;
        var value = localStorage.getItem(key);
        if (value === null) return default_value;
        value = String(value).toLowerCase();
        var n = parseInt(value, 10);
        if (isFinite(n) && String(n) === value) return n | 0;
        return (value === '1' || value === 'true' ||
                value === 'yes' || value === 'on') ? 1 : 0;
    } catch (e) {
        console.warn('[SETTINGS] localStorage read failed', e);
        return default_value;
    }
});

EM_JS(int, settings_web_local_storage_set_int,
      (const char *key_ptr, int value), {
    var key = UTF8ToString(key_ptr);
    try {
        if (typeof localStorage === 'undefined') return 0;
        localStorage.setItem(key, String(value | 0));
        return 1;
    } catch (e) {
        console.warn('[SETTINGS] localStorage write failed', e);
        return 0;
    }
});
#endif

static char *trim(char *s)
{
    char *end;
    while (*s && isspace((unsigned char)*s)) s++;
    if (*s == 0) return s;
    end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) {
        *end = '\0';
        end--;
    }
    return s;
}

static void rtrim_inplace(char *s)
{
    if (!s || !*s) return;
    char *end = s + strlen(s) - 1;
    while (end >= s && isspace((unsigned char)*end)) {
        *end = '\0';
        end--;
    }
}

static int parse_bool(const char *v)
{
    if (!v || !*v) return 0;
    if (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T')
        return 1;
    if (strncmp(v, "on", 2) == 0 || strncmp(v, "yes", 3) == 0)
        return 1;
    return 0;
}

static void settings_copy_path(char *dst, size_t dst_size, const char *src)
{
    if (!dst || dst_size == 0) return;
    if (!src) src = "";
    snprintf(dst, dst_size, "%s", src);
}

static void settings_remember_menu_options_paths(const char *save_path,
                                                 const char *seed_path)
{
    settings_copy_path(g_menu_options_save_path,
                       sizeof(g_menu_options_save_path),
                       save_path);
    settings_copy_path(g_menu_options_seed_path,
                       sizeof(g_menu_options_seed_path),
                       seed_path);
}

#if !defined(__EMSCRIPTEN__)
static int settings_file_exists(const char *path)
{
    FILE *f;
    if (!path || !*path) return 0;
    f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static void settings_default_menu_options_save_path(char *out, size_t out_size)
{
    char *base;

    if (!out || out_size == 0) return;
    base = SDL_GetBasePath();
    if (base && *base) {
        snprintf(out, out_size, "%sab3d.ini", base);
    } else {
        snprintf(out, out_size, "ab3d.ini");
    }
    if (base) SDL_free(base);
}

static int settings_path_with_suffix(char *out, size_t out_size,
                                     const char *path, const char *suffix)
{
    int n;
    if (!out || out_size == 0 || !path || !suffix) return 0;
#ifdef AB3D_DOS
    /* Keep the temporary INI in the same directory with a DOS 8.3 name.
     * Appending .tmp to ab3d.ini makes option saves fail on real DOS. */
    const char *name = path;
    for (const char *p = path; *p; p++) {
        if (*p == '/' || *p == '\\' || *p == ':') name = p + 1;
    }
    const char *extension = strrchr(name, '.');
    size_t stem = extension ? (size_t)(extension - path) : strlen(path);
    n = snprintf(out, out_size, "%.*s%s", (int)stem, path, suffix);
#else
    n = snprintf(out, out_size, "%s%s", path, suffix);
#endif
    return n >= 0 && n < (int)out_size;
}

static int settings_extract_line_key(const char *line, char *out, size_t out_size)
{
    const char *p;
    const char *eq;
    const char *end;
    size_t n;

    if (!line || !out || out_size == 0) return 0;
    out[0] = '\0';

    p = line;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p == '\0' || *p == '#' || *p == ';') return 0;

    eq = strchr(p, '=');
    if (!eq) return 0;
    end = eq;
    while (end > p && isspace((unsigned char)*(end - 1))) end--;
    n = (size_t)(end - p);
    if (n == 0 || n >= out_size) return 0;

    for (size_t i = 0; i < n; i++) {
        out[i] = (char)tolower((unsigned char)p[i]);
    }
    out[n] = '\0';
    return 1;
}

static int settings_write_menu_options_line(FILE *out,
                                            const char *line,
                                            const GameState *state,
                                            int *saw_mouse_look,
                                            int *saw_invert_mouse,
                                            int *saw_show_fps,
                                            int *saw_volume,
                                            int *saw_crosshair,
                                            int *saw_gamepad_look_speed,
                                            int *saw_run_default,
                                            int *saw_hud_bars,
                                            int *saw_run_speed)
{
    char key[64];

    if (!settings_extract_line_key(line, key, sizeof(key))) {
        return fputs(line, out) >= 0;
    }

    if (strcmp(key, "mouse_look") == 0) {
        *saw_mouse_look = 1;
        return fprintf(out, "mouse_look=%d\n",
                       state->cfg_mouse_look ? 1 : 0) >= 0;
    }
    if (strcmp(key, "mouse_look_invert_y") == 0) {
        *saw_invert_mouse = 1;
        return fprintf(out, "mouse_look_invert_y=%d\n",
                       state->cfg_mouse_look_invert_y ? 1 : 0) >= 0;
    }
    if (strcmp(key, "show_fps") == 0 || strcmp(key, "fps_counter") == 0) {
        *saw_show_fps = 1;
        return fprintf(out, "%s=%d\n",
                       strcmp(key, "fps_counter") == 0 ? "fps_counter" : "show_fps",
                       state->cfg_show_fps ? 1 : 0) >= 0;
    }
    if (strcmp(key, "volume") == 0) {
        *saw_volume = 1;
        return fprintf(out, "volume=%d\n",
                       (int)state->cfg_volume) >= 0;
    }
    if (strcmp(key, "crosshair") == 0 ||
        strcmp(key, "crosshair_colour") == 0 ||
        strcmp(key, "crosshair_color") == 0 ||
        strcmp(key, "misc.crosshair_colour") == 0) {
        *saw_crosshair = 1;
        return fprintf(out, "%s=%d\n", key,
                       (int)(state->cfg_crosshair_colour & 7u)) >= 0;
    }
    if (strcmp(key, "gamepad_look_speed") == 0 ||
        strcmp(key, "gamepad_turn_speed") == 0 ||
        strcmp(key, "pad_turn_speed") == 0) {
        *saw_gamepad_look_speed = 1;
        return fprintf(out, "%s=%d\n", key,
                       (int)state->cfg_gamepad_look_speed) >= 0;
    }
    if (strcmp(key, "run_speed") == 0) {
        *saw_run_speed = 1;
        return fprintf(out, "run_speed=%d\n", (int)state->cfg_run_speed) >= 0;
    }
    if (strcmp(key, "hud_bars") == 0) {
        *saw_hud_bars = 1;
        return fprintf(out, "hud_bars=%d\n", state->cfg_hud_bars ? 1 : 0) >= 0;
    }
    if (strcmp(key, "run_default") == 0 ||
        strcmp(key, "always_run") == 0) {
        *saw_run_default = 1;
        return fprintf(out, "%s=%d\n", key,
                       state->cfg_run_default ? 1 : 0) >= 0;
    }

    return fputs(line, out) >= 0;
}

static int settings_append_missing_menu_options(FILE *out,
                                                const GameState *state,
                                                int saw_mouse_look,
                                                int saw_invert_mouse,
                                                int saw_show_fps,
                                                int saw_volume,
                                                int saw_crosshair,
                                                int saw_gamepad_look_speed,
                                                int saw_run_default,
                                                int saw_hud_bars,
                                                int saw_run_speed)
{
    if (saw_mouse_look && saw_invert_mouse && saw_show_fps && saw_volume &&
        saw_crosshair && saw_gamepad_look_speed && saw_run_default && saw_hud_bars && saw_run_speed) {
        return 1;
    }
    if (fprintf(out, "\n# Runtime menu options\n") < 0) return 0;
    if (!saw_mouse_look &&
        fprintf(out, "mouse_look=%d\n", state->cfg_mouse_look ? 1 : 0) < 0) {
        return 0;
    }
    if (!saw_invert_mouse &&
        fprintf(out, "mouse_look_invert_y=%d\n",
                state->cfg_mouse_look_invert_y ? 1 : 0) < 0) {
        return 0;
    }
    if (!saw_show_fps &&
        fprintf(out, "show_fps=%d\n", state->cfg_show_fps ? 1 : 0) < 0) {
        return 0;
    }
    if (!saw_volume &&
        fprintf(out, "volume=%d\n", (int)state->cfg_volume) < 0) {
        return 0;
    }
    if (!saw_crosshair &&
        fprintf(out, "crosshair=%d\n",
                (int)(state->cfg_crosshair_colour & 7u)) < 0) {
        return 0;
    }
    if (!saw_gamepad_look_speed &&
        fprintf(out, "gamepad_look_speed=%d\n",
                (int)state->cfg_gamepad_look_speed) < 0) {
        return 0;
    }
    if (!saw_run_speed &&
        fprintf(out, "run_speed=%d\n", (int)state->cfg_run_speed) < 0) {
        return 0;
    }
    if (!saw_hud_bars &&
        fprintf(out, "hud_bars=%d\n", state->cfg_hud_bars ? 1 : 0) < 0) {
        return 0;
    }
    if (!saw_run_default &&
        fprintf(out, "run_default=%d\n",
                state->cfg_run_default ? 1 : 0) < 0) {
        return 0;
    }
    return 1;
}

static void settings_save_menu_options_to_ini(const GameState *state)
{
    char target[SETTINGS_PATH_MAX];
    char tmp[SETTINGS_PATH_MAX];
    const char *source = NULL;
    FILE *in = NULL;
    FILE *out = NULL;
    char line[1024];
    int saw_mouse_look = 0;
    int saw_invert_mouse = 0;
    int saw_show_fps = 0;
    int saw_volume = 0;
    int saw_crosshair = 0;
    int saw_gamepad_look_speed = 0;
    int saw_run_default = 0;
    int saw_hud_bars = 0;
    int saw_run_speed = 0;
    int ok = 1;

    if (!state) return;

    if (g_menu_options_save_path[0]) {
        settings_copy_path(target, sizeof(target), g_menu_options_save_path);
    } else {
        settings_default_menu_options_save_path(target, sizeof(target));
    }
    if (!target[0]) {
        printf("[SETTINGS] menu options save skipped: no target path\n");
        return;
    }

    if (settings_file_exists(target)) {
        source = target;
    } else if (settings_file_exists(g_menu_options_seed_path)) {
        source = g_menu_options_seed_path;
    }

    if (!settings_path_with_suffix(tmp, sizeof(tmp), target, ".tmp")) {
        printf("[SETTINGS] menu options save skipped: path too long\n");
        return;
    }

    if (source) {
        in = fopen(source, "rb");
        if (!in) {
            printf("[SETTINGS] menu options save: could not read %s\n", source);
            return;
        }
    }

    out = fopen(tmp, "wb");
    if (!out) {
        if (in) fclose(in);
        printf("[SETTINGS] menu options save: could not write %s\n", tmp);
        return;
    }

    if (in) {
        while (fgets(line, sizeof(line), in)) {
            if (!settings_write_menu_options_line(out, line, state,
                                                  &saw_mouse_look,
                                                  &saw_invert_mouse,
                                                  &saw_show_fps,
                                                  &saw_volume,
                                                  &saw_crosshair,
                                                  &saw_gamepad_look_speed,
                                                  &saw_run_default,
                                                  &saw_hud_bars,
                                                  &saw_run_speed)) {
                ok = 0;
                break;
            }
        }
        if (ferror(in)) ok = 0;
        fclose(in);
    }

    if (ok) {
        ok = settings_append_missing_menu_options(out, state,
                                                  saw_mouse_look,
                                                  saw_invert_mouse,
                                                  saw_show_fps,
                                                  saw_volume,
                                                  saw_crosshair,
                                                  saw_gamepad_look_speed,
                                                  saw_run_default,
                                                  saw_hud_bars,
                                                  saw_run_speed);
    }
    if (fclose(out) != 0) ok = 0;

    if (!ok) {
        remove(tmp);
        printf("[SETTINGS] menu options save failed while updating %s\n",
               target);
        return;
    }

    if (rename(tmp, target) != 0) {
        remove(target);
        if (rename(tmp, target) != 0) {
            remove(tmp);
            printf("[SETTINGS] menu options save failed replacing %s\n",
                   target);
            return;
        }
    }
}
#endif

static void settings_apply_persistent_menu_options(GameState *state)
{
#if defined(__EMSCRIPTEN__)
    if (!state) return;
    state->cfg_mouse_look =
        settings_web_local_storage_get_int(
            SETTINGS_WEB_MOUSE_LOOK_KEY,
            state->cfg_mouse_look ? 1 : 0) != 0;
    state->cfg_mouse_look_invert_y =
        settings_web_local_storage_get_int(
            SETTINGS_WEB_INVERT_MOUSE_KEY,
            state->cfg_mouse_look_invert_y ? 1 : 0) != 0;
    state->cfg_show_fps =
        settings_web_local_storage_get_int(
            SETTINGS_WEB_SHOW_FPS_KEY,
            state->cfg_show_fps ? 1 : 0) != 0;
    {
        int volume = settings_web_local_storage_get_int(
            SETTINGS_WEB_VOLUME_KEY,
            (int)state->cfg_volume);
        if (volume >= 0 && volume <= 100) {
            state->cfg_volume = (int16_t)volume;
        }
    }
    {
        int crosshair = settings_web_local_storage_get_int(
            SETTINGS_WEB_CROSSHAIR_KEY,
            (int)(state->cfg_crosshair_colour & 7u));
        if (crosshair >= 0 && crosshair <= 7) {
            state->cfg_crosshair_colour = (uint8_t)crosshair;
        }
    }
    {
        int speed = settings_web_local_storage_get_int(
            SETTINGS_WEB_GAMEPAD_LOOK_SPEED_KEY,
            (int)state->cfg_gamepad_look_speed);
        if (speed >= 25 && speed <= 800) {
            state->cfg_gamepad_look_speed = (int16_t)speed;
        }
    }
    {
        int speed = settings_web_local_storage_get_int(
            SETTINGS_WEB_RUN_SPEED_KEY, state->cfg_run_speed);
        if (speed >= RUN_SPEED_PERCENT_MIN && speed <= RUN_SPEED_PERCENT_MAX)
            state->cfg_run_speed = (int16_t)speed;
    }
    state->cfg_hud_bars =
        settings_web_local_storage_get_int(
            SETTINGS_WEB_HUD_BARS_KEY, state->cfg_hud_bars ? 1 : 0) != 0;
    state->cfg_run_default =
        settings_web_local_storage_get_int(
            SETTINGS_WEB_RUN_DEFAULT_KEY,
            state->cfg_run_default ? 1 : 0) != 0;
#else
    (void)state;
#endif
}

static void settings_finalize_load(GameState *state, const char *source_label)
{
    apply_runtime_constraints(state);
    settings_apply_persistent_menu_options(state);
    log_effective_settings(state, source_label);
}

static int parse_display_mode_value(const char *v, int8_t *out_mode)
{
    char buf[64];
    size_t n = 0;
    if (!v || !out_mode) return 0;

    while (*v && isspace((unsigned char)*v)) v++;
    while (*v && n + 1 < sizeof(buf)) {
        buf[n++] = (char)tolower((unsigned char)*v++);
    }
    while (n > 0 && isspace((unsigned char)buf[n - 1])) n--;
    buf[n] = '\0';

    if (strcmp(buf, "fullscreen") == 0 ||
        strcmp(buf, "fullscreen_desktop") == 0 ||
        strcmp(buf, "desktop") == 0 ||
        strcmp(buf, "1") == 0 ||
        strcmp(buf, "true") == 0 ||
        strcmp(buf, "yes") == 0 ||
        strcmp(buf, "on") == 0) {
        *out_mode = 1;
        return 1;
    }
    if (strcmp(buf, "windowed") == 0 ||
        strcmp(buf, "window") == 0 ||
        strcmp(buf, "0") == 0 ||
        strcmp(buf, "false") == 0 ||
        strcmp(buf, "no") == 0 ||
        strcmp(buf, "off") == 0) {
        *out_mode = 0;
        return 1;
    }
    return 0;
}

static const char *display_mode_to_text(int8_t mode)
{
    switch (mode) {
    case 1:  return "fullscreen";
    case 0:  return "windowed";
    default: return "build-default";
    }
}

static void apply_line(GameState *state, char *line)
{
    char *eq = strchr(line, '=');
    if (!eq) return;
    *eq = '\0';
    rtrim_inplace(line);
    char *key = trim(line);
    char *val = trim(eq + 1);
    if (*key == '\0') return;

    for (char *p = key; *p; p++) *p = (char)tolower((unsigned char)*p);

    if (strcmp(key, "start_level") == 0) {
        int n = atoi(val);
        /* INI: title menu always appears. New Game uses 0 = level 1;
         * 1..MAX_LEVELS are fixed 1-based levels. */
        if (n >= 1 && n <= MAX_LEVELS) {
            state->cfg_start_level = (int16_t)(n - 1);
        } else if (n == 0) {
            state->cfg_start_level = -1;
        } else {
            printf("[SETTINGS] start_level ignored (use 0..%d): %s\n", MAX_LEVELS, val);
        }
    } else if (strcmp(key, "infinite_health") == 0) {
        state->infinite_health = parse_bool(val) ? true : false;
    } else if (strcmp(key, "infinite_ammo") == 0) {
        state->infinite_ammo = parse_bool(val) ? true : false;
    } else if (strcmp(key, "all_weapons") == 0) {
        state->cfg_all_weapons = parse_bool(val) ? true : false;
    } else if (strcmp(key, "flamethrower_weapon") == 0 ||
               strcmp(key, "flame_thrower_weapon") == 0 ||
               strcmp(key, "flamethrower") == 0) {
        state->cfg_flamethrower_weapon = parse_bool(val) ? true : false;
    } else if (strcmp(key, "all_keys") == 0) {
        state->cfg_all_keys = parse_bool(val) ? true : false;
    } else if (strcmp(key, "mouse_look") == 0) {
        state->cfg_mouse_look = parse_bool(val) ? true : false;
    } else if (strcmp(key, "mouse_look_invert_y") == 0) {
        state->cfg_mouse_look_invert_y = parse_bool(val) ? true : false;
    } else if (strcmp(key, "gamepad_look_speed") == 0 ||
               strcmp(key, "gamepad_turn_speed") == 0 ||
               strcmp(key, "pad_turn_speed") == 0) {
        int n = atoi(val);
        if (n >= 25 && n <= 800) {
            state->cfg_gamepad_look_speed = (int16_t)n;
        } else {
            printf("[SETTINGS] gamepad_look_speed ignored (use 25..800, 100=old speed): %s\n", val);
        }
    } else if (strcmp(key, "quicksave_load") == 0 ||
               strcmp(key, "quick_save_load") == 0 ||
               strcmp(key, "quickload_save") == 0) {
        state->cfg_quicksave_load = parse_bool(val) ? true : false;
    } else if (strcmp(key, "marine_hitscan_projectiles") == 0 ||
               strcmp(key, "marine_projectiles") == 0) {
        state->cfg_marine_hitscan_projectiles = parse_bool(val) ? true : false;
    } else if (strcmp(key, "run_speed") == 0) {
        int n = atoi(val);
        if (n >= RUN_SPEED_PERCENT_MIN && n <= RUN_SPEED_PERCENT_MAX)
            state->cfg_run_speed = (int16_t)n;
        else
            printf("[SETTINGS] run_speed ignored (use 50..200): %s\n", val);
    } else if (strcmp(key, "hud_bars") == 0) {
        state->cfg_hud_bars = parse_bool(val) ? true : false;
    } else if (strcmp(key, "run_default") == 0 ||
               strcmp(key, "always_run") == 0) {
        state->cfg_run_default = parse_bool(val) ? true : false;
    } else if (strcmp(key, "footsteps_water_only") == 0 ||
               strcmp(key, "water_footsteps_only") == 0) {
        state->cfg_footsteps_water_only = parse_bool(val) ? true : false;
    } else if (strcmp(key, "crosshair") == 0 ||
               strcmp(key, "crosshair_colour") == 0 ||
               strcmp(key, "crosshair_color") == 0 ||
               strcmp(key, "misc.crosshair_colour") == 0) {
        int n = atoi(val);
        if (n >= 0 && n <= 7) {
            state->cfg_crosshair_colour = (uint8_t)n;
        } else {
            printf("[SETTINGS] crosshair ignored (use 0..7, 0=off): %s\n", val);
        }
    } else if (strcmp(key, "display_mode") == 0 ||
               strcmp(key, "window_mode") == 0) {
        int8_t mode = -1;
        if (parse_display_mode_value(val, &mode)) {
            state->cfg_display_mode = mode;
        } else {
            printf("[SETTINGS] display_mode ignored (use windowed/fullscreen): %s\n", val);
        }
    } else if (strcmp(key, "fullscreen") == 0) {
        state->cfg_display_mode = parse_bool(val) ? 1 : 0;
    } else if (strcmp(key, "render_width") == 0) {
        int n = atoi(val);
        if (n >= 96 && n <= RENDER_INTERNAL_MAX_DIM) {
            state->cfg_render_width = (int16_t)n;
        } else {
            printf("[SETTINGS] render_width ignored (use 96..%d): %s\n", RENDER_INTERNAL_MAX_DIM, val);
        }
    } else if (strcmp(key, "render_height") == 0) {
        int n = atoi(val);
        if (n >= 80 && n <= RENDER_INTERNAL_MAX_DIM) {
            state->cfg_render_height = (int16_t)n;
        } else {
            printf("[SETTINGS] render_height ignored (use 80..%d): %s\n", RENDER_INTERNAL_MAX_DIM, val);
        }
    } else if (strcmp(key, "supersampling") == 0) {
        int n = atoi(val);
        if (n >= 1 && n <= 4) {
            state->cfg_supersampling = (int16_t)n;
        } else {
            printf("[SETTINGS] supersampling ignored (use 1..4): %s\n", val);
        }
    } else if (strcmp(key, "render_threads") == 0) {
        state->cfg_render_threads = parse_bool(val) ? true : false;
    } else if (strcmp(key, "render_threads_max") == 0) {
        int n = atoi(val);
        if (n >= 0 && n <= 64) {
            state->cfg_render_threads_max = (int16_t)n;
        } else {
            printf("[SETTINGS] render_threads_max ignored (use 0..64): %s\n", val);
        }
    } else if (strcmp(key, "volume") == 0) {
        int n = atoi(val);
        if (n >= 0 && n <= 100) {
            state->cfg_volume = (int16_t)n;
        } else {
            printf("[SETTINGS] volume ignored (use 0..100): %s\n", val);
        }
    } else if (strcmp(key, "audio_buffer_samples") == 0) {
        int n = atoi(val);
        if (n == 0 || (n >= 256 && n <= 4096)) {
            state->cfg_audio_buffer_samples = (int16_t)n;
        } else {
            printf("[SETTINGS] audio_buffer_samples ignored (use 0 or 256..4096): %s\n", val);
        }
    } else if (strcmp(key, "y_proj_scale") == 0) {
        int n = atoi(val);
        if (n >= 25 && n <= 1000) {
            state->cfg_y_proj_scale = (int16_t)n;
        } else {
            printf("[SETTINGS] y_proj_scale ignored (use 25..1000, 100=default): %s\n", val);
        }
    } else if (strcmp(key, "billboard sprite rendering enhancement") == 0 ||
               strcmp(key, "billboard_sprite_rendering_enhancement") == 0) {
        state->cfg_billboard_sprite_rendering_enhancement = parse_bool(val) ? true : false;
    } else if (strcmp(key, "show_fps") == 0 || strcmp(key, "fps_counter") == 0) {
        state->cfg_show_fps = parse_bool(val) ? true : false;
    } else if (strcmp(key, "weapon_draw") == 0) {
        state->cfg_weapon_draw = parse_bool(val) ? true : false;
    } else if (strcmp(key, "post_tint") == 0) {
        state->cfg_post_tint = parse_bool(val) ? true : false;
    } else if (strcmp(key, "weapon_post_gl") == 0) {
        state->cfg_weapon_post_gl = parse_bool(val) ? true : false;
    }
}

static void apply_runtime_constraints(GameState *state)
{
#ifdef AB3D_NO_THREADS
    static int logged = 0;
    if (!logged) {
        printf("[SETTINGS] AB3D_NO_THREADS build: render_threads is ignored; forcing single-threaded renderer\n");
        logged = 1;
    }
    state->cfg_render_threads = false;
    state->cfg_render_threads_max = 0;
#else
    (void)state;
#endif
    if (state->cfg_volume < 0)
        state->cfg_volume = 0;
    else if (state->cfg_volume > 100)
        state->cfg_volume = 100;
    if (state->cfg_run_speed < RUN_SPEED_PERCENT_MIN)
        state->cfg_run_speed = RUN_SPEED_PERCENT_MIN;
    else if (state->cfg_run_speed > RUN_SPEED_PERCENT_MAX)
        state->cfg_run_speed = RUN_SPEED_PERCENT_MAX;
    if (state->cfg_gamepad_look_speed < 25)
        state->cfg_gamepad_look_speed = 25;
    else if (state->cfg_gamepad_look_speed > 800)
        state->cfg_gamepad_look_speed = 800;
    state->cfg_crosshair_colour &= 7u;
}

static void log_effective_settings(const GameState *state, const char *source_label)
{
    if (state->cfg_start_level >= 0) {
        printf("[SETTINGS] %s: start_level=%d infinite_health=%d infinite_ammo=%d all_weapons=%d flamethrower_weapon=%d all_keys=%d mouse_look=%d mouse_look_invert_y=%d gamepad_look_speed=%d quicksave_load=%d marine_hitscan_projectiles=%d run_default=%d run_speed=%d footsteps_water_only=%d crosshair=%d display_mode=%s render=%dx%d supersampling=%d render_threads=%d render_threads_max=%d volume=%d audio_buffer_samples=%d y_proj_scale=%d billboard_sprite_rendering_enhancement=%d weapon_draw=%d post_tint=%d weapon_post_gl=%d show_fps=%d\n",
               source_label,
               (int)state->cfg_start_level + 1,
               state->infinite_health ? 1 : 0,
               state->infinite_ammo ? 1 : 0,
               state->cfg_all_weapons ? 1 : 0,
               state->cfg_flamethrower_weapon ? 1 : 0,
               state->cfg_all_keys ? 1 : 0,
               state->cfg_mouse_look ? 1 : 0,
               state->cfg_mouse_look_invert_y ? 1 : 0,
               (int)state->cfg_gamepad_look_speed,
               state->cfg_quicksave_load ? 1 : 0,
               state->cfg_marine_hitscan_projectiles ? 1 : 0,
               state->cfg_run_default ? 1 : 0,
               (int)state->cfg_run_speed,
               state->cfg_footsteps_water_only ? 1 : 0,
               (int)state->cfg_crosshair_colour,
               display_mode_to_text(state->cfg_display_mode),
               (int)state->cfg_render_width,
               (int)state->cfg_render_height,
               (int)state->cfg_supersampling,
               state->cfg_render_threads ? 1 : 0,
               (int)state->cfg_render_threads_max,
               (int)state->cfg_volume,
               (int)state->cfg_audio_buffer_samples,
               (int)state->cfg_y_proj_scale,
               state->cfg_billboard_sprite_rendering_enhancement ? 1 : 0,
               state->cfg_weapon_draw ? 1 : 0,
               state->cfg_post_tint ? 1 : 0,
               state->cfg_weapon_post_gl ? 1 : 0,
               state->cfg_show_fps ? 1 : 0);
    } else {
         printf("[SETTINGS] %s: start_level=0 infinite_health=%d infinite_ammo=%d all_weapons=%d flamethrower_weapon=%d all_keys=%d mouse_look=%d mouse_look_invert_y=%d gamepad_look_speed=%d quicksave_load=%d marine_hitscan_projectiles=%d run_default=%d run_speed=%d footsteps_water_only=%d crosshair=%d display_mode=%s render=%dx%d supersampling=%d render_threads=%d render_threads_max=%d volume=%d audio_buffer_samples=%d y_proj_scale=%d billboard_sprite_rendering_enhancement=%d weapon_draw=%d post_tint=%d weapon_post_gl=%d show_fps=%d\n",
               source_label,
               state->infinite_health ? 1 : 0,
               state->infinite_ammo ? 1 : 0,
               state->cfg_all_weapons ? 1 : 0,
               state->cfg_flamethrower_weapon ? 1 : 0,
               state->cfg_all_keys ? 1 : 0,
               state->cfg_mouse_look ? 1 : 0,
               state->cfg_mouse_look_invert_y ? 1 : 0,
               (int)state->cfg_gamepad_look_speed,
               state->cfg_quicksave_load ? 1 : 0,
               state->cfg_marine_hitscan_projectiles ? 1 : 0,
               state->cfg_run_default ? 1 : 0,
               (int)state->cfg_run_speed,
               state->cfg_footsteps_water_only ? 1 : 0,
               (int)state->cfg_crosshair_colour,
               display_mode_to_text(state->cfg_display_mode),
               (int)state->cfg_render_width,
               (int)state->cfg_render_height,
               (int)state->cfg_supersampling,
               state->cfg_render_threads ? 1 : 0,
               (int)state->cfg_render_threads_max,
               (int)state->cfg_volume,
               (int)state->cfg_audio_buffer_samples,
               (int)state->cfg_y_proj_scale,
               state->cfg_billboard_sprite_rendering_enhancement ? 1 : 0,
               state->cfg_weapon_draw ? 1 : 0,
               state->cfg_post_tint ? 1 : 0,
               state->cfg_weapon_post_gl ? 1 : 0,
               state->cfg_show_fps ? 1 : 0);
    }
}

static void parse_file(GameState *state, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return;
    char buf[512];
    while (fgets(buf, sizeof(buf), f)) {
        char *cr = strchr(buf, '\r');
        if (cr) *cr = '\0';
        char *line = trim(buf);
        if (*line == '\0' || *line == '#' || *line == ';')
            continue;
        apply_line(state, line);
    }
    fclose(f);
}

static int try_load_settings_file(GameState *state, const char *path, const char *label)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    printf("[SETTINGS] Loading INI: %s%s\n", path, label ? label : "");
    parse_file(state, path);
    settings_finalize_load(state, "Loaded");
    return 1;
}

void settings_load(GameState *state)
{
    char *base = SDL_GetBasePath();
    char default_save_path[SETTINGS_PATH_MAX] = "ab3d.ini";

    if (base && *base) {
        char path_ini[SETTINGS_PATH_MAX];
        char path_tpl[SETTINGS_PATH_MAX];
        snprintf(path_ini, sizeof(path_ini), "%sab3d.ini", base);
        snprintf(path_tpl, sizeof(path_tpl), "%sab3d.ini.template", base);
        settings_copy_path(default_save_path, sizeof(default_save_path),
                           path_ini);

        if (try_load_settings_file(state, path_ini, NULL)) {
            settings_remember_menu_options_paths(path_ini, path_ini);
            SDL_free(base);
            return;
        }
        if (try_load_settings_file(state, path_tpl, " (ab3d.ini not found)")) {
            settings_remember_menu_options_paths(path_ini, path_tpl);
            SDL_free(base);
            return;
        }
    }

    /* Fallbacks for IDE/debug runs where users keep INI in project root. */
    if (try_load_settings_file(state, "ab3d.ini", " (from working directory)")) {
        settings_remember_menu_options_paths("ab3d.ini", "ab3d.ini");
        if (base) SDL_free(base);
        return;
    }
    if (try_load_settings_file(state, "ab3d.ini.template", " (from working directory)")) {
        settings_remember_menu_options_paths("ab3d.ini", "ab3d.ini.template");
        if (base) SDL_free(base);
        return;
    }
    if (try_load_settings_file(state, "data/ab3d.ini", " (from working directory data/)")) {
        settings_remember_menu_options_paths("data/ab3d.ini", "data/ab3d.ini");
        if (base) SDL_free(base);
        return;
    }

    if (base && *base) {
        printf("[SETTINGS] No ab3d.ini or ab3d.ini.template in %s (or working directory fallbacks)\n", base);
    } else {
        printf("[SETTINGS] SDL_GetBasePath unavailable and no INI found in working directory fallbacks\n");
    }
    settings_remember_menu_options_paths(default_save_path, NULL);
    if (base) SDL_free(base);
    settings_finalize_load(state, "Defaults");
}

void settings_save_menu_options(const GameState *state)
{
    if (!state) return;
#if defined(__EMSCRIPTEN__)
    if (!settings_web_local_storage_set_int(
            SETTINGS_WEB_MOUSE_LOOK_KEY,
            state->cfg_mouse_look ? 1 : 0) ||
        !settings_web_local_storage_set_int(
            SETTINGS_WEB_INVERT_MOUSE_KEY,
            state->cfg_mouse_look_invert_y ? 1 : 0) ||
        !settings_web_local_storage_set_int(
            SETTINGS_WEB_SHOW_FPS_KEY,
            state->cfg_show_fps ? 1 : 0) ||
        !settings_web_local_storage_set_int(
            SETTINGS_WEB_VOLUME_KEY,
            (int)state->cfg_volume) ||
        !settings_web_local_storage_set_int(
            SETTINGS_WEB_CROSSHAIR_KEY,
            (int)(state->cfg_crosshair_colour & 7u)) ||
        !settings_web_local_storage_set_int(
            SETTINGS_WEB_GAMEPAD_LOOK_SPEED_KEY,
            (int)state->cfg_gamepad_look_speed) ||
        !settings_web_local_storage_set_int(
            SETTINGS_WEB_RUN_SPEED_KEY,
            (int)state->cfg_run_speed) ||
        !settings_web_local_storage_set_int(
            SETTINGS_WEB_RUN_DEFAULT_KEY,
            state->cfg_run_default ? 1 : 0) ||
        !settings_web_local_storage_set_int(
            SETTINGS_WEB_HUD_BARS_KEY,
            state->cfg_hud_bars ? 1 : 0)) {
        printf("[SETTINGS] menu options localStorage save failed\n");
    }
#else
    settings_save_menu_options_to_ini(state);
#endif
}

void settings_log_recap(const GameState *state)
{
    log_effective_settings(state, "Active");
}
