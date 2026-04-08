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
#include "logging.h"
#define printf ab3d_log_printf

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
        /* INI is 1-based (1 = first level, MAX_LEVELS = last). Internal cfg_start_level stays 0-based. */
        if (n >= 1 && n <= MAX_LEVELS) {
            state->cfg_start_level = (int16_t)(n - 1);
        } else if (n == 0) {
            state->cfg_start_level = 0;
            printf("[SETTINGS] start_level=0 is deprecated; use start_level=1 for the first level\n");
        } else {
            printf("[SETTINGS] start_level ignored (use 1..%d): %s\n", MAX_LEVELS, val);
        }
    } else if (strcmp(key, "infinite_health") == 0) {
        state->infinite_health = parse_bool(val) ? true : false;
    } else if (strcmp(key, "infinite_ammo") == 0) {
        state->infinite_ammo = parse_bool(val) ? true : false;
    } else if (strcmp(key, "all_weapons") == 0) {
        state->cfg_all_weapons = parse_bool(val) ? true : false;
    } else if (strcmp(key, "all_keys") == 0) {
        state->cfg_all_keys = parse_bool(val) ? true : false;
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
}

static void log_effective_settings(const GameState *state, const char *source_label)
{
    if (state->cfg_start_level >= 0) {
        printf("[SETTINGS] %s: start_level=%d infinite_health=%d infinite_ammo=%d all_weapons=%d all_keys=%d display_mode=%s render=%dx%d supersampling=%d render_threads=%d render_threads_max=%d volume=%d y_proj_scale=%d billboard_sprite_rendering_enhancement=%d weapon_draw=%d post_tint=%d weapon_post_gl=%d show_fps=%d\n",
               source_label,
               (int)state->cfg_start_level + 1,
               state->infinite_health ? 1 : 0,
               state->infinite_ammo ? 1 : 0,
               state->cfg_all_weapons ? 1 : 0,
               state->cfg_all_keys ? 1 : 0,
               display_mode_to_text(state->cfg_display_mode),
               (int)state->cfg_render_width,
               (int)state->cfg_render_height,
               (int)state->cfg_supersampling,
               state->cfg_render_threads ? 1 : 0,
               (int)state->cfg_render_threads_max,
               (int)state->cfg_volume,
               (int)state->cfg_y_proj_scale,
               state->cfg_billboard_sprite_rendering_enhancement ? 1 : 0,
               state->cfg_weapon_draw ? 1 : 0,
               state->cfg_post_tint ? 1 : 0,
               state->cfg_weapon_post_gl ? 1 : 0,
               state->cfg_show_fps ? 1 : 0);
    } else {
        printf("[SETTINGS] %s: start_level=default infinite_health=%d infinite_ammo=%d all_weapons=%d all_keys=%d display_mode=%s render=%dx%d supersampling=%d render_threads=%d render_threads_max=%d volume=%d y_proj_scale=%d billboard_sprite_rendering_enhancement=%d weapon_draw=%d post_tint=%d weapon_post_gl=%d show_fps=%d\n",
               source_label,
               state->infinite_health ? 1 : 0,
               state->infinite_ammo ? 1 : 0,
               state->cfg_all_weapons ? 1 : 0,
               state->cfg_all_keys ? 1 : 0,
               display_mode_to_text(state->cfg_display_mode),
               (int)state->cfg_render_width,
               (int)state->cfg_render_height,
               (int)state->cfg_supersampling,
               state->cfg_render_threads ? 1 : 0,
               (int)state->cfg_render_threads_max,
               (int)state->cfg_volume,
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
    apply_runtime_constraints(state);
    log_effective_settings(state, "Loaded");
    return 1;
}

void settings_load(GameState *state)
{
    const char *base = SDL_GetBasePath();
    if (base && *base) {
        char path_ini[1024];
        char path_tpl[1024];
        snprintf(path_ini, sizeof(path_ini), "%sab3d.ini", base);
        snprintf(path_tpl, sizeof(path_tpl), "%sab3d.ini.template", base);

        if (try_load_settings_file(state, path_ini, NULL)) return;
        if (try_load_settings_file(state, path_tpl, " (ab3d.ini not found)")) return;
    }

    /* Fallbacks for IDE/debug runs where users keep INI in project root. */
    if (try_load_settings_file(state, "ab3d.ini", " (from working directory)")) return;
    if (try_load_settings_file(state, "ab3d.ini.template", " (from working directory)")) return;
    if (try_load_settings_file(state, "data/ab3d.ini", " (from working directory data/)")) return;

    if (base && *base) {
        printf("[SETTINGS] No ab3d.ini or ab3d.ini.template in %s (or working directory fallbacks)\n", base);
    } else {
        printf("[SETTINGS] SDL_GetBasePath unavailable and no INI found in working directory fallbacks\n");
    }
    apply_runtime_constraints(state);
    log_effective_settings(state, "Defaults");
}

void settings_log_recap(const GameState *state)
{
    log_effective_settings(state, "Active");
}
