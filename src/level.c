/*
 * Alien Breed 3D I - PC Port
 * level.c - Level data parsing
 *
 * Translated from: AB3DI.s blag: section (~line 722-848)
 */

#include "level.h"
#include "game_types.h"
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "logging.h"
#define printf ab3d_log_printf

#if defined(__clang__) || defined(__GNUC__)
#define AB3D_ATTR_UNUSED __attribute__((unused))
#else
#define AB3D_ATTR_UNUSED
#endif

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define AB3D_THREAD_LOCAL _Thread_local
#elif defined(_MSC_VER)
#define AB3D_THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define AB3D_THREAD_LOCAL __thread
#else
#define AB3D_THREAD_LOCAL
#endif

/* Helper to read big-endian 16-bit word from buffer */
static int16_t read_word(const uint8_t *p)
{
    return (int16_t)((p[0] << 8) | p[1]);
}

/* Helper to read big-endian 32-bit long from buffer */
static int32_t read_long(const uint8_t *p)
{
    return (int32_t)((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
}

static void level_build_zone_index_by_data_offset(LevelState *level);

/* 0=static, 1=pulse, 2=flicker, 3=fire. Amiga uses high byte only (lsr.w #8,d3; tst.b d3). */
static inline unsigned zone_bright_anim_type(int16_t word)
{
    unsigned hi = (unsigned)((uint16_t)word >> 8) & 0xFFu;
    if (hi >= 1u && hi <= 3u) return hi;
    return 0;
}
static const char *zone_anim_flag_name(unsigned t)
{
    if (t == 1) return "pulse";
    if (t == 2) return "flicker";
    if (t == 3) return "fire";
    return "static";
}

static void write_word_be(uint8_t *p, int16_t v)
{
    p[0] = (uint8_t)((uint16_t)v >> 8);
    p[1] = (uint8_t)(uint16_t)v;
}
static void write_long_be(uint8_t *p, int32_t v)
{
    p[0] = (uint8_t)((uint32_t)v >> 24);
    p[1] = (uint8_t)((uint32_t)v >> 16);
    p[2] = (uint8_t)((uint32_t)v >> 8);
    p[3] = (uint8_t)(uint32_t)v;
}

static int32_t pick_next_section_offset(int32_t floor_offset, int32_t data_size,
                                        const int32_t *candidates, int candidate_count)
{
    int32_t next = data_size;
    int found = 0;
    for (int i = 0; i < candidate_count; i++) {
        int32_t off = candidates[i];
        if (off > floor_offset && off <= data_size) {
            if (!found || off < next) {
                next = off;
                found = 1;
            }
        }
    }
    if (!found) return -1;
    return next;
}

static int level_expand_nasty_shot_pool(LevelState *level)
{
    const int old_slots = PLAYER_SHOT_SLOT_COUNT;
    const int new_slots = NASTY_SHOT_SLOT_COUNT;
    int old_point_count;
    int added_slots;
    int new_point_count;
    int next_cid;
    uint8_t *new_nasty;
    uint8_t *new_points;

    if (!level || !level->nasty_shot_data) return -1;
    if (new_slots <= old_slots) {
        level->other_nasty_data = level->nasty_shot_data + (size_t)old_slots * OBJECT_SIZE;
        return 0;
    }

    old_point_count = (level->num_object_points > 0) ? (int)level->num_object_points : 0;
    added_slots = new_slots - old_slots;
    new_point_count = old_point_count + added_slots;
    if (new_point_count > 32767) return -1;

    new_nasty = (uint8_t *)calloc((size_t)new_slots * OBJECT_SIZE + (size_t)new_slots * 64u, 1);
    if (!new_nasty) return -1;
    memcpy(new_nasty, level->nasty_shot_data, (size_t)old_slots * OBJECT_SIZE);

    new_points = (uint8_t *)calloc((size_t)new_point_count, 8u);
    if (!new_points) {
        free(new_nasty);
        return -1;
    }
    if (level->object_points && old_point_count > 0) {
        memcpy(new_points, level->object_points, (size_t)old_point_count * 8u);
    }

    next_cid = old_point_count;
    for (int i = old_slots; i < new_slots; i++) {
        uint8_t *slot = new_nasty + (size_t)i * OBJECT_SIZE;
        write_word_be(slot + 0, (int16_t)next_cid);
        write_word_be(slot + 12, (int16_t)-1);
        next_cid++;
    }

    level->nasty_shot_data = new_nasty;
    level->other_nasty_data = new_nasty + (size_t)new_slots * OBJECT_SIZE;
    level->object_points = new_points;
    level->num_object_points = (int16_t)new_point_count;
    return 0;
}

/* Forward declare for use in level_parse. */
static void log_broken_floor_line_connects(LevelState *level);

/* Map zone value from file (may be block ID at zd+0) to zone index (0..num_zones-1). */
static int zone_file_to_index(const uint8_t *ld, const uint8_t *zone_adds, int num_zones, int16_t zone_from_file)
{
    if (num_zones <= 0 || zone_from_file < 0) return -1;
    for (int z = 0; z < num_zones; z++) {
        int32_t zoff = read_long(zone_adds + (size_t)z * 4u);
        if (zoff < 0) continue;
        const uint8_t *zd = ld + zoff;
        if (read_word(zd + 0) == zone_from_file)
            return z;
    }
    if (zone_from_file < num_zones) return (int)zone_from_file;
    return -1;
}

/* Amiga door/lift format (Anims.s): 18-byte fixed header then variable wall list.
 * Header: Bottom(w), Top(w), curr(w), dir(w), Ptr(l), zone(w), conditions(w), 2 bytes at 16-17.
 * Wall list at 18: (wall_number(w), ptr(l), graphic(l)) until wall_number < 0, then +2.
 * wall_number = floor line index; ptr = offset into LEVELGRAPHICS for wall record (patch at +24 = door height).
 */
static AB3D_ATTR_UNUSED const uint8_t *skip_amiga_wall_list(const uint8_t *p)
{
    while (read_word(p) >= 0)
        p += 2 + 4 + 4;
    return p + 2;
}

/* Parse one door/lift wall list; return number of wall entries, advance *p to next door.
 * If dst is non-NULL, write packed entries:
 *   +0: fline (be16)
 *   +2: ptr_to_wall_rec (be32)   [Anims.s first long -> a1]
 *   +6: gfx_base_offset (be32)   [Anims.s second long -> a2]
 * Total: 10 bytes per entry. */
static int parse_amiga_door_wall_list(const uint8_t **p, uint8_t *dst, int max_entries)
{
    const uint8_t *q = *p;
    int n = 0;
    while (n < max_entries) {
        int16_t w = read_word(q);
        if (w < 0) break;
        if (dst) {
            write_word_be(dst + n * 10, w);
            write_long_be(dst + n * 10 + 2, read_long(q + 2));
            write_long_be(dst + n * 10 + 6, read_long(q + 6));
        }
        n++;
        q += 2 + 4 + 4;
    }
    if (read_word(q) < 0) q += 2;
    *p = q;
    return n;
}

/* Water list after lift table terminator (Anims.s DoWaterAnims): normalize zone ids to zone indices.
 * Structure:
 *   21 entries of [top(l), bot(l), curr(l), vel(w), (zone(w), gfx_off(l))* , -1(w)] */
static void normalize_amiga_water_list(uint8_t *water_src, const uint8_t *ld,
                                       const uint8_t *zone_adds, int num_zones)
{
    if (!water_src || !ld || !zone_adds || num_zones <= 0) return;
    uint8_t *p = water_src;
    for (int i = 0; i <= 20; i++) {
        p += 14; /* top, bot, curr, vel */
        int safety = 128;
        while (safety-- > 0) {
            int16_t zone = read_word(p);
            if (zone < 0) {
                p += 2;
                break;
            }
            int zidx = (zone >= 0 && zone < num_zones) ? (int)zone :
                       zone_file_to_index(ld, zone_adds, num_zones, zone);
            if (zidx >= 0 && zidx != zone)
                write_word_be(p, (int16_t)zidx);
            p += 2 + 4; /* zone + gfx offset */
        }
    }
}

/* -----------------------------------------------------------------------
 * level_parse - Parse loaded level data, resolving internal offsets
 *
 * This mirrors the "blag:" section in AB3DI.s where all offsets in the
 * level data are resolved to absolute pointers.
 * ----------------------------------------------------------------------- */
int level_parse(LevelState *level)
{
    if (!level->data || !level->graphics) {
        printf("[LEVEL] Cannot parse - data not loaded\n");
        return -1;
    }

    printf("[LEVEL] Parsing level data...\n");

    uint8_t *ld = level->data;      /* LEVELDATA */
    uint8_t *lg = level->graphics;  /* LEVELGRAPHICS */

    /* ---- Graphics data header (LEVELGRAPHICS) ----
     * Byte  0-3:  door_offset   (long)  offset from lg to door table; 0 = no doors
     * Byte  4-7:  lift_offset   (long)  offset to lift table
     * Byte  8-11: switch_offset (long)  offset to switch table
     * Byte 12-15: zone_graph_offset (long)  offset to zone graph adds
     * Byte 16+:   zone_adds     (num_zones * 4 bytes)  each long = offset into LEVELDATA to that zone's zone data
     *
     * Door table (at lg + door_offset). Entries 22 bytes each (same idea as lift: pos/top/bot), terminated by zone_id < 0.
     *   0-1:  zone_id (int16)   zone this door affects (roof height written here)
     *   2-3:  door_type (int16) 0=space/switch, 1=cond 0x900, 2=0x400, 3=0x200, 4=always open, 5=never
     *   4-7:  door_pos (int32)  current Y (*256)
     *   8-9:  door_vel (int16)  open/close speed
     *  10-13: door_top (int32) open position (*256), 14-17: door_bot (int32) closed position (*256)
     *  18-19: timer (int16)     close delay
     *  20-21: door_flags (uint16) for type 0: condition bit mask; 0 = any switch opens this door
     *
     * Switch table (at lg + switch_offset). Entries 14 bytes each, terminated by zone_id < 0.
     *   0-1:  zone_id (int16)   zone the switch is in
     *   2-3:  (reserved; byte 3 used as cooldown)
     *   4-5:  point_index (uint16) Amiga: index into Points for position; condition bit = 4 + (switch index in table), not from data
     *   6-9:  gfx_offset (long) offset into lg to switch wall's first word (for on/off patch)
     *  10-11: sw_x (int16)     switch position X (for facing check)
     *  12-13: sw_z (int16)     switch position Z
     *
     * Note: The procedural test level (stub) does not use this header and leaves door_data/switch_data
     * NULL. Use a real level file (e.g. levels/level_a/twolev.graph.bin) with non-zero door_offset
     * and switch_offset to test doors and switches.
     */
    /* Graphics header: all longs big-endian */
    int32_t door_offset = read_long(lg + 0);
    int32_t lift_offset = read_long(lg + 4);
    int32_t switch_offset = read_long(lg + 8);
    int32_t zone_graph_offset = read_long(lg + 12);
    printf("[LEVEL] Graphics header: door=%ld lift=%ld switch=%ld zone_graph=%ld\n",
           (long)door_offset, (long)lift_offset, (long)switch_offset, (long)zone_graph_offset);
    level->door_data_owned = false;
    level->switch_data_owned = false;
    level->lift_data_owned = false;

    int num_zones = (int)read_word(ld + 16);
    if (num_zones <= 0) num_zones = 256;

    /* Doors: Amiga format (match standalone). 999 terminator; 18-byte header + variable wall list. */
    level->door_wall_list = NULL;
    level->door_wall_list_offsets = NULL;
    level->door_wall_list_owned = false;
    level->num_doors = 0;
    if (door_offset <= 16) {
        level->door_data = NULL;
    } else {
        const uint8_t *door_src = lg + door_offset;
        int16_t first_w = read_word(door_src);
        if (first_w == 999) {
            level->door_data = NULL;
        } else {
            /* First pass: count doors and total wall entries */
            int nd = 0;
            int total_walls = 0;
            const uint8_t *d = door_src;
            while (read_word(d) != 999) {
                const uint8_t *wall_start = d + 18;
                int nw = parse_amiga_door_wall_list(&wall_start, NULL, 64);
                total_walls += nw;
                nd++;
                d = wall_start;
                if (nd > 256) break;
            }
            int16_t zone0 = read_word(door_src + 12);
            if (nd > 0 && nd <= 256 && zone0 >= 0 && zone0 < num_zones) {
                uint8_t *buf = (uint8_t *)malloc((size_t)(nd + 1) * 22u);
                uint8_t *wall_list = (total_walls > 0) ? (uint8_t *)malloc((size_t)total_walls * 10u) : NULL;
                uint32_t *wall_offsets = (nd > 0) ? (uint32_t *)malloc((size_t)(nd + 1) * sizeof(uint32_t)) : NULL;
                if (buf && (total_walls == 0 || (wall_list && wall_offsets))) {
                    const uint8_t *s = door_src;
                    int out_idx = 0;
                    uint32_t wall_index = 0;
                    for (int i = 0; i < nd; i++) {
                        int16_t bottom = read_word(s + 0);
                        int16_t top = read_word(s + 2);
                        int16_t curr = read_word(s + 4);
                        int16_t dir = read_word(s + 6);
                        int16_t zone = read_word(s + 12);
                        int16_t cond = read_word(s + 14);
                        int16_t door_mode = read_word(s + 16); /* high byte=open mode, low byte=close mode */
                        const uint8_t *wall_start = s + 18;
                        uint32_t door_start = wall_index;
                        int nw = parse_amiga_door_wall_list(&wall_start, wall_list ? wall_list + wall_index * 10 : NULL, 64);
                        wall_index += nw;
                        s = wall_start;
                        if (zone >= 0 && zone < num_zones) {
                            if (wall_offsets) wall_offsets[out_idx] = door_start;
                            uint8_t *t = buf + out_idx * 22;
                            write_word_be(t + 0, zone);
                            write_word_be(t + 2, door_mode);
                            /* Amiga: asr.w #2,d3 then muls #256,d3 → zone roof = curr*64, not curr*256 */
                            write_long_be(t + 4, (int32_t)curr * 64);
                            write_word_be(t + 8, dir);
                            write_long_be(t + 10, (int32_t)top * 64);
                            write_long_be(t + 14, (int32_t)bottom * 64);
                            write_word_be(t + 18, (int16_t)0);
                            write_word_be(t + 20, cond);
                            out_idx++;
                        }
                    }
                    if (wall_offsets) wall_offsets[out_idx] = wall_index;
                    write_word_be(buf + out_idx * 22, (int16_t)-1);
                    level->door_data = buf;
                    level->door_data_owned = true;
                    level->num_doors = out_idx;
                    level->door_wall_list = wall_list;
                    level->door_wall_list_offsets = wall_offsets;
                    level->door_wall_list_owned = true;
                } else {
                    free(wall_list);
                    free(wall_offsets);
                    if (buf) free(buf);
                    level->door_data = NULL;
                }
            } else {
                level->door_data = NULL;
            }
        }
    }

    /* Long 4: Offset to lifts - Amiga format (match standalone): 999 terminator, 18-byte header + variable wall list */
    level->lift_wall_list = NULL;
    level->lift_wall_list_offsets = NULL;
    level->num_lifts = 0;
    level->lift_wall_list_owned = false;
    level->water_list = NULL;
    if (lift_offset <= 16) {
        level->lift_data = NULL;
    } else {
        const uint8_t *lift_src = lg + lift_offset;
        const uint8_t *water_src = NULL;
        int16_t first_w = read_word(lift_src);
        if (first_w == 999) {
            level->lift_data = NULL;
            water_src = lift_src + 2;
        } else {
            int nl = 0;
            int total_lift_walls = 0;
            const uint8_t *d = lift_src;
            while (read_word(d) != 999) {
                const uint8_t *wall_start = d + 18;
                int nw = parse_amiga_door_wall_list(&wall_start, NULL, 64);
                total_lift_walls += nw;
                nl++;
                d = wall_start;
                if (nl > 256) break;
            }
            if (read_word(d) == 999)
                water_src = d + 2;
            /* Accept table if first lift has valid zone. Prefer raw index semantics (same as doors). */
            int16_t zone0 = read_word(lift_src + 12);
            int z0 = (zone0 >= 0 && zone0 < num_zones) ? (int)zone0 :
                     zone_file_to_index(ld, lg + 16, num_zones, zone0);
            if (nl > 0 && nl <= 256 && z0 >= 0) {
                uint8_t *buf = (uint8_t *)malloc((size_t)(nl + 1) * 20u);
                uint8_t *wall_list = (total_lift_walls > 0) ? (uint8_t *)malloc((size_t)total_lift_walls * 10u) : NULL;
                uint32_t *wall_offsets = (nl > 0) ? (uint32_t *)malloc((size_t)(nl + 1) * sizeof(uint32_t)) : NULL;
                if (buf && (total_lift_walls == 0 || (wall_list && wall_offsets))) {
                    const uint8_t *s = lift_src;
                    int out_idx = 0;
                    uint32_t wall_index = 0;
                    for (int i = 0; i < nl; i++) {
                        int16_t bottom = read_word(s + 0);
                        int16_t top = read_word(s + 2);
                        int16_t curr = read_word(s + 4);
                        int16_t dir = read_word(s + 6);
                        int16_t zone = read_word(s + 12);
                        int16_t conditions = read_word(s + 14);
                        int16_t lift_mode = read_word(s + 16); /* high byte=top behavior, low byte=bottom behavior */
                        const uint8_t *wall_start = s + 18;
                        uint32_t lift_start = wall_index;
                        int nw = parse_amiga_door_wall_list(&wall_start, wall_list ? wall_list + wall_index * 10 : NULL, 64);
                        wall_index += nw;
                        s = wall_start;
                        int zidx = (zone >= 0 && zone < num_zones) ? (int)zone :
                                   zone_file_to_index(ld, lg + 16, num_zones, zone);
                        if (zidx >= 0) {
                            if (wall_offsets) wall_offsets[out_idx] = lift_start;
                            uint8_t *t = buf + out_idx * 20;
                            write_word_be(t + 0, (int16_t)zidx);
                            write_word_be(t + 2, lift_mode);
                            write_long_be(t + 4, (int32_t)curr * 64);
                            write_word_be(t + 8, dir);
                            write_long_be(t + 10, (int32_t)top * 64);  /* lift_top = low position (×64, same as door) */
                            write_long_be(t + 14, (int32_t)bottom * 64);     /* lift_bot = high position (×64) */
                            write_word_be(t + 18, conditions);  /* Amiga conditions mask (same as door flags) */
                            out_idx++;
                        }
                    }
                    if (wall_offsets) wall_offsets[out_idx] = wall_index;
                    write_word_be(buf + out_idx * 20, (int16_t)-1);
                    level->lift_data = buf;
                    level->lift_data_owned = true;
                    level->num_lifts = out_idx;
                    level->lift_wall_list = wall_list;
                    level->lift_wall_list_offsets = wall_offsets;
                    level->lift_wall_list_owned = true;
                } else {
                    free(wall_list);
                    free(wall_offsets);
                    if (buf) free(buf);
                    level->lift_data = NULL;
                }
            } else {
                level->lift_data = NULL;
            }
        }
        if (water_src) {
            level->water_list = (uint8_t *)water_src;
            normalize_amiga_water_list(level->water_list, ld, lg + 16, num_zones);
        }
    }

    /* Long 8: Offset to switches.
     * Amiga SwitchRoutine iterates a fixed 8 entries (14 bytes each), not a sentinel table.
     * Individual slots can be disabled with zone_id < 0, including slot 0. */
    {
        const size_t switch_table_bytes = 8u * 14u;
        size_t gbc = level->graphics_byte_count;
        bool in_range = false;
        if (switch_offset > 16) {
            if (gbc == 0) {
                in_range = true; /* unknown size (e.g. synthetic/test); trust offset */
            } else if (switch_offset >= 0) {
                size_t off = (size_t)(uint32_t)switch_offset;
                if (off <= gbc && switch_table_bytes <= (gbc - off))
                    in_range = true;
            }
        }
        level->switch_data = in_range ? (uint8_t *)(lg + switch_offset) : NULL;
    }

    /* Long 12: Offset to zone graph adds */
    level->zone_graph_adds = lg + zone_graph_offset;

    /* Zone offset table starts at byte 16 of graphics data. Assume big-endian; convert if clearly LE. */
    level->zone_adds = lg + 16;
    level->zone_adds_owned = false;
    level->zone_brightness_le = false;

    /* ---- Level data header ---- */
    /* Byte 14: Number of points (word) */
    int16_t num_points = read_word(ld + 14);

    /* Byte 16: Number of zones (word) */
    level->num_zones = read_word(ld + 16);

    /* zone_adds starts at lg+16 and runs until zone_graph_adds.
     * On real Amiga data this can be num_zones+1 (extra connect slot). */
    {
        int32_t slots = level->num_zones;
        if (zone_graph_offset > 16) {
            int32_t table_bytes = zone_graph_offset - 16;
            if ((table_bytes % 4) == 0) {
                int32_t table_slots = table_bytes / 4;
                if (table_slots > 0)
                    slots = table_slots;
            }
        }
        if (slots < level->num_zones)
            slots = level->num_zones;
        if (slots < 0)
            slots = 0;
        if (slots > 32767)
            slots = 32767;
        level->num_zone_slots = (int16_t)slots;
    }

    /* Rows in zone_graph_adds (8 bytes each). Bounds draws/LOS without clamping to num_zones so
     * extra zone_adds slots (level 2) stay valid; graph index cap fixes level 3 OOB. */
    level->num_zone_graph_entries = 0;
    {
        size_t gbc = level->graphics_byte_count;
        if (level->graphics && gbc > (size_t)zone_graph_offset && zone_graph_offset >= 0) {
            int32_t gbc32 = (gbc > (size_t)INT_MAX) ? INT_MAX : (int32_t)gbc;
            int32_t zg_cands[] = { door_offset, lift_offset, switch_offset };
            int32_t zg_next = pick_next_section_offset(
                zone_graph_offset, gbc32, zg_cands,
                (int)(sizeof(zg_cands) / sizeof(zg_cands[0])));
            if (zg_next <= zone_graph_offset || zg_next > gbc32)
                zg_next = gbc32;
            int64_t span = (int64_t)zg_next - (int64_t)zone_graph_offset;
            if (span >= 8) {
                int32_t nent = (int32_t)(span / 8);
                if (nent > 32767) nent = 32767;
                level->num_zone_graph_entries = (int16_t)nent;
            }
        }
    }

    /* Zone offset table at lg+16 is big-endian. Log each zone's loaded data. */
    if (level->num_zones > 0) {
        printf("[LEVEL] Zones: %d zones, %d zone slots, %d zone_graph rows (offset table big-endian)\n",
               level->num_zones, level->num_zone_slots, (int)level->num_zone_graph_entries);
        for (int z = 0; z < level->num_zones; z++) {
            int32_t zoff = read_long(level->zone_adds + z * 4);
            size_t data_len = level->data_byte_count;
            if (zoff < 0 || (data_len != 0 && (size_t)zoff + 48u > data_len)) {
                printf("[LEVEL]   zone[%d] offset %ld - out of range (data_len=%zu)\n", z, (long)zoff, data_len);
                continue;
            }
            const uint8_t *zd = ld + zoff;
            int16_t zone_id = read_word(zd + 0);
            int32_t floor_y = read_long(zd + ZONE_OFF_FLOOR);
            int32_t roof_y = read_long(zd + ZONE_OFF_ROOF);
            int16_t bright_lo = read_word(zd + ZONE_OFF_BRIGHTNESS);
            int16_t bright_hi = read_word(zd + ZONE_OFF_UPPER_BRIGHT);
            unsigned anim_lo = zone_bright_anim_type(bright_lo);
            unsigned anim_hi = zone_bright_anim_type(bright_hi);
            printf("[LEVEL]   zone[%d] offset %ld id=%d floor=%ld roof=%ld bright=(%d,%d) anim=(%s,%s)\n",
                   z, (long)zoff, (int)zone_id, (long)floor_y, (long)roof_y, (int)bright_lo, (int)bright_hi,
                   zone_anim_flag_name(anim_lo), zone_anim_flag_name(anim_hi));
        }
        if (level->num_zone_slots > level->num_zones) {
            for (int z = level->num_zones; z < level->num_zone_slots; z++) {
                int32_t zoff = read_long(level->zone_adds + z * 4);
                size_t data_len = level->data_byte_count;
                if (zoff < 0 || (data_len != 0 && (size_t)zoff + 48u > data_len)) {
                    printf("[LEVEL]   zone_slot[%d] offset %ld - out of range (data_len=%zu)\n",
                           z, (long)zoff, data_len);
                    continue;
                }
                const uint8_t *zd = ld + zoff;
                int16_t zone_id = read_word(zd + 0);
                int32_t floor_y = read_long(zd + ZONE_OFF_FLOOR);
                int32_t roof_y = read_long(zd + ZONE_OFF_ROOF);
                printf("[LEVEL]   zone_slot[%d] offset %ld id=%d floor=%ld roof=%ld (extra slot)\n",
                       z, (long)zoff, (int)zone_id, (long)floor_y, (long)roof_y);
            }
        }
    }

    /* Byte 20: Number of object points (word) */
    level->num_object_points = read_word(ld + 20);

    /* Byte 22: Offset to points (long) */
    int32_t points_offset = read_long(ld + 22);
    level->points = ld + points_offset;
    /* Point brights follow points: points + 4*num_points + 4 */
    level->point_brights = level->points + 4 + num_points * 4;
    if (num_points > 0) {
        int point_anim_lower = 0;
        int point_anim_upper = 0;
        for (int p = 0; p < num_points; p++) {
            int16_t lo = read_word(level->point_brights + (size_t)p * 4u + 0u);
            int16_t hi = read_word(level->point_brights + (size_t)p * 4u + 2u);
            uint8_t lo_anim = (uint8_t)(((uint16_t)lo >> 8) & 0xFFu);
            uint8_t hi_anim = (uint8_t)(((uint16_t)hi >> 8) & 0xFFu);
            if ((int8_t)(lo & 0xFF) >= 0 && lo_anim != 0) point_anim_lower++;
            if ((int8_t)(hi & 0xFF) >= 0 && hi_anim != 0) point_anim_upper++;
        }
        printf("[LEVEL] Point brightness anim flags: lower=%d upper=%d (from point_brights table)\n",
               point_anim_lower, point_anim_upper);
    }

    /* Long 26: Offset to floor lines */
    int32_t floor_offset = read_long(ld + 26);
    level->floor_lines = ld + floor_offset;

    /* Long 30: Offset to object data */
    int32_t obj_offset = read_long(ld + 30);
    level->object_data = ld + obj_offset;

    /* Number of floor lines: derived at runtime (do not trust header).
     * Amiga uses 16-byte floor line entries (index * 16).
     *
     * Determine two bounds:
     * 1) Layout bound: floor block size until the nearest known following section.
     * 2) Reference bound: highest floor line index actually referenced by zone lists.
     *
     * Then clamp to the safer bound when both exist. */
    int32_t pshot_offset = read_long(ld + 34);
    int32_t nshot_offset = read_long(ld + 38);
    int32_t objpts_offset = read_long(ld + 42);
    int32_t plr1_offset = read_long(ld + 46);
    int32_t plr2_offset = read_long(ld + 50);

    int32_t next_candidates[] = {
        obj_offset,
        pshot_offset,
        nshot_offset,
        objpts_offset,
        plr1_offset,
        plr2_offset
    };
    int32_t next_after_floor = pick_next_section_offset(
        floor_offset, (int32_t)level->data_byte_count,
        next_candidates, (int)(sizeof(next_candidates) / sizeof(next_candidates[0])));
    int32_t layout_count = 0;
    if (next_after_floor > floor_offset) {
        layout_count = (next_after_floor - floor_offset) / 16;
    } else if (level->data_byte_count > 0 && (size_t)floor_offset < level->data_byte_count) {
        layout_count = (int32_t)((level->data_byte_count - (size_t)floor_offset) / 16u);
    }
    if (layout_count < 0) layout_count = 0;
    int32_t max_ref_index = -1;
    if (level->zone_adds && level->num_zone_slots > 0) {
        int zone_slots = (int)level->num_zone_slots;
        if (zone_slots < (int)level->num_zones)
            zone_slots = (int)level->num_zones;
        {
            size_t data_len = level->data_byte_count;
            for (int z = 0; z < zone_slots; z++) {
                int32_t zoff = read_long(level->zone_adds + (size_t)z * 4u);
                if (zoff < 0) continue;
                if (data_len != 0 && (size_t)zoff + 48u > data_len) continue;

                const uint8_t *zd = ld + zoff;
                int16_t list_off = read_word(zd + 32);  /* ToExitList */
                int64_t list_abs = (int64_t)zoff + (int64_t)list_off;
                if (list_abs < 0) continue;
                if (data_len != 0 && (size_t)list_abs + 2u > data_len) continue;

                const uint8_t *list = ld + (size_t)list_abs;
                for (int i = 0; i < 128; i++) {
                    if (data_len != 0 && (size_t)list_abs + (size_t)(i + 1) * 2u > data_len)
                        break;
                    {
                        int16_t entry = read_word(list + (size_t)i * 2u);
                        if (entry == -2) break;
                        if (entry >= 0 && entry > max_ref_index)
                            max_ref_index = (int32_t)entry;
                    }
                }
            }
        }
    }
    if (max_ref_index >= 0 && layout_count > 0) {
        int32_t ref_count = max_ref_index + 1;
        level->num_floor_lines = (ref_count <= layout_count) ? ref_count : layout_count;
    } else if (max_ref_index >= 0) {
        level->num_floor_lines = max_ref_index + 1;
    } else {
        level->num_floor_lines = layout_count;
    }

    /* Long 34: Offset to player shot data */
    level->player_shot_data = ld + pshot_offset;

    /* Long 38: Offset to nasty shot data */
    level->nasty_shot_data = ld + nshot_offset;
    /* Other nasty data follows after shot records. */
    level->other_nasty_data = level->nasty_shot_data + (size_t)PLAYER_SHOT_SLOT_COUNT * OBJECT_SIZE;

    /* Long 42: Offset to object points */
    level->object_points = ld + objpts_offset;

    if (level_expand_nasty_shot_pool(level) == 0) {
        printf("[LEVEL] Expanded nasty shot pool: %d slots, object_points=%d\n",
               NASTY_SHOT_SLOT_COUNT, (int)level->num_object_points);
    } else {
        printf("[LEVEL] Nasty shot pool expansion unavailable; using %d slots\n",
               PLAYER_SHOT_SLOT_COUNT);
    }

    /* Long 46: Offset to player 1 object */
    level->plr1_obj = ld + plr1_offset;

    /* Long 50: Offset to player 2 object */
    level->plr2_obj = ld + plr2_offset;

    printf("[LEVEL] Parsed: %d zones, %d points, %d obj_points, %d floor_lines\n",
           level->num_zones, num_points, level->num_object_points, level->num_floor_lines);

    /* Log floor lines data (16 bytes each: x, z, xlen, zlen, connect at 0,2,4,6,8) */
    if (level->floor_lines && level->num_floor_lines > 0) {
        printf("[LEVEL] Floor lines: offset=%ld count=%d (x, z, xlen, zlen, connect)\n",
               (long)floor_offset, (int)level->num_floor_lines);
        int log_max = level->num_floor_lines > 24 ? 24 : (int)level->num_floor_lines;
        for (int fli = 0; fli < log_max; fli++) {
            const uint8_t *fl = level->floor_lines + (unsigned)fli * 16u;
            int16_t fx = read_word(fl + 0);
            int16_t fz = read_word(fl + 2);
            int16_t fxlen = read_word(fl + 4);
            int16_t fzlen = read_word(fl + 6);
            int16_t fconn = read_word(fl + 8);
            printf("[LEVEL]   fl[%d] x=%d z=%d xlen=%d zlen=%d connect=%d\n",
                   fli, (int)fx, (int)fz, (int)fxlen, (int)fzlen, (int)fconn);
        }
        if ((int)level->num_floor_lines > log_max)
            printf("[LEVEL]   ... and %d more\n", (int)level->num_floor_lines - log_max);
    }

    /* Debug: dump doors and switches (door_flags; switch condition bit = 4 + switch_index, not from data) */
    if (level->door_data) {
        const uint8_t *door = level->door_data;
        int di = 0;
        while (1) {
            int16_t zone_id = read_word(door);
            if (zone_id < 0) break;
            int16_t door_type = read_word(door + 2);
            int32_t door_pos = read_long(door + 4);
            int32_t door_top = read_long(door + 10);
            int32_t door_bot = read_long(door + 14);
            uint16_t door_flags = (uint16_t)read_word(door + 20);
            printf("[LEVEL] door[%d] zone=%d type=%d pos=%ld top=%ld bot=%ld flags=0x%04X (%u)\n",
                   di, (int)zone_id, (int)door_type, (long)door_pos, (long)door_top, (long)door_bot, door_flags, door_flags);
            door += 22;
            di++;
        }
    }
    if (level->switch_data) {
        const uint8_t *sw = level->switch_data;
        int si = 0;
        while (1) {
            int16_t zone_id = read_word(sw);
            if (zone_id < 0) break;
            uint16_t point_index = (uint16_t)read_word(sw + 4);
            unsigned int bit_num = 4 + (si % 8);
            uint16_t cond_bit = (uint16_t)(1u << bit_num);
            int16_t sw_x = read_word(sw + 10);
            int16_t sw_z = read_word(sw + 12);
            int door_zone = -1;
            if (level->door_data) {
                const uint8_t *door = level->door_data;
                while (1) {
                    int16_t dz = read_word(door);
                    if (dz < 0) break;
                    int16_t door_type = read_word(door + 2);
                    uint16_t door_flags = (uint16_t)read_word(door + 20);
                    if (door_type == 0 && (door_flags & cond_bit) != 0) {
                        door_zone = (int)dz;
                        break;
                    }
                    door += 22;
                }
            }
            if (door_zone >= 0)
                printf("[LEVEL] switch[%d] zone=%d point_index=%u cond_bit=0x%04X (bit %u) pos=(%d,%d) opens_door_zone=%d\n",
                       si, (int)zone_id, point_index, cond_bit, bit_num, (int)sw_x, (int)sw_z, door_zone);
            else
                printf("[LEVEL] switch[%d] zone=%d point_index=%u cond_bit=0x%04X (bit %u) pos=(%d,%d)\n",
                       si, (int)zone_id, point_index, cond_bit, bit_num, (int)sw_x, (int)sw_z);
            sw += 14;
            si++;
        }
    }
    if (level->lift_data) {
        const uint8_t *lift = level->lift_data;
        int li = 0;
        while (1) {
            int16_t zone_id = read_word(lift);
            if (zone_id < 0) break;
            int16_t lift_type = read_word(lift + 2);
            int32_t lift_pos = read_long(lift + 4);
            int16_t lift_vel = read_word(lift + 8);
            int32_t lift_top = read_long(lift + 10);
            int32_t lift_bot = read_long(lift + 14);
            printf("[LEVEL] lift[%d] zone=%d type=%d pos=%ld vel=%d top=%ld bot=%ld\n",
                   li, (int)zone_id, (int)lift_type, (long)lift_pos, (int)lift_vel, (long)lift_top, (long)lift_bot);
            lift += 20;
            li++;
        }
    }
    /* Debug: dump objects in level (64 bytes each; zone < 0 = inactive) */
    if (level->object_data) {
        const uint8_t *obj = level->object_data;
        int oi = 0;
        const int max_objects = 256;
        while (oi < max_objects) {
            int16_t cid = read_word(obj + 0);
            int16_t zone = read_word(obj + 12);
            int8_t number = (int8_t)obj[16];  /* object type */
            int8_t can_see = (int8_t)obj[17];
            {
                const char *otype =
                    number == OBJ_NBR_KEY ? "key" :
                    number == OBJ_NBR_MEDIKIT ? "medikit" :
                    number == OBJ_NBR_AMMO ? "ammo" :
                    number == OBJ_NBR_BIG_GUN ? "big_gun" :
                    number == OBJ_NBR_PLR1 ? "plr1" : number == OBJ_NBR_PLR2 ? "plr2" : "other";
                /* One [LEVEL] line so ab3d_log_printf suppresses it (split printf leaked key_byte17=). */
                if (number == OBJ_NBR_KEY)
                    printf("[LEVEL] obj[%d] zone=%d cid=%d type=%d (%s) key_byte17=%d\n",
                           oi, (int)zone, (int)cid, (int)number, otype, (int)(can_see & 0xFF));
                else
                    printf("[LEVEL] obj[%d] zone=%d cid=%d type=%d (%s)\n",
                           oi, (int)zone, (int)cid, (int)number, otype);
            }
            obj += OBJECT_SIZE;
            oi++;
        }
    }

    log_broken_floor_line_connects(level);

    level_build_zone_index_by_data_offset(level);

    return 0;
}

/* -----------------------------------------------------------------------
 * level_assign_clips - Assign clip offsets to zone graph lists
 *
 * Translated from AB3DI.s assignclips loop (~line 812-843).
 *
 * Each zone has a list of graphical elements (walls, floors, etc).
 * The clip data provides pre-computed clipping polygons for these.
 * This function links the clip data into the zone graph lists.
 * ----------------------------------------------------------------------- */
void level_assign_clips(LevelState *level, int16_t num_zones)
{
    if (!level->clips || !level->zone_adds || !level->data) {
        return;
    }

    uint8_t *zone_offsets = level->zone_adds;  /* lg + 16 */
    uint8_t *ld = level->data;
    uint8_t *clips = level->clips;

    int32_t clip_byte_offset = 0;

    for (int16_t z = 0; z <= num_zones; z++) {
        /* Get zone offset from graphics data -> points into level data */
        int32_t zone_add = read_long(zone_offsets + z * 4);
        uint8_t *zone_ptr = ld + zone_add;

        /* Go to the list of graph elements (ToListOfGraph offset) */
        uint8_t *graph_list = zone_ptr + ZONE_OFF_LIST_OF_GRAPH;

        /* Walk the graph list (each entry is 8 bytes: type, clip_offset, ...) */
        while (1) {
            int16_t entry_type = read_word(graph_list);
            if (entry_type < 0) break;  /* End of zone list */

            int16_t clip_ref = read_word(graph_list + 2);
            if (clip_ref >= 0) {
                /* Assign current clip offset */
                int16_t half_offset = (int16_t)(clip_byte_offset >> 1);
                /* Write back (big-endian) */
                graph_list[2] = (uint8_t)(half_offset >> 8);
                graph_list[3] = (uint8_t)(half_offset & 0xFF);

                /* Find next clip boundary (-2 sentinel) */
                while (read_word(clips + clip_byte_offset) != -2) {
                    clip_byte_offset += 2;
                }
                clip_byte_offset += 2;  /* Skip the -2 sentinel */
            }

            graph_list += 8;  /* Next entry */
        }
    }

    /* The connect table starts after all clips */
    level->connect_table = clips + clip_byte_offset;

    printf("[LEVEL] Clips assigned, connect table at offset %d\n",
           (int)clip_byte_offset);
}

int level_zone_slot_count(const LevelState *level)
{
    if (!level) return 0;
    int slots = (int)level->num_zone_slots;
    if (slots <= 0)
        slots = (int)level->num_zones;
    if (slots < (int)level->num_zones)
        slots = (int)level->num_zones;
    if (slots < 0)
        slots = 0;
    return slots;
}

int level_zone_has_upper_layer(const LevelState *level, int16_t zone_id)
{
    if (!level || !level->zone_adds || !level->data)
        return 0;

    {
        int zone_slots = level_zone_slot_count(level);
        int zi = -1;
        if (zone_id >= 0 && zone_id < zone_slots)
            zi = (int)zone_id;
        else
            zi = level_connect_to_zone_index(level, zone_id);

        if (zi < 0 || zi >= zone_slots)
            return 0;

        {
            int32_t zoff = read_long(level->zone_adds + (size_t)zi * 4u);
            if (zoff < 0)
                return 0;

            if (level->data_byte_count > 0 &&
                (size_t)zoff + 18u > level->data_byte_count)
                return 0;

            {
                const uint8_t *zd = level->data + zoff;
                int32_t upper_floor = read_long(zd + ZONE_OFF_UPPER_FLOOR);
                int32_t upper_roof = read_long(zd + ZONE_OFF_UPPER_ROOF);
                if (upper_floor <= upper_roof)
                    return 0;
            }
        }

        if (level->zone_graph_adds && level->graphics) {
            if (level->num_zone_graph_entries > 0 && zi >= level->num_zone_graph_entries)
                return 0;

            {
                int32_t upper_gfx = read_long(level->zone_graph_adds + (size_t)zi * 8u + 4u);
                if (upper_gfx <= 0)
                    return 0;
                if (level->graphics_byte_count > 0 &&
                    (size_t)upper_gfx + 2u > level->graphics_byte_count)
                    return 0;
            }
        }
    }

    return 1;
}

int level_connect_to_zone_index(const LevelState *level, int16_t connect)
{
    typedef struct {
        const LevelState *level;
        const uint8_t *zone_adds;
        const uint8_t *data;
        int16_t connect;
        int16_t zone_slots;
        int16_t result;
    } LevelConnectCacheEntry;
    enum { k_level_connect_cache_size = 1024 };
    static AB3D_THREAD_LOCAL LevelConnectCacheEntry cache[k_level_connect_cache_size];
    int result = -1;

    if (!level->zone_adds || !level->data || connect < 0)
        return -1;
    int zone_slots = level_zone_slot_count(level);
    if (zone_slots <= 0)
        return -1;

    {
        uintptr_t mix = ((uintptr_t)level >> 4) ^
                        ((uintptr_t)level->zone_adds >> 3) ^
                        ((uintptr_t)(uint16_t)connect * 2654435761u);
        unsigned int slot = (unsigned int)(mix & (uintptr_t)(k_level_connect_cache_size - 1));
        LevelConnectCacheEntry *entry = &cache[slot];
        if (entry->level == level &&
            entry->zone_adds == level->zone_adds &&
            entry->data == level->data &&
            entry->connect == connect &&
            entry->zone_slots == zone_slots) {
            return (int)entry->result;
        }

        size_t data_len = level->data_byte_count;
        for (int z = 0; z < zone_slots; z++) {
            int32_t zoff = read_long(level->zone_adds + (size_t)z * 4u);
            if (zoff < 0) continue;
            if (data_len != 0 && (size_t)zoff + 2u > data_len) continue;
            const uint8_t *zd = level->data + zoff;
            if (read_word(zd + 0) == (int16_t)connect) {
                result = z;
                break;
            }
        }
        if (result < 0) {
            /* Amiga ObjectMove uses floor-line connect as a direct index into zoneAdds.
             * Some levels have an extra slot (zone_slots > num_zones), so allow the
             * full slot range here, not just [0, num_zones). */
            if (connect < zone_slots) {
                int32_t zoff = read_long(level->zone_adds + (size_t)connect * 4u);
                if (zoff >= 0 && (data_len == 0 || (size_t)zoff + 2u <= data_len))
                    result = (int)connect;
            }
        }

        entry->level = level;
        entry->zone_adds = level->zone_adds;
        entry->data = level->data;
        entry->connect = connect;
        entry->zone_slots = (int16_t)zone_slots;
        entry->result = (int16_t)result;
    }

    return result;
}

/* Dense map: byte offset into level->data -> first matching zone_adds slot (int16_t -1 = unused). */
static void level_build_zone_index_by_data_offset(LevelState *level)
{
    free(level->zone_index_by_data_offset);
    level->zone_index_by_data_offset = NULL;
    level->zone_index_by_data_offset_len = 0;

    if (!level || !level->zone_adds || !level->data)
        return;

    size_t len = level->data_byte_count;
    /* Skip if size unknown or map would be unreasonably large */
    if (len == 0 || len > (size_t)16u * 1024u * 1024u)
        return;

    int zone_slots = level_zone_slot_count(level);
    if (zone_slots <= 0)
        return;

    int16_t *map = (int16_t *)malloc(len * sizeof(int16_t));
    if (!map)
        return;

    memset(map, 0xFF, len * sizeof(int16_t)); /* -1 */

    for (int z = 0; z < zone_slots; z++) {
        int32_t zoff = read_long(level->zone_adds + (size_t)z * 4u);
        if (zoff < 0 || (size_t)zoff >= len)
            continue;
        if (map[(size_t)zoff] < 0)
            map[(size_t)zoff] = (int16_t)z;
    }

    level->zone_index_by_data_offset = map;
    level->zone_index_by_data_offset_len = len;
}

int level_zone_index_from_room_offset(const LevelState *level, int32_t room_offset)
{
    if (!level || !level->zone_adds || room_offset < 0)
        return -1;

    if (level->zone_index_by_data_offset &&
        (size_t)room_offset < level->zone_index_by_data_offset_len) {
        int16_t z = level->zone_index_by_data_offset[(size_t)room_offset];
        if (z >= 0)
            return (int)z;
        return -1;
    }

    {
        int zone_slots = level_zone_slot_count(level);
        for (int z = 0; z < zone_slots; z++) {
            if (read_long(level->zone_adds + (size_t)z * 4u) == room_offset)
                return z;
        }
    }
    return -1;
}

int level_zone_index_from_room_ptr(const LevelState *level, const uint8_t *room_ptr)
{
    if (!level || !level->data || !room_ptr)
        return -1;
    return level_zone_index_from_room_offset(level, (int32_t)(room_ptr - level->data));
}

static int point_on_segment_i32(int32_t px, int32_t pz,
                                int32_t x1, int32_t z1,
                                int32_t x2, int32_t z2)
{
    int64_t cross = (int64_t)(px - x1) * (int64_t)(z2 - z1) -
                    (int64_t)(pz - z1) * (int64_t)(x2 - x1);
    if (cross != 0) return 0;
    if (px < (x1 < x2 ? x1 : x2) || px > (x1 > x2 ? x1 : x2)) return 0;
    if (pz < (z1 < z2 ? z1 : z2) || pz > (z1 > z2 ? z1 : z2)) return 0;
    return 1;
}

static int zone_contains_point(const LevelState *level, int zone_index, int32_t x, int32_t z)
{
    if (!level || !level->data || !level->zone_adds || !level->floor_lines)
        return 0;
    if (zone_index < 0 || zone_index >= level_zone_slot_count(level))
        return 0;

    {
        int32_t zoff = read_long(level->zone_adds + (size_t)zone_index * 4u);
        if (zoff < 0)
            return 0;

        size_t data_len = level->data_byte_count;
        if (data_len != 0 && (size_t)zoff + 34u > data_len)
            return 0;

        const uint8_t *zd = level->data + zoff;
        int16_t list_off = read_word(zd + 32);
        int64_t list_abs = (int64_t)zoff + (int64_t)list_off;
        if (list_abs < 0)
            return 0;
        if (data_len != 0 && (size_t)list_abs + 2u > data_len)
            return 0;

        const uint8_t *list = level->data + (size_t)list_abs;
        int inside = 0;
        int edges = 0;

        for (int i = 0; i < 256; i++) {
            if (data_len != 0 && (size_t)(list - level->data) + (size_t)i * 2u + 2u > data_len)
                break;

            int16_t entry = read_word(list + i * 2);
            if (entry == -2)
                break;
            if (entry < 0)
                continue; /* -1 separator between exits and wall list */
            if (entry >= level->num_floor_lines)
                continue;

            const uint8_t *fl = level->floor_lines + (size_t)entry * 16u;
            int32_t x1 = read_word(fl + 0);
            int32_t z1 = read_word(fl + 2);
            int32_t x2 = x1 + read_word(fl + 4);
            int32_t z2 = z1 + read_word(fl + 6);
            edges++;

            if (point_on_segment_i32(x, z, x1, z1, x2, z2))
                return 1;

            if ((z1 > z) != (z2 > z)) {
                int64_t dz = (int64_t)z2 - (int64_t)z1;
                int64_t lhs = (int64_t)(x - x1) * dz;
                int64_t rhs = (int64_t)(x2 - x1) * (int64_t)(z - z1);
                if ((dz > 0 && lhs < rhs) || (dz < 0 && lhs > rhs))
                    inside ^= 1;
            }
        }

        if (edges == 0)
            return 0;
        return inside;
    }
}

int level_find_zone_for_point(const LevelState *level, int32_t x, int32_t z, int16_t hint_zone)
{
    if (!level || !level->data || !level->zone_adds || !level->floor_lines)
        return -1;

    {
        int zone_slots = level_zone_slot_count(level);
        if (zone_slots <= 0)
            return -1;

        int hint = -1;
        if (hint_zone >= 0) {
            hint = level_connect_to_zone_index(level, hint_zone);
            if (hint < 0 && hint_zone < zone_slots)
                hint = hint_zone;
            if (hint >= 0 && zone_contains_point(level, hint, x, z))
                return hint;
        }

        for (int zi = 0; zi < zone_slots; zi++) {
            if (zi == hint) continue;
            if (zone_contains_point(level, zi, x, z))
                return zi;
        }
    }
    return -1;
}

/* Floor line: 16 bytes; word at offset 8 = connect. */
/*
 * Check invariant: for each zone, for each exit in its exit list, the adjacent
 * zone (the zone fline connect points to) should also list that fline (pass back).
 * Log when it does not hold. Ignore one-sided boundaries (actual walls: fline in
 * only one zone's exit list).
 */
static void log_broken_floor_line_connects(LevelState *level)
{
    if (!level->floor_lines || !level->zone_adds || !level->data ||
        level->num_zones <= 0 || level->num_floor_lines <= 0)
        return;

    const int n = (int)level->num_floor_lines;
    int num_zones = level->num_zones;
    int *zone_a = (int *)malloc((size_t)n * sizeof(int));
    int *zone_b = (int *)malloc((size_t)n * sizeof(int));
    unsigned char *nz = (unsigned char *)calloc((size_t)n, 1);
    if (!zone_a || !zone_b || !nz) {
        free(zone_a);
        free(zone_b);
        free(nz);
        return;
    }
    for (int i = 0; i < n; i++) zone_a[i] = zone_b[i] = -1;

    /* Reverse map: which zones reference each floor line (from ToExitList). */
    for (int z = 0; z < num_zones; z++) {
        int32_t zoff = read_long(level->zone_adds + (size_t)z * 4u);
        if (zoff < 0) continue;
        const uint8_t *zd = level->data + zoff;
        int16_t list_off = read_word(zd + 32);
        const uint8_t *list = zd + list_off;
        for (int i = 0; i < 128; i++) {
            int16_t entry = read_word(list + i * 2);
            if (entry < 0) break;  /* -1 ends exit portion, -2 ends list (match movement.c) */
            if (entry < n) {
                nz[entry]++;
                if (nz[entry] == 1) zone_a[entry] = z;
                else if (nz[entry] == 2) zone_b[entry] = z;
            }
        }
    }

    const uint8_t *flines = level->floor_lines;
    int logged = 0;

    /* Pass 1: any fline with connect >= 0 that does not resolve is a broken connect. Log zone info and each exit fline info. */
    for (int f = 0; f < n; f++) {
        int16_t connect = read_word(flines + (size_t)f * 16u + 8);

        if (connect < 0) continue;
        if (level_connect_to_zone_index(level, connect) >= 0) continue;

        int correct_zone = 0; // TODO: Work this out

        /* Log only; do not patch the connect word. */
        /* Fline f has broken connect; log each zone that lists f: zone info then each exit fline info. */
        int z1 = zone_a[f];
        int z2 = (nz[f] >= 2) ? zone_b[f] : -1;
        for (int zi = 0; zi < 2; zi++) {
            int z = (zi == 0) ? z1 : z2;
            if (z < 0) continue;
            int32_t zoff = read_long(level->zone_adds + (size_t)z * 4u);
            if (zoff < 0) continue;
            const uint8_t *zd = level->data + zoff;
            printf("[LEVEL]   zone %d (fline %d broken connect %d, should be zone %d): id=%d floor=%ld roof=%ld upper_floor=%ld upper_roof=%ld water=%ld bright=%d upper_bright=%d tel_zone=%d tel_x=%d tel_z=%d\n",
                   z, f, (int)connect, correct_zone,
                   (int)read_word(zd + 0),
                   (long)read_long(zd + ZONE_OFF_FLOOR), (long)read_long(zd + ZONE_OFF_ROOF),
                   (long)read_long(zd + ZONE_OFF_UPPER_FLOOR), (long)read_long(zd + ZONE_OFF_UPPER_ROOF),
                   (long)read_long(zd + ZONE_OFF_WATER),
                   (int)read_word(zd + ZONE_OFF_BRIGHTNESS), (int)read_word(zd + ZONE_OFF_UPPER_BRIGHT),
                   (int)read_word(zd + ZONE_OFF_TEL_ZONE), (int)read_word(zd + ZONE_OFF_TEL_X), (int)read_word(zd + ZONE_OFF_TEL_Z));
            int16_t list_off = read_word(zd + 32);
            const uint8_t *list = zd + list_off;
            for (int i = 0; i < 128; i++) {
                int16_t entry = read_word(list + i * 2);
                if (entry < 0) break;
                if (entry >= n) continue;
                const uint8_t *fl = flines + (size_t)entry * 16u;
                int fl_connect = (int)read_word(fl + 8);
                if (entry == f)
                    printf("[LEVEL]     fline %d: x=%d z=%d xlen=%d zlen=%d connect=%d (should be zone %d) length=%d normal=%d away=%d\n",
                           entry,
                           (int)read_word(fl + 0), (int)read_word(fl + 2), (int)read_word(fl + 4), (int)read_word(fl + 6),
                           fl_connect, correct_zone,
                           (int)read_word(fl + 10), (int)read_word(fl + 12), (int)read_word(fl + 14));
                else
                    printf("[LEVEL]     fline %d: x=%d z=%d xlen=%d zlen=%d connect=%d length=%d normal=%d away=%d\n",
                           entry,
                           (int)read_word(fl + 0), (int)read_word(fl + 2), (int)read_word(fl + 4), (int)read_word(fl + 6),
                           fl_connect,
                           (int)read_word(fl + 10), (int)read_word(fl + 12), (int)read_word(fl + 14));
            }
            logged++;
        }
    }

    if (logged > 0)
        printf("[LEVEL] %d broken connect(s) or exit(s) where adjacent zone does not pass back\n", logged);

    free(zone_a);
    free(zone_b);
    free(nz);
}

int level_get_zone_info(const LevelState *level, int16_t zone_id, ZoneInfo *out)
{
    int zone_slots = level_zone_slot_count(level);
    if (!out || !level->zone_adds || !level->data || zone_slots <= 0)
        return -1;
    if (zone_id < 0 || zone_id >= zone_slots)
        return -1;
    int32_t zoff = read_long(level->zone_adds + (size_t)zone_id * 4u);
    size_t data_len = level->data_byte_count;
    if (zoff < 0 || (data_len != 0 && (size_t)zoff + 48u > data_len))
        return -1;
    const uint8_t *zd = level->data + zoff;
    out->zone_id = read_word(zd + 0);
    out->floor_y = read_long(zd + ZONE_OFF_FLOOR);
    out->roof_y = read_long(zd + ZONE_OFF_ROOF);
    out->upper_floor_y = read_long(zd + ZONE_OFF_UPPER_FLOOR);
    out->upper_roof_y = read_long(zd + ZONE_OFF_UPPER_ROOF);
    out->water_y = read_long(zd + ZONE_OFF_WATER);
    out->brightness = read_word(zd + ZONE_OFF_BRIGHTNESS);
    out->upper_brightness = read_word(zd + ZONE_OFF_UPPER_BRIGHT);
    out->tel_zone = read_word(zd + ZONE_OFF_TEL_ZONE);
    out->tel_x = read_word(zd + ZONE_OFF_TEL_X);
    out->tel_z = read_word(zd + ZONE_OFF_TEL_Z);
    return 0;
}

uint8_t *level_get_zone_data_ptr(LevelState *level, int16_t zone_id)
{
    int zone_slots = level_zone_slot_count(level);
    if (!level->zone_adds || !level->data || zone_slots <= 0)
        return NULL;
    if (zone_id < 0 || zone_id >= zone_slots)
        return NULL;
    int32_t zoff = read_long(level->zone_adds + (size_t)zone_id * 4u);
    size_t data_len = level->data_byte_count;
    if (zoff < 0 || (data_len != 0 && (size_t)zoff + 48u > data_len))
        return NULL;

    return level->data + zoff;
}

/* Anim type 1=pulse, 2=flicker, 3=fire. Amiga uses high byte only. */
static inline unsigned zone_anim_type_from_word(int16_t word)
{
    unsigned hi = (unsigned)((uint16_t)word >> 8) & 0xFFu;
    if (hi >= 1u && hi <= 3u) return hi;
    return 0;
}

int16_t level_get_zone_brightness(const LevelState *level, int16_t zone_id, int use_upper)
{
    int zone_slots = level_zone_slot_count(level);
    if (!level->zone_adds || !level->data || zone_slots <= 0)
        return 0;
    if (zone_id < 0 || zone_id >= zone_slots)
        return 0;

    int32_t zoff = read_long(level->zone_adds + (size_t)zone_id * 4u);
    size_t data_len = level->data_byte_count;
    if (zoff < 0 || (data_len != 0 && (size_t)zoff + 48u > data_len))
        return 0;
    const uint8_t *zd = level->data + zoff;
    int off = use_upper ? ZONE_OFF_UPPER_BRIGHT : ZONE_OFF_BRIGHTNESS;
    int16_t word;
    if (level->zone_brightness_le)
        word = (int16_t)((zd[off + 1] << 8) | zd[off]);
    else
        word = (int16_t)((zd[off] << 8) | zd[off + 1]);
    unsigned anim = zone_anim_type_from_word(word);
    if (anim == 0)
        return word;
    /* Amiga: animated zone brightness = table value only (no base, no clamp). add.w ZoneBright,d6 uses full range e.g. -10..10 */
    if (anim >= 1 && anim <= 3)
        return (int16_t)level->bright_anim_values[anim - 1];
    return word;
}

int16_t level_get_point_brightness(const LevelState *level, int16_t point_id, int use_upper)
{
    if (!level || !level->data || !level->point_brights)
        return 0;
    if (point_id < 0)
        return 0;

    {
        int num_points = (int)read_word(level->data + 14);
        if (num_points <= 0 || point_id >= num_points)
            return 0;
    }

    {
        /* Match AB3DI.s "currentPointBrights" decode exactly:
         *  - base brightness is signed low byte (ext.w d2)
         *  - high byte encodes anim select/blend factor
         *  - final result is ext.w d2 again (signed low byte) */
        const uint8_t *pb = level->point_brights + (size_t)point_id * 4u + (use_upper ? 2u : 0u);
        int16_t d2 = read_word(pb);

        if ((int8_t)(d2 & 0xFF) >= 0) {
            uint8_t d3 = (uint8_t)(((uint16_t)d2 >> 8) & 0xFFu);
            if (d3 != 0) {
                uint8_t anim = (uint8_t)(d3 & 0x0Fu);             /* and.w #$f,d3 */
                uint8_t blend = (uint8_t)((d3 >> 4) + 1u);        /* lsr.w #4,d4 ; addq #1,d4 */
                if (anim >= 1u && anim <= 3u) {
                    int16_t base = (int16_t)(int8_t)(d2 & 0xFF);  /* ext.w d2 */
                    int16_t target = (int16_t)level->bright_anim_values[anim - 1u];
                    int32_t delta = (int32_t)(target - base) * (int32_t)blend;
                    /* ASR #4 semantics (round toward -inf), not C /16 (toward zero). */
                    int32_t scaled = (delta >= 0) ? (delta / 16) : -(((-delta) + 15) / 16);
                    d2 = (int16_t)(base + (int16_t)scaled);
                }
            }
        }

        return (int16_t)(int8_t)(d2 & 0xFF); /* ext.w d2 */
    }
}

static inline int16_t swap16(int16_t v)
{
    return (int16_t)(((uint16_t)v >> 8) | ((uint16_t)v << 8));
}
static inline int32_t swap32(int32_t v)
{
    return (int32_t)(((uint32_t)v >> 24) | (((uint32_t)v >> 8) & 0xFF00u) |
                     (((uint32_t)v << 8) & 0xFF0000u) | ((uint32_t)v << 24));
}

ZoneInfo zone_info_swap_endianness(const ZoneInfo *z)
{
    ZoneInfo out = {0};
    if (!z) return out;
    out.zone_id = swap16(z->zone_id);
    out.floor_y = swap32(z->floor_y);
    out.roof_y = swap32(z->roof_y);
    out.upper_floor_y = swap32(z->upper_floor_y);
    out.upper_roof_y = swap32(z->upper_roof_y);
    out.water_y = swap32(z->water_y);
    out.brightness = swap16(z->brightness);
    out.upper_brightness = swap16(z->upper_brightness);
    out.tel_zone = swap16(z->tel_zone);
    out.tel_x = swap16(z->tel_x);
    out.tel_z = swap16(z->tel_z);
    return out;
}

/* Payload bytes after type word in zone graphics stream (matches renderer zone_gfx_entry_data_skip). */
static size_t zone_gfx_payload_skip(int16_t entry_type, const uint8_t *data)
{
    switch (entry_type) {
    case 0:
    case 13: return 28;
    case 3:
    case 12: return 0;
    case 4: return 2;
    case 5: return 28;
    case 6: return 4;
    case 1:
    case 2:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    {
        int16_t nsm1 = read_word(data + 2);
        int sides = (int)nsm1 + 1;
        if (sides < 0) sides = 0;
        if (sides > 100) sides = 100;
        return (size_t)(4 + 2 * sides + 10);
    }
    default: return 0;
    }
}

static int level_find_zone_slot_by_id_word(const LevelState *level, int16_t zone_id_word)
{
    if (!level || !level->zone_adds || !level->data) return -1;
    int slots = level_zone_slot_count(level);
    size_t data_len = level->data_byte_count;
    for (int i = 0; i < slots; i++) {
        int32_t zoff = read_long(level->zone_adds + (size_t)i * 4u);
        if (zoff < 0) continue;
        if (data_len > 0 && (size_t)zoff + 2u > data_len) continue;
        if (read_word(level->data + (size_t)zoff) == zone_id_word) return i;
    }
    return -1;
}

static int level_edge_match(int16_t a1, int16_t a2, int16_t b1, int16_t b2)
{
    return ((a1 == b1 && a2 == b2) || (a1 == b2 && a2 == b1)) ? 1 : 0;
}

static int level_l4_adjacent_door_wall_match(int16_t p1, int16_t p2, int16_t tex_id)
{
    /* Level 4 door seam: same physical doorway appears in adjacent zone streams. */
    if (tex_id == 1 || tex_id == 6 || tex_id == 8) {
        int16_t d = (int16_t)(p1 - p2);
        if (d < 0) d = (int16_t)(-d);
        if (p1 >= 266 && p1 <= 269 &&
            p2 >= 266 && p2 <= 269 &&
            (d == 1 || d == 3)) {
            return 1;
        }
    }

    if (tex_id == 1) {
        if (level_edge_match(p1, p2, 25, 23)) return 1;
        if (level_edge_match(p1, p2, 24, 26)) return 1;
        if (level_edge_match(p1, p2, 33, 32)) return 1;
        if (level_edge_match(p1, p2, 269, 268)) return 1; /* zone 124 doorway */
        if (level_edge_match(p1, p2, 267, 266)) return 1; /* zone 124 doorway */
    }
    if (tex_id == 6) {
        if (level_edge_match(p1, p2, 23, 21)) return 1;
        if (level_edge_match(p1, p2, 22, 24)) return 1;
        if (level_edge_match(p1, p2, 34, 33)) return 1;
    }
    if (tex_id == 8) {
        if (level_edge_match(p1, p2, 24, 23)) return 1;
    }
    return 0;
}

static AB3D_ATTR_UNUSED int level_patch_zone_wall_yoff(LevelState *level,
                                      int16_t zone_id_word,
                                      int16_t yoff_delta,
                                      int (*wall_match)(int16_t p1, int16_t p2, int16_t tex_id),
                                      const char *log_label)
{
    if (!level || !level->graphics || !level->zone_graph_adds || !level->zone_adds || !level->data)
        return 0;
    if (level->graphics_byte_count == 0)
        return 0;

    int zone_slot = level_find_zone_slot_by_id_word(level, zone_id_word);
    if (zone_slot < 0)
        return 0;
    if (level->num_zone_graph_entries > 0 && zone_slot >= level->num_zone_graph_entries)
        return 0;

    const uint8_t *zgraph = level->zone_graph_adds + (size_t)zone_slot * 8u;
    int32_t lower_gfx_off = read_long(zgraph + 0);
    if (lower_gfx_off <= 0)
        return 0;
    if ((size_t)lower_gfx_off + 2u > level->graphics_byte_count)
        return 0;

    uint8_t *ptr = level->graphics + (size_t)lower_gfx_off;
    uint8_t *gend = level->graphics + level->graphics_byte_count;
    int patched = 0;

    ptr += 2; /* skip stream zone word */
    for (int iter = 0; iter < 512 && ptr + 2 <= gend; iter++) {
        int16_t t = read_word(ptr);
        ptr += 2;
        if (t < 0)
            break;

        size_t skip = zone_gfx_payload_skip(t, ptr);
        if (ptr + skip > gend)
            break;

        if ((t == 0 || t == 13) && skip >= 28u) {
            int16_t p1 = read_word(ptr + 0);
            int16_t p2 = read_word(ptr + 2);
            int16_t tex_id = read_word(ptr + 12);
            if (wall_match && wall_match(p1, p2, tex_id)) {
                int16_t yoff = read_word(ptr + 10);
                write_word_be(ptr + 10, (int16_t)(yoff + yoff_delta));
                patched++;
            }
        }

        ptr += skip;
    }

    if (patched > 0 && log_label && *log_label)
        printf("[LEVELFIX] %s yoff %+d on %d wall segment(s)\n",
               log_label, (int)yoff_delta, patched);

    return patched;
}

static AB3D_ATTR_UNUSED int level_patch_all_lower_zone_walls_yoff(LevelState *level,
                                                 int16_t yoff_delta,
                                                 int (*wall_match)(int16_t p1, int16_t p2, int16_t tex_id),
                                                 const char *log_label,
                                                 int *out_zone_streams_touched)
{
    if (out_zone_streams_touched) *out_zone_streams_touched = 0;
    if (!level || !level->graphics || !level->zone_graph_adds || !level->zone_adds || !level->data)
        return 0;
    if (level->graphics_byte_count == 0)
        return 0;

    int slots = level_zone_slot_count(level);
    if (slots <= 0)
        return 0;
    if (level->num_zone_graph_entries > 0 && slots > level->num_zone_graph_entries)
        slots = level->num_zone_graph_entries;

    int patched_total = 0;
    int streams_touched = 0;

    for (int zone_slot = 0; zone_slot < slots; zone_slot++) {
        int16_t zone_id_word = -1;
        {
            int32_t zoff = read_long(level->zone_adds + (size_t)zone_slot * 4u);
            if (zoff >= 0 && (size_t)zoff + 2u <= level->data_byte_count)
                zone_id_word = read_word(level->data + (size_t)zoff);
        }

        const uint8_t *zgraph = level->zone_graph_adds + (size_t)zone_slot * 8u;
        int32_t lower_gfx_off = read_long(zgraph + 0);
        if (lower_gfx_off <= 0)
            continue;
        if ((size_t)lower_gfx_off + 2u > level->graphics_byte_count)
            continue;

        uint8_t *ptr = level->graphics + (size_t)lower_gfx_off;
        uint8_t *gend = level->graphics + level->graphics_byte_count;
        int patched_this_zone = 0;

        ptr += 2; /* skip stream zone word */
        for (int iter = 0; iter < 512 && ptr + 2 <= gend; iter++) {
            int16_t t = read_word(ptr);
            ptr += 2;
            if (t < 0)
                break;

            size_t skip = zone_gfx_payload_skip(t, ptr);
            if (ptr + skip > gend)
                break;

            if ((t == 0 || t == 13) && skip >= 28u) {
                int16_t p1 = read_word(ptr + 0);
                int16_t p2 = read_word(ptr + 2);
                int16_t tex_id = read_word(ptr + 12);
                if (wall_match && wall_match(p1, p2, tex_id)) {
                    int16_t yoff = read_word(ptr + 10);
                    write_word_be(ptr + 10, (int16_t)(yoff + yoff_delta));
                    patched_this_zone++;
                }
            }

            ptr += skip;
        }

        if (patched_this_zone > 0) {
            streams_touched++;
            patched_total += patched_this_zone;
            printf("[LEVELFIX] %s zone_id=%d slot=%d patched=%d yoff %+d\n",
                   log_label ? log_label : "wall yoff patch",
                   (int)zone_id_word, zone_slot, patched_this_zone, (int)yoff_delta);
        }
    }

    if (out_zone_streams_touched) *out_zone_streams_touched = streams_touched;
    return patched_total;
}

static AB3D_ATTR_UNUSED int level_patch_l4_door_wall_list_scroll_base(LevelState *level, int16_t scroll_delta)
{
    if (!level || !level->door_wall_list || !level->door_wall_list_offsets || !level->graphics)
        return 0;
    if (level->num_doors <= 0)
        return 0;

    int patched_entries = 0;
    int touched_doors = 0;

    for (int di = 0; di < level->num_doors; di++) {
        uint32_t start = level->door_wall_list_offsets[di];
        uint32_t end = level->door_wall_list_offsets[di + 1];
        int patched_this_door = 0;

        for (uint32_t j = start; j < end; j++) {
            uint8_t *ent = level->door_wall_list + (size_t)j * 10u;
            int32_t gfx_off = read_long(ent + 2);
            if (gfx_off < 0 || (size_t)gfx_off + 30u > level->graphics_byte_count)
                continue;

            uint8_t *wall_rec = level->graphics + (size_t)gfx_off;
            int16_t t = read_word(wall_rec + 0);
            if (!(t == 0 || t == 13))
                continue;

            int16_t p1 = read_word(wall_rec + 2);
            int16_t p2 = read_word(wall_rec + 4);
            int16_t tex_id = read_word(wall_rec + 14);
            if (!level_l4_adjacent_door_wall_match(p1, p2, tex_id))
                continue;

            int32_t gfx_base = read_long(ent + 6);
            /* door_routine writes this packed long to wall_rec+10:
             * high word -> runtime totalyoff, low word -> runtime tex_id.
             * Adjust only the high-word y-offset and keep low-word tex_id intact. */
            {
                int16_t base_yoff = (int16_t)(gfx_base >> 16);
                uint16_t base_tex = (uint16_t)(gfx_base & 0xFFFF);
                int16_t patched_yoff = (int16_t)(base_yoff + scroll_delta);
                uint32_t packed = ((uint32_t)(uint16_t)patched_yoff << 16) | (uint32_t)base_tex;
                write_long_be(ent + 6, (int32_t)packed);
            }
            patched_entries++;
            patched_this_door++;
        }

        if (patched_this_door > 0) {
            touched_doors++;
            printf("[LEVELFIX] level 4 door_wall_list door=%d patched=%d scroll %+d\n",
                   di, patched_this_door, (int)scroll_delta);
        }
    }

    printf("[LEVELFIX] level 4 door_wall_list summary: patched=%d doors=%d delta=%d\n",
           patched_entries, touched_doors, (int)scroll_delta);
    return patched_entries;
}

static void level_patch_l4_door_texture_alignment(LevelState *level)
{
    (void)level;
    /* Scale-only path now lives in renderer.c for this level-specific doorway.
     * Keep load-time log so we can verify the hook runs, without mutating yoff. */
    printf("[LEVELFIX] level 4 adjacent-door summary: patched=0 streams=0 delta=0 (scale-only)\n");
    printf("[LEVELFIX] level 4 final summary: stream_patch=%d door_list_patch=%d delta=%d\n",
           0, 0, 0);
}

void level_apply_level_specific_fixes(LevelState *level, int16_t level_num)
{
    if (!level) return;

    printf("[LEVELFIX] probe: level_index=%d level_1indexed=%d\n",
           (int)level_num, (int)level_num + 1);

    /* Level 4 (1-indexed): door panel seams need per-zone vertical texture offset nudges. */
    if (level_num == 3) {
        level_patch_l4_door_texture_alignment(level);
    }
}

static size_t zone_data_guess_len(const LevelState *level, int32_t zoff)
{
    if (!level->data || zoff < 0) return 0;
    size_t cap = level->data_byte_count;
    if (cap == 0) return 512;
    size_t start = (size_t)zoff;
    if (start >= cap) return 0;
    size_t best = cap - start;
    int n = level_zone_slot_count(level);
    for (int i = 0; i < n; i++) {
        int32_t o = read_long(level->zone_adds + (size_t)i * 4u);
        if (o > zoff && (size_t)o < cap) {
            size_t cand = (size_t)o - start;
            if (cand > 0 && cand < best) best = cand;
        }
    }
    if (best > 4096u) best = 4096u;
    return best;
}

static void hexdump_block(const char *label, const uint8_t *p, size_t len)
{
    printf("[ZONELOG] %s (%zu bytes):\n", label, len);
    for (size_t i = 0; i < len; i += 16u) {
        printf("[ZONELOG]   %04zx:", i);
        size_t n = len - i;
        if (n > 16u) n = 16u;
        for (size_t j = 0; j < n; j++)
            printf(" %02x", p[i + j]);
        printf("\n");
    }
}

static void log_gfx_poly_stream(const char *which,
                                int32_t gfx_off, const uint8_t *gbase, size_t gcount)
{
    if (gfx_off < 0 || !gbase || gcount == 0) {
        printf("[ZONELOG]   %s gfx: (none / invalid off=%ld)\n", which, (long)gfx_off);
        return;
    }
    size_t go = (size_t)gfx_off;
    if (go >= gcount) {
        printf("[ZONELOG]   %s gfx: offset %zu out of range (graphics_byte_count=%zu)\n",
               which, go, gcount);
        return;
    }
    const uint8_t *ptr = gbase + go;
    const uint8_t *gend = gbase + gcount;
    int16_t stream_zone = read_word(ptr);
    ptr += 2;
    printf("[ZONELOG]   %s stream at graphics+%ld zone_word=%d\n",
           which, (long)gfx_off, (int)stream_zone);

    int iter = 0;
    while (ptr + 2 <= gend && iter++ < 512) {
        int16_t t = read_word(ptr);
        ptr += 2;
        if (t < 0) {
            printf("[ZONELOG]     [%d] END (type=%d)\n", iter - 1, (int)t);
            break;
        }
        size_t skip = zone_gfx_payload_skip(t, ptr);
        if (ptr + skip > gend) {
            printf("[ZONELOG]     [%d] type=%d TRUNCATED (need %zu bytes)\n",
                   iter - 1, (int)t, skip);
            break;
        }

        switch (t) {
        case 0:
        case 13:
        {
            int16_t p1 = read_word(ptr + 0);
            int16_t p2 = read_word(ptr + 2);
            int16_t tex_id = read_word(ptr + 12);
            int32_t topw = read_long(ptr + 18);
            int32_t botw = read_long(ptr + 22);
            printf("[ZONELOG]     [%d] wall(%d) p1=%d p2=%d tex_id=%d top=%ld bot=%ld\n",
                   iter - 1, (int)t, (int)p1, (int)p2, (int)tex_id, (long)topw, (long)botw);
            break;
        }
        case 1:
        case 2:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        {
            int16_t ypos = read_word(ptr + 0);
            int16_t nsm1 = read_word(ptr + 2);
            int sides_full = (int)nsm1 + 1;
            if (sides_full < 0) sides_full = 0;
            if (sides_full > 100) sides_full = 100;
            int spr = sides_full < 40 ? sides_full : 40;
            printf("[ZONELOG]     [%d] floorish(%d) ypos=%d sides=%d pts=[",
                   iter - 1, (int)t, (int)ypos, sides_full);
            for (int s = 0; s < spr; s++) {
                printf("%d%s", (int)read_word(ptr + 4 + s * 2), s + 1 < spr ? "," : "");
            }
            if (sides_full > spr) printf(",...");
            printf("] scale=%d tile_off=%d bright_off=%d\n",
                   (int)read_word(ptr + 4 + sides_full * 2 + 2),
                   (int)read_word(ptr + 4 + sides_full * 2 + 4),
                   (int)read_word(ptr + 4 + sides_full * 2 + 6));
            break;
        }
        case 4:
            printf("[ZONELOG]     [%d] object draw_mode=%d\n", iter - 1, (int)read_word(ptr));
            break;
        case 5:
            printf("[ZONELOG]     [%d] arc center=%d edge=%d\n",
                   iter - 1, (int)read_word(ptr), (int)read_word(ptr + 2));
            break;
        case 6:
            printf("[ZONELOG]     [%d] light beam\n", iter - 1);
            break;
        case 12:
            printf("[ZONELOG]     [%d] backdrop (sky marker)\n", iter - 1);
            break;
        case 3:
            printf("[ZONELOG]     [%d] clip setter (no data)\n", iter - 1);
            break;
        default:
            printf("[ZONELOG]     [%d] unknown type=%d skip=%zu\n", iter - 1, (int)t, skip);
            break;
        }
        ptr += skip;
    }
    if (iter >= 512)
        printf("[ZONELOG]     ... (truncated after 512 entries)\n");
}

void level_log_player_zone_full(const GameState *state)
{
    if (!state) return;
    const PlayerState *plr = (state->mode == MODE_SLAVE) ? &state->plr2 : &state->plr1;
    int16_t zid = plr->zone;

    printf("[ZONELOG] ========== player zone dump ==========\n");
    printf("[ZONELOG] viewer=%s zone_id=%d stood_in_top=%d roompt=%ld list_of_graph=%ld angpos=%d\n",
           state->mode == MODE_SLAVE ? "plr2" : "plr1",
           (int)zid, (int)plr->stood_in_top, (long)plr->roompt, (long)plr->list_of_graph_rooms,
           (int)plr->angpos);

    const LevelState *level = &state->level;
    if (!level->zone_adds || !level->data) {
        printf("[ZONELOG] (no zone_adds/data)\n");
        return;
    }

    int slots = level_zone_slot_count(level);
    if (zid < 0 || zid >= slots) {
        printf("[ZONELOG] zone_id out of range (slots=%d)\n", slots);
        return;
    }

    int32_t zoff = read_long(level->zone_adds + (size_t)zid * 4u);
    printf("[ZONELOG] zone_adds[%d] -> data offset %ld\n", (int)zid, (long)zoff);
    if (zoff < 0 || (level->data_byte_count > 0 && (size_t)zoff >= level->data_byte_count)) {
        printf("[ZONELOG] invalid zone offset\n");
        return;
    }

    const uint8_t *zd = level->data + zoff;
    size_t zlen = zone_data_guess_len(level, zoff);
    hexdump_block("zone data (raw)", zd, zlen);

    printf("[ZONELOG] zone fields (decoded):\n");
    printf("[ZONELOG]   id_word=%d floor=%ld roof=%ld upper_floor=%ld upper_roof=%ld water=%ld\n",
           (int)read_word(zd + 0),
           (long)read_long(zd + ZONE_OFF_FLOOR),
           (long)read_long(zd + ZONE_OFF_ROOF),
           (long)read_long(zd + ZONE_OFF_UPPER_FLOOR),
           (long)read_long(zd + ZONE_OFF_UPPER_ROOF),
           (long)read_long(zd + ZONE_OFF_WATER));
    printf("[ZONELOG]   brightness=%d upper_bright=%d back=%d tel_zone=%d tel=(%d,%d) list_of_graph=%d\n",
           (int)read_word(zd + ZONE_OFF_BRIGHTNESS),
           (int)read_word(zd + ZONE_OFF_UPPER_BRIGHT),
           (int)read_word(zd + ZONE_OFF_BACK),
           (int)read_word(zd + ZONE_OFF_TEL_ZONE),
           (int)read_word(zd + ZONE_OFF_TEL_X),
           (int)read_word(zd + ZONE_OFF_TEL_Z),
           (int)read_word(zd + ZONE_OFF_LIST_OF_GRAPH));

    if (!level->zone_graph_adds || !level->graphics) {
        printf("[ZONELOG] (no zone_graph_adds/graphics)\n");
        return;
    }
    if (level->num_zone_graph_entries > 0 && zid >= level->num_zone_graph_entries) {
        printf("[ZONELOG] zone_id >= num_zone_graph_entries (%d)\n", level->num_zone_graph_entries);
        return;
    }

    const uint8_t *zgraph = level->zone_graph_adds + (size_t)zid * 8u;
    int32_t lower_gfx = read_long(zgraph + 0);
    int32_t upper_gfx = read_long(zgraph + 4);
    printf("[ZONELOG] zone_graph_adds: lower gfx_off=%ld upper gfx_off=%ld\n",
           (long)lower_gfx, (long)upper_gfx);

    size_t gbc = level->graphics_byte_count;
    const uint8_t *gbase = level->graphics;
    log_gfx_poly_stream("lower", lower_gfx, gbase, gbc);
    log_gfx_poly_stream("upper", upper_gfx, gbase, gbc);
    printf("[ZONELOG] ========== end zone dump ==========\n");
}

void level_log_zone_full(const LevelState *level, int16_t zone_id, const char *label)
{
    const char *which = (label && label[0] != '\0') ? label : "zone";
    int16_t zid = zone_id;

    printf("[ZONELOG] ========== %s zone dump ==========\n", which);
    printf("[ZONELOG] zone_id=%d\n", (int)zid);

    if (!level) {
        printf("[ZONELOG] (no level)\n");
        printf("[ZONELOG] ========== end zone dump ==========\n");
        return;
    }
    if (!level->zone_adds || !level->data) {
        printf("[ZONELOG] (no zone_adds/data)\n");
        printf("[ZONELOG] ========== end zone dump ==========\n");
        return;
    }

    {
        int slots = level_zone_slot_count(level);
        if (zid < 0 || zid >= slots) {
            printf("[ZONELOG] zone_id out of range (slots=%d)\n", slots);
            printf("[ZONELOG] ========== end zone dump ==========\n");
            return;
        }
    }

    int32_t zoff = read_long(level->zone_adds + (size_t)zid * 4u);
    printf("[ZONELOG] zone_adds[%d] -> data offset %ld\n", (int)zid, (long)zoff);
    if (zoff < 0 || (level->data_byte_count > 0 && (size_t)zoff >= level->data_byte_count)) {
        printf("[ZONELOG] invalid zone offset\n");
        printf("[ZONELOG] ========== end zone dump ==========\n");
        return;
    }

    const uint8_t *zd = level->data + zoff;
    size_t zlen = zone_data_guess_len(level, zoff);
    hexdump_block("zone data (raw)", zd, zlen);

    printf("[ZONELOG] zone fields (decoded):\n");
    printf("[ZONELOG]   id_word=%d floor=%ld roof=%ld upper_floor=%ld upper_roof=%ld water=%ld\n",
           (int)read_word(zd + 0),
           (long)read_long(zd + ZONE_OFF_FLOOR),
           (long)read_long(zd + ZONE_OFF_ROOF),
           (long)read_long(zd + ZONE_OFF_UPPER_FLOOR),
           (long)read_long(zd + ZONE_OFF_UPPER_ROOF),
           (long)read_long(zd + ZONE_OFF_WATER));
    printf("[ZONELOG]   brightness=%d upper_bright=%d back=%d tel_zone=%d tel=(%d,%d) list_of_graph=%d\n",
           (int)read_word(zd + ZONE_OFF_BRIGHTNESS),
           (int)read_word(zd + ZONE_OFF_UPPER_BRIGHT),
           (int)read_word(zd + ZONE_OFF_BACK),
           (int)read_word(zd + ZONE_OFF_TEL_ZONE),
           (int)read_word(zd + ZONE_OFF_TEL_X),
           (int)read_word(zd + ZONE_OFF_TEL_Z),
           (int)read_word(zd + ZONE_OFF_LIST_OF_GRAPH));

    if (!level->zone_graph_adds || !level->graphics) {
        printf("[ZONELOG] (no zone_graph_adds/graphics)\n");
        printf("[ZONELOG] ========== end zone dump ==========\n");
        return;
    }
    if (level->num_zone_graph_entries > 0 && zid >= level->num_zone_graph_entries) {
        printf("[ZONELOG] zone_id >= num_zone_graph_entries (%d)\n", level->num_zone_graph_entries);
        printf("[ZONELOG] ========== end zone dump ==========\n");
        return;
    }

    const uint8_t *zgraph = level->zone_graph_adds + (size_t)zid * 8u;
    int32_t lower_gfx = read_long(zgraph + 0);
    int32_t upper_gfx = read_long(zgraph + 4);
    printf("[ZONELOG] zone_graph_adds: lower gfx_off=%ld upper gfx_off=%ld\n",
           (long)lower_gfx, (long)upper_gfx);

    {
        size_t gbc = level->graphics_byte_count;
        const uint8_t *gbase = level->graphics;
        log_gfx_poly_stream("lower", lower_gfx, gbase, gbc);
        log_gfx_poly_stream("upper", upper_gfx, gbase, gbc);
    }
    printf("[ZONELOG] ========== end zone dump ==========\n");
}

void level_log_zones(const LevelState *level)
{
    if (!level->zone_adds || !level->data || level->num_zones <= 0)
        return;
    const uint8_t *ld = level->data;
    size_t data_len = level->data_byte_count;
    printf("[LEVEL] Zones: %d zones (offset table big-endian)\n", level->num_zones);
    for (int z = 0; z < level->num_zones; z++) {
        int32_t zoff = read_long(level->zone_adds + (size_t)z * 4u);
        if (zoff < 0 || (data_len != 0 && (size_t)zoff + 48u > data_len)) {
            printf("[LEVEL]   zone[%d] offset %ld - out of range (data_len=%zu)\n", z, (long)zoff, data_len);
            continue;
        }
        const uint8_t *zd = ld + zoff;
        int16_t zone_id = read_word(zd + 0);
        int32_t floor_y = read_long(zd + ZONE_OFF_FLOOR);
        int32_t roof_y = read_long(zd + ZONE_OFF_ROOF);
        int16_t bright_lo = read_word(zd + ZONE_OFF_BRIGHTNESS);
        int16_t bright_hi = read_word(zd + ZONE_OFF_UPPER_BRIGHT);
        unsigned anim_lo = zone_bright_anim_type(bright_lo);
        unsigned anim_hi = zone_bright_anim_type(bright_hi);
        printf("[LEVEL]   zone[%d] offset %ld id=%d floor=%ld roof=%ld bright=(%d,%d) anim=(%s,%s)\n",
               z, (long)zoff, (int)zone_id, (long)floor_y, (long)roof_y, (int)bright_lo, (int)bright_hi,
               zone_anim_flag_name(anim_lo), zone_anim_flag_name(anim_hi));
    }
}

int level_set_zone_roof(LevelState *level, int16_t zone_id, int32_t roof_y)
{
    uint8_t *zd = level_get_zone_data_ptr(level, zone_id);
    if (!zd) return -1;
    write_long_be(zd + ZONE_OFF_ROOF, roof_y);

    return 0;
}

int level_set_zone_floor(LevelState *level, int16_t zone_id, int32_t floor_y)
{
    uint8_t *zd = level_get_zone_data_ptr(level, zone_id);
    if (!zd) return -1;
    write_long_be(zd + ZONE_OFF_FLOOR, floor_y);

    return 0;
}

int level_set_zone_water(LevelState *level, int16_t zone_id, int32_t water_y)
{
    uint8_t *zd = level_get_zone_data_ptr(level, zone_id);
    if (!zd) return -1;
    write_long_be(zd + ZONE_OFF_WATER, water_y);

    return 0;
}
