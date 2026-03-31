/*
 * Standalone level parser - outputs door, switch and lift information.
 * Format follows Amiga source only: LevelFormat doc, Anims.s (DoorRoutine, LiftRoutine, SwitchRoutine).
 *
 * Usage: parse_level [data_dir] [level]
 *   level: 1..16 or a..p (default a)
 *   Loads data_dir/levels/level_<x>/twolev.graph.bin and twolev.bin.
 *   If data_dir omitted, tries "data", "..", "../data", "."
 *
 * Graphics file (LEVELGRAPHICS): Amiga big-endian
 *   Long 0: door_offset, Long 4: lift_offset, Long 8: switch_offset, Long 12: zone_graph_offset
 *   Byte 16+: zone_adds (not used here)
 *
 * Door/Lift (Anims.s): terminator 999. Per entry: 18-byte header then wall list.
 *   Header: Bottom(w), Top(w), curr(w), dir(w), Ptr(l), zone(w), conditions(w), 2 bytes at 16-17.
 *   Wall list at 18: (wall_number(w), ptr(l), graphic(l)) until wall_number < 0, then +2.
 *
 * Switch (Anims.s): 14 bytes per entry (adda.w #14,a0). First word = zone; zone < 0 = end.
 *   LevelFormat: NUM ZONE (w), First point (w), Ptr to graphics (l), status (l) + 2 bytes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include "sb_decompress.h"

static int16_t read_be16(const uint8_t *p)
{
    return (int16_t)((p[0] << 8) | p[1]);
}

static int32_t read_be32(const uint8_t *p)
{
    return (int32_t)((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
}

static int32_t read_le32(const uint8_t *p)
{
    return (int32_t)((p[3] << 24) | (p[2] << 16) | (p[1] << 8) | p[0]);
}

static size_t zone_gfx_entry_data_skip(int16_t entry_type, const uint8_t *data)
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
        int16_t num_sides_m1 = read_be16(data + 2);
        int sides = (int)num_sides_m1 + 1;
        if (sides < 0) sides = 0;
        if (sides > 100) sides = 100;
        return (size_t)(4 + 2 * sides + 10);
    }
    default:
        return 0;
    }
}

static int clamp_zone_limit(int num_zones, int zone_slots, int zone_graph_rows)
{
    int limit = zone_slots;
    if (num_zones > 0 && num_zones < limit) limit = num_zones;
    if (zone_graph_rows > 0 && zone_graph_rows < limit) limit = zone_graph_rows;
    if (limit < 0) limit = 0;
    return limit;
}

static void analyze_sky_synthesis(const uint8_t *graph, size_t graph_size,
                                  const uint8_t *data, size_t data_size,
                                  int num_zones,
                                  int32_t door_off, int32_t lift_off, int32_t switch_off,
                                  int32_t zone_graph_off)
{
    if (!graph || !data || zone_graph_off <= 16 || (size_t)zone_graph_off >= graph_size) {
        printf("\n--- SKY ANALYSIS ---\n(unavailable)\n");
        return;
    }

    int zone_slots = (zone_graph_off - 16) / 4;
    if (zone_slots < num_zones) zone_slots = num_zones;
    if (zone_slots < 0) zone_slots = 0;

    int32_t next = (int32_t)graph_size;
    if (door_off > zone_graph_off && door_off < next) next = door_off;
    if (lift_off > zone_graph_off && lift_off < next) next = lift_off;
    if (switch_off > zone_graph_off && switch_off < next) next = switch_off;
    if (next <= zone_graph_off) next = (int32_t)graph_size;
    int zone_graph_rows = (next - zone_graph_off) / 8;
    if (zone_graph_rows < 0) zone_graph_rows = 0;

    int zone_limit = clamp_zone_limit(num_zones, zone_slots, zone_graph_rows);

    int total_streams = 0;
    int add_zone_or_marker = 0;
    int add_marker_only = 0;
    int add_with_global_plane_block = 0;
    int add_with_dynamic_plane_dedupe = 0;
    uint8_t *global_plane_occ = (uint8_t *)calloc(65536u, 1u);
    uint8_t *dynamic_plane_occ = NULL;

    printf("\n--- SKY ANALYSIS ---\n");
    printf("zone_slots=%d zone_graph_rows=%d zone_limit=%d\n",
           zone_slots, zone_graph_rows, zone_limit);

    if (global_plane_occ) {
        for (int zone = 0; zone < zone_limit; zone++) {
            int32_t zone_off = read_be32(graph + 16 + zone * 4);
            if (zone_off < 0 || (size_t)zone_off + 48u > data_size) continue;
            const uint8_t *zd = data + zone_off;
            const uint8_t *zg = graph + zone_graph_off + zone * 8;
            int32_t lower_gfx_off = read_be32(zg + 0);
            int32_t upper_gfx_off = read_be32(zg + 4);

            for (int use_upper = 0; use_upper <= 1; use_upper++) {
                int32_t gfx_off = use_upper ? upper_gfx_off : lower_gfx_off;
                int32_t zone_floor;
                int32_t zone_roof;
                int16_t zone_floor_y;
                int16_t zone_roof_y;
                const uint8_t *scan;
                int scan_iter = 500;
                if (gfx_off <= 0 || (size_t)gfx_off + 2u > graph_size) continue;

                if (use_upper) {
                    int32_t uf = read_be32(zd + 10);
                    int32_t ur = read_be32(zd + 14);
                    zone_floor = (uf != 0) ? uf : read_be32(zd + 2);
                    zone_roof = (ur != 0) ? ur : read_be32(zd + 6);
                } else {
                    zone_floor = read_be32(zd + 2);
                    zone_roof = read_be32(zd + 6);
                }
                zone_floor_y = (int16_t)(zone_floor >> 6);
                zone_roof_y = (int16_t)(zone_roof >> 6);

                scan = graph + gfx_off + 2;
                while (scan_iter-- > 0) {
                    int16_t t = read_be16(scan);
                    scan += 2;
                    if (t < 0) break;

                    if (t == 1 || t == 2 || t == 7 || t == 8 || t == 9 || t == 10 || t == 11) {
                        int16_t ypos = read_be16(scan);
                        int sides = (int)read_be16(scan + 2) + 1;
                        int16_t plane = ypos;
                        if (sides < 0) sides = 0;
                        if (sides > 100) sides = 100;
                        if (sides >= 3) {
                            if (t == 1) plane = zone_floor_y;
                            else if (t == 2) plane = zone_roof_y;
                            global_plane_occ[(uint16_t)plane] = 1;
                        }
                    }

                    {
                        size_t skip = zone_gfx_entry_data_skip(t, scan);
                        if (skip == 0) {
                            if (t != 3 && t != 12) break;
                            continue;
                        }
                        scan += skip;
                    }
                }
            }
        }
        dynamic_plane_occ = (uint8_t *)malloc(65536u);
        if (dynamic_plane_occ) {
            memcpy(dynamic_plane_occ, global_plane_occ, 65536u);
        }
    }

    for (int zone = 0; zone < zone_limit; zone++) {
        int32_t zone_off = read_be32(graph + 16 + zone * 4);
        if (zone_off < 0 || (size_t)zone_off + 48u > data_size) continue;

        const uint8_t *zd = data + zone_off;
        int zone_back = (read_be16(zd + 36) != 0) ? 1 : 0;

        const uint8_t *zg = graph + zone_graph_off + zone * 8;
        int32_t lower_gfx_off = read_be32(zg + 0);
        int32_t upper_gfx_off = read_be32(zg + 4);

        for (int use_upper = 0; use_upper <= 1; use_upper++) {
            int32_t gfx_off = use_upper ? upper_gfx_off : lower_gfx_off;
            int32_t zone_roof;
            int16_t sky_ypos;
            const uint8_t *scan;
            int scan_iter = 500;
            int stream_marker = 0;
            int stream_plane = 0;
            int floor_polys = 0;
            int pred_add_zone_or_marker;
            int pred_add_marker_only;
            int pred_add_global_block;
            int pred_add_dynamic;

            if (gfx_off <= 0 || (size_t)gfx_off + 2u > graph_size) continue;
            total_streams++;

            if (use_upper) {
                int32_t ur = read_be32(zd + 14);
                zone_roof = (ur != 0) ? ur : read_be32(zd + 6);
            } else {
                zone_roof = read_be32(zd + 6);
            }
            sky_ypos = (int16_t)(zone_roof >> 6);

            scan = graph + gfx_off + 2;
            while (scan_iter-- > 0) {
                int16_t t = read_be16(scan);
                scan += 2;
                if (t < 0) break;

                if (t == 12) stream_marker = 1;
                if (t == 1 || t == 2 || t == 7 || t == 8 || t == 9 || t == 10 || t == 11) {
                    int16_t ypos = read_be16(scan);
                    int sides = (int)read_be16(scan + 2) + 1;
                    if (sides < 0) sides = 0;
                    if (sides > 100) sides = 100;
                    if (sides >= 3) {
                        if (t == 1) floor_polys++;
                        if (t == 2 || ypos == sky_ypos) stream_plane = 1;
                    }
                }

                {
                    size_t skip = zone_gfx_entry_data_skip(t, scan);
                    if (skip == 0) {
                        if (t != 3 && t != 12) break;
                        continue;
                    }
                    scan += skip;
                }
            }

            pred_add_zone_or_marker = ((zone_back || stream_marker) && !stream_plane) ? floor_polys : 0;
            pred_add_marker_only = (stream_marker && !stream_plane) ? floor_polys : 0;
            pred_add_global_block = ((zone_back || stream_marker) &&
                                     !(global_plane_occ && global_plane_occ[(uint16_t)sky_ypos]))
                                        ? floor_polys : 0;
            pred_add_dynamic = ((zone_back || stream_marker) && !stream_plane &&
                                floor_polys > 0 &&
                                !(dynamic_plane_occ && dynamic_plane_occ[(uint16_t)sky_ypos]))
                                   ? 1 : 0;
            add_zone_or_marker += pred_add_zone_or_marker;
            add_marker_only += pred_add_marker_only;
            add_with_global_plane_block += pred_add_global_block;
            add_with_dynamic_plane_dedupe += pred_add_dynamic;
            if (pred_add_dynamic && dynamic_plane_occ) {
                dynamic_plane_occ[(uint16_t)sky_ypos] = 1;
            }

            if (pred_add_zone_or_marker > 0 || stream_marker || stream_plane || zone_back) {
                printf("zone=%d stream=%s back=%d marker=%d floors=%d stream_plane=%d"
                       " add(zone|marker)=%d add(marker_only)=%d"
                       " add(global_plane_block)=%d add(dynamic_dedupe)=%d\n",
                       zone, use_upper ? "upper" : "lower", zone_back, stream_marker,
                       floor_polys, stream_plane,
                       pred_add_zone_or_marker, pred_add_marker_only,
                       pred_add_global_block, pred_add_dynamic);
            }
        }
    }

    printf("SUMMARY streams=%d predicted_add(zone|marker)=%d predicted_add(marker_only)=%d"
           " predicted_add(global_plane_block)=%d predicted_add(dynamic_dedupe)=%d\n",
           total_streams, add_zone_or_marker, add_marker_only,
           add_with_global_plane_block, add_with_dynamic_plane_dedupe);
    free(global_plane_occ);
    free(dynamic_plane_occ);
}

/* Amiga wall list: (wall_number(w), ptr(l), graphic(l)) until wall_number < 0, then +2 (Anims.s) */
static const uint8_t *skip_wall_list(const uint8_t *p)
{
    while (read_be16(p) >= 0)
        p += 2 + 4 + 4;
    return p + 2;
}

/* Load file; if =SB= compressed, decompress. Uses sb_load_file from sb_decompress. */
static int load_level_file(const char *path, uint8_t **out, size_t *size)
{
    return sb_load_file(path, out, size);
}

int main(int argc, char **argv)
{
    /* Usage:
     *   parse_level [data_dir] [level]
     *   level can be 1..16 or a..p (default: a) */
    const char *base = (argc > 1) ? argv[1] : "data";
    char level_ch = 'a';
    static const char *try_bases[] = { "data", "..", "../data", ".", NULL };
    char path[1024];
    uint8_t *graph = NULL, *data = NULL;
    size_t graph_size = 0, data_size = 0;
    const char *used_base = NULL;

    if (argc > 2 && argv[2] && argv[2][0]) {
        if (isdigit((unsigned char)argv[2][0])) {
            int n = atoi(argv[2]);
            if (n >= 1 && n <= 16) level_ch = (char)('a' + (n - 1));
        } else if (isalpha((unsigned char)argv[2][0])) {
            level_ch = (char)tolower((unsigned char)argv[2][0]);
            if (level_ch < 'a' || level_ch > 'p') level_ch = 'a';
        }
    }

    if (argc > 1) {
        snprintf(path, sizeof(path), "%s/levels/level_%c/twolev.graph.bin", base, level_ch);
        if (load_level_file(path, &graph, &graph_size) == 0) used_base = base;
    }
    if (!used_base) {
        for (int i = 0; try_bases[i]; i++) {
            snprintf(path, sizeof(path), "%s/levels/level_%c/twolev.graph.bin", try_bases[i], level_ch);
            if (load_level_file(path, &graph, &graph_size) == 0) {
                used_base = try_bases[i];
                break;
            }
        }
    }
    if (!graph) {
        fprintf(stderr, "Cannot open graphics file. Run from project root or: parse_level <data_dir> [level]\n");
        return 1;
    }

    base = used_base ? used_base : ".";
    snprintf(path, sizeof(path), "%s/levels/level_%c/twolev.bin", base, level_ch);
    if (load_level_file(path, &data, &data_size) != 0) {
        snprintf(path, sizeof(path), "levels/level_%c/twolev.bin", level_ch);
        if (load_level_file(path, &data, &data_size) != 0) {
            fprintf(stderr, "Cannot open level data file (twolev.bin)\n");
            free(graph);
            return 1;
        }
    }

    /* Graphics header: 4 longs (Amiga BE). Try LE if BE offsets look invalid. */
    int32_t door_off = read_be32(graph + 0);
    int32_t lift_off = read_be32(graph + 4);
    int32_t switch_off = read_be32(graph + 8);
    int32_t zone_graph_off = read_be32(graph + 12);
    {
        int be_ok = (door_off >= 0 && (uint32_t)door_off <= 0x00FFFFFFu &&
                     lift_off >= 0 && (uint32_t)lift_off <= 0x00FFFFFFu &&
                     switch_off >= 0 && (uint32_t)switch_off <= 0x00FFFFFFu &&
                     zone_graph_off >= 0 && (uint32_t)zone_graph_off <= 0x00FFFFFFu);
        if (!be_ok) {
            door_off = read_le32(graph + 0);
            lift_off = read_le32(graph + 4);
            switch_off = read_le32(graph + 8);
            zone_graph_off = read_le32(graph + 12);
            if (door_off >= 0 && lift_off >= 0)
                printf("[parse_level] Graphics header interpreted as little-endian\n");
        }
    }

    int num_zones = (int)read_be16(data + 16);
    if (num_zones <= 0) num_zones = 256; /* unknown */

    printf("[parse_level] Level: level_%c\n", level_ch);
    printf("[parse_level] Graphics: %zu bytes. Level data: %zu bytes. num_zones=%d\n", graph_size, data_size, num_zones);
    printf("[parse_level] Offsets: door=%ld lift=%ld switch=%ld zone_graph=%ld\n",
           (long)door_off, (long)lift_off, (long)switch_off, (long)zone_graph_off);

    if (door_off < 0 || (size_t)door_off >= graph_size) {
        printf("[parse_level] Note: header not in Amiga format (or wrong endianness). First 16 bytes hex:\n");
        for (int i = 0; i < 16 && i < (int)graph_size; i++) printf(" %02X", graph[i]);
        printf("\n");
    }

    /* ----- Doors (Anims.s DoorRoutine: 999 terminator, 18-byte header + wall list) ----- */
    printf("\n--- DOORS ---\n");
    if (door_off < 0 || (size_t)door_off >= graph_size) {
        printf("(door offset out of range)\n");
    } else {
        const uint8_t *p = graph + door_off;
        int16_t w = read_be16(p);
        if (w == 999) {
            printf("(no doors - 999 terminator at start)\n");
        } else {
            int idx = 0;
            while (read_be16(p) != 999) {
                int16_t bottom = read_be16(p + 0), top = read_be16(p + 2), curr = read_be16(p + 4), dir = read_be16(p + 6);
                int16_t zone = read_be16(p + 12);
                int16_t conditions = read_be16(p + 14);
                int16_t mode = read_be16(p + 16);
                printf("  door[%d] zone=%d bottom=%d top=%d curr=%d dir=%d conditions=0x%04X mode=0x%04X\n",
                       idx, (int)zone, (int)bottom, (int)top, (int)curr, (int)dir,
                       (unsigned)(uint16_t)conditions, (unsigned)(uint16_t)mode);
                p = skip_wall_list(p + 18);
                idx++;
                if (idx > 512) break;
            }
            printf("  total doors: %d\n", idx);
        }
    }

    /* ----- Lifts (Anims.s LiftRoutine: same as doors) ----- */
    printf("\n--- LIFTS ---\n");
    if (lift_off < 0 || (size_t)lift_off >= graph_size) {
        printf("(lift offset out of range)\n");
    } else {
        const uint8_t *p = graph + lift_off;
        int16_t w = read_be16(p);
        if (w == 999) {
            printf("(no lifts - 999 terminator at start)\n");
        } else {
            int idx = 0;
            while (read_be16(p) != 999) {
                int16_t bottom = read_be16(p + 0), top = read_be16(p + 2), curr = read_be16(p + 4), dir = read_be16(p + 6);
                int16_t zone = read_be16(p + 12);
                int16_t conditions = read_be16(p + 14);
                int16_t mode = read_be16(p + 16);
                printf("  lift[%d] zone=%d bottom=%d top=%d curr=%d dir=%d conditions=0x%04X mode=0x%04X\n",
                       idx, (int)zone, (int)bottom, (int)top, (int)curr, (int)dir,
                       (unsigned)(uint16_t)conditions, (unsigned)(uint16_t)mode);
                p = skip_wall_list(p + 18);
                idx++;
                if (idx > 512) break;
            }
            printf("  total lifts: %d\n", idx);
        }
    }

    /* ----- Switches (Anims.s SwitchRoutine: 14 bytes per entry, zone at 0, zone < 0 = end) ----- */
    printf("\n--- SWITCHES ---\n");
    if (switch_off < 0 || (size_t)switch_off >= graph_size) {
        printf("(switch offset out of range)\n");
    } else {
        const uint8_t *p = graph + switch_off;
        int idx = 0;
        while (1) {
            int16_t zone = read_be16(p);
            if (zone < 0) break;
            /* LevelFormat: NUM ZONE (w), First point (w), Ptr (l), status (l); Anims 14 bytes so +2 more */
            int16_t first_pt = read_be16(p + 2);
            int32_t ptr_gfx = read_be32(p + 4);
            int32_t status = read_be32(p + 8);
            printf("  switch[%d] zone=%d first_pt=%d ptr_gfx=%ld status=0x%lX\n",
                   idx, (int)zone, (int)first_pt, (long)ptr_gfx, (long)(unsigned)status);
            p += 14;
            idx++;
            if (idx > 256) break;
        }
        printf("  total switches: %d\n", idx);
    }

    /* ----- OBJECTS (from level data) ----- */
    {
        int32_t obj_off = read_be32(data + 30);
        printf("\n--- OBJECTS ---\n");
        printf("object_offset=%ld\n", (long)obj_off);
        if (obj_off < 0 || (size_t)obj_off >= data_size) {
            printf("(object offset out of range)\n");
        } else {
            const uint8_t *o = data + obj_off;
            for (int i = 0; i < 256; i++, o += 64) {
                int16_t cid = read_be16(o + 0);
                int16_t zone = read_be16(o + 12);
                int16_t unk14 = read_be16(o + 14);
                int16_t groom = read_be16(o + 26);
                int8_t width_or_3d = (int8_t)o[6];
                int8_t number = (int8_t)o[16];
                int16_t vect = read_be16(o + 8);
                int16_t frame = read_be16(o + 10);
                if (cid < 0) break;
                if (width_or_3d == (int8_t)0xFF || vect == 2 || number == 2 || groom != zone) {
                    printf("  obj[%d] cid=%d zone=%d unk14=%d groom=%d n=%d w3d=%d vect=%d frame=%d in_top=%d\n",
                           i, (int)cid, (int)zone, (int)unk14, (int)groom, (int)number, (int)width_or_3d,
                           (int)vect, (int)frame, (int)(int8_t)o[63]);
                }
            }
        }
    }

    analyze_sky_synthesis(graph, graph_size, data, data_size, num_zones,
                          door_off, lift_off, switch_off, zone_graph_off);

    free(graph);
    free(data);
    return 0;
}
