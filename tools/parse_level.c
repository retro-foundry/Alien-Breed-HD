/*
 * Standalone level parser - outputs door, switch and lift information.
 * Format follows Amiga source only: LevelFormat doc, Anims.s (DoorRoutine, LiftRoutine, SwitchRoutine).
 *
 * Usage: parse_level [data_dir]
 *   Loads data_dir/levels/level_a/twolev.graph.bin and data_dir/levels/level_a/twolev.bin
 *   (level one = level_a). If data_dir omitted, tries "data", "..", "../data", "."
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
    /* Try base paths so it works from build/ or project root. Level one = level_a. */
    const char *base = (argc > 1) ? argv[1] : "data";
    static const char *try_bases[] = { "data", "..", "../data", ".", NULL };
    char path[1024];
    uint8_t *graph = NULL, *data = NULL;
    size_t graph_size = 0, data_size = 0;
    const char *used_base = NULL;

    if (argc > 1) {
        snprintf(path, sizeof(path), "%s/levels/level_a/twolev.graph.bin", base);
        if (load_level_file(path, &graph, &graph_size) == 0) used_base = base;
    }
    if (!used_base) {
        for (int i = 0; try_bases[i]; i++) {
            snprintf(path, sizeof(path), "%s/levels/level_a/twolev.graph.bin", try_bases[i]);
            if (load_level_file(path, &graph, &graph_size) == 0) {
                used_base = try_bases[i];
                break;
            }
        }
    }
    if (!graph) {
        fprintf(stderr, "Cannot open graphics file. Run from project root or: parse_level <data_dir>\n");
        return 1;
    }

    base = used_base ? used_base : ".";
    snprintf(path, sizeof(path), "%s/levels/level_a/twolev.bin", base);
    if (load_level_file(path, &data, &data_size) != 0) {
        snprintf(path, sizeof(path), "levels/level_a/twolev.bin");
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

    free(graph);
    free(data);
    return 0;
}
