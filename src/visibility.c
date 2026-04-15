/*
 * Alien Breed 3D I - PC Port
 * visibility.c - Zone ordering and line-of-sight (full implementation)
 *
 * Translated from: OrderZones.s, AB3DI.s (CanItBeSeen)
 */

#include "visibility.h"
#include "level.h"
#include "math_tables.h"
#include <stdio.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Helper: read big-endian values from level data
 * ----------------------------------------------------------------------- */
static int16_t read_be16(const uint8_t *p)
{
    return (int16_t)((p[0] << 8) | p[1]);
}

static int32_t read_be32(const uint8_t *p)
{
    return (int32_t)((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
}

/* 68020 btst/bset on data registers uses bit index modulo 32. */
static inline uint32_t reg_bit32(unsigned int bit_index)
{
    return (uint32_t)1u << (bit_index & 31u);
}

static inline int reg_btst32(uint32_t value, unsigned int bit_index)
{
    return (value & reg_bit32(bit_index)) != 0u;
}

/* Resolve ListOfGraphRooms entry word to a concrete zone id.
 * Amiga data stores a graph index when graph tables are present; otherwise
 * some synthetic data uses direct zone ids. */
static int resolve_lgr_entry_zone_id(const LevelState *level, int16_t entry_word, int16_t *out_zone_id)
{
    if (!out_zone_id || entry_word < 0) return 0;
    if (level && level->zone_graph_adds && level->graphics) {
        if (level->num_zone_graph_entries > 0 && entry_word >= level->num_zone_graph_entries)
            return 0;
        {
            uint32_t gfx_off = (uint32_t)read_be32(level->zone_graph_adds + (unsigned)entry_word * 8u);
            if (level->graphics_byte_count > 0 &&
                ((size_t)gfx_off + 2u > level->graphics_byte_count))
                return 0;
            *out_zone_id = read_be16(level->graphics + gfx_off);
            return 1;
        }
    }

    *out_zone_id = entry_word;
    return 1;
}

/* Floor line offsets. Amiga side test uses words at 4 and 6 directly: result = dx*word6 - dz*word4. */
#define FLINE_SIZE    16
#define FLINE_X       0
#define FLINE_Z       2
#define FLINE_WORD4   4   /* Amiga muls 4(a1,d0.w),d3; our layout: xlen */
#define FLINE_WORD6   6   /* Amiga muls 6(a1,d0.w),d2; our layout: zlen */
#define FLINE_XLEN    FLINE_WORD4
#define FLINE_ZLEN    FLINE_WORD6
#define FLINE_CONNECT 8
#define FLINE_DIVISOR 10   /* Amiga divs 10(a2) for crossing height */

/* Zone data offsets (Defs.i ToZoneFloor etc.) */
#define ZONE_FLOOR_HEIGHT   2
#define ZONE_ROOF_HEIGHT    6
#define ZONE_UPPER_FLOOR    10
#define ZONE_UPPER_ROOF     14
#define ZONE_OFF_UPPER_FLOOR 10
#define ZONE_OFF_UPPER_ROOF 14
#define ZONE_EXIT_LIST      32
#define ZONE_LIST_OF_GRAPH  48

/* Door table entry (22 bytes). door_pos at closed position = door blocked. */
#define DOOR_ENTRY_SIZE     22
#define DOOR_ZONE_ID        0
#define DOOR_TYPE           2
#define DOOR_POS            4
#define DOOR_BOT            14
#define DOOR_WALL_ENTRY_SIZE 10
/* door_type: 4 = always open, 5 = never open */
#define DOOR_TYPE_ALWAYS_OPEN  4
#define DOOR_TYPE_NEVER_OPEN  5

static int resolve_zone_index_from_hint_or_room(const LevelState *level,
                                                const uint8_t *room,
                                                int16_t zone_hint)
{
    int zone_slots;
    int zi;

    if (!level) return -1;

    zone_slots = level_zone_slot_count(level);
    zi = -1;

    if (zone_hint >= 0) {
        zi = level_connect_to_zone_index(level, zone_hint);
        if (zi < 0 && zone_hint < zone_slots)
            zi = zone_hint;
    }

    if (zi < 0 && room)
        zi = level_zone_index_from_room_ptr(level, room);

    return zi;
}

/* Return true if exit line_idx from zone current_zone_id is blocked by a closed door. */
static bool is_exit_blocked_by_door(const LevelState *level, int16_t current_zone_id,
                                    int16_t line_idx)
{
    if (!level->door_data || !level->door_wall_list || !level->door_wall_list_offsets) return false;
    if (current_zone_id < 0) return false;

    const uint8_t *door = level->door_data;
    for (int door_idx = 0; door_idx < level->num_doors; door_idx++, door += DOOR_ENTRY_SIZE) {
        int16_t zone_id = (int16_t)read_be16(door + DOOR_ZONE_ID);
        if (zone_id != current_zone_id) continue;

        int16_t dtype = (int16_t)read_be16(door + DOOR_TYPE);
        if (dtype == DOOR_TYPE_ALWAYS_OPEN) continue;
        if (dtype == DOOR_TYPE_NEVER_OPEN) return true;  /* door never opens → always block */

        uint32_t start = level->door_wall_list_offsets[door_idx];
        uint32_t end   = level->door_wall_list_offsets[door_idx + 1];
        for (uint32_t j = start; j < end; j++) {
            const uint8_t *ent = level->door_wall_list + j * DOOR_WALL_ENTRY_SIZE;
            int16_t fline = (int16_t)read_be16(ent);
            if (fline != line_idx) continue;

            /* This exit is this door. Block if door is closed (pos at bot). */
            int32_t door_pos = (int32_t)read_be32(door + DOOR_POS);
            int32_t door_bot = (int32_t)read_be32(door + DOOR_BOT);
            if (door_pos >= door_bot) return true;
            break;
        }
    }
    return false;
}

/* Return 1 if node_a appears before node_b when walking from head. */
static int node_before(int head, int node_a, int node_b,
                      const int *next)
{
    int n = head;
    while (n >= 0) {
        if (n == node_a) return 1;
        if (n == node_b) return 0;
        n = next[n];
    }
    return 0;
}

/* Unlink node from list; insert it before node 'before' (so node is drawn before that one). */
static void move_before(int node, int before,
                        int *next, int *prev)
{
    int p = prev[node], n = next[node];
    if (p >= 0) next[p] = n;
    if (n >= 0) prev[n] = p;
    prev[node] = prev[before];
    next[node] = before;
    if (prev[before] >= 0)
        next[prev[before]] = node;
    prev[before] = node;
}

/* -----------------------------------------------------------------------
 * order_zones - Amiga OrderZones: traverse list from current zone, then
 * reorder by portal (exit-line) side test so back-to-front order is correct.
 *
 * 1. Traverse ListOfGraphRooms (viewer's zone list at zone_data + 48).
 * 2. Build linked list in that order.
 * 3. RunThroughList + InsertList: for each zone, look at exit lines; if
 *    connected zone is in list and further away (viewer in front of line),
 *    but currently drawn before current, move current in front of connected.
 * 4. Output final list order.
 * ----------------------------------------------------------------------- */
void order_zones(ZoneOrder *out, const LevelState *level,
                 int32_t viewer_x, int32_t viewer_z,
                 int32_t move_dx, int32_t move_dz,
                 int viewer_angle,
                 const uint8_t *list_of_graph_rooms)
{
    (void)move_dx;
    (void)move_dz;
    (void)viewer_angle;
    out->count = 0;

    {
        if (!level->data || !level->zone_adds || !level->floor_lines) {
            return;
        }

        uint8_t to_draw_tab[256];
        memset(to_draw_tab, 0, sizeof(to_draw_tab));

        /* WorkSpace[zone_id] = long at offset 4 in list entry (Amiga settodraw). */
        uint32_t workspace[256];
        memset(workspace, 0, sizeof(workspace));

        int16_t zone_list[256];
        int num_zones = 0;

        if (list_of_graph_rooms) {
            const uint8_t *lgr = list_of_graph_rooms;
            int zone_slots = level_zone_slot_count(level);
            while (num_zones < MAX_ORDER_ENTRIES) {
                int16_t entry_word = read_be16(lgr);
                int16_t zid = -1;
                if (entry_word < 0) break;
                if (resolve_lgr_entry_zone_id(level, entry_word, &zid) &&
                    zid >= 0 && zid < 256 &&
                    (zone_slots <= 0 || zid < zone_slots)) {
                    if (!to_draw_tab[zid]) {
                        to_draw_tab[zid] = 1;
                        workspace[zid] = read_be32(lgr + 4);
                        zone_list[num_zones++] = zid;
                    } else {
                        /* Multiple graph entries can alias one zone id; preserve any gate bits. */
                        workspace[zid] |= read_be32(lgr + 4);
                    }
                }
                lgr += 8;
            }
        }
        if (num_zones == 0) return;

        /* Narrow fallback path for the known problematic viewpoint:
         * when the viewer is in zone 53 and both zones 56/58 are in the visible set.
         * This keeps the strict geometric reorder from affecting other zones. */
        int apply_zone53_fallback = 0;
        if (zone_list[0] == 53 && to_draw_tab[56] && to_draw_tab[58]) {
            apply_zone53_fallback = 1;
        }

        /* Linked list by node index: next[i], prev[i], zone_id[i]. Head = 0, tail = num_zones-1. */
        int next[256], prev[256];
        int16_t node_zone[256];
        for (int i = 0; i < num_zones; i++) {
            node_zone[i] = zone_list[i];
            prev[i] = i - 1;
            next[i] = i + 1;
        }
        prev[0] = -1;
        next[num_zones - 1] = -1;
        int head = 0, tail = num_zones - 1;

        /* zone_id (0..255) -> linked-list node index; -1 if not in this frame's visible set */
        int zone_id_to_node[256];
        for (int zi = 0; zi < 256; zi++)
            zone_id_to_node[zi] = -1;
        for (int i = 0; i < num_zones; i++) {
            int16_t zid = node_zone[i];
            if (zid >= 0 && zid < 256)
                zone_id_to_node[zid] = i;
        }

        /* RunThroughList: multiple passes, each pass walk list from tail to head.
         * Early-exit: if a pass made no reorderings the list is already stable. */
        enum { k_order_passes = 100 };
        for (int pass = 0; pass < k_order_passes; pass++) {
            int moved = 0;
            int node = tail;
            while (node >= 0) {
                int16_t cur_zone = node_zone[node];
                int32_t zone_off = read_be32(level->zone_adds + (int)cur_zone * 4);
                if (zone_off != 0) {
                    const uint8_t *zone_data = level->data + zone_off;
                    int16_t exit_rel = read_be16(zone_data + ZONE_EXIT_LIST);
                    if (exit_rel != 0) {
                        const uint8_t *exit_list = zone_data + exit_rel;
                        uint32_t d6 = workspace[cur_zone];
                        for (int ei = 0; ei < 64; ei++) {
                            int16_t line_idx = read_be16(exit_list + ei * 2);
                            const uint8_t *fline;
                            int32_t lx, lz, word4, word6;
                            int connect_index;
                            if (line_idx < 0) break;
                            if (line_idx >= level->num_floor_lines) continue;

                            fline = level->floor_lines + (int)line_idx * FLINE_SIZE;
                            connect_index = level_connect_to_zone_index(level, read_be16(fline + FLINE_CONNECT));
                            if (connect_index < 0 || connect_index >= 256 || !to_draw_tab[connect_index]) continue;

                            lx = (int32_t)read_be16(fline + FLINE_X);
                            lz = (int32_t)read_be16(fline + FLINE_Z);
                            word4 = (int32_t)read_be16(fline + FLINE_WORD4);
                            word6 = (int32_t)read_be16(fline + FLINE_WORD6);

                            /* Amiga InsertList bit flow:
                             *   b   = d7 (indrawlist gate)
                             *   b+1 = evaluated marker
                             *   b+2 = cached "mustdo" marker
                             * d7 advances by 3 per exit; btst/bset are register ops (bit index mod 32).
                             * WorkSpace=0 fallback: when d6 is 0, still run side test and reorder. */
                            unsigned int b = (unsigned)(ei * 3);
                            if (d6 != 0 && !reg_btst32(d6, b)) continue;

                            /* Amiga InsertList: side = dx*word6 - dz*word4; ble PutDone => reorder when side > 0 */
                            if (!reg_btst32(d6, b + 1u)) {
                                int32_t dx, dz, side;
                                d6 |= reg_bit32(b + 1u); /* mark evaluated */
                                dx = viewer_x - lx;
                                dz = viewer_z - lz;
                                side = dx * word6 - dz * word4;
                                if (side <= 0) continue;
                                d6 |= reg_bit32(b + 2u);  /* mustdo */
                            } else {
                                if (!reg_btst32(d6, b + 2u)) continue;
                            }

                            /* mustdo: connected is further; if it's earlier in list, move current in front of it (Amiga iscloser). */
                            {
                                int conn_node = (connect_index >= 0 && connect_index < 256)
                                                    ? zone_id_to_node[connect_index]
                                                    : -1;
                                if (conn_node < 0) continue;
                                if (!node_before(head, conn_node, node, next)) continue;
                                if (node == head) head = (next[node] >= 0) ? next[node] : node;
                                if (node == tail) tail = (prev[node] >= 0) ? prev[node] : node;
                                move_before(node, conn_node, next, prev);
                                if (conn_node == head) head = node;
                                moved = 1;
                            }
                        }
                        workspace[cur_zone] = d6;  /* Amiga allinlist: move.l d6,(a6) */
                    }
                }
                node = prev[node];
            }
            if (!moved) break;
        }

        if (apply_zone53_fallback) {
            enum { k_strict_order_passes = 16 };
            for (int pass = 0; pass < k_strict_order_passes; pass++) {
                int moved = 0;
                int node = tail;
                while (node >= 0) {
                    int16_t cur_zone = node_zone[node];
                    int32_t zone_off = read_be32(level->zone_adds + (int)cur_zone * 4);
                    if (zone_off != 0) {
                        const uint8_t *zone_data = level->data + zone_off;
                        int16_t exit_rel = read_be16(zone_data + ZONE_EXIT_LIST);
                        if (exit_rel != 0) {
                            const uint8_t *exit_list = zone_data + exit_rel;
                            for (int ei = 0; ei < 128; ei++) {
                                int16_t line_idx = read_be16(exit_list + ei * 2);
                                int connect_index, conn_node;
                                const uint8_t *fline;
                                int32_t lx, lz, word4, word6, dx, dz, side;

                                if (line_idx < 0) break;
                                if (line_idx >= level->num_floor_lines) continue;

                                fline = level->floor_lines + (int)line_idx * FLINE_SIZE;
                                connect_index = level_connect_to_zone_index(level, read_be16(fline + FLINE_CONNECT));
                                if (connect_index < 0 || connect_index >= 256 || !to_draw_tab[connect_index]) continue;

                                lx = (int32_t)read_be16(fline + FLINE_X);
                                lz = (int32_t)read_be16(fline + FLINE_Z);
                                word4 = (int32_t)read_be16(fline + FLINE_WORD4);
                                word6 = (int32_t)read_be16(fline + FLINE_WORD6);
                                dx = viewer_x - lx;
                                dz = viewer_z - lz;
                                side = dx * word6 - dz * word4;
                                if (side <= 0) continue;

                                conn_node = zone_id_to_node[connect_index];
                                if (conn_node < 0) continue;
                                if (!node_before(head, conn_node, node, next)) continue;
                                if (node == head) head = (next[node] >= 0) ? next[node] : node;
                                if (node == tail) tail = (prev[node] >= 0) ? prev[node] : node;
                                move_before(node, conn_node, next, prev);
                                if (conn_node == head) head = node;
                                moved = 1;
                            }
                        }
                    }
                    node = prev[node];
                }
                if (!moved) break;
            }
        }

        /* Output final order (walk from head) */
        {
            int n = head;
            int out_i = 0;
            while (n >= 0 && out_i < MAX_ORDER_ENTRIES) {
                out->zones[out_i++] = node_zone[n];
                n = next[n];
            }
            out->count = out_i;
        }
    }
}

/* -----------------------------------------------------------------------
 * can_it_be_seen - Line-of-sight (ObjectMove.s CanItBeSeen)
 *
 * 1. Same room: visible only if viewer_top == target_top.
 * 2. Else: to_room must be in from_room list-of-graph (when graph available).
 * 3. Clip points: left/right clip test when clips/points available.
 * 4. GoThroughZones: exit cross test, crossing height (fline+10 divisor),
 *    clearance vs current room (ViewerTop), GotIn (entry_top), target_top at end.
 * ----------------------------------------------------------------------- */
uint8_t can_it_be_seen(const LevelState *level,
                       const uint8_t *from_room, const uint8_t *to_room,
                       int16_t to_zone_id,
                       int16_t viewer_x, int16_t viewer_z, int16_t viewer_y,
                       int16_t target_x, int16_t target_z, int16_t target_y,
                       int8_t viewer_top, int8_t target_top,
                       int full_height)
{
    int current_zone_idx;
    int target_zone_idx;

    if (!level->data || !level->floor_lines) {
        return 0;
    }

    if (!level->zone_adds) return 0;
    int zone_slots = level_zone_slot_count(level);
    if (zone_slots <= 0) return 0;

    current_zone_idx = resolve_zone_index_from_hint_or_room(level, from_room, -1);
    target_zone_idx = resolve_zone_index_from_hint_or_room(level, to_room, to_zone_id);

    if (!full_height) {
        if (!level_zone_has_upper_layer(level, (int16_t)current_zone_idx))
            viewer_top = 0;
        if (!level_zone_has_upper_layer(level, (int16_t)target_zone_idx))
            target_top = 0;
    }

    /* Same room (insameroom): normally visible only on same floor.
     * Full-height callers (e.g. blast/hitscan paths) intentionally
     * allow cross-section visibility in the same zone. */
    if (from_room == to_room) {
        if (!full_height &&
            !level_zone_has_upper_layer(level, (int16_t)current_zone_idx)) {
            viewer_top = 0;
            target_top = 0;
        }
        if (full_height || viewer_top == target_top)
            return 0x03u;
        return 0u;
    }

    /* to_zone_id is the target's zone id (caller passes it; do not read from to_room) */
    const uint8_t *list = from_room + ZONE_LIST_OF_GRAPH;
    bool in_list = false;
    int16_t clip_off = -1;

    /* InList: is to_room in from_room's list-of-graph?
     *
     * The level uses Amiga format: each list entry's first word is a graph index
     * (into zone_graph_adds), NOT a zone ID.  When Amiga graph data is available
     * we MUST use that lookup exclusively.  The old "direct format" shortcut
     * (word0 == to_zone_id) would fire as a spurious match because graph indices
     * (0, 1, 2 …) coincide numerically with small zone IDs, causing enemies to
     * see through walls into unrelated zones.
     *
     * Fallback to direct-format only when the graph tables are absent. */
    bool amiga_graph = (level->zone_graph_adds != NULL && level->graphics != NULL);
    {
        const uint8_t *entry = list;
        for (int i = 0; i < 256; i++) {
            int16_t word0 = read_be16(entry);
            if (word0 < 0) break;

            if (amiga_graph) {
                /* Amiga format: word0 = graph index -> offset in graphics -> zone id */
                if (level->num_zone_graph_entries > 0 && word0 >= level->num_zone_graph_entries) {
                    entry += 8;
                    continue;
                }
                uint32_t gfx_off = (uint32_t)read_be32(level->zone_graph_adds + (unsigned)word0 * 8u);
                int16_t entry_zone_id = read_be16(level->graphics + gfx_off);
                if (entry_zone_id == to_zone_id) {
                    in_list = true;
                    clip_off = read_be16(entry + 2);
                    break;
                }
            } else {
                /* Direct format fallback: first word is zone id (no graph data present) */
                if (word0 == to_zone_id) {
                    in_list = true;
                    clip_off = read_be16(entry + 2);
                    break;
                }
            }

            entry += 8;
        }
        if (!in_list) {
            if (amiga_graph)
                return 0;   /* target not reachable from this zone */
            in_list = true; /* no graph data at all: let GoThroughZones decide */
        }
    }

    int32_t dx = (int32_t)target_x - (int32_t)viewer_x;
    int32_t dz = (int32_t)target_z - (int32_t)viewer_z;

    /* Clip check (left/right) when clips/points available.
     * Amiga: d4 = (pz-vz)*dx - (px-vx)*dz; left=ble outlist, right=bge outlist. */
    if (in_list && clip_off >= 0 && level->clips && level->points) {
        const uint8_t *clip_ptr = level->clips + (unsigned)clip_off * 2u;
        for (;;) {
            int16_t pt_idx = read_be16(clip_ptr);
            if (pt_idx < 0) break;
            int16_t px = read_be16(level->points + (unsigned)pt_idx * 4u);
            int16_t pz = read_be16(level->points + (unsigned)pt_idx * 4u + 2u);
            /* Amiga: (pz-vz)*dx - (px-vx)*dz <= 0 → not visible */
            int32_t cross = ((int32_t)pz - (int32_t)viewer_z) * dx - ((int32_t)px - (int32_t)viewer_x) * dz;
            if (cross <= 0) return 0;
            clip_ptr += 2;
        }
        clip_ptr += 2;
        for (;;) {
            int16_t pt_idx = read_be16(clip_ptr);
            if (pt_idx < 0) break;
            int16_t px = read_be16(level->points + (unsigned)pt_idx * 4u);
            int16_t pz = read_be16(level->points + (unsigned)pt_idx * 4u + 2u);
            /* Amiga: (pz-vz)*dx - (px-vx)*dz >= 0 → not visible */
            int32_t cross = ((int32_t)pz - (int32_t)viewer_z) * dx - ((int32_t)px - (int32_t)viewer_x) * dz;
            if (cross >= 0) return 0;
            clip_ptr += 2;
        }
    }

    /* GoThroughZones (Amiga ObjectMove.s lines 1564-1664).
     * For each exit of current_room, test if viewer→target ray crosses the exit line,
     * then advance into the next zone. Repeat until we reach to_room or fail.
     * Exits that are closed doors block line-of-sight (enemies must not see through doors). */
    int32_t dy = (int32_t)target_y - (int32_t)viewer_y;
    const uint8_t *current_room = from_room;
    int8_t d2 = viewer_top;
    int current_zone_id = current_zone_idx;

    for (int depth = 0; depth < 20; depth++) {
        int16_t exit_rel = read_be16(current_room + ZONE_EXIT_LIST);
        if (exit_rel == 0) break;

        const uint8_t *exit_list = current_room + exit_rel;
        bool found_exit = false;

        for (int i = 0; i < 50; i++) {
            int16_t line_idx = read_be16(exit_list + i * 2);
            /* Negative sentinel → end of list, no path found → not visible */
            if (line_idx < 0) break;

            const uint8_t *fline = level->floor_lines + (unsigned)line_idx * (unsigned)FLINE_SIZE;
            int16_t lx    = read_be16(fline + FLINE_X);
            int16_t lz    = read_be16(fline + FLINE_Z);
            int16_t lxlen = read_be16(fline + FLINE_XLEN);
            int16_t lzlen = read_be16(fline + FLINE_ZLEN);
            int16_t connect = read_be16(fline + FLINE_CONNECT);

            /* Amiga viewer side test (FindWayOut):
             * d4 = (lz-vz)*dx - (lx-vx)*dz  > 0 → viewer on correct side */
            int32_t lx_off = (int32_t)lx - (int32_t)viewer_x;  /* lx - vx */
            int32_t lz_off = (int32_t)lz - (int32_t)viewer_z;  /* lz - vz */
            int32_t viewer_side = lz_off * dx - lx_off * dz;
            if (viewer_side <= 0) continue;

            /* Amiga target side test (using line END = start + (lxlen,lzlen)):
             * d6 = (lz+lzlen-vz)*dx - (lx+lxlen-vx)*dz  < 0 → target on correct side */
            int32_t le_x_off = lx_off + (int32_t)lxlen;  /* lx+lxlen - vx */
            int32_t le_z_off = lz_off + (int32_t)lzlen;  /* lz+lzlen - vz */
            int32_t target_side = le_z_off * dx - le_x_off * dz;
            if (target_side >= 0) continue;

            /* Wall (no exit) → not visible (Amiga: blt outlist) */
            if (connect < 0) return 0;
            int connect_index = level_connect_to_zone_index(level, connect);
            if (connect_index < 0 || connect_index >= zone_slots) return 0;

            /* Closed door blocks this exit. */
            if (is_exit_blocked_by_door(level, (int16_t)current_zone_id, line_idx)) return 0;

            /* Height at which ray crosses this exit line.
             * Amiga d4 = (tz-lz)*lxlen - (tx-lx)*lzlen  (target signed distance, negated)
             * Amiga d5 = (vx-lx)*lzlen - (vz-lz)*lxlen  (viewer signed distance)
             * crossing_y = viewer_y + d5*dy / (d5+d4) */
            int16_t divisor = read_be16(fline + FLINE_DIVISOR);
            if (divisor == 0) divisor = 1;
            int32_t num_t = (int32_t)(target_z - lz) * (int32_t)lxlen
                          - (int32_t)(target_x - lx) * (int32_t)lzlen; /* = Amiga d4 */
            int32_t num_v = (int32_t)(viewer_x - lx) * (int32_t)lzlen
                          - (int32_t)(viewer_z - lz) * (int32_t)lxlen; /* = Amiga d5 */
            num_t /= (int32_t)divisor;
            num_v /= (int32_t)divisor;
            int32_t den = num_v + num_t;  /* Amiga: d5 + d4 */
            int32_t cross_y_16 = (int32_t)viewer_y;
            if (den != 0) {
                cross_y_16 = (int32_t)viewer_y + (int32_t)dy * num_v / den;
            }
            int32_t cross_y = cross_y_16 << 7;

            /* Advance into next zone (Amiga: GotIn) */
            int32_t next_off = read_be32(level->zone_adds + (unsigned)connect_index * 4u);
            const uint8_t *next_zone = level->data + next_off;

            int8_t entry_top;
            if (full_height) {
                /* Hitscan: check crossing height against the FULL height
                 * envelope of the zone (both lower and upper sections)
                 * rather than only the viewer's current section.  This
                 * allows cross-section shots while still rejecting exits
                 * at heights outside the zone's physical bounds —
                 * preventing the ray from routing around closed doors
                 * through height-separated alternative exits. */
                int32_t floor_h = read_be32(current_room + ZONE_FLOOR_HEIGHT);
                int32_t roof_h  = read_be32(current_room + ZONE_ROOF_HEIGHT);
                if (level_zone_has_upper_layer(level, (int16_t)current_zone_id)) {
                    int32_t cur_up_roof = read_be32(current_room + ZONE_UPPER_ROOF);
                    if (cur_up_roof < roof_h) roof_h = cur_up_roof;
                }
                if (cross_y < roof_h || cross_y > floor_h) return 0;

                int32_t next_floor = read_be32(next_zone + ZONE_FLOOR_HEIGHT);
                int32_t next_roof  = read_be32(next_zone + ZONE_ROOF_HEIGHT);
                if (level_zone_has_upper_layer(level, (int16_t)connect_index)) {
                    int32_t next_up_roof = read_be32(next_zone + ZONE_UPPER_ROOF);
                    if (next_up_roof < next_roof) next_roof = next_up_roof;
                }
                if (cross_y < next_roof || cross_y > next_floor) return 0;

                entry_top = 0;
            } else {
                /* Current room height clearance (Amiga: comparewithbottom / top section) */
                int32_t floor_h = read_be32(current_room + ZONE_FLOOR_HEIGHT);
                int32_t roof_h  = read_be32(current_room + ZONE_ROOF_HEIGHT);
                if (d2 && level_zone_has_upper_layer(level, (int16_t)current_zone_id)) {
                    floor_h = read_be32(current_room + ZONE_UPPER_FLOOR);
                    roof_h  = read_be32(current_room + ZONE_UPPER_ROOF);
                } else {
                    d2 = 0;
                }
                if (cross_y < roof_h || cross_y > floor_h) return 0;

                int32_t next_floor = read_be32(next_zone + ZONE_FLOOR_HEIGHT);
                int32_t next_roof  = read_be32(next_zone + ZONE_ROOF_HEIGHT);
                if (cross_y > next_floor) return 0;

                /* Amiga: bgt.s GotIn → if cross_y > next_roof → enter at bottom */
                if (cross_y > next_roof) {
                    entry_top = 0;
                } else {
                    if (!level_zone_has_upper_layer(level, (int16_t)connect_index))
                        return 0;
                    entry_top = 1;
                    int32_t up_floor = read_be32(next_zone + ZONE_UPPER_FLOOR);
                    int32_t up_roof  = read_be32(next_zone + ZONE_UPPER_ROOF);
                    if (cross_y > up_floor) return 0;
                    if (cross_y < up_roof)  return 0;
                }
            }

            if (next_zone == to_room) {
                if (full_height) return 0x03u;
                return (entry_top == target_top) ? 0x03u : 0u;
            }
            current_room = next_zone;
            d2 = entry_top;
            current_zone_id = connect_index;
            found_exit = true;
            break;
        }
        if (!found_exit) break;
    }

    return 0;
}
