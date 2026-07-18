/*
 * Alien Breed 3D I - PC Port
 * renderer_3dobj.h - 3D vector/polygon object rendering
 *
 * Translated from: ObjDraw3.ChipRam.s PolygonObj / doapoly
 *
 * The Amiga original draws two kinds of objects:
 *   - 2D billboard sprites (BitMapObj): obj[6] != 0xFF
 *   - 3D vector/polygon objects (PolygonObj): obj[6] == 0xFF
 *
 * 3D objects are small polygon meshes stored in .vec binary files.
 * The POLYOBJECTS table (10 entries) maps vect_number -> vec data.
 *
 * .vec file layout (big-endian words):
 *   [0]       word: num_points
 *   [2]       word: num_frames
 *   [4..]     word[num_frames]: frame_offsets (each = byte offset from file
 *             start to that frame's point data)
 *   [4+num_frames*2..] : part list, entries = { word poly_off, word sort_off }
 *             where poly_off = byte offset to polygon list from file start,
 *             sort_off = byte offset into boxrot array for depth sort point.
 *             Terminated by a word with the sign bit set (typically 0xFFFF).
 *   [at poly_off]: polygon list (terminated by a word with high bit set).
 *     Each polygon (ObjDraw3 doapoly):
 *       [0..1]  word: lines_to_draw (d0)
 *       [2..3]  word: preholes
 *       [4..]   (d0+2) vertex records, 4 bytes each:
 *                [0..1] word point_index
 *                [2]    byte u
 *                [3]    byte v
 *              (first d0+1 point indices are polygon perimeter order;
 *               final record repeats the first point for edge stepping)
 *       [12+d0*4 .. 13+d0*4] word texture_map_index
 *       [14+d0*4 .. 15+d0*4] word brightness divisor
 *       [16+d0*4 .. 17+d0*4] word pregour flag
 *       Total polygon bytes advanced by polyloop: 18 + d0*4
 *   [at each frame_offset]: num_points * (word x, word y, word z) = 6 bytes/point
 */

#ifndef RENDERER_3DOBJ_H
#define RENDERER_3DOBJ_H

#include <stdint.h>
#include <stddef.h>
#include "game_state.h"
#include "renderer.h"

/* Number of vector object slots matching POLYOBJECTS in ObjDraw3.ChipRam.s */
#define POLY_OBJECTS_COUNT  10

/* Safety caps */
#define MAX_POLY_POINTS     128  /* Robot.vec=67, MediPac.vec=85; keep full meshes */
#define MAX_POLY_PARTS      32
#define MAX_POLY_FRAMES     32

/*
 * Parsed vector object.
 * The raw data pointer is owned by io.c (never freed here).
 */
typedef struct {
    const uint8_t *data;       /* raw .vec bytes */
    size_t         size;       /* file size */
    int            num_points; /* vertices per frame */
    int            num_frames;
    uint16_t  frame_off[MAX_POLY_FRAMES]; /* byte offsets to frame point data */
    int       num_parts;
    uint16_t  part_poly_off[MAX_POLY_PARTS]; /* byte offset to polygon list */
    uint16_t  part_sort_off[MAX_POLY_PARTS]; /* byte offset into boxrot for sort key */
} VecObject;

/* Global POLYOBJECTS table (indices 0-9) */
extern VecObject g_poly_objects[POLY_OBJECTS_COUNT];

/*
 * Parse a .vec binary file into slot (0..POLY_OBJECTS_COUNT-1).
 * data must remain valid for the lifetime of the VecObject.
 * Returns 1 on success, 0 on failure.
 */
int poly_obj_load(int slot, const uint8_t *data, size_t size);

/* Set shared texture assets used by 3D object polygons.
 * texture_maps points to TextureMaps (expected 65536 bytes).
 * texture_pal points to OldTexturePalScaled (expected 15*256*2 bytes). */
void poly_obj_set_texture_assets(const uint8_t *texture_maps, size_t texture_maps_size,
                                 const uint8_t *texture_pal, size_t texture_pal_size);

/* Enable/disable reading objVectFrameNumber for 3D polygon objects.
 * enabled = 0: strict Amiga PolygonObj behaviour (frame 0 only, default).
 * enabled != 0: use object frame number (allows animated 3D objects). */
void poly_obj_set_use_object_frame(int enabled);

/*
 * Draw a 3D vector/polygon object at its view-space position.
 *   obj    - raw 64-byte object data (Amiga big-endian)
 *   orp    - view-space centre of object (from RotateObjectPts)
 *   state  - game state (for player angle)
 *
 * Vertical placement matches ObjDraw3.ChipRam.s PolygonObj / convtoscr:
 *   d2 = ext.l 2(a0) ; asl.l #7,d2 ; sub.l yoff,d2  (a0 advanced +2 after first word)
 *   i.e. y_adjust = ((int32_t)(int16_t)obj[4] << 7) - yoff — no TOPOFROOM/BOTOFROOM path.
 */
void draw_3d_vector_object(const uint8_t *obj, const ObjRotatedPoint *orp, GameState *state,
                           int clip_left, int clip_right, int clip_top, int clip_bot);

#endif /* RENDERER_3DOBJ_H */
