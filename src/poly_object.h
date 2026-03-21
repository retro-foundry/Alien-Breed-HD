/*
 * Alien Breed 3D I - PC Port
 * poly_object.h - Minimal poly (vector) object format for 3D sprites
 *
 * Matches the Amiga .vec layout used by PolygonObj in ObjDraw3.ChipRam.s.
 * See docs/vec_format.md for the full .vec binary specification.
 */

#ifndef POLY_OBJECT_H
#define POLY_OBJECT_H

#include <stdint.h>

/* -----------------------------------------------------------------------
 * Limits (from Amiga: boxrot/boxonscr 250 points, PartBuffer 32 parts)
 * ----------------------------------------------------------------------- */
#define MAX_POLY_POINTS  250
#define MAX_POLY_FRAMES  32
#define MAX_POLY_PARTS   32
#define MAX_POLY_VERTS   4   /* triangle or quad per polygon */

/* Number of vector objects in POLYOBJECTS table (robot, medipac, ... gaspipe) */
#define NUM_POLY_OBJECTS 10
/* POLYOBJECTS index for exitsign.vec (ObjDraw3 / renderer_3dobj.c) — hung from ceiling */
#define POLY_SLOT_EXIT_SIGN 2

/* -----------------------------------------------------------------------
 * In-memory point: object-space or screen-space
 * ----------------------------------------------------------------------- */
typedef struct poly_point_3 {
	int16_t x, y, z;
} poly_point_3_t;

typedef struct poly_point_2 {
	int16_t x, y;
} poly_point_2_t;

/* -----------------------------------------------------------------------
 * Single polygon (triangle or quad) for one part
 * lines_to_draw = 3 or 4; vertex_indices index into shared point array
 * ----------------------------------------------------------------------- */
typedef struct poly_polygon {
	uint16_t lines_to_draw;   /* 3 = tri, 4 = quad; 0xFFFF = end of list */
	uint16_t preholes;
	uint16_t vertex_indices[MAX_POLY_VERTS];
	uint16_t texture_offs;
	uint16_t divisor;
} poly_polygon_t;

/* Record size in bytes: 18 + lines_to_draw*4 (matches doapoly advance) */
#define POLY_POLYGON_RECORD_SIZE(lines)  (18u + (uint32_t)(lines) * 4u)

/* -----------------------------------------------------------------------
 * One part: sort point index + pointer to polygon list (in raw .vec or
 * converted blob, this is an offset; when loaded, can be pointer or offset).
 * ----------------------------------------------------------------------- */
typedef struct poly_part {
	uint32_t polygon_data_offset;  /* byte offset from start of object, or pointer when resolved */
	uint16_t sort_point_index;     /* point index for depth-sort */
} poly_part_t;

/* -----------------------------------------------------------------------
 * Minimal poly object: header + frame offsets + part list + frame data.
 * Can represent raw .vec in memory or a converted/flattened form.
 * ----------------------------------------------------------------------- */
typedef struct poly_object {
	uint16_t num_points;
	uint16_t num_frames;
	/* Frame offsets: num_frames words, then part list (part_offset, sort_point) terminated by -1.
	 * Then frame data: for each frame, num_points * (x,y,z) int16 = 6 bytes per point. */
	const uint8_t *data;   /* points to raw .vec after header, or converted blob */
} poly_object_t;

/* -----------------------------------------------------------------------
 * Optional: C-friendly “flattened” format for one loaded object
 * (points per frame, list of parts with polygon count and polygon array).
 * Used by converter output or runtime loader.
 * ----------------------------------------------------------------------- */
typedef struct poly_object_flat {
	uint16_t num_points;
	uint16_t num_frames;
	const poly_point_3_t *frame_points;   /* num_frames * num_points */
	uint16_t num_parts;
	const poly_part_t *parts;              /* sort_point only; polygon_data = offset or ptr */
	const poly_polygon_t *polygons;        /* packed per-part polygons; part ends with lines_to_draw == 0xFFFF */
} poly_object_flat_t;

#endif /* POLY_OBJECT_H */
