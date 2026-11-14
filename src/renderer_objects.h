#ifndef AB3D_RENDERER_OBJECTS_H
#define AB3D_RENDERER_OBJECTS_H

#include "renderer.h"

enum {
    RENDERER_OBJECT_SHOTS = 256,
    RENDERER_OBJECT_EXPLOSIONS = 256 + NASTY_SHOT_SLOT_COUNT + PLAYER_SHOT_SLOT_COUNT,
    RENDERER_OBJECT_SLOTS = RENDERER_OBJECT_EXPLOSIONS + MAX_EXPLOSIONS
};

/* ObjDraw candidates are rebuilt after RotateObjectPts and remain immutable
 * until the world pass completes. Room queries retain original slot order. */
void renderer_objects_prepare(const LevelState *level, const ObjRotatedPoint *points, int32_t near_z);
unsigned renderer_objects_for_room(const LevelState *level, int zone,
                               const int16_t *adjacent, unsigned adjacent_count, uint8_t out[256]);
const uint8_t *renderer_objects_shots(unsigned pool, unsigned *count);
/* The output holds nasty shots followed by player shots, each in slot order. */
void renderer_objects_shots_for_room(const LevelState *level, int zone,
                                 const int16_t *adjacent, unsigned adjacent_count,
                                 unsigned counts[2], uint8_t out[NASTY_SHOT_SLOT_COUNT + PLAYER_SHOT_SLOT_COUNT]);

/* Add the rooms reached by a billboard footprint before starting any workers.
 * Object/shot queries retain their original slot ordering. */
void renderer_objects_add_room(const LevelState *level, int zone, unsigned slot);
int renderer_objects_in_room(const LevelState *level, int zone, unsigned slot);
int renderer_objects_room_is_exact(const LevelState *level, int zone);

#endif
