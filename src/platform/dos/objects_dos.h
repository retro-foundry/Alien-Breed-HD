#ifndef AB3D_OBJECTS_DOS_H
#define AB3D_OBJECTS_DOS_H

#include "renderer.h"

/* ObjDraw candidates are rebuilt after RotateObjectPts and remain immutable
 * until the world pass completes. Room queries retain original slot order. */
void dos_objects_prepare(const LevelState *level, const ObjRotatedPoint *points, int32_t near_z);
unsigned dos_objects_for_room(const LevelState *level, int zone,
                               const int16_t *adjacent, unsigned adjacent_count, uint8_t out[256]);
const uint8_t *dos_objects_shots(unsigned pool, unsigned *count);

#endif
