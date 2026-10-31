#ifndef AB3D_RENDERER_LIMITS_H
#define AB3D_RENDERER_LIMITS_H

/* Current port's bounded per-zone, per-storey graphics traversal. All draw
 * passes and editor playable-output validation share this declaration. This
 * is a port safety bound, not a claimed original Amiga format limit. */
/* A JSON numeric literal is also a C integer constant. Import the same file
 * in the editor and its Node/browser tests, without build-time copies. */
enum {
    AB3D_ZONE_GRAPHICS_RECORD_LIMIT =
#include "renderer_graphics_limit.json"
};

#endif
