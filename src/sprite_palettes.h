/*
 * Alien Breed 3D I - PC Port
 * sprite_palettes.h - Embedded sprite palette tables (from data/pal, .pal files)
 *
 * Data is generated at configure time by tools/pal_to_header.py into
 * sprite_palettes_data.h. DOS RGB565 embeds the same tables with native color
 * words in sprite_palettes_rgb565_data.h. No runtime .pal loading.
 */
#ifndef SPRITE_PALETTES_H
#define SPRITE_PALETTES_H

#include "renderer.h"
#include <stddef.h>

extern const uint8_t *sprite_pal_embedded[MAX_SPRITE_TYPES];
extern const size_t sprite_pal_embedded_size[MAX_SPRITE_TYPES];

#endif
