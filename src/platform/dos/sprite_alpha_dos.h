#ifndef AB3D_SPRITE_ALPHA_DOS_H
#define AB3D_SPRITE_ALPHA_DOS_H

#include <stdint.h>
#include "raster_dos.h"

enum { DOS_SPRITE_ALPHA_ROWS=64 };
/* Conservative source-index occupancy; palette entries may be opaque black.
 * Original masks remain valid after io_apply_sprite_asset_fixes clears texels. */
/* Bind loaded source banks once; reset before their buffers are freed. */
void dos_sprite_alpha_reset(void);
void dos_sprite_alpha_bind(unsigned type,const uint8_t *wad,size_t wad_bytes,
                            const uint8_t *ptr,size_t ptr_bytes);
const uint32_t *dos_sprite_alpha_source(const DosSpriteRows *sprite);

#endif
