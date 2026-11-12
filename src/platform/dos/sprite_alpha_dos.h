#ifndef AB3D_SPRITE_ALPHA_DOS_H
#define AB3D_SPRITE_ALPHA_DOS_H

#include <stdint.h>

enum { DOS_SPRITE_ALPHA_ROWS=64 };
/* Conservative source-index occupancy; palette entries may be opaque black.
 * Original masks remain valid after io_apply_sprite_asset_fixes clears texels. */
const uint32_t *dos_sprite_alpha_rows(unsigned type,uint32_t ptr_offset,
                                      unsigned down_strip,unsigned columns);

#endif
