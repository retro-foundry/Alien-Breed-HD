#ifndef AB3D_RASTER_DOS_H
#define AB3D_RASTER_DOS_H

#include "renderer_pixel.h"

uint8_t *dos_load_wall(uint8_t *data, size_t bytes, size_t *source_bytes);
enum { DOS_WALL_PALETTE_BYTES=2048, DOS_WALL_HEADER_BYTES=2052 };

/* Three planes retain the original packed-word ordering. Level-authored
 * VALSHIFT still selects the word offset; each sampled column is contiguous. */
static inline const uint8_t *dos_wall_column(const uint8_t *texture, unsigned column, unsigned rows)
{
    uint32_t words=*(const uint32_t *)(const void *)(texture-sizeof(uint32_t));
    return texture+(column/3)*rows+(column%3)*words;
}
void dos_raster_wall(RendererPixel *out, unsigned count, unsigned stride,
                      const uint8_t *column, const RendererPixel *palette,
                      uint32_t y, uint32_t dy, unsigned mask);

/* Draw the core and its one/two side extensions together: copies is 2 or 3. */
void dos_raster_wall_edges(RendererPixel *out, unsigned count, unsigned stride,
                            const uint8_t *column, const RendererPixel *palette,
                            uint32_t y, uint32_t dy, unsigned mask, unsigned copies);

typedef struct {
    const uint8_t *texels;
    const RendererPixel *palette;
} DosWallColumn;
void dos_raster_floor_edge(int16_t *left, int16_t *right, int16_t *left_bright, int16_t *right_bright,
                            unsigned count, uint32_t x, uint32_t dx, uint32_t bright, uint32_t dbright);
void dos_raster_floor_edges_init(int16_t *left, int16_t *right,
                                  unsigned count, int16_t empty_left);
/* Native callers can supply one palette and leave column palette fields unused.
 * NULL retains automatic classification for spans with per-column brightness. */
void dos_raster_wall_rows(RendererPixel *out, unsigned width, unsigned height, unsigned pitch,
                           const DosWallColumn *columns, const RendererPixel *shared_palette,
                           uint32_t y, uint32_t dy, unsigned mask);

/* AB3DI.s pastfloorbright sampling: V*1024 + U*4, wrapping at 64. */
#ifdef AB3D_NATIVE_RGB565
typedef struct {
    const uint8_t *wad, *ptr;
    size_t wad_bytes, ptr_bytes;
    const RendererPixel *palette;
    const int *rows;
    uint32_t ptr_offset, u, du;
    unsigned columns, down_strip;
} DosSpriteRows;
typedef struct { unsigned visible, drawn; } DosSpriteCoverage;
/* Native, unclipped rectangle after the renderer applies its screen/plane
 * bounds. Transparent texels leave the destination untouched, including
 * opaque palette entries whose RGB565 word happens to be zero. */
DosSpriteCoverage dos_raster_sprite_rows(RendererPixel *out, unsigned width,
                                        unsigned height, unsigned pitch,
                                        const DosSpriteRows *sprite);
void dos_raster_prepare_floor(unsigned slot,const uint8_t *texture,const uint8_t *palettes);
#endif
void dos_raster_floor(RendererPixel *out, unsigned count, const uint8_t *texture,
                       const RendererPixel *palette, uint32_t u, uint32_t v,
                       uint32_t du, uint32_t dv);
#ifdef AB3D_NATIVE_RGB565
void dos_raster_floor_dark(RendererPixel *out,unsigned count,const uint8_t *texture,
                            const RendererPixel *palette,uint32_t u,uint32_t v,uint32_t du,uint32_t dv);
#endif
static inline void dos_raster_floor_shaded(RendererPixel *out,unsigned count,const uint8_t *texture,
                            const RendererPixel (*palettes)[256],unsigned shade,
                            uint32_t u,uint32_t v,uint32_t du,uint32_t dv)
{
#ifdef AB3D_NATIVE_RGB565
    if(shade==14){dos_raster_floor_dark(out,count,texture,palettes[shade],u,v,du,dv);return;}
#endif
    dos_raster_floor(out,count,texture,palettes[shade],u,v,du,dv);
}

void dos_raster_floor_gouraud(RendererPixel *out, unsigned count, const uint8_t *texture,
                               const RendererPixel (*palettes)[256],
                               uint32_t u, uint32_t v, uint32_t du, uint32_t dv,
                               int32_t level, int32_t dlevel);

#endif
