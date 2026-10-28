#ifndef AB3D_RENDERER_PIXEL_H
#define AB3D_RENDERER_PIXEL_H
#include <stdint.h>
/* Source palettes use Amiga 0RGB. DOS caches use the display's native RGB565
 * words, so a completed framebuffer needs no color conversion or transpose. */
static inline uint16_t renderer_pixel_from_c12(uint16_t c)
{
#ifdef AB3D_NATIVE_RGB565
    unsigned r=(c>>8)&15,g=(c>>4)&15,b=c&15;
    return (uint16_t)((((r<<1)|(r>>3))<<11)|(((g<<2)|(g>>2))<<5)|((b<<1)|(b>>3)));
#else
    return c;
#endif
}
static inline uint16_t renderer_pixel_to_c12(uint16_t p)
{
#ifdef AB3D_NATIVE_RGB565
    return (uint16_t)((((p>>12)&15)<<8)|(((p>>7)&15)<<4)|((p>>1)&15));
#else
    return p;
#endif
}
static inline uint16_t renderer_pixel_from_be(const uint8_t *p)
{ return renderer_pixel_from_c12((uint16_t)((p[0]<<8)|p[1])); }
static inline unsigned renderer_pixel_gb_index(uint16_t p)
{
#ifdef AB3D_NATIVE_RGB565
    return (((p>>7)&15)<<4)|((p>>1)&15);
#else
    return p&255;
#endif
}
static inline uint16_t renderer_pixel_without_red(uint16_t p)
{
#ifdef AB3D_NATIVE_RGB565
    return p&0x07ff;
#else
    return p&0x00ff;
#endif
}
#endif
