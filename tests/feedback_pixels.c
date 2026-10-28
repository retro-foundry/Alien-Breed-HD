#include "renderer_pixel.h"
#include <assert.h>
#include <stdio.h>
static unsigned char seen[65536];
int main(void)
{
    for(unsigned c=0;c<4096;c++) {
        uint16_t pixel=renderer_pixel_from_c12((uint16_t)c);
        assert(renderer_pixel_to_c12(pixel)==c);
        assert(!seen[pixel]);seen[pixel]=1;
        assert(renderer_pixel_gb_index(pixel)==(c&255));
        assert(renderer_pixel_to_c12(renderer_pixel_without_red(pixel))==(c&255));
        unsigned char bytes[2]={(unsigned char)(c>>8),(unsigned char)c};
        assert(renderer_pixel_from_be(bytes)==pixel);
    }
#ifdef AB3D_NATIVE_RGB565
    assert(!seen[1]); /* SDL weapon color key is outside the original colors. */
#endif
    puts("PASS: all 4096 colors are distinct and exact; water indices and underwater masks match source");
    return 0;
}
