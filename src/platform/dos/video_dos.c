#include <SDL3/SDL.h>
#include "video_dos.h"
#include <string.h>

/* Convert two RGB565 pixels at a time. Constant-size memcpy keeps unaligned
 * row starts valid C and compiles to ordinary 486 word/dword loads/stores. */
void dos_video_convert_row(void *destination,const void *source,unsigned pixels,SDL_PixelFormat format)
{
    unsigned char *out=destination;
    const unsigned char *in=source;
    if(format==SDL_PIXELFORMAT_XRGB1555) {
        for(;pixels>=2;pixels-=2,in+=4,out+=4) {
            uint32_t pair;memcpy(&pair,in,4);
            pair=((pair&UINT32_C(0xffc0ffc0))>>1)|(pair&UINT32_C(0x001f001f));
            memcpy(out,&pair,4);
        }
        if(pixels){uint16_t p;memcpy(&p,in,2);p=(uint16_t)(((p&0xffc0)>>1)|(p&31));memcpy(out,&p,2);}
    } else {
        for(;pixels>=2;pixels-=2,in+=4,out+=4) {
            uint32_t pair;memcpy(&pair,in,4);
            pair=((pair>>4)&UINT32_C(0x0f000f00))|((pair>>3)&UINT32_C(0x00f000f0))|((pair>>1)&UINT32_C(0x000f000f));
            memcpy(out,&pair,4);
        }
        if(pixels){uint16_t p;memcpy(&p,in,2);p=(uint16_t)(((p>>4)&0xf00)|((p>>3)&0xf0)|((p>>1)&15));memcpy(out,&p,2);}
    }
}
