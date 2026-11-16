#ifndef AB3D_VIDEO_DOS_H
#define AB3D_VIDEO_DOS_H
#include <stdint.h>

/* VBE direct-color masks describe the hardware layout. The engine retains
 * RGB565; SDL converts only when the selected two-byte display differs. */
enum { DOS_VIDEO_UNAVAILABLE = 86 };
extern int dos_force_vga;
void dos_video_convert_row(void *destination,const void *source,unsigned pixels,SDL_PixelFormat format);

static inline int dos_video_color_rank(SDL_PixelFormat format)
{
    switch(format) {
    case SDL_PIXELFORMAT_RGB565:return 0;
    case SDL_PIXELFORMAT_XRGB1555:return 1;
    case SDL_PIXELFORMAT_XRGB4444:return 2;
    default:return -1;
    }
}

static inline const SDL_DisplayMode *dos_video_choose_mode(SDL_DisplayMode *const *modes,
    int count,int width,int height,int indexed)
{
    const SDL_DisplayMode *best=NULL;
    int best_rank=4;
    for(int i=0;modes && i<count;i++) {
        const SDL_DisplayMode *mode=modes[i];
        if(!mode || mode->w<width || mode->h<height)continue;
        int rank=indexed?(mode->format==SDL_PIXELFORMAT_INDEX8?0:-1):dos_video_color_rank(mode->format);
        if(rank<0)continue;
        if(!best || rank<best_rank || (rank==best_rank &&
           (uint64_t)mode->w*mode->h<(uint64_t)best->w*best->h)) {
            best=mode;best_rank=rank;
        }
    }
    return best;
}

static inline uint16_t dos_video_color(SDL_PixelFormat format,unsigned red,unsigned green,unsigned blue)
{
    if(format==SDL_PIXELFORMAT_XRGB1555)
        return (uint16_t)(((red>>3)<<10)|((green>>3)<<5)|(blue>>3));
    if(format==SDL_PIXELFORMAT_XRGB4444)
        return (uint16_t)(((red>>4)<<8)|((green>>4)<<4)|(blue>>4));
    return (uint16_t)(((red>>3)<<11)|((green>>2)<<5)|(blue>>3));
}
#endif
