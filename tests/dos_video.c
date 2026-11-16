/* VBE mode priority and preservation of the original 4096 source colors. */
#include <SDL3/SDL.h>
#include "platform/dos/video_dos.h"
#include <stdio.h>
#include <stdlib.h>
#define CHECK(c) do { if(!(c)){printf("DOS video failure at line %d: %s\n",__LINE__,#c);exit(1);} } while(0)
static void check_rows(void)
{
    static const unsigned widths[]={0,1,2,3,7,127,128};
    uint16_t source[130],actual[134],expected[134];
    for(unsigned f=0;f<2;f++)for(unsigned base=0;base<65536;base+=128)
    for(unsigned offset=0;offset<2;offset++)for(unsigned w=0;w<7;w++) {
        for(unsigned i=0;i<130;i++)source[i]=(uint16_t)(base+i);
        for(unsigned i=0;i<134;i++)actual[i]=expected[i]=0xa5a5;
        for(unsigned i=0;i<widths[w];i++) {
            unsigned p=source[i+offset],r=p>>11,g=(p>>5)&63,b=p&31;
            expected[i+offset+2]=(uint16_t)(f?((r>>1)<<8)|((g>>2)<<4)|(b>>1):(r<<10)|((g>>1)<<5)|b);
        }
        dos_video_convert_row(actual+offset+2,source+offset,widths[w],f?SDL_PIXELFORMAT_XRGB4444:SDL_PIXELFORMAT_XRGB1555);
        CHECK(!memcmp(actual,expected,sizeof(actual)));
    }
    puts("DOS video: all 65536 RGB565 inputs, odd widths, unaligned rows and guards preserved");
}
#ifdef AB3D_DOS
#include "platform/dos/ui_dos.h"
static void check_tints(void)
{
    static const struct {SDL_PixelFormat format;unsigned rs,gs,rm,gm,bm,rl,gl,bl;} formats[]={
        {SDL_PIXELFORMAT_RGB565,11,5,31,63,31,3,2,3},
        {SDL_PIXELFORMAT_XRGB1555,10,5,31,31,31,3,3,3},
        {SDL_PIXELFORMAT_XRGB4444,8,4,15,15,15,4,4,4}};
    static const unsigned alphas[]={0,55,110,128,220,255};
    static const unsigned targets[][3]={{255,255,255},{255,0,0},{17,119,238}};
    unsigned checked=0;
    for(unsigned f=0;f<3;f++) {
        SDL_Surface *surface=SDL_CreateSurface(128,512,formats[f].format);CHECK(surface);
        DosUiRenderer renderer={0};renderer.surface=surface;
        SDL_Rect clip={2,1,124,510},rect={1,2,126,509};
        CHECK(SDL_SetSurfaceClipRect(surface,&clip));
        unsigned rs=formats[f].rs,gs=formats[f].gs,rm=formats[f].rm,gm=formats[f].gm,bm=formats[f].bm;
        for(unsigned t=0;t<3;t++)for(unsigned a=0;a<6;a++) {
            unsigned alpha=alphas[a],mask=(rm<<rs)|(gm<<gs)|bm;
            for(int y=0;y<512;y++)for(int x=0;x<128;x++)
                ((Uint16 *)((Uint8 *)surface->pixels+y*surface->pitch))[x]=(Uint16)((y*128+x)&mask);
            CHECK(dos_ui_tint(&renderer,&rect,targets[t][0],targets[t][1],targets[t][2],alpha));
            for(int y=0;y<512;y++)for(int x=0;x<128;x++) {
                unsigned original=(y*128+x)&mask,expected=original;
                if(x>=2 && x<126 && y>=2 && y<511) {
                    unsigned r=(((original>>rs)&rm)*(255-alpha)+(targets[t][0]>>formats[f].rl)*alpha+127)/255;
                    unsigned g=(((original>>gs)&gm)*(255-alpha)+(targets[t][1]>>formats[f].gl)*alpha+127)/255;
                    unsigned b=((original&bm)*(255-alpha)+(targets[t][2]>>formats[f].bl)*alpha+127)/255;
                    expected=(r<<rs)|(g<<gs)|b;
                }
                CHECK(((Uint16 *)((Uint8 *)surface->pixels+y*surface->pitch))[x]==expected);checked++;
            }
        }
        CHECK(dos_ui_color(&renderer,17,119,238,255)==0);
        CHECK(renderer.color==dos_video_color(formats[f].format,17,119,238));
        SDL_DestroySurface(surface);
    }
    printf("DOS video: %u clipped tint pixels match independent scalar equations\n",checked);
}
#endif

int main(void)
{
    check_rows();
    SDL_DisplayMode modes[7]={0};
    SDL_PixelFormat formats[]={SDL_PIXELFORMAT_INDEX8,SDL_PIXELFORMAT_XRGB4444,
        SDL_PIXELFORMAT_XRGB1555,SDL_PIXELFORMAT_RGB565,SDL_PIXELFORMAT_RGB565,
        SDL_PIXELFORMAT_RGB24,SDL_PIXELFORMAT_RGB565};
    SDL_DisplayMode *pointers[7];
    for(int i=0;i<7;i++) {
        modes[i].format=formats[i];modes[i].w=320;modes[i].h=200;pointers[i]=modes+i;
    }
    modes[3].w=640;modes[3].h=480;modes[6].w=160;modes[6].h=100;
    CHECK(dos_video_choose_mode(pointers,7,320,200,0)==modes+4);
    CHECK(dos_video_choose_mode(pointers,7,320,240,0)==modes+3);
    CHECK(dos_video_choose_mode(pointers,3,320,200,0)==modes+2);
    CHECK(dos_video_choose_mode(pointers,2,320,200,0)==modes+1);
    CHECK(!dos_video_choose_mode(pointers,1,320,200,0));
    CHECK(dos_video_choose_mode(pointers,7,320,200,1)==modes);
    CHECK(!dos_video_choose_mode(pointers,7,320,240,1));
    CHECK(!dos_video_choose_mode(NULL,7,320,200,0));
    CHECK(!dos_video_choose_mode(pointers,0,320,200,0));
    for(int f=1;f<4;f++) {
        unsigned char used[65536]={0};
        for(unsigned c=0;c<4096;c++) {
            unsigned red=((c>>8)&15)*17,green=((c>>4)&15)*17,blue=(c&15)*17;
            unsigned p=dos_video_color(formats[f],red,green,blue);
            CHECK(!used[p]);used[p]=1;
            unsigned rshift=f==1?8:f==2?10:11,gshift=f==1?4:5;
            unsigned rbits=f==1?4:5,gbits=f==1?4:f==2?5:6,bbits=rbits;
            CHECK(((p>>rshift)&((1u<<rbits)-1))==(red>>(8-rbits)));
            CHECK(((p>>gshift)&((1u<<gbits)-1))==(green>>(8-gbits)));
            CHECK((p&((1u<<bbits)-1))==(blue>>(8-bbits)));
        }
    }
    puts("DOS video: ordered mode selection and all 4096 colors preserved in RGB565/555/444");
#ifdef AB3D_DOS
    check_tints();
#endif
    return 0;
}
