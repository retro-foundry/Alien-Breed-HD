/* Synchronous DOS UI primitives: SDL fills/blits on the native VESA surface. */
#include <SDL3/SDL.h>
#include "ui_dos.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef AB3D_DOS_INDEXED
#include "palette_dos.h"
#endif

#ifdef AB3D_NATIVE_RGB565
extern bool Ab3dDos_BeginCanvas(SDL_Window *window,SDL_Surface *source,void **backup);
extern void Ab3dDos_EndCanvas(SDL_Window *window);
extern bool Ab3dDos_MaterializeCanvas(SDL_Window *window);
extern void Ab3dDos_DiscardCanvasReadback(void);

void dos_ui_materialize_canvas(DosUiRenderer *r)
{
    if(!r || !r->canvas_surface_stale || r->window_pixels)return;
    if(!Ab3dDos_MaterializeCanvas(r->window)) {
        fprintf(stderr,"DOS canvas readback failed: %s\n",SDL_GetError());exit(1);
    }
    r->canvas_surface_stale=false;
}

void dos_ui_discard_canvas_readback(DosUiRenderer *r)
{
    if(!r || !r->canvas_surface_stale)return;
    r->canvas_surface_stale=false;
    Ab3dDos_DiscardCanvasReadback();
}

/* The two native allocations have the same word/dword alignment and pitch.
 * Select the leading/trailing word once, then retain both row pointers and
 * stride in registers throughout the rectangle. */
__attribute__((always_inline)) static inline
void copy_native_rectangle(void *destination,const void *source,unsigned pixels,
                            unsigned rows,unsigned pitch)
{
    if(!pixels || !rows)return;
    unsigned head=((uintptr_t)source&2)!=0;
    unsigned pairs=(pixels-head)/2,tail=(pixels-head)&1,skip=pitch-pixels*2;
#define COPY_RECTANGLE(HEAD,TAIL) \
    __asm__ volatile("1:\n\t" HEAD \
                     "movl %%ebx, %%ecx\n\t" \
                     "rep movsl\n\t" TAIL \
                     "addl %[skip], %%esi\n\t" \
                     "addl %[skip], %%edi\n\t" \
                     "decl %%edx\n\t" \
                     "jnz 1b" \
                     : "+&D"(destination), "+&S"(source), "+&d"(rows) \
                     : "b"(pairs), [skip] "r"(skip) : "ecx", "cc", "memory")
    if(head) {
        if(tail)COPY_RECTANGLE("movsw\n\t","movsw\n\t");
        else COPY_RECTANGLE("movsw\n\t","");
    } else {
        if(tail)COPY_RECTANGLE("","movsw\n\t");
        else COPY_RECTANGLE("","");
    }
#undef COPY_RECTANGLE
}

void dos_ui_check_canvas_copy(void)
{
    _Alignas(16) Uint16 source[8*64],actual[8*64],expected[8*64];
    static const unsigned widths[]={0,1,2,3,4,7,8,9,10,15,16,31};
    static const unsigned heights[]={0,1,2,4};
    for(unsigned i=0;i<8*64;i++)source[i]=(Uint16)(i*131+17);
    unsigned checked=0;
    for(unsigned x=0;x<4;x++)for(unsigned w=0;w<12;w++)for(unsigned h=0;h<4;h++) {
        memset(actual,0xa5,sizeof(actual));memcpy(expected,actual,sizeof(actual));
        unsigned start=64+x;
        for(unsigned row=0;row<heights[h];row++)
            memcpy(expected+start+row*64,source+start+row*64,widths[w]*2);
        copy_native_rectangle(actual+start,source+start,widths[w],heights[h],128);
        if(memcmp(actual,expected,sizeof(actual))) {
            fprintf(stderr,"DOS strided copy differs at x=%u width=%u height=%u\n",x,widths[w],heights[h]);exit(1);
        }
        checked++;
    }
    printf("[DOSCHECK] %u strided rectangle copies retain word alignment, odd sizes and row guards\n",checked);
}

static void copy_canvas_rects(DosUiRenderer *r,bool restore)
{
    SDL_Surface *s=r->surface;
    Uint8 *destination=restore?s->pixels:r->window_pixels;
    const Uint8 *source=restore?r->window_pixels:s->pixels;
    for(int n=0;n<r->saved_count;n++) {
        SDL_Rect a=r->saved_rects[n];
        int offset=a.y*s->pitch+a.x*2;
        copy_native_rectangle(destination+offset,source+offset,a.w,a.h,s->pitch);
    }
}

bool dos_ui_prepare_canvas(DosUiRenderer *r,const SDL_Rect *rects,int count)
{
    if(!r || r->window_pixels || !rects || count<1 || count>3)return false;
    r->saved_count=0;
    SDL_Rect bounds={0,0,r->surface->w,r->surface->h};
    for(int n=0;n<count;n++)
        if(!SDL_GetRectIntersection(&rects[n],&bounds,&r->saved_rects[n]))return false;
    r->saved_count=count;
    return true;
}

bool dos_ui_begin_canvas(DosUiRenderer *r,SDL_Surface *source)
{
    SDL_Rect clip;
    if(!r || r->window_pixels || !source || r->saved_count<1 || r->saved_count>3 ||
       r->viewport.x || r->viewport.y || r->viewport.w!=source->w || r->viewport.h!=source->h ||
       !SDL_GetSurfaceClipRect(r->surface,&clip) || clip.x || clip.y ||
       clip.w!=source->w || clip.h!=source->h)return false;
    if(!Ab3dDos_BeginCanvas(r->window,source,&r->window_pixels))return false;
    copy_canvas_rects(r,false);
    return true;
}

void dos_ui_restore_canvas(DosUiRenderer *r)
{
    if(!r || !r->window_pixels)return;
    copy_canvas_rects(r,true);
    Ab3dDos_EndCanvas(r->window);
    r->window_pixels=NULL;
    r->canvas_surface_stale=true;
}
#endif

DosUiRenderer *dos_ui_create_renderer(SDL_Window *window)
{
    SDL_Surface *s=SDL_GetWindowSurface(window);
    if(!s || s->format!=
#ifdef AB3D_DOS_INDEXED
       SDL_PIXELFORMAT_INDEX8
#else
       SDL_PIXELFORMAT_RGB565
#endif
       ){SDL_SetError("DOS UI requires native pixel format");return NULL;}
    DosUiRenderer *r=calloc(1,sizeof(*r));if(!r){SDL_OutOfMemory();return NULL;}
    r->window=window;r->surface=s;r->viewport=(SDL_Rect){0,0,s->w,s->h};return r;
}
void dos_ui_destroy_renderer(DosUiRenderer *r){
#ifdef AB3D_NATIVE_RGB565
    dos_ui_restore_canvas(r);
#endif
    free(r);
}
static bool apply_clip(DosUiRenderer *r)
{
    SDL_Rect a=r->viewport;
    if(r->clipped){SDL_Rect b=r->clip;b.x+=a.x;b.y+=a.y;
        if(!SDL_GetRectIntersection(&a,&b,&a))a=(SDL_Rect){0,0,0,0};}
    SDL_SetSurfaceClipRect(r->surface,&a);return true;
}
bool dos_ui_viewport(DosUiRenderer *r,const SDL_Rect *rect)
{if(!r)return SDL_SetError("Missing DOS renderer");r->viewport=rect?*rect:(SDL_Rect){0,0,r->surface->w,r->surface->h};return apply_clip(r);}
bool dos_ui_clip(DosUiRenderer *r,const SDL_Rect *rect)
{if(!r)return SDL_SetError("Missing DOS renderer");r->clipped=rect!=NULL;if(rect)r->clip=*rect;return apply_clip(r);}
int dos_ui_color(DosUiRenderer *r,Uint8 red,Uint8 green,Uint8 blue,Uint8 alpha)
{if(!r || alpha!=255){SDL_SetError("DOS UI colors must be opaque");return -1;}
#ifdef AB3D_DOS_INDEXED
 r->color=dos_c12_to_index[((red>>4)<<8)|((green>>4)<<4)|(blue>>4)];
#else
 r->color=(Uint16)(((red>>3)<<11)|((green>>2)<<5)|(blue>>3));
#endif
 return 0;}
int dos_ui_fill(DosUiRenderer *r,const SDL_Rect *rect)
{if(!r)return -1;
#ifdef AB3D_NATIVE_RGB565
 dos_ui_materialize_canvas(r);
#endif
 SDL_Rect a=rect?*rect:(SDL_Rect){0,0,r->viewport.w,r->viewport.h};
 a.x+=r->viewport.x;a.y+=r->viewport.y;return SDL_FillSurfaceRect(r->surface,&a,r->color)?0:-1;}
int dos_ui_clear(DosUiRenderer *r)
{if(!r)return -1;
#ifdef AB3D_NATIVE_RGB565
 dos_ui_materialize_canvas(r);
#endif
 SDL_SetSurfaceClipRect(r->surface,NULL);bool ok=SDL_FillSurfaceRect(r->surface,NULL,r->color);apply_clip(r);return ok?0:-1;}
int dos_ui_line(DosUiRenderer *r,int x0,int y0,int x1,int y1)
{
    if(!r)return -1;
#ifdef AB3D_NATIVE_RGB565
    dos_ui_materialize_canvas(r);
#endif
    SDL_Surface *surface=r->surface;SDL_Rect clip;
    if(!SDL_GetSurfaceClipRect(surface,&clip) || !SDL_LockSurface(surface))return -1;
    x0+=r->viewport.x;x1+=r->viewport.x;y0+=r->viewport.y;y1+=r->viewport.y;
    int dx=abs(x1-x0),dy=-abs(y1-y0),sx=x0<x1?1:-1,sy=y0<y1?1:-1,e=dx+dy;
    /* Same Bresenham pixels as the solid fill path, with one SDL lock per line. */
    for(;;) {
        if(x0>=clip.x && x0<clip.x+clip.w && y0>=clip.y && y0<clip.y+clip.h) {
#ifdef AB3D_DOS_INDEXED
            *((Uint8 *)surface->pixels+y0*surface->pitch+x0)=(Uint8)r->color;
#else
            *(Uint16 *)((Uint8 *)surface->pixels+y0*surface->pitch+x0*2)=r->color;
#endif
        }
        if(x0==x1 && y0==y1)break;
        int twice=e*2;if(twice>=dy){e+=dy;x0+=sx;}if(twice<=dx){e+=dx;y0+=sy;}
    }
    SDL_UnlockSurface(surface);return 0;
}
int dos_ui_outline(DosUiRenderer *r,const SDL_Rect *rect)
{
    if(!r)return -1;SDL_Rect a=rect?*rect:(SDL_Rect){0,0,r->viewport.w,r->viewport.h};
    if(a.w<1 || a.h<1)return 0;
    return dos_ui_line(r,a.x,a.y,a.x+a.w-1,a.y) || dos_ui_line(r,a.x,a.y+a.h-1,a.x+a.w-1,a.y+a.h-1) ||
           dos_ui_line(r,a.x,a.y,a.x,a.y+a.h-1) || dos_ui_line(r,a.x+a.w-1,a.y,a.x+a.w-1,a.y+a.h-1)?-1:0;
}
bool dos_ui_present(DosUiRenderer *r){
#ifdef AB3D_NATIVE_RGB565
    dos_ui_materialize_canvas(r);
#endif
    return r && SDL_UpdateWindowSurface(r->window);
}
SDL_Surface *dos_ui_readback(DosUiRenderer *r,const SDL_Rect *rect)
{
    if(!r)return NULL;
#ifdef AB3D_NATIVE_RGB565
    dos_ui_materialize_canvas(r);
#endif
    SDL_Rect a=rect?*rect:(SDL_Rect){0,0,r->surface->w,r->surface->h};
    SDL_Surface *copy=SDL_CreateSurface(a.w,a.h,SDL_PIXELFORMAT_RGB565);
    if(copy && !SDL_BlitSurface(r->surface,&a,copy,NULL)){SDL_DestroySurface(copy);return NULL;}return copy;
}
static int edge(int ax,int ay,int bx,int by,int x,int y){return (bx-ax)*(y-ay)-(by-ay)*(x-ax);}
int dos_ui_geometry(DosUiRenderer *r,DosUiImage *image,const SDL_Vertex *v,int count,const int *indices,int n)
{
    if(image || !indices || n%3){SDL_SetError("DOS map uses solid indexed triangles");return -1;}
    for(int i=0;i<n;i+=3){int ids[3]={indices[i],indices[i+1],indices[i+2]},x[3],y[3];
        for(int j=0;j<3;j++){if(ids[j]<0 || ids[j]>=count)return -1;x[j]=(int)v[ids[j]].position.x*2;y[j]=(int)v[ids[j]].position.y*2;}
        if(edge(x[0],y[0],x[1],y[1],x[2],y[2])<0){int t=x[1];x[1]=x[2];x[2]=t;t=y[1];y[1]=y[2];y[2]=t;}
        SDL_FColor c=v[ids[0]].color;if(dos_ui_color(r,c.r*255,c.g*255,c.b*255,255)<0)return -1;
        int minx=SDL_max(0,SDL_min(x[0],SDL_min(x[1],x[2]))/2),maxx=SDL_min(r->viewport.w-1,SDL_max(x[0],SDL_max(x[1],x[2]))/2);
        int miny=SDL_max(0,SDL_min(y[0],SDL_min(y[1],y[2]))/2),maxy=SDL_min(r->viewport.h-1,SDL_max(y[0],SDL_max(y[1],y[2]))/2);
        for(int py=miny;py<=maxy;py++)for(int px=minx;px<=maxx;px++){bool inside=true;
            for(int j=0;j<3;j++){int k=(j+1)%3,e=edge(x[j],y[j],x[k],y[k],px*2+1,py*2+1);
                bool top_left=y[k]<y[j] || (y[k]==y[j] && x[k]>x[j]);if(e<0 || (!e && !top_left)){inside=false;break;}}
            if(inside){SDL_Rect p={px,py,1,1};if(dos_ui_fill(r,&p)<0)return -1;}}
    }return 0;
}
