/* Synchronous DOS UI primitives: SDL fills/blits on the native VESA surface. */
#include <SDL3/SDL.h>
#include "ui_dos.h"
#include <stdlib.h>

DosUiRenderer *dos_ui_create_renderer(SDL_Window *window)
{
    SDL_Surface *s=SDL_GetWindowSurface(window);
    if(!s || s->format!=SDL_PIXELFORMAT_RGB565){SDL_SetError("DOS UI requires RGB565");return NULL;}
    DosUiRenderer *r=calloc(1,sizeof(*r));if(!r){SDL_OutOfMemory();return NULL;}
    r->window=window;r->surface=s;r->viewport=(SDL_Rect){0,0,s->w,s->h};return r;
}
void dos_ui_destroy_renderer(DosUiRenderer *r){free(r);}
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
 r->color=(Uint16)(((red>>3)<<11)|((green>>2)<<5)|(blue>>3));return 0;}
int dos_ui_fill(DosUiRenderer *r,const SDL_Rect *rect)
{if(!r)return -1;SDL_Rect a=rect?*rect:(SDL_Rect){0,0,r->viewport.w,r->viewport.h};
 a.x+=r->viewport.x;a.y+=r->viewport.y;return SDL_FillSurfaceRect(r->surface,&a,r->color)?0:-1;}
int dos_ui_clear(DosUiRenderer *r)
{if(!r)return -1;SDL_SetSurfaceClipRect(r->surface,NULL);bool ok=SDL_FillSurfaceRect(r->surface,NULL,r->color);apply_clip(r);return ok?0:-1;}
int dos_ui_line(DosUiRenderer *r,int x0,int y0,int x1,int y1)
{
    if(!r)return -1;
    SDL_Surface *surface=r->surface;SDL_Rect clip;
    if(!SDL_GetSurfaceClipRect(surface,&clip) || !SDL_LockSurface(surface))return -1;
    x0+=r->viewport.x;x1+=r->viewport.x;y0+=r->viewport.y;y1+=r->viewport.y;
    int dx=abs(x1-x0),dy=-abs(y1-y0),sx=x0<x1?1:-1,sy=y0<y1?1:-1,e=dx+dy;
    /* Same Bresenham pixels as the solid fill path, with one SDL lock per line. */
    for(;;) {
        if(x0>=clip.x && x0<clip.x+clip.w && y0>=clip.y && y0<clip.y+clip.h)
            *(Uint16 *)((Uint8 *)surface->pixels+y0*surface->pitch+x0*2)=r->color;
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
bool dos_ui_present(DosUiRenderer *r){return r && SDL_UpdateWindowSurface(r->window);}
SDL_Surface *dos_ui_readback(DosUiRenderer *r,const SDL_Rect *rect)
{
    if(!r)return NULL;SDL_Rect a=rect?*rect:(SDL_Rect){0,0,r->surface->w,r->surface->h};
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
