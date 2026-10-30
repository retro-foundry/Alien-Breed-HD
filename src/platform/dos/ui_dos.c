/* UI pixels are converted once at upload, then copied as native opaque RGB565
 * runs. Alpha is only an on/off cutout mask; disabled items use cached grey. */
#include <SDL3/SDL.h>
#include "ui_dos.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef AB3D_DOS_INDEXED
#include "palette_dos.h"
static void release_indexed(DosUiImage *i)
{
    SDL_DestroySurface(i->indexed_normal);i->indexed_normal=NULL;
    SDL_DestroySurface(i->indexed_grey);i->indexed_grey=NULL;
    if(i->indexed_strips)for(int n=0;n<i->strip_count;n++)SDL_DestroySurface(i->indexed_strips[n]);
    free(i->indexed_strips);i->indexed_strips=NULL;
}
static void refresh_indexed(DosUiImage *i)
{if(i->palette_epoch!=dos_palette_epoch){release_indexed(i);i->palette_epoch=dos_palette_epoch;}}
#endif

static DosUiImage *images;

static void ui_fail(const char *operation)
{
    fprintf(stderr, "fatal: DOS UI %s: %s\n", operation, SDL_GetError());
    exit(1);
}

static Uint16 rgb565(Uint8 r, Uint8 g, Uint8 b)
{
    return (Uint16)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
}

DosUiImage *dos_ui_create_image(int width,int height,SDL_TextureAccess access)
{
    DosUiImage *i=calloc(1,sizeof(*i));
    if(!i)return NULL;
    i->normal=SDL_CreateSurface(width,height,SDL_PIXELFORMAT_RGB565);
    if(!i->normal){free(i);return NULL;}
    i->width=width;i->height=height;
    i->alpha=i->red=i->green=i->blue=255;i->access=access;
    i->next=images;images=i;return i;
}

/* Native title rows occupy separate small allocations. Returning to the menu
 * must not require a contiguous 160 KiB hole in the level-loading heap. */
DosUiImage *dos_ui_create_striped_image(int width,int height)
{
    DosUiImage *i=calloc(1,sizeof(*i));if(!i)return NULL;
    i->width=width;i->height=height;i->strip_count=(height+15)/16;
    i->alpha=i->red=i->green=i->blue=255;i->access=SDL_TEXTUREACCESS_STATIC;
    i->next=images;images=i;
    i->strips=calloc((size_t)i->strip_count,sizeof(*i->strips));
    if(!i->strips){dos_ui_texture_destroy(i);return NULL;}
    for(int n=0;n<i->strip_count;n++) {
        int rows=SDL_min(16,height-n*16);
        i->strips[n]=SDL_CreateSurface(width,rows,SDL_PIXELFORMAT_RGB565);
        if(!i->strips[n] || !SDL_FillSurfaceRect(i->strips[n],NULL,0) ||
           !SDL_SetSurfaceBlendMode(i->strips[n],SDL_BLENDMODE_NONE)) {
            dos_ui_texture_destroy(i);return NULL;
        }
    }
    return i;
}

void dos_ui_texture_destroy(DosUiImage *texture)
{
    DosUiImage **link = &images;
    while (*link) {
        DosUiImage *i = *link;
        if (i == texture) {
            *link = i->next;
#ifdef AB3D_DOS_INDEXED
            release_indexed(i);
#endif
            SDL_DestroySurface(i->normal);
            SDL_DestroySurface(i->grey);
            if(i->strips)for(int n=0;n<i->strip_count;n++)SDL_DestroySurface(i->strips[n]);
            free(i->strips);free(i);
            return;
        }
        link = &i->next;
    }
}

void dos_ui_texture_upload(DosUiImage *image, const SDL_Rect *rect, const void *pixels, int pitch)
{
#ifdef AB3D_DOS_INDEXED
    if(image)release_indexed(image);
#endif
    SDL_Surface *src = image?image->normal:NULL;
    if (!src || src->format!=SDL_PIXELFORMAT_RGB565) ui_fail("native texture surface");
    if(rect || !pixels || pitch<src->w*4)ui_fail("expected full ARGB8888 UI upload");
    SDL_DestroySurface(image->grey);image->grey=NULL;
    if (!SDL_LockSurface(src)) ui_fail("upload lock");
    if(!SDL_SetSurfaceColorKey(src,false,0))ui_fail("reset mask");
    unsigned char used[65536 / 8] = {0};
    bool masked = false;
    for (int y = 0; y < src->h; y++) {
        const Uint32 *in = (const Uint32 *)((const Uint8 *)pixels + y * pitch);
        Uint16 *out = (Uint16 *)((Uint8 *)image->normal->pixels + y * image->normal->pitch);
        for (int x = 0; x < src->w; x++) {
            Uint8 r=in[x]>>16,g=in[x]>>8,b=in[x],a=in[x]>>24;
            Uint16 color = rgb565(r, g, b);
            out[x] = color;
            if (a) used[color >> 3] |= (1u << (color & 7));
            else masked = true;
        }
    }
    if (masked) {
        unsigned key;
        for (key = 1; key < 65536 && (used[key >> 3] & (1u << (key & 7))); key++);
        if (key == 65536) {
            if (used[0] & 1) ui_fail("no unused mask color");
            key = 0;
        }
        for (int y = 0; y < src->h; y++) {
            const Uint32 *in = (const Uint32 *)((const Uint8 *)pixels + y * pitch);
            Uint16 *out = (Uint16 *)((Uint8 *)image->normal->pixels + y * image->normal->pitch);
            for (int x = 0; x < src->w; x++) {
                Uint8 a=in[x]>>24;
                if (!a) out[x] = (Uint16)key;
            }
        }
        if (!SDL_SetSurfaceColorKey(image->normal, true, key)) ui_fail("mask");
    }
    SDL_UnlockSurface(src);
    if (!SDL_SetSurfaceBlendMode(image->normal, SDL_BLENDMODE_NONE) ||
        !SDL_SetSurfaceRLE(image->normal, true)) ui_fail("opaque cache");
}

static SDL_Surface *grey_image(DosUiImage *image)
{
    if (image->grey) return image->grey;
    SDL_Surface *src = image->normal;
    image->grey = SDL_CreateSurface(src->w, src->h, SDL_PIXELFORMAT_RGB565);
    if (!image->grey || !SDL_LockSurface(src)) ui_fail("grey cache");
    Uint32 key;
    bool masked = SDL_SurfaceHasColorKey(src);
    if (masked && !SDL_GetSurfaceColorKey(src, &key)) ui_fail("grey mask");
    for (int y = 0; y < src->h; y++) {
        const Uint16 *in = (const Uint16 *)((const Uint8 *)src->pixels + y * src->pitch);
        Uint16 *out = (Uint16 *)((Uint8 *)image->grey->pixels + y * image->grey->pitch);
        for (int x = 0; x < src->w; x++) {
            Uint16 c = in[x];
            if (masked && c == key) { out[x] = 1; continue; }
            unsigned r = (c >> 11) * 255 / 31;
            unsigned g = ((c >> 5) & 63) * 255 / 63;
            unsigned b = (c & 31) * 255 / 31;
            unsigned value = SDL_max(r, SDL_max(g, b)) * 2 / 3;
            out[x] = rgb565(value, value, value);
        }
    }
    SDL_UnlockSurface(src);
    if ((masked && !SDL_SetSurfaceColorKey(image->grey, true, 1)) ||
        !SDL_SetSurfaceBlendMode(image->grey, SDL_BLENDMODE_NONE) ||
        !SDL_SetSurfaceRLE(image->grey, true)) ui_fail("opaque grey cache");
    return image->grey;
}

void dos_ui_check_images(void)
{
    int count = 0;
    for (DosUiImage *i = images; i; i = i->next) {
        if (!i->grey) continue;
        SDL_BlendMode blend;
        if (!SDL_GetSurfaceBlendMode(i->grey, &blend) || blend != SDL_BLENDMODE_NONE)
            ui_fail("grey image must be opaque");
        Uint32 normal_key = 0, grey_key = 0;
        bool masked = SDL_SurfaceHasColorKey(i->normal);
        if (masked != SDL_SurfaceHasColorKey(i->grey)) ui_fail("grey mask missing");
        if (masked && (!SDL_GetSurfaceColorKey(i->normal, &normal_key) ||
                       !SDL_GetSurfaceColorKey(i->grey, &grey_key))) ui_fail("check mask");
        if (!SDL_LockSurface(i->normal) || !SDL_LockSurface(i->grey)) ui_fail("check lock");
        for (int y = 0; y < i->normal->h; y++) {
            const Uint16 *normal = (Uint16 *)((Uint8 *)i->normal->pixels + y * i->normal->pitch);
            const Uint16 *grey = (Uint16 *)((Uint8 *)i->grey->pixels + y * i->grey->pitch);
            for (int x = 0; x < i->normal->w; x++) {
                bool cutout = masked && normal[x] == normal_key;
                if (cutout != (masked && grey[x] == grey_key)) ui_fail("grey cutout differs");
                if (!cutout && ((grey[x] >> 11) != (grey[x] & 31) ||
                                (grey[x] >> 11) != ((grey[x] >> 6) & 31)))
                    ui_fail("disabled pixel is not grey");
            }
        }
        SDL_UnlockSurface(i->grey);
        SDL_UnlockSurface(i->normal);
        count++;
    }
    if (count < 4) ui_fail("missing disabled key images");
    printf("[DOSCHECK] %d opaque grey UI images preserve exact cutout masks\n", count);
}

#ifdef AB3D_DOS_INDEXED
void dos_ui_check_indexed_images(void)
{
    unsigned checked=0;
    for(DosUiImage *i=images;i;i=i->next) {
        if(i->palette_epoch!=dos_palette_epoch)continue; /* Unused cached images rebuild on next blit. */
        SDL_Surface *sources[]={i->normal,i->grey};
        SDL_Surface *targets[]={i->indexed_normal,i->indexed_grey};
        for(unsigned n=0;n<2;n++) {
            SDL_Surface *src=sources[n],*dst=targets[n];
            if(!dst)continue;
            Uint32 key=0;bool mask=SDL_SurfaceHasColorKey(src);
            if(mask && !SDL_GetSurfaceColorKey(src,&key))ui_fail("indexed cache source mask");
            if(!SDL_LockSurface(src) || !SDL_LockSurface(dst))ui_fail("indexed cache lock");
            for(int y=0;y<src->h;y++) {
                const Uint16 *a=(const Uint16 *)((const Uint8 *)src->pixels+y*src->pitch);
                const Uint8 *b=(const Uint8 *)dst->pixels+y*dst->pitch;
                for(int x=0;x<src->w;x++) {
                    unsigned c=a[x],c12=(((c>>12)&15)<<8)|(((c>>7)&15)<<4)|((c>>1)&15);
                    unsigned expected=(mask && c==key)?255:dos_c12_to_index[c12];
                    if(b[x]!=expected)ui_fail("stale indexed UI pixels or changed cutout");
                }
            }
            SDL_UnlockSurface(dst);SDL_UnlockSurface(src);checked++;
        }
    }
    if(!checked)ui_fail("no indexed UI caches exercised");
    printf("[DOSCHECK] %u indexed UI caches preserve current palette and exact cutouts\n",checked);
}
#endif

int dos_ui_blit(DosUiRenderer *renderer, DosUiImage *image,
                const SDL_Rect *src, const SDL_Rect *dst)
{
    if (!image) ui_fail("texture was not uploaded");
    Uint8 alpha=image->alpha,r=image->red,g=image->green,b=image->blue;
    if (!alpha) return 0;
    if (r != 255 || g != 255 || b != 255) ui_fail("unexpected color modulation");
    if(image->strips) {
#ifdef AB3D_DOS_INDEXED
        refresh_indexed(image);
        if(!image->indexed_strips)image->indexed_strips=calloc(image->strip_count,sizeof(SDL_Surface *));
        if(!image->indexed_strips)ui_fail("indexed title cache");
#endif
        if(src || !dst || dst->w!=image->width || dst->h!=image->height || alpha!=255)
            ui_fail("strip picture must be opaque and 1:1");
        if(!renderer)ui_fail("missing strip target");
        for(int n=0;n<image->strip_count;n++) {
            SDL_Rect target={dst->x+renderer->viewport.x,dst->y+renderer->viewport.y+n*16,image->width,image->strips[n]->h};
            SDL_Surface *strip=image->strips[n];
#ifdef AB3D_DOS_INDEXED
            if(!image->indexed_strips[n])image->indexed_strips[n]=dos_palette_convert(strip);
            strip=image->indexed_strips[n];
#endif
            if(!SDL_BlitSurface(strip,NULL,renderer->surface,&target))ui_fail("native picture strip");
        }
        return 0;
    }
    SDL_Surface *pixels = alpha == 255 ? image->normal : grey_image(image);
#ifdef AB3D_DOS_INDEXED
    refresh_indexed(image);
    SDL_Surface **native=alpha==255?&image->indexed_normal:&image->indexed_grey;
    if(!*native)*native=dos_palette_convert(pixels);
    pixels=*native;
#endif
    if(!SDL_SetSurfaceBlendMode(pixels,SDL_BLENDMODE_NONE))ui_fail("opaque texture");
    SDL_Rect area = src ? *src : (SDL_Rect){0, 0, pixels->w, pixels->h};
    if (!dst || dst->w != area.w || dst->h != area.h) ui_fail("UI must be 1:1");
    SDL_Surface *screen = renderer?renderer->surface:NULL;
    if (!screen || screen->format !=
#ifdef AB3D_DOS_INDEXED
        SDL_PIXELFORMAT_INDEX8
#else
        SDL_PIXELFORMAT_RGB565
#endif
        ) ui_fail("native DOS target");
    SDL_Rect target = *dst;
    target.x+=renderer->viewport.x;target.y+=renderer->viewport.y;
    if (!SDL_BlitSurface(pixels, &area, screen, &target)) ui_fail("opaque blit");
    return 0;
}
