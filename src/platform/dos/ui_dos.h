#ifndef AB3D_UI_DOS_H
#define AB3D_UI_DOS_H

/* Immediate native SDL surfaces; no SDL_Renderer or duplicate texture pixels. */
typedef struct DosUiImage {
    SDL_Surface *normal,*grey;
    SDL_Surface **strips;
    int strip_count,width,height;
    Uint8 alpha,red,green,blue;
    SDL_TextureAccess access;
    struct DosUiImage *next;
} DosUiImage;
typedef struct DosUiRenderer {
    SDL_Window *window;
    SDL_Surface *surface;
    SDL_Rect viewport,clip;
    bool clipped;
    Uint16 color;
} DosUiRenderer;
DosUiRenderer *dos_ui_create_renderer(SDL_Window *window);
void dos_ui_destroy_renderer(DosUiRenderer *renderer);
DosUiImage *dos_ui_create_image(int width,int height,SDL_TextureAccess access);
DosUiImage *dos_ui_create_striped_image(int width,int height);
void dos_ui_texture_upload(DosUiImage *image,const SDL_Rect *rect,const void *pixels,int pitch);
void dos_ui_texture_destroy(DosUiImage *image);
void dos_ui_check_images(void);
int dos_ui_blit(DosUiRenderer *renderer, DosUiImage *texture,
                const SDL_Rect *src, const SDL_Rect *dst);
int dos_ui_color(DosUiRenderer *r,Uint8 red,Uint8 green,Uint8 blue,Uint8 alpha);
int dos_ui_clear(DosUiRenderer *r);
int dos_ui_fill(DosUiRenderer *r,const SDL_Rect *rect);
int dos_ui_line(DosUiRenderer *r,int x0,int y0,int x1,int y1);
int dos_ui_outline(DosUiRenderer *r,const SDL_Rect *rect);
bool dos_ui_viewport(DosUiRenderer *r,const SDL_Rect *rect);
bool dos_ui_clip(DosUiRenderer *r,const SDL_Rect *rect);
bool dos_ui_present(DosUiRenderer *r);
SDL_Surface *dos_ui_readback(DosUiRenderer *r,const SDL_Rect *rect);
int dos_ui_geometry(DosUiRenderer *r,DosUiImage *image,const SDL_Vertex *v,int count,const int *indices,int n);

#endif
