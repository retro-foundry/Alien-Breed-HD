#ifndef AB3D_PALETTE_DOS_H
#define AB3D_PALETTE_DOS_H
#ifndef SDL_DISABLE_OLD_NAMES
#define SDL_DISABLE_OLD_NAMES 1
#endif
#include <SDL3/SDL.h>
#define DOS_PIXEL_FORMAT SDL_PIXELFORMAT_INDEX8
#define DOS_PIXEL_BYTES 1
#define DOS_PIXEL_KEY 255
extern uint8_t dos_c12_to_index[4096];
extern uint16_t dos_index_to_c12[256];
extern unsigned dos_palette_epoch;
void dos_palette_load(int level);
void dos_palette_attach(SDL_Surface *surface);
void dos_palette_install(SDL_Window *window);
SDL_Surface *dos_palette_convert(SDL_Surface *source);
#endif
