#ifndef AB3D_TEXT_DOS_H
#define AB3D_TEXT_DOS_H

#include <stdint.h>
#include <SDL3/SDL.h>

enum { DOS_TEXT_COLUMNS=80,DOS_TEXT_ROWS=25 };
void dos_text_begin(SDL_Window *window);
void dos_text_end(SDL_Window *window);
int dos_text_active(void);
void dos_text_present(const char *const *lines,unsigned count,int alpha);
void dos_text_read(uint16_t cells[DOS_TEXT_ROWS][DOS_TEXT_COLUMNS]);
void dos_text_capture(const char *filename);
void dos_text_check_graphics_palette(SDL_Surface *surface);

#endif
