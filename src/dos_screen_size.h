#ifndef AB3D_DOS_SCREEN_SIZE_H
#define AB3D_DOS_SCREEN_SIZE_H

/* Optional DOS view sizes; the native video mode and HUD keep their size. */
enum { DOS_SCREEN_SIZE_MIN = 40, DOS_SCREEN_SIZE_MAX = 100,
       DOS_SCREEN_SIZE_STEP = 20, DOS_SCREEN_SIZE_DEFAULT = 100 };

static inline int dos_screen_size_valid(int percent)
{
    return percent >= DOS_SCREEN_SIZE_MIN && percent <= DOS_SCREEN_SIZE_MAX &&
           percent % DOS_SCREEN_SIZE_STEP == 0;
}

static inline int dos_screen_size_adjust(int percent, int direction)
{
    percent += direction * DOS_SCREEN_SIZE_STEP;
    if (percent < DOS_SCREEN_SIZE_MIN) percent = DOS_SCREEN_SIZE_MIN;
    if (percent > DOS_SCREEN_SIZE_MAX) percent = DOS_SCREEN_SIZE_MAX;
    return percent;
}

static inline void dos_screen_size_dimensions(int width, int height, int percent,
                                               int *view_width, int *view_height)
{
    *view_width = width * percent / 100;
    *view_height = height * percent / 100;
}

#endif
