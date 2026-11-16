#include "audio.h"
#include <stdint.h>
#include <math.h>

int audio_distance_volume(int noise_volume, int dx, int dz)
{
    /* SoundPlayer.s MakeSomeNoise's inverse-distance curve and loudness cap.
     * Keep fractional distance/gain until the final 0..255 mixer conversion:
     * rounding first to Paula's integer 0..64 caused audible volume steps. */
    uint64_t squared = (uint64_t)((int64_t)dx * dx) +
                       (uint64_t)((int64_t)dz * dz);
    if (noise_volume <= 0) return 0;
    if (squared == 0) return 255;
    int64_t numerator = (int64_t)noise_volume * 64;
    if (numerator > 32767) numerator = 32767;
#ifdef AB3D_DOS
    /* Same MakeSomeNoise inverse-distance curve, retaining the PC port's
     * fractional gain until its final 0..255 conversion. For candidate v:
     *   v <= numerator*255 / (16*(sqrt(squared)+4))
     * iff squared*(16*v)^2 <= (numerator*255-64*v)^2.
     * Reject v=1 first; the remaining products then fit in uint64_t even
     * for callers with full-range int coordinates. No distance quantization. */
    uint64_t gain = (uint64_t)numerator * 255u;
    if (squared > (gain - 64u) * (gain - 64u) / 256u) return 0;
    unsigned low = 0, high = 256;
    while (high - low > 1) {
        unsigned v = (low + high) / 2;
        uint64_t remaining = gain - 64u * v;
        if (squared * (256u * v * v) <= remaining * remaining) low = v;
        else high = v;
    }
    return (int)low;
#else
    double volume = (double)numerator / (sqrt((double)squared) / 4.0 + 1.0);
    if (volume >= 64.0) return 255;
    return (int)(volume * 255.0 / 64.0);
#endif
}
