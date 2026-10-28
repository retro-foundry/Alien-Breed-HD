#include "audio.h"
#include <stdint.h>
#include <math.h>

int audio_distance_volume(int noise_volume, int dx, int dz)
{
    /* SoundPlayer.s MakeSomeNoise's inverse-distance curve and loudness cap.
     * Keep fractional distance/gain until the final 0..255 mixer conversion:
     * rounding first to Paula's integer 0..64 caused audible volume steps. */
    int64_t squared = (int64_t)dx * dx + (int64_t)dz * dz;
    if (noise_volume <= 0) return 0;
    if (squared == 0) return 255;
    int64_t numerator = (int64_t)noise_volume * 64;
    if (numerator > 32767) numerator = 32767;
    double volume = (double)numerator / (sqrt((double)squared) / 4.0 + 1.0);
    if (volume >= 64.0) return 255;
    return (int)(volume * 255.0 / 64.0);
}
