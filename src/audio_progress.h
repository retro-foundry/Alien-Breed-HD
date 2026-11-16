#ifndef AB3D_AUDIO_PROGRESS_H
#define AB3D_AUDIO_PROGRESS_H
#include <stdint.h>

#ifdef AB3D_DOS
/* Unsigned 0.32, retaining the original binary32 music-position phases. */
typedef uint32_t AudioProgress;
#define AUDIO_PROGRESS_START UINT32_C(0)
#define AUDIO_PROGRESS_END UINT32_MAX

static inline AudioProgress audio_progress_ratio(uint32_t position,uint32_t length)
{
    if(!length || position>=length)return AUDIO_PROGRESS_END;
    if(!position)return AUDIO_PROGRESS_START;
    /* DJGPP retains both integer inputs in x87 precision until the result
     * is passed as binary32. Round that quotient once, ties to even. */
    uint64_t numerator=position,denominator=length;
    unsigned shift=23;
    while((numerator<<shift)<(denominator<<23))shift++;
    uint64_t scaled=numerator<<shift,significand=scaled/denominator,remainder=scaled%denominator;
    significand+=(remainder*2>denominator || (remainder*2==denominator && (significand&1)));
    uint64_t phase=shift<=32?significand<<(32-shift):significand>>(shift-32);
    return phase>UINT32_MAX?AUDIO_PROGRESS_END:(AudioProgress)phase;
}

static inline int audio_progress_alpha(AudioProgress phase,int maximum)
{
    return (int)(((uint64_t)phase*(unsigned)maximum+UINT64_C(0x80000000))>>32);
}
#else
typedef float AudioProgress;
#define AUDIO_PROGRESS_START 0.0f
#define AUDIO_PROGRESS_END 1.0f
static inline AudioProgress audio_progress_ratio(uint32_t position,uint32_t length)
{
    return length?(float)position/(float)length:AUDIO_PROGRESS_END;
}
static inline int audio_progress_alpha(AudioProgress phase,int maximum)
{
    if(phase<0.0f)phase=0.0f;
    if(phase>1.0f)phase=1.0f;
    return (int)(phase*(float)maximum+0.5f);
}
#endif
#endif
