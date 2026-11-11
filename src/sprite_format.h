#ifndef AB3D_SPRITE_FORMAT_H
#define AB3D_SPRITE_FORMAT_H

#include <stdint.h>
#include <string.h>

/* DOS package ABS1 header: magic/version and little-endian payload length.
 * The payload remains one word per Amiga word; PTR offsets are unchanged. */
enum { SPRITE_NATIVE_HEADER_BYTES = 8 };
static inline size_t sprite_native_payload_size(const uint8_t header[8],size_t file_bytes)
{
    uint32_t bytes=(uint32_t)header[4]|((uint32_t)header[5]<<8)|
        ((uint32_t)header[6]<<16)|((uint32_t)header[7]<<24);
    if (memcmp(header,"ABS1",4) || file_bytes<8 || !bytes || (bytes&1) ||
        bytes!=file_bytes-8) return 0;
    return bytes;
}

/* ObjDraw3.ChipRam.s BitMapObj packs three five-bit pixels per word.
 * DOS transports the same words in native byte order; desktop assets retain
 * the original Amiga order. memcpy permits unaligned samples without aliasing. */
static inline uint16_t sprite_word_read(const uint8_t *p)
{
#ifdef AB3D_DOS
    uint16_t word;
    memcpy(&word,p,sizeof(word));
    return word;
#else
    return (uint16_t)(((unsigned)p[0]<<8)|p[1]);
#endif
}

static inline void sprite_word_write(uint8_t *p,uint16_t word)
{
#ifdef AB3D_DOS
    memcpy(p,&word,sizeof(word));
#else
    p[0]=(uint8_t)(word>>8);
    p[1]=(uint8_t)word;
#endif
}

#endif
