#ifndef AB3D_WATER_FORMAT_H
#define AB3D_WATER_FORMAT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum { WATER_NATIVE_HEADER_BYTES=8, WATER_FILE_BYTES=65536,
       WATER_PHASE_BYTES=4096, WATER_PHASE_COUNT=8 };

/* DOS ABW1 stores eight contiguous phase planes, followed by all original
 * odd bytes in word order. It is a permutation of the complete WaterFile. */
static inline size_t water_native_payload_size(const uint8_t header[8],size_t file_bytes)
{
    if(file_bytes!=WATER_NATIVE_HEADER_BYTES+WATER_FILE_BYTES || memcmp(header,"ABW1",4))return 0;
    uint32_t bytes=(uint32_t)header[4]|((uint32_t)header[5]<<8)|
        ((uint32_t)header[6]<<16)|((uint32_t)header[7]<<24);
    return bytes==WATER_FILE_BYTES?bytes:0;
}

/* The caller retains the original source-byte bounds checks. Stock
 * texturedwater reads even offsets; retaining odd bytes also keeps the
 * complete original data addressable. Desktop assets remain in Amiga order. */
static inline uint8_t water_source_byte(const uint8_t *data,size_t offset)
{
#ifdef AB3D_DOS
    if(offset&1u)return data[32768u+(offset>>1)];
    offset=((offset&0x300u)<<5)|((offset&2u)<<11)|
           ((offset&0xfc00u)>>4)|((offset&0xfcu)>>2);
#endif
    return data[offset];
}

#endif
