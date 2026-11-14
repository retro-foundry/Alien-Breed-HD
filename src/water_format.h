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

static inline size_t water_phase_index(size_t offset)
{
    return ((offset&0x300u)<<5)|((offset&2u)<<11)|
           ((offset&0xfc00u)>>4)|((offset&0xfcu)>>2);
}

/* The caller retains the original source-byte bounds checks. Stock
 * texturedwater reads even offsets; retaining odd bytes also keeps the
 * complete original data addressable. Desktop assets remain in Amiga order. */
static inline uint8_t water_source_byte(const uint8_t *data,size_t offset)
{
#ifdef AB3D_DOS
    if(offset&1u)return data[32768u+(offset>>1)];
    offset=water_phase_index(offset);
#endif
    return data[offset];
}

#if defined(AB3D_DOS) && defined(AB3D_NATIVE_RGB565)
enum { WATER_PHASE_LEVEL_SHIFT=10, WATER_PHASE_RUN_SHIFT=4, WATER_PHASE_RUN_MAX=4 };
typedef struct {
    const uint8_t *data;
    size_t bytes;
    uint8_t prepared, odd_value;
} WaterSourceView;

/* The owner passes a newly loaded ABW1 bank and a zeroed descriptor. Reset
 * the descriptor when freeing/reloading that bank; repeated preparation of
 * the same live view is harmless. No file-format or bank-size change.
 * AB3DI.s texturedwater needs level*1024. Bits 4..5 additionally record a
 * conservative equal-level run, minus one, around the selected 64-texel row.
 * Native Brighten's duplicate slots ignore these low green bits after the
 * offset is doubled; the hint never changes a sampled colour. A uniform odd
 * stream leaves room for these words while its actual value stays in the view.
 * Other banks retain the complete byte-phase representation. */
static inline int water_source_prepare(WaterSourceView *view,uint8_t *data,size_t bytes)
{
    if(view->prepared && view->data==data && view->bytes==bytes)return 1;
    *view=(WaterSourceView){data,bytes,0,0};
    if(!data || bytes!=WATER_FILE_BYTES)return 0;
    const unsigned phases=WATER_PHASE_COUNT*WATER_PHASE_BYTES;
    uint8_t odd=data[phases];
    for(unsigned i=0;i<phases;i++)
        if(data[i]>63 || data[phases+i]!=odd)return 0;
    for(unsigned i=phases;i;i--) {
        uint16_t level=(uint16_t)data[i-1]<<WATER_PHASE_LEVEL_SHIFT;
        memcpy(data+(i-1)*sizeof(level),&level,sizeof(level));
    }
    for(unsigned row=0;row<phases;row+=64)for(unsigned u=0;u<64;u++) {
        uint16_t level;
        memcpy(&level,data+(row+u)*sizeof(level),sizeof(level));
        unsigned run=1;
        while(run<WATER_PHASE_RUN_MAX) {
            uint16_t next;
            memcpy(&next,data+(row+((u+run)&63))*sizeof(next),sizeof(next));
            if((next>>WATER_PHASE_LEVEL_SHIFT)!=(level>>WATER_PHASE_LEVEL_SHIFT))break;
            run++;
        }
        level|=(uint16_t)((run-1)<<WATER_PHASE_RUN_SHIFT);
        memcpy(data+(row+u)*sizeof(level),&level,sizeof(level));
    }
    view->prepared=1;view->odd_value=odd;
    return 1;
}

/* Logical source bounds remain the caller's responsibility, as for
 * water_source_byte. Every original even and odd byte stays addressable. */
static inline uint8_t water_view_byte(const WaterSourceView *view,size_t offset)
{
    if(!view->prepared)return water_source_byte(view->data,offset);
    if(offset&1u)return view->odd_value;
    uint16_t level;
    memcpy(&level,view->data+water_phase_index(offset)*sizeof(level),sizeof(level));
    return (uint8_t)(level>>WATER_PHASE_LEVEL_SHIFT);
}
#endif

#endif
