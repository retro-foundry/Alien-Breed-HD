#include "sprite_alpha_dos.h"
#include <stddef.h>
#include <string.h>

typedef struct { unsigned type,count;const uint32_t (*rows)[3]; } DosSpriteAlphaBank;
#include "sprite_alpha_data.h"

typedef struct {
    const uint8_t *wad,*ptr;
    size_t wad_bytes,ptr_bytes;
} DosSpriteAlphaSource;
static DosSpriteAlphaSource sprite_alpha_sources[sizeof(sprite_alpha_banks)/sizeof(*sprite_alpha_banks)];

void dos_sprite_alpha_reset(void)
{
    memset(sprite_alpha_sources,0,sizeof(sprite_alpha_sources));
}

void dos_sprite_alpha_bind(unsigned type,const uint8_t *wad,size_t wad_bytes,
                            const uint8_t *ptr,size_t ptr_bytes)
{
    for(unsigned i=0;i<sizeof(sprite_alpha_banks)/sizeof(*sprite_alpha_banks);i++)
        if(sprite_alpha_banks[i].type==type)
            sprite_alpha_sources[i]=(DosSpriteAlphaSource){wad,ptr,wad_bytes,ptr_bytes};
}

static const uint32_t *sprite_alpha_find(const DosSpriteAlphaBank *bank,uint32_t key)
{
    unsigned low=0,high=bank->count;
    while(low<high) {
        unsigned mid=low+(high-low)/2;
        if(bank->rows[mid][0]<key)low=mid+1;else high=mid;
    }
    return low<bank->count && bank->rows[low][0]==key?bank->rows[low]+1:NULL;
}

const uint32_t *dos_sprite_alpha_source(const DosSpriteRows *sprite)
{
    if(sprite->ptr_offset>65535 || sprite->down_strip>255 || sprite->columns>255)return NULL;
    uint32_t key=(sprite->ptr_offset<<16)|(sprite->down_strip<<8)|sprite->columns;
    for(unsigned i=0;i<sizeof(sprite_alpha_banks)/sizeof(*sprite_alpha_banks);i++) {
        const DosSpriteAlphaSource *source=sprite_alpha_sources+i;
        if(source->wad && source->wad==sprite->wad && source->ptr==sprite->ptr &&
           source->wad_bytes==sprite->wad_bytes && source->ptr_bytes==sprite->ptr_bytes)
            return sprite_alpha_find(sprite_alpha_banks+i,key);
    }
    return NULL;
}
