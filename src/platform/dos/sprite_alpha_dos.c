#include "sprite_alpha_dos.h"
#include <stddef.h>

typedef struct { unsigned type,count;const uint32_t (*rows)[3]; } DosSpriteAlphaBank;
#include "sprite_alpha_data.h"

const uint32_t *dos_sprite_alpha_rows(unsigned type,uint32_t ptr_offset,
                                      unsigned down_strip,unsigned columns)
{
    if(ptr_offset>65535 || down_strip>255 || columns>255)return NULL;
    uint32_t key=(ptr_offset<<16)|(down_strip<<8)|columns;
    for(unsigned i=0;i<sizeof(sprite_alpha_banks)/sizeof(*sprite_alpha_banks);i++) {
        const DosSpriteAlphaBank *bank=sprite_alpha_banks+i;
        if(bank->type!=type)continue;
        unsigned low=0,high=bank->count;
        while(low<high) {
            unsigned mid=low+(high-low)/2;
            if(bank->rows[mid][0]<key)low=mid+1;else high=mid;
        }
        return low<bank->count && bank->rows[low][0]==key?bank->rows[low]+1:NULL;
    }
    return NULL;
}
