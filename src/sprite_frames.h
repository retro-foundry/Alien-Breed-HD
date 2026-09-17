#ifndef AB3D_SPRITE_FRAMES_H
#define AB3D_SPRITE_FRAMES_H
#include <stdint.h>
#include <stddef.h>
#include "renderer.h"

/* -----------------------------------------------------------------------
 * Sprite FRAMES tables (from ObjDraw3.ChipRam.s).
 *
 * Each frame entry is {ptr_offset, down_strip}:
 *   ptr_offset = byte offset into .ptr table for the starting column
 *   down_strip = vertical row offset within each column's data
 *
 * The Amiga stores these as dc.w pairs; ptr_offset values are in bytes
 * (column_index * 4 since each PTR entry is 4 bytes).
 * ----------------------------------------------------------------------- */
typedef struct { uint16_t ptr_off; uint16_t down_strip; } SpriteFrame;

/* Type 0 - ALIEN2: 64 cols/frame, walking 0-15, exploding 16-31, dying 32-33 */
static const SpriteFrame frames_alien[] = {
    {0,0},{64*4,0},{64*4*2,0},{64*4*3,0},{64*4*4,0},{64*4*5,0},{64*4*6,0},{64*4*7,0},
    {64*4*8,0},{64*4*9,0},{64*4*10,0},{64*4*11,0},{64*4*12,0},{64*4*13,0},{64*4*14,0},{64*4*15,0},
    /* frames 16-19: gib set 1 (Explode1Anim: vect=0 frames 16-19) */
    {4*(64*16),0},{4*(64*16+16),0},{4*(64*16+32),0},{4*(64*16+48),0},
    /* frames 20-23: gib set 2 (Explode2Anim: vect=0 frames 20-23) */
    {4*(64*16),16},{4*(64*16+16),16},{4*(64*16+32),16},{4*(64*16+48),16},
    /* frames 24-27: gib set 3 (Explode3Anim: vect=0 frames 24-27) */
    {4*(64*16),32},{4*(64*16+16),32},{4*(64*16+32),32},{4*(64*16+48),32},
    /* frames 28-31: gib set 4 (Explode4Anim: vect=0 frames 28-31) */
    {4*(64*16),48},{4*(64*16+16),48},{4*(64*16+32),48},{4*(64*16+48),48},
    /* frames 32-33: death animation (alien body lying on floor) */
    {64*4*17,0},{64*4*18,0}
};

/* Type 1 - PICKUPS: variable layout */
static const SpriteFrame frames_pickups[] = {
    {0,0},                /* 0: medikit */
    {0,32},               /* 1: big gun */
    {64*4,32},            /* 2: bullet */
    {32*4,0},             /* 3: ammo */
    {64*4,0},             /* 4: battery */
    {192*4,0},            /* 5: rockets */
    {128*4,0},{(128+16)*4,0},{(128+32)*4,0},{(128+48)*4,0},   /* 6-9: gunpop */
    {128*4,16},{(128+16)*4,16},{(128+32)*4,16},{(128+48)*4,16}, /* 10-13 */
    {128*4,32},{(128+16)*4,32},{(128+32)*4,32},{(64+16)*4,32}, /* 14-17 */
    {64*4,48},{(64+16)*4,48},                                   /* 18-19 */
    {(64+32)*4,0},        /* 20: rocket launcher */
    {64*4,32},{(64+16)*4,32},{(64+16)*4,48},{64*4,48},          /* 21-24: grenade */
    {128*4,32},           /* 25: shotgun */
    {256*4,0},            /* 26: grenade launcher */
    {64*3*4,32},          /* 27: shotgun shells*4 */
    {(64*3+32)*4,0},      /* 28: shotgun shells*20 */
    {(64*3+32)*4,32}      /* 29: grenade clip */
};

/* Type 2 - BIGBULLET */
static const SpriteFrame frames_bigbullet[] = {
    {0,0},{0,32},{32*4,0},{32*4,32},{64*4,0},{64*4,32},{96*4,0},{96*4,32},
    {128*4,0},{128*4,32},{32*5*4,0},{32*5*4,32},{32*6*4,0},{32*6*4,32},
    {32*7*4,0},{32*7*4,32},{32*8*4,0},{32*8*4,32},{32*9*4,0},{32*9*4,32}
};

/* Type 4 - FLYINGMONSTER: 64 cols/frame, 21 frames */
static const SpriteFrame frames_flying[] = {
    {0,0},{64*4,0},{64*4*2,0},{64*4*3,0},{64*4*4,0},{64*4*5,0},{64*4*6,0},{64*4*7,0},
    {64*4*8,0},{64*4*9,0},{64*4*10,0},{64*4*11,0},{64*4*12,0},{64*4*13,0},{64*4*14,0},{64*4*15,0},
    {64*4*16,0},{64*4*17,0},{64*4*18,0},{64*4*19,0},{64*4*20,0}
};

/* Type 5 - KEYS: 4 frames */
static const SpriteFrame frames_keys[] = {
    {0,0},{0,32},{32*4,0},{32*4,32}
};

/* Type 6 - ROCKETS: 12 frames */
static const SpriteFrame frames_rockets[] = {
    {0,0},{32*4,0},{0,32},{32*4,32},
    {64*4,0},{(64+32)*4,0},{64*4,32},{(64+32)*4,32},
    {128*4,0},{(128+32)*4,0},{128*4,32},{(128+32)*4,32}
};

/* Type 7 - BARREL: 1 frame */
static const SpriteFrame frames_barrel[] = { {0,0} };

/* Type 8 - EXPLOSION: 9 frames (uses explosion.wad/ptr, not bigbullet) */
static const SpriteFrame frames_explosion[] = {
    {0,0},{64*4,0},{64*4*2,0},{64*4*3,0},{64*4*4,0},{64*4*5,0},{64*4*6,0},{64*4*7,0},{64*4*8,0}
};

/* Type 9 - GUNS (in-hand): handled by gun renderer, but provide frames */
static const SpriteFrame frames_guns[] = {
    {96*4*20,0},{96*4*21,0},{96*4*22,0},{96*4*23,0},
    {96*4*4,0},{96*4*5,0},{96*4*6,0},{96*4*7,0},
    {96*4*16,0},{96*4*17,0},{96*4*18,0},{96*4*19,0},
    {96*4*12,0},{96*4*13,0},{96*4*14,0},{96*4*15,0},
    {96*4*24,0},{96*4*25,0},{96*4*26,0},{96*4*27,0},
    {0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
    {96*4*0,0},{96*4*1,0},{96*4*2,0},{96*4*3,0}
};

/* Type 10 - MARINE: 19 frames, 64 cols each */
static const SpriteFrame frames_marine[] = {
    {0,0},{64*4,0},{(64*2)*4,0},{(64*3)*4,0},{(64*4)*4,0},{(64*5)*4,0},
    {(64*6)*4,0},{(64*7)*4,0},{(64*8)*4,0},{(64*9)*4,0},{(64*10)*4,0},
    {(64*11)*4,0},{(64*12)*4,0},{(64*13)*4,0},{(64*14)*4,0},{(64*15)*4,0},
    {(64*16)*4,0},{(64*17)*4,0},{(64*18)*4,0}
};

/* Type 11 - BIGALIEN: 4 frames, 128 cols each */
static const SpriteFrame frames_bigalien[] = {
    {0,0},{128*4,0},{128*4*2,0},{128*4*3,0}
};

/* Type 12 - LAMPS: 1 frame */
static const SpriteFrame frames_lamps[] = { {0,0} };

/* Type 13 - WORM: 21 frames, 90 cols each */
static const SpriteFrame frames_worm[] = {
    {0,0},{90*4,0},{90*4*2,0},{90*4*3,0},{90*4*4,0},{90*4*5,0},{90*4*6,0},{90*4*7,0},
    {90*4*8,0},{90*4*9,0},{90*4*10,0},{90*4*11,0},{90*4*12,0},{90*4*13,0},{90*4*14,0},{90*4*15,0},
    {90*4*16,0},{90*4*17,0},{90*4*18,0},{90*4*19,0},{90*4*20,0}
};

/* Type 14 - BIGCLAWS: 18 frames, 128 cols each */
static const SpriteFrame frames_bigclaws[] = {
    {0,0},{128*4,0},{128*4*2,0},{128*4*3,0},{128*4*4,0},{128*4*5,0},{128*4*6,0},{128*4*7,0},
    {128*4*8,0},{128*4*9,0},{128*4*10,0},{128*4*11,0},{128*4*12,0},{128*4*13,0},{128*4*14,0},{128*4*15,0},
    {128*4*16,0},{128*4*17,0}
};

/* Type 15 - TREE: 22 frames, 64 cols each (with repeats) */
static const SpriteFrame frames_tree[] = {
    {0,0},{64*4,0},{64*2*4,0},{64*3*4,0},
    {0,0},{64*4,0},{64*2*4,0},{64*3*4,0},
    {0,0},{64*4,0},{64*2*4,0},{64*3*4,0},
    {0,0},{64*4,0},{64*2*4,0},{64*3*4,0},
    {0,0},{0,0},
    {32*8*4,0},{32*9*4,0},{32*10*4,0},{32*11*4,0}
};

/* Master lookup: frames table + count per sprite type */
static const struct {
    const SpriteFrame *frames;
    int count;
} sprite_frames_table[MAX_SPRITE_TYPES] = {
    { frames_alien,     sizeof(frames_alien)/sizeof(SpriteFrame) },       /* 0 */
    { frames_pickups,   sizeof(frames_pickups)/sizeof(SpriteFrame) },     /* 1 */
    { frames_bigbullet, sizeof(frames_bigbullet)/sizeof(SpriteFrame) },   /* 2 */
    { NULL, 0 },                                                          /* 3 ugly monster */
    { frames_flying,    sizeof(frames_flying)/sizeof(SpriteFrame) },      /* 4 */
    { frames_keys,      sizeof(frames_keys)/sizeof(SpriteFrame) },        /* 5 */
    { frames_rockets,   sizeof(frames_rockets)/sizeof(SpriteFrame) },     /* 6 */
    { frames_barrel,    sizeof(frames_barrel)/sizeof(SpriteFrame) },      /* 7 */
    { frames_explosion, sizeof(frames_explosion)/sizeof(SpriteFrame) },   /* 8 */
    { frames_guns,      sizeof(frames_guns)/sizeof(SpriteFrame) },        /* 9 */
    { frames_marine,    sizeof(frames_marine)/sizeof(SpriteFrame) },      /* 10 */
    { frames_bigalien,  sizeof(frames_bigalien)/sizeof(SpriteFrame) },    /* 11 */
    { frames_lamps,     sizeof(frames_lamps)/sizeof(SpriteFrame) },       /* 12 */
    { frames_worm,      sizeof(frames_worm)/sizeof(SpriteFrame) },        /* 13 */
    { frames_bigclaws,  sizeof(frames_bigclaws)/sizeof(SpriteFrame) },    /* 14 */
    { frames_tree,      sizeof(frames_tree)/sizeof(SpriteFrame) },        /* 15 */
    { frames_marine,    sizeof(frames_marine)/sizeof(SpriteFrame) },      /* 16 tough (same as 10) */
    { frames_marine,    sizeof(frames_marine)/sizeof(SpriteFrame) },      /* 17 flame (same as 10) */
    { NULL, 0 }, { NULL, 0 }
};

#endif
