#ifndef AB3D_AUDIO_SAMPLES_H
#define AB3D_AUDIO_SAMPLES_H
#define NUM_NAMED_SFX 28
/* Amiga LoadFromDisk.s SFX_NAMES: size in bytes (one byte per sample, 8-bit signed). */
static const unsigned int amiga_sfx_sizes[NUM_NAMED_SFX] = {
    4400, 7200, 5400, 4600, 3400, 8400, 8000, 4000, 8600, 6200,
    1200, 4000, 2200, 3000, 5600, 11600, 7200, 7400, 9200, 5000,
    4000, 8800, 9000, 1800, 3400, 1600, 11000, 8400
};
/* Amiga LoadFromDisk.s SFX_NAMES order: sample_id -> disk/sounds/<name> (we use sounds/<name>.wav or raw) */
static const char *const sfx_names[NUM_NAMED_SFX] = {
    "scream",      /* 0 */
    "fire!",       /* 1 ShootName */
    "munch",       /* 2 */
    "shoot.dm",    /* 3 PooGunName */
    "collect",     /* 4 */
    "newdoor",     /* 5 */
    "splash",      /* 6 BassName */
    "footstep3",   /* 7 StompName */
    "lowscream",   /* 8 */
    "baddiegun",   /* 9 */
    "switch",      /* 10 */
    "switch1",     /* 11 ReloadName */
    "noammo",      /* 12 */
    "splotch",     /* 13 */
    "splatpop",    /* 14 */
    "boom",        /* 15 */
    "newhiss",     /* 16 */
    "howl1",       /* 17 */
    "howl2",       /* 18 */
    "pant",        /* 19 */
    "whoosh",      /* 20 */
    "shotgun",     /* 21 ShotGunName */
    "flame",       /* 22 */
    "muffledfoot", /* 23 (Amiga: MuffledFoot) */
    "footclop",    /* 24 */
    "footclank",   /* 25 */
    "teleport",    /* 26 */
    "halfwormpain" /* 27 (Amiga: HALFWORMPAIN) */
};


#endif
