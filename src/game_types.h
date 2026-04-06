/*
 * Alien Breed 3D I - PC Port
 * game_types.h - Core data types translated from Defs.i / AB3DI.i
 *
 * All values and offsets derived from the original 68000 assembly source.
 */

#ifndef GAME_TYPES_H
#define GAME_TYPES_H

#include <stdint.h>
#include <stdbool.h>

/* -----------------------------------------------------------------------
 * Fixed-point helpers
 * The Amiga code uses 16.16 and other fixed-point representations.
 * We use int32_t for 16.16 fixed point, int16_t for angles, etc.
 * ----------------------------------------------------------------------- */
typedef int32_t fixed16_t;  /* 16.16 fixed point */

#define FIXED_SHIFT     16
#define FIXED_ONE       (1 << FIXED_SHIFT)
#define INT_TO_FIXED(x) ((fixed16_t)(x) << FIXED_SHIFT)
#define FIXED_TO_INT(x) ((x) >> FIXED_SHIFT)

/* -----------------------------------------------------------------------
 * Screen / rendering constants (from AB3DI.i)
 * ----------------------------------------------------------------------- */
#define MAX_SCR_DIV     8
#define MAX_3D_DIV      5

#define SCR_WIDTH       104     /* Amiga chunky width in longwords */
#define SCR_HEIGHT      80      /* Amiga screen height in lines */

#define FROM_PT_OFFSET  10
#define WIDTH_OFFSET    (104 * 4)
#define MID_OFFSET      (104 * 4 * 40)

#define PLAYER_HEIGHT       (12 * 1024)
#define PLAYER_CROUCHED     (8 * 1024)
#define PLAYER_MAX_ENERGY   127

/* -----------------------------------------------------------------------
 * Memory block sizes (from AB3DI.i)
 * ----------------------------------------------------------------------- */
#define TEXT_SCR_SIZE        (10240 * 4)
#define LEVEL_DATA_SIZE      120000
#define LEVEL_GRAPHICS_SIZE  50000
#define LEVEL_CLIPS_SIZE     40000
#define TITLE_SCR_ADDR_SIZE  (10240 * 7)
#define OPT_SPR_ADDR_SIZE    (258 * 16 * 5)
#define PANEL_SIZE           30720
#define FLOOR_TILE_SIZE      65536

/* -----------------------------------------------------------------------
 * Object constants (from Defs.i)
 * ----------------------------------------------------------------------- */
#define OBJECT_SIZE        64  /* bytes per object */
/* ObjDraw3.ChipRam.s: cmp.b #$ff,6(a0); bne BitMapObj; bsr PolygonObj.
 * When byte at offset 6 is OBJ_3D_SPRITE the object is a 3D polygon sprite; else billboard. */
#define OBJ_3D_SPRITE      0xFF

/* Object type numbers (stored at offset 16 in object) */
typedef enum {
    OBJ_NBR_DEAD            = -1,
    OBJ_NBR_ALIEN           = 0,
    OBJ_NBR_MEDIKIT         = 1,
    OBJ_NBR_BULLET          = 2,
    OBJ_NBR_BIG_GUN         = 3,
    OBJ_NBR_KEY             = 4,
    OBJ_NBR_PLR1            = 5,  /* Marine */
    OBJ_NBR_ROBOT           = 6,
    OBJ_NBR_BIG_NASTY       = 7,
    OBJ_NBR_FLYING_NASTY    = 8,
    OBJ_NBR_AMMO            = 9,
    OBJ_NBR_BARREL          = 10,
    OBJ_NBR_PLR2            = 11,
    OBJ_NBR_MARINE          = 12,
    OBJ_NBR_WORM            = 13,
    OBJ_NBR_HUGE_RED_THING  = 14,
    OBJ_NBR_SMALL_RED_THING = 15,
    OBJ_NBR_TREE            = 16,
    OBJ_NBR_EYEBALL         = 17,
    OBJ_NBR_TOUGH_MARINE    = 18,
    OBJ_NBR_FLAME_MARINE    = 19,
    OBJ_NBR_GAS_PIPE        = 20,
} ObjNumber;

/* -----------------------------------------------------------------------
 * Generic object structure (64 bytes, from Defs.i)
 *
 * Data is stored in Amiga big-endian format. All multi-byte field access
 * MUST go through the OBJ_* accessor macros below. Single-byte fields
 * (number, can_see, worry, in_top) can be accessed directly via .obj.
 *
 * The struct inside the union is kept for reference/documentation of the
 * layout, but direct access to its multi-byte fields is INCORRECT on
 * little-endian platforms. Always use the OBJ_* macros instead.
 * ----------------------------------------------------------------------- */
typedef union {
    uint8_t raw[64];

    struct {
        /* offset 0 */  int16_t  collision_id;      /* .w - USE OBJ_CID() */
        /* offset 2 */  int16_t  unknown2;          /* .w */
        /* offset 4 */  int16_t  unknown4;          /* .w */
        /* offset 6 */  int8_t   width_or_3d;      /* .b: world_w, or OBJ_3D_SPRITE if 3D polygon */
        /* offset 7 */  int8_t   world_height;     /* .b: world_h (when billboard) */
        /* offset 8 */  int16_t  dead_frame_h;      /* .w - USE OBJ_DEADH() */
        /* offset 10 */ int16_t  dead_frame_l;      /* .w - USE OBJ_DEADL() */
        /* offset 12 */ int16_t  zone;              /* .w - USE OBJ_ZONE() */
        /* offset 14 */ uint8_t  src_cols;          /* .b - sprite source columns (ObjDraw3 BitMapObj) */
        /* offset 15 */ uint8_t  src_rows;          /* .b - sprite source rows (ObjDraw3 BitMapObj) */
        /* offset 16 */ int8_t   number;            /* .b - direct access OK */
        /* offset 17 */ int8_t   can_see;           /* .b - direct access OK - visibility or keys */
        /* offset 18 */ uint8_t  type_data[44];     /* type-dependent fields; GraphicRoom at raw+26 */
        /* offset 62 */ int8_t   worry;             /* .b - direct access OK */
        /* offset 63 */ int8_t   in_top;            /* .b - direct access OK */
    } obj;
} GameObject;

_Static_assert(sizeof(GameObject) == 64, "GameObject must be 64 bytes");

/* -----------------------------------------------------------------------
 * Big-endian object field accessors
 *
 * All Amiga data is big-endian. These macros read/write multi-byte
 * fields correctly on any platform.
 * ----------------------------------------------------------------------- */

/* Generic BE read/write on raw byte pointer */
static inline int16_t obj_w(const uint8_t *p) {
    return (int16_t)((p[0] << 8) | p[1]);
}
static inline void obj_sw(uint8_t *p, int16_t v) {
    p[0] = (uint8_t)((uint16_t)v >> 8);
    p[1] = (uint8_t)v;
}
static inline int32_t obj_l(const uint8_t *p) {
    return (int32_t)(((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                     ((uint32_t)p[2] << 8) | (uint32_t)p[3]);
}
static inline void obj_sl(uint8_t *p, int32_t v) {
    p[0] = (uint8_t)((uint32_t)v >> 24);
    p[1] = (uint8_t)((uint32_t)v >> 16);
    p[2] = (uint8_t)((uint32_t)v >> 8);
    p[3] = (uint8_t)v;
}

/* Common header fields - READ (p = GameObject*) */
#define OBJ_CID(p)       obj_w((p)->raw + 0)   /* collision_id */
#define OBJ_ZONE(p)      obj_w((p)->raw + 12)   /* zone */
#define OBJ_GROOM(p)     obj_w((p)->raw + 26)   /* graphic_room */
#define OBJ_DEADH(p)     obj_w((p)->raw + 8)    /* dead_frame_h */
#define OBJ_DEADL(p)     obj_w((p)->raw + 10)   /* dead_frame_l */

/* Common header fields - WRITE (p = GameObject*, v = value) */
#define OBJ_SET_CID(p, v)    obj_sw((p)->raw + 0, (int16_t)(v))
#define OBJ_SET_ZONE(p, v)   obj_sw((p)->raw + 12, (int16_t)(v))
#define OBJ_SET_GROOM(p, v)  obj_sw((p)->raw + 26, (int16_t)(v))
#define OBJ_SET_DEADH(p, v)  obj_sw((p)->raw + 8, (int16_t)(v))
#define OBJ_SET_DEADL(p, v)  obj_sw((p)->raw + 10, (int16_t)(v))

/* type_data field access (td_off = offset within type_data, i.e. from byte 18) */
#define OBJ_TD_W(p, td_off)         obj_w((p)->raw + 18 + (td_off))
#define OBJ_TD_L(p, td_off)         obj_l((p)->raw + 18 + (td_off))
#define OBJ_SET_TD_W(p, td_off, v)  obj_sw((p)->raw + 18 + (td_off), (int16_t)(v))
#define OBJ_SET_TD_L(p, td_off, v)  obj_sl((p)->raw + 18 + (td_off), (int32_t)(v))

/* -----------------------------------------------------------------------
 * Shot field accessors (td_off relative to byte 18)
 *
 * READ: SHOT_XVEL(o) where o is a GameObject value/ref
 * WRITE: SHOT_SET_XVEL(o, v)
 * ----------------------------------------------------------------------- */
#define SHOT_XVEL(o)      obj_l((o).raw + 18)   /* td[0]  abs 18 */
#define SHOT_ZVEL(o)      obj_l((o).raw + 22)   /* td[4]  abs 22 */
#define SHOT_POWER(o)     (*(int8_t*)&(o).raw[28])  /* td[10] byte - OK */
#define SHOT_ANIM(o)      (*(uint8_t*)&(o).raw[52]) /* td[34] byte: shotanim counter (Defs.i shotanim) */
#define SHOT_STATUS(o)    (*(int8_t*)&(o).raw[30])  /* td[12] byte - OK */
#define SHOT_SIZE(o)      (*(int8_t*)&(o).raw[31])  /* td[13] byte - OK */
#define SHOT_YVEL(o)      obj_w((o).raw + 42)   /* td[24] abs 42 */
#define SHOT_ACCYPOS(o)   obj_l((o).raw + 44)   /* td[26] abs 44 */
#define SHOT_GRAV(o)      obj_w((o).raw + 54)   /* td[36] abs 54 */
#define SHOT_LIFE(o)      obj_w((o).raw + 58)   /* td[40] abs 58 */
#define SHOT_FLAGS(o)     obj_w((o).raw + 60)   /* td[42] abs 60 */

#define SHOT_SET_XVEL(o, v)    obj_sl((o).raw + 18, (int32_t)(v))
#define SHOT_SET_ZVEL(o, v)    obj_sl((o).raw + 22, (int32_t)(v))
#define SHOT_SET_YVEL(o, v)    obj_sw((o).raw + 42, (int16_t)(v))
#define SHOT_SET_ACCYPOS(o, v) obj_sl((o).raw + 44, (int32_t)(v))
#define SHOT_SET_GRAV(o, v)    obj_sw((o).raw + 54, (int16_t)(v))
#define SHOT_SET_LIFE(o, v)    obj_sw((o).raw + 58, (int16_t)(v))
#define SHOT_SET_FLAGS(o, v)   obj_sw((o).raw + 60, (int16_t)(v))

/* -----------------------------------------------------------------------
 * Nasty/enemy field accessors
 * ----------------------------------------------------------------------- */
#define NASTY_LIVES(o)    (*(int8_t*)&(o).raw[18])   /* td[0]  byte - OK */
#define NASTY_DAMAGE(o)   (*(int8_t*)&(o).raw[19])   /* td[1]  byte - OK */
#define NASTY_MAXSPD(o)   obj_w((o).raw + 20)   /* td[2]  abs 20 */
#define NASTY_CURRSPD(o)  obj_w((o).raw + 22)   /* td[4]  abs 22 */
#define NASTY_FACING(o)   obj_w((o).raw + 30)   /* td[12] abs 30 */
#define NASTY_TIMER(o)    obj_w((o).raw + 34)   /* td[16] abs 34 */
#define NASTY_EFLAGS(o)   obj_l((o).raw + 36)   /* td[18] abs 36 */
#define NASTY_IMPACTX(o)  obj_w((o).raw + 42)   /* td[24] abs 42 */
#define NASTY_IMPACTZ(o)  obj_w((o).raw + 44)   /* td[26] abs 44 */

#define NASTY_SET_DAMAGE(p, v)   (*(int8_t*)&(p)->raw[19] = (int8_t)(v))
#define NASTY_SET_MAXSPD(o, v)   obj_sw((o).raw + 20, (int16_t)(v))
#define NASTY_SET_CURRSPD(o, v)  obj_sw((o).raw + 22, (int16_t)(v))
#define NASTY_SET_FACING(o, v)   obj_sw((o).raw + 30, (int16_t)(v))
#define NASTY_SET_TIMER(o, v)    obj_sw((o).raw + 34, (int16_t)(v))
#define NASTY_SET_EFLAGS(o, v)   obj_sl((o).raw + 36, (int32_t)(v))
#define NASTY_SET_IMPACTX(p, v)  obj_sw((p)->raw + 42, (int16_t)(v))
#define NASTY_SET_IMPACTZ(p, v)  obj_sw((p)->raw + 44, (int16_t)(v))

/* -----------------------------------------------------------------------
 * Door definitions (from Defs.i)
 * ----------------------------------------------------------------------- */
typedef enum {
    DR_PLR_SPC  = 0,
    DR_PLR      = 1,
    DR_BUL      = 2,
    DR_ALIEN    = 3,
    DR_TIMEOUT  = 4,
    DR_NEVER    = 5,
} DoorRule;

typedef enum {
    DL_TIMEOUT  = 0,
    DL_NEVER    = 1,
} DoorLock;

/* -----------------------------------------------------------------------
 * Zone data offsets (from Defs.i)
 * These are byte offsets into a zone record in level data.
 * ----------------------------------------------------------------------- */
#define ZONE_OFF_FLOOR          2
#define ZONE_OFF_ROOF           6
#define ZONE_OFF_UPPER_FLOOR    10
#define ZONE_OFF_UPPER_ROOF     14
#define ZONE_OFF_WATER          18
#define ZONE_OFF_BRIGHTNESS     22
#define ZONE_OFF_UPPER_BRIGHT   24
#define ZONE_OFF_CPT            26
#define ZONE_OFF_WALL_LIST      28
#define ZONE_OFF_EXIT_LIST      32
#define ZONE_OFF_PTS            34
#define ZONE_OFF_BACK           36
#define ZONE_OFF_TEL_ZONE       38
#define ZONE_OFF_TEL_X          40
#define ZONE_OFF_TEL_Z          42
#define ZONE_OFF_FLOOR_NOISE    44
#define ZONE_OFF_UPPER_FLOOR_NOISE 46
#define ZONE_OFF_LIST_OF_GRAPH  48

/* -----------------------------------------------------------------------
 * Gun data (each gun entry is 32 bytes)
 * ----------------------------------------------------------------------- */
#define GUN_DATA_SIZE   32
#define MAX_GUNS        8
#define MAX_LEVELS      16
#define PASSWORD_LENGTH 16
#define PASSWORD_STORAGE_SIZE (MAX_LEVELS * (PASSWORD_LENGTH + 1))

typedef struct {
    int16_t  ammo;              /* 0: Ammo left */
    int16_t  unknown2;          /* 2 */
    int16_t  unknown4;          /* 4 */
    int8_t   unknown6;          /* 6 */
    int8_t   visible;           /* 7: 0 or $ff - weapon visible/acquired */
    int16_t  unknown8;          /* 8 */
    int16_t  fire_rate;         /* 10 */
    uint8_t  _pad[20];          /* 12-31 */
} GunEntry;

_Static_assert(sizeof(GunEntry) == 32, "GunEntry must be 32 bytes");

/* -----------------------------------------------------------------------
 * Game mode enumeration (mors variable)
 *   'n' = single player
 *   'm' = master (multiplayer)
 *   's' = slave (multiplayer)
 *   'q' = quit/exit
 * ----------------------------------------------------------------------- */
typedef enum {
    MODE_SINGLE = 'n',
    MODE_MASTER = 'm',
    MODE_SLAVE  = 's',
    MODE_QUIT   = 'q',
} GameMode;

/* -----------------------------------------------------------------------
 * Input control mode
 * ----------------------------------------------------------------------- */
typedef struct {
    bool keys;
    bool path;
    bool mouse;
    bool joy;
    bool mouse_kbd;
} ControlMode;

/* -----------------------------------------------------------------------
 * Key control bindings (from ControlLoop.s CONTROLBUFFER)
 * Values are Amiga raw key codes; will be remapped for PC
 * ----------------------------------------------------------------------- */
typedef struct {
    uint8_t turn_left;
    uint8_t turn_right;
    uint8_t forward;
    uint8_t backward;
    uint8_t fire;
    uint8_t operate;
    uint8_t run;
    uint8_t force_sidestep;
    uint8_t sidestep_left;
    uint8_t sidestep_right;
    uint8_t duck;
    uint8_t look_behind;
} KeyBindings;

#endif /* GAME_TYPES_H */
