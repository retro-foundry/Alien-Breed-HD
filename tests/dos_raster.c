/* Compare the DOS kernels with the existing WallRoutine3/AB3DI expressions. */
#include "raster_dos.h"
#include "renderer_fixed.h"
#include "objects_dos.h"
#include "sprite_palettes.h"
#include "level.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reference contains the original big-endian .pal bytes; the linked
 * sprite_palettes.c contains the actual native data used by the game. */
#define sprite_pal_embedded sprite_pal_source
#define sprite_pal_embedded_size sprite_pal_source_size
#include "sprite_palettes_source.h"
#undef sprite_pal_embedded
#undef sprite_pal_embedded_size

static uint32_t random_state = 17;
static uint32_t next_value(void)
{
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
}

__attribute__((noinline)) static int64_t reference_divide(int64_t n, int64_t d) { return n / d; }

static void check_division(void)
{
    for (int64_t n = -65537; n <= 65537; n += 127) {
        static const int offsets[] = {-8388609, -8388608, -1, 0, 1, 8388607, 8388608};
        for (unsigned i = 0; i < sizeof(offsets) / sizeof(*offsets); i++) {
            int64_t value = n * 16777216 + offsets[i];
            int32_t rounded = (int32_t)((value < 0 ? value - 8388608 : value + 8388608) / 16777216);
            assert(renderer_project_round_24(value) == rounded);
        }
    }
    static const int64_t numerators[] = {INT64_MIN, INT64_MIN + 1, -4294967297LL,
        INT32_MIN, -1, 0, 1, INT32_MAX, 4294967297LL, INT64_MAX};
    static const int32_t divisors[] = {INT32_MIN, -16777216, -65536, -255, -3, -1,
        1, 3, 255, 65536, 16777216, INT32_MAX};
    unsigned checked = 0;
    for (unsigned n = 0; n < sizeof(numerators) / sizeof(*numerators); n++) {
        for (unsigned d = 0; d < sizeof(divisors) / sizeof(*divisors); d++) {
            if (numerators[n] == INT64_MIN && divisors[d] == -1) continue;
            assert(renderer_divide_i64_i32(numerators[n], divisors[d]) == reference_divide(numerators[n], divisors[d]));
            checked++;
        }
    }
    for (unsigned i = 0; i < 4096; i++) {
        int64_t n = (int64_t)(((uint64_t)next_value() << 32) | next_value());
        int32_t d = (int32_t)(next_value() | 1u);
        assert(renderer_divide_i64_i32(n, d) == reference_divide(n, d));
        checked++;
    }
    printf("PASS: %u signed 64/32 divisions, including wide quotients\n", checked);
}

static void check_palette(void)
{
    uint8_t palette[8192];
    for (unsigned c = 0; c < 4096; c++) { palette[c * 2] = c >> 8; palette[c * 2 + 1] = c; }
    renderer_pixel_prepare_palette(palette, sizeof(palette));
    for (unsigned c = 0; c < 4096; c++)
        assert(renderer_pixel_to_c12(renderer_pixel_from_palette(palette + c * 2)) == c);
    uint8_t packaged[8192];
    FILE *file=fopen("COLORS.BIN","rb");
    assert(file);
    size_t bytes=fread(packaged,1,sizeof(packaged),file);
    int closed=fclose(file);
    assert(bytes==sizeof(packaged) && !closed && !memcmp(packaged,palette,sizeof(palette)));
    puts("PASS: all 4096 prepared palette colors retain their source values");
}

static void check_water_projection(void)
{
    static const int32_t depths[] = {1, 2, 3, 127, 255, 256, 257, 65535, 65536, INT32_MAX};
    static const int32_t heights[] = {INT32_MIN, -65537, -257, -1, 0, 1, 257, 65537, INT32_MAX};
    unsigned checked = 0;
    RendererRoundCache cache = {0};
    for (unsigned y = 0; y < sizeof(heights) / sizeof(*heights); y++) {
        for (unsigned z = 0; z < sizeof(depths) / sizeof(*depths); z++) {
            for (int scale = 1; scale <= 2048; scale *= 2) {
                int64_t num = (int64_t)heights[y] * scale * 8;
                int64_t expanded = num * 256, den = (int64_t)depths[z] * 256;
                int64_t expected = reference_divide(expanded >= 0 ? expanded + den / 2 : expanded - den / 2, den);
                assert(renderer_divide_round_i64_i32(num, depths[z]) == expected);
                assert(renderer_divide_round_cached(&cache, num, depths[z]) == (int32_t)expected);
                assert(renderer_divide_round_cached(&cache, num, depths[z]) == (int32_t)expected);
                int64_t changed = num + (int64_t)depths[z] * 3;
                int64_t changed_expected = reference_divide(changed >= 0 ? changed + depths[z] / 2 : changed - depths[z] / 2, depths[z]);
                assert(renderer_divide_round_cached(&cache, changed, depths[z]) == (int32_t)changed_expected);
                assert(renderer_divide_round_cached(&cache, num, depths[z]) == (int32_t)expected);
                if (depths[z] < INT32_MAX) {
                    int32_t new_depth = depths[z] + 1;
                    int64_t new_expected = reference_divide(num >= 0 ? num + new_depth / 2 : num - new_depth / 2, new_depth);
                    assert(renderer_divide_round_cached(&cache, num, new_depth) == (int32_t)new_expected);
                }
                checked++;
            }
        }
    }
    for (int sine = -65536; sine <= 65536; sine += 127) {
        for (int distance = 16; distance <= 32000; distance += 997) {
            int32_t expected = reference_divide((int64_t)sine * 256, (int64_t)(distance + 300) * 64);
            assert(renderer_water_vertical_refraction(sine, distance) == expected);
            checked++;
        }
    }
    printf("PASS: %u water projection and refraction cases match expanded fixed-point expressions\n", checked);
}

static void check_floors(void)
{
    uint8_t *texture = malloc(65536);
    assert(texture);
    for (unsigned i = 0; i < 65536; i++) texture[i] = next_value() >> 24;
    RendererPixel palettes[15][256], actual[322], expected[322];
    for (unsigned l = 0; l < 15; l++) {
        for (unsigned t = 0; t < 256; t++) palettes[l][t] = renderer_pixel_from_c12(l * 256 + t);
    }
    for (unsigned c = 0; c < 4096; c++) {
        unsigned count = c % 321;
        uint32_t u = next_value(), v = next_value(), du = next_value(), dv = next_value();
        if (c % 3 == 0) du = 0;
        if (c % 3 == 1) dv = 0;
        /* Dense sub-texel steps exercise exact positive carries and negative
         * borrows, zero steps, both axes and all unrolled tails. */
        static const uint32_t small_steps[]={0,1,8192,16384,32767,32768,32769,65535,
            65536,0u-1,0u-8192,0u-16384,0u-32767,0u-32768,0u-32769,0u-65535,
            123,4095,4096,4097,8191,8193,0u-123,0u-4095,
            0u-4096,0u-4097,0u-8191,0u-8193,17,0u-17,7777,0u-7777};
        if(c%6==0)dv=small_steps[(c/6)%32];
        if(c%6==1)du=small_steps[(c/6)%32];
        int32_t left = c % 15, right = (c / 15) % 15;
        int32_t level = left * 65536;
        int32_t dlevel = count > 1 ? (right - left) * 65536 / (int32_t)(count - 1) : 0;
        memset(actual, 0x91, sizeof(actual)); memcpy(expected, actual, sizeof(actual));
        uint32_t a = u, b = v;
        int32_t shade = level;
        for (unsigned x = 0; x < count; x++) {
            /* AB3DI d5.w = (V&63)*256 + (U&63), then sample d5.w*4. */
            unsigned offset = (((b >> 16) & 63) * 256 + ((a >> 16) & 63)) * 4;
            expected[x + 1] = palettes[shade >> 16][texture[offset]];
            a += du; b += dv; shade += dlevel;
        }
        dos_raster_floor_gouraud(actual + 1, count, texture, palettes, u, v, du, dv, level, dlevel);
        assert(!memcmp(actual, expected, sizeof(actual)));
        for (unsigned x = 0; x < count; x++) {
            unsigned offset = (((v >> 16) & 63) * 256 + ((u >> 16) & 63)) * 4;
            expected[x + 1] = palettes[left][texture[offset]];
            u += du; v += dv;
        }
        /* Reconstruct the original accumulators after the reference walk. */
        u -= du * count; v -= dv * count;
        dos_raster_floor(actual + 1, count, texture, palettes[left], u, v, du, dv);
        assert(!memcmp(actual, expected, sizeof(actual)));
    }
    free(texture);
    puts("PASS: 8192 floor spans retain UV wrapping, shade gradients and bounds");
}

static void check_sprite_palettes(void)
{
    unsigned checked=0;
    for(unsigned p=0;p<MAX_SPRITE_TYPES;p++) {
        assert(sprite_pal_embedded_size[p]==sprite_pal_source_size[p]);
        assert((sprite_pal_embedded[p]!=NULL)==(sprite_pal_source[p]!=NULL));
        for(size_t i=0;i+1<sprite_pal_source_size[p];i+=2) {
            unsigned source=(sprite_pal_source[p][i]<<8)|sprite_pal_source[p][i+1];
            uint16_t pixel=renderer_pixel_from_palette(sprite_pal_embedded[p]+i);
            assert(pixel==renderer_pixel_from_c12(source));
            assert(renderer_pixel_to_c12(pixel)==(source&4095));
            checked++;
        }
    }
    printf("PASS: %u embedded sprite shade colors match the original palettes\n",checked);
}

static void check_edges(void)
{
    int16_t actual[4][320], expected[4][320];
    for (unsigned count = 0; count <= 311; count++) {
        for (unsigned offset = 0; offset < 4; offset++) {
            memset(actual, 0x91, sizeof(actual)); memcpy(expected, actual, sizeof(actual));
            for (unsigned i = offset; i < offset + count; i++) {
                expected[0][i] = 320; expected[1][i] = -1;
            }
            dos_raster_floor_edges_init(actual[0] + offset, actual[1] + offset,
                                         count, 320);
            assert(!memcmp(actual, expected, sizeof(actual)));
        }
    }
    for (unsigned c = 0; c < 4096; c++) {
        unsigned count = c % 321;
        int64_t x = (int32_t)(next_value() & 0x3fffffff) - 0x20000000;
        int64_t dx = (int32_t)(next_value() % 200001) - 100000;
        if (c % 8 == 0) { x = INT32_MAX; dx = 0; }
        if (c % 8 == 1) { x = INT32_MIN; dx = 0; }
        int64_t bright = (int32_t)next_value(), db = (int32_t)next_value();
        for (unsigned i = 0; i < 320; i++) {
            actual[0][i] = 320; actual[1][i] = -1;
            actual[2][i] = actual[3][i] = -99;
        }
        memcpy(expected, actual, sizeof(actual));
        int64_t a = x, b = bright;
        for (unsigned i = 0; i < count; i++) {
            int64_t left = a >> 16, right = a <= 0 ? left : (a + 65535) >> 16;
            if (left < expected[0][i]) { expected[0][i] = left; expected[2][i] = b >> 16; }
            if (right > expected[1][i]) { expected[1][i] = right > INT16_MAX ? INT16_MAX : right; expected[3][i] = b >> 16; }
            a += dx; b += db;
        }
        dos_raster_floor_edge(actual[0], actual[1], actual[2], actual[3], count, x, dx, bright, db);
        assert(!memcmp(actual, expected, sizeof(actual)));
    }
    puts("PASS: 4096 floor edges match the wide fixed-point walk");
}

static void check_walls(void)
{
    FILE *file=fopen("WALL.BIN","rb");
    assert(file && !fseek(file,0,SEEK_END));
    long bytes=ftell(file);
    assert(bytes>0 && !fseek(file,0,SEEK_SET));
    uint8_t *data = malloc((size_t)bytes);
    assert(data);
    assert(fread(data,1,(size_t)bytes,file)==(size_t)bytes && !fclose(file));
    size_t source_bytes;
    uint8_t *native_palette=dos_load_wall(data,(size_t)bytes,&source_bytes);
    assert(source_bytes==2048+65536*2);
    for(unsigned i=0;i<1024;i++)
        assert(renderer_pixel_from_palette(native_palette+i*2)==renderer_pixel_from_c12((i*37)&4095));
    const uint8_t *texels=native_palette+DOS_WALL_HEADER_BYTES;
    for (unsigned w = 0; w < 65536; w++)
        for (unsigned c = 0; c < 3; c++) assert(texels[c * 65536 + w] == ((w >> (c * 5)) & 31));
    RendererPixel palettes[4][32];
    static RendererPixel actual[320 * 160], expected[320 * 160];
    for (unsigned l = 0; l < 4; l++)
        for (unsigned t = 0; t < 32; t++) palettes[l][t] = renderer_pixel_from_c12((t * 97 + l * 73) & 4095);
    const RendererPixel *pal = palettes[0];
    for (unsigned c = 0; c < 96; c++) {
        unsigned rows = 1u << (c % 5 + 4), mask = rows - 1;
        uint32_t y = next_value(), dy = c % 2 ? next_value() : 8192;
        if(c%8==0)dy=0;
        if(c%8==2)dy=32768;
        if(c%8==4)dy=32769;
        if(c%8==6)dy=65535;
        if(c%9==0)mask=0;
        if(c%11==0)mask &= 0x55;
        unsigned width = 1 + c % 33;
        DosWallColumn columns[33];
        for (unsigned col = 0; col < 33; col++) {
            columns[col].texels = dos_wall_column(texels,col,rows);
            columns[col].palette = palettes[c % 3 ? 0 : col % 4];
        }
        memset(actual, 0x91, sizeof(actual)); memcpy(expected, actual, sizeof(actual));
        uint32_t a = y;
        for (unsigned row = 0; row < 150; row++) {
            unsigned r = (a >> 16) & mask;
            for (unsigned col = 0; col < width; col++) {
                unsigned word = ((col / 3) * rows + r) & 65535;
                expected[row * 40 + col] = columns[col].palette[(word >> ((col % 3) * 5)) & 31];
            }
            a += dy;
        }
        dos_raster_wall_rows(actual, width, 150, 40, columns, y, dy, mask);
        assert(!memcmp(actual, expected, sizeof(actual)));
        memset(actual, 0x91, sizeof(actual)); memset(expected, 0x91, sizeof(expected));
        a = y;
        unsigned count = c * 23 % 160, stride = c % 3 == 2 ? 40 : 320, base_word=c*17;
        for (unsigned row = 0; row < count; row++) {
            unsigned word = ((a >> 16) & mask)+base_word;
            expected[row * stride] = pal[word & 31];
            a += dy;
        }
        dos_raster_wall(actual, count, stride, texels+base_word, pal, y, dy, mask);
        assert(!memcmp(actual, expected, sizeof(actual)));
    }
    free(data);
    puts("PASS: all packed wall words and 192 row/column rasters preserve samples");
}

static void put_word(uint8_t *p, int value) { p[0] = (unsigned)value >> 8; p[1] = value; }
static int get_word(const uint8_t *p) { return (int16_t)((p[0] << 8) | p[1]); }

static void check_objects(void)
{
    static LevelState level;
    static uint8_t objects[256 * OBJECT_SIZE], point_data[256 * 8];
    static uint8_t nasty[NASTY_SHOT_SLOT_COUNT * OBJECT_SIZE], player[PLAYER_SHOT_SLOT_COUNT * OBJECT_SIZE];
    static ObjRotatedPoint points[256];
    const int near_z = ROT_Z_FROM_INT(50);
    level.object_data = objects; level.object_points = point_data; level.num_object_points = 256;
    level.nasty_shot_data = nasty; level.player_shot_data = player; level.draw_group_count = 64;
    for (unsigned z = 0; z < 64; z++) level.draw_groups[z] = z / 2;
    for (unsigned frame = 0; frame < 32; frame++) {
        unsigned count = frame == 31 ? 0 : (frame * 43) % 257;
        memset(objects, 0, sizeof(objects));
        for (unsigned i = 0; i < 256; i++) {
            uint8_t *obj = objects + i * OBJECT_SIZE;
            put_word(obj, i % 27 ? i : 300);
            put_word(obj + 12, i % 29 ? (i + frame) % 64 : -1);
            if (frame % 8 == 7 && i == 5) put_word(obj + 12, 300);
            obj[6] = i % 11 ? 32 : OBJ_3D_SPRITE;
            points[i].z = ROT_Z_FROM_INT((int)((i * 7 + frame) % 151) - 25);
        }
        if (count < 256) put_word(objects + count * OBJECT_SIZE, -1);
        for (unsigned pool = 0; pool < 2; pool++) {
            uint8_t *data = pool ? player : nasty;
            unsigned slots = pool ? PLAYER_SHOT_SLOT_COUNT : NASTY_SHOT_SLOT_COUNT;
            for (unsigned i = 0; i < slots; i++) {
                uint8_t *shot = data + i * OBJECT_SIZE;
                put_word(shot, (i * 3) % 256);
                put_word(shot + 12, i % 7 ? i % 64 : -1);
                shot[16] = i % 5 ? OBJ_NBR_BULLET : OBJ_NBR_ALIEN;
            }
        }
        dos_objects_prepare(&level, points, near_z);
        int indexed = 1;
        for (unsigned i = 0; i < count; i++) {
            const uint8_t *obj = objects + i * OBJECT_SIZE;
            int p = get_word(obj), z = get_word(obj + 12);
            if ((unsigned)p >= 256 || z < 0 || points[p].z <= (obj[6] == OBJ_3D_SPRITE ? 0 : near_z)) continue;
            if (z >= 256) indexed = 0;
        }
        for (int zone = 0; zone < 64; zone++) {
            int16_t adjacent[] = {zone - 1, zone + 1, zone + 2, -1};
            uint8_t actual[256], expected[256];
            unsigned total = 0;
            for (unsigned i = 0; i < count; i++) {
                const uint8_t *obj = objects + i * OBJECT_SIZE;
                int p = get_word(obj), z = get_word(obj + 12);
                int polygon = obj[6] == OBJ_3D_SPRITE;
                if ((unsigned)p >= 256 || z < 0 || points[p].z <= (polygon ? 0 : near_z)) continue;
                int connected = 0;
                for (unsigned a = 0; a < 4; a++) if (adjacent[a] == z) connected = 1;
                if (indexed && !level_same_draw_room(&level, z, zone) && (polygon || !connected)) continue;
                expected[total++] = i;
            }
            unsigned found = dos_objects_for_room(&level, zone, adjacent, 4, actual);
            assert(found == total && !memcmp(actual, expected, total));
        }
        for (unsigned pool = 0; pool < 2; pool++) {
            const uint8_t *data = pool ? player : nasty;
            unsigned slots = pool ? PLAYER_SHOT_SLOT_COUNT : NASTY_SHOT_SLOT_COUNT;
            uint8_t expected[NASTY_SHOT_SLOT_COUNT + PLAYER_SHOT_SLOT_COUNT];
            unsigned total = 0, found;
            for (unsigned i = 0; i < slots; i++) {
                const uint8_t *obj = data + i * OBJECT_SIZE;
                int point = get_word(obj);
                if (get_word(obj + 12) >= 0 && obj[16] == OBJ_NBR_BULLET &&
                    point >= 0 && point < 256 && points[point].z > near_z) expected[total++] = i;
            }
            const uint8_t *actual = dos_objects_shots(pool, &found);
            assert(found == total && !memcmp(actual, expected, total));
        }
    }
    level.num_object_points = -1;
    dos_objects_prepare(&level, points, near_z);
    uint8_t empty[256];
    assert(dos_objects_for_room(&level, 0, NULL, 0, empty) == 0);
    for (unsigned pool = 0; pool < 2; pool++) {
        unsigned total;
        dos_objects_shots(pool, &total);
        assert(total == 0);
    }
    puts("PASS: 2048 room candidate lists and both shot pools retain ObjDraw slot ordering and rejects");
}

int main(void)
{
    check_division(); check_palette(); check_water_projection(); check_sprite_palettes(); check_floors(); check_edges(); check_walls();
    check_objects();
    puts("DOS raster regression passed");
    return 0;
}
