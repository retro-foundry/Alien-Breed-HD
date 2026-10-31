/* Check every authored door face against its initial texture origin, then
 * replay opening/closing with Anims.s DoorRoutine's absolute Y scroll. */
#include "../src/objects.c"
#include "sb_decompress.h"
#include <assert.h>
#include <stdlib.h>

GameState g_state;
void audio_play_sample(int sample, int volume) { (void)sample; (void)volume; }

typedef struct {
    uint32_t gfx_offset;
    uint32_t initial_texture;
    int32_t initial_y;
    int door;
} DoorFace;

int main(void)
{
    int doors = 0, faces = 0, samples = 0, nonzero_tiles = 0, shifted_doors = 0;
    for (int map = 0; map < 16; map++) {
        static GameState state;
        memset(&state, 0, sizeof(state));
        state.plr1.zone = state.plr2.zone = -1;
        state.temp_frames = 1;
        char path[128]; size_t bytes;
        snprintf(path, sizeof(path), "amiga/data/levels/level_%c/twolev.bin", 'a' + map);
        assert(!sb_load_file(path, &state.level.data, &bytes)); state.level.data_byte_count = bytes;
        snprintf(path, sizeof(path), "amiga/data/levels/level_%c/twolev.graph.bin", 'a' + map);
        assert(!sb_load_file(path, &state.level.graphics, &bytes)); state.level.graphics_byte_count = bytes;
        assert(!level_parse(&state.level));
        LevelState *level = &state.level;
        assert(level->num_doors > 0);
        unsigned count = level->door_wall_list_offsets[level->num_doors];
        DoorFace *expected = calloc(count, sizeof(*expected));
        uint8_t *original_graphics = malloc(level->graphics_byte_count);
        uint8_t *original_list = malloc((size_t)count * 10);
        uint8_t *patched = calloc(level->graphics_byte_count, 1);
        assert(expected && original_graphics && original_list && patched);
        memcpy(original_graphics, level->graphics, level->graphics_byte_count);
        memcpy(original_list, level->door_wall_list, (size_t)count * 10);
        for (int di = 0; di < level->num_doors; di++) {
            uint8_t *door = level->door_data + di * 22;
            for (unsigned j = level->door_wall_list_offsets[di];
                 j < level->door_wall_list_offsets[di + 1]; j++) {
                DoorFace *face = &expected[j];
                face->door = di;
                face->gfx_offset = (uint32_t)be32(level->door_wall_list + j * 10 + 2);
                assert((size_t)face->gfx_offset + 30 <= level->graphics_byte_count);
                uint8_t *wall = level->graphics + face->gfx_offset;
                assert(be16(wall) == 0 || be16(wall) == 13);
                face->initial_texture = (uint32_t)be32(wall + 10);
                face->initial_y = be32(door + 4);
                if (face->initial_texture >> 16) nonzero_tiles++;
                if ((face->initial_y >> 8) & 255) shifted_doors++;
                memset(patched + face->gfx_offset + 10, 1, 4);
                memset(patched + face->gfx_offset + 24, 1, 4);
            }
        }
        /* Closed -> fully open -> closed, including partial openings. */
        for (int sample = 0; sample <= 32; sample++) {
            int progress = sample <= 16 ? sample : 32 - sample;
            for (int di = 0; di < level->num_doors; di++) {
                uint8_t *door = level->door_data + di * 22;
                int32_t top = be32(door + 10), closed = be32(door + 14);
                wbe32(door + 4, closed + (int32_t)(((int64_t)top - closed) * progress / 16));
                wbe16(door + 8, 0);
            }
            door_routine(&state);
            for (unsigned j = 0; j < count; j++) {
                const DoorFace *face = &expected[j];
                const uint8_t *wall = level->graphics + face->gfx_offset;
                int32_t current_y = be32(level->door_data + face->door * 22 + 4);
                /* Preserve the authored tile. Vertical displacement is in world
                 * texels and wraps over 256 rows before the wall's VALAND mask. */
                int32_t delta = (face->initial_y >> 8) - (current_y >> 8);
                uint16_t yoff = (uint16_t)(((face->initial_texture & 255u) + delta) & 255u);
                uint32_t texture = (face->initial_texture & 0xFFFF0000u) | yoff;
                uint32_t actual = (uint32_t)be32(wall + 10);
                if (actual != texture) {
                    fprintf(stderr, "Door alignment failed: level=%d zone=%d gfx=%u progress=%d expected=%08x actual=%08x\n",
                            map + 1, be16(level->door_data + face->door * 22),
                            face->gfx_offset, progress, (unsigned)texture, (unsigned)actual);
                    return 1;
                }
                assert(be32(wall + 24) == current_y);
                samples++;
            }
            /* Keep legacy wall-list/save bytes and every other graphics field. */
            assert(!memcmp(original_list, level->door_wall_list, (size_t)count * 10));
            for (size_t off = 0; off < level->graphics_byte_count; off++)
                if (!patched[off]) assert(level->graphics[off] == original_graphics[off]);
        }
        doors += level->num_doors; faces += count;
        free(expected); free(original_graphics); free(original_list); free(patched);
    }
    assert(faces == 245 && nonzero_tiles == 25 && shifted_doors > 0);
    printf("PASS: %d doors / %d faces across all 16 levels, %d opening/closing texture samples; %d tile offsets and %d nonzero-height faces\n",
           doors, faces, samples, nonzero_tiles, shifted_doors);
    return 0;
}
