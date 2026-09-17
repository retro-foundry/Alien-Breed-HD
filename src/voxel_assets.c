#include "voxel_assets.h"
#include "sprite_frames.h"
#include "voxel_marine_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static VoxelModel models[MAX_SPRITE_TYPES][MAX_SPRITE_FRAMES];
static VoxelModel red_death_poses[10];
static uint8_t ready[MAX_SPRITE_TYPES][MAX_SPRITE_FRAMES];
static uint8_t directional_slot[MAX_SPRITE_TYPES];
static size_t bank_bytes;
static int bank_count;

static int load_authored_marine(VoxelModel *m, int pose, const VoxelObservation views[4])
{
    const MarineVoxelData *data = &marine_voxel_data[pose];
    for (int q = 0; q < 4; ++q) {
        uint32_t hash = 2166136261u;
        for (int p = 0; p < 64 * 64; ++p) hash = (hash ^ views[q].pixels[p]) * 16777619u;
        if (hash != data->source_hash[q]) {
            fprintf(stderr, "[VOXEL] Marine frame %d differs from its authored references; regenerate the model with tools/voxel_author.py (assets/voxels/marine.json).\n", views[q].source.frame);
            return 0;
        }
    }
    memset(m, 0, sizeof(*m));
    m->size_x = m->size_y = m->size_z = MARINE_VOXEL_GRID;
    m->run_count = data->count;
    /* Model data is immutable throughout rendering. Reference the baked
     * arrays directly, avoiding a second RAM copy and startup conversion. */
    m->borrowed_data = 1;
    m->columns = (uint32_t *)data->columns;
    m->runs = (VoxelRun *)data->runs;
    m->source_width = m->source_height = 64; m->source_views = 4;
    m->reconstruction.shape = VOXEL_AUTHORED_PARTS;
    m->reconstruction.max_axis = MARINE_VOXEL_GRID; m->reconstruction.sampling_scale = 1;
    memcpy(m->sources, data->sources, sizeof(m->sources));
    memcpy(m->source_iou, data->iou, sizeof(m->source_iou));
    memcpy(m->source_false_positive, data->fp, sizeof(m->source_false_positive));
    memcpy(m->source_false_negative, data->fn, sizeof(m->source_false_negative));
    return 1;
}

void voxel_assets_free(void)
{
    for (int s = 0; s < MAX_SPRITE_TYPES; ++s)
        for (int f = 0; f < MAX_SPRITE_FRAMES; ++f) voxel_free(&models[s][f]);
    for (int f = 0; f < 10; ++f) voxel_free(&red_death_poses[f]);
    memset(ready, 0, sizeof(ready));
    memset(directional_slot, 0, sizeof(directional_slot));
    bank_bytes = 0; bank_count = 0;
}

size_t voxel_assets_bytes(void) { return bank_bytes; }
int voxel_assets_count(void) { return bank_count; }

const VoxelModel *voxel_assets_model(int slot, int frame, int corpse)
{
    if (slot == 16 || slot == 17) slot = 10; /* Palette variants share geometry. */
    if (slot < 0 || slot >= MAX_SPRITE_TYPES) return NULL;
    /* BitMapObj uses the first PTR entry for an out-of-table frame. */
    if (frame < 0 || frame >= sprite_frames_table[slot].count) frame = 0;
    /* BigRedThing.s uses frames 0..9 sequentially on death, independently
     * of the live angle*4+step interpretation (game_data.c death_frames). */
    if (corpse && slot == 14 && frame < 10 && red_death_poses[frame].columns)
        return &red_death_poses[frame];
    if (slot == 15 && frame < 18) frame = frame < 16 ? frame & 3 : 0;
    if (slot == 9 && frame >= 20 && frame < 28) frame = 28;
    if (directional_slot[slot] && frame < 16) {
        frame &= 3;
    }
    return ready[slot][frame] ? &models[slot][frame] : NULL;
}

static int decode(int slot, int frame, int width, int height, uint8_t *out)
{
    const RendererState *r = &g_renderer;
    SpriteFrame f = sprite_frames_table[slot].frames[frame];
    memset(out, 0, (size_t)width * height);
    for (int x = 0; x < width; ++x) {
        size_t p = (size_t)f.ptr_off + (size_t)x * 4;
        if (p + 4 > r->sprite_ptr_size[slot]) return 0;
        const uint8_t *entry = r->sprite_ptr[slot] + p;
        size_t offset = ((size_t)entry[1] << 16) | ((size_t)entry[2] << 8) | entry[3];
        if (!entry[0] && !offset) continue;
        if (entry[0] > 2 || offset + 2u * (f.down_strip + height) > r->sprite_wad_size[slot]) return 0;
        for (int y = 0; y < height; ++y) {
            size_t q = offset + 2u * (f.down_strip + y);
            unsigned word = ((unsigned)r->sprite_wad[slot][q] << 8) | r->sprite_wad[slot][q + 1];
            out[y * width + x] = (uint8_t)((word >> (entry[0] * 5)) & 31);
        }
    }
    return 1;
}

int voxel_assets_build(void)
{
    /* Source: ObjDraw3.ChipRam.s FRAMES, objects.c enemy_update_anim_with_step
     * and Anims.s BulletSizes. MutantMarine.s:112-117 calls ViewpointToDraw,
     * multiplies direction by four, then adds alframe+2 (the walking step).
     * AlienControl.s:129-142 returns towards=0, right=1, away=2, left=3.
     * Thus frames {step,step+4,step+8,step+12} are one pose, not four poses.
     * The port's enemy_viewpoint applies its existing 180-degree convention;
     * local view cameras {-Z,-X,+Z,+X} preserve that convention in renderer.c.
     * Every loaded world billboard is converted. */
    static const int slots[] = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8_t (*pixels)[128 * 128] = malloc(4 * 128 * 128);
    voxel_assets_free();
    if (!pixels) goto fail;
    for (size_t si = 0; si < sizeof(slots) / sizeof(slots[0]); ++si) {
        int s = slots[si];
        /* Slot 11 is absent in the distributed original assets, as io.c records. */
        if (!g_renderer.sprite_wad[s] && !g_renderer.sprite_ptr[s]) continue;
        if (!g_renderer.sprite_wad[s] || !g_renderer.sprite_ptr[s]) goto fail;
        directional_slot[s] = (uint8_t)(s == 0 || s == 4 || s == 10 || s == 13 || s == 14);
        for (int f = 0; f < sprite_frames_table[s].count; ++f) {
            if (directional_slot[s] && f >= 4 && f < 16) continue;
            if (s == 15 && f >= 4 && f < 18) continue;
            if (s == 9 && f >= 20 && f < 28) continue;
            int width = 64, height = 64;
            if (s == 1 || s == 2 || s == 5 || s == 6) width = height = 32;
            if (s == 0 && f >= 16 && f < 32) width = height = 16;
            if (s == 1 && f >= 6 && f < 20) width = height = 16;
            if (s == 1 && f >= 21 && f <= 24) width = height = 16;
            if (s == 1 && (f == 1 || f == 25 || f == 26)) width = 64;
            if (s == 9) { width = 96; height = 58; }
            if (s == 11 || s == 14) width = height = 128;
            if (s == 13) { width = 90; height = 100; }
            /* Tree's eyeball animation occupies half-width strips (18..21). */
            if (s == 15 && f >= 18) width = 32;
            int count = directional_slot[s] && f < 4 ? 4 : 1;
            const uint8_t *views[4] = {pixels[0], pixels[1], pixels[2], pixels[3]};
            VoxelObservation observations[4] = {0};
            for (int v = 0; v < count; ++v) {
                if (!decode(s, f + v * 4, width, height, pixels[v])) {
                    fprintf(stderr, "[VOXEL] Invalid sprite data: slot %d frame %d (%dx%d)\n", s, f + v * 4, width, height);
                    goto fail;
                }
                observations[v].pixels = views[v];
                observations[v].source.quarter_turn = v;
                observations[v].source.frame = f + v * 4;
                observations[v].source.weight = 1;
            }
            VoxelModel *m = &models[s][f];
            /* Only BARREL is a solid of revolution. Other props keep their
             * silhouettes; other directional art uses one source-pixel slack.
             * AB3D has no patch offsets or mirror flags: preserve full canvases
             * and the angle*4+step mapping, not per-view alpha-bound centres. */
            VoxelBuildOptions options = {
                .shape = s == 7 ? VOXEL_AXISYMMETRIC : count == 4 ? VOXEL_VISUAL_HULL : VOXEL_INFLATED_SINGLE_VIEW,
                .outside_tolerance = count == 4 ? 1.0f : 0.0f
            };
            if (s == 10 && count == 4) {
                /* Explicit part/depth corrections and visibility-aware paint
                 * are baked offline. The runtime still reads three-byte runs. */
                if (!load_authored_marine(m, f, observations)) goto fail;
            } else if (!voxel_build_observations(m, observations, width, height, count, &options)) goto fail;
            ready[s][f] = 1;
            ++bank_count;
            bank_bytes += ((size_t)m->size_x * m->size_z + 1) * sizeof(uint32_t) + (size_t)m->run_count * sizeof(VoxelRun);
        }
    }
    if (g_renderer.sprite_wad[14] && g_renderer.sprite_ptr[14]) {
        for (int f = 0; f < 10; ++f) {
            VoxelObservation view = {pixels[0], {0,0,0,0,1,f}};
            VoxelModel *m = &red_death_poses[f];
            if (!decode(14, f, 128, 128, pixels[0]) || !voxel_build_observations(m, &view, 128, 128, 1, NULL)) goto fail;
            ++bank_count;
            bank_bytes += ((size_t)m->size_x * m->size_z + 1) * sizeof(uint32_t) + (size_t)m->run_count * sizeof(VoxelRun);
        }
    }
    free(pixels);
    return 1;
fail:
    free(pixels);
    fprintf(stderr, "[VOXEL] Asset conversion failed; check the original sprite WAD/PTR files and available memory.\n");
    voxel_assets_free();
    return 0;
}
