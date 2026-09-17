#include "renderer.h"
#include "voxel_assets.h"
#include "sprite_palettes.h"
#include "sprite_frames.h"
#include "sb_decompress.h"
#include "voxel_reconstruction.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

RendererState g_renderer;

static void require(int condition, const char *message)
{
    if (!condition) { fprintf(stderr, "%s\n", message); exit(1); }
}

int main(int argc, char **argv)
{
    static const struct { int slot; const char *name; } assets[] = {
        {0,"alien2"}, {1,"pickups"}, {4,"flyingalien"}, {5,"keys"},
        {7,"barrel"}, {10,"newmarine"}, {12,"lamps"}, {13,"worm"},
        {14,"bigclaws"}, {15,"tree"}, {2,"bigbullet"}, {6,"ROCKETS"},
        {8,"explosion"}, {9,"newgunsinhand"}
    };
    require(argc >= 2 && argc <= 4, "Usage: voxel_assets_test <data-dir> [preview.ppm] [existing-report-directory]");
    for (size_t i = 0; i < sizeof(assets) / sizeof(assets[0]); ++i) {
        char path[1024]; uint8_t *data = NULL; size_t size = 0;
        int s = assets[i].slot;
        snprintf(path, sizeof(path), "%s/includes/%s.wad", argv[1], assets[i].name);
        require(sb_load_file(path, &data, &size) == 0, path);
        g_renderer.sprite_wad[s] = data; g_renderer.sprite_wad_size[s] = size;
        snprintf(path, sizeof(path), "%s/includes/%s.ptr", argv[1], assets[i].name);
        require(sb_load_file(path, &data, &size) == 0, path);
        g_renderer.sprite_ptr[s] = data; g_renderer.sprite_ptr_size[s] = size;
    }
    if (argc == 4 && !strcmp(argv[2], "--authoring-inputs")) {
        voxel_reconstruction_export_references(argv[3]);
        for (int s = 0; s < MAX_SPRITE_TYPES; ++s) {
            free((void *)g_renderer.sprite_wad[s]); free((void *)g_renderer.sprite_ptr[s]);
        }
        return 0;
    }
    clock_t start = clock();
    require(voxel_assets_build(), "Real asset conversion failed");
    printf("Converted %d models, %zu bytes in %.3f seconds\n", voxel_assets_count(),
           voxel_assets_bytes(), (double)(clock() - start) / CLOCKS_PER_SEC);
    int populated = 0, empty = 0;
    for (int s = 0; s < MAX_SPRITE_TYPES; ++s) for (int f = 0; f < sprite_frames_table[s].count; ++f) {
        const VoxelModel *m = voxel_assets_model(s, f, 0);
        if (!m) continue;
        if (!m->run_count) {
            require(s == 1 && f >= 14 && f <= 16, "Nonempty original sprite lost all geometry");
            ++empty; printf("Empty source/model: slot %d frame %d\n", s, f); continue;
        }
        ++populated;
        require(m->columns[m->size_x * m->size_z] == m->run_count, "Bad column sentinel");
        for (unsigned r = 0; r < m->run_count; ++r) {
            require(m->runs[r].y0 < m->runs[r].y1 && m->runs[r].y1 <= m->size_y, "Bad run bounds");
            require(m->runs[r].color > 0 && m->runs[r].color < 32, "Bad palette index");
        }
    }
    require(populated > 100, "Missing enemy/prop coverage");
    require(voxel_assets_model(10, 0, 0) == voxel_assets_model(16, 0, 0), "Marine geometry not shared");
    require(voxel_assets_model(10, 0, 0) == voxel_assets_model(10, 12, 0), "Direction changed pose geometry");
    require(voxel_assets_model(10, 0, 0) != voxel_assets_model(10, 1, 0), "Different walking poses were merged");
    require(voxel_assets_model(10, 0, 0)->borrowed_data &&
            voxel_assets_model(10, 0, 0)->reconstruction.shape == VOXEL_AUTHORED_PARTS,
            "Marine did not load the baked authored volume");
    require(voxel_assets_model(14, 4, 1) != voxel_assets_model(14, 4, 0), "Death pose interpreted as a walking direction");
    require(voxel_assets_model(8, 0, 0) && voxel_assets_model(0, 16, 0) &&
            voxel_assets_model(2, 0, 0) && voxel_assets_model(6, 0, 0) &&
            voxel_assets_model(1, 21, 0), "Effects missing voxel models");
    printf("Validated %d frame mappings (%d empty source poses)\n", populated, empty);
    /* Every unique conversion records its source reprojections, including
     * single-view poses/effects and the separate BigRedThing death models. */
    FILE *metrics = NULL, *inputs = NULL;
    if (argc >= 4) {
        char path[1024];
        require(snprintf(path, sizeof(path), "%s/all-models.csv", argv[3]) < (int)sizeof(path), "Report path too long");
        metrics = fopen(path, "wb"); require(metrics != NULL, "Create the report directory before running the test");
        fprintf(metrics, "slot,pose,corpse,width,height,view,source_frame,yaw,pivot_x,ground_y,shift_x,shift_y,mirrored,weight,transparent_index,mode,outside_px,inside_px,support,iou,false_positive,false_negative,input_sheet\n");
        require(snprintf(path, sizeof(path), "%s/inputs.html", argv[3]) < (int)sizeof(path), "Report path too long");
        inputs = fopen(path, "wb"); require(inputs != NULL, "Cannot open input audit index");
        fprintf(inputs, "<!doctype html><meta charset='utf-8'><title>Voxel input audit</title>"
                "<style>body{background:#14161b;color:#eee;font:16px system-ui;max-width:1000px;margin:32px auto}img{width:100%%}a{color:#8cf}</style>"
                "<h1>All unique voxel inputs</h1><p><a href='index.html'>Turntables and geometry metrics</a> / "
                "<a href='all-models.csv'>Source metadata and error counts</a>. Palette index 0 is transparent; "
                "other indices remain opaque, including black. Cyan guides show the registered pivot and ground.</p>\n");
    }
    const VoxelModel *seen[256]; int seen_count = 0;
    for (int corpse = 0; corpse < 2; ++corpse)
        for (int s = 0; s < MAX_SPRITE_TYPES; ++s)
            for (int f = 0; f < sprite_frames_table[s].count; ++f) {
                const VoxelModel *m = voxel_assets_model(s, f, corpse);
                if (!m) continue;
                int duplicate = 0;
                for (int i = 0; i < seen_count; ++i) duplicate |= seen[i] == m;
                if (duplicate) continue;
                require(seen_count < 256, "Unique-model test capacity exceeded");
                seen[seen_count++] = m;
                require(m->source_views == 1 || m->source_views == 4, "Missing source reprojection");
                const char *mode = m->reconstruction.shape == VOXEL_AXISYMMETRIC ? "lathe" :
                                   m->reconstruction.shape == VOXEL_AUTHORED_PARTS ? "authored_parts" :
                                   m->reconstruction.shape == VOXEL_VISUAL_HULL ? "visual_hull" : "single_view_inflation";
                for (int v = 0; v < m->source_views; ++v) {
                    const VoxelSource *source = &m->sources[v];
                    require(isfinite(m->source_iou[v]) && m->source_iou[v] >= 0 && m->source_iou[v] <= 1,
                            "Invalid source reprojection metric");
                    if (m->source_views == 4) {
                        require(source->frame == (f & 3) + 4 * v && source->quarter_turn == v,
                                "Directional frames do not match the original viewpoint/animation grouping");
                        require(!source->mirrored && source->weight == 1, "Original independent views were treated as generated mirrors");
                    }
                    if (metrics) fprintf(metrics, "%d,%d,%d,%d,%d,%d,%d,%d,%.1f,%d,%d,%d,%d,%.3f,0,%s,%.3f,%.3f,%.3f,%.6f,%u,%u,input-%d-%d-%d.svg\n",
                        s, f, corpse, m->source_width, m->source_height, v, source->frame, source->quarter_turn * 90,
                        m->source_width * 0.5, m->source_height, source->shift_x, source->shift_y, source->mirrored, source->weight,
                        mode, m->reconstruction.outside_tolerance, m->reconstruction.inside_tolerance,
                        m->reconstruction.required_support, m->source_iou[v], m->source_false_positive[v],
                        m->source_false_negative[v], s, f, corpse);
                }
                if (inputs) {
                    voxel_reconstruction_inputs(argv[3], s, f, corpse, m);
                    fprintf(inputs, "<h2>Slot %d / pose %d / corpse %d / %s</h2><img loading='lazy' src='input-%d-%d-%d.svg' alt='Source and mask audit'>\n",
                            s, f, corpse, mode, s, f, corpse);
                }
            }
    require(seen_count == voxel_assets_count(), "Some converted models were not validated");
    if (metrics) require(!ferror(metrics) && fclose(metrics) == 0, "Cannot finish model metrics");
    if (inputs) require(!ferror(inputs) && fclose(inputs) == 0, "Cannot finish input audit index");
    voxel_reconstruction_check(argc >= 4 ? argv[3] : NULL);

    if (argc >= 3) {
        enum { TILE = 192, W = TILE * 4, H = TILE * 10 };
        uint8_t *image = calloc((size_t)W * H, 3);
        require(image != NULL, "Preview allocation failed");
        start = clock();
        for (int i = 0; i < 10; ++i) for (int a = 0; a < 4; ++a) {
            int slot = assets[i].slot;
            const VoxelModel *m = voxel_assets_model(slot, 0, 0);
            require(m && m->run_count, "Empty first model");
            float angle = (float)a * 0.65f;
            VoxelView v = {0, 0.22f, 3.5f, 2, 2, cosf(angle), sinf(angle), 192, 96, 65, 0.01f};
            uint8_t indices[TILE];
            for (int x = 0; x < TILE; ++x) {
                voxel_trace_column(m, &v, x, 0, TILE, indices, NULL, NULL);
                for (int y = 0; y < TILE; ++y) {
                    uint8_t *p = image + ((size_t)(i * TILE + y) * W + a * TILE + x) * 3;
                    if (indices[y]) {
                        const uint8_t *pal = sprite_pal_embedded[slot] + indices[y] * 2;
                        unsigned color = ((unsigned)pal[0] << 8) | pal[1];
                        p[0] = (uint8_t)(((color >> 8) & 15) * 17);
                        p[1] = (uint8_t)(((color >> 4) & 15) * 17);
                        p[2] = (uint8_t)((color & 15) * 17);
                    } else { p[0] = 28; p[1] = 30; p[2] = 36; }
                }
            }
        }
        printf("40 model views (192x192) rendered in %.3f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC);
        FILE *out = fopen(argv[2], "wb");
        require(out != NULL, "Cannot open preview output");
        fprintf(out, "P6\n%d %d\n255\n", W, H);
        require(fwrite(image, 3, (size_t)W * H, out) == (size_t)W * H, "Preview write failed");
        require(fclose(out) == 0, "Preview close failed");
        free(image);
    }
    voxel_assets_free();
    for (int s = 0; s < MAX_SPRITE_TYPES; ++s) {
        free((void *)g_renderer.sprite_wad[s]);
        free((void *)g_renderer.sprite_ptr[s]);
    }
    return 0;
}
