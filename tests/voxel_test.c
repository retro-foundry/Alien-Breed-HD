#include "voxel.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

/* Independent oracle: intersect every occupied cell's box with a 3D ray.
 * Intentionally slow; this validates the column/run projection, including
 * holes, off-axis perspective, rotations, near-plane crossings and top faces. */
static float oracle(const VoxelModel *m, const VoxelView *v, int sx, int sy)
{
    float dx = (sx + 0.5f - v->screen_x) / v->focal;
    float dy = (sy + 0.5f - v->screen_y) / v->focal;
    float origin[3] = {-v->center_x * v->cosine + v->center_z * v->sine,
                      -v->center_y, -v->center_x * v->sine - v->center_z * v->cosine};
    float ray[3] = {dx * v->cosine - v->sine, dy, dx * v->sine + v->cosine};
    float best = FLT_MAX;
    for (int z = 0; z < m->size_z; ++z) for (int x = 0; x < m->size_x; ++x) {
        int c = z * m->size_x + x;
        for (unsigned ri = m->columns[c]; ri < m->columns[c + 1]; ++ri) {
            const VoxelRun *r = m->runs + ri;
            float lo[3] = {(x / (float)m->size_x - 0.5f) * v->width,
                           (r->y0 / (float)m->size_y - 0.5f) * v->height,
                           (z / (float)m->size_z - 0.5f) * v->width};
            float hi[3] = {lo[0] + v->width / m->size_x,
                           (r->y1 / (float)m->size_y - 0.5f) * v->height,
                           lo[2] + v->width / m->size_z};
            float entry = v->near_z, leave = FLT_MAX;
            for (int axis = 0; axis < 3; ++axis) {
                if (fabsf(ray[axis]) < 1e-8f) {
                    if (origin[axis] < lo[axis] || origin[axis] >= hi[axis]) leave = -1;
                } else {
                    float a = (lo[axis] - origin[axis]) / ray[axis];
                    float b = (hi[axis] - origin[axis]) / ray[axis];
                    entry = fmaxf(entry, fminf(a, b));
                    leave = fminf(leave, fmaxf(a, b));
                }
            }
            if (entry < leave && entry < best) best = entry;
        }
    }
    return best;
}

static void collect_depth(void *context, int row, uint8_t index, float distance)
{
    (void)index;
    ((float *)context)[row] = distance;
}

static int occupied(const VoxelModel *m, int x, int y, int z)
{
    int c = x + z * m->size_x;
    for (unsigned i = m->columns[c]; i < m->columns[c + 1]; ++i)
        if (y >= m->runs[i].y0 && y < m->runs[i].y1) return 1;
    return 0;
}

static int asymmetric_volume(int x, int y, int z)
{
    if (y >= 1 && y < 4) return x >= 2 && x < 4 && z >= 5 && z < 8;
    return y >= 4 && y < 10 && x >= 1 && x < 5 && z >= 6 && z < 10;
}

static void visual_hull_invariants(void)
{
    enum { N = 12 };
    uint8_t pixels[4][N * N] = {{0}};
    VoxelObservation views[4] = {0};
    for (int q = 0; q < 4; ++q) {
        views[q].pixels = pixels[q]; views[q].source.quarter_turn = q;
        views[q].source.weight = 1; views[q].source.frame = 100 + q;
    }
    /* Ground-truth asymmetric solid and its four independent camera images.
     * Palette index 7 may be black; opacity depends only on index != 0. */
    for (int y = 0; y < N; ++y) for (int z = 0; z < N; ++z) for (int x = 0; x < N; ++x)
        if (asymmetric_volume(x, y, z)) {
            pixels[0][y * N + x] = 7;
            pixels[1][y * N + N - 1 - z] = 7;
            pixels[2][y * N + N - 1 - x] = 7;
            pixels[3][y * N + z] = 7;
        }
    CHECK(pixels[0][4 * N + 1] && !pixels[0][4 * N + 10]);
    CHECK(pixels[2][4 * N + 10] && !pixels[2][4 * N + 1]);
    CHECK(pixels[1][4 * N + 2] && pixels[3][4 * N + 9]);
    VoxelBuildOptions hard = {VOXEL_VISUAL_HULL, 0};
    VoxelModel previous = {0};
    for (int count = 1; count <= 4; ++count) {
        VoxelModel next;
        CHECK(voxel_build_observations(&next, views, N, N, count, &hard));
        for (int y = 0; y < N; ++y) for (int z = 0; z < N; ++z) for (int x = 0; x < N; ++x) {
            int now = occupied(&next, x, y, z);
            if (count > 1) CHECK(!now || occupied(&previous, x, y, z));
            if (count == 4) CHECK(now == asymmetric_volume(x, y, z));
        }
        voxel_free(&previous); previous = next;
    }
    for (unsigned i = 0; i < previous.run_count; ++i) CHECK(previous.runs[i].color == 7);
    /* All 24 permutations retain the same occupancy AND deterministic colours. */
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) if (b != a)
        for (int c = 0; c < 4; ++c) if (c != a && c != b) {
            int d = 6 - a - b - c;
            VoxelObservation reordered[4] = {views[a], views[b], views[c], views[d]};
            VoxelModel next;
            CHECK(voxel_build_observations(&next, reordered, N, N, 4, &hard));
            CHECK(next.run_count == previous.run_count);
            CHECK(!memcmp(next.columns, previous.columns, (N * N + 1) * sizeof(*next.columns)));
            CHECK(!memcmp(next.runs, previous.runs, next.run_count * sizeof(*next.runs)));
            voxel_free(&next);
        }
    /* An explicit mirrored observation is interpreted once, not added as
     * another body. Metadata applies before opacity and palette sampling. */
    for (int y = 0; y < N; ++y) for (int x = 0; x < N / 2; ++x) {
        uint8_t t = pixels[3][y * N + x];
        pixels[3][y * N + x] = pixels[3][y * N + N - 1 - x];
        pixels[3][y * N + N - 1 - x] = t;
    }
    views[3].source.mirrored = 1; views[3].source.weight = 0.5f;
    VoxelModel mirrored;
    CHECK(voxel_build_observations(&mirrored, views, N, N, 4, &hard));
    CHECK(mirrored.run_count == previous.run_count);
    CHECK(!memcmp(mirrored.runs, previous.runs, previous.run_count * sizeof(*previous.runs)));
    voxel_free(&mirrored);

    /* 1x and 2x sampling have the same strict hull at source-voxel centres. */
    hard.max_axis = 24; hard.sampling_scale = 2;
    VoxelModel twice;
    CHECK(voxel_build_observations(&twice, views, N, N, 4, &hard));
    for (int y = 0; y < 24; ++y) for (int z = 0; z < 24; ++z) for (int x = 0; x < 24; ++x)
        CHECK(occupied(&twice, x, y, z) == occupied(&previous, x / 2, y / 2, z / 2));
    voxel_free(&twice); voxel_free(&previous);
    hard.shape = VOXEL_INFLATED_SINGLE_VIEW;
    CHECK(!voxel_build_observations(&twice, views, N, N, 4, &hard));

    /* Three observations plus a one-pixel boundary discrepancy may survive;
     * a four-pixel contradiction must carve, despite the other three views. */
    uint8_t narrow[4][64] = {{0}};
    int columns[4] = {3,3,3,4}; /* Candidate x=3,z=4; opposite X is 4. */
    for (int q = 0; q < 4; ++q) {
        views[q].pixels = narrow[q]; views[q].source.mirrored = 0; views[q].source.weight = 1;
        for (int y = 0; y < 8; ++y) narrow[q][y * 8 + columns[q]] = 7;
    }
    VoxelBuildOptions soft = {VOXEL_VISUAL_HULL, 2, 0.5f, 0.75f};
    CHECK(voxel_build_observations(&twice, views, 8, 8, 4, &soft));
    CHECK(occupied(&twice, 3, 4, 4));
    VoxelModel soft_twice;
    soft.sampling_scale = 2;
    CHECK(voxel_build_observations(&soft_twice, views, 8, 8, 4, &soft));
    for (int y = 0; y < 16; ++y) for (int z = 0; z < 16; ++z) for (int x = 0; x < 16; ++x)
        CHECK(occupied(&soft_twice, x, y, z) == occupied(&twice, x / 2, y / 2, z / 2));
    voxel_free(&soft_twice); voxel_free(&twice);
    soft.sampling_scale = 1;
    /* Two strong views cannot retain a point, even within every outer bound. */
    memset(narrow[1], 0, 64);
    for (int y = 0; y < 8; ++y) narrow[1][y * 8 + 2] = 7;
    CHECK(voxel_build_observations(&twice, views, 8, 8, 4, &soft));
    CHECK(!occupied(&twice, 3, 4, 4)); voxel_free(&twice);
    memset(narrow[1], 0, 64);
    for (int y = 0; y < 8; ++y) narrow[1][y * 8 + 3] = 7;
    memset(narrow[2], 0, 64);
    for (int y = 0; y < 8; ++y) narrow[2][y * 8] = 7;
    CHECK(voxel_build_observations(&twice, views, 8, 8, 4, &soft));
    CHECK(!occupied(&twice, 3, 4, 4)); voxel_free(&twice);
    views[2].source.shift_x = 3;
    CHECK(!voxel_build_observations(&twice, views, 8, 8, 4, &soft));
}

static void reconstruction_contract(void)
{
    CHECK(sizeof(VoxelRun) == 3);
    uint8_t pixels[4][64];
    const uint8_t *views[4] = {pixels[0], pixels[1], pixels[2], pixels[3]};
    VoxelModel m;
    for (int v = 0; v < 4; ++v) memset(pixels[v], v + 1, 64);
    CHECK(voxel_build(&m, views, 8, 8, 4, NULL));
    for (int v = 0; v < 4; ++v) CHECK(m.source_iou[v] == 1);
    /* Cardinal surface centres select their visible source palette. Once
     * baked, the same voxel material also applies to its top/bottom faces. */
    const float cs[4] = {1,0,-1,0}, sn[4] = {0,-1,0,1};
    uint8_t output[2];
    for (int v = 0; v < 4; ++v) {
        VoxelView camera = {0,0,5,2,2,cs[v],sn[v],20,0.5f,0.5f,0.1f};
        voxel_trace_column(&m, &camera, 0, 0, 1, output, NULL, NULL);
        CHECK(output[0] == v + 1);
    }
    for (unsigned i = 0; i < m.run_count; ++i) m.runs[i].color = 31;
    VoxelView camera = {0,2,5,2,2,1,0,4.9f,0.5f,0.5f,0.1f};
    voxel_trace_column(&m, &camera, 0, 1, 2, output, NULL, NULL);
    CHECK(output[1] == 31);
    camera.center_y = -2; camera.screen_y = 1.5f;
    voxel_trace_column(&m, &camera, 0, 0, 1, output, NULL, NULL);
    CHECK(output[0] == 31);
    voxel_free(&m);

    /* Background in even ONE direction carves the volume. Opposite view
     * horizontal coordinates must be mirrored, not accumulated as geometry. */
    for (int y = 0; y < 8; ++y) memset(pixels[2] + y * 8, 0, 4);
    CHECK(voxel_build(&m, views, 8, 8, 4, NULL));
    for (int z = 0; z < 8; ++z) for (int x = 0; x < 8; ++x)
        CHECK((m.columns[z * 8 + x + 1] > m.columns[z * 8 + x]) == (x < 4));
    voxel_free(&m);
    memset(pixels[2], 0, 64);
    VoxelBuildOptions tolerant = {VOXEL_VISUAL_HULL, 1};
    CHECK(voxel_build(&m, views, 8, 8, 4, &tolerant));
    CHECK(m.run_count == 0); /* No N-1 or majority fallback. */
    voxel_free(&m);

    /* Sub-voxel registration tolerance is applied in SOURCE pixels. A thin
     * painted column must survive a one-pixel shift without inventing colour. */
    memset(pixels, 0, sizeof(pixels));
    for (int y = 0; y < 8; ++y) for (int v = 0; v < 4; ++v) pixels[v][y * 8 + 3] = (uint8_t)(v + 1);
    CHECK(voxel_build(&m, views, 8, 8, 4, NULL));
    CHECK(m.run_count == 0); voxel_free(&m);
    CHECK(voxel_build(&m, views, 8, 8, 4, &tolerant));
    CHECK(m.run_count > 0);
    for (unsigned r = 0; r < m.run_count; ++r)
        CHECK(m.runs[r].color >= 1 && m.runs[r].color <= 4);
    voxel_free(&m);

    VoxelBuildOptions cylinder = {VOXEL_AXISYMMETRIC, 0};
    CHECK(!voxel_build(&m, views, 8, 8, 4, &cylinder));
    memset(pixels[0], 1, 64);
    CHECK(voxel_build(&m, views, 8, 8, 1, &cylinder));
    /* A solid circle spans both full diameters but excludes its square's
     * corners. This distinguishes revolution from an equal-width square. */
    CHECK(m.columns[1] == m.columns[0]);
    CHECK(m.columns[4] > m.columns[3]);
    CHECK(m.columns[25] > m.columns[24]);
    CHECK(m.source_iou[0] == 1);
    for (int z = 0; z < 8; ++z) for (int x = 0; x < 8; ++x) {
        float dx = x + 0.5f - 4, dz = z + 0.5f - 4;
        CHECK(occupied(&m, x, 4, z) == (dx * dx + dz * dz <= 16));
    }
    voxel_free(&m);
    tolerant.outside_tolerance = 3;
    CHECK(!voxel_build(&m, views, 8, 8, 4, &tolerant));
}

int main(void)
{
    visual_hull_invariants();
    reconstruction_contract();
    uint8_t pixels[4][64];
    const uint8_t *views[4] = {pixels[0], pixels[1], pixels[2], pixels[3]};
    for (int v = 0; v < 4; ++v) for (int i = 0; i < 64; ++i)
        pixels[v][i] = (uint8_t)(v + 1);
    VoxelModel m;
    CHECK(voxel_build(&m, views, 8, 8, 4, NULL));
    CHECK(m.run_count == 64);
    CHECK(m.runs[0].y0 == 0 && m.runs[0].y1 == 8);
    voxel_free(&m);
    for (int v = 0; v < 4; ++v) {
        pixels[v][2 * 8 + 3] = 0;
        pixels[v][3 * 8 + 3] = 0;
        pixels[v][5 * 8 + 6] = 0;
    }
    int comparisons = 0;
    for (int mode = 1; mode <= 4; mode += 3) {
        CHECK(voxel_build(&m, views, 8, 8, mode, NULL));
        for (int scenario = 0; scenario < 12; ++scenario) {
            float angle = scenario * 0.573f;
            VoxelView v = {0.31f, -0.8f, 4.2f, 2.0f, 3.0f,
                           cosf(angle), sinf(angle), 31.0f, 20.0f, 18.0f, 0.1f};
            if (scenario == 10) v.center_z = 0.5f;
            if (scenario == 11) { v.center_z = -0.3f; v.center_y = 0.1f; }
            uint8_t index[40], split[40]; float depth[40], split_depth[40];
            int bounds[4];
            CHECK(voxel_screen_bounds(&m, &v, 40, 40, bounds));
            for (int x = 0; x < 40; ++x) {
                for (int y = 0; y < 40; ++y) depth[y] = split_depth[y] = FLT_MAX;
                voxel_trace_column(&m, &v, x, 0, 40, index, collect_depth, depth);
                voxel_trace_column(&m, &v, x, 0, 17, split, collect_depth, split_depth);
                voxel_trace_column(&m, &v, x, 17, 40, split, collect_depth, split_depth);
                CHECK(memcmp(index, split, 40) == 0);
                CHECK(memcmp(depth, split_depth, sizeof(depth)) == 0);
                for (int y = 0; y < 40; ++y) {
                    float expected = oracle(&m, &v, x, y);
                    CHECK((index[y] == 0) == (expected == FLT_MAX));
                    if (index[y]) {
                        if (fabsf(depth[y] - expected) >= 0.0001f)
                            fprintf(stderr, "mode=%d scenario=%d x=%d y=%d actual=%.9f expected=%.9f\n", mode, scenario, x, y, depth[y], expected);
                        CHECK(fabsf(depth[y] - expected) < 0.0001f);
                        CHECK(x >= bounds[0] && x < bounds[2] && y >= bounds[1] && y < bounds[3]);
                    }
                    ++comparisons;
                }
            }
        }
        voxel_free(&m);
    }
    memset(pixels, 0, sizeof(pixels));
    CHECK(voxel_build(&m, views, 8, 8, 4, NULL));
    CHECK(m.run_count == 0);
    voxel_free(&m);
    CHECK(!voxel_build(&m, views, 129, 8, 1, NULL));
    CHECK(!voxel_build(&m, views, 8, 8, 2, NULL));
    printf("Voxel oracle: %d ray/depth comparisons and split equivalence passed\n", comparisons);
    return 0;
}
