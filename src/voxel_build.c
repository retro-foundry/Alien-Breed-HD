#include "voxel.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int occupied_at(const uint8_t *solid, int nx, int ny, int x, int y, int z)
{
    return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nx &&
           solid[(y * nx + z) * nx + x] != 0;
}

/* Bake a single material per voxel. Choosing another sprite for each tiny
 * stair-step face repeats painted eyes/armour edges at oblique angles. Instead
 * choose the visible camera nearest the local surface normal, then propagate
 * observed material into unobserved interior/top surfaces. All colours remain
 * exact source palette indices; this never changes occupancy. */
static int colour_hull(uint8_t *solid, int nx, int ny, const uint8_t *sampled,
                       unsigned available, const float weights[4])
{
    size_t cells = (size_t)nx * nx * ny;
    uint32_t *queue = malloc(cells * sizeof(*queue));
    if (!queue) return 0;
    size_t head = 0, tail = 0;
    for (int y = 0; y < ny; ++y) for (int z = 0; z < nx; ++z) for (int x = 0; x < nx; ++x) {
        size_t p = (size_t)(y * nx + z) * nx + x;
        if (!solid[p]) continue;
        int visible[4], any = 0;
        int distances[4] = {z, x, nx - 1 - z, nx - 1 - x};
        for (int v = 0; v < 4; ++v) {
            visible[v] = (available >> v) & 1;
            for (int d = 1; d <= distances[v]; ++d) {
                int xx = x + (v == 1 ? -d : v == 3 ? d : 0);
                int zz = z + (v == 0 ? -d : v == 2 ? d : 0);
                if (solid[(y * nx + zz) * nx + xx]) { visible[v] = 0; break; }
            }
            any |= visible[v];
        }
        if (!any) continue;
        /* A five-cell neighbourhood estimates the underlying surface normal,
         * without smoothing geometry or interpolating the pixel-art palette. */
        int normal_x = 0, normal_z = 0;
        for (int dy = -2; dy <= 2; ++dy) for (int d = -2; d <= 2; ++d) {
            normal_x += occupied_at(solid, nx, ny, x - 2, y + dy, z + d) -
                        occupied_at(solid, nx, ny, x + 2, y + dy, z + d);
            normal_z += occupied_at(solid, nx, ny, x + d, y + dy, z - 2) -
                        occupied_at(solid, nx, ny, x + d, y + dy, z + 2);
        }
        int scores[4] = {-normal_z, -normal_x, normal_z, normal_x};
        int best = -1;
        for (int v = 0; v < 4; ++v)
            if (visible[v] && (best < 0 || scores[v] > scores[best] ||
                (scores[v] == scores[best] && weights[v] > weights[best]))) best = v;
        int u = best == 0 ? x : best == 1 ? nx - 1 - z : best == 2 ? nx - 1 - x : z;
        solid[p] = sampled[(best * ny + y) * nx + u];
        queue[tail++] = (uint32_t)p;
    }
    /* Multi-source breadth-first material extension, confined to the hull.
     * Empty gaps are never filled, and no new palette values are generated. */
    while (head < tail) {
        uint32_t p = queue[head++];
        int x = (int)(p % nx), z = (int)(p / nx % nx), y = (int)(p / (nx * nx));
        const int offsets[6] = {-1, 1, -nx, nx, -nx * nx, nx * nx};
        const int valid[6] = {x > 0, x + 1 < nx, z > 0, z + 1 < nx, y > 0, y + 1 < ny};
        for (int d = 0; d < 6; ++d) if (valid[d]) {
            uint32_t q = (uint32_t)((int)p + offsets[d]);
            if (solid[q] == 255) { solid[q] = solid[p]; queue[tail++] = q; }
        }
    }
    free(queue);
    return 1;
}


static uint8_t source_pixel(const VoxelObservation *view, int w, int h, int x, int y)
{
    if (x < 0 || x >= w || y < 0 || y >= h) return 0;
    if (view->source.mirrored) x = w - 1 - x;
    return view->pixels[y * w + x];
}

/* Signed Euclidean distance between pixel CENTRES, truncated beyond the
 * largest permitted tolerance. Foreground is positive, background negative.
 * Thus an immediately adjacent background pixel is -1, not -0.5: a 0.5-pixel
 * support threshold does not grow all four silhouettes by a whole pixel.
 * Radius three computes every relevant distance exactly for limits <=2.
 * Also copy the nearest opaque texel for later colouring; never use RGB. */
static float source_distance(const VoxelObservation *view, int w, int h,
                             int x, int y, uint8_t *colour)
{
    uint8_t original = source_pixel(view, w, h, x, y);
    int best_squared = 10;
    *colour = original;
    for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx) {
        int squared = dx * dx + dy * dy;
        if (!squared || squared >= best_squared) continue;
        uint8_t other = source_pixel(view, w, h, x + dx, y + dy);
        if ((other != 0) == (original != 0)) continue;
        best_squared = squared;
        if (!original) *colour = other;
    }
    float distance = sqrtf((float)best_squared);
    return original ? distance : -distance;
}

int voxel_build(VoxelModel *m, const uint8_t *pixels[4], int width, int height,
                int count, const VoxelBuildOptions *options)
{
    if (!pixels || (count != 1 && count != 4)) return 0;
    VoxelObservation views[4] = {0};
    for (int v = 0; v < count; ++v) {
        views[v].pixels = pixels[v];
        views[v].source.quarter_turn = v;
        views[v].source.weight = 1;
        views[v].source.frame = v;
    }
    return voxel_build_observations(m, views, width, height, count, options);
}

int voxel_build_observations(VoxelModel *m, const VoxelObservation *views,
                             int width, int height, int count,
                             const VoxelBuildOptions *options)
{
    VoxelBuildOptions opt = {0};
    opt.shape = count == 1 ? VOXEL_INFLATED_SINGLE_VIEW : VOXEL_VISUAL_HULL;
    if (options) opt = *options;
    if (!opt.max_axis) opt.max_axis = 48;
    if (!opt.sampling_scale) opt.sampling_scale = 1;
    if (!m || !views || width < 1 || width > 128 || height < 1 || height > 128 ||
        count < 1 || count > 4 || opt.max_axis < 1 || opt.max_axis > 128 ||
        opt.sampling_scale < 1 || opt.sampling_scale > 2 ||
        opt.shape < VOXEL_VISUAL_HULL || opt.shape > VOXEL_INFLATED_SINGLE_VIEW ||
        !isfinite(opt.outside_tolerance) || opt.outside_tolerance < 0 || opt.outside_tolerance > 2 ||
        !isfinite(opt.inside_tolerance) || opt.inside_tolerance < 0 ||
        opt.inside_tolerance > opt.outside_tolerance ||
        !isfinite(opt.required_support) ||
        (opt.required_support != 0 && (opt.required_support < 0.75f || opt.required_support > 1)) ||
        (opt.shape != VOXEL_VISUAL_HULL && (count != 1 || views[0].source.quarter_turn != 0))) return 0;
    unsigned available = 0;
    float weights[4] = {0}, total_weight = 0;
    for (int v = 0; v < count; ++v) {
        const VoxelSource *s = &views[v].source;
        if (!views[v].pixels || s->quarter_turn < 0 || s->quarter_turn > 3 ||
            (available & (1u << s->quarter_turn)) || s->shift_x < -2 || s->shift_x > 2 ||
            s->shift_y < -2 || s->shift_y > 2 ||
            s->mirrored < 0 || s->mirrored > 1 || !isfinite(s->weight) || s->weight <= 0 || s->weight > 1)
            return 0;
        for (int i = 0; i < width * height; ++i) if (views[v].pixels[i] > 31) return 0;
        available |= 1u << s->quarter_turn;
        weights[s->quarter_turn] = s->weight;
    }
    /* Canonical summation keeps fractional confidence order-independent. */
    for (int q = 0; q < 4; ++q) total_weight += weights[q];
    memset(m, 0, sizeof(*m));
    int nx = width * opt.sampling_scale, ny = height * opt.sampling_scale;
    if (nx > opt.max_axis) nx = opt.max_axis;
    if (ny > opt.max_axis) ny = opt.max_axis;
    int nz = nx;
    size_t plane = (size_t)nx * ny, cells = plane * nz;
    uint8_t *sampled = calloc(4 * plane, 1);
    float *fields = calloc(4 * plane, sizeof(*fields));
    uint8_t *projections = calloc(4 * plane, 1);
    uint16_t *distance = calloc(plane, sizeof(*distance));
    uint8_t *solid = calloc(cells, 1);
    int radii[128] = {0};
    if (!sampled || !fields || !projections || !distance || !solid) goto fail;

    /* One source scale and canvas for every view. The canonical quarter-turn
     * index, not input order, controls the geometry and material projection. */
    for (int v = 0; v < count; ++v) {
        const VoxelSource *s = &views[v].source;
        size_t base = (size_t)s->quarter_turn * plane;
        for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
            int sx = (2 * x + 1) * width / (2 * nx) + s->shift_x;
            int sy = (2 * y + 1) * height / (2 * ny) + s->shift_y;
            fields[base + y * nx + x] = source_distance(&views[v], width, height, sx, sy,
                                                       &sampled[base + y * nx + x]);
        }
    }
    if (opt.shape == VOXEL_AXISYMMETRIC) {
        for (int y = 0; y < ny; ++y) {
            int sy = (2 * y + 1) * height / (2 * ny) + views[0].source.shift_y;
            int left = width, right = -1;
            for (int x = 0; x < width; ++x) if (source_pixel(&views[0], width, height, x, sy)) {
                if (x < left) left = x;
                right = x;
            }
            radii[y] = right >= left ? (right - left + 1) * nx : 0;
        }
    }
    if (opt.shape == VOXEL_INFLATED_SINGLE_VIEW) {
        for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
            int d = fields[y * nx + x] >= -opt.outside_tolerance ? 1 + nx + ny : 0;
            if (d && (!x || !y || x == nx - 1 || y == ny - 1)) d = 1;
            distance[y * nx + x] = (uint16_t)d;
        }
        for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
            int i = y * nx + x, d = distance[i];
            if (x && distance[i - 1] + 1 < d) d = distance[i - 1] + 1;
            if (y && distance[i - nx] + 1 < d) d = distance[i - nx] + 1;
            distance[i] = (uint16_t)d;
        }
        for (int y = ny - 1; y >= 0; --y) for (int x = nx - 1; x >= 0; --x) {
            int i = y * nx + x, d = distance[i];
            if (x + 1 < nx && distance[i + 1] + 1 < d) d = distance[i + 1] + 1;
            if (y + 1 < ny && distance[i + nx] + 1 < d) d = distance[i + nx] + 1;
            distance[i] = (uint16_t)d;
        }
    }

    /* Every observation constrains this ONE initially bounded solid. No
     * source pixels/planes are ever inserted or accumulated into geometry. */
    for (int y = 0; y < ny; ++y) for (int z = 0; z < nz; ++z) for (int x = 0; x < nx; ++x) {
        int occupied = 1;
        if (opt.shape == VOXEL_AXISYMMETRIC) {
            int dx = (2 * x + 1 - nx) * width, dz = (2 * z + 1 - nz) * width;
            occupied = radii[y] && (int64_t)dx * dx + (int64_t)dz * dz <= (int64_t)radii[y] * radii[y];
        } else if (opt.shape == VOXEL_VISUAL_HULL) {
            int u[4] = {x, nx - 1 - z, nx - 1 - x, z};
            float support = 0;
            for (int q = 0; q < 4; ++q) if (available & (1u << q)) {
                float d = fields[q * plane + y * nx + u[q]];
                if (d < -opt.outside_tolerance) { occupied = 0; break; }
                if (d >= -opt.inside_tolerance) support += weights[q];
            }
            if (opt.required_support && support < opt.required_support * total_weight) occupied = 0;
        } else {
            int d = distance[y * nx + x], half = d < nx / 3 ? d : nx / 3;
            if (half < 1) half = 1;
            occupied = d && abs(2 * z + 1 - nz) <= 2 * half;
        }
        solid[(y * nz + z) * nx + x] = occupied ? 255 : 0;
    }
    if (opt.shape == VOXEL_VISUAL_HULL && !colour_hull(solid, nx, ny, sampled, available, weights)) goto fail;

    m->columns = malloc(((size_t)nx * nz + 1) * sizeof(*m->columns));
    m->runs = malloc(cells * sizeof(*m->runs));
    if (!m->columns || !m->runs) goto fail;
    m->size_x = nx; m->size_y = ny; m->size_z = nz;
    for (int z = 0; z < nz; ++z) for (int x = 0; x < nx; ++x) {
        m->columns[z * nx + x] = m->run_count;
        VoxelRun *last = NULL;
        for (int y = 0; y < ny; ++y) {
            uint8_t color;
            if (!solid[(y * nz + z) * nx + x]) { last = NULL; continue; }
            if (opt.shape == VOXEL_VISUAL_HULL) {
                color = solid[(y * nz + z) * nx + x];
            } else if (opt.shape == VOXEL_AXISYMMETRIC) {
                int sy = (2 * y + 1) * height / (2 * ny) + views[0].source.shift_y;
                int sx = (2 * x + 1) * width / (2 * nx) + views[0].source.shift_x;
                color = source_pixel(&views[0], width, height, sx, sy);
                for (int d = 1; !color && d < width; ++d) {
                    color = source_pixel(&views[0], width, height, sx - d, sy);
                    if (!color) color = source_pixel(&views[0], width, height, sx + d, sy);
                }
            } else {
                color = sampled[y * nx + x];
            }
            if (!color || color > 31) goto fail;
            if (last && last->y1 == y && last->color == color) last->y1 = (uint8_t)(y + 1);
            else {
                last = &m->runs[m->run_count++];
                last->y0 = (uint8_t)y; last->y1 = (uint8_t)(y + 1); last->color = color;
            }
        }
    }
    m->columns[nx * nz] = m->run_count;
    for (int y = 0; y < ny; ++y) for (int z = 0; z < nz; ++z) for (int x = 0; x < nx; ++x) {
        if (!solid[(y * nz + z) * nx + x]) continue;
        projections[y * nx + x] = 1;
        projections[plane + y * nx + nx - 1 - z] = 1;
        projections[2 * plane + y * nx + nx - 1 - x] = 1;
        projections[3 * plane + y * nx + z] = 1;
    }
    m->source_width = width; m->source_height = height; m->source_views = count;
    m->reconstruction = opt;
    for (int v = 0; v < count; ++v) {
        m->sources[v] = views[v].source;
        const VoxelSource *s = &views[v].source;
        unsigned overlap = 0, combined = 0;
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            int xx = (int)floorf((x + 0.5f - s->shift_x) * nx / width);
            int yy = (int)floorf((y + 0.5f - s->shift_y) * ny / height);
            int original = source_pixel(&views[v], width, height, x, y) != 0;
            int projected = xx >= 0 && xx < nx && yy >= 0 && yy < ny &&
                            projections[s->quarter_turn * plane + yy * nx + xx];
            overlap += original && projected; combined += original || projected;
            m->source_false_positive[v] += projected && !original;
            m->source_false_negative[v] += original && !projected;
        }
        m->source_iou[v] = combined ? (float)overlap / combined : 1;
    }
    free(sampled); free(fields); free(projections); free(distance); free(solid);
    if (m->run_count) {
        VoxelRun *compact = realloc(m->runs, (size_t)m->run_count * sizeof(*m->runs));
        if (compact) m->runs = compact;
    } else { free(m->runs); m->runs = NULL; }
    return 1;
fail:
    free(sampled); free(fields); free(projections); free(distance); free(solid);
    voxel_free(m);
    return 0;
}
