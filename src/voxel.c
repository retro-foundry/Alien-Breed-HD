#include "voxel.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void voxel_free(VoxelModel *m)
{
    if (!m->borrowed_data) {
        free(m->columns);
        free(m->runs);
    }
    memset(m, 0, sizeof(*m));
}

static int clip_axis(float origin, float direction, float lo, float hi,
                     float *entry, float *exit)
{
    if (fabsf(direction) < 1e-8f) return origin >= lo && origin < hi;
    float a = (lo - origin) / direction, b = (hi - origin) / direction;
    if (a > b) { float t = a; a = b; b = t; }
    if (a > *entry) *entry = a;
    if (b < *exit) *exit = b;
    return *entry < *exit;
}

int voxel_screen_bounds(const VoxelModel *m, const VoxelView *v,
                        int w, int h, int bounds[4])
{
    if (!m || !m->run_count || v->width <= 0 || v->height <= 0 || v->focal <= 0) return 0;
    float extent = v->width * 0.5f * (fabsf(v->cosine) + fabsf(v->sine));
    float near = v->center_z - extent, far = v->center_z + extent;
    if (far <= v->near_z) return 0;
    if (near < v->near_z) near = v->near_z;
    float xmin = (float)w, xmax = 0, ymin = (float)h, ymax = 0;
    for (int iz = 0; iz < 2; ++iz) for (int s = -1; s <= 1; s += 2) {
        float z = iz ? far : near;
        float x = v->screen_x + v->focal * (v->center_x + s * extent) / z;
        float y = v->screen_y + v->focal * (v->center_y + s * v->height * 0.5f) / z;
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
    }
    /* Clamp before integer conversion: a near-plane crossing can project far outside int. */
    bounds[0] = (int)floorf(fmaxf(0, fminf((float)w, xmin)));
    bounds[1] = (int)floorf(fmaxf(0, fminf((float)h, ymin)));
    bounds[2] = (int)ceilf(fmaxf(0, fminf((float)w, xmax)));
    bounds[3] = (int)ceilf(fmaxf(0, fminf((float)h, ymax)));
    return bounds[0] < bounds[2] && bounds[1] < bounds[3];
}

void voxel_trace_column(const VoxelModel *m, const VoxelView *v,
                        int column, int top, int bottom, uint8_t *indices,
                        VoxelEmitPixel emit, void *context)
{
    memset(indices + top, 0, (size_t)(bottom - top));
    float inverse_focal = 1.0f / v->focal;
    float ray_x = (column + 0.5f - v->screen_x) * inverse_focal;
    float dx = ray_x * v->cosine - v->sine;
    float dz = ray_x * v->sine + v->cosine;
    float ox = -v->center_x * v->cosine + v->center_z * v->sine;
    float oz = -v->center_x * v->sine - v->center_z * v->cosine;
    float half = v->width * 0.5f;
    float entry_x = v->near_z, entry_z = v->near_z, end = FLT_MAX;
    if (!clip_axis(ox, dx, -half, half, &entry_x, &end) ||
        !clip_axis(oz, dz, -half, half, &entry_z, &end)) return;
    float t = fmaxf(entry_x, entry_z);
    if (t >= end) return;
    float cell_x = v->width / m->size_x, cell_z = v->width / m->size_z;
    int x = (int)floorf((ox + dx * t + half) / cell_x);
    int z = (int)floorf((oz + dz * t + half) / cell_z);
    if (x < 0) x = 0;
    if (x >= m->size_x) x = m->size_x - 1;
    if (z < 0) z = 0;
    if (z >= m->size_z) z = m->size_z - 1;
    int step_x = dx >= 0 ? 1 : -1, step_z = dz >= 0 ? 1 : -1;
    float tx = fabsf(dx) < 1e-8f ? FLT_MAX :
        ((x + (step_x > 0)) * cell_x - half - ox) / dx;
    float tz = fabsf(dz) < 1e-8f ? FLT_MAX :
        ((z + (step_z > 0)) * cell_z - half - oz) / dz;
    float delta_x = fabsf(dx) < 1e-8f ? FLT_MAX : fabsf(cell_x / dx);
    float delta_z = fabsf(dz) < 1e-8f ? FLT_MAX : fabsf(cell_z / dz);
    int horizon = (int)ceilf(v->screen_y - 0.5f);
    float inverse_height = 1.0f / m->size_y;
    int covered = 0;
    /* A ray crosses at most nx+nz cells. Vertical runs avoid a 3D ray per pixel. */
    for (int steps = 0; steps <= m->size_x + m->size_z; ++steps) {
        float next = fminf(end, fminf(tx, tz));
        float entry_scale = v->focal / t, exit_scale = v->focal / next;
        uint32_t c = (uint32_t)(z * m->size_x + x);
        /* Positive-Y rays encounter runs top-to-bottom; negative-Y rays
         * encounter them bottom-to-top. Combined with front-to-back X/Z DDA,
         * the first covered pixel is the nearest surface, without a Z buffer. */
        for (int half_screen = 0; half_screen < 2; ++half_screen) {
          int pass_top = top, pass_bottom = bottom;
          if (half_screen == 0 && pass_top < horizon) pass_top = horizon;
          if (half_screen == 1 && pass_bottom > horizon) pass_bottom = horizon;
          if (pass_top >= pass_bottom) continue;
          int begin = (int)m->columns[c], finish = (int)m->columns[c + 1];
          for (int n = begin; n < finish; ++n) {
            int ri = half_screen ? finish - 1 - (n - begin) : n;
            const VoxelRun *r = &m->runs[ri];
            float y0 = v->center_y + (r->y0 * inverse_height - 0.5f) * v->height;
            float y1 = v->center_y + (r->y1 * inverse_height - 0.5f) * v->height;
            float a = v->screen_y + y0 * (y0 < 0 ? entry_scale : exit_scale);
            float b = v->screen_y + y1 * (y1 > 0 ? entry_scale : exit_scale);
            int first = (int)fmaxf((float)pass_top, fminf((float)pass_bottom, ceilf(a - 0.5f)));
            int last = (int)fmaxf((float)pass_top, fminf((float)pass_bottom, ceilf(b - 0.5f)));
            for (int row = first; row < last; ++row) {
                if (indices[row]) continue;
                float dy = (row + 0.5f - v->screen_y) * inverse_focal;
                float y_at_entry = dy * t;
                float enter = t;
                if (y_at_entry < y0) {
                    if (dy <= 0) continue;
                    enter = y0 / dy;
                } else if (y_at_entry >= y1) {
                    if (dy >= 0) continue;
                    enter = y1 / dy;
                }
                if (enter >= next) continue;
                indices[row] = r->color;
                if (emit) emit(context, row, indices[row], enter);
                ++covered;
            }
          }
        }
        if (next >= end || covered == bottom - top) break;
        t = next;
        int cross_x = tx <= tz, cross_z = tz <= tx;
        if (cross_x) { x += step_x; tx += delta_x; }
        if (cross_z) { z += step_z; tz += delta_z; }
        if (x < 0 || x >= m->size_x || z < 0 || z >= m->size_z) break;
    }
}
