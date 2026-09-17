#ifndef AB3D_VOXEL_H
#define AB3D_VOXEL_H

#include <stddef.h>
#include <stdint.h>

/* Immutable, palette-indexed vertical runs. y1 is exclusive. A voxel has
 * one fixed material on every face, independent of the viewing direction. */
typedef struct {
    uint8_t y0, y1, color;
} VoxelRun;

typedef enum {
    VOXEL_VISUAL_HULL,
    VOXEL_AXISYMMETRIC,
    VOXEL_INFLATED_SINGLE_VIEW,
    VOXEL_AUTHORED_PARTS /* Baked offline; never selected by silhouette carving. */
} VoxelShape;

typedef struct {
    int quarter_turn; /* Camera yaw: 0=-Z, 1=-X, 2=+Z, 3=+X. */
    int shift_x, shift_y; /* Explicit registration, at most +/-2 source pixels. */
    int mirrored;
    float weight; /* Original observation 1; generated mirrors may be lower. */
    int frame; /* Source frame ID, retained for diagnostics. */
} VoxelSource;

typedef struct {
    const uint8_t *pixels; /* Palette index 0 alone is transparent. */
    VoxelSource source;
} VoxelObservation;

typedef struct {
    VoxelShape shape;
    float outside_tolerance; /* Hard rejection beyond 0..2 source pixels. */
    float inside_tolerance;  /* Strong support within this distance of a mask. */
    float required_support;  /* 0=hard bounds only; else 0.75..1 strong weight. */
    int max_axis;            /* 0=48; up to 128 for authoring/validation. */
    int sampling_scale;      /* 0/1=native source scale, 2=double. */
} VoxelBuildOptions;

typedef struct {
    int size_x, size_y, size_z;
    uint32_t *columns; /* x + z*size_x, plus one terminal offset */
    VoxelRun *runs;
    uint32_t run_count;
    int borrowed_data; /* Static baked arrays: immutable, not freed by model. */
    /* Orthographic silhouette IoU against each unmodified input view,
     * measured at source resolution during conversion, not during drawing. */
    int source_width, source_height, source_views;
    VoxelSource sources[4];
    VoxelBuildOptions reconstruction;
    float source_iou[4];
    uint32_t source_false_positive[4], source_false_negative[4];
} VoxelModel;

/* Input views are row-major palette indices, 0 transparent, on one shared
 * canvas/pivot. Direction v has image X = {x, -z, -x, z}. Animation poses
 * are separate calls. This convenience wrapper accepts canonical unmirrored
 * frames; voxel_build_observations resolves explicit mirroring itself.
 * Four views always constrain one hull. NULL options select a strict hull
 * for multiple observations, or silhouette inflation for a single view.
 * Conversion belongs to asset loading, never to a render worker. */
int voxel_build(VoxelModel *model, const uint8_t *views[4],
                int width, int height, int view_count, const VoxelBuildOptions *options);
/* Explicit observations may be passed in any order. A partial hard hull is
 * useful for monotonicity checks: another observation can only carve it.
 * Authoritative frame/yaw/registration metadata is separate from sheet order. */
int voxel_build_observations(VoxelModel *model, const VoxelObservation *views,
                             int width, int height, int view_count,
                             const VoxelBuildOptions *options);
void voxel_free(VoxelModel *model);

/* Ray-column projection. Model occupies [-w/2,w/2] x [-h/2,h/2]
 * x [-w/2,w/2], rotated about Y by cos/sin. All distances use view Z
 * units. The caller owns scratch/output, so strips run independently. */
typedef struct {
    float center_x, center_y, center_z;
    float width, height;
    float cosine, sine;
    float focal, screen_x, screen_y, near_z;
} VoxelView;

/* Front-to-back traversal emits each covered row once. Only a one-column
 * coverage mask is stored; hit distance is transient, for wall/plane clipping.
 * There is no model or scene depth buffer. emit may be NULL. */
typedef void (*VoxelEmitPixel)(void *context, int row, uint8_t index, float distance);
void voxel_trace_column(const VoxelModel *model, const VoxelView *view,
                        int column, int top, int bottom,
                        uint8_t *coverage, VoxelEmitPixel emit, void *context);
int voxel_screen_bounds(const VoxelModel *model, const VoxelView *view,
                        int screen_width, int screen_height, int bounds[4]);

#endif
