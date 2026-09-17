#ifndef AB3D_VOXEL_ASSETS_H
#define AB3D_VOXEL_ASSETS_H
#include "voxel.h"

/* Called after io_load_objects has decoded/patched all original WADs. */
int voxel_assets_build(void);
void voxel_assets_free(void);
const VoxelModel *voxel_assets_model(int slot, int frame, int corpse);
size_t voxel_assets_bytes(void);
int voxel_assets_count(void);

#endif
