#ifndef AB3D_VOXEL_RECONSTRUCTION_H
#define AB3D_VOXEL_RECONSTRUCTION_H

#include "voxel.h"

void voxel_reconstruction_inputs(const char *directory, int slot, int frame, int corpse,
                                 const VoxelModel *model);
void voxel_reconstruction_check(const char *report_directory);
void voxel_reconstruction_export_references(const char *directory);

#endif
