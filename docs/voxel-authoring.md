# Marine reconstruction and authoring

The marine now uses explicit part volumes and a fixed, visibility-aware palette
bake. The old silhouette hull is retained as a reproducible diagnostic, not as
the marine's runtime geometry. The barrel is unchanged. Voxels remain the default;
the software renderer still has no depth buffer.

## Diagnosis and evidence

The earlier silhouette improvements did not establish convincing anatomy.
`voxel_build.c:colour_hull` already checked occlusion along its four cardinal
camera rays. It did **not** simply project colour through every occupied cell.
However, its geometry had no distinction between gun, hands, arms and torso,
and its material propagation could cross those regions. The source art also
contains incompatible outlines and different painted heights for the weapon.

`tools/voxel_author.py:legacy_bake` independently reproduces the previous C bake
and requires every palette index to match the frozen exported model. The report
compares three controlled cases with identical cameras and scale:

1. Frozen hull and original colour assignment.
2. The same occupied cells with first-hit and material-region restrictions.
3. Editable part geometry with that same restricted colour method.

The neutral geometry sheets show the original hull's broad arm/weapon/chest
mass between source angles. Recolouring alone preserves that mass and loses
painted details; it is not the shipped correction. The edited model provides
separate depth for the weapon, forearms and chest. This is an authored
interpretation, not recovery of an original 3D model.

## Correction data and provenance

`assets/voxels/marine.json` is the editable source. It contains:

- Original WAD/PTR resource names, the `direction * 4 + step` frame rule,
  full 64x64 canvas, pivot, baseline and transparent index.
- Camera direction, nominal orthographic projection, pixel aspect and the
  provenance/confidence of assumptions. Cardinal angle order is established by
  `MutantMarine.s` and `AlienControl.s:ViewpointToDraw`; original camera pitch and
  projection are not recovered from the painted sprites.
- Explicit part primitives, dimensions, transforms, helmet tilt, per-pose leg
  guides and weapon centres. Occluded depths are identified as inferred.
- Editable source-image material-region polygons, original palette materials
  for unseen cells and landmarks.

The 23 parts per pose cover the helmet/fins/neck, chest/back/abdomen/pelvis,
shoulders, upper arms, forearms, hands, weapon, thighs, shins and feet. Shapes
are superellipsoids and capsules, sampled at voxel centres on a 48-cubed grid.
Later named parts own overlapping cells. Contact between parts is permitted;
there is no silhouette-based deletion of an occluded limb.

The weapon's X position comes from the front orange muzzle pixels; its forward
depth comes from the independent side drawings. Front and side muzzle heights
disagree by 4.5..6.5 native pixels. The authored Y guide uses their equal-weight
midpoint and records that compromise. Hands and forearm ends follow the weapon.
Other dimensions were fitted by inspecting the labelled front/profile drawings;
they are explicit modelling choices, not calibrated measurements. No global
optimisation score or IoU threshold certifies the model.

The retained one-pixel back-view offsets come from the previous bounded
silhouette alignment. They are recorded as such, not described as camera
calibration. A source pixel aspect of 1 matches this port's square marine world
dimensions and current voxel projection; it does not establish the historical
display or the original artist's camera.

## Colour assignment

For each original camera, `first_hits` generates source-resolution voxel IDs and
depths. `bake` requires an exact first-hit ID and matching depth before using a
source pixel. Background and out-of-bounds samples are rejected. An additional
material-region check prevents, for example, a weapon pixel from painting a
chest cell. Analytic part normals select stable directional patches. All input
views are processed in canonical order, including deterministic ties.

Colours are copied as palette indices, without interpolation or RGB averaging.
Each exported voxel has one fixed colour for every face. Unobserved cells retain
the part's explicit material; colours are not propagated through the character.
The source-ID map marks these inferred materials in magenta and confidence maps
mark them red. The reported confidence is a facing/observation score, not a
calibrated probability. Interior cells are included in inferred-cell counts;
those counts are not the percentage of visible surface that is inferred.

Depth/ID buffers belong exclusively to the **offline authoring tool**. The game
renders the same compressed three-byte vertical runs as before, with one column
of coverage scratch. `voxel_assets.c:load_authored_marine` references the compiled
run/column arrays directly: no duplicate heap copy, part fitting, depth buffer,
or colour bake at runtime. Slots 16/17 still share the marine geometry with their
own palettes. Original animation selection, dimensions, facing and gameplay are
unchanged.

## Reproduction

Normal builds use the checked-in generated `src/voxel_marine_data.h`; Python
authoring packages are not game/build dependencies. Tested authoring versions:
Python 3, NumPy 2.2.3 and Pillow 11.1.0. The tool uses no neural model or external
editor. The optional packages can be installed in a dedicated environment:

```sh
python -m pip install numpy==2.2.3 Pillow==11.1.0
```

From the repository root, export exact source indices and both the frozen hull
baseline and current runtime models:

```sh
cmake --build build --config Release --target voxel_assets_test
mkdir build/voxel-report
build/Release/voxel_assets_test amiga/data build/voxel-preview.ppm build/voxel-report
```

Edit `assets/voxels/marine.json`, then regenerate:

```sh
python tools/voxel_author.py --snapshots build/voxel-report --output build/voxel-authoring --runtime-header src/voxel_marine_data.h
cmake --build build --config Release
build/Release/voxel_assets_test amiga/data build/voxel-preview.ppm build/voxel-report
python tools/voxel_author.py --snapshots build/voxel-report --output build/voxel-authoring --verify-runtime build/voxel-report
python tests/voxel_author_test.py
ctest --test-dir build -C Release --output-on-failure -R voxel
```

The last authoring command compares reloaded exports byte-for-byte with the C
runtime geometry and palette indices. Runtime source hashes prevent silently
using the authored correction with different marine art. Changing those source
files requires re-exporting their references and regenerating the correction.
The independent input-export mode works even when the old authored bank rejects
the changed source hashes:

```sh
build/Release/voxel_assets_test amiga/data --authoring-inputs build/voxel-report
python tools/voxel_author.py --snapshots build/voxel-report --output build/voxel-authoring --runtime-header src/voxel_marine_data.h
```

Open `build/voxel-authoring/index.html`. It includes original sprites with
landmarks, editable material regions, same-camera reprojections and differences,
colour and neutral turntables, normals, depth, part IDs, source ownership,
confidence, horizontal/vertical slices, elevated and top-down views. Separate
JSON metrics report missing/excess silhouette pixels and palette/RGB errors on
mutually covered foreground. A strict undilated hull is shown for diagnosis.

`marine-0.vox` through `marine-3.vox` and `barrel.vox` use the published
[MagicaVoxel version-150 format](https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt).
The paired JSON files preserve origin, pose, part transforms and provenance.
Cell indices convert from runtime `(x,y,z)` with Y down to VOX
`(x,z,size_y-1-y)` with Z up. A separate affine matrix in each marine sidecar
maps source-unit part coordinates and the bottom pivot to VOX boundary
coordinates. Palette index 1 remains opaque black.
The tool exports, reloads and verifies occupancy, palette and sidecar data, then
renders the reloaded volume with an independent 3D DDA. No MagicaVoxel/Vengi GUI
was available or used, so external-editor compatibility has not been visually
certified. Keep the JSON sidecar when sharing the VOX model.

## Results and limits

The new synthetic fixtures verify foreground-block occlusion, semantic material
rejection, opaque black, all 24 input-order permutations, asymmetric camera IDs,
agreement between cardinal projection and independent 3D traversal, strict-hull
monotonicity, and VOX/run-format round trips. The native ray oracle, real asset
checks and renderer worker-strip comparisons also pass. Barrel geometry is
preserved and the cylinder fixture compares occupancy to its analytic sampling
rule.

The authored marine is coherent as a connected volume, with 5,893..6,034 voxels
per pose. **It is not a closer pixel reproduction of every original drawing.**
Current source-angle silhouette IoUs are:

| Pose | Front | Right | Back | Left |
| --- | --- | --- | --- | --- |
| 0 | 0.775 | 0.745 | 0.632 | 0.745 |
| 1 | 0.683 | 0.686 | 0.792 | 0.686 |
| 2 | 0.717 | 0.709 | 0.613 | 0.709 |
| 3 | 0.619 | 0.638 | 0.592 | 0.638 |

Exact palette agreement on shared foreground is about 44..54%, versus 68..73%
for the old painted hull. For pose 0 the mean absolute RGB error rises from
roughly 12..14 to 20..26 on a 0..255 scale. These regressions are visible in the
report; they are not attributed entirely to source inconsistency. The inferred
part dimensions and coarse material masks also contribute. More detailed
fitting/painting can improve likeness without giving up separate depth.

Historical 0.85 median/0.75 minimum silhouette gates remain useful measurements
for older hull assets. They are no longer an acceptance test for the authored
marine. Strict global reprojection mode deliberately reports its deviations
alongside the ten pre-existing other-enemy conflicts. Passing connectivity or
IoU alone is not evidence that the modelling work is visually finished.
