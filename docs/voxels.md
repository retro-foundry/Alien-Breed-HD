# Voxel world objects

Select **Options > Objects > Voxels**, or set `voxel_objects=1` in `ab3d.ini`.
The setting defaults to voxels and persists through the normal desktop INI
or browser local-storage path. Loading a saved game preserves the current
rendering choice. The new setting is appended to `GameState` to preserve the
field offsets in existing full saves.

Every loaded world billboard has a voxel model: enemies, corpses, decorations,
keys, weapons and ammunition on the floor, flying projectiles, impact effects,
flames, explosions and gibs. The first-person weapon and HUD are screen
overlays, not world billboards; existing vector objects use their polygon path.
Voxel mode never substitutes a sprite for a missing converted model.

## Asset conversion

Original packed WAD/PTR art and five-bit palette indices remain authoritative.
`src/sprite_frames.h` contains the original frame tables. Palette index 0 alone
is transparent, including when nonzero palette entries are black.

- **Marines:** four baked part models, one per walking step, with explicit depth
  guides and visibility-aware fixed colours. `assets/voxels/marine.json` is the
  editable source; `src/voxel_marine_data.h` is generated run/column data.
  The bank references these immutable arrays without allocating another copy.
  See [the authoring diagnosis, report and commands](voxel-authoring.md).
- **Other directional enemies:** four observations constrain one visual hull.
  Their normal conversion uses one source pixel of silhouette tolerance.
  No sprite planes are accumulated into the volume.
- **Barrel:** an explicitly tagged surface of revolution retains circular X/Z
  sections and the existing accepted radius profile.
- **Single-view objects, death poses and effects:** thickness is inferred from
  the source silhouette. These are approximations, not recovered 3D assets.

`MutantMarine.s` multiplies `ViewpointToDraw` by four and adds the walking step.
`AlienControl.s:ViewpointToDraw` supplies towards/right/away/left as 0/1/2/3.
Thus `{step, step+4, step+8, step+12}` belong to one pose. Local camera directions
are `-Z,-X,+Z,+X`, with image X mappings `{x,-z,-x,z}`; they preserve the port's
existing `objects.c:enemy_viewpoint` convention. Different poses are never
combined, and actor facing controls rotation.

Full source canvases and origins are retained. Any marine registration or
inferred geometry is recorded in the correction manifest. The original drawings
are not assumed to be consistent calibrated orthographic photographs.

Palette variants and repeated frames share geometry. BigRedThing's sequential
death frames 0..9 retain separate single-view models. Existing palettes,
brightness, animation selection, dimensions, bottom anchoring and gameplay rules
are preserved. Absent optional sheets remain absent.

Each voxel has one fixed palette material on all faces. Original hull models
choose visible source cameras and extend material through occupied cells.
Marine authoring instead checks source-resolution first-hit IDs/depths and
material-region ownership; unseen cells use explicit per-part materials.
Neither method averages conflicting RGB colours. Authoring buffers are never
used by the frame renderer.

The grid is capped at 48 cells per axis for current game assets. Vertical runs
store two Y endpoints and one palette index (three bytes). The bank contains
172 models, with 2,361,753 bytes (about 2.25 MiB) of column/run storage.
All runtime models are immutable and shared by renderer workers.
`voxel_assets_build` rejects marine source art that differs from its baked
references rather than quietly substituting a hull.

## Rendering without a depth buffer

The world still uses its existing back-to-front zone/object ordering, room
planes, portal ownership and two wall spans per screen column. No framebuffer
depth allocation or clear pass is added.

Inside a model, each screen column walks the X/Z grid from front to back.
Vertical runs are visited top to bottom below the view horizon, and bottom to
top above it. This ordering makes the first covered pixel the nearest voxel
surface, including top/bottom faces and views from inside the volume. A single
column of coverage bytes prevents a farther voxel overwriting it. Hit distance
exists only while emitting a pixel for the existing wall/room clipping checks;
it is never stored in a depth buffer.

The renderer writes the normal colour-word and RGB targets, uses the existing
worker strips and contributes to the existing object profiling counters. There
are no per-frame model conversions, sorts of individual voxels, heap allocations
or cube triangle meshes. Scratch is one byte per possible screen row per worker
(8 KiB at this port's 8192-row maximum), independent of screen width.

The existing painter ordering also determines interactions between separate
objects and rooms. It does not provide arbitrary intersecting-solid geometry
occlusion. The kernel currently uses floating-point projection and traversal;
its memory architecture requires no Z buffer, but performance on a particular
retro CPU still needs target-specific measurement.

## Validation and reports

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DAB3D_BUILD_RENDER_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure -R voxel
```

- `voxel_ray_oracle`: 38,400 rays against independent box intersections,
  covering holes, rotations, near-plane crossings, inside views and strip
  splitting. Also checks hard-hull monotonicity, all view permutations,
  mirroring, asymmetric coordinates, opacity, source-scale tolerance and
  analytic centre-sampled cylinder occupancy.
- `voxel_original_assets`: all unique conversions, frame grouping, palettes,
  separate death poses and effect coverage. Independently reprojects the models
  and compares stored metrics. Marine connectivity and plane-area checks catch
  structural regressions; they do not certify anatomical quality.
- `voxel_level_rendering`: 24 frozen views across three original levels, with
  one/two/four renderer strips and six projectile/explosion cases.

To export source/mask audits, sixteen-angle turntables, model snapshots and
quality measurements:

```sh
mkdir build/voxel-report
build/Release/voxel_assets_test amiga/data build/voxel-preview.ppm build/voxel-report
```

`index.html` covers all walking poses, the barrel and key; `inputs.html` audits
all 172 unique inputs. CSV files record source metadata, silhouette errors and
geometry statistics. `marine-*-baseline.json` reproduces the pre-part hull;
`marine-*-runtime.json` exports the actual current game model. The PPM preview
uses the game's perspective voxel kernel. No test opens a window or modifies
saved games/settings.

For neutral/colour controls, material ownership, depth, normals, part slices
and exported/reloaded VOX models, use the separate
[marine authoring workflow](voxel-authoring.md). It includes synthetic
occlusion/ownership tests and a byte-for-byte comparison with the runtime bank.

The historical median IoU >= 0.85 / minimum >= 0.75 targets remain measurements,
configurable through `AB3D_VOXEL_MEDIAN_IOU` and `AB3D_VOXEL_MIN_IOU`.
Regular tests enforce them for flying poses and worm poses 1/3.
`AB3D_VOXEL_STRICT_REPROJECTION=1` rejects every remaining target miss: the
four authored marine poses and ten pre-existing other-enemy cases currently
deviate. Marine anatomy is not accepted by passing those thresholds, nor are
its lower source-image scores concealed. The authoring report separates shape
and colour errors and documents the chosen depth compromises.

The barrel retains 46 circular occupied slices and its measured turntable
width of 30 pixels at the diagnostic scale. The key front silhouette retains
IoU 1.000 and remains thinner in depth than in width.
