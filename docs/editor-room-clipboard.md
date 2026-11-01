# Room clipboard compiler contracts

Room clipboard work must preserve imported content as well as authored outlines.
The existing new-room constructor is not a faithful imported-room duplicator.
This is implementation evidence for the remaining R9/R11 work, not a completed
clipboard feature or a reduction to copying only simple rooms.

## Existing contracts

- `roomSource.ts` separates stable editable IDs, original runtime zone identity
  (`originZone`) and native settings provenance (`recordZone`). Runtime origins
  are unique. A copy needs a new editable ID and no original runtime identity;
  reusing `originZone` would make starts, objects and controllers ambiguous.
- Wall-source IDs already preserve per-edge native material provenance and are
  validated against `recordZone` for both imported and new rooms. They survive
  normalization and T-junction splitting. Do not replace these with nulls and
  then claim a material-preserving copy.
- `wallGeometry.ts` originally branched on `originZone === null`, repeating the
  selected new-room wall materials on each perimeter edge. The native path
  preserves individual wall records, sampling, phase, decorative fragments,
  heights and stream identities. Copying an imported room needs that provenance
  without also borrowing its original runtime identity. `roomMaterials.ts` now
  makes this decision from the retained native wall provenance separately.
- `cellGraphics.ts` similarly inserts new-room walls as a batch, but preserves
  imported wall records at their original draw positions. Arc and LightDraw
  point references are translated only for an established whole-room translation.
  Ordered water/object/clip/backdrop records must stay in their native order.
- `portalWalls.ts` distinguishes original openings from new or closed openings
  using material provenance and original/template correspondence. A copy needs
  explicit correspondence between
  its copied neighbour and the source relationship. Copying only one side of an
  opening needs a valid closure appearance; copying both must retain an opening.
- `sourceBuild.ts` reconstructs visibility and collision from source geometry.
  Copied source rooms must enter that same pipeline with fresh generated IDs;
  copied clip offsets, ordering masks or generated cells are not editable source.

## Positional and behavioural relationships

`roomGroups.ts` now moves starts, active non-avatar objects and incoming teleport
destinations with a room group. Copy differs from move: it must allocate fresh
object/CID records, remap internal copied destinations, and retain external
destinations according to the selected copy operation. A copy must not move the
existing player starts or redirect incoming links to the original rooms.

`objectAuthoring.ts` and `objectClipboard.ts` provide snapshot native records,
capacity checks, independent CIDs and atomic file relocation. Room clipboard
assembly should reuse those contracts and expose one history command for the
whole geometry/resource/object/link change. Any error must leave all source and
history unchanged, including failures near native room/object/point limits.

Original moving controllers are remapped by origin during source compilation.
A room clone cannot inherit a working door or lift merely by sharing its settings
template. It needs an independently allocated controller and synchronized wall
attachments, ranges and triggers. Switch targets need the same explicit mapping.
Do not silently turn copied mechanisms into static rooms or retain references to
the original moving walls. This is a dependency on the remaining R7 authoring work.

## Required verification

Check native source records and actual engine rendering for rooms with different
perimeter materials, partial decorative walls, upper/lower layers, water and
point-owning draw records. Check one-sided and paired portal copies, internal and
external teleport/switch destinations, independent door/lift cycles, captured
settings after original edits/deletion, capacity rejection, cancellation,
save/reopen and whole-operation undo. Original-level byte passthrough and simple
new-room copies do not establish this contract.

## Material compiler checkpoint

The wall, stream-order, topology/portal, appearance and lighting stages now
distinguish inherited native walls from a new perimeter-material selection.
Fresh room identities can retain their wall records, native closed boundaries,
paired openings, ordered arc/beam graphics and point-lighting provenance.
One-sided inherited openings still require an explicit closure appearance.
Source saves use the existing validated wall-source fields; no generated tables
or copied original zone identities are written into the editable baseline.

All 720 model tests pass. Tests compare translated interior decorations and
ordered arc/beam streams with and without original runtime identities; a static
native-material copy compiles through project save/reopen. Both Chromium/Firefox
native-render cases pass, inspecting four deterministic camera directions, as
do 16 adjacent portal/group/sprite cases. This is compiler acceptance and
record-preservation evidence, not complete original/copy frame parity for every
content class. Controller/switch copies are explicitly rejected before build
until independent authoring exists. Full clipboard assembly/UI, arbitrary
partial-surface reshaping and remaining relationship/profile work stay open.

Exact whole-room translations now preserve partial floor/roof/water polygon
footprints, point order, closure references, materials, sampling and native
lighting roles, including interior points. Translation detection tolerates
collinear portal splits without removing editable wall identities. Whole native
draw volumes retain their polygon records independently of convex collision
partitioning; arbitrary shape changes still require explicit footprint mapping.
The 723-test full model run passes, plus the subsequent partial-water/translation
focused checks. All 20 focused Chromium/Firefox native-copy, room-group, height
and sprite cases pass. These checks extend compiler support; they do not expose
or establish completion of the full room clipboard operation.

Room-source version 4 now supports explicit native property snapshots: water,
packed lower/upper brightness, control bytes, sky and floor sounds. Template
edits cannot change a room's captured values. Save/reopen, geometry/height edits,
native header assembly and Undo retain the snapshot; removing it explicitly
restores template inheritance. This addresses property ownership needed by room
copy assembly. Shared material resources remain separate dependencies.

All 727 model tests and four native Chromium/Firefox property-probe cases pass,
including static/animated words and later template changes. The native probe is
read-only and returns authored packed words, not evaluated animation intensity.
The lighting inspector now offers independent brightness controls only for the
native bump/smooth surface paths that consume zone brightness. Point-shaded
walls/floors and native water retain their own lighting contracts. A valid bump
material fixture viewed from a bounded inspection eye height proves actual
pixel changes and restoration through Undo/Redo and inheritance reset in both
browsers. All 40 focused controls/editor/property/height cases pass. The
inspection-height capture API changes only the paused camera and rejects heights
outside the room's air band. Full room clipboard authoring remains incomplete.

## Shared water animation attachments

Native-material copies now receive their own generated polygon pointers in the
source water animation group. `amiga/Anims.s` `DoWaterAnims` and
`src/objects.c` `do_water_anims` advance one group's 24.8 height and write every
listed room's water plane plus its polygon's quarter-unit height word. Copying
that relationship preserves the original group header, range, velocity and phase;
it does not allocate an invented independent oscillator. Original runtime room
identities remain unique. Door/lift/switch copies retain their explicit guards.

The original levels do not populate an oscillating group, so verification uses
an explicitly constructed native-format group with an original water polygon.
Save/reopen and source compilation prove distinct graphics pointers, translated
polygon coordinates and unchanged group headers/source bytes. The engine probe
reads actual polygon and zone water writes in one browser callback; both browsers
prove multiple changing heights stay equal between original and copy. The probe
only inspects loaded buffers and rejects invalid polygon offsets.

Native water entries can legally control a room different from the referenced
polygon owner. Existing cross-room relationships are preserved; copying that
polygon without an explicit room relationship map is rejected before export.
Independent water authoring, general cross-room mapping, complete clipboard
assembly and the remaining R7/R9/R11 requirements are still open.

## Moving-controller dependency inspection

`mechanismDependencies.ts` now resolves the controlled volume, primary surface,
each moving graphics owner, each collision-line incidence, and selected key /
switch conditions separately. This follows `Anims.s` DoorRoutine/LiftRoutine's
zone-plane, wall and floorline-flag writes. A graphics owner is not assumed to
own the associated collision line. All controllers in all sixteen original maps
resolve against their actual source records. Slot-before-file-alias tests also
cover the inspector's corrected lift lookup.

The mechanism inspector exposes these relationships in a collapsible attachment
table, with keyboard-accessible room-framing links and explanatory tooltips.
Framing keeps the mechanism selected and does not alter source. Moving-room copy
diagnostics identify the graphics-owning rooms and point to that inspection view.
This is dependency inspection needed for independent controller allocation, not
the controller allocation or complete room clipboard operation itself.
