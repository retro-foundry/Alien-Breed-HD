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
- `wallGeometry.ts` currently branches on `originZone === null`: it repeats the
  selected new-room wall materials on each perimeter edge. The imported path
  preserves individual wall records, sampling, phase, decorative fragments,
  heights and stream identities. Copying an imported room needs that provenance
  without also borrowing its original runtime identity.
- `cellGraphics.ts` similarly inserts new-room walls as a batch, but preserves
  imported wall records at their original draw positions. Arc and LightDraw
  point references are translated only for an established whole-room translation.
  Ordered water/object/clip/backdrop records must stay in their native order.
- `portalWalls.ts` distinguishes original openings from new or closed openings
  using original zone identities. A copy needs explicit correspondence between
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
