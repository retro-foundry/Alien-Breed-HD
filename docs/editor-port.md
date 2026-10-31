# React level editor port

The requested end state is a complete, friendly replacement for the AMOS level
editor, running directly in a browser on Windows and Linux. React and TypeScript
are the implementation chosen by the user. This work is ongoing: a viewer or a
map patcher alone does not satisfy the objective.

The user's additional P0/P1/P2 priorities and their completion evidence are
recorded in [editor-requirements.md](editor-requirements.md). All twelve
requirements are part of the goal; the checkpoint below is partial progress.

## Authority and evidence

- Original editor: `amiga/archive/amos/leved.amos`, SHA-256
  `26b7f6fa8255515509a3850e92f263ceebcec7e8b70952d2fa0b107e0459b779`.
- Readable listing: `amiga/archive/amos/leveld.asc`. `LEVELSAVE` (719),
  `LEVELLOAD` (2847), and `INITDEFORDER` (5691) define source project I/O,
  compilation, and visibility generation. The tokenized program was listed
  independently with [amostools](https://github.com/kyz/amostools), commit
  `cf30aa53f8422201fbad4e59ebb59649c7964326`; its routine locations agree within
  one line. The tool reports unresolved extension instructions; their use must
  be resolved before claiming a complete translation.
- This editor predates the shipped binary format. Its compiler writes a 1,600
  byte text prefix and graphics records that do not match the shipped files.
  Do not assume that reproducing its old compiler bytes makes a playable map.
- Current runtime authority: `src/level.c`, `src/game_types.h`, `src/renderer.c`,
  `src/objects.c`, `amiga/AB3DI.s`, and the 16 shipped level triplets in
  `amiga/data/levels/`. `amiga/doc/LevelFormat` contains older offsets too;
  check each field against current code and actual bytes.
- Compression authority: `src/sb_decompress.c` (`=SB=`, LH6 parameters).
  Level 1 decodes to 68,380 data bytes and 23,540 graphics bytes. Its runtime
  header begins at byte zero; graphics entries use the current word opcodes.
- The original embedded geometry helper has now been extracted and executed
  directly. See [editor-geometry.md](editor-geometry.md) for addresses, hashes,
  behavioral coverage and remaining PVS/compiler proof.

## Full scope and completion gates

Each item needs implementation and direct verification before completion.

| Requirement | Evidence needed | State |
| --- | --- | --- |
| React app on Windows, Linux and web | Reproducible static build; Chromium and Firefox browser runs; documented local/offline use | Static build and Windows Chromium/Firefox runs pass; Linux CI/offline delivery pending |
| Open all original levels | Parse all 16 actual level triplets; labelled 2D geometry and object checks | All 16 pass model and browser checks |
| Open/save AMOS projects | `.dat`, `.obj`, `.links` field contracts and round trips from original save/load routines | Checked reader/writer and constructed blank-source contracts pass; real authored-file, UI and compiler integration pending |
| Lossless game data import/export | Exact unchanged round trips, including unknown records and visibility data | All 16 exact triplet round trips pass |
| Geometry creation and editing | Points, sectors, shared portals, split/delete/join operations, upper/lower rooms; playable exports | Pending |
| Automatic geometry orientation | Either drawing/import direction produces the same valid geometry; shared portals paired automatically; no user-facing winding setting | Normalisation, partial-wall pairing and concave partition tests pass; canvas/compiler integration pending |
| Materials and lighting | Wall/floor/roof textures, alignment/scaling, vertex/zone light and animations; rendered comparisons | Pending |
| Objects and starts | Add/move/delete real object types, player starts, exit zone; engine playback | Pending |
| Doors/lifts/switches/water/teleports | Original conditions, links, heights, speed and animation records; engine scenarios | Pending |
| Control paths and visibility | Control points, links, zone assignments and ordering; original algorithm contracts and engine scenarios | All ordering masks/wall links rebuild; exact PVS/clips assemble for all 16 maps and preserve tested engine spawn views; broader scenarios, UI integration and control-path authoring pending |
| Modern editing workflow | Selection, drag, zoom/pan, snapping, numeric inspector, undo/redo, shortcuts and contextual help | Working for supported record edits; geometry tools pending |
| Reliable documents | Portable project files, autosave/recovery, clear dirty state, safe async loading, actionable import/export errors | Browser round trips, recovery and stale-load rejection pass; broader failure corpus pending |
| Actual game playtest | Exported map runs in the existing game engine; no substitute renderer presented as engine proof | All 16 original maps draw in the actual engine in Chromium/Firefox; record edits reload, chosen slot and isolated playtest pass; authored geometry scenarios pending |
| Robustness and accessibility | Malformed/truncated file corpus, bounded allocations, invalid topology checks, keyboard and responsive-layout runs | Pending |
| Delivery | Build/test instructions, CI artifact, usage guide, verified deployment when requested | Pending |

## Implementation order

1. Establish checked binary readers and a source-backed document model. Keep
   original bytes so unsupported fields cannot disappear on save. Build
   executable format fixtures against all shipped levels.
2. Build the working React canvas/inspector and document history around real
   imports, with errors, persistence, and round-trip exports from the outset.
3. Translate AMOS source-project records and compiler subsystems into readable
   domain modules, adapting proven current-format differences explicitly.
   Add topology editing and every original authoring feature to the same model.
4. Wire exported levels to the actual Emscripten game, with scoped file
   replacement. Validate each risky compiler change with the existing native
   parser and actual engine captures before declaring its feature complete.
5. Exercise full authoring scenarios, crash/recovery cases, all original maps,
   cross-browser use, and clean builds. Audit the table above against actual
   evidence; retain this goal until all requested capabilities are verified.

## Oracle plan

Use shipped binary triplets as immutable round-trip oracles. Run the existing
C decompressor/parser independently of the TypeScript implementation and record
counts, offsets and hashes. For new compiler features, compare a small authored
scenario with `LEVELSAVE`/`INITDEFORDER` dataflow and the current runtime's
consumer, then load the exported bytes in the native and web game. A working
AMOS runtime with its original extensions is not currently established; do not
claim emulator parity until it is actually run. Static contracts plus shipped
fixtures can establish current-format support independently.

## Current verified checkpoint

- `editor/` contains the React application, checked binary readers, pure
  TypeScript LH6 decoding, map/record models, atomic byte commands, project ZIP
  I/O, IndexedDB recovery, and the canvas/inspector interface.
- Unit/format tests cover all 48 C-decoder hashes, all 16 untouched
  binary triplet round trips, reversible field edits and all 2,496 room winding
  checks, 330 emulated original assembly geometry fixtures and independent
  rebuilds of all 49,797 original runtime ordering masks. Source-project tests
  check the AMOS blank-file layout, extra zone slot, text/banks, trailing-byte
  preservation and bounded writes; no live AMOS source-project run is claimed.
  Browser tests
  cover both Chromium and Firefox, including actual engine rendering of all
  16 maps, isolated playtests, focused-field saves and narrow-window navigation.
  A cancelled backup download with unavailable recovery storage retains the
  dirty marker and unload protection. Only a completed IndexedDB save clears
  that marker; requesting a download cannot claim that a file was saved.
- 104 unit/format/scenario tests currently pass. The portal compiler handles all
  16 maps, derives legacy-safe common clip bounds for alternate routes, and
  supplies exact sightline witnesses. All sixteen maps now pass deterministic
  binary assembly, content retention and matching original/rebuilt engine
  spawn-view checksums in Chromium/Firefox. See the geometry notes for measured
  differences from the shipped PVS and the remaining scenario requirements.
- `npm run build` produces the static app. Browser screenshots have been
  inspected. `.github/workflows/editor.yml` adds Windows/Linux browser CI;
  that workflow has not yet run remotely.
- Current `ToBack` is a **word** at zone offset 36 (`amiga/includes/Defs.i`).
  Every shipped value is 0 or -1. Do not expose its second byte as echo: that
  belongs to the older editor layout, not this runtime. Likewise, offset 28
  is `ToWallList` in the current definitions; its older background-SFX meaning
  is not proven for the current runtime. Original bytes remain preserved.
- All original collision outlines and floor/roof/water polygons tested have
  negative area in X/Z. The geometry layer handles either authoring direction,
  closing/repeated clicks, partial shared walls and concave decomposition.
  It still needs to be connected to the room creation tools and full compiler.
- Next substantial work is source-project UI integration, the geometry/compiler
  pipeline, broader renderer comparisons, and full
  authoring scenarios. Do not replace the full goal with a viewer or
  record-editing-only deliverable.

### Compiler layout facts for continued work

- Original zone boundary lists precede their 48-byte headers. Every original
  zone region is fully accounted for by boundary/header/visibility/rotation
  data; rebuilding the suffix is idempotent. Preserve the prefix's object,
  shot, point, brightness and floor-line tables until their own compiler stage
  is implemented and independently checked.
- Door/lift `+8` fields point into LEVELGRAPHICS, not level data. Expanding a
  graphics prologue must relocate them together with wall-record pointers and
  switch/water graphics references; texture-base values have different meaning.
- Rotation lists are word indices terminated by -1, with no count prefix.
  Curved-wall entries reference center/edge level points; the reader now
  captures both for correct rebuilt rotation lists.
- The original shot-pool expander in `src/level.c` allocates separate buffers;
  it does not insert bytes into level data. Source-file layout stays stable.
- The shipped runtime has **no exit-zone word before FloorLines**. Those bytes
  are the final reserved point brightness values. Current level completion uses
  `end_zones[16]` in `src/game_data.c` (read directly by `game_loop.c`). The older
  AMOS saver did emit EZONE there, but that contract does not apply to these
  shipped files. The mistaken runtime model properties were removed. The source
  project must own level-slot/exit configuration and export it through supported
  game setup; do not infer an exit from a filename or overwrite brightness bytes.
- Original source-project LSP is the lift's **starting endpoint**, not its speed
  (`LEVELSAVE` 1913–1918). Its seven option words remain opaque until their
  separate contracts are checked.
- Generated files must remain separate from editable source/history. The
  current fixed-byte command history cannot directly own a variable-size
  geometry/PVS replacement. Complete that source-model transition before
  wiring topology editing or automatic rebuilding into the app.
- `roomSource.ts` now provides the room portion of that transition: immutable
  stable identities, explicit original-record references, isolated drawing
  drafts, atomic create/outline-edit/multi-delete commands and unlimited source
  undo/redo. Drafts validate whole-coordinate limits and complete outlines;
  stale drafts/commands fail without changing source. All 2,496 original
  outlines import, and gesture-direction/cancel/stale-edit/history scenarios
  pass. This module is not yet wired into the canvas, project archives or full
  compiler; it does not claim that new rooms are playable yet. Original record
  references retain known surface/mechanism data without inventing defaults.
  Dependency checks, inter-room topology validation and geometry compilation
  remain necessary before exposing those commands in the application.
- `roomTopology.ts` adds the source-to-runtime topology stage. It distinguishes
  a room's original zone from its surface-record template, decomposes concave
  outlines, joins partial shared walls, preserves imported closed boundaries,
  retains existing point indices, and returns stable-source-to-runtime-cell
  mappings. It checks the current 256-zone/2,048-point/ten-edge limits and
  reciprocal portals before returning output. All sixteen imports match the
  separate visibility importer cell-for-cell; new concave rooms, partial joins,
  direction independence and deletion/portal closing have scenario coverage.
  The complete suite now has 144 passing tests. This stage does not emit game
  files yet: surface generation, graphics-pointer relocation, collision-line
  normals and object/mechanism/teleporter remapping remain required. Source
  topology is not enabled in the deployed application until those stages and
  the canvas/project integration are verified in the actual engine.
- `collisionBuild.ts` now generates primary collision records and adjoining
  solid-corner lists from compiled cells. Its formulas come from `LEVELSAVE`
  (`amiga/archive/amos/leveld.asc` 979–1102) and the actual displacement-byte
  consumer in `src/movement.c:get_floorline_shift_offsets`. All 11,190 shipped
  collision records rebuild byte-for-byte, including signed displacement
  components and zero initial touch flags; all 2,496 secondary corner-wall
  lists rebuild independently from primary geometry. Lengths and displacement
  components truncate towards zero, rather than round. Coordinate/span/length
  overflow and missing points/destinations reject before output is returned.
  The suite now has 162 passing tests. Complete-file assembly, surface records,
  pointer relocation and dependency remapping are still pending; this is not
  evidence that newly drawn rooms are playable yet.
