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
- 301 unit/format/scenario tests currently pass. The portal compiler handles all
  16 maps, derives legacy-safe common clip bounds for alternate routes, and
  supplies exact sightline witnesses. All sixteen maps now pass deterministic
  binary assembly, content retention and matching original/rebuilt engine
  spawn-view checksums in Chromium/Firefox. See the geometry notes for measured
  differences from the shipped PVS and the remaining scenario requirements.
- `npm run build` produces the static app. Browser screenshots have been
  inspected. `.github/workflows/editor.yml` adds Windows/Linux browser CI;
  remote runs have now built the WebAssembly engine. The first Windows/Linux
  unit jobs failed because the former single all-map visibility test exceeded
  its five-second limit; it is now split into one complete rebuild test per
  map. The subsequent full Windows and Linux jobs both passed, including
  browser checks and static artifacts (run 41219390025, commit c1e3994).
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
- `surfaces.ts` provides typed current-format straight/see-through wall and
  floor/roof/water polygon records, used directly by the level reader. Tests
  decode and re-encode every supported surface in all sixteen maps byte for
  byte. Horizontal/vertical phase, authored sampling masks, scale, material,
  brightness and exact fixed/quarter-unit heights are retained separately.
  `polygonForCell` binds an explicitly selected polygon material to generated
  convex-cell point IDs without mutating the template. Contracts come from
  `renderer.c` wall/polygon cases, `WallRoutine3.ChipMem.s:itsawalldraw` and
  `AB3DI.s:itsafloordraw`; older AMOS packed-point output is not copied into
  current-format records. The suite now has 181 tests. Full wall-span/material
  authoring, curved surfaces, stream assembly/relocation and UI integration
  remain required before enabling geometry exports.
- `graphicsBuild.ts` reassembles graphics streams and relocates door/lift
  height-record and wall pointers, switch pointers, all 21 water animation
  groups, and the graphics/animation table headers. It can grow the zone
  prologue while leaving extra slots empty for the data compiler. Every shipped
  graphics file reconstructs exactly in original order; reversing streams and
  reconstructing again recovers the original bytes. Tests also compare the
  actual relocated target records, retain animation texture bases unchanged,
  and reject missing/out-of-range references. The current suite has 199 tests.
  This work exposed and corrected the switch-reader layout: byte +2 is type,
  +3 countdown, word +4 the point pair, long +6 the graphics pointer, and the
  final four bytes are runtime state, not X/Z coordinates. The authority is
  `Anims.s:SwitchRoutine` and `objects.c`'s switch loop. Chromium and Firefox
  both load all sixteen rebuilt maps with reversed streams and expanded zone
  prologues; every original/rebuilt spawn-view checksum matches. This stage still needs
  generated surfaces and complete source dependency remapping before new-room
  exports can be enabled.
- `dataBuild.ts` rebuilds current point coordinates/brightness and collision
  tables while preserving original control data, object points, shot pools,
  object records and player avatars. It accounts for every prefix byte before
  writing, relocates ZoneAdds after table growth, and rejects table truncation,
  invalid fields, missing collision destinations and unsupported layouts.
  All sixteen data files reconstruct exactly. Scenario checks expand both
  point and collision tables and verify unchanged room-region bytes and
  preserved relationships after relocation. The suite now has 216 tests.
  The word immediately **after** collision records is the zone count in all
  original maps, matching `LEVELSAVE`'s `DK[NZ]`; it is distinct from the
  obsolete exit-zone word before FloorLines. New room headers and dependency
  remapping are separate stages: this table writer does not claim that a new
  unused point or collision record is a complete authored room.
  Combined data-table expansion, visibility rebuilding and reversed/relocated
  graphics pass all sixteen original spawn-frame comparisons in Chromium and
  Firefox, exercising all three assembly stages together in the actual engine.
- `roomReferences.ts` resolves original player starts, active objects,
  teleport destinations, switch point-pair midpoints and mechanism groups to
  compiled source cells. Internal seams use a deterministic containing cell;
  deleted rooms and points stranded by edits produce explicit errors, not
  silently disabled dependencies. All sixteen original maps resolve, including
  their concave splits. Six originally out-of-outline objects retain their
  original relationships with warnings only while their source room remains
  unchanged. Object `+26` is GraphicRoom/runtime state, not another authored
  zone: this stage does not guess a remapping for it. Exit/game configuration
  and complete mechanism/graphics reference rewriting remain pending.
  The source suite now has 237 tests.
- Current editor containment now uses primary outlines only. Adjoining
  secondary collision fences had incorrectly rejected valid positions,
  including the original level 3 start. All 32 original starts and 52 teleport
  destinations pass containment; the browser regression moves that level 3
  start and undoes it. Actual game movement/collision code is unchanged.
- The target game slot is now project metadata rather than filename-derived
  preview state. Project format v2 stores it; v1 backups migrate to explicit
  slot 1 because they contain no saved slot. Imports from actual game level
  directories retain that directory's slot. Slot edits participate in the same
  undo/redo and autosave/dirty protection as binary edits, and the Project
  panel makes the setting available without loading the 3D engine. Exported
  game ZIPs use the chosen `levels/level_a/`–`levels/level_p/` directory and
  include installation instructions. The unit suite has 240 tests. Exit-zone
  configuration, source-room archive persistence and the full geometry build
  remain required; saving a slot does not establish those features.
  Chrome/Firefox checks cover slot undo/redo, project download/reopen, recovery,
  2D/3D view changes, actual engine slot selection and matching export paths.
  A slot-only edit with unavailable storage keeps the dirty marker/unload
  protection after a cancelled backup and cannot be discarded by opening a
  different map. All 28 targeted browser checks pass alongside the unit suite.
- Project v3 adds explicit exit setup, retaining v2 slots and migrating older
  unspecified exits to the original slot behavior. Level exit is available in
  Project, participates in undo and dirty/recovery tracking, and is written to
  game exports as `twolev.cfg`. This is a documented port extension, not an
  invented field in the original geometry. Its eight bytes are ASCII `ABEX`,
  big-endian version word 1 and signed zone word: -2 original slot, -1 disabled,
  0..255 explicit runtime zone. Exports always include it, including an explicit
  original-exit reset so old setup cannot survive replacing a custom map.
  `io_load_level_exit_zone` validates present files after level parsing; absent
  files retain the original table. The actual game loop now reads the state's
  end-zone table, initialised from `game_data.c` as before. Custom exit tests
  exercise actual completion, disabled completion, original reset and game ZIP
  re-import in Chromium/Firefox. The Windows native game and WebAssembly host
  build. Full source-room exit remapping (including concave splits), strict
  legacy export profiles and complete geometry authoring remain pending.
  All 32 browser scenarios pass in Chromium/Firefox, including all-original
  engine loads and combined data/visibility/graphics frame comparisons. The
  complete unit suite has 244 passing checks; see [the format and installation
  guide](level-setup.md) for the sidecar's exact contract.
- Project v4 stores `room-source.json` alongside the binary baseline, target
  slot and exit setup. Room source is part of the editor document and shared
  undo/redo commands, and participates in dirty tracking and recovery. Its
  checked schema retains stable identities, original/material references,
  concave outlines and the identity counter; partial gestures stay out of saved
  source. All sixteen original imports round-trip both representations without
  changing game bytes. Mixed source/record/setup edits undo together, while
  deleted/empty source remains recoverable. Unsupported fields, duplicate IDs,
  stale counters, missing templates and malformed outlines are rejected rather
  than dropped. Older project versions derive source from their known game
  geometry. Changed room source opens with an explicit limitation message and
  stale game export/preview is blocked until full compilation is integrated.
  Source archive/history integration is now in place; actual canvas topology
  tools and the complete source compiler remain required. The suite has 265
  passing unit checks and 22 targeted Chromium/Firefox browser checks, including
  backup, recovery and corrupt-source rejection without losing the open source.
- `zoneBuild.ts` assembles room records from explicit headers, primary/corner
  collision lists, visibility entries and rotation points. It owns both zone
  counts and all signed header-relative list offsets, and returns the new
  ZoneAdds addresses. Visibility rebuilding now uses this shared writer rather
  than an import-specific emitter. All 2,496 original room records reconstruct
  byte-for-byte; tests also reverse their physical order and reparse the
  relocated headers, and assemble an additional header with updated count and
  graphics tables. References are checked against the actual point/collision
  tables, with self-visibility, identity, size and uniqueness validation.
  The suite now has 283 passing unit checks. An extra header is assembly
  evidence only: generated surfaces/dependency rewriting and the complete
  source compiler are still required before exposing room creation.
  Chromium and Firefox both preserve all sixteen original spawn-frame
  checksums after combining data-table expansion, visibility rebuilding,
  reversed room blocks and reversed/relocated graphics streams.
- React now exposes source room drawing, corner dragging/numeric edits and
  deletion. A retained settings-template selector also supports empty source
  projects. Drawing uses either direction; Enter/first-corner completes one
  command, Backspace/Ctrl+Z removes a draft point, and Escape, tool changes and
  pointer cancellation do not commit unfinished work. Shared vertex edits move
  the incident portal fan atomically, inserting the corner in a larger wall at
  a T join. Crossings and overlaps reject before source changes. All sixteen
  original planar arrangements pass this validator. Source picking/drawing,
  fit bounds, list rows and inspector fields show changed outlines; selected
  outlines recenter and their fields retain keyboard-focus tooltips. Numeric
  fields now share a separate component rather than coupling inspector panels.
  Room tools and their control help state that game compilation is unfinished;
  source saves remain available and stale previews stop. The full source
  geometry compiler and playable new-room validation remain required.
  The unit suite has 301 passing checks. A full 40-scenario browser run passed;
  after layout/template refinements, 32 targeted Chrome/Firefox checks passed,
  including direction-independent gestures, cancellation, drag/coordinate
  edits, rejected crossings/overlap, deletion/undo and recovery from empty
  source. Updated room-inspector screenshots were visually checked.
- Graphics assembly now accepts validated replacement records of different
  lengths and additional room streams. Imported and generated records share
  one opcode decoder. Relocation tracks individual records and polygon material
  tails, retaining animation targets when a corner array changes size; pointers
  into replaced corner arrays reject until explicitly remapped. All sixteen
  original maps retain their animation targets in resized-record unit checks.
  Those checks deliberately use repeated corners to stress byte relocation;
  they are not visual-equivalence checks, since the renderer rasterizes the
  resulting zero-length edges too.
  An integration fixture assembles an isolated new room from explicit geometry
  and original material records using the shared data, room and graphics
  writers. Both browsers load its actual new spawn zone, pick its rendered
  walls and start gameplay. Its textured inspection view was visually checked.
  This exposed a preview-host initialization bug: game-loop context setup reset
  interpolation after the host set it, leaving paused views at the previous
  camera. The host now selects the current camera after context initialization
  and when stopping play. The unit suite has 319 passing checks. After rebuilding
  the WebAssembly host, all 44 Chromium/Firefox scenarios pass, including all
  sixteen original-map spawn-view comparisons through the combined writers.
  This fixture is compiler-stage evidence, not the complete source compiler:
  arbitrary drawn/edited/deleted rooms still cannot build or export gameplay.
  Keep that UI guard and defer the finished-editor deployment until the source
  compiler and remaining requirements are ready. The requested deployment
  destination is `../retro-foundry.github.io/abhd-ed`.
- The source reference planner now supplies per-cell teleport destinations,
  including inherited teleports on newly drawn rooms. It validates the actual
  selected template rather than an unused original room's teleport settings.
  `roomDataReferences.ts` writes resolved player-start zone words, every
  object's authored `OBJ_ZONE` field and generated room headers. It checks the
  preserved object-prefix offsets/counts and rejects unresolved dependencies
  before producing output. Opaque graphic-room state, mechanisms and graphics
  dependencies remain separate compiler work.
  All sixteen maps pass reference-write checks after reversing source-room
  order and convex decomposition, with byte comparisons proving unrelated
  data and header fields are retained. Tests cover new-room template teleports,
  deleted destinations, unresolved starts and stale prefix offsets. The shared
  new-room engine fixture now uses this writer; its native spawn, rendered wall
  picking and gameplay start still pass in Chromium and Firefox. The build and
  all 337 unit checks pass. Full source compilation and UI integration remain
  required; changed outlines are still guarded from stale exports/playtests.
