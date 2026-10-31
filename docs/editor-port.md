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
| React app on Windows, Linux and web | Reproducible static build; Chromium and Firefox browser runs; documented local/offline use | Source-build integration passes Windows/Ubuntu CI and Chromium/Firefox; later milestones need their own CI; offline delivery pending |
| Open all original levels | Parse all 16 actual level triplets; labelled 2D geometry and object checks | All 16 pass model and browser checks |
| Open/save AMOS projects | `.dat`, `.obj`, `.links` field contracts and round trips from original save/load routines | Checked reader/writer and constructed blank-source contracts pass; real authored-file, UI and compiler integration pending |
| Lossless game data import/export | Exact unchanged round trips, including unknown records and visibility data | All 16 exact triplet round trips pass |
| Geometry creation and editing | Points, sectors, shared portals, split/delete/join operations, upper/lower rooms; playable exports | Source outlines, static new-room openings, imported doorway closures and replacements with authored neighbours build/play/export; unassigned boundaries, broader topology operations and controller groups pending |
| Automatic geometry orientation | Either drawing/import direction produces the same valid geometry; shared portals paired automatically; no user-facing winding setting | Normalisation, partial-wall pairing and concave partition tests pass; canvas/compiler integration pending |
| Materials and lighting | Wall/floor/roof textures, alignment/scaling, vertex/zone light and animations; rendered comparisons | Pending |
| Objects and starts | Add/move/delete real object types, player starts, exit zone; engine playback | Pending |
| Doors/lifts/switches/water/teleports | Original conditions, links, heights, speed and animation records; engine scenarios | Pending |
| Control paths and visibility | Control points, links, zone assignments and ordering; original algorithm contracts and engine scenarios | All ordering masks/wall links rebuild; exact PVS/clips assemble for all 16 maps and preserve tested engine spawn views; broader scenarios, UI integration and control-path authoring pending |
| Modern editing workflow | Selection, drag, zoom/pan, snapping, numeric inspector, undo/redo, shortcuts and contextual help | Working for supported record edits; geometry tools pending |
| Reliable documents | Portable project files, autosave/recovery, clear dirty state, safe async loading, actionable import/export errors | Browser round trips, recovery and stale-load rejection pass; broader failure corpus pending |
| Actual game playtest | Exported map runs in the existing game engine; no substitute renderer presented as engine proof | All 16 originals draw; normal source build/export, renumbered room visits/picking, authored starts/objects/teleports pass in Chromium/Firefox; broader geometry/gameplay scenarios pending |
| Robustness and accessibility | Malformed/truncated file corpus, bounded allocations, invalid topology checks, keyboard and responsive-layout runs | Pending |
| Delivery | Build/test instructions, CI artifact, usage guide, verified deployment when requested | Pending |

## Implementation order

Current integration: `playableBuild.ts` is the shared preview/export entry point.
Supported source edits compile automatically, with explicit appearance choices
for new rooms; generated artifacts never replace editable source in saves.
Static new-room joins generate wall openings automatically. Imported doorway
closures require an explicit saved wall style and lighting selection; they
replace the closed edge's old wall pieces while retaining other authored walls.
Historical checkpoint notes below that retain the old blanket source-build guard
are superseded by this integration. Unsupported portal/controller/surface cases
still fail explicitly, and the remaining requirements are not complete.

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
  At that checkpoint the UI guard remained in place; the later source-build
  integration supersedes it. Defer the finished-editor deployment until the
  remaining requirements are ready. The requested deployment
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
- The data-table writer now separates a complete-build prefix from the
  retained-original-room relocation path. `buildDataPrefix` accepts regenerated
  point/brightness and collision tables and the target room count, writes the
  last-zone header and count trailer correctly, and preserves actual control,
  object, shot-pool and avatar records. This removes the original collision
  count restriction for complete geometry assembly. It is not independently a
  playable file: fresh room records, graphics and reference maps are required.
  The original-room path retains its table-shrinking guards. All sixteen
  original prefixes reconstruct exactly, and a reduced-table case checks
  retained prefix bytes, relocated collision offsets and rejected invalid
  room/point counts or neighbours. The generated-room fixture now assembles its
  prefix directly with the new room count. Build and 338 unit checks pass.
  Both browser engines pass the generated-room scenario and all sixteen
  original spawn-view comparisons through the combined rebuild (four tests).
- Visibility membership, ordering masks, rotation references, clip lists and
  solid-wall point links now have one shared binary-table writer for imported
  logical rooms and generated convex cells. `buildSourceVisibility` consumes
  compiled topology and the generated graphics' actual point references;
  source builds include all points instead of retaining the original reserved
  switch-point gap. Imported rebuilds preserve their prior room ordering and
  original link-table length. The new-room fixture uses this writer for its
  isolated room's visibility, rotation list and solid links. A generated joined
  corridor test walks the native left/-1/right/-2 clip format, checks self-first
  ordering masks and the continuous outer solid boundary, and verifies rejection
  of missing graphics, invalid points or mismatched room identities.
  This audit also found a runtime off-by-one: header +14 is the last point
  index, as confirmed by AB3DI.s's `Points + 4 + index*4` brightness address.
  Brightness and renderer clip bounds now share `level_point_count`, allowing
  the final generated point. A C regression checks final-point lower/upper
  brightness, one-point counts and invalid bounds; it runs in the editor CI.
  The Windows game and WebAssembly preview build successfully, and all 340
  editor unit checks pass. After rebuilding the engine, both browsers pass the
  generated-room scenario and all sixteen combined-rebuild spawn comparisons
  (four browser tests). Full source surface/mechanism compilation and UI
  integration remain required before enabling edited-room exports/playtests.
- Editable source now retains one original boundary-wall identity per directed
  edge (`wallSources`). Automatic winding/start-corner normalization, corner
  moves and shared T-junction splits carry those identities with the geometry.
  Convex-cell compilation consumes them instead of trying to rediscover a
  moved wall at its old coordinates; internal decomposition edges remain
  unassigned. This is required to retain per-wall materials and authored
  closed-boundary flags during surface compilation.
  Room-source snapshots are now version 2 inside the existing v4 project
  container. Version 1 snapshots migrate exact geometric wall matches on
  imported rooms; already moved/unknown edges remain unassigned rather than
  receiving guessed materials. New-room edges also remain unassigned. Unknown,
  incomplete and out-of-template references reject without replacing source.
  All sixteen original source imports and legacy migrations retain their wall
  identities. Tests cover a moved corner that changes the canonical first
  corner, history, moved-wall project round trips and retained solid closure
  across a moved shared boundary. The build and 344 unit checks pass. Surface
  generation, material authoring and full source compiler integration remain
  required; edited-room gameplay/export guards remain in place.
  All 46 Chromium/Firefox scenarios pass, including the new imported-wall edit,
  downloaded-project and browser-recovery check and all original engine views.
- `surfaceGeometry.ts` now binds complete floor, roof and water material
  records to generated convex cells. It retains original records exactly when
  both the footprint and point identities are unchanged, otherwise writes the
  generated corners with the original height, sampling scale, tile and light.
  It retains record offsets as dependency identities for the graphics/animation
  assembler. Partial or degenerate authored polygons remain byte-preserved on
  unchanged geometry; regenerating them needs an explicit geometry mapping
  rather than silently expanding their material across the whole new room.
  Tests force polygon regeneration for every material template in all sixteen
  maps, including both concave imports, and check reindexed points separately.
  Wall-span binding uses the selected record's authored texels/world-unit ratio
  over an explicit physical interval. Shared split endpoints receive identical
  texture coordinates; horizontal/vertical phase, masks, heights and light stay
  separate. This preserves authored density rather than guessing the lost AMOS
  repeat/fit mode from a texture's dimensions. All 7,628 original straight wall
  records bind back byte-for-byte. Span continuity, resize, limits and invalid
  dependencies are covered. The generated-room fixture now uses both surface
  binders; its textured view was inspected, and native spawn, wall picking and
  gameplay start pass in Chromium and Firefox. Build and 364 unit checks pass.
  Complete room stream generation, wall attachments, moving-wall dependencies,
  material authoring and source compiler/UI integration remain required.
- Graphics assembly now accepts complete lower/upper record lists for an
  existing room, with optional original-record identities. This supports
  insertion, removal, reordering, changed record lengths and adding an upper
  stream without treating every original record as mandatory. Identity-bearing
  records retain their role and layer; duplicate/missing identities and mixed
  whole-stream/individual edits reject. A removed animation target rejects
  instead of binding to an arbitrary identical-looking record. Imported stream
  lists preserve absent versus existing empty upper streams.
  All sixteen maps round-trip exactly through complete record lists. Reordered
  record tests retain door/lift headers and wall targets, switch targets and all
  twenty-one water animation groups. Tests also cover insertion/removal, upper
  streams, role conflicts and invalid inputs without mutating the original.
  The engine comparison pipeline now routes every original room through this
  path, alongside table expansion, visibility rebuilding and physical stream/
  room reordering. The build and 382 unit checks pass. Whole-level room identity
  changes, deleted-room dependencies, cloned mechanism/water links and complete
  source stream generation remain required before source compiler integration.
  Both browsers preserve all sixteen original spawn views through complete
  stream assembly, and both generated-room scenarios pass (four tests).
  Resized polygons also produce identical files through complete-stream and
  individual-record inputs across all sixteen maps.
- Animation tables now have a shared import/assembly model for 18-byte moving
  room headers and variable wall lists, all 21 water groups, and eight fixed
  switch slots. Authored room/line/point and graphics relationships are explicit;
  rule/runtime bytes, wall texture phase and trailing opaque bytes are retained.
  The graphics builder now emits these tables instead of copying and patching
  the old tail, and accepts explicitly regenerated animation tables. Source
  layouts with unexplained inter-section data reject rather than losing bytes.
  Every original map's animation region reconstructs byte-for-byte. Tests
  verify relocated section/graphics addresses, changed room and collision-line
  references, expanded water lists, absent versus empty-present tables, fixed
  switch slots, reserved terminators and limits. The parser now includes the
  final permitted entry at its 256-mechanism and 64-wall bounds; dedicated
  assembly/reparse tests cover those boundaries. Active switch pairs and total
  generated graphics size are validated. The build and 402 unit checks pass.
  This is animation-file assembly evidence, not completed authoring or source
  dependency mapping: automatic allocation/cloning of moving walls, water links
  and switch attachments across changed room geometry remains required.
  Runtime review also found two water-list consumers capped at 128 reads,
  which left the terminator unread for a full group and misparsed following
  headers. Normalization and animation now allow 256 entries plus the
  terminator. A C regression checks 0/1/127/128/256 entries, normalized room
  aliases, subsequent group boundaries, live heights and graphics writes. It
  fails against the old normalizer and independently against the old updater;
  the fixed consumers pass. The regression runs in the editor CI. Windows and
  WebAssembly builds pass. Imported-animation validation before preview and the
  remaining source dependency/authoring stages still need completion.
  After rebuilding the engine, both browsers pass the generated-room test and
  all sixteen original-map spawn comparisons through animation reassembly
  (four browser tests).
- Validated level imports and preview loads now check live animation
  relationships before handing bytes to the native engine. The validator uses
  the shared table decoder, checks room and physical collision-line bounds,
  polygon-height targets for moving rooms, moving-wall record targets, switch
  activation point pairs and phase/material targets, and water polygon targets.
  SwitchRoutine pointers address wall byte +10, not the record start; all 49
  active shipped switches follow that contract. Inactive switch slots retain
  unused stored values without dereferencing them. Room-file aliases are
  accepted where the native animation loader resolves them.
  `openLevel` is the validated import boundary; the low-level `parseLevel`
  remains available for inspecting damaged data. Preview validation failures
  stop playback and report the link that needs repair. Corrupt candidates do
  not replace the open project. Tests reject out-of-file and misaligned/incorrect
  record targets, missing rooms and collision lines, broken switch point pairs
  and invalid water links, while accepting valid generated water animations and
  ignored inactive-slot fields. Build and all 412 unit checks pass.
  This validates animation references, not every authored gameplay constraint;
  full source compilation, attachment remapping and mechanism authoring remain
  required.
  All 48 Chromium/Firefox scenarios pass, including malformed archive/triplet
  rejection with retained preview/project/recovery and all original engine
  views. Runtime decoder review found another required checkpoint for generated
  moving rooms: `level_parse` still uses data-header +16 (last zone index) as
  the door/lift normalization count. The following checkpoint addresses this
  before arbitrary generated mechanism allocation is enabled.
- Native animation decoding now uses the graphics prologue's full slot count
  before normalizing doors/lifts/water, rather than the inclusive last index
  in data-header +16. The later stored header value remains unchanged for
  existing inclusive loops. A shared parser helper checks the 1..256 capacity,
  table alignment and consistency with the authored room count.
  A C regression using original Level 1 data fails against the old decoder
  when a door is assigned to its final existing room. The fixed decoder retains
  both final-slot doors and lifts, including a lift file alias resolving to that
  final slot. It also retains every original controller across all sixteen
  maps. Boundary tests cover one slot, maximum capacity and invalid tables.
  The regression is included in CI using embedded original map files.
  Import validation now distinguishes direct door/switch indices from the
  fallback file aliases supported by native lift/water normalization. Windows
  and WebAssembly builds and all 413 editor unit checks pass. Full moving-room
  authoring, generated dependency allocation and source compiler integration
  remain required.
  Before/after native comparison also identifies a shipped-map defect: Level
  14 contains ten authored lifts, but the previous decoder retained only nine.
  The corrected decoder retains all ten. Both browser engines pass the
  generated-room test and all sixteen rebuilt/original spawn comparisons after
  rebuilding the WebAssembly host (four tests).
- Graphics assembly now supports a validated permutation of runtime room IDs,
  including stream headers, graphics rows, ZoneAdds and imported door, lift,
  water and switch room references. Explicit animation tables use their supplied
  target IDs. Native lift/water aliases resolve before remapping; source graphics
  identities remain available for pointer relocation.
  An integration fixture reverses every room ID across all sixteen maps and
  rebuilds zone blocks, clip ordering, portal links, teleports, starts, object
  rooms and exit configuration. Unit checks also require identity mapping to
  preserve original graphics bytes. All 446 unit checks and the build pass.
  Chromium and Firefox pass native comparisons at both player starts and eight
  directions around both the original and remapped numeric room 53, where
  present. Every sampled frame matches its original checksum. This covers room
  renumbering, not all viewpoints or gameplay. The visibility batch now runs as
  sixteen separate test cases so each map has its own existing timeout.
  Arbitrary deletion, one-to-many geometry and animation attachment allocation,
  source-reference authoring and full compiler/UI integration remain required.
  Deployment remains pending that working authoring pipeline.
- Added a generated graphics assembler whose room set is independent of the
  imported room count. It accepts complete generated streams, dense runtime
  IDs and compiled zone-data offsets, so removed rooms do not have to survive
  as unused original slots. Present empty upper streams remain distinct from
  absent streams. All sixteen original graphics files rebuild byte for byte
  through this path.
  Generated copies retain explicit source record identities. If an animation
  target has multiple copies, the compiler must select its exact room, layer
  and record; missing or ambiguous attachments fail before output is returned.
  The animation writer supplies door/lift/water/switch context and controlled
  room, plus the collision line for moving walls. Polygon-tail relocation is
  shared with the existing writer. Rule bytes, texture phases and inactive
  switch fields retain their existing format contracts from `amiga/Anims.s`.
  Tests cover independent controllers addressing duplicated surfaces, explicit
  attachment selection for every animation kind, room removal, absent/empty
  streams, missing links and the full 256-room boundary.
  The build, type checks and 465-test unit run pass; the subsequently added
  capacity regression also passes. Both browsers pass the all-sixteen-map
  renumbering comparisons using this assembler, including both starts and
  the room-53 camera sweeps described above (two native engine tests).
  Source geometry-to-stream generation and automatic attachment allocation
  still need implementation before normal UI builds can use this assembler.
- Wall geometry now binds original boundary material records to changed and
  split source edges, retaining record identities for animation mapping.
  Partial boundary decorations get regenerated endpoint IDs in the returned
  compiler point table. Texture density, horizontal/vertical phases, sampling
  masks, brightness and fixed-point heights retain their authored meanings
  from `WallRoutine3.ChipMem.s`. Interior decorations support exact whole-room
  translation. New-room perimeter materials are supplied explicitly; concave
  decomposition does not generate walls across its internal portals.
  An audit of all 7,628 shipped straight walls found 7,580 exact boundary
  matches, 46 partial spans and two interior decorations. Unchanged room walls
  retain their bytes. Translation/regeneration checks cover every original
  straight wall across all sixteen maps, including total segment lengths and
  strip coverage when a room decomposes. Further cases check stretched walls,
  split-edge texture continuity, partial decorations and concave perimeters.
  A diagonal decoration exposed floating-point cancellation one texel below
  an exact boundary; the shared span binder now snaps only values within eight
  scaled machine epsilons of an integer and retains truncation elsewhere.
  The generated-room integration fixture now uses this wall stage. Its actual
  native render, picking, spawn and gameplay checks pass in both browsers.
  All 503 unit checks and the build pass.
  This remains a compiler stage: general non-rigid interior-decoration mapping,
  stream/control ordering, new portal height openings, attachment allocation,
  source-reference authoring and full UI build/export integration are required.
  Fractional decoration endpoints fail with a grid-alignment diagnostic rather
  than exporting rounded, off-boundary geometry. No new editor deployment yet.
- Complete cell graphics now combine regenerated polygons and walls at their
  original stream positions, retaining clip setters, object draw passes and
  backdrop controls in order. Upper-stream absence remains distinct from an
  empty stream. New-room wall batches use an explicit insertion position in
  the selected material template, including placement relative to water and
  before/after-water object passes. The compiler returns all generated graphics
  point references for the visibility/rotation stage.
  `AB3DI.s:LightDraw` consumes two point indices. Opcode 6 parsing now exposes
  and validates both instead of treating them as unreferenced payload bytes.
  Arc and light-beam points translate with an imported room, retaining all
  other payload fields; unsupported reshaping requires geometry mapping.
  All sixteen maps pass ordered cell-stream checks. Whole-room records remain
  byte exact; decomposed rooms preserve source record order. Dedicated cases
  check water/object sequencing, invalid insertion settings, arc/beam point
  movement, complete rotation references and unchanged input geometry.
  The generated-room fixture uses the complete stream builder. Its native
  render, selection, spawn and playtest checks pass in Chromium and Firefox.
  All 522 unit checks and the build pass. Source animation attachment mapping,
  point brightness generation, portal height openings, general decorative
  geometry mapping and full compiler/UI integration remain required before
  deployment.
- Generated point lighting now has an explicit compiler contract: preserve a
  retained identity, copy a selected original point, sample an original segment
  by a rational fraction, or assign lower/upper words directly. New or moved
  identities without assignments fail before data assembly. Coordinate matches
  do not silently select lighting from another original point.
  Current `AB3DI.s:currentPointBrights` and `level_get_point_brightness` define
  the packed words: signed low-byte brightness and high-byte animation controls.
  Segment sampling is a new authoring operation; it interpolates signed bases
  with truncation towards zero, retaining matching active control bits rather
  than interpolating packed words. Endpoints and equal words remain exact.
  Incompatible active animation settings require explicit point lighting.
  Lower and upper channels are handled independently; original inactive bits
  remain exact on retention/copy. The older AMOS LEVELSAVE brightness layout
  is not used as the current runtime format.
  All original point words across all sixteen maps pass retention checks.
  Tests cover signed/static and animated sampling, endpoint retention, explicit
  assignments, missing bindings, invalid targets and unchanged input data.
  The generated-room fixture now uses this lighting builder and its final
  graphics point table. All 540 unit checks, the build and Chromium/Firefox
  native generated-room tests pass.
  Automatic source-geometry lighting provenance and lighting authoring controls
  remain required, alongside source animation attachments, portal openings,
  decorative mappings and complete compiler/UI build/export integration.
- Automatic geometry lighting now derives imported-room assignments from
  source boundary identities and graphics-record endpoint roles. Partial wall
  decorations, arcs and beams retain their own endpoint lighting. Polygon
  corners use their own material record's source points; inserted straight-edge
  points sample the lower/upper channels separately. New rooms receive one
  explicit initial lighting selection rather than per-point assignments.
  Level 1 exposed coincident points with distinct lighting, including separate
  material records inside a room. The lighting assembler now allocates separate
  graphics identities for those values, patches only the affected draw records
  and polygon closing corners, and refreshes rotation references. Collision
  vertices and portal geometry remain shared. Identical copies are reused, and
  exceeding 2,048 points fails without mutating the source graphics.
  All sixteen maps pass unchanged-point retention and translated-geometry
  checks. Every generated material point is compared with its corresponding
  original endpoint or explicit segment sample, including all packed words.
  A separate gradient case checks an inserted midpoint with static lower and
  animated upper lighting. Level 1's visibility membership, clipping prefix and
  existing solid-wall links remain identical after graphics lighting copies;
  only unused link slots and required rotation identities are added.
  The native generated-room fixture now supplies a room-level lighting choice
  and uses automatic assignment. All 558 unit checks, the build, and both
  browser engine tests pass. These native runs cover that generated room, not
  full translated-level gameplay. General nonuniform bent-wall light sampling,
  automatic animation attachment allocation, portal height openings, remaining
  decorative mappings and complete compiler/UI integration are still required.
- Animation attachments now bind native moving-room polygon targets, animated
  collision lines and wall records to generated identities. The mapper checks
  the compiled line coordinates/neighbours against their advertised identities,
  preserves native rule/state/texture-phase bytes, and resolves graphics copies
  explicitly for the graphics assembler. New rooms using an original material
  template do not inherit that original room's animation controllers.
  Water polygons can fan out across runtime cells within their existing shared
  native group. Switch activation pairs retain their stored geometry, including
  authored offsets: unchanged pairs stay byte exact; translated/rotated walls
  carry those offsets with them and allocate consecutive point pairs as needed.
  Rotation snaps the generated pair to whole coordinates and enforces bounds.
  Copying the stored pair is necessary: regenerating a nominal 64-unit segment
  does not reproduce all 49 shipped activation pairs.
  Moving-room volumes and switched volumes that split into several cells still
  require synchronized controller grouping. The mapper rejects that case and
  repeated collision-flag writes instead of creating independent controllers
  with altered trigger behaviour. Those authoring capabilities remain required.
  Tests cover all sixteen original and translated attachment sets, all 49 switch
  pairs, independent water polygon targets and controlled-volume rejection.
  The native isolated-room fixture now uses this attachment mapper and the
  generated graphics assembler. Rendering, selection, spawn and playtests pass
  in both browsers. Collision assembly also exposes its typed segments, and
  visibility assembly accepts an original order hint while rebuilding masks.
  A local complete-source compiler prototype now assembles all runtime tables
  and passes deterministic data builds for all sixteen maps. Its broader native
  comparison remains unverified: Level 2 room 67 changes pixels after convex
  decomposition. Paired captures expose shared wall-endpoint draw ordering and
  a remaining small floor/sprite region. Experimental static-batch changes were
  removed; they did not establish complete fidelity. The complete compiler and
  this failing browser scenario remain local work in progress, outside this
  checkpoint. Logical draw-room membership versus collision cells needs further
  tracing before the full Build & Play/export path or deployment is enabled.
- The complete source-build API now connects topology, material streams,
  lighting, collision, animation attachments, source references, visibility,
  zone assembly and graphics relocation into a validated output document.
  Source bytes and room history remain unchanged; repeat builds are identical.
  New portal spans, split controllers and split exit destinations still fail
  explicitly where their authoring/runtime adapters are not implemented.
  The native decomposition mismatch was isolated using Level 2's marine: with
  that same sprite hidden in both inputs, the remaining floor-region pixels
  matched. Rendering ownership must therefore remain an authored-room identity
  even though collision and AI ownership use convex runtime cells.
  `buildDrawTopology` reconstructs each authored perimeter. Draw cells retain
  their source-ordered static material batch; the native renderer treats
  sprites, shots and effects in that authored draw room as local, avoiding
  artificial sprite-spill clipping. Gameplay/collision room IDs remain distinct.
  This does not implement synchronized door/lift/switch controller groups.
  Optional `twolev.cfg` version 2 carries those draw identities: `ABEX`, BE word
  version 2, signed exit word, unsigned slot-count word, then one representative
  runtime-room word per slot. Owners must be in range and own themselves.
  Version 1 remains byte exact. The pure native setup decoder applies valid
  files atomically; malformed sizes, roots and exit destinations leave state
  unchanged. Parsing new geometry clears old grouping, and missing/v1 setup
  retains original per-zone rendering. Limits include the full 256 slots.
  Grouping survives project/game archives and commands. The iframe validates
  the extended setup before loading it. The browser harness now builds the
  current production app itself and reserves its test server, preventing a
  stale decoder/runtime from being mistaken for a source-compiler failure.
  Complete rebuild comparisons retain direct shipped-frame checks at both
  player starts. Views inside decomposed imports use an independent
  visibility-only reference with original geometry/graphics/object ownership.
  A separate Level 13 case proves why: its shipped white gap is repaired by
  rebuilding only visibility/clipping tables, and the full compiler matches
  that repaired frame exactly. The test does not accept arbitrary pixel drift.
  Native setup/lifecycle regressions and Windows/WebAssembly builds pass.
  All 595 unit checks and all 56 Chromium/Firefox scenarios pass, including
  both full-source map comparisons and the isolated sprite/visibility cases.
  The source build remains outside the normal UI
  workflow until stable source gameplay references, portal spans, controller
  grouping and other outstanding authoring requirements are completed.

- Authored player starts now reference stable editable room IDs rather than
  generated convex cells. The Starts inspector offers room names and whole-unit
  coordinates for source edits; map drawing/hit-testing and navigator focus use
  the authored location. Dragging locates the editable room, including newly
  drawn rooms. Original unchanged-map start edits retain their binary workflow.
  Source start commands participate in atomic undo/redo, stale-edit checks and
  saved-change tracking. Project format 5 stores `player-starts.json` alongside
  the original binary baseline and room source; versions 1–4 remain readable,
  and saves without source starts remain version 4. Invalid schemas and missing
  files are rejected. A deleted/reshaped start room remains recoverable as source
  but blocks compilation with an actionable dependency error.
  Source compilation locates each start's containing convex cell and patches
  both header coordinates/room and its multiplayer avatar's room, graphics room
  and 16.16 object-point coordinates, following `commands.setStart` and header
  offsets +46/+50. Opaque import bytes remain unchanged in the source project.
  Both player starts, split-room lookup, save/recovery and protected avatar edits
  have direct tests; native browser spawn tests exercise a newly drawn room.
  Validation: production build and all 601 model tests pass; the four authored
  start checks pass in Chromium/Firefox, along with the 20 existing browser
  checks for editing, project persistence, recovery and rejected imports.
  Existing-object and teleport destination source authoring, portal wall spans,
  and the normal source build/export integration remain unfinished. No finished
  editor deployment has been made.

- Existing objects now have stable authored room placements as well. Objects
  keep their original record/CID identities and other binary attributes; an
  explicit null room means deletion. The model, inspector room picker, map
  drawing/hit-testing, drag handling, navigator focus, dirty state and recovery
  all use authored placements. Source-only rooms are not misrepresented as the
  template's old zone, and unresolved room IDs remain repairable dependencies.
  The shared room-position form supplies controls/help for both objects and
  player starts. Ordinary decimal coordinates are retained in editable source;
  the compiler rounds them to the native 16.16 point precision, following
  `commands.moveObject`, and locates the resulting point's convex cell.
  Builds patch OBJ_ZONE (+12), OBJ_GROOM (+26) and the CID's point-table entry;
  deletion patches both room fields to -1. Player avatars remain controlled by
  Starts, including their map markers. Invalid upper-floor destinations and
  shared point-slot moves are rejected before a corrupt placement can be built.
  Project format 6 adds `object-placements.json`, with optional source starts;
  formats 1–5 remain readable and saves without object placements keep their
  previous format. Missing/malformed files, duplicate IDs and missing original
  records are rejected. Geometry-only matching and full source-vs-runtime
  matching are now separate checks, so preview/export cannot use stale gameplay
  bytes when geometry itself is unchanged. Native browser comparisons isolate
  a moved health pickup in a newly drawn room against its deleted counterpart.
  Production build and all 607 model tests pass. The four new Chromium/Firefox
  checks pass for browser placement/drag/delete/undo/recovery and real-engine
  present/deleted sprite rendering; existing room, start, persistence and
  general editing browser regressions also pass.
  Adding new object records, source teleport links, portal openings and normal
  Build & Play integration remain pending; this does not complete the editor.

- Authored teleporter links now use stable source-room IDs for both emitter and
  destination. Per-room overrides preserve the imported template until changed;
  an explicit null destination disables an inherited link. The source compiler
  locates the landing point's convex cell and writes destination/X/Z to header
  +38/+40/+42 in every emitted cell of the logical source room. Missing or
  reshaped destination rooms are recoverable source dependencies and block the
  build. Removing an emitter removes its runtime volume, while retaining its
  source override for undo; it does not create an orphan runtime teleporter.
  Inspector controls, landing-point validation, destination framing, map link
  arrows/highlights, undo/redo and dirty/recovery tracking are integrated. The
  unchanged original-map binary teleporter workflow remains available. Project
  format 7 adds `teleporter-links.json`; source starts/objects remain optional,
  earlier formats remain readable, and malformed/missing link data is rejected.
  The editor host exposes a read-only actual player-position query. Native
  browser tests verify the simulation moves from an authored room to the exact
  saved destination room and coordinates after starting play, rather than only
  asserting compiled header values. Source-model tests also cover concave
  emitter/destination cells, disabled inheritance, stale commands, missing and
  reshaped destinations, and retained import bytes. Production/WebAssembly
  builds and all 612 model tests pass. Six Chromium/Firefox teleporter scenarios
  pass, alongside 42 surrounding editor/room/start/object/recovery/engine checks,
  including original-map loading in both browsers. Portal openings, new object
  creation, full mechanism authoring and normal source Build & Play integration
  remain pending; no finished-editor deployment has been made.

- Animation room resolution is now shared by import validation, animation
  remapping, geometry attachments and source dependency planning. This fixes a
  valid lift file-ID alias being reported as a deleted room by the reference
  stage after the other stages had accepted it. The rule follows
  `src/level.c` lift parsing and `normalize_amiga_water_list`: in-range words
  address raw room slots; only out-of-range lift/water words resolve stored
  file IDs. Doors and switches continue to require direct room indices. An
  in-range missing slot cannot redirect to an unrelated stored file ID.
  Regression fixtures compile lift/water aliases to exactly the same complete
  generated triplets as their canonical equivalents, retain source bytes, and
  diagnose deletion of the resolved lift volume. This clears one compiler
  integration edge case; normal source Build & Play remains unfinished.

- CI exposed a real initial-preview/play race in the narrow Firefox controls
  scenario. Build & Play could become enabled while the automatic inspection
  reload was still queued. If the play load's reply took longer than that
  debounce, the inspection request overwrote the pending play intent and left
  the engine paused. Explicit play requests now cancel queued inspection
  reloads; automatic loads cannot replace a pending play request, and the
  debounce lifecycle observes the starting state. A deterministic regression
  clicks the earliest enabled play button and delays replies from the actual
  engine. It reproduced the old paused state before the fix. The native game
  simulation and its controls are unchanged. Failed load replies also clear
  pending play/starting state, retain an actionable error, and suppress automatic
  retries for that failed document; a changed document or explicit retry/reload
  can try again. Reload engine clears the old intent and starts a fresh host.
  A second browser regression exercises the real iframe's invalid-slot rejection
  followed by a successful reload/play sequence. Production build and all 615
  model tests pass. Both new scenarios and the original narrow controls case
  pass in Chromium/Firefox, alongside existing native map loading, authored
  spawn/object/teleport, persistence and source-guard browser regressions. The
  new commit still needs its own remote CI result; the earlier failure was
  investigated rather than treated as a completed cross-platform check.

- Normal Preview, Build & Play and Export level now share the immutable source
  artifact service in `playableBuild.ts`. Original unchanged maps retain exact
  imported files; edited source compiles through the full geometry/graphics/
  lighting/animation/reference/visibility pipeline. New rooms have explicit
  inspector wall-style and template-lighting selections. The selected source
  wall retains texture, phase and sampling, with its perimeter span taken from
  the room template's actual floor/ceiling. Insertion remains at the selected
  record's ordered draw-stream location. These settings are undoable, recoverable
  and stored in project format 8 (`room-appearances.json`); older source projects
  remain readable and can assign missing appearances before building.
  Asset preparation packages the real `src/game_data.c` ENDZONES array as game
  configuration, avoiding an independent default-exit table. The compiler maps
  that source exit to generated rooms for both preview and export. Compiled
  viewport visits and picks map runtime cells back to original/editor room IDs.
  Async builds reject stale results after source/project/revision changes;
  opening a newer project while export waits cannot download old source files.
  Engine errors/stops/picks now carry revision identity too, and late responses
  cannot invalidate a newer load. Newer successful loads clear old iframe errors.
  Native UI tests author two rooms with reversed source ordering, select their
  appearance, export through the real button, inspect/pick generated rooms,
  start gameplay and teleport to the saved destination. Saving afterward proves
  original binary files and editable source/gameplay were retained. Missing
  appearance still blocks preview/export with actionable feedback; new portal
  wall openings, moving-volume controller groups and all other incomplete
  requirements remain outstanding. Production build and all 620 model checks
  pass; 66 surrounding Chromium/Firefox scenarios pass, followed by focused
  stale-engine-error, build/export cancellation and integrated source-play
  reruns. An earlier Ubuntu CI drag failure was traced to the test assuming
  that anchored wheel zoom left a marker at canvas centre. Integer wheel-event
  rounding on Linux moved the world centre. The test now derives the rendered
  marker position from the current transform and deliberately uses integer
  wheel coordinates, retaining the exact expected 16-unit drag result. Its
  Chromium/Firefox reruns pass locally; fresh remote CI is still required.
  Deployment remains deferred.

- New static doorway openings now cut wall graphics when an authored room
  touches a previously closed imported boundary, or another authored room.
  Destination floor/ceiling air bands leave steps, lintels and any solid band
  between lower and upper storeys. Upper-storey availability follows the native
  graphics pointer and height tests. Existing imported portal records retain
  their authored bytes. Texture columns keep their original distance and phase;
  lighting interpolation keeps the full original wall anchors even when the
  opening separates the remaining wall into disconnected pieces. Logical draw
  cells remap external neighbours to their retained draw-room identities.
  A shipped Level 1 fixture opens part of a real solid wall, checks exported
  graphics and lighting, and crosses the opening with normal game input in
  Chromium and Firefox. Both browser scenarios and all 623 model tests pass;
  the production build passes. Screenshots confirm the opening into the new
  room. New openings into moving doors/lifts, replacing existing portals,
  assigning closing-wall materials and broader controller authoring remain
  unfinished. This milestone does not complete the editor or its deployment.

- Imported doorway closure now rebuilds solid wall graphics after an adjoining
  room is deleted or moved away. The remaining room's inspector provides a
  Doorway closure appearance, using actual template wall records and point
  lighting. No texture is guessed. Missing lower/required upper selections
  block the build with room-specific instructions. Existing wall records on
  closed edge segments are removed before full-height fills are inserted at
  the chosen style's draw-stream position, avoiding overlapping old steps and
  lintels. Fill texture columns follow the source wall chain through inserted
  corners; generated-wall lighting identities remain separate where other
  materials need different values. Existing wall styles and original baseline
  files remain authored, and assigning a closure appearance without closing a
  portal leaves every generated graphics record unchanged. The existing v8
  appearance fields now also store imported-room closure selections and retain
  atomic undo/redo and recovery behavior. Moving-volume closures still fail
  explicitly until synchronized animation attachments can be authored.
  Model scenarios cover deletion, exact saved settings, undo/redo, continuous
  texture columns, separate upper/lower spans, missing-style rejection and the
  moving-volume restriction. Browser scenarios delete the adjoining room via
  the actual UI, repair its visible build error, save original files and source,
  undo/redo the appearance, and hold movement against the resulting native
  collision wall in Chromium and Firefox. Screenshots show the opening filled.
  Broader topology/controller and other outstanding requirements remain open;
  deployment is still deferred.
  Source-build integration commit `fa701e6` has now completed its separate
  Windows/Ubuntu editor CI successfully, including the actual-engine browser
  suites. This does not substitute for CI of the later portal/closure changes.
  All 628 model checks, production build and TypeScript validation pass for
  closure authoring. All 84 surrounding browser scenarios pass, including
  complete original/source/visibility/renumbered-map engine comparisons in
  Chromium and Firefox. A fresh production rebuild and all four focused
  opening/closure scenarios also pass after the final split-wall and upper-layer
  validation changes. Closure commit `06cd281` is pushed; its own remote editor
  CI is still running and is not counted as passed here.

- An imported static portal can now connect to a replacement authored room.
  The saved appearance is presented as Doorway repair appearance and handles
  both closing and replacing a doorway. Replacements remove the old edge's
  material fragments, bind the selected style to the source wall chain and
  subtract the new destination's native air bands. This retains the correct
  steps/lintels or removes all old portal pieces for matching heights. Original
  unchanged portals still retain their authored graphics. Replacement identity
  comes from source provenance rather than the runtime number, which can be
  reused after deletion. The existing project appearance fields, explicit
  lighting and ordered stream insertion are shared with closures; no material
  is inferred from the room being removed. Missing appearance remains an
  actionable build error. Moving-volume joins and boundaries without source
  provenance still need their own authoring/attachment implementation.
  All 631 model tests and production build pass. Six focused browser scenarios
  pass in Chromium/Firefox, covering openings, solid closures and actual
  keyboard movement through the replacement doorway after UI appearance
  selection and export. Additional full-build model scenarios cover a changed
  step/lintel and a full-height opening, with project/source preservation.
  Screenshot camera directions now follow renderer.c's player-arrow
  sine/cosine convention; inspected captures show the replacement opening
  rather than the wall behind the player. The six browser scenarios pass
  again after this test-camera correction; game rendering/input is unchanged.
  The surrounding 84 browser cases passed at the prior closure milestone;
  that full suite has not been repeated for this replacement change.
  The complete editor requirements and deployment remain unfinished.
