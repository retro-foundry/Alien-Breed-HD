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
| Geometry creation and editing | Points, sectors, shared portals, split/delete/join operations, upper/lower rooms; playable exports | Source outlines/redrawing, explicit new boundary walls, static openings, imported doorway closures and replacements build/play/export; partial decorations, broader topology operations and controller groups pending |
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

- Redraw outline now replaces a selected room's perimeter and corner count as
  one undoable command while keeping its source identity. The old room remains
  committed throughout the draft; Escape, Cancel room and tool switches discard
  it. Save during an unfinished draft preserves only committed geometry. The
  action explains its controls and material requirements in status/tooltips.
  Imported edges without original wall provenance now use the saved Wall repair
  appearance rather than failing unconditionally. The compiler distinguishes
  true perimeter edges from internal decomposition seams, emits selected wall
  materials (or destination step/lintel spans), and gives newly authored
  polygon corners explicit selected initial lighting. Existing authored corner
  and material lighting remains separate where required. New boundaries into
  authored neighbours can open fully without a solid seam; required lighting
  is retained even when an opening emits no wall records. Missing appearance
  still stops the build with a room-specific instruction. Moving volumes and
  unsupported partial/interior decorations retain their explicit restrictions.
  A real imported-room fixture cuts a corner into a new diagonal boundary,
  compiles the complete map, checks actual wall/polygon lighting references and
  preserves original files. UI scenarios save an unfinished draft, cancel by
  Escape/tool switch, redraw in either direction, save and undo/redo the same
  room, repair its missing appearance, export five collision edges and start
  the real game. Inspected engine captures show the new textured wall.
  These scenarios pass in Chromium/Firefox alongside doorway opening/closure/
  replacement checks. All 634 model checks pass, including complete original
  source builds and a moving-door redraw rejection; production build passes.
  Eight focused Chromium/Firefox opening/closure/replacement/redraw scenarios
  pass, followed by two reverse-direction redraw reruns and 26 surrounding
  controls, tool cancellation, source persistence and build/export scenarios.
  The full editor scope and deployment remain unfinished.

- Playable output now enforces the current renderer's 500-command traversal
  bound per runtime room/storey. The previous structural stream parser/writer
  accepted 4,095 records while renderer.c's seven passes stopped after 500,
  allowing a valid-looking export whose tail was never drawn. Native draw and
  inspection passes now use `renderer_limits.h`, whose enum includes the numeric
  literal in `renderer_graphics_limit.json`; the editor imports that same JSON
  value directly in browser and Node tests. No duplicate limit table or generated
  copy is maintained, and the existing native bound itself is unchanged.
  Source stream assembly reports the stable editor room identity; generated
  graphics and unchanged-file preview/export paths enforce the same budget.
  The structural importer/project writer still preserves oversized files so
  they can be saved and repaired. Errors identify room, lower/upper floor,
  actual count and supported count; Help explains that draw commands share the
  budget. This is explicitly a port safety bound, not a claimed original Amiga
  format limit or completion of the strict original-engine export profile.
  Model scenarios accept exactly 500 lower/upper records, reject 501 through
  all playable paths (including retries and complete source assembly), and
  round-trip oversized project files unchanged. The native boundary fixture
  prepends real no-payload Backdrop commands before the original draws; four
  actual-engine camera hashes still match the shipped map at 500, proving the
  final draws execute. Browser checks reject over-budget inspection/play/export
  and retain a downloadable editable backup. Windows and WebAssembly builds,
  production editor build and all 637 model checks pass. Twelve rebuilt-runtime
  Chromium/Firefox scenarios pass, including all sixteen original maps, budget
  boundary/rejection, source export/play and imported-room redraw. Remaining
  editor requirements and deployment are still unfinished.

- Per-room numerical height editing now follows AMOS `DEFHEIGHT`,
  `DEFROOFHEIGHT`, `DEFUPPERFLOORHEIGHT` and `DEFUPPERROOFHEIGHT`; water remains
  separately authored as in `DEFWATERHEIGHT`. Room-source schema v3 stores four
  optional floor/ceiling overrides without modifying original files or shared
  templates. Existing source schemas remain readable, and ordinary snapshots
  still use v2. Shape moves/redraws retain height overrides. Validation checks
  native WORD quarter-unit plane bounds, positive air clearance, separated
  upper/lower storeys and active upper material availability. Exact restoration
  of authored closed-door/sentinel values remains possible. Older source
  versions reject height fields rather than silently dropping them.
  The inspector applies all fields atomically, copies shown heights into a
  reviewable draft, restores template heights, explains direction/units and
  keeps errors beside the form. Undo/redo, source archives and effective room
  readouts include the overrides. Runtime cells carry room settings through
  collision/header assembly, wall endpoint anchoring and floor/roof draw planes.
  Existing portal height changes rebuild both sides with explicit selected
  repair materials where original full-span material is absent; known solid
  wall material is reused when opening into an authored room. Unanchored
  decorative heights and water stay authored. Moving ranges and curves retain
  explicit unsupported-build errors until their attachments can be edited.
  Model scenarios cover template independence, quarter heights, upper/lower
  spans, save/restore, undo/redraw retention, invalid input/schema downgrade,
  moving-door rejection/reset and both sides of an imported height-changing
  doorway. Real-engine browser scenarios edit imported and authored rooms,
  copy/reset/undo, preserve source backups, compare upper/lower rendered views
  and walk up a generated step. A read-only preview probe reports simulation
  feet (`s_yoff + s_height`, player.c floor-target contract), avoiding rendered
  head bob in the physical floor assertion. All 643 model checks and the
  production build pass. Twenty-eight surrounding Chromium/Firefox scenarios
  pass, followed by fresh height and workspace-recovery checks in both engines.
  The preceding renderer-budget commit `40dacbd` also completed remote editor
  CI successfully; height commit `a5da947` subsequently passed its own complete
  editor and normal CI workflows.
  This does not complete graphical height handles, animated attachments, other
  editor requirements or deployment.

- Automatic straight stair flights now author adjoining source rooms from
  endpoints, width, step count, starting/final floors and clearance. Cross
  sections are snapped once and shared by both neighbours; floors use native
  quarter units. Generated rooms carry independent height overrides and the
  explicitly chosen original wall styles/lighting. The native player's existing
  40-unit step allowance is shared through a numeric JSON include between C and
  TypeScript; movement behavior is unchanged. Strict risers below that allowance
  are checked in both directions. Invalid dimensions, snapped flat/collapsed
  steps, overlapping rooms, bad materials and exhausted convex runtime capacity
  fail before applying source. Batch room insertion validates the group once.
  The modal shows a labelled plan, local errors and control descriptions.
  Creation is one undoable source/appearance command; Escape, Cancel and saving
  a project while drafting cannot commit partial stairs. Tooltips now render
  inside native dialogs so the top-layer backdrop does not obscure descriptions.
  All 647 model checks pass, Windows and WebAssembly builds succeed, and twelve
  Chromium/Firefox control and stair scenarios pass. The stair cases export
  real native files, preserve original source files, verify undo/redo and
  cancelled drafts, and walk the player up four risers and back down while
  checking the actual simulation floor height. Rendered staircase views were
  inspected. Graphical height handles, moving/controller attachment authoring,
  other incomplete requirements and final deployment remain open. The requested
  deployment directory is the sibling website repository's `abhd-ed/` folder;
  deployment is reserved for the completed editor.

- Graphical lower/upper height handles now show the actual air bands in a
  section diagram beside the existing numerical height form. This is an editor
  diagram, not a substitute renderer: Split view continues to use the game
  engine, which rebuilds after a completed edit. Pointer capture keeps the
  gesture provisional until release; Escape, cancelled capture or changing
  rooms discards it. The displayed proposal is committed as one undo command,
  with quarter-unit values and the existing source/compiler height validation.
  Invalid crossings show a local explanation and leave source unchanged.
  Keyboard Up/Down uses quarters, Shift uses four units, and repeated edits
  retain focus. Original AMOS height routines remain the authority documented
  above; native rendering and movement code are unchanged in this UI milestone.
  Shared tooltips now describe SVG controls and reposition focused descriptions
  when browser focus scrolls a control into view; unrelated pointer-out events
  no longer clear another control's description. Browser scenarios verify
  cancelled drags, source-only saves while dragging, atomic undo/redo, independent
  upper/lower planes, invalid relationships, focus and keyboard descriptions.
  Real exported game files and actual player simulation verify dragged floor
  heights, and Split-view rendered changes are compared and visually inspected.
  The production build and 17 focused height/stair/history model checks pass.
  Six surrounding numerical-height browser cases pass, followed by sixteen
  fresh graphical-height, control and stair scenarios in Chromium/Firefox.
  Animated mechanisms, other open requirements and deployment remain unfinished.

- Visual wall-style browsers now accompany the lower/upper selectors in room
  appearance and stair authoring. Whole native texture sheets identify each
  template's authored styles; choosing a tile changes the form draft, and the
  existing Apply/Create action commits it transactionally. Search and preview
  shade controls have keyboard help/tooltips. Alignment, sampling, transparency
  and corner-lighting choices remain explicit source data. Preview shade never
  changes saved lighting. Missing or damaged sheets show actionable errors and
  retry; thumbnail failures do not replace source or fabricate pixel data.
  Asset preparation reads the actual `src/io.c` wall table, declared unpacked
  lengths and dimension overrides, then packages all fourteen original WADs
  with a validated catalog. TypeScript decodes the original SB files, 3x5-bit
  strip packing and all 32 RGB12 palette levels. Pixel oracles compile the
  existing native `io_dump_textures`, dimension, RGB and BMP routines in a
  temporary C harness and feed them independently decompressed native files.
  All fourteen sheets match both native raw-byte and complete middle-shade
  RGBA hashes. A native IO fingerprint requires reviewed fixture regeneration
  after IO changes. All 664 model tests pass. Eighteen fresh Chromium/Firefox
  cases cover native canvas hashes/catalog metadata, filtering, preview shade,
  draft/apply/undo/source backups, real game exports and renderer views,
  missing-asset retry, stair-dialog selection and surrounding controls/stairs.
  The production build passes and browser/engine views were visually inspected.
  CI now also triggers on native wall assets and their oracle tool. This
  advances R8; global material assignment, PNG conversion/import, portable
  custom assets, remaining requirements and final deployment are still open.

- Portable native wall overrides now connect project source, history, material
  thumbnails, generated level artifacts, runtime preview and game export.
  Native slots/filenames/dimensions are generated from `io.c` into the model
  catalog as well as packaged assets. Override records cannot redefine those
  paths or sizes. Imported raw/SB sheets decode before one stale-checked edit;
  Restore original sheet and undo/redo update every wall using that material.
  These advanced controls explain their immediate shared-sheet effect, separate
  from the wall-style form's Apply action. Pending file reads are cancelled on
  project, material-slot or asset changes. Original level binary files remain
  the baseline.
  Project format v9 declares required editable source files, optional draw
  setup and wall slots, stores decoded WADs separately, rejects damaged/missing
  resources and schema downgrades, and retains legacy v1–v8 loading. Game ZIPs
  include native shared wall paths and installation notes explaining their
  global effect; re-import restores them. Compiled artifacts retain overrides.
  Workspace dirty tracking and completed autosaves include asset identities.
  The iframe validates a complete replacement set before writing assets, keeps
  immutable packaged originals, restores omitted overrides, and reloads native
  wall buffers only when bytes change. `io_load_walls` now releases its previous
  owned buffers before reloading; the editor bridge stops simulation during
  asset replacement. The native pixel oracles were regenerated after this IO
  ownership change; all pixel hashes remain unchanged.
  Windows/WebAssembly/production builds and all 667 model tests pass. Twenty
  Chromium/Firefox scenarios cover asset import/errors, save/export/re-import,
  source-only preservation, repeated renderer restore/undo/redo, workspace
  recovery, playtesting, switching to a project without overrides, materials,
  controls and stairs. Browser/engine output was inspected. Four additional
  Chromium/Firefox engine scenarios pass, including all sixteen original maps
  and edit/playtest isolation from normal game saves. PNG conversion still needs its
  original packing/palette authority checked against
  `amiga/archive/amos/256wallchunk.amos`; this is a completed native asset path,
  not completion of PNG authoring, other R8/R11 resources, the full goal or
  deployment.

- PNG wall import now resizes a still image to the selected native sheet,
  matches its existing palette, composites alpha onto the material background
  and packs native five-bit pixels while preserving all authored lighting
  palettes. Fitted/native comparison and adjusted-pixel counts precede Apply;
  Cancel/Escape and stale asset/project changes keep source unchanged. Apply
  commits original PNG and native output as one undoable asset edit; identical
  reapply closes review without adding history or dropping source. Archive
  capacity is validated before committing the import. Project v10 explicitly
  declares PNG source slots and retains their original bytes separately from
  WADs. Missing originals/schema downgrades fail loudly, legacy projects remain
  readable and native game ZIPs contain WADs only. The original PNG can be
  downloaded unchanged after workspace recovery. Restore original sheet is
  exposed alongside PNG import rather than requiring native-file tools.
  Source inspection corrected the earlier converter assumption:
  `256wallchunk.amos` stops before its inactive `.256wad` tail, which uses a
  different palette representation. The new import policy and its limited use
  of the stored colour-distance expression are documented in
  [editor-png.md](editor-png.md); native layout comes from the active C loader
  and renderer, not a claim of running that AMOS tool.
  All 671 model checks and production builds pass. Browser checks in Chromium
  and Firefox cover 4K resizing, transparency, invalid input, review/cancel,
  no-op reapply, atomic undo/redo, native renderer changes, exports and unchanged
  original PNG recovery/downloads. Surrounding material/native-asset and controls
  regressions also pass. New-palette generation, broader material assignment,
  other resource conversions and remaining editor/deployment work remain open.

- New object authoring checkpoint: Add object clones an actual current-level
  enemy, pickup or scenery record into a fresh source record and position slot.
  Destination room, coordinates, lower/upper floor and floor-relative or manual
  height are explicit, labelled controls with tooltips. Cancel/Escape preserves
  source; Create is one undoable edit. Players and reserved projectile pools are
  excluded, and the native 250-record capacity is generated from game_state.h.
  Source-table relocation updates object/pool/avatar and geometry pointers plus
  graphics room offsets without replacing editable room source. Native header
  word 20 counts active position slots; its extra trailing padding point moves
  so native projectile expansion cannot reuse the new actor's position.
  Object Y now displays the renderer's actual half-world-unit scale (raw word
  multiplied by 128 in 24.8), correcting the former quarter-unit UI conversion
  without changing untouched native bytes. Saved projects retain complete new
  records, source placements, history and native game export. Model validation
  covers all 16 original prefixes, capacity, stale commands and save/build
  round trips. Browser validation exercises native position/height probes,
  invalid drafts, saving during a draft, undo/redo and portable project records.
  All 677 model checks, the production build and 16 Chromium/Firefox object and
  controls regression scenarios pass.
  A cross-level object library and broader object configuration remain open;
  this checkpoint does not complete the full editor or trigger deployment.

- Key configuration checkpoint: the object inspector now offers colour choices
  grounded in `leved.amos` KEYPUT/KEYSAVE: green/red/yellow/blue use artwork
  `5*65536+CO` and condition mask `2^CO`. One command edits both fields; custom
  artwork/masks are shown explicitly and left intact until a colour is chosen.
  The native `object_handle_key` pickup routine consumes the resulting mask.
  Original-level tests check every imported key and all four choices, atomic
  undo, custom records and portable project saves. Chromium/Firefox gameplay
  places a new blue key beyond the native pickup radius, walks into it and
  verifies its removal and condition bit 8, with no other key bits granted.
  Production builds, all 681 model tests and all four Chromium/Firefox object
  authoring scenarios pass. The native simulation is unchanged; the preview
  host exposes a read-only conditions probe for gameplay verification.

- Ammunition configuration checkpoint: the object inspector offers named pistol,
  plasma, rocket, grenade and shotgun ammunition choices, with tooltips and
  explicit preservation of custom records. One command changes AmmoType at +18
  and the inspection artwork to match `object_handle_ammo` and its native
  `ammo_graphic_table`, now generated from game_data.c during asset preparation.
  Weapon-pickup IDs are deliberately not used as ammunition indices. The native
  handler rejects flamethrower clips; unused weapon slots are not offered.
  Model checks cover complete-field isolation, custom artwork, atomic undo,
  invalid types and portable saves. Native browser gameplay checks the selected
  plasma clip's removal and the actual player ammo increase from the original
  `default_plr1_guns.ammo_in_clip` and `ItsAnAmmoClip` scaling, without modifying
  any other weapon's ammunition or the simulation.
  The production build, all 683 model tests and six Chromium/Firefox object
  authoring/playback scenarios pass. Broader weapon/enemy configuration and
  remaining full-editor requirements are still open; deployment remains pending.

- Weapon pickup configuration checkpoint: named plasma, rocket, grenade and
  shotgun choices use the original `leved.amos` BIGGUNGRAPH/BIGGUNDIM tables and
  BIGGUNSAVE layout. Artwork, source/world dimensions, grant ID and height are
  one command. The BIGGUNSAVE floor anchor is preserved when changing between
  tall and short artwork, including deliberate half-unit offsets. Custom records
  are retained until a choice; polygon artwork and height overflow fail before
  mutation. Models exercise all original weapon records, every supported choice,
  saves and undo/redo. Native browser gameplay checks that rocket pickup ID 1
  actually grants player gun 2 and its original AmmoInGuns amount, with no other
  weapon ownership or ammunition changed. The preview adds only a read-only
  weapon-ownership probe; the game simulation remains unchanged.
  The production build, 686 model tests and eight Chromium/Firefox object
  authoring/gameplay scenarios pass. The full editor remains in progress.

- Visibility inspection checkpoint: a map overlay now reads the exact shared
  preview/export artifact via preparePlayableLevel. Amber identifies the selected
  runtime cell, cyan its potentially visible cells and green its actual connected
  floor-line portals. Concave source rooms expose individual compiled cells;
  the table shows native zone, clip and ordering-mask records without treating
  conservative candidates as exact line of sight. Visibility is shared by both
  storeys, as the native table is. Source identity guards and effect cancellation
  discard stale builds; a failed source build removes old overlays and exposes
  the compiler error with retry. Undo restores valid inspection. Model checks
  compare all original zone entries/portals, disconnected new rooms and concave
  cell mapping. Browser checks verify exact compiled entries, switching convex
  cells, keyboard toggling, no source mutation and moving-door failure/undo.
  Full runtime closed-door/occlusion validation remains required for R3.
  All 690 model tests, production builds and 14 Chromium/Firefox visibility and
  controls regression scenarios pass. The final memoized inspection also passes
  all four focused browser cases. No native gameplay code changed.

- Clip inspection checkpoint: the visibility table now distinguishes saved
  reference flags from actual word offsets resolved in `level_assign_clips`
  stream order. A target selector shows its left/right clipping boundary points,
  with native point IDs and a legend. Malformed boundary lists fail visibly
  without crashing the map. Frame visibility fits the current inspected cells
  without source edits. Visual review caught authored-room fills hiding the
  diagnostic colours; visibility now draws after those fills and point markers.
  Browser pixel checks protect the amber/cyan shading, alongside framing, target
  selection and exact offset display. An independent DataView stream walk checks
  offsets and both boundary lists across all 16 original maps. All 691 model
  tests, production builds and four Chromium/Firefox inspection scenarios pass.
  This is inspection tooling; broader runtime PVS/occlusion validation remains
  open, along with the other full-editor requirements and final deployment.

- Visibility panel interaction checkpoint: controls can now be hidden without
  disabling diagnostic drawing, with cell and target choices retained when
  reopened. Long tables contain overscroll. Chromium/Firefox checks confirm
  scrolling leaves map scale/centre unchanged and hiding controls leaves the
  actual canvas pixels unchanged. All six focused browser cases and production
  builds pass. This change touches UI only; remaining full-editor work stays open.

- Object duplication checkpoint: the inspector and Ctrl/Command+D now open a
  seeded placement draft for the selected active enemy, pickup or scenery record.
  It uses editable source positions rather than stale native coordinates, retains
  fractional coordinates, storey and exact object height, and starts with floor
  matching off. Settings come from the selected native record and creation uses
  the existing validated table relocation, with one undo and a fresh CID.
  Cancel/Escape leaves source unchanged. Add object explicitly clears the seed.
  Player/projectile/deleted records cannot be duplicated; shortcuts are guarded
  during text entry and modal workflows. Help and tooltips explain the workflow.
  Browser checks verify cancellation, shortcuts, exact placement, copied health,
  undo/redo, saved records and both independent native positions in the engine.
  Multiselect, batch transforms and clipboard workflows remain required for R9.
  All 20 Chromium/Firefox object duplication, pickup gameplay and controls
  regression scenarios pass, together with production builds. No model or native
  simulation changes were needed for this workflow.

- Object group checkpoint: Shift/Ctrl/Command click now adds/removes object
  selections in the map and navigator, with matching marker/list highlights.
  The group inspector offers decimal X/Z translation, coordinate alignment to
  the first selected object and deletion. Each operation builds one source-object
  command, validates every destination and commits no partial result if any
  object is invalid. Player avatars are excluded. Heights and native source bytes
  remain intact; normal compiled preview/export consumes the authored placements.
  Group deletion undo restores the entire selection. Clear/Escape and plain-click
  single selection are supported; duplicate refuses to silently copy only one
  member of a group. Source selection is memoized to avoid mouse-hover redraws.
  Tests check partial failures, decimal moves, alignment, group deletion, native
  positions, project round trips and single-action undo. Room groups, group dragging,
  clipboard and other broader R9 workflows remain required.
  All 694 model tests, production builds and eight Chromium/Firefox group,
  duplication and existing-object regression scenarios pass. Final map gesture
  checks also pass in both browsers. No native simulation changes were made.

- Group dragging checkpoint: pressing a selected object marker retains the group
  and previews all selected positions with one shared, grid-snapped translation.
  Release calls the same atomic group command as the inspector; invalid targets
  reject the complete movement. Escape, pointer cancel/capture loss, focus loss,
  tool/source changes or changes to the selection cancel the provisional move.
  Spacing and fractional starting positions are preserved by snapping the offset
  rather than snapping each object independently. Tests save a snapped drag,
  undo it once, cancel another with Escape and release afterward to prove no
  delayed commit. Existing single-object dragging remains covered. Room groups,
  clipboard and the other full-editor requirements remain open.
  Production builds and all six Chromium/Firefox group/single-object regression
  scenarios pass. The final focused cases also verify zero-snapped gestures do
  not create edits. This checkpoint changes interaction only, not native gameplay.

- Storey-aware placement checkpoint: automatic object destinations now filter
  room candidates by the object's upper/lower floor before choosing one, retaining
  the current room when possible. This fixes a shared-boundary case where a
  one-storey room appeared first and rejected an upper object even though the
  adjoining two-storey destination was valid. The same resolver serves single
  source-object dragging and group moves/alignment; explicit room choices retain
  their normal validation. A source fixture places a single-storey room before
  the two-storey target on a common edge. Models verify compatible placement,
  native upper flags and complete rejection inside the single-storey interior.
  Browser tests check both failure and success, saved room bindings and actual
  native object zones/positions/heights. This changes editor destination choice,
  not game mechanics or automatic object heights.
  Direct native-record moves also enforce upper-floor compatibility, and automatic
  native-zone selection filters incompatible storeys before choosing a boundary
  candidate. Focused source-object/model tests cover that separate path.
  All 697 model tests and production builds pass. Eight Chromium/Firefox
  group/existing-object scenarios pass, followed by both focused upper-floor
  checks after the native-record guard was added. Full-editor work remains open.

- Selection framing checkpoint: Frame selected objects and Shift+F now fit the
  current source selection into the map, including groups spread across rooms.
  Single objects, source room outlines, native zones, points and player starts
  are supported by the same framing action. Object/start coordinates come from
  editable source metadata, not stale original record coordinates. Framing from
  3D switches back to the map and waits for usable layout dimensions. Subsequent
  edits, selection changes and pointer updates do not reapply an old framing
  request, preserving manual pan/zoom. Visibility framing shares the same bounds
  calculation. Eight Chromium/Firefox framing/visibility scenarios and production
  builds pass, covering source coordinates, screen bounds, no dirtying, keyboard
  use, panning followed by edits and 3D-to-map layout. Full-editor work remains open.

- Keyboard deletion checkpoint: Delete now invokes existing object removal or
  atomic group deletion, skipping inactive selections and preserving one-action
  undo. The shortcut is guarded during numeric/text entry, room drawing, busy
  operations and all editor dialogs; key-repeat cannot remove subsequent records.
  Help and tooltips expose the control. Browser scenarios exercise group deletion
  and one-step restoration, single-object deletion/undo, numeric-field safety,
  object-dialog safety and help-dialog safety alongside native group placement.
  The shortcut reuses existing removal commands; native simulation is unchanged.
  Review also found native player-avatar deletion was guarded only after a source
  start was authored. The removal API now rejects both native avatar records in
  all cases, and the inspector disables their delete control. A direct model test
  protects both header-referenced avatars. Normal object removal is unchanged.
  All 698 model tests and the final production build pass. Fourteen Chromium/
  Firefox group/controls checks pass, followed by four focused group checks after
  the avatar guard. Full-editor requirements and deployment remain open.

- Group duplication checkpoint: the group inspector can now duplicate the
  selection with explicit X/Z offsets; Ctrl/Command+D copies it in place and
  selects the complete new batch. Every native record and position identity is
  independent, preserving complete type-specific bytes, spacing, storey and exact
  height. Creation is staged in isolated immutable documents and exposed as one
  source-table relocation command, so invalid destinations, unsupported templates
  or insufficient capacity cannot append a partial batch. One undo restores all
  original files and source placements. Keyboard repeat is guarded. Models cover
  native bytes/CIDs, capacity, upper-floor destinations, saves and undo/redo;
  browsers check offset/in-place batches, rejection, keyboard repeat and actual
  native positions/heights/health. Clipboard and room-group workflows remain open.
  All 702 model tests, the final production build and four Chromium/Firefox
  single/group duplication scenarios pass. No native simulation changes were made.

- Select-all checkpoint: an explicit object-list control selects current active
  search matches across both floors. Ctrl/Command+A follows that scope outside
  text fields in the object list, and selects only the displayed floor when the
  map has focus. Map selection respects Show objects and ignores the list search.
  Both player avatar records and inactive/deleted placements are excluded by a
  shared selection model. Text inputs retain native text selection; dialog,
  drawing and busy guards prevent unintended selection changes. No source or
  history edits are created. Models check search, floor, deleted objects and
  avatars; browser checks cover both scopes, text input and hidden markers.
  Clipboard, room-group and other full-editor workflows remain open.
  All 703 model tests, production builds and four Chromium/Firefox select-all/
  group-copy scenarios pass. Native gameplay is unchanged.

- Rectangle selection checkpoint: Box select objects / B starts a one-shot
  selection rectangle using authored coordinates and the displayed floor. Every
  drag direction is normalized; edges are inclusive, and Shift adds to the prior
  selection. Hidden markers and player avatars are excluded. Completion returns
  to Select for immediate group movement. Escape during a pointer edit is handled
  on the canvas before global selection clearing, preserving the prior selection
  and preventing a later pointer release from committing a cancelled gesture.
  Tool/source/storey changes, focus loss and pointer cancellation discard drafts.
  Review also found toolbar groups could shrink to zero width as tools were added;
  tool groups now retain their width inside the existing scrolling toolbar.
  Models check directions, edges, floor filtering and invalid coordinates;
  browser scenarios cover directions, Shift addition, Escape and hidden markers.
  Room selection/clipboard and remaining full-editor workflows stay open.
  All 704 model tests, production builds and 16 Chromium/Firefox rectangle,
  group-editing and controls cases pass. The focused authored-position model
  check also passes. Visual review confirms the rectangle preview and floor
  controls remain usable. No native simulation changes were made.

- Group object heights checkpoint: the group inspector now applies a shared
  half-world-unit height offset or aligns absolute heights to the first selected
  record. The command edits only native Y words, preserving source X/Z placements
  and lower/upper bindings. It validates the entire batch before execution;
  overflow identifies the offending object in world units and applies nothing.
  Zero offsets preserve no-op history. Models check exact scale, alignment,
  partial overflow, source identity, saves/builds and undo/redo. Browser scenarios
  check upper-floor records through the actual engine's 24.8 height probe, native
  floor flags in saved projects and failed edits without partial changes.
  Full moving-height authoring, clipboard, room groups and remaining editor work
  are still required before final deployment.
  All 706 model tests, production builds and six Chromium/Firefox height/group
  scenarios pass. Both focused cases also pass with source-relative upper-floor
  fixture heights. Native simulation code is unchanged.

- Moving-room range checkpoint: existing door/lift destinations now have an
  atomic draft/apply form with quarter-world-unit inputs, current-position readout,
  immediate validation, field Escape/reset and undo. `level.c` converts the source
  top/bottom/current words by 64 into runtime 24.8; `door_routine` and `lift_routine`
  own subsequent movement and attached-wall updates. This edit changes only the
  two endpoint words, preserves the initial position and requires it inside the
  new range. Native signed-word limits and the first-word 999 table terminator
  are enforced before mutation, with errors expressed in user-facing world units.
  All original records are checked for field isolation and undo; project saves
  and native host probes verify loaded door/lift endpoints in Chromium/Firefox.
  These checks establish configuration loading, not complete authored mechanism
  motion coverage. New controllers, synchronized moving-geometry edits and broader
  R7 runtime/restriction scenarios remain open before final deployment.
  All 708 model tests and four Chromium/Firefox range scenarios pass. The final
  world-unit validation passes the focused model checks and production build.
  The native simulation is unchanged; only read-only editor inspection was added.

- Moving-range runtime verification: native door/lift state inspection now
  reports attached-wall count, live collision roof/floor and how many attached
  wall edges differ from the actual controller position. The probes are read-only
  and follow `door_routine`'s +24 / `lift_routine`'s +20 moving wall fields.
  Browser scenarios use the existing Timer expires rule with Never at the upper
  endpoint, change the destination by a quarter-unit fraction and run normal
  gameplay. Both controllers are observed in transit, with collision plane and
  every attached wall synchronized, then clamped at the exact edited endpoint.
  Post-play project checks retain the authored initial position and native zone/
  wall bytes instead of leaking runtime animation state into editable source.
  These checks extend R7 motion evidence; controller creation, lower-end/cyclic
  scenarios, neighbouring moving volumes and synchronized geometry authoring
  remain open before full-editor completion and deployment.
  All eight Chromium/Firefox loading/motion scenarios pass. The four focused
  motion scenarios also pass post-play source-isolation assertions. WebAssembly
  host rebuilds and production builds pass; the gameplay implementation is unchanged.

- Moving-range cycle checkpoint: timer-driven door and lift scenarios now travel
  to the edited upper endpoint and return to the lower endpoint. Sampled runtime
  controller positions stay in range; collision planes and every attached moving
  wall remain synchronized throughout the observed cycle. Post-play source
  isolation is checked as before. Tracing endpoint triggers also exposed an
  authoring trap: a stationary lift or gated door can be stranded by putting its
  fixed initial position between both endpoints. Range edits now retain an endpoint
  there for those controllers. Moving starts remain allowed, as does the original
  `NotGoBackUp` ungated-door occupancy exception; this is not a new gameplay rule.
  UI guidance and invalid-draft checks explain the constraint without modifying
  imported source. New controllers, moving-volume geometry and remaining R7/full
  editor requirements remain open.
  The shared activation check also rejects adding a gate to an idle intermediate-
  start ungated door, closing the corresponding rules-editing path. Existing
  imported bytes are retained until an explicit edit; moving and ungated exceptions
  have model coverage. No game simulation code is changed by these validations.
  All 709 model tests and production builds pass. Twelve Chromium/Firefox
  loading/motion/cycle cases pass; final focused range checks and both gate-edit
  cases also pass after sharing the activation validation.

- Browser CI contract checkpoint: Windows and Linux browser runs exposed six
  failures caused by inventory fields being added to a helper whose callers
  assert exact player position/play-state objects. Those positions were correct.
  The position helper now retains its original four-field contract; pickup checks
  use a separate inventory helper backed by the same native snapshot query.
  Existing exact movement assertions remain unchanged. The focused browser run
  covering source starts, room redraw, teleports and object authoring passed
  locally in Chromium and Firefox. Full cross-platform CI remains to be checked.

- Modal keyboard checkpoint: Help, Recent Workspaces and the archive level
  chooser now contain global editor shortcuts. Undo/redo, tool selection,
  search, open and save cannot act on the workspace behind these panels.
  Escape closes the panel without clearing the selected room/object or changing
  tools, and existing focus restoration returns to the opening control.
  New browser regressions check each attempted background shortcut and verify
  Undo still works after closing Help and Recent Workspaces. TypeScript checks
  pass; those new browser cases await a fresh build after the running baseline
  suite. The baseline has passed all 77 Chromium cases and is running Firefox.

- Object clipboard checkpoint: Copy objects / Paste objects and Ctrl/Command
  C/V capture single objects or groups as immutable document snapshots. Later
  edits, deletion or undo of their originals cannot alter the captured native
  records. Paste allocates independent collision identities, retains exact
  positions, heights, storeys and settings, validates the entire batch, and
  commits one undoable relocation. Capacity and destination errors cannot leave
  partial copies. Toolbar tooltips and Help explain in-place placement and
  subsequent dragging/inspector movement; normal text-field shortcuts remain
  native. Clipboard lifetime is the current editor session and resource identity
  restricts pasting to the same workspace. Cross-project resource transfer, room
  clipboard workflows and other R9/R11 requirements remain open.
  TypeScript and the full 712-test model run pass; all four clipboard tests pass
  after adding whole-batch capacity and exact-record preservation checks. New
  single/group browser/native scenarios await the running baseline suite before
  a fresh production build, together with the modal keyboard regressions.

- Browser verification checkpoint: all 154 baseline cases passed across Chromium
  and Firefox, including all-map room-renumbering, source-rebuild and visibility
  frame comparisons. A fresh production build then exercised the modal and
  clipboard changes plus adjacent controls. This exposed a real tooltip focus
  regression when the wider toolbar scrolls: pointer enter/leave events generated
  by scrolling could replace keyboard help and remove its ARIA description.
  Focused control help now retains priority until blur, dismissal or interaction.
  Clipboard browser fixtures also now compare the native object type with the
  actual Health-pickup template instead of an incorrect hard-coded zero.
  All 22 focused Chromium/Firefox cases pass after those corrections, including
  single/group clipboard saves, atomic undo/redo, native heights/types, modal
  containment, tooltip focus, duplication and Select All. Cross-platform CI and
  the remaining full-editor requirements still need verification/completion.

- Paste placement checkpoint: Place pasted objects / Ctrl/Command+Shift+V opens
  a transactional placement dialog. Its X/Z anchor belongs to the first captured
  object; the shared offset preserves every other object's spacing and exact
  height/storey/settings. Use map cursor fills the anchor from the last map
  position. Whole-batch validation disables Paste here for invalid destinations;
  Escape or Cancel paste discards the draft and retains the previous selection.
  Native in-place Ctrl/Command+V remains available. Dialog text, field tooltips,
  toolbar descriptions and Help explain placement and cancellation.
  Production builds and all 20 focused Chromium/Firefox controls/modal/clipboard
  cases pass, including map-anchor placement, rejection without source changes,
  cancellation, one-step undo/redo and exact saved positions/heights. This extends
  same-workspace object clipboard authoring; room groups, cross-project resource
  transfer and the other full-editor requirements remain open.

- Room selection checkpoint: Shift/Ctrl/Command clicks now add/remove imported
  and authored rooms in the navigator and map. Room groups have visible outlines,
  accessible pressed states, a dedicated inspector and union framing via button
  or Shift+F. Delete selected rooms / Delete removes the full group in one source
  command; Undo restores it. Linked starts/objects/teleports remain source data
  for explicit repair, rather than being silently removed. Escape clears the
  group without editing. Single-room creation still selects a just-committed room
  before React updates the workspace reference; only modifier toggles validate
  against that reference. Tooltips and Help describe grouping and shortcuts.
  Production builds and all 30 focused Chromium/Firefox cases pass: imported/map
  toggles, framing, atomic deletion, exact retained starts/native bytes in saves,
  playable Undo restoration, existing room drawing/corner cancellation/overlap
  rejection, object groups, clipboard placement and modal containment.
  Room transforms, room clipboard, cross-project resources and the other
  full-editor requirements remain open. Windows CI passed on e16eaa3; latest
  cross-platform CI results still need verification.

- Room translation checkpoint: the group inspector now applies whole-unit X/Z
  offsets to selected room boundaries and their positional dependencies as one
  immutable source command. Geometry is validated after the simultaneous move,
  so temporary overlap with another selected room does not reject a valid final
  arrangement. Shared portal points propagate through their incident room fan,
  including partial-edge T joins; other neighbour corners stay anchored. Wall
  identities, room settings and heights are retained. Starts, active non-avatar
  objects and incoming teleport destinations follow their selected destination
  rooms, including fractional object positions. Avatars continue to follow the
  source starts through the existing compiler. Invalid geometry, coordinates or
  positional links reject the whole command; zero offsets create no history.
  The existing compiler still governs moving-volume and decorative-geometry
  limitations; this does not complete unrestricted transforms or R7.
  All 716 model tests pass, including native compile/save/undo, simultaneous
  movement and shared T-junction cases. All 20 focused Chromium/Firefox room/
  object cases pass. Browser/native checks verify moved player/object positions,
  exact saved inbound teleport coordinates and unchanged original native bytes,
  rejection without edits, and one-Undo native restoration. Room clipboard,
  rotations/reflections and remaining full-editor requirements stay open.

- Shared-boundary dependency checkpoint: translation now also validates starts,
  active non-avatar objects and teleport destinations in unselected rooms whose
  shared boundary deforms. A geometrically valid move that would leave one of
  those dependencies outside is rejected atomically with its subject and room,
  retaining source/history. Valid neighbour deformations keep those positional
  records unchanged. Focused model coverage exercises starts, objects and inbound
  links independently; all 13 room-group/source tests pass. All eight Chromium/
  Firefox room-group cases pass, including rejected deformation without edits,
  exact saved shared corners and a valid T-junction deformation loaded by the
  actual engine with its neighbouring start retained.
  CI run 41270633486 (e16eaa3) completed: Windows passed; Linux failed only the
  two controls tooltip-focus checks. Their scroll/focus symptoms match the issue
  subsequently addressed in c683627. Newer Linux CI must still verify that fix.

- Native room-material checkpoint: material provenance is now independent of
  original runtime zone identity across wall geometry, ordered streams, portal
  topology, appearance selection and point lighting. Existing wall-source fields
  let a fresh room identity retain individual materials and decorative records;
  original closed boundaries remain closed. One-sided inherited portals still
  need an explicit closure style. Native-material copies of moving rooms or
  switch rooms fail with an independent-control-record diagnostic instead of
  silently becoming inert copies. All 720 model tests pass, including exact
  translated decoration and arc/beam stream comparisons with fresh identities,
  static native-copy save/build and behavioural-copy rejection. Both native
  Chromium/Firefox inspection cases pass (four repeatable views); 16 adjacent
  portal/group/sprite cases also pass. This is a room-clipboard compiler
  prerequisite, not complete clipboard UI or all-content render parity.

- Partial-surface translation checkpoint: proven native whole-room translations
  now carry their exact floor/roof/water footprints, point order, closing corner,
  heights, material/sampling values and lighting roles. Partial polygons are not
  expanded to the perimeter. Interior decorative vertices retain their own native
  lighting instead of seeking a boundary interpolation. Exact-shape comparison
  tolerates inserted collinear portal points without changing editable source;
  different shapes/sizes remain unsupported without explicit footprint mapping.
  Whole draw-volume polygons are retained separately from convex collision cells.
  The full 723-test model run passes, followed by six focused material/translation
  checks including a partial water plane. All 20 focused Chromium/Firefox native
  copy/group/height/sprite cases pass, including a partial floor with a distinct
  interior lighting point in the actual engine. Production builds pass.
  CI run 41272068999 (c683627) completed successfully on both Windows and Linux,
  confirming the earlier tooltip fix; newer compiler changes still need their CI
  results checked. Full room clipboard UI, independent mechanisms and remaining
  editor requirements stay open.

- Room property snapshot checkpoint: room-source v4 retains explicit water,
  packed lower/upper brightness, control, sky and sound settings independently
  of later native-template edits. Values are native-range validated, flow through
  room settings and compiled cell headers, persist through save/reopen and source
  transforms, and support one-command inheritance reset/Undo. Earlier source
  versions remain readable; original imports without overrides remain unchanged.
  All 727 model tests pass. Four Chromium/Firefox cases read every property from
  the actual engine after template changes, including static and animated packed
  words. The added host/protocol probe is read-only; gameplay is unchanged.
  Thirty-two adjacent editor/group/height browser checks also passed. A local
  lighting-inspector experiment and its pixel test remain uncommitted: native
  values/reloads are verified but the visual brightness assertion still fails.
  Investigate that rendering/view fixture before shipping the lighting controls.
  This checkpoint does not complete room clipboard UI or the broader goal.

- Room lighting inspector checkpoint: the pending pixel assertion is resolved.
  The inspector exposes source-owned brightness/animation only for native bump/
  smooth surfaces, preserving the point-lighting contracts of walls/Gouraud
  surfaces and the native distance shading of water. Tooltips and Help explain
  those material roles and inheritance reset. Native pixel verification uses a
  valid bump material and an explicit near-floor inspection eye height; a normal
  standing view can omit the tested surface. The paused capture host now accepts
  optional native fixed-point eye height within the room air band; invalid
  heights are rejected before changing the camera. Simulation/renderer behaviour
  is unchanged. Both Chromium/Firefox cases prove actual pixel changes, exact
  restoration by Undo/Redo and inheritance reset, native property readback and
  unchanged original level bytes in project saves. All 40 focused browser cases
  pass, covering adjacent controls, editor recovery, original-level UI, native
  properties and heights. Full clipboard UI and remaining requirements stay open.

  The final 727-test model run also passes. CI run 41274356974 (67e4cbb)
  subsequently completed with two Windows Chromium timer-cycle endpoint failures
  in mechanism-ranges.spec.ts (door/lift). These need investigation; do not infer
  current cross-platform readiness from the older c683627 green run or the local
  lighting checks. The lighting change includes no simulation/renderer edits.

  Mechanism cycle investigation: `game_loop_tick` performs one controller pass
  per display frame. `door_routine` and `lift_routine` clamp at each destination
  and can immediately select the reverse velocity, leaving an exact endpoint
  observable for a single frame. Cross-frame postMessage polling is therefore
  insufficient to prove continuous automatic travel on a loaded CI runner.
  The preview host now supports a bounded, read-only animation-frame trace in
  the engine iframe. Cycle tests consume that trace and still require the exact
  top followed by exact bottom, in-range positions, synchronized collision plane
  and every attached wall. No controller or simulation timing was changed.
  Production TypeScript/Vite build and all 12 focused Chromium/Firefox mechanism
  cases pass locally after rebuilding the preview runtime. Full CI remains
  required before deployment readiness can be claimed.

  Native water-copy attachments now preserve the shared `DoWaterAnims` group
  header/phase while allocating distinct generated polygon pointers for copied
  native-material rooms. Original cross-room water writes remain intact; copying
  those relationships requires an explicit mapping and is rejected without one.
  The preview has a read-only bounded polygon-height probe, allowing browser
  tests to compare actual polygon writes and zone water planes in the same
  callback. An explicit native-format oscillating fixture is necessary because
  the original levels do not populate one. All 729 model tests and six focused
  Chromium/Firefox water/material cases pass; production build/runtime rebuild
  also pass. Full clipboard, independent moving controllers and final deployment
  requirements remain open; this does not claim complete room-copy authoring.

  Moving-controller inspection now separates the controlled volume, primary
  surface, each moving graphics owner and each collision-line incidence, plus
  key/switch requirements. It is grounded in native DoorRoutine/LiftRoutine
  writes and checked for every original controller in all sixteen maps. The
  inspector's collapsible attachment table provides room-framing links with
  keyboard support/tooltips while retaining mechanism selection. Copied moving
  room diagnostics identify live graphics owners instead of only giving a
  generic controller error. The inspector now resolves lift slot/file aliases
  with the same precedence as the compiler/runtime. All 746 model tests and 16
  Chromium/Firefox attachment/range/motion cases pass, including full door/lift
  cycles. The production build passes and the attachment layout was inspected.
  This establishes dependency inspection for R7; independent controller
  allocation and complete room-copy authoring are still required.

  The source compiler now accepts an explicit copied-controller plan with
  independent graphics-owner and collision-room correspondence. It allocates
  separate native door/lift entries and generated attachment identities, with
  independent settings headers. Validation rejects incomplete mappings, mismatched
  translation groups, duplicate moving-plane controllers, changed initial
  positions, changed heights and missing doorway neighbours. Original and copy
  wall matching share the same ambiguity/count checks. Normal builds retain the
  guard unless this explicit plan is supplied. Both Chromium and Firefox prove
  original controllers stay still while copies move to different destinations,
  with every sampled plane/wall synchronized after native initialization. All
  18 focused motion/range/water browser cases, all 749 model tests and production
  builds pass. No runtime movement/timing code changed. Editable-source plan
  persistence, room-copy UI, switch allocation/remapping and broader grouped /
  transformed moving-volume authoring remain open before R7/R9 are complete.

  Room-source v5 now persists immutable copied-controller settings and attachment
  plans; versions 1–4 remain readable. `setRoomControllerCopies` captures native
  settings in one source/history command, and normal preview/export consumes
  those plans without an external compiler option. Geometry/property/height
  edits retain plans; deleting a controlled room removes its plan with the same
  Undo. Native template rule/range edits do not mutate captured copy settings.
  Strict restoration rejects damaged/missing fields, duplicate keys/controllers
  and invalid native bytes/ranges/rules while later geometry dependency problems
  remain editable and produce build diagnostics. All 753 model tests and 12
  Chromium/Firefox native-source save/reopen and adjacent room-group cases pass;
  production builds pass. Browser reopen checks wait for the previous workspace
  iframe to detach before checking the replacement engine's authored state.
  Room-copy UI, ordinary copied-mechanism controls, switches and the remaining
  full-editor requirements are still open. Deployment remains pending readiness.

  Copied door/lift settings now have inspector controls for ranges, native
  opening/closing rules and shared level key/switch requirements. Range drafts
  apply atomically, Escape/reset cancel drafts and Undo/Redo restore committed
  source edits. Imported and copied controls share a range form, rule UI and
  native range/activation validation. Copied changes preserve original bytes and
  attachment mappings, and native playtests reach the edited destination with
  synchronized planes/walls while the original stays still. The occupied-door
  exception is preserved and a new gate that would strand an idle intermediate
  door is rejected. Visible notes/tooltips explain controls and condition scope.
  All 756 model tests, 26 Chromium/Firefox copied/original motion and controls
  cases, and the production build pass; the copied-control layout was inspected.
  Room-copy/creation UI, switches and broader motion authoring remain required.

  The compiler now allocates independent switch copies into unused native slots,
  preserving the eight-slot / condition-bit contract. Activation and graphics
  rooms are mapped separately; copies get fresh point pairs and graphics phase
  targets while originals remain unchanged. Shared point/wall transformation
  logic validates both paths. Explicit condition remapping preserves key/external
  requirements and rejects aliases. The tested native switch doorway needs its
  adjoining corridor rooms to retain the wall attachment; incomplete mappings
  remain actionable build errors. A read-only bounded runtime switch probe allows
  actual Space activation tests to prove the new condition bit/wall phase changes
  without affecting the original. All 759 model tests, ten focused Chromium/
  Firefox switch/copy-control/attachment cases and production/runtime builds pass.
  CI 41277373768 (408c592) and 41277738298 (490e95f) also completed successfully on
  Windows/Linux; newer changes still need their own full CI evidence. Editable
  switch-source persistence, ordinary switch/room-copy authoring and remaining
  full-editor requirements stay open. No gameplay routines were changed.

  Room-source v6 now persists copied switch slots/room relationships and frozen
  native settings alongside motion plans. Both lists are explicit in v6 and
  versions 1–5 remain readable. Normal preview/export consumes saved switches
  automatically; source edits retain both plan types, controller-room deletion
  removes its switch plan with the same Undo, and template changes do not mutate
  captured settings. Structural corruption is rejected; geometry dependency
  problems stay repairable source with build diagnostics. All 763 model tests
  pass, including combined switch/motion projects through group movement and
  save/reopen. Fourteen focused Chromium/Firefox source motion/group/switch cases
  pass, with the two native switch source/save/reopen cases rerun after the final
  strict v6 contract change. Production builds pass. Switch authoring UI, full
  room clipboard assembly and remaining R1–R12 requirements are still open;
  deployment remains pending readiness.

  Native switch-slot safety: the C routine's extra unconditional wall refresh
  wrote unused/off slot graphics fields, unlike `Anims.s` nobutt. It could corrupt
  header bytes at zero or dereference stale positive offsets. Refresh now requires
  an active slot, and all switch phase/texture writes (including use/timer reset)
  validate the entire six-byte span against the actual graphics buffer. Native
  activation, condition bits, countdown behavior and audio are preserved.
  `feedback_switch_slots.c` fails on the old unused-slot write and passes after
  the fix, with canaries and short/negative/large/boundary cases, active toggles
  and timer paths. Windows native and WebAssembly SAFE_HEAP executions pass;
  Windows Release and editor runtime builds pass; all 18 focused Chromium/Firefox
  switch/copy-control/motion cases pass with the rebuilt engine. CI runs the new
  regression, and workflow actionlint passes. Switch authoring and the broader
  editor requirements remain open; deployment is still pending readiness.

  Original and copied switch inspectors now expose independent native reset-mode
  controls with Undo/Redo. Shared UI and setters preserve countdown/state and
  attachments, including existing nonzero native reset flags. Copied sections
  explain assigned Switch N requirements, template/activation rooms, Space use
  and graphics-room framing; tooltips explain immediate edits and Undo. The
  native timer is retained without guessed seconds/duration controls. All 765
  model tests, 14 Chromium/Firefox switch settings/source save/reopen/controls
  cases and the production build pass; layout was inspected. Native browser
  checks prove mode changes start countdown activity after actual Space input
  while the original remains unchanged. Full creation/placement and clipboard
  switch-slot/condition allocation remain required before R7/R9 completion.

  An atomic room-group copy model now assembles source IDs, captured heights /
  native properties/material provenance, objects, teleport links, independent
  motions and switches. Internal destinations/condition bits remap; external
  references, starts and global exit remain unchanged. Existing native or copied
  controller relationships require complete adjoining-room mappings. Switch
  allocation reserves unused bits already claimed by existing gates and copied
  external requirements, avoiding accidental gate activation. Provisional full
  native compilation checks the result before one undoable command is returned.
  All 774 model tests pass, including actual object snapshots, partial floor
  footprints, save/reopen/Undo, original/copy controller branches and a real
  eight-switch capacity boundary with atomic rejection. Eight focused Chromium /
  Firefox circuit/motion/switch cases pass; final circuit cases were rerun after
  gate-bit reservation changed. Actual Space input activates only the pasted
  circuit, with synchronized native moving walls/plane and preserved source on
  save. Production builds pass. Room-copy/placement UI, dependency/material review,
  broader transforms/resources and remaining full-editor requirements stay open;
  this does not claim completed R9 or deployment readiness.

  Room clipboard UI now captures required motion/switch/material attachments,
  lists selected and attached rooms, and validates placement before applying.
  Copy/Paste/Duplicate buttons and keyboard shortcuts have tooltips and help;
  coordinates or the snapped map cursor place the marked group anchor. Escape
  cancels, successful paste selects/frames the group, and one Undo reverses it.
  All 775 model tests and 20 Chromium/Firefox room-placement/object-clipboard /
  controls cases pass. Production builds pass and placement layout was inspected.
  Broader transforms, resource ownership, independent controller creation and
  remaining R1–R12 work are still open. Deploy to the requested sibling site's
  abhd-ed directory only when the complete editor is ready.

  Native 3D room picking now follows the clicked pixel instead of always querying
  the centre. renderer_get_pixel_pick shares the existing captured visibility
  buffers with the unchanged centre-query API; display_render_pixel uses the
  actual presentation rectangle to account for scaling/letterboxing and reject
  outside coordinates. The editor host remains read-only while inspecting.
  Picks retain the camera, map generated cells to authored/original source rooms,
  and Enter keeps centre-crosshair selection accessible from the keyboard.
  Explicit map/list selections continue to visit rooms. Help and the README
  describe these controls and the current room-only selection scope.

  Windows Release, WebAssembly runtime and production editor builds pass, as do
  all 775 model tests. Four final Chromium/Firefox native-picking cases pass:
  authored stairs distinguish off-centre rooms, upper-floor picks resolve
  deliberately remapped generated IDs after resizing, selection preserves the
  camera, outside coordinates reject, and later list navigation still works.
  Fourteen existing controls/engine cases also pass, including all 16 original
  maps in both browsers and isolated playtest storage. Individual wall/floor /
  object picking, edit overlays and the remaining R1–R12 work are still required;
  this checkpoint does not establish full R10 or deployment readiness.

  Surface inspection now captures the native graphics record/stream at each
  world pixel, using the existing wall/floor raster coverage and an opt-in pair
  of provenance buffers. Normal game startup allocates neither buffer. Backdrop,
  opaque sprite and polygon-object writes clear covered surface identities;
  transparent pixels retain the surface behind them. The host exposes read-only
  record and colour-word samples; the UI checks picks against its exact compiled
  artifact and shows kind, lower/upper stream, material and authored height.
  Picking preserves the camera, while navigation/rebuilds clear the readout.
  Inspector edits remain explicitly scoped to the selected room. See
  editor-surface-picking.md for lifecycle and source-identity constraints.

  All 777 model tests and 24 Chromium/Firefox controls/picking/occlusion/engine
  cases pass, including all 16 original maps in both browsers. Native pixel
  tests distinguish walls/floors, ceilings/water and remapped upper-room records.
  Independent before/after colour words prove opaque sprite and polygon-object
  coverage while gaps inside their observed silhouette bounds remain pickable.
  Surface readout layout was inspected. Linux native and WebAssembly SAFE_HEAP
  buffer regressions pass; Windows Release, WebAssembly runtime and production
  editor builds pass. CI runs the capture regression, and workflow actionlint
  passes. Stable source surface ownership/overrides, individual editing, object
  picking, graphical overlays and the rest of R1–R12 remain required. Deployment
  remains pending full readiness at the requested abhd-ed destination.

  Native 3D object picking now shares the optional provenance buffer with surface
  capture. Opaque sprite texels and polygon pixels carry tagged source record
  IDs; transparent gaps retain underlying surface/object identity. Immutable
  per-draw polygon tags keep worker state separate. Projectiles/explosions have
  no authored object identity. The host reports record ID plus CID, and the
  model checks both against the compiled artifact and active editable source,
  rejecting unknown/deleted records and player avatars. A successful pick opens
  the object's existing inspector, clears navigator search, selects/reveals its
  Objects row and keeps camera, page scroll and keyboard focus unchanged.
  Supported edits use existing commands and Undo; help describes the controls.

  All 779 model tests pass. Twenty-six focused Chromium/Firefox controls,
  surface/object picking, occlusion and engine cases pass, including all 16
  original maps in both browsers. Native colour comparisons prove sprite /
  polygon coverage and transparent gaps; lower/upper-floor objects select their
  own source record, height edits affect only the copy in the native engine,
  Undo restores its height, and clicking a gap selects the surface behind it.
  Twenty-two controls/surface/object cases pass again after adding scoped list
  scrolling and explicit no-scroll canvas focus for Firefox, including viewport
  visibility and retained iframe focus/page scroll. Linux native and
  WebAssembly SAFE_HEAP capture regressions cover object/surface tag separation,
  object ID zero, bounds, invalid frames and disabled buffers. Windows Release,
  WebAssembly runtime and production editor builds pass. Stable individual
  surface edits, iframe integration for global editing shortcuts, graphical edit
  overlays and the remaining full R1–R12 scope
  remain required; deployment is still pending readiness.

  Paused 3D inspection now forwards supported scalar key intent through a shared
  inspection-shortcuts.mjs policy to the existing editor keyboard handler.
  There is no synthetic DOM event or separate editing-command implementation.
  Parent checks cover origin, iframe identity, payload, loaded-frame lifetime,
  current revision/source for selection commands, and play state. Existing modal,
  draft and text-field rules remain in the common handler. History/save/open and
  navigation stay available while preview builds; selection commands require the
  matching frame. Inspection Select All uses the editing floor independently of
  map marker visibility, while map/list scopes retain their existing rules.
  The host stages the shared module with runtime assets; missing-module errors
  preserve source and can be retried by reloading. Controls help lists the keys.

  All 783 model tests pass. A 34-case Chromium/Firefox run covers inspection
  shortcuts, all 16 engine maps, startup races, map/list select-all, object
  clipboard and room placement. A final 28-case run passes after help/feedback
  changes, covering shortcuts/controls, authored rooms, native picking and object
  duplication. Portable save/open, paste/Delete history, modal/text/playtest
  isolation and module-error recovery are exercised. Rapid Undo/save is verified
  while a real completed engine load reply is held, then released with the final
  native object height checked.

  Older 6a7d067 CI exposed a real Ctrl+D feedback regression for starts/no object
  selection: room-copy routing silently skipped the existing object eligibility
  message. That feedback is restored, and the affected duplication cases pass in
  both browsers. Its Firefox authored-room/picking failures also pass against the
  current local host. Full current CI is still pending; this does not claim a
  green entire matrix. Runtime staging and production builds pass. Individual
  surface edits, graphical overlays, remaining R1–R12 work and final packaging /
  deployment remain open.

- Individual wall-material checkpoint: paused native surface picking now offers
  Paint this wall for imported/copied straight faces. The dialog has sheet
  previews, scoped help/tooltips, Cancel/Escape, Apply and Reset; room appearance
  can reset all face overrides. Source v7 stores per-room native-record and
  lower/upper identities, independently of generated graphics offsets. Only four
  sheet/sampling fields change; geometry, phases, brightness, transparency and
  controller attachments retain their contracts. Switch faces are protected.
  Compiler provenance follows emitted pieces, copied rooms, clipboard placement,
  source save/reopen and Undo/Redo. Deleted source-face targets produce a repair
  diagnostic; saving remains possible. Packed sampling bounds, signed phases,
  zero-field fallback and lift base phases are checked before preview, without
  silently changing alignment. New/repaired perimeter faces, curves, floor and
  ceiling paint and graphical editing handles remain open.

  All 790 model tests and the production build pass. Earlier browser regression
  runs passed 34 cases covering inspection controls/picking/shortcuts and room
  placement. The new targeted Chromium/Firefox checks exercise lower/upper wall
  paint, native pixel changes confined to the chosen face, copied-room
  independence, portable saves, cancel/reset/history, modal shortcut isolation,
  and synchronized native door/lift movement. All eight final targeted browser
  cases pass. Full CI and final deployment remain pending.

- Authored boundary painting: newly drawn perimeters and repaired doorway fills
  now use the same native 3D material dialog. Graphics provenance carries a
  transient source-edge index. On first paint, source v8 allocates persistent
  room-local edge IDs and lower/upper overrides; generated record offsets are
  never used as authored edge IDs. Normalization, corner moves, T-junction
  splitting, decomposition, translation, save/reopen and copied rooms retain
  these IDs. Split side pieces and step/lintel bands share the source boundary's
  material. New edges from a redraw receive fresh IDs, and orphaned paints remain
  saveable with an explicit restore/reset diagnostic before compilation.
  Reset painted room walls clears both native-face and boundary overrides.
  Help, tooltips and the editor README explain the supported controls and scope.

  Lower/upper model checks retain geometry, strip span, phases and other faces;
  doorway repairs, portable v8 saves, independent clipboard copies, reversed
  winding, normalization rotation and partial-portal splits are covered.
  A 20-case Chromium/Firefox run passes new boundary paint, existing native face
  paint, door/lift motion and native picking. The reopen browser test waits for
  replacement of the previous iframe, avoiding captures sent to a retired frame.
  The all-map weapon/controller model loops have been parameterized by map to
  avoid the five-second single-test timeout while retaining their original
  assertions and explicit nonempty reference coverage.

  Final validation: all 828 model tests in 70 files pass. The production build
  and a final 16-case Chromium/Firefox run pass boundary painting, native pixel
  changes confined to the picked boundary, save/reopen, history, controls/help,
  tooltips and engine-load lifecycle checks. The earlier 20-case regression run
  also passed retained native face painting, native picking and moving walls.

  CI audit: d9e4e57 Windows ran 226 browser cases, with ten Firefox failures and
  216 passes. Downloaded traces show WebGL2 context creation denied
  (`AllowWebgl2:false`), and screenshots show a blank viewport. This explains
  the missing native picks in that environment; it is not a green release gate.
  Runtime initialization must surface an actionable renderer-startup failure,
  and the browser CI rendering environment still needs investigation. Final
  packaging/deployment, curves and floor/ceiling painting, graphical overlays
  and the other open R1–R12 requirements remain pending.

- Renderer startup checkpoint: the editor host now checks actual display
  readiness (SDL window/backend, presentation texture/path and framebuffer)
  after setup and before loading shared assets or starting its frame loop.
  `editor_ready` is set only after successful initialization. The iframe sends
  Ready only after that check, so denied WebGL2 context creation cannot appear
  as a successfully loaded but blank preview. An actionable error explains
  graphics/browser recovery, keeps portable Save available and offers Reload
  engine. The toolbar reports Engine preview unavailable. The error overlay is
  opaque to prevent duplicate iframe text showing through it.

  Browser fault injection denies real graphics context creation for the first
  iframe lifetime; native SDL/host initialization detects that failure. The
  test saves the unchanged decoded original files, reloads with context creation
  restored, and verifies native picking and an actual playtest. No renderer or
  game simulation is replaced. Native Windows and WebAssembly builds pass, and
  a 34-case Chromium/Firefox regression run passes startup/reload, controls,
  authored rooms, native surface/object picks, occlusion and inspection keys.

  Firefox test launches now request `webgl.force-enabled`, permitting real
  software WebGL on GPU-less runners. This is confined to the test browser;
  it changes no user settings and bypasses no pixel/picking assertions. The
  rationale is Mozilla's [GPU-less WebGL2 report](https://bugzilla.mozilla.org/show_bug.cgi?id=1968347),
  which describes the same AllowWebgl2 denial observed in the Windows trace.
  Full Windows/Linux CI must still prove the corrected environment.
  CI runs the real startup/recovery cases before the full browser suite, so an
  unusable graphics environment fails early with its diagnostic artifacts rather
  than spending the full matrix on blank-preview timeouts.

  The older Linux job also exhausted the shared five-minute all-map source-build
  test budget while taking a canvas screenshot. Those comparisons are now
  independent per-map cases with their original camera sets, frame-hash checks,
  visibility-only references and diagnostic images retained. Each map gets an
  explicit two-minute bound instead of sharing a deadline with fifteen others.
  The final 34-case run passes all sixteen maps in Chromium and Firefox, plus
  startup failure/save/reload/play recovery in both browsers. Level 13's full
  camera set takes about 90 seconds in Chromium and 66 seconds in Firefox on
  this machine; full CI still needs to verify its per-map budget on runners.
  The production build, TypeScript checks and native Windows build pass.
  Remaining feature requirements and final deployment remain open.

- 2D picking report (Edge and Brave, default published map): the deployed source
  was still `956397a`, before the primary-outline containment correction. Its
  secondary collision fences made a Level 1 click at (-482.56, 278.08) select
  Zone 0 instead of the visible Zone 1. This is reproduced in a browser against
  that source, then corrected in the focused published-preview branch
  `fix/published-map-selection` (`66acb46`). The branch retains the published
  project format and all four runtime files' hashes. Its 121 model tests and
  complete 30-case Chromium/Firefox suite pass, including all sixteen maps and
  playtesting. Installed Edge and Brave pass 18 selection cases at device scale
  factors 1, 1.25 and 2 across three viewport sizes.

  The current development editor also had invisible marker/corner hit margins:
  start markers captured a 13-pixel circle, small object dots a 9-pixel circle,
  and six-pixel corner handles a 9-pixel circle. Picking now shares their visible
  drawing geometry, stroke widths and actual start-label metrics. Visible
  markers remain selectable and draggable. Room selection is separated from
  navigation: a map click preserves pan/zoom, while navigator visits and explicit
  framing still move the view. The new tests reproduce these three captured-zone
  clicks and the authored-room recentering before the fix. A 40-case interaction
  run passes room/corner editing, groups, starts, framing and box selection in
  Chromium/Firefox. Edge/Brave pass 24 development selection cases. A final
  30-case repeated selection run passes after making the test wait for explicit
  Fit completion before projecting its click coordinates.

  Site commit `3059e08` deploys the focused preview patch to `abhd-ed`. Pages
  deployment succeeds; live installed Edge and Brave both load the new hashed
  bundle, select Zone 1 at the regression point and render the 138-zone 3D map.
  Broader editor feature work and its full release gates remain open. In
  particular, the newer development runtime's Linux Firefox CI startup probe
  still rejects its graphics environment and needs further investigation.

- Publish the implemented authoring tools: the user explicitly requested room
  drawing and wall editing on the website. The release now promotes the current
  authoring UI rather than the earlier read-only-outline preview. Unfinished
  floor-bank painting remains separate work. Room creation, corner dragging /
  coordinates, imported-outline redraw, heights, source saves, Undo and compiled
  gameplay are exercised against the packaged build.

  `npm run build:web` validates the staged bridge against its source, hashes the
  complete five-file native bundle, builds the UI with that content-versioned
  runtime path and writes `dist/build-info.json`. Cached earlier pages can retain
  their runtime while a refreshed authoring editor loads the matching new one.
  The runtime builder uses relative entry/output paths so local build paths do
  not enter the published loader. The packaged asset hashes and public-path
  hygiene checks pass. CI builds this package and serves the existing artifact
  during browser checks rather than replacing it with an unversioned rebuild.

  A 63-case packaged-build run passes in installed Edge, Brave and Firefox,
  including drawing, winding-independent creation, corners, redraw, heights,
  clipboard mechanisms, native gameplay, selection and inspection shortcuts.
  After rebuilding the loader with relative paths, a final 16-case Edge/Brave
  run passes creation, corner editing, redraw, save/export and native play.
  The deployed runtime directory is `runtime/r-ad32e275c8d0e039/`. Live-site
  verification follows publication; the remaining R1–R12 work and Linux CI
  graphics-environment issue remain open.

  Publication is verified: editor source `2831d8b` is deployed in site commit
  `4edbafa`, followed by `f1485aa` / `23159d8` to preserve native bundle bytes
  through Git's Windows line-ending filters. All committed runtime blobs match
  the release manifest. Pages deployment succeeds. On the public URL, installed
  Edge and Brave each create Room 138, edit a corner, Undo, apply its appearance,
  download editable source, compile its 3D view and start/stop native gameplay.
  The live iframe uses `runtime/r-ad32e275c8d0e039/frame.html`. Room drawing and
  wall/corner editing are now published; this does not close the remaining
  editor-port requirements.


- AMOS editing parity is now an explicit P0 minimum (R0), with all 38 original
  menu operations inventoried in `docs/editor-amos-parity.md`. Imported data
  preservation does not count as an authoring workflow. Independent controllers,
  navigation graph editing, original material modes and ambient/echo controls
  remain open requirements.

  Normal floor/ceiling painting now supports all 16 original patterns and repeat
  shifts -5 through 2 on lower and upper source surfaces. The picker uses native
  rendered-surface provenance, thumbnails use the original interleaved floor bank
  and RGB12 palettes, and compilation changes only texture/scale. Source format
  v9 retains overrides through saves, room copies and geometry edits; reset and
  Undo restore original materials. Moving door/lift attachment metadata survives.
  Water, bump modes and sky ceilings are explicitly outside this normal brush.

  Validation: all 835 model tests and 22 Chromium/Firefox browser regressions
  pass, including per-surface pixels, lower/upper independent paint, exact Undo,
  reopening source and synchronized door/lift motion. This advances the full editor
  goal; it does not establish complete AMOS parity.


  The exact packaged build also passes all eight installed Edge/Brave normal
  floor/ceiling cases. The reset oracle now waits for the changed preview
  revision before comparing pixels (the first packaged run exposed a stale-frame
  race in that assertion). Source bcf1d94 is published by site commit b9ad98d;
  runtime resource hashes remain unchanged and floor-bank files preserve their
  original bytes through explicit Git attributes. The public URL now serves the new frontend
  (`index-Be-5ouop.js`); both downloaded floor resources match the packaged
  original bytes exactly.


- Existing objects now support in-place complete type/settings replacement.
  The inspector's Object type template / Apply object type workflow copies a
  real current-level source record's artwork, dimensions, health and behaviour
  while preserving CID, Y, logical/graphic room and storey. This follows the
  64-byte object layout in `amiga/includes/Defs.i` and `src/level.c`; changing
  only the type byte would retain incompatible old parameters. Player avatars,
  reserved projectile pools and removed targets are protected. Template choice
  is a draft, and Apply/save/reopen/Undo preserve the object's identity and
  authored placement. The full original object library and independent door/
  lift creation remain explicit parity gaps.

  All 852 model tests pass with `vitest run --maxWorkers=2`; the default worker
  count first caused 13 five-second timeouts in heavy existing geometry tests.
  Four Chromium/Firefox authoring/duplication checks and two exact packaged
  installed Edge/Brave type-change checks pass, including native object state.
  The web build retains the content-versioned native bundle.

  Publication: source `11ecc6e`, site `5ecc07b`. The public URL now serves
  `index-Dz42MKM9.js`; its delivered script includes the object-type selector
  and Apply action. The worktree is clean after the release checkpoint. This
  is current-level template replacement, not completion of the full object
  library or independent AMOS door/lift authoring.


- Independent door/lift authoring: the generated-graphics compiler now accepts
  exact controller plane/wall targets by compiled room, lower/upper stream and
  record index. New moving graphics no longer need an imported source byte
  offset or a copied native room. This is a compiler-stage identity only; it is
  deliberately not a generated offset stored in editable source. Existing
  imported animation relocation remains byte-identical.

  The controller writer forwards these explicit targets to generated assembly,
  which resolves and validates the actual opcode and controlled lower plane.
  Missing, wrong-room, wrong-stream and wrong-opcode targets reject before any
  file is emitted. Tests use newly generated records without source offsets,
  including two indistinguishable wall copies to prove exact selection. Native
  headers, collision line identities and legacy wall texture bases stay intact.
  All 854 model tests pass with two workers, including the original 16-map exact
  assembly checks. Sixteen Chromium/Firefox native range/motion checks also
  pass, covering edited destinations, collision, wall geometry and independent
  copied-controller movement. This is not yet a published Make door/Make lift
  workflow.

  Next integration: create source-owned controller settings independently of
  copied-controller templates, generate moving lintel/riser bands over the full
  travel range (including zero-height resting bands), bind both sides of portal
  collision boundaries, preserve them through save/Undo/copy/geometry changes,
  then expose creation/removal and native motion/collision/playtest acceptance.
  AMOS `DEFDOOR` / `DEFLIFT` assign groups to rooms and wall edges independently;
  current native `DoorRoutine` changes the lower roof and wall bottoms, while
  `LiftRoutine` changes the lower floor and wall tops. Those relationships, not
  a room-name/type flag, are the required authoring contract. The older AMOS
  settings/header variant must not be confused with the current 18-byte native
  header followed by 10-byte wall records.


- Independent moving-room compiler integration now generates door lintels,
  lift risers and static pit walls from a newly authored convex room and its
  actual reverse portal boundaries. Zero-height resting door bands are retained
  for later movement. Solid sides cover the complete travel envelope. New bands
  enter the existing wall batch, and exact targets are rebound after subsequent
  insertions so two controllers sharing a static neighbour keep their own walls.
  Both collision-line directions refer to the same moving graphic; original
  controller targets cannot be silently taken over.

  `AuthoredMovingRoom` is explicit compiler input in source-room identity with
  selected native settings/material. `buildSourceLevel` accepts these plans;
  `buildPlayableLevel`/editable project source do not yet produce them. This
  intermediate step is not published as a Make door/Make lift UI. It currently
  requires a fresh convex lower moving volume and one complete plane; adjacent
  moving groups, imported conversion and multi-plane grouping remain required.
  Source-owned settings, starting closed doors, history, save/reopen/copy and
  creation/removal controls are the next integration work.

  Evidence: `objects.c` DoorRoutine patches room roof, wall bottom and legacy
  4096-byte texture base with absolute row scrolling; LiftRoutine patches room
  floor/wall top and `renderer_lift_wall_texture_info_for_gfx_off` consumes packed
  texture words. The new compiler preserves these distinct decoder contracts.
  Anims.s DoWaterAnims requires 21 groups after a lift table; authoring the first
  lift emits empty structural groups with no target writes. Invalid ranges,
  wrong initial planes, missing portals/targets, duplicate controllers and
  incompatible decorative ownership fail before emitting a native artifact.

  Native Chromium/Firefox tests compile fresh adjacent rooms (no copied
  controller/room geometry), open the complete generated level in the actual
  engine and cycle both new mechanisms. Plane and wall samples stay synchronized;
  paused endpoint captures give distinct nonzero pixel hashes and show the open
  versus closed door and raised versus lowered platform. The existing copied
  mechanism motion checks pass too. The initial lift test incorrectly requested
  an automatic cycle from a manually-lowered fixture; its rule now explicitly
  selects native timer lowering. Inspection captures are correctly paused because
  the native capture API deliberately returns zero during gameplay.

  Final validation: all 859 model tests pass with two workers, TypeScript passes,
  and eight Chromium/Firefox native motion cases pass (four independently
  authored, four existing copied mechanisms). The four new cases additionally
  pass rendered endpoint captures from the adjoining room; these images were
  inspected to confirm an open corridor/closed door and raised/lowered pit.
  A separate shared-neighbour two-door model case verifies each final graphic
  pointer still targets its own portal edge after both wall-band insertions.


- Make door / Make lift is now an editable authoring workflow for newly drawn,
  connected convex lower rooms. The room inspector offers destinations, initial
  endpoint, original trigger/condition rules and a native moving-wall material
  browser. Apply validates the complete generated artifact before committing
  one source change; Cancel/Escape discards the draft. Remove restores the static
  editable room. Imported conversion/reassignment, adjacent moving groups and
  multi-plane volumes remain parity gaps.

  Room source v10 stores per-room kind, 18 native settings bytes and a stable
  source material identity. It never saves generated pointers or cell/line IDs.
  Settings survive constructors, corner moves, archive save/reopen and room
  clipboard. Copy automatically captures adjoining rooms; internal copied switch
  conditions remap and existing requirements reserve their native slots.
  Creation/removal and all settings share normal Undo/Redo.

  Source air clearance remains editable and valid while generated topology
  receives the initial moving plane. Doors can start fully closed without
  storing an invalid zero-clearance outline. Missing resting solid walls are
  generated across the full travel envelope, so opening that initial closed
  volume reveals complete walls. Lifts can choose either initial endpoint.

  Validation: all 862 model tests pass with two workers and TypeScript passes.
  Eight Chromium/Firefox browser cases verify new-room native motion and the
  complete create/cancel/error/save/undo/reopen/remove workflow. The closed-door
  authoring case uses a real Space press from the adjoining room to open it.
  Four installed Edge/Brave cases pass against the exact packaged release.
  Runtime bundle r-ad32e275c8d0e039 remains unchanged. Publication is the next
  release step; this does not close AMOS parity or the full editor goal.

  Publication: source 86bbd09 is deployed in site commit 423e49d. The public
  index serves index-CgCvR2uB.js, and all four live installed Edge/Brave workflows
  pass creation, draft cancellation/errors, save, Undo/Redo, reopening, actual
  Space use/native motion and removal for both doors and lifts.

  A preview refinement follows: selecting a newly authored closed door keeps
  its inspector selected while moving the inspection camera into an adjoining
  room and looking toward the door, instead of putting the eye inside its sealed
  volume. Six exact packaged Edge/Brave/Firefox workflows pass with an explicit
  assertion that this camera is outside while the door stays selected. Editable
  data and game starts remain unchanged. The remaining full parity work stays
  open, including imported-room conversion and grouped moving volumes.

  Final release verification: source ff8af3d, site 3c64790, frontend
  index-QE5jBBy9.js. The public URL serves the refined build. Both final live
  Edge create/save/reopen/play/remove cases pass, including Space use and the
  closed-door camera/selection assertion. Earlier live Edge/Brave creation cases
  and all six final packaged Edge/Brave/Firefox workflows pass. Source and site
  changes are committed and pushed; unrelated site abhd/abhd.zip is untouched.


- Object inspector redesign: Type is a direct named selector at the top,
  grouped as Enemies, Pickups and Scenery. It applies immediately as one Undo;
  the duplicate read-only Type row, Collision ID, template explanation and
  separate Apply action are removed. Type-specific controls follow the selector,
  and Duplicate/Delete share one compact row. Tooltips explain Undo and retained
  position without making implementation details part of normal editing.

  The type catalog reads pristine active records from all 16 released native
  maps, excluding reserved projectile pools and player avatars. Fifteen unique
  released classes are available independently of the current level; artwork,
  dimensions, health and type parameters change together. CID, position, height,
  logical/graphic room and storey retain their identity. Original AMOS-only
  definitions absent from released maps are not fabricated; the complete object
  bank, variants and placement workflow remain part of the full goal.

  All 878 model tests pass. Six Chromium/Firefox browser checks pass, including
  direct selection, compact ordering, Undo/Redo/save, protected duplication,
  library failure/retry and a native class absent from the starting map. The
  rendered inspector screenshot was reviewed. Packaged and live verification
  follow before reporting the redesign as published.

  The exact packaged redesign passes six installed Edge/Brave/Firefox checks,
  including the cross-level native class change and asset-failure recovery.
  This release updates the inspector; the separate Add/Duplicate placement
  workflow and original AMOS-only object banks remain follow-up work.

  Published and verified: source 011239f, site 08fd374. The public index loads
  index-C5kGN_-q.js and index-wazFJgEI.css. All four live installed Edge/Brave
  checks pass direct named selection, cross-level native class loading, history,
  saved bytes and library failure/retry. Runtime manifest hashes are unchanged;
  unrelated site files are untouched.


- Eligible existing rooms now support independent door/lift creation without
  replacing their original source identities. Native movement/switch graphics
  and collision ownership are checked before conversion. Existing owners remain
  editable through a Go to existing link that selects and frames their room;
  adjacent independent moving volumes still need explicit synchronized grouping.
  Original starts, actors and controller settings retain their links/state.

  Static geometry is compiled at its editable clearance first. The initial
  moving plane/header is then applied to generated graphics, and the mechanism
  owns its lintel/riser bands. Closing an imported room no longer requires the
  user to assign repair appearances to every neighbouring room. Point-coordinate
  aliases are matched geometrically, and changing a generated band opcode drops
  incompatible imported record provenance. New bands cannot overwrite existing
  moving or switch targets.

  Moving-wall material choices include the map's existing native styles, not
  only walls already drawn in the selected room. Thus a portal-only volume can
  choose its face material explicitly. Selected source styles are bounds-checked
  against the native texture bank. Archives retain only stable source settings;
  generated pointers never replace imported IDs.

  A picked authored moving face now offers Edit door/lift, selecting its source
  controller. The ordinary wall-paint action is suppressed for that generated
  face because the controller would replace a neighbour's paint during rebuild.
  Normal wall painting and floor/ceiling pattern painting remain independent.

  All 895 model tests pass, including door/lift conversion on all 16 released
  maps, original starts/actor positions, existing controllers, save/reopen and
  cross-room material selection. Fourteen Chromium/Firefox workflow/attachment
  checks pass; ten further conversion/picking/normal-paint checks pass with the
  contextual moving-face action. Packaged and live release checks follow. This
  advances R0 while controller reassignment, grouped/multi-plane/curve ownership,
  navigation authoring and the rest of the full editor goal remain open.

  Final local verification: all 895 model tests pass after the ownership and
  moving-face changes, and TypeScript passes. The exact packaged build passes
  15 installed Edge/Brave/Firefox cases covering original-room conversion,
  generated-face Edit routing, existing-owner navigation, fresh-room creation,
  archives/history and native movement. The native bundle remains unchanged.

  Publication verified: source 940e363, site 0c6159a, frontend
  index-Dh0puXrA.js. All six live installed Edge/Brave cases pass conversion,
  save/history/native motion, moving-face Edit routing and existing-controller
  navigation. The source worktree is clean, deployment resources match the native
  manifest, and unrelated site abhd/abhd.zip remains untouched. Full AMOS parity
  and the remaining R0/R1-R12 editor requirements remain active work.


## Axis and editable 3D views

The game renderer is now explicitly **Engine view**. **Editor view** offers Top,
Bottom, Front, Back, Left, Right, 3D orthographic and 3D perspective; Split view
combines the selected editing projection with the native engine. Top retains
room drawing and rectangle selection. Other projections expose editable source
room volumes, current lower/upper planes, actor anchors and player starts.
Native coordinates remain authoritative (larger Y is down); the scene does not
claim to reproduce decorative graphics or material rendering.

Camera projection, inverse rays, nearest-surface picking, near-plane clipping,
cursor-anchored zoom and constrained transform math live in `editorCamera.ts`.
`editorScene.ts` derives volumes from room outlines and `roomSettings`, and
emits existing validated commands for corners, heights and actor placement.
One drag commits one history entry; previews and rejected/cancelled gestures do
not alter the saved document. Actor XY/ZY dragging and XYZ handles preserve
locked dimensions; room height handles keep water independent. Object anchors
are included when framing, even outside the room's floor/ceiling interval.

User controls are in tooltips, the viewport footer and Help. Keyboard handles,
upper/lower editing, perspective click placement, saved source and actual native
object coordinates are covered by model/browser scenarios. Exact game surfaces,
materials and gameplay remain available in Engine view. Complete AMOS parity
and the remaining requirements above remain open.

Verification for the published axis/3D release: 922 model tests passed. The
48 packaged controls, shortcut, height and viewport regression checks passed
across installed Edge, Brave and Firefox. All five viewport scenarios also
passed against the final package (the side-axis keyboard case was checked
separately). Ten further checks passed on the public deployment in Edge and
Brave, including native object coordinates, upper-floor placement, save/reopen,
height cancellation and corner Undo. Publication is site commit `18bd235`,
JavaScript `index-BEa1x5u2.js`, CSS `index-DwRnkAfo.css`; Pages run `41315345996`
completed successfully. All five native runtime resource hashes remain verified
under `runtime/r-ad32e275c8d0e039/`.


## Ceiling usability and positive room height

Room heights now lead with a positive headroom size instead of requiring an
inverted absolute ceiling coordinate. `setRoomHeadroom` derives the selected
ceiling from its existing floor and delegates to `setRoomHeights`; floors,
water and the other storey remain unchanged. Native range/storey constraints,
project storage and compilation stay shared. The side-view button frames the
selected room; the section labels usable air and separates coincident lower
ceiling/upper floor handles. Show ceilings explicitly describes visibility only.
Absolute positions remain available in a readable two-column layout. This
addresses the user's reported ceiling confusion while the remaining AMOS and
R0-R12 requirements continue to apply.

The ceiling usability release passed all 926 model tests, 27 packaged height/
viewport regressions across Edge, Brave and Firefox, and six focused headroom
checks against the final package. Four further checks passed on the public
site in Edge and Brave, including unchanged native player floor height and
independent upper/lower editing. The final publication is site `75ecf3c`,
JavaScript `index-Dq_YeS1p.js`, CSS `index-BepOojkE.css`; Pages run `41316825535`
completed successfully. Native runtime hashes remain unchanged and verified.


## Direct outline corners and texture discovery

The original `PTADD` at `leveld.asc:7597` creates whole-coordinate authoring
points; source outline editing now exposes insertion on an existing edge and
safe shared-corner removal, without requiring a complete redraw. Slanted edges
use their integer lattice, shared segments split on both sides and keep their
native wall/paint identities. Removal changes the incident room fan atomically,
merges identical split material identities and assigns genuinely new boundaries
new identities. It rejects invalid outlines, fewer than three corners and
stranded starts, active objects or teleport destinations. Save/Undo reuse the
existing source schema. Unattached/decorative points remain an explicit gap;
this does not establish the complete AMOS Add/Delete Point operation.

The room inspector now leads with Textures and Paint room textures, routing to
the actual renderer's existing wall/floor/ceiling pickers. Wall repair appearance
is identified separately. Help explains the selection, Apply/Cancel, storey and
moving-surface workflows; unsupported special material targets are explicitly
labelled. Material formats and native gameplay are unchanged.

Verification: all 947 model tests passed, including split/compile/save/merge
checks on one native edge in each of the 16 original maps and protected shared
corner dependencies. Nine initial Edge scenarios exercised all lower/upper wall
and normal floor/ceiling painters through the new inspector entry, plus native
corner pixels/collision. Fifteen packaged cross-browser checks then passed on
Edge, Brave and Firefox; the final package additionally passed all 12 corner
scenarios, including pointer-position double-click insertion and Space-pan
isolation. Four live checks passed in Edge and Brave for native pixel-preserving
corner edits and all 16 normal lower-floor patterns through Paint room textures.
Site commit `b865ca5` serves JavaScript `index-B71ShpUG.js` and CSS
`index-Bgk5Y1Rh.css`; Pages run `41319077670` succeeded. Five native resource
hashes remain verified under `runtime/r-ad32e275c8d0e039/`.


## Default textured FPS editor and native material banks (historical integration)

The embedded-engine default described here was withdrawn after user review.
The editable geometry correction below supersedes that view integration; native
material-bank support and runtime line-ending fixes remain in place.

Editor view now opens in 3D perspective using the actual game renderer, original
textures and lighting. WASD moves the paused inspection camera, right drag looks,
Q/E changes height and Shift increases camera speed. Controls are visible below
the viewport and in Help. Axis and orthographic views retain geometry handles;
Split opens a map alongside the renderer. `?view=top` and the other documented
view IDs select an initial projection explicitly; otherwise perspective opens.
Inspection navigation does not run actor AI or modify saved player starts.

Native floor picking inverts the renderer's active projection and view rotation,
including scaling and look position, to place palette objects on the clicked
floor. Floor picks preserve the actual upper/lower storey. Escape cancels repeat
placement. Valid camera position, angle and look survive source rebuilds and
Undo; invalidated positions return inside the selected room with a notice.

Floor and ceiling painting now includes normal, chunky bump and smooth bump
material banks, with the original bytes/palettes. AMOS FGPUT/RGPUT and its
compiler map normal floor/roof 1/2 to chunky bump 8/9. Original AB3DI.s polyloop
and BumpLine also implement smooth 10/11 with one fixed texture; its pattern
selector is therefore omitted. Explicit mode overrides use room-source version
11 and reject older-version downgrades; legacy overrides remain readable.
Same-plane opcode conversions require an explicit source override; geometry,
brightness, record provenance and moving-controller attachments stay intact.
Ceiling height anchoring and native labels now correctly classify opcodes 9/11.
Sky/no-roof authoring and the remaining AMOS/R0-R12 requirements remain open.

Verification so far: all 961 model tests pass, including twelve lower/upper
floor/ceiling mode conversions, portable source/history and native-bank palette
sampling. Six initial packaged Edge viewport scenarios passed, including default
FPS navigation, native floor click placement and camera preservation on Undo.
Final cross-browser publication evidence is recorded after deployment below.

The initial spawn camera is cached before any navigation, so painting immediately
after opening also preserves the view. Returning from a running playtest to the
perspective editor stops simulation and rebuilds editable inspection. Runtime
bridge files are pinned to LF in Git: Ubuntu builds the native artifact for both
CI hosts, and Windows must compare exactly the same bridge bytes rather than
fail the deliberate stale-runtime guard because of checkout line endings.


## Editable textured 3D correction: both storeys and height gestures

Perspective and orthographic again use the editable spatial viewport. The
perspective editor remains the default; Engine view stays separate and Split
pairs the two. Native compiled wall/polygon records provide the textured mesh,
not opaque room-volume boxes: portal openings, partial polygons, source surface
identity, wall strip endpoints, phase, masks, repeat and saved overrides remain
available to editing and paint. Original indexed sheets and palette levels
supply colours. Authored point/wall and room/polygon brightness are shown without
game-distance darkening so an overview remains readable. This is an authoring
renderer, not a claim of pixel-identical game rendering.

Both storeys and their objects are visible in both 3D projections. Lower/Upper
still chooses the active editing layer without filtering the mesh or reframing
the camera. Selected rooms expose LF/LC/UF/UC handles; shared-height boundaries
have separate handles. Pointer drags preview validated native heights and commit
one ordinary command on release; Escape cancels. Alt-drag on an actual boundary
floor/ceiling is a direct height gesture. Save/Undo retain independent storey
heights, source corner geometry and controller validation.

Surface picking and Paint room textures operate in the geometry editor. Wall
and polygon dialogs reuse the same source commands as native playtest/export.
Paint rebuilds the mesh while retaining the editing camera. Palette objects can
be placed on either visible storey. Perspective offers Orbit and Walk modes:
Walk starts from player one inside this same editable mesh, with WASD movement,
right-drag looking, Q/E vertical movement and Shift acceleration. It does not
run the game or modify player starts. Objects remain editable markers; animated
water and curved-wall graphics still require Engine view for exact appearance.
Texture/GPU failures are explicit and retryable; geometry outlines and axis
editing remain available. Help and inspector text now describe this workflow.

Initial verification: all 963 model tests pass, including all 16 native meshes
and independent four-plane source/history checks. Five initial Edge editing
scenarios proved real textured pixels, both projections, all four height drags,
Walk placement, and direct wall/floor painting without a game iframe. Final
cross-browser and publication evidence follows after the release checks.


Release verification: all 36 packaged editing/axis regressions passed across
Edge, Brave and Firefox, including both-storey four-handle pointer drags, direct
Alt-drag floor/ceiling editing, material-load failure/retry, Walk placement and
wall/floor painting in the editable mesh. Six further Edge scenarios passed in
the separate native renderer for lower/upper wall and all floor/ceiling pattern
painting, saves, cancellation, history and unchanged player positions.

The final package is published as site commit `e2bcd5d`, JavaScript
`index-Dv4NXA0e.js`, CSS `index-CiVQzSh2.css`. Pages run `41326488319` succeeded.
All 14 public Edge/Brave scenarios passed, proving that the live default is the
editable 3D viewport and exercising both-storey heights, direct surface drags,
paint, Walk object placement and texture retry. The five native runtime hashes
remain unchanged under `runtime/r-c8b6aa3eff4f5614/`. Source changes are committed
and pushed; full Windows/Linux editor CI is running separately. This milestone
does not complete the remaining AMOS and R0-R12 scope.


## Fixed-size height diagram and retained doorway materials

The height diagram is now a fixed 400 CSS pixels high. Its SVG uses the same
400-unit height and a 352-unit plotting area; pointer conversion still uses the
screen transform and native quarter-unit validation. Both single- and two-storey
diagrams keep that height as their values change.

The reported ceiling drag exposed a compiler error: height changes made an
existing doorway look like a new/replaced connection, requiring repair styles
on neighbouring Room 5 and turning the whole textured editor into outlines.
Retained native connections now have a separate height-only path. Existing
bands retain source identities and sampling; newly exposed solid areas inherit
an actual wall style from the room's perimeter, other storey or adjoining room.
Switch graphics are not replicated. Intentional original omissions remain;
new bands have source boundary identities and explicit native stream insertion
positions rather than being appended after water/object drawing controls.
Door/lift apertures use their complete authored swept ranges, preserving their
moving attachments instead of sealing them at their initial position. Real new
or replaced boundaries still require explicit appearance and controller setup.
No room-appearance choice or new project schema is required for height edits.

Verification: all 966 model tests passed, including height-only portal edits in
all 16 original levels and the reported -40 ceiling / 240 floor combination,
portable saves and Undo/Redo. Eighteen packaged height/headroom checks passed
across Edge, Brave and Firefox, including the larger diagram, live texture
retention, actual native door travel with synchronized wall attachments,
collision/rendering, cancellation and keyboard height edits. Firefox's pointer
regression scrolls the complete SVG into view before dragging its lower handle.

Published site commit `766af5c` serves JavaScript `index-B5QxB2pK.js` and CSS
`index-UvzKxj_Z.css`; Pages run `41330535581` succeeded. All four live Edge/Brave
ceiling-drag and native door-motion regressions passed. The five native runtime
hashes remain unchanged under `runtime/r-c8b6aa3eff4f5614/`. Remaining AMOS and
R0-R12 scope is still open.


## Selection gestures across editable views

Left click selects and left drag draws a crossing zone selection rectangle in
Top, orthographic and perspective views. Shift adds zones; either rectangle
direction works. Selection does not edit source, add Undo entries or move the
camera. Room groups can be framed and remain highlighted in 3D.

Middle drag pans; right drag pans in Top and orbits in 3D (looks in Walk).
Space-left and Alt-left no longer navigate or adjust heights. Explicit corner,
LF/LC/UF/UC and object-axis handles remain editable; Top objects, groups and
player starts have a separate Move selection handle. Object placement, room
drawing and the dedicated object rectangle tool keep their explicit workflows.
Help, inspector guidance and tooltips explain the current controls.

The generated doorway-band material picker also retains the compiled native
wall as its initial style when no explicit room appearance exists (3fb9727).
This avoids an undefined style when painting geometry exposed by height edits.

Verification: all 969 model tests passed. Twenty-four controls/selection checks
passed across Edge, Brave and Firefox. After updating the object-group test's
stale library-dialog selector, all 39 final editing regressions passed across
those browsers: crossing selection in both directions, additive Shift clicks,
unchanged source/camera/Undo during selection, explicit object handles and
middle navigation, both-storey height handles, Alt-left selection, texture
painting with portable Undo, native object positions and atomic group edits.
Implementation is 7a9433f; the selector correction is 68b48d6. No native runtime
or project schema changed for selection.

Published site afb050f serves index-iVVJKPbC.js and index-Cx88Qus4.css.
Pages run 41333340819 succeeded. All six public Edge/Brave selection regressions
passed (Top, orthographic and perspective), including unchanged source and
camera during selection, Shift grouping, explicit movement handles and middle
navigation. All five native runtime hashes still match r-c8b6aa3eff4f5614.
Remaining AMOS and R0-R12 scope stays open.


## Multi-zone floor and ceiling handles

Room groups now retain LF/LC handles in editable 3D and UF/UC when every selected
room has an active upper storey. A selected room anchors the handles;
a gesture applies its height delta to the matching plane in every selected
room, preserving existing offsets. The shared spatial command builds a complete
provisional RoomSource, validates each room with the existing quarter-unit and
storey rules, and publishes one command only if every room succeeds. Preview,
Escape, keyboard nudges, portable saves and one-step Undo use that same path.
Single-room corner handles remain tied to single-room selection. Inspector
notes and handle tooltips explain group behavior and upper-storey requirements.

Verification: 20 relevant model tests passed, including preserving relative
floor heights, saved snapshots, one Undo/Redo and rejection after a later room
fails validation. All 30 packaged Edge/Brave/Firefox height and selection
regressions passed. The six group tests drag and cancel each of LF/LC/UF/UC in
perspective and orthographic views, compare both rooms' saved plane values,
undo each gesture once, and retain selection, camera and textured preview.
Implementation: d73621e. No project schema or native runtime change.

Published site 7cf4e5d serves index-DyLwurzD.js with unchanged
index-Cx88Qus4.css and native r-c8b6aa3eff4f5614 (all five hashes checked).
Pages run 41333976458 succeeded. All four public Edge/Brave group-height
regressions passed, covering both 3D projections and all four planes with
cancellation, saved relative values and one-step Undo. Remaining AMOS/R0-R12
scope stays open.


## Editor terminology and material choices (2026-11-05)

The room inspector now explains zones as map areas rather than assuming each is
an entire visible room. Height controls explain the two spaces sharing one
outline and the template workflow for drawing a two-storey room. Toolbar
buttons say Lower storey / Upper storey, and template options show their storey
count. The existing labelled side diagram remains the height editing guide.
Ceiling help distinguishes openings between storeys from sky openings and
states that new opening authoring is unsupported; ceiling visibility changes
only the editing view.

At the user's request, unused chunky bump and smooth bump choices are disabled
in the surface material selector. Existing saved materials remain readable and
preserved; the native material model and renderer are unchanged. The picker,
controls help and README use the same wording. Browser selectors follow the
new storey button labels. No substitute geometry or rendering was added.

Validation: TypeScript and production build pass. All 12 Chromium editing-view,
height and ceiling regressions pass. All four floor/ceiling material scenarios
pass in Chromium and Firefox, checking disabled bump options, normal painting,
portable saves and Undo/Redo. Seven polygon-material model tests pass.
`git diff --check` passes. New ceiling/sky opening authoring remains unsupported.


## Room surface, storey and opening authoring (2026-11-05)

This completes the authoring workflows raised in the editor feedback and
supersedes the earlier ceiling/sky-authoring limitation above. Floor and ceiling
textures now have direct inspector buttons, independently for each storey and
source polygon. They open the same native material picker as 3D surface picking,
including pattern thumbnails, repeat, Cancel, Apply and Undo. Imported outdoor
rooms can create a missing ceiling from an explicit normal material reference.
Unused bump choices remain disabled; existing saved material modes are retained.

Add upper storey and Remove upper storey now edit actual source geometry.
Creation records explicit straight-wall, normal floor/ceiling and corner-lighting
references, plus validated independent upper heights. Starting pattern choices
are available even when the lower room contains partial decorative surfaces.
Upper surfaces retain independent paint and lighting without replacing lower
records. The geometry-lighting planner distinguishes storey-specific authored
claims when one original material supplies several surfaces.

Ceiling open to sky removes the top ceiling from the saved graphics and emits
the native backdrop marker. Rectangular opening authoring offers sky or a
connection between storeys, with a plan preview and explicit wall/lighting
choices. It partitions the footprint into the opening and surrounding native
zones. An inter-storey opening has a tall single-storey shaft, so native portal
movement permits falling from the upper floor. The shaft retains the upper
ceiling pattern and lower floor pattern. Starts, objects and incoming teleport
landings are reassigned by position. An upper object in the proposed shaft must
be moved first. Native coordinate/zone limits, explicit exit ownership and
moving mechanisms are validated before one atomic history entry. Concave
outlines use existing convex decomposition; bounds must remain inside the room
and cuts must meet whole native coordinates. No invisible collision floor or
replacement game behavior is introduced.

Original authority: leveld.asc DEFUPPERFLOORHEIGHT, DEFUPPERROOFHEIGHT,
DEFFLOORGRAPH/DEFROOFGRAPH and RGPUT's NONE (SKY); current native renderer.c
opcodes 1/2/12 and backdrop cache; level.c level_zone_has_upper_layer and
movement.c checkotherwalls/FindRoom air-band selection. Room-source version 12
stores the additional authoring data. Project saves/recovery, clipboard copies,
playtest and export continue to use the shared source compiler. Other AMOS and
R0-R12 requirements remain tracked separately.


Verification for this checkpoint: production TypeScript/Vite build passes.
The full model suite reports 975 passes and one pre-existing failure:
wallTextures.test.ts expects io.c hash bb436acfb7680a3b954e27365341b294816862948486c9118fc5f3cd7af19933,
while the committed HEAD source normalizes to b7bb7c82b7aaf1b5d7d711e88d31a8d53412d9f313970ff869791b3122f2b16c.
Neither that source nor its oracle was changed by this editor task. Six new
storey/opening model scenarios pass, including inherited-storey removal,
clipboard/project round trips and adding a separately lit upper storey above a
partial lower floor. The upper stream includes LEVELSAVE's dry ObjDraw tail
(opcode 4, full-room clip mode 2), so upper objects have a native draw pass.

All eight Chromium/Firefox authoring scenarios pass, including direct lower
and upper floor/ceiling paint, Cancel, portable saves, Undo, missing-ceiling
creation, rectangular skylights, sky/frame changes and opening creation. Native
play walks from an upper landing onto the authored upper floor, then falls
through the opening and settles on the lower floor. Four existing native height
handle regressions also pass across both browsers. After moving direct texture
controls above room-outline editing, both browser visibility checks pass without
scrolling, and the inspector screenshots were visually reviewed. Existing
material-bank and editable-3D scenarios passed earlier in this change.
`git diff --check` passes. Remaining unrelated AMOS/R0-R12 work stays open.


## Dotted zone connections (2026-11-05)

Top view now has a solid-wall/dotted-connection legend and an expandable diagram
with steps for drawing adjoining rooms. The room-outline inspector and controls
help expose the same explanation: draw an ordinary room beside another room
with an exactly shared edge, finish and apply appearance. Partial shared edges
connect too; corner contact and overlapping rooms do not create a passage.
Heights and doors still control passability, and solid boundaries between two
imported zones remain preserved by the existing compiler rules.

New/edited outlines previously drew every edge solid. Map connection strokes
now use compileRoomTopology plus buildDrawTopology, matching export's actual
exterior boundary pairing and omitting internal convex seams. Shared edges draw
once, keep their dashes under selection and use selection priority. Invalid
compiler geometry keeps editable outlines with an explicit connection diagnostic
instead of guessed or stale portal classifications. Native game behavior and
saved project formats are unchanged by this presentation change.

Verification: production TypeScript/Vite build passes. Two model scenarios cover
whole and partial edge pairing, corner contact and concave decomposition seams.
All 18 Chromium/Firefox browser checks pass: selected authored-edge pixels have
visible dash gaps; help and drawing cancellation leave history unchanged;
existing room tools, recovery and native doorway walking/closure remain valid.
The expanded guide screenshot was visually reviewed. No placeholder connections
or game-engine changes were added.


## Explicit existing wall connections (2026-11-05)

The room inspector now exposes Wall connections (solid / dotted) for existing
adjoining rooms. Open, Close and Use automatic update reciprocal source boundary
states. Closing has explicit wall/lighting choices on each side and a cancellable
draft; Apply rebuilds collision, graphics, visibility and clipping as one Undo.
Opening also cuts a solid boundary re-imported from compiled game data. Door/lift
and switch-owned rooms retain their mechanism protections. Normal drawing still
pairs new shared edges automatically.

Authority: leveld.asc DEFWALL (ordinary WT 0/1 toggle; special b/v wall types
remain separate unfinished work). Room-source version 13 stores stable boundary
IDs and neighbour-specific overrides. T-junction/corner insertion retains the
connection identity; copied internal room relationships receive new neighbour
IDs. Skylight/shaft partitions retarget outside connection ownership instead of
reopening a previously closed boundary. Conflicting reciprocal states reject
compilation rather than selecting one side. No native gameplay rule changed.

Two model scenarios cover native close/reopen, strict game-data re-import,
Undo/project/clipboard round trips, inserted corners and skylight ownership.
All 70 targeted model checks pass. All 10 Chromium/Firefox dotted-edge,
portal-opening/closure and explicit-wall checks pass, including blocked movement
against the closed wall and crossing the re-imported solid wall after opening.
TypeScript/Vite build passes. Broader R0-R12 requirements remain open in the
current completion audit; no final deployment is claimed.

The pre-existing native wall-oracle hash failure was investigated against the
committed io.c changes (17ce3d8 and 0e1e505). The standalone dump harness now
includes the actual renderer_pixel.h used by native palette conversion. Rebuilt
the independent C =SB= unpacker and reran editor_wall_oracles.py with clang.
All 14 unpacked sheet hashes, dimensions and RGBA hashes are identical to the
previous oracle; only the reviewed source hash and new pixel-header hash differ.
The model guards both dependencies, and all 17 wall texture checks now pass.


Checkpoint verification after wall connection and oracle integration: all 980
model tests pass (88 files) with two workers and a 30-second per-test limit.
The prior native-source hash failure is resolved by reviewed C regeneration,
not by suppressing or weakening the assertion. The relevant 10 native-browser
scenarios pass in Chromium and Firefox. Final commit/push/deployment remains
pending completion of the remaining authoring and R0-R12 gates; the public site
has not been replaced with this partial checkpoint.

The dotted-line help now leads with the existing-room workflow: select either
room, expand Wall connections (solid / dotted), and choose Open edge beside the
neighbour. A separate illustrated sequence covers drawing an adjoining room.
It explicitly defines a zone as a room or room section, explains the missing
neighbour case, and directs ceiling/sky holes to Storeys and openings. README
instructions match the in-editor guide. The targeted connection/storey model
checks and both browsers' dotted-line and native wall open/close checks pass.

### Source-room exit authoring

Level exit now includes newly drawn rooms and lists each current room once.
Room-source v14 stores the selected stable exitRoom identity; choosing a numeric,
original-slot or disabled exit clears that override atomically with Undo.
Geometry edits and room copies retain the one project exit. Deleting its room
or partitioning it into an opening requires reassignment first. Missing imported
exit rooms produce a visible selector diagnostic and a failed build.

The compiler maps the source exit to its generated representative and retains
the existing ABEX v2 draw-room membership. Native game_loop.c uses that explicit
membership for completion: all collision cells of one concave authored room
form one exit. Without draw groups this remains AB3DI.s addExit/ENDZONES's exact
room comparison. No original binary level fields or completion conditions change.

Rebuilt the real single-threaded preview. The C feedback_draw_rooms check passes,
including disabled, unrelated, grouped and legacy identities. All 983 model tests
pass. Chromium and Firefox each verify source selection, Undo/Redo, source save,
game export, completion from every generated cell, disabled free playtesting,
and original-slot/numeric exits. The broader audit and final release remain open.

After the selector cleanup, all four exit browser scenarios pass again. Ten
additional Chromium/Firefox storey, shaft, direct texture and reciprocal wall
collision scenarios pass against the rebuilt native preview.

Next control-point work must distinguish LEVELSAVE's eight-byte runtime
coordinates at data +54 and room control bytes +26/+27 from LEVELSAVE/LEVELLOAD's
separate 100x100 twolev.links source graph. readAmosProject/writeAmosProject
already preserve that source graph. Native level.connect_table is the rendering
point/clip table consumed in AB3DI.s and renderer.c, not navigation links; it
must not be repurposed for this authoring feature.

### Control-point and link-graph authoring

Room-source v15 now stores stable control identities, owned lower/upper points,
explicit room assignments and an optional original-format 100x100 link bank.
An absent graph is distinct from an explicitly empty bank. The inspector imports
the matching twolev.links or creates a new graph on request; native maps cannot
supply the missing source graph. Download writes the actual bank, while project
saves retain all metadata and inactive imported bank bytes until compaction.

Implemented PLACECONTPT/CPTMOVE diagram gestures, exact room/coordinate/storey
editing, independent lower/upper assignments, all CONNECTCONT physical and
visual link pairs, AUTOLINK preserving physical pairs, and CPTNEARTOZONE's nearest
visible lower assignment (single/all rooms). The automatic tests use geometric
shared-edge traversal independent of wall type, doors and heights, as the AMOS
routines do; concave outlines are handled through convex clipping. Failed
all-room assignment applies nothing. Graphical/numerical edits, source saves,
Undo, point deletion/link compaction, copied room controls, whole-room movement
and skylight rebinding preserve stable relationships. Upper shaft points must
be moved off the removed floor first. The table limit is 100 points.

The compiler writes LEVELSAVE's point coordinates/height/storey records at data
+54, recomputes owned midpoint heights and maps assignments to bytes +26/+27.
After object/start relocation, it relocates every absolute prefix pointer for
the new control count, then lets normal room/graphics assembly resolve fresh
zone offsets. The original object/shot pools and renderer connect_table remain
unchanged in meaning. These edits preserve original authoring data and do not
introduce new AI/gameplay behavior.

The Top-view control overlay shows both storeys, physical direction arrows and
visual dashes, with all/physical/visual/none link filters from the original menu.
The inspector's room diagram is editable; its Escape handler owns cancellation.
Point labels follow current table order while persisted identities remain stable.
User instructions in Help and README cover placement, assignments, links and
the source/runtime distinction.

The full model-suite checkpoint passes 1008 checks (90 files). The current
navigation target passes 26 checks, including a subsequent inherited native
reference-compaction case, the sixteen original maps, relocation, graph
validation, copies, height changes, room moves/partitions, occluded bends,
concavity and corner-only contacts. All
14 Chromium/Firefox navigation, dotted-connection, storey/texture/shaft and exit
scenarios pass. The navigation workflow includes real native rendering and
player movement after export compilation. Broader R0-R12 work and final
commit/push/deployment remain open.

### Independent water surfaces and animation groups

Room-source v16 adds per-room water coverage/height/group settings and overrides
for the native 21 water-animation slots. The controls follow DEFWATERANIM and
LEVELSAVE in leveld.asc, Anims.s DoWaterAnims, and objects.c do_water_anims.
Static lower-storey water can retain existing partial polygon shapes/patterns or
cover a complete authored outline using an explicitly selected pattern. Room
assignments can join, leave or change groups independently. Disabling a group
holds its members at the configured starting height; enabling it restores the
attachments. Removing water also moves the collision plane below the floor,
using the original dry-room convention. Template inheritance is undoable.

Group top/bottom/current values retain native 24.8 precision and initial signed
velocity. The game retains its original fixed +/-128 velocity after reversal.
Polygon heights use the same arithmetic-shift quarter-unit rounding as the
native routine. Initial collision and polygon heights agree to that native
precision in paused inspection as well as play. Invalid/missing group settings,
material references and ranges produce actionable diagnostics before writing.

The water pass writes LEVELSAVE's object-below/water/object-above sequence,
preserves existing pool geometry when requested, and generates convex pieces
for whole concave rooms. Direct compiler-owned graphics targets bind every
water polygon to its controlled runtime cell. Explicit overrides remove old
inherited group writes; untouched originals and native-material copies retain
their existing sharing. The native limit is 256 animated polygon attachments
per group; decomposition counts towards it and over-limit builds stop clearly.
If the imported level has no lift/animation table, assembly writes the required
lift 999 terminator followed by all 21 slots so the native loader finds WaterList.

Project saves/Undo preserve source settings. Copies retain shared group IDs and
restore captured settings when the target workspace predates group creation.
Room deletion drops its generated water attachments, and normal room partition
metadata carries water into the resulting rooms. Native water picks now direct
users to Water and animation groups rather than claiming no authoring control.
Inspector height readouts use authored water settings. Closed water/group drafts
are unmounted, so empty unused form fields cannot block Save project.

Eight water model scenarios cover static/dynamic creation, shared membership,
pause/reassignment/removal, existing pool preservation, missing native tables,
copies/concavity/deletion/Undo, damaged source and native phase precision. The
full model-suite checkpoint passes 1017 tests in 91 files. The 104-check targeted
animation/source regression and 18 Chromium/Firefox authoring/import/navigation
checks pass. Both browsers verify shared polygon/collision motion and a map
whose imported baseline has no animation tables. Full editor completion and
final commit/push/deployment remain governed by the open R0-R12 audit.

Final water browser checks additionally capture a valid inspection camera,
sample native picks to find a rendered opcode-7 surface, click it and verify the
water-authoring guidance. Both Chromium and Firefox pass the visual/picking,
shared-motion, pause/removal and no-animation-table scenarios. The source-water
target also rejects a restored room assignment whose group is unconfigured;
no unused-slot default can become an implicit runtime animation.

### Automatic starter textures for new rooms

The user requested automatic defaults so finishing a room immediately gives a
working textured preview. The user-facing finishRoomDraft command now commits
the outline and a saved RoomAppearance together as one Undo. It selects real
wall styles, normal floor/ceiling material records and actual lighting, preferring
the selected template and using level materials for missing styles. Two-storey
rooms get independent lower/upper surface references. Missing resources fail
before committing the new room; no placeholder texture or unsaved preview-only
material is generated.

The appearance's optional surfaces recipe creates complete planes for the new
outline, so partial template decorations or independent control geometry cannot
prevent a fresh room from compiling. Imported rooms retain their original
geometry. Draw order retains the appropriate object/water clips; inherited sky
settings apply only to the active top storey. Wall-style changes and redraws
retain the saved surface recipe. Individual textures remain paintable; stairs,
room copies and opening partitions preserve the recipe. Older manually authored
appearance files remain readable without silently replacing their geometry.

Inspector sections now use distinct React keys per section and room. Shared
keys had left duplicate floor/ceiling controls after changing selections; the
new-room browser test checks that only one pair is present. Help, the drawing
tooltip, dotted-line guide and README no longer require a separate Apply
appearance step for new rooms.

The full model checkpoint passes 1035 tests in 92 files. The current 20-check
creation target additionally covers both-storey defaults, opening partitions,
independent control geometry and material choices for every template in all
16 maps. The 26 Chromium/Firefox default-texture, painting, stair, opening and
storey checks pass, plus ten room-tool scenarios. Native frame picking verifies
real walls and planes without an appearance setup step. TypeScript/Vite build
and diff checks pass. The broader authoring/release audit remains open; this
checkpoint has not been committed, pushed or deployed as a finished editor.

### Independent lower-wall switches

The room inspector now creates numbered wall switches independently of imported
switch templates. A keyboard-accessible numbered room diagram selects the wall;
the inspector supports wall reassignment, manual/automatic reset and removal.
Instructions explain using Space nearby and linking the same switch number in a
door or lift's requirements. Native surface picking identifies the switch and
links to its activation room, including imported cross-room graphics owners.

The source contract is `leveld.asc`'s `DEFSWITCH` (3905) and `LEVELSAVE`'s
activation points (901), panel (1413/1445), and eight slots (1962), encoded for
the current native format. Panels use actual texture 11, 32 strips, a 64-unit
consecutive point pair and heights floor-64 through floor-32. Native
`objects.c:switch_routine` retains ownership of condition bit 4+slot, phase and
timer behaviour; no new game rules or placeholder resources are introduced.

Room-source version 17 stores slot, stable boundary identity and reset mode.
Whole-room moves and safe straight-wall corner insertion preserve the panel;
missing, shortened, bent or open backing walls give explicit build diagnostics.
The compiler requires full solid backing in one generated cell, respects the
2,048-point/eight-switch limits and preserves explicit moving-surface identities
when inserting the panel. Converting a switch room into a moving room and
deleting it are protected. A required switch must be unlinked before removal.

Room clipboard copies allocate independent slots and remap copied internal door
and lift requirements. Captured clipboard source survives Undo of the original
switch. Maps without any imported switch table can create and copy switches;
unused native records remain inactive. Save/Undo retain editable identities,
never generated graphics addresses or runtime activation state. Reassigning or
removing imported switch controllers remains unfinished, as do the wider
controller grouping and authoring audit requirements.

Validation: all 1043 model tests in 93 files pass. Six switch model scenarios
cover native records, save/Undo/Redo, split-wall moves, reassignment/removal,
slot protection, absent imported tables, independent copies and gated-controller
requirement remapping. Twelve Chromium/Firefox switch/door/lift regression
checks pass. Sixteen additional browser checks cover new-switch rendering and
picking, native activation and gated synchronized door motion, automatic starter
textures, direct floor/ceiling textures, skylights, upper-storey shafts and
solid/dotted wall collision. TypeScript/Vite builds and diff checks pass. This
checkpoint does not close the release audit or publish an incomplete editor.

### Independent straight-wall alignment and brightness

The wall paint dialog now has an Alignment and brightness section for signed
per-face brightness, horizontal tile offset and vertical row offset. Empty fields
inherit source values. Texture, alignment and brightness apply together as one
validated Undo transaction; material reset and alignment/brightness reset are
independent. The room inspector also clears settings whose walls were removed
by an outline edit, allowing the source to be repaired without a working preview.
README and in-app help explain the units and the effect on corner lighting.

`wallSettings.ts` owns the editable settings and strict room-source version 18
schema. Source faces and stable authored boundary IDs have independent lower and
upper values. Saves retain original game files; room moves, corner insertion,
clipboard capture/paste and opening partitions preserve only the relevant face
settings. Removing an upper storey removes its settings. Opening partitions use
actual generated provenance to retain imported face overrides, and validate the
original settings before partitioning rather than dropping broken source.

`wallSettingsBuild.ts` applies settings after geometry/material assembly, keeping
points, strip spans, heights, transparency and draw order intact. Its contracts
are AMOS `DEFLOWERWALLBRIGHT`/`DEFUPPERWALLBRIGHT` (3825/3789) and the current
native WallRoutine's signed `wallbrightoff`, FROMTILE (*16 columns), and totalyoff
(rows). Newly selected columns are checked against the actual painted sheet.
Brightness/row-only edits preserve retained legacy sampling. No replacement
textures, guessed lighting or game rules are introduced.

Native animation bases are synchronized separately. DoorRoutine decodes a
4096-byte legacy tile base and adds absolute height scroll; the compiler inverts
that scroll for the chosen initial row. LiftRoutine keeps a stable decoded base
in the current renderer, so the lift list stores the chosen tile/row directly.
Signed legacy door base bit patterns survive row-only changes. Unedited original
controllers keep their own bases, requirements and collision relationships.
The preview bridge adds read-only native wall phase/base/brightness inspection
for these checks; gameplay routines have not changed.

Validation: all 1050 model tests in 94 files pass, including seven new settings
scenarios covering independent storeys, saves/history, copied controllers,
signed legacy bases, material resets, safe boundary splits, opening partitions
and missing-wall recovery. Twenty Chromium/Firefox checks pass for boundary
painting, editable 3D and actual native door/lift phase/base/brightness during
synchronized motion. Eight additional wall-paint/settings checks pass for modal
cancellation, atomic history, portable source and isolated native pixel changes
from brightness and row alignment. The native preview was rebuilt; current
TypeScript/Vite and diff checks pass.

The broader gates remain open: original repeat-window/fit options, transparency,
bands, curves, special-face lighting and the full WB/ZZ mapping still need their
complete workflows and evidence. This checkpoint is not a finished-editor
commit, push or deployment.

### Original wall repeat/fit and explicit corner lighting

The wall paint dialog now exposes the original 2, 4, 8, 16, 32, 64, 128 and
256-column repeat windows and natural/up/down fit. `wallRepeat.ts` follows
`leveld.asc` WGPUT and LEVELSAVE's half-world-unit column count. Up adds a full
repeat even on an exact multiple; down keeps at least one full repeat. Source
room version 19 records the explicit width/fit without inferring it from an
imported strip span. Choosing Inherit original repeat clears that override.
Texture and settings still commit together as one Undo. Reset wall settings
restores repeat/alignment/brightness while keeping the saved material.

Fitting uses the complete source face or stable authored boundary, including
deformed chains, imported partial/decorative faces, source copies and moving-wall
attachments. Generated pieces share texture columns at cut endpoints. Widths
are checked against the actual selected sheet; a narrow sheet is not silently
substituted when the user chooses a wide window. The normal source/clipboard
schema upgrade survives pasting captured settings after Undo of the original.

The deformed-native-face test exposed the existing need for explicit lighting
when an inserted point leaves its old straight gradient. `cornerLights.ts` and
the new Corner lighting inspector provide that authoring path. Users choose both
native brightness words or copy an existing point, optionally scope the values
to an individual normal floor/ceiling/wall, and choose Static/Pulse/Flicker/Fire
with a native 1/16–16/16 blend. Source version 20 retains coordinates, scope and
exact words. Values move/copy with the corner; removing a corner, sky ceiling or
upper storey removes only its obsolete assignments. Opening partitions preserve
the scopes which actually draw that corner. Reset/inherit and draft cancellation
are available even when another geometry error prevents preview.

`geometryLighting.ts` consumes explicit values before trying inherited gradients
and separates coincident graphics identities for scoped values. Each normal
lower floor, lower ceiling, upper floor and upper ceiling can therefore retain
its own lighting at the same physical corner. Native `level_get_point_brightness`
and the normal Gouraud floor/wall shaders consume the rebuilt words. Copying a
native point preserves unknown retained word bits until a user deliberately
edits its decoded controls. The animation encoding follows currentPointBrights:
signed base byte, low-nibble effect and high-nibble blend-minus-one.

These features do not close the full authoring audit. Original preset/selector
options beyond the current native light effects, special-surface/curve controls,
transparent walls/bands and the other R0–R12 gaps remain release requirements.

Validation: the full model suite passed 1,100 tests across 96 files. A subsequent
focused run passed all 94 room-creation, corner-lighting, geometry-lighting,
wall-repeat and storey tests. Automatic starter materials passed again in both
Chromium and Firefox: finish drawing, inspect textured 3D geometry, save/reload
the appearance, Undo/Redo creation and render the room in the native preview.
The browser run also completed the TypeScript and production-build checks.

### Original static whole-room lighting presets

`roomLightPresets.ts` implements `leveld.asc` PRESETS / UPPERPRESETS keys 1–6
as one source transaction per storey. The six floor/ceiling word pairs are
255/1, 1/255, 1/241, 241/1, 241/15 and 15/241. Keeping 255 and 241 preserves
the original unsigned low-byte encoding of -1 and -15. The inspector exposes
the original preset names under Corner lighting → Whole-room lighting.
Individual normal plane scopes retain independent corner identities; wall
lighting and the opposite native brightness word are preserved. Effective
compiled lighting supplies that opposite word, including existing explicit
overrides. The proposal validates both its input build and complete result
before committing one Undo. Save/reopen retains source assignments.

Fifteen model checks cover the six presets on each storey, preserved animated
opposite words, shared lower/upper material references, concave new-room cells,
invalid input, save/reopen and Undo/Redo. Chromium and Firefox both verify
changed native floor/ceiling pixels on each storey, exact pixel restoration
after Undo, Redo and the saved four-plane assignments. The existing four
corner-lighting browser checks also pass. The initial preset browser failure
was a test synchronization error: Undo must wait for the new preview revision
before reading pixels. No renderer changes were needed.

The four original glowing presets remain open. PRESETS / UPPERPRESETS keys
7–0 derive selectors 1–5 from point numbers and their complements. The current
native decoder only animates selectors 1–3. `Anims.s` BrightAnimHandler populates
three entries of a twenty-word table; the remaining original table semantics
and stable source point numbering require explicit parity evidence before
exposing those presets. This does not close R0 or R10, or the broader release
audit. User instructions and the parity inventory reflect the supported six.

Final checkpoint validation: all 1,115 model tests across 97 files passed;
TypeScript, the browser production build and `git diff --check` passed. The
new preset checks passed in both browsers, as did all four existing native
corner-lighting checks. Commit, push and deployment remain pending completion
of the full release gates.

### Original glowing presets and steady point-light selectors

The four remaining PRESETS / UPPERPRESETS choices (keys 7, 8, 9 and 0) now
author ceiling glow, floor glow, matching floor/ceiling glow and complementary
floor/ceiling glow. The original formula is `(point mod 5) + 1`, with a zero
base and full-strength `$F000` blend; out-of-sync floors substitute `1000-point`.
The modern authoring point number is the current geometric corner identity,
resolved before separate graphics lighting clones. Assignments store the
resulting native words once, so moving/copying/rebuilding does not re-randomize
the choice. Original out-of-sync numbering above 1,000 is rejected explicitly
rather than inventing a negative-modulo extension. The other presets remain
available in levels with higher current-runtime point numbers.

Source audit resolved selectors 4–15: `Anims.s` reserves twenty zero words at
brightAnimTable and BrightAnimHandler updates only the first three, terminated
by the fourth pointer's -1. No other source writes that table. Original point
decoding indexes its low-nibble selector and blends toward that entry. The C
decoder previously kept the base for selectors above 3; it now blends toward
zero for 4–15. Zone-level brightness encoding is separate and unchanged.
The corner codec retains selectors 1–15; the normal picker names Pulse,
Flicker and Fire, while advanced Original animation selector exposes the
complete 0–15 selection. Steady selectors have an explicit descriptive label.

`feedback_points.c`, now included in the native feedback runner, checks 61,440
point evaluations (15 selectors × 16 strengths × 128 bases × two words), signed
rounding and negative-base bypass. Model coverage adds all four glowing choices
on each storey with exact compiled words, save/reopen, move and clipboard
preservation. A read-only preview query reports native packed and evaluated
point lighting; it does not advance animation or inject values. Browser
acceptance observes the real playtest and then renders fixed inspection cameras
to distinguish native animation from ordinary camera movement.

All ten intended presets are authored. This does not close the full point-light
audit: original raw-word edge cases (including selector-zero words carrying
nonzero high-byte blend bits), unrestricted decorative/special geometry and
other R0–R12 requirements remain open. Unused bump choices remain disabled.

Validation: all 1,123 model tests across 97 files passed. The rebuilt native
point harness passed all 61,440 evaluations. Chromium and Firefox passed all
eight preset applications (four glowing choices × two storeys), source saves,
the advanced selector-15 versus Static-0 exact pixel comparison and actual
playtest animation on both storeys. The existing static-preset, independent
corner-lighting, edit/reload/playtest-isolation and all-sixteen-original-map
browser checks also passed with the corrected decoder. TypeScript, production
build and whitespace checks passed.

The animation test initially assumed toolbar Stop retained runtime lighting.
EnginePreview intentionally reloads the editable artifact on Stop, resetting
that table. The final check observes native values during real gameplay, then
uses the existing play bridge to freeze the simulation while the React playtest
remains mounted. Sixteen fixed cameras near floors/ceilings compare the actual
animated frame before ordinary Stop restores the authored level. Pixel samples
come from renderer_get_cw_buffer before SDL's damage overlay, so a red playtest
tint cannot count as evidence of a lighting change. No animation values or game
timesteps are injected by this check. This checkpoint remains uncommitted and
undeployed while the full release audit is open.

### Reciprocal see-through barriers

AMOS DEFWALL's `b` and `v` choices toggle WT 0↔3 and 0↔4. LEVELSAVE emits
the same opcode-13 graphics for both special types and exports their collision
neighbour as -1. Rendering visibility still crosses those boundaries. Source
version 21 adds `barrier:true` to an explicitly open visibility connection;
both source owners must agree. Earlier versions reject the new field instead
of treating it as an ordinary opening. Stable boundary IDs preserve inserted
corners, copies, captured paste after Undo and skylight partitions.

The compiler now projects barrier edges into a separate collision topology.
Visibility/clip generation retains the connected graph; native collision and
corner walls use the solid projection. Animation attachment validation checks
the actual collision projection rather than assuming it equals visibility.
`authorSeeThroughBarriers` creates full-height lower/upper opcode-13 faces from
explicit saved wall recipes, removes their superseded boundary graphics and
supplies real lighting provenance. Original material density stays continuous
along split boundary chains, and independent paint/repeat/alignment/brightness
continue through the normal boundary pipeline. Controlled volumes remain
protected by their existing mechanism dependency rules.

The renderer previously clamped opcode 13's brightness and disabled clipping,
but still rasterized zero texels as opaque. Its new dedicated path follows
WallRoutine3.ChipMem.s ScreenWallstripdrawthru PACK0/1/2: zero leaves colour,
draw tags and pick provenance untouched. Solid runs alone mark picking. The
path avoids artificial side/cap extension across holes. Opaque wall loops stay
on their existing paths; DOS opaque fast paths are gated only for see-through
faces. The context flag occupies an existing padding byte and is restored after
each wall. This enables the original optional ENABLESEEWALL rendering behavior
for the current port; strict original-engine capability profiles remain open.

The room inspector provides See-through barrier alongside Open/Close edge,
with explicit materials/lighting on both sides, draft cancellation and one Undo.
Open edge removes the barrier, and Close edge makes it solid. An orange dash-dot
map line distinguishes a visible-but-blocked boundary from a dotted passage.
README, in-app controls and the map's connection help explain the workflow.

Model checks cover collision versus visibility, both storeys, strict v21 restore,
reciprocal diagnostics, saved source, Undo/Redo, default and explicit repeat
continuity, captured clipboard and skylight partitions. Browser acceptance
requires both foreground barrier picks and visible/pickable destination geometry,
exact destination colour words through holes, blocked native walking and actual
passage after reopening. Barrier/open colour comparisons keep the same saved
appearance selections; undoing creation also removes those selections and is a
different material/lighting baseline.

This checkpoint does not finish the authoring audit. Imported raw special-wall
provenance and unrestricted reconstruction, other wall bands/curves, remaining
mechanism ownership and the other R0–R12 release requirements remain open.

Validation: the full model run passed 1,126 tests across 98 files. After the
final split-chain continuity adjustment and additional upper-storey/skylight
case, the focused barrier/portal/animation-attachment run passed all 25 checks.
The rebuilt native preview passed sixteen Chromium/Firefox checks covering
barriers, lower/upper wall and boundary paint, repaired doorways, engine edit
reload/playtest isolation and all sixteen original maps. Ordinary wall closure
and reopening also passed in both browsers. TypeScript, production build and
`git diff --check` passed. No commit, push or deployment occurred at this
checkpoint; the full release gates above remain authoritative.

### Incremental deployment requested by the user

The user requested keeping the `abhd-ed` deployment current during ongoing
authoring work. This supersedes waiting for the full audit before publishing
incremental packages; it does not close any unfinished authoring gate.

Built the matching native preview and `npm run build:web` package with runtime
revision `e08a341fc410c6231f753bfed26af951122b5dbd43a72d538b297338fde0d4d4`.
Thirty Chromium/Firefox authoring checks passed against that exact packaged
build using `EDITOR_USE_PACKAGED_BUILD=1`: automatic textures, direct plane
editing, dotted connections, upper storeys/openings, lighting presets,
see-through barriers, navigation, switches, exits and animated water.

Copied the eighty package files into the website checkout's `abhd-ed` folder,
retaining old hashed assets for existing tabs. Verified destination and staged
runtime bytes against the manifest, then committed/pushed only that folder as
website commit `7a4a6d71ea3eab6eb28d423aeb1ac7bb1e1eeb96`. Added deployment-local
Git attributes for binary native payloads. The unrelated `abhd/abhd.zip` was
preserved. Generated Emscripten JavaScript retains its exact compiler bytes;
Git's whitespace checker reports its existing trailing line-ending whitespace.

GitHub Pages run `42081405269` completed successfully. Every one of the eighty
public package files matched its local SHA-256. The live
https://retro-foundry.github.io/abhd-ed/ editor loaded Level 1, direct floor
texture controls and the matching versioned native engine in both Chromium and
Firefox, with rendered pixels and no page errors or engine alerts. README and
requirements now record the incremental deployment workflow. The source
authoring audit remains active; source checkpoint commit/push is still pending.

### Retained full-height native barriers

`importedBarriers.ts` derives native WT3/4-style boundaries from full-height
opcode-13 records covering a solid collision line and an opposing shared native
boundary. Contiguous face intervals may cover the line together. The source
snapshot stays unchanged: no guessed portal overrides, material choices or
lighting words are inserted on import. Raw, untouched levels still retain
their original files through the existing matching-source path.

Room topology and imported visibility now retain sight connections across these
boundaries while the collision projection preserves each side's actual native
neighbour word. One-sided native collision is retained rather than made
reciprocal by inference. Explicit modern Open/Close actions still update both
sides. A retained barrier is not mistaken for a newly opened doorway requiring
a repair recipe during unrelated height/property edits. Native graphics remain
source-owned and keep their material/lighting identities.

Explicit Open trims the old face into the destination air bands; any retained
frame uses the ordinary wall opcode. Generated opcode changes require an
explicit override on that original barrier's source boundary. Explicit Close
uses the normal two-sided repair appearance workflow. Removing the adjoining
room now requests a repair style and replaces the old barrier with solid
graphics. Partial decorative strips do not establish a full-room visibility
connection; unrestricted compound/partial graphics remain an open gate.

Eight model checks cover untouched source import, unrelated property rebuilds,
both storeys' exact native wall fields, legacy visibility, explicit open/close,
portable saves and Undo/Redo, inserted corners, independent group copies,
one-sided collision, decorative-strip exclusion and deletion repairs. The final
model run passed all 1,135 tests across 99 files. Six Chromium/Firefox
checks passed imported-barrier height edits and real hole picking, blocked
walking, opening and Undo alongside authored barriers and ordinary closures.
Instructions now explain automatic recognition and retained native settings.

The closure regression also verifies choosing a texture style whose original
face was see-through. Appearance recipes now supply texture/sampling while the
connection mode owns opacity: normal perimeter/repair walls emit opcode 0 and
explicit barriers emit 13. Close no longer accidentally inherits transparency
from a material prototype. Retained imported graphics keep their native opcode
until the user explicitly changes that connection.

The source checkpoint includes the preceding deployed authoring work and this
import fix. It is an incremental checkpoint, with the R0–R12 audit still open.

Checkpoint delivery: source commit
`27e3ce959ab96e49bb95d370d3a8c2c154562354` was pushed to the game repository's
main branch. The final closure correction passed all 1,135 model tests and
sixteen packaged Chromium/Firefox checks covering imported/authored barriers,
ordinary wall closure, default textures, direct surface editing and storey
openings. Earlier packaged checks also verified all sixteen original maps.

Website commit `5c3c3061273d090e278f73aca7cca6d975fd06c0` publishes
`index-BHtgFhjG.js` with the unchanged matching native runtime `r-e08a341fc410c623`.
GitHub Pages run `42086233780` completed successfully. All eighty public package
files matched their local SHA-256. Both live browsers loaded the tested saved
native-barrier project, recognised its derived barrier after the height edit,
showed direct texture controls and rendered its native preview without page
errors or engine alerts. Historical hashed assets remain available; the
unrelated website archive was preserved. This advances authoring and delivery,
but the current completion audit remains open.


### Imported switch reassignment and removal

Room-source version 22 records retired original switch slots and native settings
provenance on adopted wall switches. Original binary records remain the source
for independent copies and Undo; they are not patched into inactive templates.
The compiler omits retired panels from their original owners, clears every
runtime timer/state/pointer byte before disabling the controller, and can reuse
that condition number for a fresh panel. `Anims.s SwitchRoutine` and the current
`objects.c:switch_routine` timer-before-zone order require that complete clear.
Independent copied controllers retain their own graphics and raw settings.
Copying an already stripped room cannot resurrect an orphan switch panel.

Original-switch settings now expose a destination room and wall, a validated
Move action and requirement-protected removal. Adopted and newly authored
switches can subsequently move between rooms without clearing their mechanism
requirements. The switch number stays fixed, so existing key/switch condition
masks are unchanged. Every Apply builds the whole proposal before committing
one Undo. Mechanism attachment links follow the actual authored/copied switch
room. Static-room eligibility now distinguishes a fresh room from its native
material template and continues protecting copied switch attachments after
retiring their original controller.

The native inspection bridge reports an inactive slot with a null panel phase
instead of treating inspection as an engine failure or reading a stale pointer.
Runtime gameplay remains the existing native switch routine.

Twenty-one model cases include every active switch across all sixteen original
levels, exact opaque settings preservation, original-panel removal, new-slot
reuse, copied templates surviving original retirement, clipboard capture across
Undo, further adopted-switch moves, strict source-version validation and
portable saves. The full model suite passed 1,156 tests across 100 files before
the final copied-attachment guard; focused model checks cover that guard too.
The final proposal passed thirty focused model checks after the attachment and
fresh-template eligibility corrections. The package passed twelve exact
Chromium/Firefox checks for imported-switch move/remove, native switch-gated
door movement, saved source/Undo, copied-switch independence, direct textures,
skylights, upper-floor walking and falling through an opening. The preceding
broader package run passed twenty-four browser checks including authored and
copied switches, reset timers, default textures and retained native barriers.

Delivery package: `index-D1YRsso5.js` with runtime
`r-3a3fddd130e89f0e`. All eighty package files match the staged website blobs,
including the exact native runtime hashes. Source commit
`6b4bb7a5bc838bf0bde371cfa3f7e12238ae0c4c` and website commit
`0a6bdf2431b08c125f27e70a4fc02205e768c118` were pushed to their main branches.
GitHub Pages run `42090280092` completed successfully. All eighty public files
match the tested package by SHA-256. Live Chromium and Firefox each opened the
saved moved-switch and removed-switch projects, showed the authored settings,
loaded the matching versioned native runtime and rendered more than one million
nonblack pixels. Native inspection reported the adopted switch in its new room
and the retired switch as inactive with no panel phase, while its independent
copy remained active. All four live checks had no page errors or engine alerts.
The unrelated website archive and historical assets were preserved.

The complete R0-R12 audit remains open, including native copied-controller
reassignment/removal, door/lift controller groups and the other recorded gaps.


### Copied switch reassignment and removal

Copied switch settings now expose the same validated destination room/wall and
requirement-protected Remove controls as original switches. A moved copy keeps
its switch number, so its door/lift requirements remain connected. Its new
panel appears under Wall switches in the destination room and supports further
wall/room/reset edits. The original and other copies retain their own records.
Every Apply builds the complete proposal before committing one Undo.

Room-source version 23 stores retired copied activation/graphics attachments,
explicit native slot clearing and the moved copy's complete fourteen settings
bytes. A single decoder validates captured bytes for both copied and moved
switches. Native pointers/points are rebuilt, while reset, timer and opaque
state bytes remain independent. Retired slots are cleared before reuse because
SwitchRoutine's timer precedes its active-zone check. Retiring a copy cannot
clear an active original switch. The old panel is removed only from that copy's
graphics owner, and later room copies cannot resurrect it. Separate activation
and graphics owners are retired together. The sixteen shipped maps use matching
owners; a valid explicit fixture using native points and independent native
references covers the separate-owner format contract.

Seven model scenarios cover movement, removal, active timers/opaque state,
slot reuse, saved source/Undo/Redo, geometry edits, captured clipboard pastes
after Undo, repeated moves and subsequent copied-group requirement remapping.
A required copied door remains linked after movement; removal is rejected until
its requirement is cleared. Damaged settings and unsupported source versions
remain explicit errors. Browser checks exercise the actual copied native door,
its moving-wall synchronization, visible panel picking and controller state,
plus removal without disabling the original.

README, in-app help and the parity/requirements audit describe the controls.
The remaining R7 gate concerns door/lift controller reassignment/groups and
engine-profile constraints. The complete R0-R12 objective remains open.


Validation: the final model run passed all 1,163 tests across 101 files. Sixteen
checks passed against the exact packaged editor in Chromium and Firefox,
covering copied/original switch move/remove, native copied-door activation and
moving walls, save/Undo, direct surface textures, skylights, upper-storey
creation and native walking/falling through openings. An earlier fourteen-check
browser run also passed existing copied circuits, clipboard gate remapping and
reset timers. The production build/typecheck and source whitespace checks pass.
All eighty files in `index-Bdn8KV3U.js`'s package match the staged website blobs;
the native runtime remains `r-3a3fddd130e89f0e`.

Source commit `14dc1697e5687908bc8ce70e9e322cb316f1149b` and website commit
`da8dea5039edcdaec7b35002635b4bab64b27469` are pushed. GitHub Pages run
`42093134892` completed successfully. All eighty public package files match
local SHA-256. Live Chromium and Firefox each opened the moved-copy and
removed-copy saved projects, showed the expected source controls and loaded the
matching native runtime. The moved copy activated its own condition bit and
opened its copied door without changing the original switch state. The removed
copy reported zone -1, false state, zero timer and no panel phase while the
original retained its own native record. All four live checks rendered more
than one million nonblack pixels with no page errors or engine alerts.
The historical website assets and unrelated archive were preserved. The broader
completion audit remains active.


### Controller settings transfer and authored reassignment

`DEFDOOR`/`DEFLIFT` in `amiga/archive/amos/leveld.asc:4659/4728` implement the
original </> copy-settings operation. The new workflow transfers selected
supported native settings between original, copied and authored controllers of
the same kind. Trigger rules are selected by default, while destinations and
key/switch requirements are explicit choices. Destination current/direction,
room/graphics/collision attachments and material remain its own. A captured
settings value contains no native pointers or generated identities and remains
independent of later source edits.

The current native 18-byte header is the format authority. Final range,
requirement and endpoint validation happens together before constructing the
command, so a legal combined edit does not pass through an invalid intermediate
state. Each native destination emits only the selected range/rule/mask patches;
copied and authored destinations retain their own source metadata. The UI builds
the complete proposal before applying one Undo, and rejects physical geometry
conflicts without changing the document. Reset transfer draft/Escape discards
pending choices. Native source labels resolve lift file-ID aliases to actual
editable room identities.

Authored doors/lifts can also move to another connected static room. Source
removal and destination assignment are one source transaction; the compiler
regenerates moving bands, plane and collision attachments in their new owners.
The original room returns to its editable static geometry, and the destination
is selected automatically. Existing native/copied ownership is protected.
Cancel move draft/Escape discards the target; Undo/Redo and project archives
preserve the assignment. No new project/source format is needed.

This closes settings transfer and authored reassignment, with instructions in
README and in-app help. Original/copied-controller retirement/reassignment,
synchronized groups, grouped planes and the broader R0-R12 audit remain open.
The original AMOS multi-zone group model cannot be represented by simply
copying the current runtime's one-plane-per-controller record; synchronization
still needs its own compiler/runtime contract and acceptance evidence.


Final validation passed all 1,187 model tests across 102 files. Twenty-four
transfer-focused model cases include the original/copied/authored matrix for
both kinds, exact target identity/state preservation, selected fields,
captured values after source edits, joint range/requirement validation, valid
and invalid authored moves, saves/history and native owner names/settings over
all sixteen original levels. Eighteen exact packaged Chromium/Firefox checks
passed settings transfer and native motion, authored moves and regenerated
attachments, destination selection, draft cancellation, Undo/Redo and saved
projects, plus default room textures, direct surface editing and storey openings.
The preceding sixteen-check browser run also passed existing creation/removal
and copied-controller settings workflows. Typecheck/build and source whitespace
checks pass. The release entry is `index-LDPwhi-m.js`, using the unchanged
`r-3a3fddd130e89f0e` native runtime. All eighty staged website files match their
local package SHA-256. Deployment/public verification follows the checkpoint.


### Paused native controller projection and closed-room inspection

The transfer checkpoint (`3461526c09927b71e867ce64fe73625f66a1a212`, website
`4b9ceb2398e5d34004b32fbf8dbda96a74ea2bc4`, Pages run `42096848944`)
passed public file hashes. Expanded live verification then exposed an existing
paused-preview mismatch: a copied door reported current height 0 while its
room roof remained at -96 until the first play tick. Source archives and native
playback retained the correct settings, but paused inspection was incomplete.

`objects_apply_moving_positions` now projects the normalized native current
positions into live room planes and wall records before the first editor draw.
It reuses the same extracted wall transform as DoorRoutine/LiftRoutine,
including door texture phase and the lift renderer's existing phase ownership.
It does not run triggers, advance positions or timers, change velocity or floor
line event flags, move players, or play audio. Normal simulation applies the
same wall writes in its original per-wall order. The preview host invokes only
the geometry projection on load. Read-only velocity/timer inspection supports
regression proof without advancing a fake initialization tick.

Room visits now provide adjoining views and let the native bridge choose using
live lower/upper air heights. This extends the existing authored closed-door
camera workflow to original/copied doors and controllers paused after play.
The inspector remains selected on the requested room. If no usable view exists,
the camera stays put and the preview explains how to select an open room,
without entering renderer failure state. Visiting valid air restores normal
status. Object/start and explicit capture requests retain their own camera
contracts. README and in-app help describe paused poses and closed-room visits.

Twenty native/browser checks first passed pose initialization, unchanged
velocity/timers for every controller in all sixteen maps, movement cycles and
wall texture/brightness behavior. A broader thirty-check run passed that native
runtime with closed-room visits and the existing room/texture/opening scenarios.
The final exact package then passed ten focused Chromium/Firefox checks with
stronger adjoining-view coordinates, no-air status recovery, original/copied/
authored paused-state checks, transfers, moves and native motion. The model
suite remains the validated 1,187 tests; model authoring code did not change in
this native projection correction.

Corrected package: `index-Cp8vFwv4.js` with runtime
`r-969287ddd046f5e9` (full revision
`969287ddd046f5e9a33770befe19211c1a632b20fdc3220f97c293f476e1c264`).
The full R0-R12 completion audit remains open.
