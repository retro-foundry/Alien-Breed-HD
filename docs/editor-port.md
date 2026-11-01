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
