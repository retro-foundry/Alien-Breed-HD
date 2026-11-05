# Editor modernisation requirements

This is the acceptance checklist supplied by the user, alongside the original
goal to port the AMOS editor directly to React for browsers on Windows and
Linux. Drawing direction, convexity and portal orientation are editor
responsibilities. The user must not need to reason about clockwise versus
anticlockwise geometry.

The historical examples concern the Alien Breed 3D II / TKG editor and later
LevelEd 3. They provide regression scenarios, not evidence that AB3D I uses
TKG's formats or that every historical defect survives in later editors.
The compatibility authority here remains AB3D I's actual source and files.
Karl Churchill's [Software that Shouldn't Exist (Part 2)](https://www.linkedin.com/pulse/software-shouldnt-exist-part-2-karl-churchill-fasae)
discusses that editor history and renderer/PVS problems. The user's supplied
LevelEditor v3.03 changelog reference still needs to be located and checked.

## Required results and evidence

| ID | Priority | Required result | Proof needed before completion |
| --- | --- | --- | --- |
| R0 | P0 | At least the original AMOS editor's editing capabilities, with modern equivalent workflows and no user-managed winding | Close every original operation and option in `editor-amos-parity.md`, including save/Undo and actual engine proof; published drawing tools alone do not meet this gate |
| R1 | P0 | Transactional zone creation/edit/delete; changing tools cannot commit a partial room; neighbouring walls and shared edges stay valid | Cancel/switch-tool/point-limit/delete scenarios with undo and actual exported game geometry |
| R2 | P0 | Automatic winding and portal orientation, convex runtime zones, automatic concave decomposition; duplicate/intersection/acute-corner/disconnection diagnostics | Both gesture directions and rotations produce equivalent playable rooms; partial-edge joins and malformed outlines tested in UI and compiler |
| R3 | P0 | Correct PVS and clipping generation; overlays and inspection; stale-data detection; full rebuilding after geometry changes | Generated visibility compared with original contracts and independent ray/portal scenarios; closed-door, corridor and occluded-room engine runs; runtime fixes when required |
| R4 | P0 | Reliable atomic project saves and builds; editable source separate from generated game files; correct target level slot; clear compiler errors; one-click Build & Play | Cancelled dialogs, failed storage, wrong-slot attempts and failed builds retain source; exported map starts in the actual engine |
| R5 | P0 | Original-level round trips and compiler rebuild compatibility | All original imports retain relationships, materials, objects, mechanisms and heights; untouched byte round trips **and** independently rebuilt data/engine comparisons. Byte passthrough alone does not prove compiler fidelity |
| R6 | P1 | Numerical and graphical height editing, height copy, valid floor/ceiling relationships, automatic stairs, clear upper/lower editing and simultaneous 3D preview | Interactive height/stair scenarios, immediate validation, correct runtime collision and rendered height/texture evidence |
| R7 | P1 | Safe doors/lifts/switches/teleports with destination links, highlighting and immediate validation; original engine restrictions enforced in legacy export | Conflicting mechanisms, duplicate lift assignment, neighbouring moving walls, speed limits and invalid/blocked destinations tested against the selected engine profile |
| R8 | P1 | Visual material browser, PNG import and automatic original-format conversion with correct palettes/sampling | Real asset conversions compared with source routines and actual engine pixels; missing/bad resources produce useful diagnostics |
| R9 | P1 | Conventional select/multiselect, copy/paste, duplicate, align, snapping, anchored pan/zoom, search, shortcuts and unlimited undo | Multi-item atomic edits, clipboard workflows, drag cancellation and undo beyond 100 actions; no hard undo cap; keyboard and responsive UI runs |
| R10 | P1 | Interactive simultaneous 3D viewport with wall/floor/object picking and height, light, texture and portal overlays | Actual engine rendering/picking drives matching editor selections and edits; upper/lower and portal scenarios verified |
| R11 | P1 | Unified project configuration, asset browser and dependency checking for level files, materials, objects, animations, enemies, weapons and sounds | A portable project opens independently of Amiga assigns/manual game-link setup; missing resources warn without preventing startup or losing source |
| R12 | P2 | Expose supported port rendering/material/resource capabilities; strict original-engine export profile | Profile limits tied to actual engine/source declarations, compatible exports accepted, incompatible extensions rejected before writing files |

## Current completion audit (2026-11-05)

The full R0-R12 objective remains active. The table below supersedes stale
feature-status statements in the historical notes that follow. Individual
implemented workflows do not close the broader release gates.

| Gate | Current evidence | Remaining evidence or implementation |
| --- | --- | --- |
| R0 | Original 38-operation menu inventory; source drawing, texture/height/storey/sky authoring, native wall connections and control-point/graph authoring | Every Partial/Missing entry in editor-amos-parity.md, remaining original options and full save/Undo/native acceptance |
| R1 | Transactional draw/redraw/corners/delete, portal repair, explicit reciprocal open/close, cancelled drafts and native wall collision | Unattached/decorative geometry and all topology/controller dependency cases |
| R2 | Automatic winding, shared/partial edges, original outline corpus, convex decomposition and explicit diagnostic paths | Full unrestricted decoration/curve reassignment and the complete original topology acceptance corpus |
| R3 | Rebuilt original ordering/clip fixtures, map overlays and compiled original-view checks | Independent closed-door/corridor/occluded-room visibility proof over the required scope |
| R4 | Source-preserving project archives/recovery, shared preview/export compiler, automatic saved starter room textures, playtest and stable new/concave-room exits | Complete authoring integration, all failure cases and final coherent published package |
| R5 | Original untouched round trips and compiled original-source/native comparisons | Full relationships/material/mechanism parity across the remaining authoring cases |
| R6 | Numerical/graphical planes, stairs, upper-storey creation/removal, sky and playable inter-storey shafts | Synchronized editing of controlled geometry, grouped moving planes and remaining original height options |
| R7 | Static/new door/lift creation and authored reassignment, same-kind settings transfer across original/copied/authored controllers, ranges/rules, copied controllers/switches, independent lower-wall switch creation/reassignment/reset/removal, imported/copied switch reassignment/removal, teleports, independent shared water groups and native movement | Original/copied door/lift controller reassignment, synchronized groups and complete engine-profile constraints |
| R8 | Wall PNG conversion and source retention, native material browser, direct floor/ceiling patterns, straight-wall alignment/repeat/brightness and native controller-base synchronization; authored and retained full-height native barriers with hole picking and separate collision/visibility | Bands/curves, compound/partial special-boundary reconstruction and remaining special-face lighting, new palettes and floor/other resource conversions |
| R9 | Unlimited history regression, map/3D selection, object groups, room groups and same-workspace clipboards | Remaining transforms, cross-project resources and complete keyboard/responsive coverage |
| R10 | Editable 3D plus native engine, wall/surface/object picking, four height handles, independent normal floor/ceiling corner lighting, all ten original room presets and native point selectors 1–15, portal overlays and both-storey control-point/link overlays | Original raw-lighting-word edge cases, curve/special-surface lighting and geometry editing, exact remaining overlays and complete upper/lower acceptance |
| R11 | Portable levels, wall assets/PNG sources and named cross-level object library | Unified configuration/dependency browser for remaining resource classes and complete object/enemy parameters |
| R12 | Shared current-runtime graphics limit and explicit port setup files | Capability declarations, strict original-engine export profile, all original limits and incompatible-extension rejection |

User-requested unused bump options must remain disabled. Preserving imported
bump records is required, but re-enabling those options is not a completion gate.

Final delivery remains: finish the open gates; update user instructions and
requirements evidence; commit and push the relevant source; build and validate
a coherent runtime/editor package; deploy it to
`retro-foundry.github.io/abhd-ed`; verify the public artifact and required
browser/native scenarios. Keep unrelated ongoing DOS changes separate from
editor commits. Do not publish the current partial state as full completion.

Deployment cadence update: the user explicitly requested keeping
`retro-foundry.github.io/abhd-ed` current while authoring work continues.
Publish verified incremental editor/runtime packages without waiting for every
release gate to close. Record package validation and public artifact checks;
the deployments do not establish full authoring completion.

## Historical implementation notes

R8/R10 now include individual material painting on picked imported/copied
straight native faces and new/repaired authored boundaries, with previews,
Cancel/Apply/Reset, source-persistent lower/upper provenance and Undo. Sampling
compatibility is checked before native preview; switch faces remain protected.
Boundary IDs survive normalization, T-junction splits and corner/room moves.
Curves, floor/ceiling painting and graphical editing handles remain open. See
`editor-surface-picking.md` for the supported scope and remaining limits.

The binary model, original-file preservation, supported record edits, recovery
and basic React UI are implemented and tested. They cover parts of R4, R5, R7,
R9 and R11. The geometry normaliser has unit/source-fixture coverage for part of
R2, including all 2,496 original outlines. Source drawing, corner dragging and
coordinate editing, shared T-junction moves, deletion, cancellation, overlap
validation and source saves are now wired into the React canvas/inspector.
An atomic source-build API now assembles complete runtime tables and validates
all sixteen original maps, including decomposed-room draw identities. Connecting
it to normal source editing is now implemented for supported geometry, authored
starts/objects/teleports and explicit new-room appearance. Preview and export
share the validated artifact, retain editable source, and map generated room
IDs back to editor selections. Static new-room joins generate their openings,
and imported doorway closures use explicit saved material/lighting selections.
Static imported portals can also reconnect to replacement authored rooms using
the saved repair appearance and destination height bands. Redraw outline now
changes an imported room's corner count transactionally; new boundaries use
explicit selected wall styles and initial point lighting. Controller groups and
partial decorative geometry assignment remain required for
unrestricted geometry authoring. Broader coverage is still needed
before claiming the complete original editor workflow.
Undo keeps the full action chain and has a 250-action
undo/redo regression.
R9 now includes object multiselect in the map/list and atomic group translation,
alignment and deletion, with invalid destinations rejecting the whole edit.
Browser/native checks cover gestures, source round trips and group undo. Room
groups now support atomic movement and a validated same-project placement
clipboard with attached-room capture. Broader transforms and cross-project
clipboard workflows remain open. Single objects and
groups now support a session clipboard within the same workspace, preserving
captured settings after original edits, deletion or undo; paste validates the
whole batch and creates independent records as one undoable action. Browser
checks cover save/reopen records and actual native object types/heights. Object groups now also support
snapped/cancellable map dragging, atomic duplication, selection framing,
keyboard deletion, scoped Select All and rectangle selection in either direction.
New enemy/pickup/scenery records can now be created from current-level templates
with independent native positions, floor matching, capacity validation and atomic
undo. Named key, ammunition and weapon pickup settings update related native
artwork and behaviour fields together. Actual browser gameplay verifies key
conditions, ammunition grants and weapon ownership. A cross-level object library,
broader enemy configuration and the other resource classes remain open.
Paused 3D inspection now shares editor shortcut handling with the map/list,
including history, save/open, clipboard, duplication, Delete, help and search.
Modal/text/playtest isolation and delayed-build Undo/save have native-browser
coverage. Frame-dependent actions reject stale document revisions; inspection
Select All follows the editing floor without inheriting hidden map markers.
The current port's per-room/storey graphics command budget is now shared with
the native renderer and enforced for preview/build/game export, while editable
oversized files remain saveable. This covers one concrete R12 resource limit;
the broader capability model and original-engine export profile remain open.
Numerical lower/upper floor and ceiling editing, height copy, reset, validation
and undo now use per-room source overrides. Native header/surface/wall rebuilds
and browser gameplay verify quarter-unit floor standing and a step transition;
upper-storey view changes are checked in the actual engine. This advances R6;
straight stair creation now generates adjoining native rooms as one undoable
source edit, with a labelled plan, actual material choices, shared climb-limit
validation and tested walking in both directions. A section diagram now adds
graphical floor/ceiling handles, quarter-unit keyboard adjustments, provisional
validation, cancellation and one-edit drag commits alongside the actual engine
preview. Exported collision and native player standing heights are checked for
dragged floors. Animated-height authoring is still required.

R8 now has visual lower/upper wall style selection in room and stair forms,
using packaged native texture sheets, original palette levels, search and
explicit draft/apply behavior. Every decoded sheet matches independent native
C dump pixels. Errors and retries preserve source.
Native wall overrides now persist independently of original levels, pass
through compiled preview/export and restore on undo, reopening and recovery.
Raw/SB WAD import is available in advanced material tools. This establishes the
wall-asset path for R8/R11. PNG wall import now resizes automatically, matches
the existing native palette, previews before applying and preserves original
PNGs in v10 project backups/recovery. Native pixel packing and actual renderer
changes are checked. New-palette authoring, broader material assignment,
floor/other resource conversions and the other resource classes remain open.

The actual game engine is now embedded beside the map, with zone visits,
lighting-edit reloads and isolated playtests. All 16 maps load and draw in both
Chromium and Firefox. Native pixel picking now selects the room at the clicked
position, retains the inspection camera and maps generated cells back to source
rooms. Enter selects at the crosshair. Authored stair views, native upper-floor
views with remapped IDs, viewport resizing and later list navigation are checked
in both browsers. Native graphics provenance now additionally identifies picked
walls, floors, ceilings and water, with stream/material/authored-height readouts.
Actual colour-word comparisons verify opaque sprite/polygon-object coverage and
transparent gaps. Visible sprite and polygon objects now select their own
authored record in the object inspector, with native record/CID validation and
camera preservation. Browser checks cover lower/upper objects, independent
height edits, Undo and selecting through transparent gaps. Surface inspector
edits still apply to the room; stable individual surface editing and graphical
edit overlays are pending.
Original assembly geometry helpers have 330 emulated parity fixtures, and all
49,797 current ordering masks independently rebuild. The exact portal model
processes all 16 maps, including concave imports, emits common clip bounds and
reproduces original solid-wall links. Generated-file and engine comparisons
now cover deterministic assembly and unchanged engine spawn views in both
browsers. Source history and project archives are integrated. Source build
integration now includes rebuilt visibility in normal supported-geometry
preview/export. Broader occlusion/gameplay scenes and unrestricted authored
geometry still need verification.
The map now offers visibility/portal overlays and a per-runtime-cell table
inspector backed by the same artifact as preview/export. Concave source rooms
expose their compiled cells separately. Native zone/clip/mask records are shown
with a conservative-visibility explanation. Source edits invalidate old results;
failed builds clear overlays and Undo restores valid inspection. This advances
R3 inspection, while broader closed-door and occlusion validation remain open.

The remaining compiler/authoring integration, topology/diagnostic tools, PVS
inspection, heights/stairs, asset pipeline, multiselect/clipboard, full 3D
picking/overlays, full mechanism authoring and export profiles remain required.
Do not mark this goal complete based only on the existing format/model/UI tests.
