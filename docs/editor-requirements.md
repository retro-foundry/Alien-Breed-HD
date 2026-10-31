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

## Current state

The binary model, original-file preservation, supported record edits, recovery
and basic React UI are implemented and tested. They cover parts of R4, R5, R7,
R9 and R11. The geometry normaliser has unit/source-fixture coverage for part of
R2, including all 2,496 original outlines, but still needs authoring/compiler
integration. Undo now keeps the full action chain and has a 250-action
undo/redo regression.

The actual game engine is now embedded beside the map, with zone visits,
lighting-edit reloads and isolated playtests. All 16 maps load and draw in both
Chromium and Firefox. Full wall/floor/object picking and overlays are pending.
Original assembly geometry helpers have 330 emulated parity fixtures, and all
49,797 current ordering masks independently rebuild. The exact portal model
processes all 16 maps, including concave imports, emits common clip bounds and
reproduces original solid-wall links. Generated-file and engine comparisons
now cover deterministic assembly and unchanged engine spawn views in both
browsers. Broader scenes, geometry edits and source/history integration are
still required before enabling automatic rebuilt visibility in exports.

The full compiler, topology tools, geometry validation UI, PVS generation and
inspection, heights/stairs, asset pipeline, multiselect/clipboard, full 3D
picking/overlays, full mechanism authoring and export profiles remain required.
Do not mark this goal complete based only on the existing format/model/UI tests.
