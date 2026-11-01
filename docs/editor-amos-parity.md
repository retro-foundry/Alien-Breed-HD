# AMOS editing parity: minimum release requirement

The user requires at least the editing capabilities of the original AMOS
editor. A modern replacement workflow counts only when it can produce the
equivalent editable result, preserve it through saves/Undo, and run the exported
result in the game. Published room drawing alone does not establish parity.
Automatic winding, portal pairing and compilation remain editor responsibilities.

The initial authoritative inventory is the complete `BUT$(0)` through `BUT$(37)`
menu in `amiga/archive/amos/leveld.asc:280`. Its dispatch starts at line 3738.
The tokenized `leved.amos`, project banks, dialogs, keyboard modifiers and export
routines also need coverage; this menu is a baseline, not proof that the audit
has found every original capability. Historical TKG editor descriptions do not
replace the AB3D I source as the format/behaviour authority.

## Coverage against the original menu

“Implemented” below is scoped to the named operation, not the complete editor.
“Partial” and “Missing” are release gaps, even where imported bytes round-trip.

| ID | AMOS operation | Current coverage / remaining requirement |
| --- | --- | --- |
| 0 | Add Point | Partial: drawing/redrawing creates corners; independent point insertion and shared-edge splitting still need the complete workflow |
| 1 | Define New Zone | Implemented: draw/cancel/commit editable rooms, choose appearance, compile/play/save |
| 2 | Add Object | Partial: current-level templates; the original complete object library and all parameters remain required |
| 3 | Delete Point | Partial: redraw can remove corners; direct safe vertex removal/shared-edge repair remains required |
| 4 | Toggle Wall/Delete Zone | Partial: undoable room deletion and generated portal repairs; explicit open/close of existing wall edges remains required |
| 5 | Move Object | Implemented: map drag and numerical position, native floor/room validation and Undo |
| 6 | Move Point | Implemented for room corners/shared junctions; independent decorative/curve/control geometry remains a gap |
| 7 | Define Viewing Order | Partial: automatic rebuilding/inspection and original-view comparisons; remaining independent occlusion scenarios and full editing equivalence need proof |
| 8 | Delete Object | Implemented: individual/group deletion, native records, save and Undo |
| 9 | Upper Roof Height | Implemented: numerical/graphical height, copy, preview and native upper-storey checks |
| 10 | Upper Roof Point Brightness | Partial: point-light editing exists; exact four-channel AMOS authoring equivalence still needs verification |
| 11 | Set Player Start / End Zone | Partial: authored starts and exit settings exist; complete new-room/end-zone workflow remains subject to its native/source acceptance cases |
| 12 | Upper Floor Height | Implemented: independent upper floor, relationship validation and preview |
| 13 | Upper Floor Point Brightness | Partial: as ID 10 |
| 14 | Define Door Zone | Partial: imported/copy settings and linked graphics work; create/remove/reassign an independent door on arbitrary valid new geometry remains required |
| 15 | Define Roof Height | Implemented: lower-storey ceiling numerical/graphical editing and native checks |
| 16 | Lower Roof Point Brightness | Partial: as ID 10 |
| 17 | Define Lift Zone | Partial: imported/copy settings and native motion work; independent creation/removal/reassignment and full controller grouping remain required |
| 18 | Define Floor Height | Implemented: lower floor editing, stairs and native standing/crouched collision checks |
| 19 | Lower Floor Point Brightness | Partial: as ID 10 |
| 20 | Define Teleporter | Implemented source links/destinations and native teleport checks; retains explicit invalid-link diagnostics |
| 21 | Define Roof Graphic | Partial: all 16 normal patterns and original repeat choices implemented; bump mode and no-roof/sky authoring remain required |
| 22 | Define Wall Graphic | Partial: independent straight face/boundary paint; complete alignment/repeat, transparency, bands, curves and other original options remain required |
| 23 | Define Floor Graphic | Partial: all 16 normal patterns and original repeat choices implemented; bump-mode assignment remains required |
| 24 | UPPER Roof Graphic | Partial: independent upper normal patterns/repeat implemented; bump and sky remain required |
| 25 | UPPER Wall Graphic | Partial: same requirements as ID 22 on upper walls |
| 26 | UPPER Floor Graphic | Partial: independent upper normal patterns/repeat implemented; bump remains required |
| 27 | Water Height | Implemented: independent source height and native plane checks |
| 28 | Add Control Point | Missing: editable independent navigation points |
| 29 | Link Zone to Cpt | Missing: graphical lower-storey navigation assignment and validation |
| 30 | Water Height Anim | Partial: imported/copied animation settings and actual movement; independent authoring/group assignment remains required |
| 31 | Link Control Points | Missing: editable navigation graph links with native constraints |
| 32 | Link Upper Zone to Cpt | Missing: independent upper-storey navigation assignment |
| 33 | Upper Wall Bright | Partial: room/point lighting exists; independent original per-wall brightness assignment remains required |
| 34 | Lower Wall Bright | Partial: as ID 33 on lower walls |
| 35 | Move Control Point | Missing: transactional map/numerical movement and dependent-link updates |
| 36 | Set Background SFX for zone | Missing: original ambient sound masks. Existing floor footstep-sound settings are a different field |
| 37 | Set Echo Level of zone | Missing: explicit echo authoring with native audio verification |

## Source contracts for the next work

- `DEFFLOORGRAPH` / `DEFROOFGRAPH` at 4936/4964 and upper variants at
  5125/5153 assign per-zone patterns and repeat. `FGPUT` / `RGPUT` at 5219/5186
  select all 16 normal patterns, repeat shifts -5 through 2, bump mode and a
  seventeenth no-roof choice. The compiler writes tile offset
  `(pattern / 4) * 256 + (pattern mod 4)`. Normal patterns and repeat are the
  active implementation; bump and no-roof remain explicit gaps.
- `DEFDOOR` / `DEFLIFT` at 4659/4728 assign controller groups and copy settings.
  Existing controller copies do not replace independent group creation.
- Control-point procedures and `CPTNEARTOZONE` / `CPTNEARTOUPPERZONE` retain
  distinct navigation semantics; rendering visibility is a different system.
- Zone output writes `BSFX` at byte 28 and echo at byte 37. Footstep sounds,
  backdrop and ambient effects must remain distinct source fields.

The parity gate stays open until every gap has a complete workflow, appropriate
original-source evidence, save/Undo tests and real engine verification. Further
modernisation and existing R1–R12 requirements remain part of the full goal.
