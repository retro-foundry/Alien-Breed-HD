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
| 0 | Add Point | Partial: direct outline-edge insertion, shared-edge splitting, drawing/redrawing and whole-coordinate lattice snapping are implemented; unattached/decorative point creation still needs the complete workflow |
| 1 | Define New Zone | Implemented: draw/cancel/commit editable rooms, choose appearance, compile/play/save |
| 2 | Add Object | Partial: deduplicated named variants from all 16 maps, click-to-place with floor matching and Undo, click-to-duplicate with custom settings, and direct named in-place types; the original complete object library and all parameters remain required |
| 3 | Delete Point | Partial: direct safe outline-corner removal updates the incident room fan, validates geometry and protects starts/objects/teleports; control-point deletion is implemented; independent decorative geometry deletion remains required |
| 4 | Toggle Wall/Delete Zone | Partial: undoable deletion, portal repair and open/close/automatic connections include authored and retained full-height native see-through barriers, solid collision versus visible neighbours, save/copy/Undo and native hole picking; compound/partial special-boundary reconstruction remains part of unrestricted editing |
| 5 | Move Object | Implemented: map drag and numerical position, native floor/room validation and Undo |
| 6 | Move Point | Implemented for room corners/shared junctions and control points; independent decorative/curve geometry remains a gap |
| 7 | Define Viewing Order | Partial: automatic rebuilding/inspection and original-view comparisons; remaining independent occlusion scenarios and full editing equivalence need proof |
| 8 | Delete Object | Implemented: individual/group deletion, native records, save and Undo |
| 9 | Upper Roof Height | Implemented: numerical/graphical height, copy, preview and native upper-storey checks |
| 10 | Upper Roof Point Brightness | Partial: four independent normal floor/ceiling corner channels, all ten whole-room presets, selectors 1–15 and blend strengths have saved source, copy/move/Undo and native shading workflows; unrestricted decorative/special geometry lighting and original raw-word edge cases remain to be audited |
| 11 | Set Player Start / End Zone | Implemented: authored starts and original/new-room exits with stable source references, save/Undo/export and native completion from every section of a concave exit room; disabled and original-slot exits verified in both browsers |
| 12 | Upper Floor Height | Implemented: independent upper floor, relationship validation and preview |
| 13 | Upper Floor Point Brightness | Partial: as ID 10 |
| 14 | Define Door Zone | Partial: independent eligible original/new connected convex lower doors can be created/removed, start closed, save/Undo/copy and run natively; controller reassignment and grouped/multi-plane geometry remain required |
| 15 | Define Roof Height | Implemented: lower-storey ceiling numerical/graphical editing and native checks |
| 16 | Lower Roof Point Brightness | Partial: as ID 10 |
| 17 | Define Lift Zone | Partial: independent eligible original/new connected convex lower lifts can be created/removed, save/Undo/copy and run natively; controller reassignment and full grouping remain required |
| 18 | Define Floor Height | Implemented: lower floor editing, stairs and native standing/crouched collision checks |
| 19 | Lower Floor Point Brightness | Partial: as ID 10 |
| 20 | Define Teleporter | Implemented source links/destinations and native teleport checks; retains explicit invalid-link diagnostics |
| 21 | Define Roof Graphic | Implemented for normal patterns/repeat, top ceilings open to sky and rectangular skylights; unused bump choices are disabled at the user's request while imported modes remain preserved |
| 22 | Define Wall Graphic | Partial: independent straight face/boundary paint, tile/row alignment, signed brightness, moving-wall base synchronization and original repeat-window/fit; authored and retained full-height native see-through barriers preserve material/light provenance with native hole picking and collision; bands, curves, compound special-boundary reconstruction and other original options remain required |
| 23 | Define Floor Graphic | Implemented: normal patterns with original repeat choices and direct inspector editing; imported bump modes are preserved, with new bump choices disabled as requested |
| 24 | UPPER Roof Graphic | Implemented for independent upper patterns/repeat, sky ceilings and playable inter-storey openings; unused bump choices remain disabled as requested |
| 25 | UPPER Wall Graphic | Partial: same requirements as ID 22 on upper walls |
| 26 | UPPER Floor Graphic | Implemented: independent upper floor patterns/repeat and direct inspector editing; new bump choices stay disabled as requested |
| 27 | Water Height | Implemented: independent source height and native plane checks |
| 28 | Add Control Point | Implemented: lower/upper diagram placement, numerical positions, room containment, midpoint heights, native 100-point limit, portable source and rebuilt game tables |
| 29 | Link Zone to Cpt | Implemented: independent lower assignments, nearest visible point and atomic all-room automatic assignment, clear diagnostics and native control bytes |
| 30 | Water Height Anim | Implemented: independent static water and 21 shared animation groups, range/starting height/direction, pause/resume, existing-pool preservation or whole-room coverage, room assignment/reassignment/removal, copy/save/Undo, concave attachments and native polygon/collision movement |
| 31 | Link Control Points | Implemented: original physical/one-way/visual pairs, explicit graph import/create, automatic visual linking preserving physical pairs, filtered overlays, raw twolev.links download and source/Undo persistence |
| 32 | Link Upper Zone to Cpt | Implemented: independent upper assignment and storey validation without changing lower references |
| 33 | Upper Wall Bright | Partial: independent straight-face/boundary brightness offsets are authored separately per storey, saved/copied/undone and verified in native rendering; curve/special-face and full original WB/ZZ field coverage still need completion |
| 34 | Lower Wall Bright | Partial: as ID 33 on lower walls |
| 35 | Move Control Point | Implemented: diagram dragging/cancellation, exact coordinates and storey/room transfer, dependent room moves/copies/skylight partitions, link/assignment compaction on deletion and Undo |
| 36 | Set Background SFX for zone | Missing: original ambient sound masks. Existing floor footstep-sound settings are a different field |
| 37 | Set Echo Level of zone | Missing: explicit echo authoring with native audio verification |

## Source contracts for the next work

- `DEFFLOORGRAPH` / `DEFROOFGRAPH` at 4936/4964 and upper variants at
  5125/5153 assign per-zone patterns and repeat. `FGPUT` / `RGPUT` at 5219/5186
  select all 16 normal patterns, repeat shifts -5 through 2, bump mode and a
  seventeenth no-roof choice. The compiler writes tile offset
  `(pattern / 4) * 256 + (pattern mod 4)`. Normal, chunky bump and smooth bump banks and repeat are implemented.
  No-roof/sky authoring now has saved source controls and native rendering/movement verification. Native opcodes 8/9 and 10/11
  preserve the floor/ceiling role; the smooth bank uses one fixed texture.
- `DEFDOOR` / `DEFLIFT` at 4659/4728 assign controller groups and copy settings.
  Existing controller copies do not replace independent group creation.
- `DEFSWITCH` at 3905 selects one of eight slots, assigns a wall and toggles
  automatic reset. Independent lower-wall switches now have a numbered room
  diagram, creation/reassignment/reset/removal controls, source/Undo persistence,
  independent clipboard slots and copied internal requirement remapping.
  `LEVELSAVE`'s panel and activation-pair geometry (901, 1413/1445, 1962) are
  compiled into the current native format; real SwitchRoutine condition bits,
  panel picking and switch-gated door movement are verified in both browsers.
  Reassignment/removal of imported switch controllers remains a separate gap.
- Control-point procedures and `CPTNEARTOZONE` / `CPTNEARTOUPPERZONE` retain
  distinct navigation semantics; rendering visibility is a different system.
- Zone output writes `BSFX` at byte 28 and echo at byte 37. Footstep sounds,
  backdrop and ambient effects must remain distinct source fields.

- `DEFWALL` at 7320 toggles ordinary WT 0/1 boundaries and excludes controlled
  wall types. The current editor now records reciprocal overrides on stable
  source boundary identities, rebuilds actual collision/wall/visibility tables,
  and verifies closing/reopening in the native engine. The `b`/`v` toggles write
  WT 3/4; both LEVELSAVE branches emit opcode 13 and a solid -1 collision
  neighbour while retaining visibility. The modern See-through barrier action
  authors that common result reciprocally. The native renderer now follows
  ScreenWallstripdrawthru's zero-texel skip (originally behind ENABLESEEWALL).
  Full-height raw barriers now derive their visibility and original collision
  directly from native wall/line records and opposing source boundaries; source
  edits preserve them without requiring invented appearance selections. Partial
  decorative strips and compound special geometry remain separate work.

The parity gate stays open until every gap has a complete workflow, appropriate
original-source evidence, save/Undo tests and real engine verification. Further
modernisation and existing R1–R12 requirements remain part of the full goal.
