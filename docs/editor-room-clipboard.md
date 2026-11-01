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
- `wallGeometry.ts` originally branched on `originZone === null`, repeating the
  selected new-room wall materials on each perimeter edge. The native path
  preserves individual wall records, sampling, phase, decorative fragments,
  heights and stream identities. Copying an imported room needs that provenance
  without also borrowing its original runtime identity. `roomMaterials.ts` now
  makes this decision from the retained native wall provenance separately.
- `cellGraphics.ts` similarly inserts new-room walls as a batch, but preserves
  imported wall records at their original draw positions. Arc and LightDraw
  point references are translated only for an established whole-room translation.
  Ordered water/object/clip/backdrop records must stay in their native order.
- `portalWalls.ts` distinguishes original openings from new or closed openings
  using material provenance and original/template correspondence. A copy needs
  explicit correspondence between
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

## Material compiler checkpoint

The wall, stream-order, topology/portal, appearance and lighting stages now
distinguish inherited native walls from a new perimeter-material selection.
Fresh room identities can retain their wall records, native closed boundaries,
paired openings, ordered arc/beam graphics and point-lighting provenance.
One-sided inherited openings still require an explicit closure appearance.
Source saves use the existing validated wall-source fields; no generated tables
or copied original zone identities are written into the editable baseline.

All 720 model tests pass. Tests compare translated interior decorations and
ordered arc/beam streams with and without original runtime identities; a static
native-material copy compiles through project save/reopen. Both Chromium/Firefox
native-render cases pass, inspecting four deterministic camera directions, as
do 16 adjacent portal/group/sprite cases. This is compiler acceptance and
record-preservation evidence, not complete original/copy frame parity for every
content class. Controller/switch copies are explicitly rejected before build
until independent authoring exists. Full clipboard assembly/UI, arbitrary
partial-surface reshaping and remaining relationship/profile work stay open.

Exact whole-room translations now preserve partial floor/roof/water polygon
footprints, point order, closure references, materials, sampling and native
lighting roles, including interior points. Translation detection tolerates
collinear portal splits without removing editable wall identities. Whole native
draw volumes retain their polygon records independently of convex collision
partitioning; arbitrary shape changes still require explicit footprint mapping.
The 723-test full model run passes, plus the subsequent partial-water/translation
focused checks. All 20 focused Chromium/Firefox native-copy, room-group, height
and sprite cases pass. These checks extend compiler support; they do not expose
or establish completion of the full room clipboard operation.

Room-source version 4 now supports explicit native property snapshots: water,
packed lower/upper brightness, control bytes, sky and floor sounds. Template
edits cannot change a room's captured values. Save/reopen, geometry/height edits,
native header assembly and Undo retain the snapshot; removing it explicitly
restores template inheritance. This addresses property ownership needed by room
copy assembly. Shared material resources remain separate dependencies.

All 727 model tests and four native Chromium/Firefox property-probe cases pass,
including static/animated words and later template changes. The native probe is
read-only and returns authored packed words, not evaluated animation intensity.
The lighting inspector now offers independent brightness controls only for the
native bump/smooth surface paths that consume zone brightness. Point-shaded
walls/floors and native water retain their own lighting contracts. A valid bump
material fixture viewed from a bounded inspection eye height proves actual
pixel changes and restoration through Undo/Redo and inheritance reset in both
browsers. All 40 focused controls/editor/property/height cases pass. The
inspection-height capture API changes only the paused camera and rejects heights
outside the room's air band. Full room clipboard authoring remains incomplete.

## Shared water animation attachments

Native-material copies now receive their own generated polygon pointers in the
source water animation group. `amiga/Anims.s` `DoWaterAnims` and
`src/objects.c` `do_water_anims` advance one group's 24.8 height and write every
listed room's water plane plus its polygon's quarter-unit height word. Copying
that relationship preserves the original group header, range, velocity and phase;
it does not allocate an invented independent oscillator. Original runtime room
identities remain unique. Door/lift/switch copies retain their explicit guards.

The original levels do not populate an oscillating group, so verification uses
an explicitly constructed native-format group with an original water polygon.
Save/reopen and source compilation prove distinct graphics pointers, translated
polygon coordinates and unchanged group headers/source bytes. The engine probe
reads actual polygon and zone water writes in one browser callback; both browsers
prove multiple changing heights stay equal between original and copy. The probe
only inspects loaded buffers and rejects invalid polygon offsets.

Native water entries can legally control a room different from the referenced
polygon owner. Existing cross-room relationships are preserved; copying that
polygon without an explicit room relationship map is rejected before export.
Independent water authoring, general cross-room mapping, complete clipboard
assembly and the remaining R7/R9/R11 requirements are still open.

## Moving-controller dependency inspection

`mechanismDependencies.ts` now resolves the controlled volume, primary surface,
each moving graphics owner, each collision-line incidence, and selected key /
switch conditions separately. This follows `Anims.s` DoorRoutine/LiftRoutine's
zone-plane, wall and floorline-flag writes. A graphics owner is not assumed to
own the associated collision line. All controllers in all sixteen original maps
resolve against their actual source records. Slot-before-file-alias tests also
cover the inspector's corrected lift lookup.

The mechanism inspector exposes these relationships in a collapsible attachment
table, with keyboard-accessible room-framing links and explanatory tooltips.
Framing keeps the mechanism selected and does not alter source. Moving-room copy
diagnostics identify the graphics-owning rooms and point to that inspection view.
This is dependency inspection needed for independent controller allocation, not
the controller allocation or complete room clipboard operation itself.

## Independent moving-controller compiler allocation

`SourceBuildOptions.controllerCopies` now accepts explicit native controller /
source-room correspondence, separate graphics-owner mappings and collision-line
room mappings. `movingControllerCopies.ts` validates the complete relationship
before topology compilation: fresh controlled identities, matching native
templates, a common exact translation, unique moving-plane controllers, complete
attachments and valid native header settings. Initial position changes require
synchronized source geometry and are rejected. Copied moving volumes must retain
one convex cell, original heights and boundary relationships; missing adjoining
doorway rooms remain an actionable error.

The animation compiler allocates separate door/lift table entries and binds their
surface and each wall/trigger line to the explicitly mapped generated copies.
Original and copied attachment candidates are separate. Both use the same wall
matching, ambiguity and count validation. Header range/rules/conditions/velocity
can differ while the native pointer and runtime-room fields are rebuilt. Native
DoorRoutine/LiftRoutine remain unchanged.

Compiler fixtures copy an original level-1 controller together with its graphics
owners, collision owners and doorway neighbours. Originals are set to Never;
copies use Timer and a different destination. Model checks verify distinct
runtime rooms, graphics pointers and collision lines, settings and compiled-native
save/reopen, plus atomic rejection of incomplete mappings, mismatched translated
groups, duplicate controllers and changed initial positions. Native geometry is
synchronized on the first simulation update, as in the original routines; the
browser test observes both that initialization and subsequent motion isolation.

Editable-source integration is recorded below. Ordinary authoring controls,
the room-copy command, switch allocation/internal condition remapping, grouped
moving cells and broader transforms still require implementation before complete
room clipboard authoring or R7 completion can be claimed. Normal builds keep
rejecting moving copies that do not provide the explicit controller plan.

## Persisted copied-controller source

Room-source version 5 stores controller plans alongside stable source room IDs.
Settings bytes and attachment pairs are copied into frozen arrays, avoiding
mutable Maps/typed arrays in history or saved source. Capturing a plan without
an explicit header snapshots its current native settings; subsequent original
range/rule changes do not alter the copy. Native graphics pointers and runtime
zone fields are still rebuilt from validated attachment identities.

`setRoomControllerCopies` produces one source/history change. Normal preview and
game export compile the persisted plans automatically. Room creation, redraw,
height/property changes and shared-boundary/group moves retain them. Deleting a
controlled copied room removes its controller in the same source edit; Undo
restores both. Removing a required attachment room still requires repair before
building, while its editable source remains saveable. Structural corruption,
duplicate attachment keys/controllers, invalid bytes/ranges/rules and unsupported
source fields are rejected explicitly on restoration. Versions 1–4 remain readable.

Model verification covers source save/reopen, independent settings after native
template changes, normal export, immutable snapshots, group moves, deletion and
whole-operation Undo/Redo. Browser verification imports actual editable v5 source,
uses normal Build & play, saves after motion, reopens that downloaded project and
checks the reset controller's authored range. Workspace replacement is confirmed
by the old iframe detaching before querying the replacement engine, avoiding a
stale-preview readiness race. Room-copy/ordinary mechanism authoring UI and the
remaining relationship work are still required.

## Copied motion settings inspector

Copied door/lift rooms now expose their captured ranges, opening/closing rules
and key/switch requirements in the inspector. Range edits validate both draft
destinations and commit together; Escape cancels a field, Reset clears the draft,
and Undo/Redo restores each committed source edit. Rule and requirement changes
are immediate source edits. Tooltips and visible notes explain quarter units,
height direction, fixed initial position, shared level conditions and playtesting.

Imported and copied mechanisms share the range form, rule controls and native
range/activation validator. Copied setters preserve the native pointer/attachment
plan and original files; only the selected captured settings change. The native
ungated occupied-door exception is retained, while adding a gate to a stationary
intermediate door is rejected. Existing original-controller controls keep their
labels and behavior.

Browser checks exercise invalid drafts, cancellation, range Undo/Redo, rule/key
Undo, source save and actual native movement to the edited destination while the
original stays still. Both browsers pass those cases and adjacent imported
mechanism cycles and keyboard/tooltip checks. Room copying/creation UI, switch
allocation, broader motion authoring and remaining requirements are still open.

## Independent switch compiler allocation

`SourceBuildOptions.switchCopies` now supplies explicit native template slot,
unused destination slot, copied activation room and copied graphics room. Native
`Anims.s` / `objects.c` SwitchRoutine owns exactly eight slots, each controlling
bit `4 + slot`; the compiler preserves that contract. Slot reuse, shared graphics
attachments, missing owners and inconsistent whole-room translations are rejected.
Original slots remain intact and the new slot gets independently allocated
activation points and a relocated copied wall-phase pointer. Original and copied
switches share the existing point-pair/wall transformation and point-limit checks.

`remapSwitchConditions` maps explicitly copied internal switch requirements while
preserving keys and external bits, rejecting condition aliases. Clipboard assembly
must provide that explicit correspondence; the compiler does not guess which
external switch relationships the user intended to copy.

The switch fixture retains adjoining native doorway/corridor rooms. Closing the
tested doorway would remove its switch wall, so a copied switch without its
complete graphics attachment remains an error. Model checks cover slot/target
rejection, independent graphics/point identities, native settings, compiled-native
save/reopen and condition remapping. A bounded read-only runtime probe exposes
actual switch state and phase writes. In both browsers, pressing Space near the
copied activation pair sets only its new condition bit and wall phase; the
original switch stays unchanged. All 759 model tests and ten focused native
switch/copy-controls/attachment browser cases pass; production/runtime builds pass.

Editable switch-source integration is recorded below. Full clipboard assembly,
ordinary switch authoring controls and remaining relationship/UI work are still
required before R7/R9/R11 completion can be claimed.

## Persisted switch-copy source

Room-source v6 stores copied switch slot/room correspondence and all fourteen
captured native settings bytes in frozen source records. Both switch and moving
controller lists are explicit in v6, so dropping a list is diagnosed as damage;
versions 1–5 remain readable. Native settings, slot bounds, duplicate destinations,
unexpected fields and malformed records are checked on restoration. Geometry
dependencies remain separately validated at build time, preserving repairable
editable projects.

`setRoomSwitchCopies` produces one source/history command; normal preview/export
automatically consumes the stored plan. Subsequent native template settings do
not mutate the captured copy. Geometry operations carry both kinds of controller
plan together. Removing a copied activation room removes its switch plan in the
same edit, and Undo restores it. A required graphics room removed separately
still needs attachment repair before building.

All 763 model tests pass, including combined moving/switch plans surviving a
shared group move and v6 save/reopen, settings independence, normal export,
deletion/Undo and corruption rejection. Fourteen focused Chromium/Firefox source
motion, room-group and switch cases pass; the final switch cases were rerun after
tightening the v6 list contract. Browser tests open actual editable source,
activate the copied switch, save, reopen the downloaded project and verify its
authored state resets without changing original files. Production builds pass.
Switch authoring UI and complete room clipboard assembly remain open.

## Native unused-slot write safety

Native inspection found that the C switch routine's per-tick wall refresh also
wrote through unused slot graphics fields. In `Anims.s`, `nobutt` advances unused
off records without that unconditional write. The port could therefore corrupt
graphics headers at offset zero or write outside the buffer through stale
positive offsets, even though editable/native formats correctly preserve unused
stored fields rather than treating them as active references.

The refresh now applies only to active slots. Every switch graphics write,
including use and timer-reset paths, requires the complete six-byte phase/texture
write span inside the loaded graphics buffer. Native condition bits, activation,
timer behavior and audio events remain unchanged, including the original timed
reset branch for a stored inactive/on record with a valid target.

`tests/feedback_switch_slots.c` reproduces the old unused-slot corruption and
checks active toggles, idle slots, negative/large/partial-span offsets, zero/short
buffer lengths, boundary canaries, condition bits, timer paths and audio events.
It passes as a Windows executable and in WebAssembly with SAFE_HEAP, and runs in
editor CI. The Windows Release game and editor preview runtime build; all 18
focused Chromium/Firefox switch/copy-control/motion cases pass with that runtime.
Workflow actionlint also passes. This fixes a port-specific memory write issue;
it does not complete switch authoring or the room clipboard workflow.

## Switch reset-mode inspector

Original and copied switches now expose a shared After activation control:
Use again to reset / Automatically resets. Setters follow `Anims.s` backtoend's
zero/nonzero byte test and preserve existing nonzero values, countdown/state and
attachment bytes. Original edits patch only their settings byte; copy edits
change only the selected captured source record. Both participate in Undo/Redo.

Copied sections display their assigned condition number, original template and
activation room, with a graphics-room framing link. Notes/tooltips explain Space
activation in Build & play and matching Switch N under a door/lift's Requires all
selected. Framing preserves selection/source. No invented seconds/duration field
is offered; the runtime's native timing remains authoritative.

All 765 model tests and 14 Chromium/Firefox reset-mode, source save/reopen and
controls cases pass. Tests cover original/copy independence, preserving nonzero
native flags, invalid targets, Undo/Redo, saved bytes and native countdown
activity after Space activation, without assuming a wall-clock duration. The
existing native switch regression covers the timer's terminal reset branch.
The production build passes and the inspector layout was reviewed. Full switch
creation/placement, automatic clipboard slot/condition allocation and complete
room clipboard assembly remain open.

## Atomic room-group copy assembly

`roomClipboard.ts` now captures immutable document/selection snapshots and
assembles translated same-project room groups as one command. Copies receive
fresh source IDs with retained wall/material provenance, captured four-plane
heights and native room properties. Authored appearance is retained; otherwise
closure appearance is derived from the room's own native wall material when
available. Rooms without a suitable material still require their adjoining
geometry or an explicit repair; no missing switch/moving wall is fabricated.

Internal teleport destinations remap to new source identities and translated
coordinates; external destinations remain external. Incoming links, player
starts and the global exit are unchanged. Active supported non-avatar records
use the existing full native object copier with fresh CIDs, room bindings and
translated positions, preserving captured settings after original edits.
Reserved/unsupported active content is an error rather than silently omitted.

`roomClipboardControllers.ts` rebinds original and previously copied door/lift
graphics and collision relationships, and allocates switch slots independently.
Internal switch conditions remap to the new circuit; keys/external conditions
keep their meanings. Allocation reserves both active/assigned slots and unused
bits referenced by existing gates or external copied requirements, preventing a
new switch from taking over an unrelated gate. Required adjoining room mappings
must be complete. Full native compilation validates the provisional result
before returning one source/files/object/link/appearance history command. Any
failure leaves the destination untouched.

All 774 model tests pass. Copy cases cover original/copy motion groups, actual
object snapshots, internal/external links, partial native floor footprints,
save/reopen, Undo/Redo, overlap/attachment errors, reserved gate bits and a real
eight-active-switch capacity boundary. Eight focused Chromium/Firefox circuit,
motion and switch cases pass; the two circuit cases were rerun after reservation
validation changed. Native Space activation opens only the newly pasted door
through its newly pasted switch; the source circuit remains inactive, every
sampled moving wall/plane stays synchronized and saved source retains the plan.
Production builds pass.

This is the validated copy model, not a completed clipboard UI. Room selection /
placement dialogs, dependency expansion and appearance review, transforms beyond
translation, cross-project resource transfer and remaining editor requirements
stay open. Shared native material resources retain their existing project
provenance; broader independent resource authoring remains R8/R11 work.
