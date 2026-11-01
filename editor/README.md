# Alien Breed level editor

A React/TypeScript editor that runs directly in a browser on Windows and Linux.
The full AMOS editor port is in progress. Track the complete requirements,
source contracts and remaining work in [the port notes](../docs/editor-port.md).

## Run locally

Use Node.js 22.12 or newer:

```sh
cd editor
npm ci
npm run dev
```

Open the address printed by Vite. The original 16 map triplets are copied from
the repository into ignored `public/reference/` during development and builds.
Opening local files does not upload them anywhere.

Engine view requires WebGL2. If its renderer cannot start, the editor shows
recovery steps and keeps the map editable and saveable. After enabling browser
hardware acceleration, updating the graphics driver or changing browsers, use
**Reload engine** to retry. A failed renderer does not count as a loaded map.

## Publish a static web build

Build the native preview with `tools/build_editor_runtime.py` from the repository
root, then run `npm run build:web` in `editor`. Deploy `editor/dist` as a whole.
The package puts the matching native engine and bridge in a content-versioned
directory, so refreshing an older published editor loads the new runtime too.
`dist/build-info.json` records its resource hashes. To test that exact package
without rebuilding it, set `EDITOR_USE_PACKAGED_BUILD=1` before Playwright.

## Editing views

**Editor view** has Top, Bottom, Front, Back, Left, Right, 3D orthographic and
3D perspective projections, chosen beside the viewport tabs. **Engine view**
is the actual game renderer and playtest. **Split view** keeps the chosen
editing projection alongside the engine. The editable volumes represent room
outlines, native heights and actor anchors; game-specific decorative surfaces
and material appearance are inspected in Engine view.

- Click rooms, objects or starts to select them. The inspector, source document
  and history are shared by every view.
- Drag selected room corners to reshape outlines. Front/back lock Z; left/right
  lock X. In 3D, corner drags stay on the room's horizontal floor plane.
- Drag **F** / **C** to change the displayed storey's floor/ceiling. Invalid
  heights show a diagnostic and are rejected atomically. Water stays independent.
- Selected objects expose X/Y/Z handles. Direct drags use the visible axis plane;
  3D direct drags keep height fixed. Y increases downwards. Starts stay floor-bound.
- Each release is one Undo; Escape discards a drag. Tab/arrow keys nudge handles;
  Shift increases the increment. Grid snapping applies; Alt uses finer increments.
- Wheel zooms at the pointer. Space/middle drag pans. Right drag orbits 3D views.
  F fits the level; Shift+F frames selection. Ceilings/Wireframe affect display.
- Add/duplicate use click placement in editing views. Top retains room drawing
  and rectangle selection; choosing those tools switches to Top automatically.

## Understanding ceilings

Select a room and edit **Room height (headroom)** for its normal, positive
floor-to-ceiling size. A larger size raises the ceiling; the floor and water
stay fixed. Two-storey rooms have independent lower/upper headroom controls.
**View room from side** frames the room and its F/C handles. Each headroom edit
is one Undo and uses the same validated native planes as direct height edits.

The shaded section labels each usable room. When an upper floor meets the
lower ceiling, its two handles are separated left/right. **Absolute plane
positions** still expose native coordinates (larger numbers are lower down).
**Show ceilings** only affects editing-view visibility, leaving game geometry
intact. Open or closed doors use their mechanism movement controls.

## Painting walls, floors and ceilings

Select a room, then **Paint room textures** in its inspector. Engine view opens
that room. Stop a running playtest, click the surface, choose **Paint this wall**
or **Paint this surface**, pick a material, then **Apply**. Arrow/look controls
reach ceilings and floors. Escape cancels a draft and Undo restores the material.
Moving faces offer **Edit door/lift**. Straight walls and normal floor-bank
patterns are supported; special surfaces without painting controls say so.
**Wall repair appearance** configures newly created boundaries and changed
openings; individual existing surfaces use the painting workflow above.

## Adding and removing outline corners

Select a room in Top view and use an edge's **+** button, or double-click an
edge. Slanted edges snap along their native whole-coordinate positions. Edges
with no interior whole-coordinate point explain why insertion is unavailable.
The inspector's **Edit room corners** also has Add corner after / Remove corner
controls. Shared boundaries update together and each operation is one Undo.
Deletion preserves the minimum three corners and rejects stranded starts,
objects and teleport destinations. Newly shaped imported boundaries may need
an explicit Wall repair appearance, just as redrawn boundaries do.

## Current working workflows

- Select a connected original or newly drawn room with no inward corners.
  Apply room appearance for a newly drawn room, then select **Make door** or
  **Make lift** in the inspector. Set destinations,
  starting endpoint and trigger rules, then **Apply mechanism**. Smaller height
  numbers are higher. Doors initially use the room's ceiling/floor as open and
  closed destinations and start closed. Lifts need two different destinations.
  **Browse moving wall materials** changes the moving face style. Space uses a
  manually triggered mechanism in **Build & play**. Cancel/Escape discards the
  draft; Apply/Remove each supports Undo. Save project retains the mechanism;
  room copy automatically includes adjoining rooms and creates independent
  controllers. Original room identities and their starts/actors remain linked. Existing
  movement/switch ownership is protected; its inspector links to the current
  controller when needed. Reassignment and grouped moving volumes remain gaps.

- **Add object** opens a searchable library with one choice per supported variant:
  named key colours, weapon ammunition, enemies and scenery. Choose a tile, then
  click inside a room on the map. The displayed floor determines height. Repeat
  clicks add more; Escape or **Cancel placement** ends placement. Grid snapping
  applies; hold Alt for finer placement. Each click is independently undoable.
  **Duplicate** / **Ctrl+ D** also uses click placement, retaining custom settings
  and the source object's offset above the destination floor.

- Select an object and choose **Type** at the top of its inspector. Named
  choices are grouped as Enemies, Pickups and Scenery, using pristine original
  released records from all 16 maps. Selection applies immediately as one Undo,
  including artwork, dimensions, health and behaviour settings. Position,
  height, room, floor and identity stay fixed. Pickup-specific settings follow
  the type selector. Player avatars and projectile pools stay protected.

- In paused Engine view, click a normal floor or ceiling and choose **Paint this surface**.
  Select any of the original 16 patterns and a texture repeat from 1/32? to 4?.
  Apply commits one undoable edit; Escape cancels. Lower and upper storeys and
  copied rooms stay independent. Saving and room copy/paste retain the paint.
  **Reset surface material** restores the picked surface; **Reset painted room
  surfaces** clears all surface overrides for that room. Bump mapping, water
  materials and no-roof/sky authoring are not part of this normal-pattern brush.

- In paused Engine view, click a straight wall and choose **Paint this wall**.
  Choose a native style using the sheet previews, then Apply, or Escape to cancel.
  Paint affects the picked native face or authored boundary and its generated
  pieces, including step/lintel bands around openings. Other rooms keep their
  materials. Alignment, heights, lighting, transparency and movement stay intact;
  incompatible sampling produces an error. New perimeters and repaired doorways
  support the same workflow. Save project, room copy/paste and Undo retain paint.
  **Reset wall material** restores that wall; **Reset painted room walls** in the
  room appearance inspector also clears missing-face dependencies after redraws.

- Select an existing door/lift room to edit its movement destinations. **Apply
  range** commits both endpoints together; Undo restores them. Values use world
  units in 0.25 steps, with smaller values higher. The range must contain the
  fixed initial position. Rules, native speeds and attached graphics stay intact.
  Stationary lifts and gated doors must keep one endpoint at that initial height
  so their trigger rules can start movement. Ungated doors retain the original
  game's player-entry opening behaviour, including intermediate starts.
  Escape restores one draft field; Reset range draft discards both. Creating new
  controllers and synchronizing edits to moving-room geometry remain in progress.
- Shift/Ctrl/Command click objects in the map or object list to add/remove them
  from a group. The group inspector moves them by X/Z offsets, aligns either
  coordinate to the first selected object, or deletes the selection. Every edit
  validates the whole group and is one undoable action. Heights are preserved;
  destinations outside editable rooms reject the complete operation. Clicking an
  unselected object selects it alone; Escape or Clear object selection clears the group.
  Drag a selected marker to preview moving the whole group. Grid snapping applies
  to the shared offset, preserving spacing and fractional starting positions.
  Release commits one edit; Escape, pointer cancellation or focus loss discards it.
  Moves keep each object's floor: on shared boundaries, upper-floor objects choose
  a room with an upper floor. Moving them into a single-floor room rejects the
  whole group. The current room is preferred when it remains compatible.
  **Frame selected objects** fits the group's authored positions in the map.
  **Shift+F** also frames a single object, room or player start. Framing changes
  only the view and stays explicit: later edits do not undo your pan or zoom.
  **Delete** removes selected objects as one undoable action. It is ignored while
  editing a field, drawing a room or using a dialog. Ctrl/Command+Z restores the
  object or complete group.
  **Duplicate selected objects** uses the group inspector's X/Z offsets to copy
  the batch. **Ctrl/Command+D** copies a group in place. Copies are selected and
  retain spacing, settings, floors and exact heights with independent native
  records. One Undo removes the complete batch; capacity or invalid destinations
  reject every copy. Hold-to-repeat is disabled for this shortcut.
  **Select all matching objects** selects the active search results across both
  floors. **Ctrl/Command+A** does the same outside fields in the object list;
  on the map it selects only the displayed floor and ignores the list search.
  Hidden objects and player avatars are excluded from map selection. In text
  fields the shortcut keeps its normal text-selection behaviour.
  **Box select objects** or **B** starts a rectangle on the displayed floor.
  Drag in any direction; hold Shift to add. Completion returns to Select so the
  group can be moved immediately. Escape during a gesture cancels it and retains
  the previous selection. Hidden markers and player avatars are excluded.
  The group inspector's **Object heights** section applies a shared half-unit
  offset or aligns absolute heights to the first selected object. Negative offsets
  raise objects. These edits preserve X/Z and floor bindings, are undoable as a
  group, and reject the whole operation if any height would exceed the native range.
- Toggle **Visibility overlay** on the map, then select a room. Amber marks its
  selected runtime cell, cyan its potential visibility and green its portals.
  Concave rooms offer each compiled cell separately. Expand Runtime visibility
  table for exact zone, clip and mask entries. Preview/export share this artifact;
  edits rebuild it, and errors clear outdated overlays. These are conservative
  candidates rather than exact line of sight; both floors share the table.
  **Frame visibility** fits the inspected area without changing source. Choose a
  clip target to show its left (blue) and right (pink) boundary points. The table
  distinguishes saved flags from actual word offsets assigned by the native loader.
  **Hide visibility controls** clears the panel out of the way while keeping its
  diagnostic drawing. Show controls restores your cell/target choices. Scrolling
  a long table keeps the map view fixed.
- Placement uses the native 250-record limit, including reserved avatar and
  projectile pools. Those reserved records cannot be copied. Added objects and
  duplicates have independent native position slots and portable saved records.
- Select a key pickup and choose **Key colour** to change both its artwork and
  the door requirement it unlocks. Green, red, yellow and blue correspond to
  Key 1–4 in mechanism requirements. Undo restores both values. Custom artwork
  or condition combinations are retained until you explicitly choose a colour.
- Select an ammunition pickup and choose **Ammunition for** by weapon name.
  Its type and native artwork change together, with one undo. The game retains
  control of the pickup amount. Custom records are preserved until a choice is
  made; unsupported flamethrower/unused ammunition slots are excluded.
- Select a weapon pickup and choose plasma, rocket, grenade or shotgun by name.
  Artwork, size and weapon grant change together as one undoable edit. Changing
  between tall and short pickups preserves its floor-relative position, including
  manual height offsets. Custom records remain intact until you make a choice.
- Import a PNG in a room's material browser. It resizes to the native sheet,
  matches its existing colours and preserves its lighting palettes. Compare
  fitted/native previews, then Apply PNG as one undoable shared-material edit.
  Cancel or Escape discards the conversion. The original PNG stays in the
  project and can be downloaded unchanged after reopening or recovery.
- Keep native wall-sheet replacements inside the project, with undo/redo,
  workspace recovery, preview and game export. Advanced Native sheet tools in
  the material browser import compatible raw/SB WADs; Restore original sheet
  is available beside PNG import.
  An imported sheet changes all walls using that shared material. Exported
  installation notes identify shared wall files as well as the target level.
- Browse lower/upper wall styles visually in the room and stair forms. Search
  by material/style and preview the original palette shades. Choosing a tile
  fills the form; Apply commits the authored style as one undoable edit.
  Missing material files show an error and retry without changing source.
- Drag floor/ceiling handles in a room's section diagram or adjust them with
  Up/Down (quarters) and Shift (4 units). Release applies one undoable edit;
  Escape cancels. Invalid air-band crossings leave source unchanged. Split view
  shows the actual game renderer and rebuilds after applying the height.
- Build a straight stair flight from endpoints, width, count and floor heights.
  The labelled layout checks overlap, shared boundaries, quarter-unit heights,
  native climb allowance and room limits before one undoable creation. Choose
  actual template wall styles and lighting; Escape cancels without saving a
  partial flight. Build & play verifies clearance and joins in the game.
- Open any shipped map, a ZIP containing game levels, a saved `.ab3d` project,
  or the three `twolev` files selected together.
- Pan/zoom the map, inspect lower and upper floors, select zones and objects,
  and search the map contents.
- Edit static/animated zone light, teleport destinations, door/lift rules and
  conditions, existing object positions/heights/health, and player starts.
- Place player starts in edited or newly drawn rooms using the Starts inspector
  or by dragging their map markers. These source starts survive undo, project
  saves and workspace recovery. Preview, Build & play and Export level compile
  the authored source without replacing it with generated runtime cells.
- Move existing enemies, pickups and scenery into edited/new rooms using the
  Object room selector or map dragging. Deletion is undoable, decimal coordinates
  are accepted, and project copies/recovery retain the authored positions.
- Link edited/new teleporter rooms to a destination room and landing point.
  Show destination frames the target and highlights the link on the map.
  Broken destinations remain editable; disable/repair them without losing source.
- Undo/redo binary edits, download portable project copies, export a game
  triplet ZIP, and recover the previous workspace through browser storage.
- Open a simultaneous 2D/3D view, visit selected zones, inspect lighting with
  the actual game renderer, and build/play the current map in a chosen slot.
  In inspection, click a surface to identify its type, lower/upper floor, material
  and authored height, and select its room without moving the camera. Arrow keys
  turn/look and clear the readout; Enter picks under the centre crosshair.
  Map/list selections still visit their rooms. Picking uses the native captured
  world frame and accounts for viewport scaling and letterboxing. Opaque object
  pixels hide surfaces; transparent gaps keep the surface behind them pickable.
  Surface inspector settings apply to the selected room. Clicking a visible
  enemy, pickup or scenery object opens that object's settings and reveals it in
  the Objects list, clearing any navigator search. This also keeps the camera
  still. Object edits use the same Undo as map/list edits. Player avatars remain
  controlled through Starts. Individual surface edits remain in progress.
  Inspection also supports the editor's Ctrl/Command shortcuts for copy/paste,
  duplicate, Undo/Redo, save and open, plus Delete, help and search. Ctrl/Command+A
  selects objects on the current editing floor, even if map markers are hidden.
  History and saving stay available while the native preview rebuilds. Playtests
  keep their game controls; editor shortcuts apply to paused inspection.
  Playtests use separate browser storage from the normal game.
- Choose the target game slot in Project or the 3D toolbar. The choice supports
  undo/redo and is saved in project copies and browser recovery; changing views
  or the project name does not change it. Game ZIPs contain the matching
  `levels/level_a/` through `levels/level_p/` directory and installation notes.
  Extract those ZIPs into the game's data directory. Older v1 project backups
  did not store a slot and open at slot 1; select the intended slot before use.
- Choose Level exit in Project: retain the original slot exit, finish when
  entering a selected zone, or disable completion for free playtesting. It is
  saved with undo/recovery and exported as `twolev.cfg`. Custom exits require
  the updated game port; original Amiga engines keep their built-in exits.
- Reject incomplete/corrupt input and invalid moves without replacing the
  document being edited. Imports are transactional and stale asynchronous
  loads cannot overwrite newer work.

Room outline drawing/editing/deletion is available as editable source. Choose
a room settings template, click Draw room, then click corners in either
direction. Enter or clicking the first corner finishes; Backspace removes the
last corner, and Escape/changing tools cancels without committing. Select an
outline to drag its corners or edit coordinates in the inspector. Shared
corners and T joins move together; crossing and overlapping outlines reject
atomically. Delete room supports undo. Templates remain available after
deleting all outlines.

To change an existing room's shape or corner count, select it and choose
Redraw outline. Click the replacement corners in either direction, then press
Enter to apply the whole outline as one undoable edit. Escape, Cancel room or
changing tools discards the draft and keeps the old room. Saving during a draft
saves only committed geometry. The room identity, starts, objects and links
remain associated; reassign anything that falls outside the new outline.
New boundary walls require a Wall repair appearance with an explicit wall style
and corner lighting before preview/export. Existing wall references retain
their styles, and internal decomposition seams do not acquire walls.

For a new room, choose its Lower wall style and Corner lighting in the inspector
and click Apply appearance (also choose an upper wall style for two-floor
templates). Styles come from the selected room template, retaining texture,
alignment and sampling; perimeter walls span that room's floor and ceiling.
Then open 3D view or Split view to inspect it, Build & play to start from player
one, or Export level to download the same compiled result. Source room visits
and viewport picking follow compiler numbering automatically. Save project
always keeps the editable outlines, appearance and gameplay links.

Touching a new room against a static room boundary creates the doorway
automatically. Its wall graphics leave the correct steps and lintels for the
destination heights. If deleting or moving an imported neighbour closes an old
doorway, select the remaining room, choose its Wall repair appearance and
click Apply appearance. This fills the closed opening with the chosen wall
style and lighting; existing wall styles remain authored. The choice is saved
with the project and supports undo/redo. A missing closure choice stops a build
with instructions rather than exporting a collision wall with a visual hole.
If a new room replaces the old neighbour, the same repair appearance generates
the steps and lintels around the new destination's opening. Matching heights
remove the old wall pieces entirely, leaving a full-height doorway.

The application is not yet a complete replacement for the original editor.
Broader material/asset authoring, source AMOS project adapters, partial decorative
geometry and openings into moving volumes, split mechanism/controller groups, control paths and full 3D surface
picking are still being implemented. Unsupported builds produce an actionable
error and preserve source; they never export stale geometry in its place.

Project format v8 stores new-room and doorway-closure appearance alongside the editable
room/gameplay source and original binary baseline. Formats v1–v7 remain readable;
projects without the newer fields retain their earlier format. Incomplete
projects can be saved/recovered before all build dependencies are assigned.
Appearance and geometry edits use the same undo/history model.

Room heights are independent of the shared material template. Select a room,
edit its floor/ceiling fields in Room heights and choose Apply heights. Enter
also submits the full set; Escape restores the focused field. Values use native
quarter units and smaller values are higher. Upper fields appear for an active
two-storey template. Copy shown heights from another room loads a draft for
review; Apply heights commits it. Restore template heights removes overrides.
All changes support undo/redo and portable source saves. Collision headers,
anchored walls, floor/roof surfaces and doorway openings rebuild together.
Changed imported doorways may need Wall repair appearance on both sides; the
build error identifies the room. Water retains its independently authored
height. Moving-door/lift ranges and curved-wall height attachments still need
their dedicated authoring work; unsupported builds preserve editable source.
Height-bearing room source uses schema v3; ordinary projects retain schema v2
and earlier project/room-source formats remain readable.

Preview and game export enforce the current port's graphics-command budget per
room and floor. Walls, surfaces and draw controls share that budget; Help shows
the value used by the engine. An oversized file can still be opened and saved
as an editable project, but building/playing it stops with the affected room,
floor, command count and limit. This prevents the engine silently omitting the
end of a draw stream. It does not implement the separate original-Amiga export
profile, which remains part of the unfinished editor work.

Room-source snapshots also retain each imported wall's identity when corners
move, including shared-wall splits. New saves use room-source version 2; older
version 1 saves still open. If an older save already moved a wall, its material
association is recovered only where the saved geometry proves the match.

Room geometry accepts either drawing direction. The shared geometry module
normalises winding, pairs opposite shared edges, splits partially
shared walls and decomposes concave outlines. These contracts are tested
against the orientation in all 2,496 original room outlines. There is no user-facing
clockwise/anticlockwise setting.

## Build and verification

Build the engine preview first, from the repository root, with Emscripten
5.0.0, CMake, Ninja, Python and ffmpeg available. Follow the main README's web
toolchain setup:

```sh
emcmake cmake --preset emscripten-release -G Ninja
python tools/build_editor_runtime.py
```

The helper builds the game and links a separate editor host against its real
renderer/simulation objects. Output goes to ignored `editor/public/runtime/`.
The React application can run without this output for 2D work; prepare it to use
3D/playtest and before producing a complete distribution. CI builds it on Linux
and includes it in both Windows/Linux browser test jobs and static artifacts.

Then, from `editor/`:

```sh
npm test
npm run build
npx playwright install chromium firefox
npm run test:e2e
```

`dist/` is a static production build. Browser tests run that build in Chromium
and Firefox, covering all 16 maps in the actual engine and map UI, editing/undo,
file round trips, focused-field saves, autosave recovery, malformed inputs,
invalid positions, narrow layouts and asynchronous-load races.

Compression fixtures were generated using the independent C decompressor.
From the repository root, regenerate them only after reviewing source changes:

```sh
python tools/editor_oracles.py --unpacker <path-to-sb_unpack>
```

Visibility geometry fixtures execute the original editor's embedded 68020 bank
using the development-only Unicorn 2.1.4 package:

```sh
python tools/editor_geometry_oracles.py
```

These validate the ordering-mask and portal-point primitives. They do not yet
prove a full visibility compiler; see [the geometry evidence](../docs/editor-geometry.md).

Wall sheet fixtures execute the current `io.c` texture-dump functions as C,
using the independent native SB decompressor for their input. Regenerate after
reviewing changes to native wall loading or palettes:

```sh
python tools/editor_wall_oracles.py --unpacker <path-to-sb_unpack> --cc clang
```

The tool extracts the actual native dump/dimension/BMP functions and tables
into a temporary harness; it does not replace them with an editor-side oracle.
Model checks compare all 14 decompressed sheets and every middle-shade RGBA
pixel. Browser checks compare packaged catalog dimensions and canvas pixels.
Native wall overrides now use project format v9: original level files stay
unchanged, while native material bytes and required source-file declarations
travel with the project. Preview restoration and game ZIP re-import are tested.
Projects retaining PNG sources use v10; v1–v9 projects remain readable. Game
exports carry the converted native WAD, while projects keep the original image.
See [PNG conversion contracts](../docs/editor-png.md). New-palette generation,
broader resource authoring and export profiles remain in progress.

## Shortcuts

| Action | Shortcut |
| --- | --- |
| Open files | Ctrl/Cmd+O |
| Download project copy | Ctrl/Cmd+S |
| Undo / redo | Ctrl/Cmd+Z / Ctrl/Cmd+Shift+Z |
| Select / pan | V / H |
| Temporary pan | Space+drag |
| Zoom | Wheel, anchored at pointer |
| Frame map | F |
| Search map contents | / |
| Move without snapping | Alt+drag |
| Help | ? |
| Clear selection / close dialog | Escape |
