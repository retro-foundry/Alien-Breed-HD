# Native surface inspection

The editor captures graphics provenance in the actual game rasterizer. It does
not reconstruct visibility with a second JavaScript renderer or use projected
object rectangles as coverage. `renderer_draw_zone_ctx` identifies the current
native graphics record and lower/upper stream; the same wall-column and
floor-span writers that update room picking also write surface provenance.
The original `AB3DI.s` polyloop opcodes remain the authority for record meaning.

`renderer_set_graphic_picking` opts into two row-major uint32 buffers. Normal game
startup leaves it disabled and allocates neither buffer. Editor startup enables
it explicitly; allocation failure is reported and aborts preview initialization.
Resizing recreates both buffers with the existing renderer allocation lifecycle.
The added storage is eight bytes per internal raster pixel. Clearing, drawing
and swapping follow the existing room-pick frame lifecycle. A query requires a
valid captured frame and in-range coordinates; missing coverage returns none.

Surface words encode `graphics record offset << 1 | upper`. The high bit instead
tags an object record ID, with UINT32_MAX reserved for none. Editor graphics files
are bounded to 16 MiB, so their offsets never occupy that tag bit. Graphic queries
reject object tags; object queries reject surface words and invalid frames/bounds.
Wall edge-extension pixels carry the same record as their colour writes.
Fast floor-column runs and floor-row spans use the current polygon record.
Backdrop writes clear provenance. Opaque sprite texels and filled/textured
polygon-object pixel writers tag their source record ID. Projectiles and
explosions clear provenance; transparent texels/polygon holes retain whatever
surface or object was already there. Polygon draw calls carry an immutable tag
through their local pipeline, without a shared current-object global. This identifies world surfaces/objects;
frontend HUD overlays are outside the software world's provenance buffers.

`editor_pick_graphic` exposes offset, stream and opcode as read-only data for a
normalized presented pixel, using the actual display rectangle. `editor_pixel_at`
reads the corresponding colour word with the native column/row layout; browser
oracles compare this raster with provenance to verify actual object coverage.
One synchronous iframe `pickAt` gathers room and surface from the captured frame.
Parent messages retain the existing origin, iframe, revision and source checks.

`resolveEngineSurface` checks the reported record against the exact compiled
document's zone/stream, including opcode agreement. The readout shows surface
kind, stream, material when the reader supplies one, and authored height. Floor /
ceiling heights come from the compiled room header; other polygons retain their
own stored height. These are authored values, not a live moving-plane monitor.
The current inspector still edits room settings, and the readout says so.
Object picks include both record slot and CID. `resolveEngineObject` checks the
compiled artifact and its source together, rejects missing/deleted/inactive
records, mismatched point identities and player avatars, and returns the authored
object ID. Original record slots remain stable through compilation and native
shot-pool expansion. Runtime-only records cannot select a different source actor
or a room hidden behind them. Picking opens the object's existing inspector;
height and other supported edits use normal commands, rebuilds and Undo.
The Objects list clears its search and scrolls only its own contents to reveal
the selected row. Inspection clicks explicitly focus the canvas without page
scrolling, including Firefox where the click alone did not focus it.

Picking preserves the inspection camera. Camera navigation, turns, rebuilds and
playtest loads discard the previous surface readout.

Generated offsets are ephemeral inspection identities. They must never be used
as editable-source addresses: compilation relocates streams, decomposes rooms,
creates portal bands and fans out native templates. Individual surface editing
still needs stable source ownership/provenance and saved overrides, plus Undo,
rebuild and native-pixel evidence. Graphical edit overlays and broader 3D
authoring also remain required for full R10 completion.

## Inspection keyboard ownership

`inspection-shortcuts.mjs` is shared by the iframe and React. The build host
stages the same module beside `frame.html`; a failed import reports an engine
error and preserves source, with Reload engine able to retry. The matcher accepts
only supported scalar key/modifier/repeat fields and excludes Alt/composition
input, camera arrows/Enter and ordinary game keys.

In paused inspection, the iframe prevents the browser default for a matched key
and posts key intent. The parent validates origin, iframe identity, payload,
loaded-frame lifetime and play state before calling the existing editor keyboard
handler. It does not dispatch synthetic DOM events or accept editing commands
from the payload. Modal/draft/text-field rules stay owned by that handler.
Playtests retain native game input and cannot mutate editor history through this
bridge, including delayed shortcut messages.

Copy/paste/duplicate/select-all/Delete require the matching compiled document and
revision. Undo/Redo, save/open and navigation operate on the current workspace
even while its native preview rebuilds. The iframe is replaced when changing
workspace, so old frames cannot target another project's history. This keeps
rapid Undo and portable backups usable while a real engine completion reply is
pending, without allowing stale frame-dependent selection commands.

Ctrl/Command+A in inspection selects active objects on the current editing floor,
independently of the map's marker-visibility toggle and navigator search. The map
retains its own hidden-marker rule; the object list retains search/all-floor
selection. Arrow keys and Enter remain native inspection-camera/pick controls.

## Individual straight-wall materials

In paused 3D inspection, click a straight wall and choose
**Paint this wall**. The material dialog shows native sheet previews, explains
its scope and provides Cancel/Escape, Apply and Reset. Applying is one undoable
source edit. Switch-control faces are protected. Room appearance also offers
**Reset painted room walls**, including overrides orphaned by an outline edit.

The compiler records source provenance separately from emitted byte offsets.
Paint is keyed by source room, lower/upper stream and native source record;
it follows split pieces and remains independent across copied rooms. Source
snapshot v7 stores the four sheet/sampling fields and retains the original files.
Wall geometry, strip span, horizontal/vertical phase, brightness, record type and
moving-controller attachments remain unchanged. Clipboard placement, save/open,
Undo and Redo preserve these overrides.

Newly drawn perimeter walls and repaired doorway fills are also paintable.
They use source snapshot v8: each room allocates boundary identities on first
paint and retains an identity-to-current-edge list and independent lower/upper
overrides. Generated records carry transient source-edge indices back to the
matching source document. Normalization, shared-corner moves, T-junction splits,
translation and copied rooms preserve identities. All emitted pieces of a
painted boundary, including step/lintel bands around openings, share its paint.
A redraw allocates fresh IDs for genuinely new edges; it does not silently
transfer paint to an unrelated boundary. Missing painted boundaries can be saved
and restored or cleared using Reset painted room walls before building.

Applying and compiling validate packed texture addressing against the sheet's
payload, including signed phase, native zero-field dimension fallback and lift
wall-list base phases. Incompatible styles produce an actionable error instead
of changing alignment or submitting an unsafe native preview. Missing source
faces remain saveable but must be restored or reset before compilation.

This does not yet provide painting for curves, floors or ceilings. Those need their own stable source identities and
editing controls; generated byte offsets are never persisted as authoring IDs.
