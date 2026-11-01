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

Each stored word encodes `graphics record offset << 1 | upper`. UINT32_MAX is
none. Wall edge-extension pixels carry the same record as their colour writes.
Fast floor-column runs and floor-row spans use the current polygon record.
Backdrop writes clear provenance. Opaque sprite texels and the filled/textured
polygon-object pixel writers clear it too; transparent texels/polygon holes keep
whatever visible surface was already there. This identifies world surfaces;
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
Picking preserves the inspection camera. Camera navigation, turns, rebuilds and
playtest loads discard the previous surface readout.

Generated offsets are ephemeral inspection identities. They must never be used
as editable-source addresses: compilation relocates streams, decomposes rooms,
creates portal bands and fans out native templates. Individual surface editing
still needs stable source ownership/provenance and saved overrides, plus Undo,
rebuild and native-pixel evidence. Object selection and graphical edit overlays
also remain required for full R10 completion.
