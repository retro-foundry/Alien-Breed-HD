# PNG wall import contracts

PNG import is a modern authoring feature that produces the current game's
native wall format. It preserves the chosen material's palette rather than
claiming a recovered original PNG converter or generating new lighting ramps.

## Authority

- `src/io.c:io_load_walls`, `wall_texture_dims_from_size` and
  `io_dump_textures`, plus `amiga/WallChunk.s` and the renderer wall loops,
  establish the native layout: 2048 palette bytes followed by vertical strips
  containing three five-bit pixels per big-endian word. Existing native C
  oracles cover all fourteen sheets and their decoded pixels.
- `amiga/archive/amos/256wallchunk.amos`, SHA-256
  `da94e92be3540782a0e2d1746b5a109031688307973102fd6d087c8990542361`,
  was independently listed with `kyz/amostools` commit
  `cf30aa53f8422201fbad4e59ebb59649c7964326`. This snapshot has an unconditional
  `End` after saving font palette data. Its inactive tail describes `.256wad`
  output whose palette entries are global 256-colour indices, not native RGB12.
  Unknown extension pixel-access calls remain unresolved. No executed original
  converter, native palette-generation algorithm or full AMOS parity is claimed.
- The inactive tail's `3*abs(R) + 4*abs(G) + 2*abs(B)` colour-distance expression
  supplies the modern matching policy. Ties choose the first existing colour.
  Native packing is independently grounded in the active loader/renderer.

## Import and storage

The browser decodes a still PNG and resizes it with nearest-neighbour sampling
to the selected sheet's native dimensions. Alpha is composited onto palette
entry zero's background colour. Pixels are matched to the existing full-bright
palette, while all 32 authored lighting palettes remain byte-for-byte intact.
The two-image preview and adjusted-pixel count expose conversion loss before
Apply. Dimensions, alignment masks and wall-style settings remain authored.
This mode cannot introduce colours absent from the existing palette.

Structural checks follow the [W3C PNG specification](https://www.w3.org/TR/png/):
signature, chunk bounds/order, header settings, checksums and image ending are
checked before browser pixel decoding. Source files are limited to 16 MiB and
4096 × 4096 pixels. APNG input is explicitly rejected. PNG fixtures are generated
independently using Python's `zlib` and `binascii.crc32`:

```sh
python tools/editor_png_fixtures.py
```

Cancel/Escape and stale project/material changes discard a conversion. Apply
creates one asset command containing both native bytes and original PNG bytes.
An identical reapply closes review without adding history or losing the source.
Archive capacity is checked before the image edit is committed.

Project v10 declares PNG-bearing slots and stores their unchanged originals in
`assets/wall-sources/<slot>.png`, separately from converted WADs. Missing sources,
duplicate slots and schema downgrades fail explicitly. Older project formats
remain readable, and non-PNG wall overrides retain v9. Downloads/recovery retain
the original image. Game exports contain native WADs only; use the project copy
to retain editable image sources.

## Evidence and remaining work

Tests cover native packed-word order across strips, palette preservation,
weighted matching and alpha composition, independent PNG fixtures/checksums,
v10 restoration/downgrade rejection and source retention. Chromium/Firefox runs
cover 4K resizing, transparency, comparison/cancellation, identical reapply,
atomic undo/redo, game export, original source downloads and workspace recovery
with the actual game renderer. No simulation or native renderer changes are
introduced by this feature.

New-palette authoring, broader material assignment, floor/other resource
conversion, remaining editor requirements and final deployment remain open.
