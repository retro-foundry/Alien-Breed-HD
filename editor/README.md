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

## Current working workflows

- Open any shipped map, a ZIP containing game levels, a saved `.ab3d` project,
  or the three `twolev` files selected together.
- Pan/zoom the map, inspect lower and upper floors, select zones and objects,
  and search the map contents.
- Edit static/animated zone light, teleport destinations, door/lift rules and
  conditions, existing object positions/heights/health, and player starts.
- Undo/redo binary edits, download portable project copies, export a game
  triplet ZIP, and recover the previous workspace through browser storage.
- Open a simultaneous 2D/3D view, visit selected zones, inspect lighting with
  the actual game renderer, and build/play the current map in a chosen slot.
  Playtests use separate browser storage from the normal game.
- Reject incomplete/corrupt input and invalid moves without replacing the
  document being edited. Imports are transactional and stale asynchronous
  loads cannot overwrite newer work.

Geometry creation/topology editing, material authoring, source AMOS project
adapters, the complete compiler, control-path authoring and full 3D surface
picking are still being implemented. The application is not yet
a complete replacement for the original editor.

Room geometry will accept either drawing direction. The shared geometry module
already normalises winding, pairs opposite shared edges, splits partially
shared walls and decomposes concave outlines. These contracts are tested
against the orientation in all 2,496 original room outlines; canvas/compiler
integration remains part of the geometry work. There is no user-facing
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
