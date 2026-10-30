# DOS / 486 build

Target: 486 DX2/DX4 with x87 FPU, 4 MB RAM, DOS 5+, VESA RGB565 video and
Sound Blaster audio. A VBE 2 BIOS or UNIVBE is recommended. The build was
validated in DOSBox Staging 0.82.2 with `core=normal`, `cputype=486_slow`,
`machine=svga_s3`, 4 MB RAM and fixed CPU cycles. Real hardware remains untested.

## Build from Windows

Install CMake 3.16+, a native C compiler (MSVC on Windows), Ninja, Git,
Python 3.9+ and ffmpeg with libopenmpt support.
Run from the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup_dos.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_dos.ps1
```

Setup downloads DJGPP GCC 12.2 and CWSDPMI release 7, verifies their archive
SHA256 checksums, then builds a private SDL checkout pinned to
`c699512adcc139ec1a355ff97cb2e5dbab3c9ac2` with the tracked DOS patch.
Dependencies live under ignored `build/dos-deps`; no system compiler is replaced.
This is a Windows cross build, not a compiler installation on the DOS PC.

An existing GloomAmiga installation can supply the compiler and SDL Git cache:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup_dos.ps1 `
  -DjgppRoot ../GloomAmiga/build/dos-toolchain/djgpp-mingw/djgpp `
  -SdlCache ../GloomAmiga/build/dos-sdl/SDL
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_dos.ps1
```

The cache is cloned without hard links; the reference project is not modified.
The build command stages assets, regenerates the same sprite palette header as
the desktop build, compiles `ABHD.EXE`, sets a 128 KB stack, and creates
`build/dos/ABHD-DOS.zip`. The archive includes CWSDPMI, its redistribution notice,
SDL's license, game data, music, `AB3D.INI`, `README.TXT` and `ABHD.BAT`.
Only the distribution manifest is archived, excluding local saves, logs and tests.

### Direct 8-bit trial

The default build retains exact RGB565 colors. To build the optional 8-bit
version separately:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_dos.ps1 -BuildDir build/dos-indexed -Indexed
```

Its package is `build/dos-indexed/ABHD-DOS.zip`. This version uses native
320x200 indexed VGA through SDL, with the same 1:1 world resolution and UI
sizes. It needs no RGB565 VESA mode. The two color buffers each occupy 64,000
bytes rather than 128,000. The game writes indices directly; there is no
full-frame color conversion. Gameplay and source texture sampling are unchanged.

`tools/dos_palette.py` builds the menu and 16 level palettes offline. It uses
original sprite/weapon/floor/sky/polygon palettes, with wall usage counted from
each level's original zone graphics streams. Weighted median cut selects actual
source colors; neutral/UI colors are reserved. Each palette has 255 opaque
entries and a separate cutout index. A packaged 4096-entry map resolves every
original 0RGB word. Some shades merge, so color accuracy is reduced. Water and
underwater tint decode palette representatives through the original color
tables and map their results back to indices.

World palette caches use the native pixel type. Weapon and UI cutouts remain
exact. UI source images retain their RGB565 pixels and acquire indexed SDL
caches when first used under a new palette. Palette changes invalidate weapon,
water, sky and UI caches; no UI quantization runs during steady-state blits.

Under the 128-frame level-1 comparison settings below, the 8-bit trial measures
5.27 FPS at 20,000 cycles and 10.57 FPS at 40,000, versus 4.82/9.67 FPS for the
current RGB565 build: about 9% faster in this starting view. Both builds run
with 4 MB; the performance comparison itself uses 16 MB consistently.

All 16 indexed level starts pass the 4 MB runner. RGB565 captures still match
the original reference byte-for-byte. The indexed world comparison checks
43,400 central world pixels per level against those RGB565 colors transformed
by the packaged palette map. It excludes HUD regions because the native
200-line VGA and 240-line VESA modes place UI differently. Run both variants
and compare with:

```powershell
python tools/check_dos_indexed.py --dosbox <path-to-dosbox.exe>
```

Omit `--dosbox` to compare existing captures. The 4 MB indexed `--exercise`
check also covers 24 weapon frames, palette changes, original water/tint
tables, UI cache colors/cutouts, saves, menus and Sound Blaster underruns.

GitHub Actions builds the DOS ZIP on main-branch pushes and pull requests.
Version-tag releases also publish `ABHD-DOS.zip`, with its checksum and build
provenance, alongside the Windows and Linux packages. CI uses the same setup
and build scripts on a clean Windows runner without the local Gloom toolchain.

## Run on DOS

Extract the whole ZIP into a writable directory and run `ABHD.BAT` from there.
Use a lean DOS boot with XMS enabled and EMS disabled. The launcher starts
CWSDPMI with `-s-` so the 4 MB budget cannot be hidden by disk swapping.
Aim for about 600 KB free conventional RAM at the DOS prompt: four original 64 KB texture banks,
one 64 KB sky/gun bank, the original sound samples and the audio DMA buffer
live there. They are addressed directly through SDL's enabled near-pointer
mapping; there is no extra per-frame copy. Extra resident drivers, asset
replacements and larger physical video modes can exceed the tested budget.

Set `BLASTER` to the actual card resources first. If unset, the launcher uses
`A220 I7 D1 H5 T6`. SB16 signed 16-bit output and SB Pro unsigned 8-bit output
were exercised in DOSBox. Sound can be explicitly disabled:

```dos
SET AB3D_DOS_SOUND=0
ABHD
```

The DOS build includes keyboard and mouse input; gamepad/HID drivers are omitted
to keep the executable small. Keyboard controls match the desktop port: arrows/W/S move, A/D strafe, Ctrl
fires, Space operates, 1-6 select weapons, Esc opens the menu, Enter selects,
Tab opens the map, F5/F9 save/load. DOS menus default to keyboard input.
`SET AB3D_DOS_MOUSE=1` enables the DOS mouse driver transport and captures the
mouse in-game; Esc releases it for menus. This path was smoke-tested with DOSBox's
mouse driver, but mouse accuracy needs real-hardware testing.

The smallest RGB565 mode of at least 320x200 is selected. S3 in DOSBox selects
320x240, with a centered 320x200 game viewport. A BIOS exposing only larger
RGB565 modes uses that larger mode while retaining a centered 1:1 viewport.
Plain 256-color VGA output is not supported:
RGB565 retains every distinct original 12-bit color without inventing a palette.
DOS always uses the hardware full screen; the desktop window preference has no
effect. Errors and diagnostics are written beside the executable in `AB3D.LOG`.

`AB3D.INI` defaults to true 320x200 world rendering, presented 1:1 without
enlargement, one rendering thread and normal game rules. DOS requires
`supersampling=1`. Menus, HUD, fonts and the map draw at native pixel sizes
with SDL; weapons continue scaling to the game view. Long level text wraps
into pages, with a key press advancing to the next page before dismissal.
Text is laid out inside the visible screen even when the native title artwork
is cropped. Key ownership underlines stay above the bottom margin, and bars
leave room above the key row in 200-line output.
Map lines are clipped to the UI viewport. The runner checks that the 200-line
HUD captures leave every pixel below that viewport black.
Change `render_width`/`render_height` only after measuring your PC. The staged
INI preserves local edits during rebuilds; the ZIP always contains clean defaults.

## Performance decisions

- Engine and SDL use `-march=i386 -mtune=i486`, size optimization and LTO;
  unused sections are discarded. SDL SIMD assembly is disabled.
  The DOS CPUID probe is disabled because an EFLAGS.ID probe under DPMI can
  falsely report support on a 486. The executable runs with a 486 instruction set.
- The desktop renderer's 32 MB RGB-to-Amiga lookup is omitted. Existing small
  tables calculate the same 12-bit values. Initial buffers use the requested
  resolution directly, and aligned allocation has a DJGPP implementation.
- The zone-offset lookup uses sorted offset/slot triples on DOS, retaining
  first-slot behavior for duplicate offsets. This replaces the desktop table
  that occupies twice the level-data size. Diagnostics compare against the
  original zoneAdds linear search, including adjacent offsets and misses.
- Automap reservation follows the map's line count instead of preallocating
  262,144 walls (about 6 MB including the hash). Existing dynamic growth retains
  discovered segments. Allocation failures stop with an error.
- DOS palette caches and framebuffers hold native RGB565, in row-major order.
  SDL surfaces borrow the two renderer buffers and blit directly in the same
  format. There is no full-frame color conversion, transpose, texture upload or
  automatic scaling. The DOS framebuffer backend copies the result to VESA.
  Desktop ARGB, diagnostic pick buffers and byte-sized pixel tag buffers are
  not allocated. Tag stores and their pointer increments compile out of DOS
  raster loops, saving 128,000 framebuffer bytes at 320x200. Water reads native
  color words directly. Desktop tag tracing retains its buffers; requesting
  tag-dependent floor tracing on DOS reports that it needs a desktop build. Thread-local
  caches become ordinary static storage in this single-threaded target, avoiding
  emulated TLS allocations. The unused GL line scratch buffer is omitted.
  Large debug snapshots allocate only when explicitly requested on desktop.
  Alpha LUTs use a small exact cache and division instead of 320 KB of tables.
  Cache reservations start small and grow with actual sky/map content. DJGPP's
  allocator receives a contiguous 1.5 MiB heap up front to reduce wasted 64 KB
  DPMI-zone padding. Separately allocated shot/point arrays are freed using the
  exact level-data extent, fixing leaks on reload and level transitions.
- Weapons retain their original scale and sampling. Each sampled animation
  frame is cached in a native SDL surface, using RLE and a color key outside
  the 4096 source colors. Empty margins are cropped and only the selected
  weapon is retained. It decodes directly from the original PTR/WAD/palette
  without a temporary RGBA image. This skips transparent runs instead of resampling
  thousands of empty pixels every frame. The former opaque weapon tag runs and
  their cache are omitted; the SDL surface already retains the cutout mask.
- UI draws synchronously with SDL3 surface fills/blits, without an SDL renderer
  queue or OpenGL. PNG fonts are converted offline to native RGB565 with their
  authored cutout masks. The font packs the original sampled 8x13 glyphs
  without unused cell padding; glyph size and layout stay identical. The DOS
  executable and SDL omit PNG/JPEG decoders.
  The original planar title streams a plane at a time into its final surface
  and is released before a level loads. Its native SDL strips avoid requiring
  a contiguous 160 KB allocation when returning to the menu. Opaque
  color-key/RLE blits preserve their cutout shapes without alpha blending.
  Uncollected keys and disabled labels use cached grey images; collected keys
  retain their colors. HUD backgrounds and map lines use solid colors. Menus
  and transitions use solid fills, and damage feedback uses a red border.
- Original `LoadFromDisk.s SFX_NAMES` samples mix at 8007 Hz with integer math.
  Voice selection, sample mapping and `SoundPlayer.s MakeSomeNoise` distance
  volume follow the existing port. Underwater filtering uses a fixed-point
  version of its 2.2 kHz filter. No gameplay timers, AI or damage rules change.
- A bounded manual queue feeds the Sound Blaster IRQ ring. The manual audio
  hint is applied before SDL initializes its audio scheduler. Event polling runs
  every frame without yielding to a cooperative SDL audio thread. Audio is pumped
  around input, presentation and between world sections; the profiler counts
  IRQ-ring underruns. Music is converted offline to mono 8-bit 8007 Hz WAVs
  and streamed in 4 KB blocks rather than decoded as tracker modules on a 486.
- Assets use deterministic `DATA/Axxxxxx.BIN` names, with a generated logical
  path table. Original world/sprite/sample formats are retained; UI images and
  music have the offline derivatives described above. Shared marine sprites and
  the in-hand/pickup gun bank reuse one allocation with explicit ownership. The supplied binary FloorPalScaled
  is checked byte-for-byte against its assembly before packaging, avoiding slow
  assembly text parsing at startup. Rotated saves use `AUTO1.BIN`..`AUTO4.BIN`.

Original compressed level files are unpacked offline using a host tool built
from the existing `sb_decompress.c`. DOS loads the exact decoded bytes through
the existing raw-file path. This removes the roughly 134 KB decoder workspace
and compressed input buffer from each level-load peak, and avoids spending
486 time on decompression. Original source files stay untouched; the host tool
is not needed on the DOS PC.

## Memory budget

DOS full-save loading first validates the entire payload with a 4 KB buffer,
keeps the file open, then reads chunks directly into the reloaded level. This
preserves the existing v2/v3 format without holding two complete levels in RAM.
Truncated payloads are rejected before the live state changes. Desktop/web
continue to stage their saves in memory.

The runner defaults to `memsize=4`, `ems=false`, `umb=false` and `CWSDPMI -s-`.
`[DOSMEM]` reports XMS pages; that figure excludes conventional banks and is
not a whole-machine RAM measurement. The emulator's 4 MB configuration is the
whole-machine limit. The 128 KB stack is included. This is not a 2 MB build.

## Reproduce the checks

Use an installed DOSBox Staging executable. For example:

```powershell
python tools/check_dos.py --dosbox ../GloomAmiga/build/dos-toolchain/dosbox-staging/dosbox-staging-v0.82.2/dosbox.exe --sound off
python tools/check_dos.py --dosbox ../GloomAmiga/build/dos-toolchain/dosbox-staging/dosbox-staging-v0.82.2/dosbox.exe --exercise
python tools/check_dos.py --dosbox ../GloomAmiga/build/dos-toolchain/dosbox-staging/dosbox-staging-v0.82.2/dosbox.exe --cycles 40000 --exercise
python tools/check_dos.py --dosbox ../GloomAmiga/build/dos-toolchain/dosbox-staging/dosbox-staging-v0.82.2/dosbox.exe --exercise --sb sbpro2 --mouse
1..16 | ForEach-Object {
  python tools/check_dos.py --dosbox ../GloomAmiga/build/dos-toolchain/dosbox-staging/dosbox-staging-v0.82.2/dosbox.exe --cycles 40000 --frames 4 --level $_ --frontend --mouse
  if ($LASTEXITCODE -ne 0) { throw "DOS level $_ failed" }
}
```

The runner creates isolated trees under `build/dos/checks`. Each contains its
exact emulator config, startup log and a real, nonblank `FRAME.BMP`. It checks
the requested map, frame count, successful teardown and zero audio underruns.
`--exercise` additionally checks aligned allocation, navigates New Game through
SDL key events, streams title music, captures `MENU.BMP`, writes/reads all five
autosave headers and the quicksave, restores full saves eight times before and eight times after
gameplay, verifies level/object/map payload hashes, rejects truncated payloads,
and mixes shotgun/flame effects. It captures
the map, numeric HUD, grey/collected keys and options screen, verifies native UI blit dimensions
and exact cutout masks in opaque grey images,
checks all 16 level narratives for lost glyphs, and compares cached scaled
weapon color pixels against the original CPU draw routine and verifies that
both DOS tag buffer pointers remain null. Save tests
write only inside that isolated test directory.
Screen-fit diagnostics check every narrative page and the main, options,
autosave, confirmation and pause menus at both 320x200 and 320x240. HUD
diagnostics draw empty, one-unit and full bars/numbers, all four collected
keys and their underlines, the four-digit FPS counter, crosshair and map;
every UI image and fill is checked against the output bounds. `HUD200.BMP`,
`BAR200.BMP` and their 240-line counterparts capture those layouts.

The pixel oracle runs the production pipeline with 4 MB and builds the original v1.0.2 renderer from commit
`487c1aae41ae3443b8ea118b4d14550647597c61` in an ignored snapshot, with only
DOS memory/API adaptations, and compares complete captures against the native
pipeline using a 16 MB reference profile. Its color conversion exists only in
that test snapshot:

```powershell
python tools/check_dos_pixels.py --dosbox ../GloomAmiga/build/dos-toolchain/dosbox-staging/dosbox-staging-v0.82.2/dosbox.exe
```

Both variants force exactly one 20 ms logic tick per diagnostic frame, including
the interpolation remainder, for deterministic comparison.
Those diagnostic FPS values are not performance results. The standalone pixel
regression also proves all 4096 colors remain distinct and that water indices,
underwater masks and the reserved weapon color key preserve source behavior.

Measured level 1, true 320x200 at 1:1, 32 frames, 4 MB, 320x240 output:

| Fixed cycles | Scenario | Average FPS | Average frame | p95 frame |
| --- | --- | ---: | ---: | ---: |
| 40,000 | SB16 exercise, mouse enabled | 8.70 | 114.93 ms | 115.86 ms |
| 20,000 | SB Pro exercise, mouse enabled | 4.27 | 234.41 ms | 237.93 ms |

The first frame includes cold cache setup; the worst frame in that sample was
187.56 ms at 40,000 cycles and 381.49 ms at 20,000 cycles. At 40,000 cycles,
presentation averaged 7.62 ms and world rendering 100.64 ms.
Regular frames remain renderer-bound; increasing resolution is
expensive. These initial emulator profiles do not establish that every DX2/DX4
configuration will be fast enough. Both sound exercises had zero measured ring
underruns. All 16 levels passed title-to-game startup/render checks at 40,000 cycles
with sound and mouse enabled, 4 MB, EMS/UMB disabled and swapping disabled.
The full exercise also passed on the larger level 4. Menu/title captures remain
byte-identical after packing the font and storing the title in native strips.
These are short stationary checks, not a complete playthrough or a sustained
combat stress test. Recheck longer combat and transitions on actual hardware.

Cycles are not MHz. These profiles provide repeatable comparisons and do not
establish DX2/66 or DX4/100 hardware frame rates. `cycles=max` was used only for
early smoke testing and is not used for the reported performance results.

Windows Release and the single-threaded web Release also build. The existing
movement, settings, audio/filter, enemy sound/shotgun, HUD, shooting, shared ammo
and 96-case level 15 teleporter regressions pass through `tools/check_feedback.py`.

### Renderer optimization measurements

The first renderer pass removes software 64-bit division from wall texture
column selection using an exact 386 IDIV and signed remainder correction.
The near plane bounds inverse Z to at most 2^24; original signed 16-bit texture
coordinates keep the reduced quotient within 32 bits. The desktop expression
remains unchanged. Wall palettes also skip unused ARGB cache writes when RGB
raster expansion is disabled. Floor projection/camera setup now runs once per
polygon, including clipped subspans, and row distance is calculated once per span.
These preserve the existing `WallRoutine3.ChipMem.s` wall raster and `AB3DI.s
pastfloorbright` sampling and shading. No detail or gameplay settings are reduced.

Level 1 stationary starting view, 128 real-time frames, DOSBox Staging 0.82.2,
normal 486 core, 320x200, 16 MB for comparison with Duke shareware, SB16 effects
and mouse enabled, music off, host OpenGL output and normal process priority:

| Fixed cycles | Original | Arithmetic/setup pass | DOS tags removed |
|---|---:|---:|---:|
| 20,000 | 4.46 FPS | 4.59 FPS | 4.82 FPS |
| 40,000 | 8.95 FPS | 9.20 FPS | 9.67 FPS |

The arithmetic/setup pass gains about 3%, and removing DOS pixel tags adds
about 5%, for about 8% overall in this view. Tag removal also saves 125 KiB of
framebuffer allocations plus the weapon tag cache. A substantial gap to Duke's
starting view remains.
Optimizing the renderer translation units with `-O2` increased executable size
without improving this scene, so the size flags remain unchanged.
The DOS frontend diagnostic compares the hardware division against the former
64-bit expression at signed column boundaries and near-plane extremes.
Validation: 312 arithmetic cases passed on the DOS path; all 16 deterministic
level-start captures match the saved original-renderer reference byte-for-byte
with 4 MB. The 4 MB sound/mouse exercise passed save reloads, menus, weapon
color pixels, absent tag-buffer assertions and zero audio underruns. Windows
Release also builds.

### DOS cache audit

The optional indexed build still defaults to off. DOS already uses row-major
color buffers, no SSE2 raster path, and ordinary GCC pixel stores. Unsupported
prefetch instructions are not emitted for the forced i386/i486 target. Forcing
the prefetch macros off gave no material improvement and was not retained.

The retained DOS-only floor changes keep `BumpMap.s`/`AB3DI.s` Gouraud sampling:
prepare just the palette levels between the clamped gradient endpoints, skip
already prepared levels, and use a one-pixel loop for changing gradients.
Constant-shade and desktop loops retain their existing batching. With DJGPP
12.2 and the normal size/LTO flags, the indexed floor-span routine shrinks from
7,332 to 6,598 bytes (about 10%). This is the routine's code size, not a whole
renderer or frame-rate reduction.

Matched indexed starting-view profiles at 40,000 fixed cycles, normal 486 core,
16 MB, sound effects/mouse on and music off:

| Level | Frames | Before audit | After audit |
|---|---:|---:|---:|
| 1 | 128 | 10.57 FPS | 10.58 FPS |
| 6 | 64 | 10.20 FPS | 10.27 FPS |
| 15 | 64 | 16.59 FPS | 16.81 FPS |

The RGB565 level-1 profile moves from 9.67 to 9.69 FPS. These are small gains;
the level-1 change is effectively flat. A compact scalar wall loop reduced
code size but dropped the indexed level-1 result to 10.24 FPS, so it was
discarded. DOSBox timing does not establish actual 486 cache behavior; real
hardware measurements remain necessary for that comparison.
Both DOS packages were rebuilt. All 16 RGB565 captures remain byte-identical
to the original reference, and all 16 indexed center-world comparisons match
the same palette mapping. Both 4 MB sound/mouse exercises pass saves, menus
and weapon checks with zero audio underruns; Windows Release also builds.
