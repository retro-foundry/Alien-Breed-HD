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
The build command stages assets, regenerates the desktop sprite palette header
and a private native RGB565 version for DOS, compiles `ABHD.EXE`, sets a 128 KB stack, and creates
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
320x200 indexed VGA through SDL, with 1:1 world pixels and native UI sizes.
It needs no RGB565 VESA mode. The retained drawing canvas occupies 64,000 bytes;
the default RGB565 canvas uses 153,600 bytes at 320x240. SDL owns a separate
presentation surface. There is no
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
exact. UI source images retain their RGB565 pixels (or an exact grey palette
for disabled images) and acquire indexed SDL
caches when first used under a new palette. Palette changes invalidate weapon,
water, sky and UI caches; no UI quantization runs during steady-state blits.

Under the 128-frame level-1 comparison settings below, the 8-bit trial measures
5.27 FPS at 20,000 cycles and 10.57 FPS at 40,000, versus 4.82/9.67 FPS for the
then-current RGB565 build: about 9% faster in this starting view. Both builds run
with 4 MB; the performance comparison itself uses 16 MB consistently.

All 16 indexed level starts pass the 4 MB runner. RGB565 captures still match
the original reference byte-for-byte. The indexed world comparison checks
43,400 central world pixels per level against those RGB565 colors transformed
by the packaged palette map. Both comparisons request the same 320x240 mode;
the standard indexed package still defaults to 320x200 VGA. The comparison
excludes HUD regions. Run both variants and compare with:

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

The smallest RGB565 mode meeting the configured minimum size is selected.
The world fills that entire mode behind the keys and HUD, with 1:1 pixels and
no border. S3 in DOSBox uses 320x240. An existing INI requesting 320x200 also
fills its selected 320x240 mode. Larger native modes cost more CPU and RAM;
4 MB is validated at 320x240 RGB565 and 320x200 indexed VGA.
Plain 256-color VGA output requires the optional indexed build:
RGB565 retains every distinct original 12-bit color without inventing a palette.
DOS always uses the hardware full screen; the desktop window preference has no
effect. Errors and diagnostics are written beside the executable in `AB3D.LOG`.

`AB3D.INI` defaults to true 320x240 RGB565 world rendering, presented 1:1 without
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

- Engine uses `-march=i486 -mtune=i486`; the SDL dependency remains i386-safe.
  Engine code uses whole-program `-O3` and LTO, without fast-math;
  unused sections are discarded. SDL SIMD assembly is disabled.
  The DOS CPUID probe is disabled because an EFLAGS.ID probe under DPMI can
  falsely report support on a 486. The executable runs with a 486 instruction set.
- The desktop renderer's 32 MB RGB-to-Amiga lookup is omitted. Existing small
  tables calculate the same 12-bit values. Buffers are allocated after selecting
  the native video mode, and aligned allocation has a DJGPP implementation.
- The zone-offset lookup uses sorted offset/slot triples on DOS, retaining
  first-slot behavior for duplicate offsets. This replaces the desktop table
  that occupies twice the level-data size. Diagnostics compare against the
  original zoneAdds linear search, including adjacent offsets and misses.
- Automap reservation follows the map's line count instead of preallocating
  262,144 walls (about 6 MB including the hash). Existing dynamic growth retains
  discovered segments. Allocation failures stop with an error.
- DOS palette caches and framebuffers hold native RGB565, in row-major order.
  One retained renderer canvas copies to SDL's presentation surface in the same
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
  allocator receives a contiguous 1.25 MiB heap up front to reduce wasted 64 KB
  DPMI-zone padding. Separately allocated shot/point arrays are freed using the
  exact level-data extent, fixing leaks on reload and level transitions.
- Weapons retain their original scale and sampling. Each sampled animation
  frame is cached in a native SDL surface, using a color key outside
  the 4096 source colors. Empty margins are cropped and only the selected
  weapon is retained. It decodes directly from the original PTR/WAD/palette
  without a temporary RGBA image. One buffer, sized from the largest authored
  frame, is reserved before loading a level and reused on animation changes.
  At 320x240 it occupies 15,312 bytes. Original 32-color weapon indices convert
  exactly to RGB565 through SDL; no color quantization is introduced. SDL's
  additional RLE copy is omitted. The former opaque weapon tag runs and
  their cache are omitted; the SDL surface already retains the cutout mask.
- UI draws synchronously with SDL3 surface fills/blits, without an SDL renderer
  queue or OpenGL. PNG fonts are converted offline to native RGB565 with their
  authored cutout masks. The font packs the original sampled 8x13 glyphs
  without unused cell padding; glyph size and layout stay identical. The DOS
  executable and SDL omit PNG/JPEG decoders.
  The original planar title streams a plane at a time into its final surface
  and is released before a level loads. Its native SDL strips avoid requiring
  a contiguous 160 KB allocation when returning to the menu. Opaque
  color-key blits preserve their cutout shapes without alpha blending. UI
  source bitmaps omit SDL's duplicate RLE storage to leave room for live caches.
  Uncollected keys and disabled labels use cached grey images; collected keys
  retain their colors. Grey images store the original dim transform exactly
  in a one-byte image with one shared immutable grey palette, index 255 reserved
  for cutouts and no RLE duplicate. This leaves room for the full-height view
  during level-4 reloads.
  HUD fonts, strips and all four key images are prepared before loading a
  level; bitmap invalidation follows sprite assets rather than key ownership.
  Key-to-frame lookup shares one authored object-table scan per HUD draw.
  HUD backgrounds and map lines use solid colors. Menus
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
  path table. Original geometry/sprite/sample formats are retained; walls, UI
  images and music have offline derivatives. Shared marine sprites and
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
preserves the existing full-save versions without holding two complete levels in RAM.
Truncated payloads are rejected before the live state changes. Desktop/web
continue to stage their saves in memory.
Saving discovered map walls also streams the existing disk records through a
64-record stack buffer, avoiding another live map allocation.

The runner defaults to `memsize=4`, `ems=false`, `umb=false` and `CWSDPMI -s-`.
`[DOSMEM]` reports XMS pages; that figure excludes conventional banks and is
not a whole-machine RAM measurement. The emulator's 4 MB configuration is the
whole-machine limit. The 128 KB stack is included. This is not a 2 MB build.
DOS retains one drawing canvas and a separate SDL presentation surface. The
two renderer color pointers alias that canvas; water samples the active image.
Weapon comparisons borrow SDL's independent pixels after capture, rather than
retaining another complete world buffer solely for diagnostics.

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
The emulator window is hidden by default on Windows, and host mouse capture is
disabled for repeatable benchmarks. `--mouse` still exercises the guest mouse
driver each frame. Use `--visible` for interactive checks. The default height
comes from the packaged INI; `--height` requests a minimum video height.
Full-surface builds must report a 1:1 viewport at (0,0), with renderer and
capture dimensions equal to the selected video mode.
`--exercise` additionally checks aligned allocation, navigates New Game through
SDL key events, streams title music, captures `MENU.BMP`, writes/reads all five
autosave headers and the quicksave, restores full saves eight times before and eight times after
gameplay, verifies level/object/map payload hashes, rejects truncated payloads,
and mixes shotgun/flame effects. It captures
the map, numeric HUD, grey/collected keys and options screen, verifies native UI blit dimensions
and exact cutout masks in opaque grey images,
checks all 16 level narratives for lost glyphs, and compares cached scaled
weapon color pixels against the original CPU draw routine, both with the level
still resident and after returning to menus, and verifies that
both DOS tag buffer pointers remain null. Save tests
write only inside that isolated test directory.
Screen-fit diagnostics check every narrative page and the main, options,
autosave, confirmation and pause menus at both 320x200 and 320x240. HUD
diagnostics draw empty, one-unit and full bars/numbers, all four collected
keys and their underlines, the four-digit FPS counter, crosshair and map;
every UI image and fill is checked against the output bounds. `HUD200.BMP`,
`BAR200.BMP` and their 240-line counterparts capture those layouts.

For performance-only pixel parity, preserve a separate native build from
before the optimization, then compare both at the same native dimensions:

```powershell
python tools/check_dos_pixels.py --dosbox <dosbox.exe> --reference-build <preserved-native-build> --height 240
```

The historical pixel oracle instead builds the original v1.0.2 renderer from commit
`487c1aae41ae3443b8ea118b4d14550647597c61` in an ignored snapshot, with only
DOS memory/API adaptations, and compares complete captures against the native
pipeline using a 16 MB reference profile. Its color conversion exists only in
that test snapshot. This is a historical behavior comparison and currently
differs from the unmodified native baseline as described below:

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
At that stage, optimizing the renderer translation units with `-O2` increased
executable size without improving this scene, so the size flags were retained.
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

### Performance target and initial renderer pass

The requested target is approximately **3x the pre-change FPS**, preserving
the existing visuals on a 486 DX2/DX4. **That target remains open.** This pass
initially improved level 1 by about 23%; it did not establish acceptable hardware speed.

The comparison baseline is the native DOS executable built from `de9d543`,
preserved before editing. Both builds use DOSBox Staging 0.82.2, normal 486
core, fixed cycles, 4 MB, EMS/UMB/swapping disabled, 320x200 at 1:1, SB16 sound
effects and mouse enabled, music off, and real-time logic ticks. Runs are
sequential. These figures use the final normal build, without detailed
renderer instrumentation:

| Cycles | Level | Frames | Before FPS | After FPS | Before p95 | After p95 |
|---|---|---:|---:|---:|---:|---:|
| 20,000 | 1 | 128 | 4.69 | 5.79 | 213.00 ms | 172.30 ms |
| 40,000 | 1 | 128 | 9.41 | 11.60 | 106.21 ms | 86.04 ms |
| 40,000 | 6 | 64 | 9.25 | 10.74 | 110.14 ms | 94.94 ms |
| 40,000 | 15 | 64 | 14.99 | 17.54 | 70.65 ms | 60.41 ms |

At 40,000 cycles, rendering falls from 95.70 to 75.70 ms/frame; presentation
remains 6.86 ms. The corresponding 3x target is 28.23 FPS, or 35.42 ms for the
entire frame. Substantial world-rendering work remains before that is met.
The stationary views remain short comparisons, not a moving/combat playthrough
or real-hardware measurement.

Retained changes preserve `WallRoutine3.ChipMem.s ScreenWallstripdraw`,
`AB3DI.s pastfloorbright` and `ObjDraw3.ChipRam.s ObjDraw` sampling/order:

- DOS already renders in screen order with `AB3D_CW_COL_MAJOR=0`. There is no
  OpenGL rotation or transpose in its presentation path.
- RGB565 wall/floor palettes are prepared in their owned RAM at load time.
  Drawing reads native shade tables directly; all 4096 source colors remain
  distinct. Indexed palettes retain their existing per-level mapping.
- Wall words expand once from three packed five-bit texels into three byte
  indices. Word order is retained because level-authored VALSHIFT values can
  interpret the same asset at different column heights. This costs 73,840
  extra heap bytes for the supplied wall set; disk assets remain unchanged.
- Constant-depth RGB565 walls draw in horizontal rows, including exact side
  and cap extensions and original column-clip registration order. Repeated
  sampled rows use a copy. Perspective-varying walls use a small column kernel.
- DOS floor kernels retain full wrapping 16.16 UVs and Gouraud levels. Exact
  axis-aligned spans avoid the constant UV coordinate's per-pixel work. Visible
  edge walks use 32 bits only after proving their endpoints and step fit;
  wider projections retain the existing 64-bit walk.
- Exact 64/32 projection/interpolation division uses 386 DIV, including a
  high-word step for wide quotients. Per-frame object candidates retain the
  original slot order while sharing camera/point rejects between rooms.

A 128 KiB shaded-column cache was slower (11.51 FPS in its matched level-1
trial) and was removed. Whole-program `-O3` also failed that pass's 4 MB startup
check, so it retained size optimization. The retained-canvas change below
allows the stronger optimization in the current build. No resolution, shading, texture
sampling, input cadence or gameplay timing cuts were introduced.

All 16 RGB565 starts match the preserved native baseline byte-for-byte, and
all 16 indexed center-world comparisons match the same source palette map.
Both variants pass the 4 MB SB16/mouse exercises on levels 1 and 4, including
save reloads, menus, original weapon-color comparisons and zero audio
underruns. Windows Release and `tools/check_feedback.py` pass.

The historical v1.0.2 whole-frame oracle is not the current renderer baseline:
the unmodified `de9d543` executable also differs in 810 central level-1
pixels. Its frozen vector-object header now stays paired with its source,
fixing a reference-build API mismatch; the historical image difference remains.
Use a separately preserved native build to check performance-only changes:

```powershell
python tools/check_dos_pixels.py --dosbox <dosbox.exe> --reference-build <preserved-native-build>
python tools/check_dos_indexed.py --dosbox <dosbox.exe>
python tools/check_dos_raster.py --dosbox <dosbox.exe>
```

The full-surface performance acceptance command below remains unmet.
`--min-fps` rejects fixed diagnostic ticks and fails below the supplied real-time average.
`--render-profile` separately reports wall, floor, sprite, object-pass and
backdrop costs; its counter/statistics overhead must not be included in the
before/after FPS comparison.

The standalone 4 MB raster check exercises the actual DJGPP/i486 kernels:
4,215 signed division cases, all 4096 prepared palette colors, 8,192 floor
spans, 4,096 edge walks, every packed wall word and 192 wall row/column rasters.

### Further DOS renderer work

The follow-up keeps exact colors, texture samples, clipping and object order.
The approximately 3x target remains open. DOS still draws directly in screen
order; no rotated framebuffer or OpenGL transpose is introduced.

- RGB565 wall columns, wall rows and floor spans use small 486 assembly loops.
  Axis-aligned floor spans process eight pixels per iteration with exact tails;
  general UVs retain their full 32-bit wrapping. Gouraud shade runs stop at
  exact fixed-point boundaries.
- Native floor spans use the prepared palette directly, avoiding the desktop
  cache setup. Floor and sky edge arrays initialize only their clipped row
  band. Empty edges skip division; reversed edges reuse the same exact slope.
- Wall clipping handles an empty or overlapping first span directly while
  retaining the original two-span merge and replacement policy.
- Sky source-column offsets are shared across spans, keyed by the view's pan,
  projection and asset layout. Sampling and endian conversion remain exact.
- Per-frame room masks select objects by their authored draw groups and exact
  adjacent source zones. Original slot order, billboard spill, polygon rejects
  and equal-depth sorting remain intact. Both live projectile lists are built
  once per frame instead of scanning 120 slots for every room.
- In that pass, only `renderer.c` used `-O2`; the rest retained `-Os` and LTO. Reserving
  1.25 MiB rather than 1.5 MiB leaves space for the larger optimized text in
  the same 4 MB runtime budget. A floor-row math cache gave less than 1% and
  was removed.

`tools/check_dos.py --turn` holds the real right-turn key through the normal
input path and verifies that the camera moves. For exact moving-view parity,
use the same diagnostic in the preserved original build and fixed ticks:

```powershell
python tools/check_dos_pixels.py --dosbox <dosbox.exe> --reference-build <original-with-turn-diagnostic> --levels 1 4 6 15 --frames 48 --turn
```

These comparisons use the original `de9d543` renderer with only the camera
diagnostic added to `main.c`. They compare complete captures after 48 frames
with fixed logic ticks, including oblique sampling and changed room visibility.
Real-time turning profiles also work, but faster builds advance farther in
the same frame count, so their views and elapsed game time differ.

Before the full-surface change, measurements used the same normal-486, 4 MB, RGB565,
sound/mouse-on settings as the native `de9d543` baseline above. These runs
were sequential, with detailed instrumentation off:

| Cycles | Level | Frames | Original FPS | Initial pass FPS | Current FPS | Original p95 | Current p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| 20,000 | 1 | 128 | 4.69 | 5.79 | 8.29 | 213.00 ms | 117.91 ms |
| 40,000 | 1 | 128 | 9.41 | 11.60 | 16.61 | 106.21 ms | 58.86 ms |
| 40,000 | 6 | 64 | 9.25 | 10.74 | 15.25 | 110.14 ms | 64.74 ms |
| 40,000 | 15 | 64 | 14.99 | 17.54 | 23.51 | 70.65 ms | 42.30 ms |

Level 1 gains about 77% over the original, or 43% over the initial pass.
Normal rendering falls from 95.70 to 48.75 ms/frame. The complete frame is
60.20 ms, leaving another 24.78 ms to remove for the 35.42 ms / 28.23 FPS
target. The real-time `--min-fps 28.23` acceptance check **still fails**.
In a separate instrumented run, walls take 23.8 ms (13.9 ms in pixel kernels),
floor spans 10.3 ms, floor geometry 3.4 ms, object passes 5.8 ms including
sprites, and backdrop work 1.9 ms. Instrumented costs include counter overhead.

Validation on the final packages: all 16 complete RGB565 level-start captures
are byte-identical to the original native baseline; four 48-frame turning
captures also match exactly. All 16 indexed center-world comparisons pass.
Both packages pass the 4 MB SB16/mouse exercises on levels 1 and 4, including
eight quickloads, menus, weapon frames and zero audio underruns. The DJGPP
raster/object tests, Windows Release build and feedback regression suite pass.
The raster tests also cover signed projection half-pixel boundaries, clipped
edge initialization canaries and 2,048 room object lists with both shot pools.
This remains emulator evidence; real 486 hardware has not been measured.

### Full native view and latest measurements

The DOS renderer now initializes after video-mode selection and draws the
entire native surface behind the HUD and keys. A 320x200 request on S3 still
selects 320x240 RGB565, but now renders all 240 rows instead of centering a
200-row image. Presentation neither clears a border nor rotates or scales
the image. Indexed VGA similarly fills its native 320x200 surface.

Additional exact wall work shares a palette across flat wall rows, unrolls
eight row samples at a time and addresses eight vertical samples directly in
320-pixel modes. Mixed-palette rows, other widths and remaining samples keep
their exact paths. A bounded native projection loop retains full 64-bit
interpolants; repeated projection factors are cached by their actual inputs.
No texture samples, shading, game ticks or input polls are omitted.

The comparison below renders **320x240 in every build**, at 40,000 fixed
cycles, normal 486 core, 4 MB, SB16 and guest mouse enabled, 128 real-time
frames on level 1. Runs are sequential, with detailed profiling disabled.

| Build | FPS | Render time | Average frame | p95 frame |
|---|---:|---:|---:|---:|
| Original `de9d543` | 8.60 | 105.00 ms | 116.23 ms | 116.21 ms |
| Previous renderer pass | 15.37 | 52.98 ms | 65.06 ms | 63.74 ms |
| Full native view, wall loops and prepared UI | 17.04 | 48.33 ms | 58.69 ms | 58.66 ms |
| Exact floor reuse, native palettes and prepared UI blits | 17.97 | 45.34 ms | 55.65 ms | 55.98 ms |
| Retained canvas, contiguous wall streams and `-O3` | 18.87 | 43.05 ms | 53.00 ms | 53.31 ms |
| Exact wall wrapping | 19.10 | 42.40 ms | 52.35 ms | 52.65 ms |
| Shared projections and contiguous presentation copy | 20.66 | 41.43 ms | 48.40 ms | 49.03 ms |
| Repeated automap wall updates | 20.97 | 40.73 ms | 47.70 ms | 48.31 ms |
| Frame lookup for sprite spill order | 21.04 | 40.56 ms | 47.53 ms | 48.14 ms |
| Connected-room cache slots without pointer mixing | 22.28 | 37.92 ms | 44.89 ms | 45.42 ms |
| Inline DOS wall clip registration | 22.33 | 37.83 ms | 44.79 ms | 45.32 ms |
| Exact 32-bit texture walk for flat walls | 22.45 | 37.58 ms | 44.54 ms | 45.07 ms |
| Wall clip guards once per native segment | 22.58 | 37.33 ms | 44.29 ms | 44.82 ms |
| Reuse exact wall endpoint values | 22.81 | 36.87 ms | 43.83 ms | 44.35 ms |
| Carry-based projection rounding | 22.83 | 36.83 ms | 43.80 ms | 44.31 ms |
| Clip native columns before texture/depth work | 22.87 | 36.76 ms | 43.72 ms | 44.23 ms |
| Shade wall core and side pixels together | 23.06 | 36.40 ms | 43.36 ms | 43.86 ms |
| Classify identical darkest-shade floor rows at load | 23.19 | 36.15 ms | 43.12 ms | 43.62 ms |

This is a 2.70x increase over the original at the same rendering size.
The approximately 3x target remains unmet: it requires 25.80 FPS, or 38.76 ms
per frame, for this full-height baseline. The earlier 28.23 FPS target and
320x200 tables above describe the earlier, smaller view.

```powershell
python tools/check_dos.py --dosbox <dosbox.exe> --height 240 --cycles 40000 --frames 128 --mouse --min-fps 25.80
```

The raster regression covers both native and general wall pitches, zero-length
columns, all eight unroll tails, mixed palettes and widths 1..33. Grey UI checks
also verify each opaque palette color against the original dim transform.
Diagnostic BMP captures stream through bounded stack row/file buffers, so a
live 4 MB scene does not need additional heap space merely to save its image.
HUD diagnostic state also uses the reserved stack. Weapon cache invalidation
follows shared gun assets, view dimensions, animation and palette changes;
level geometry reloads retain it. All 24 weapon frames are checked with the
level still loaded, including the larger level 4.

Final package captures match the preserved original byte-for-byte on all 16
level starts and four 48-frame turning views. An old 320x200 INI produces the
same complete native capture as 320x240, including projection and aiming after
the shared-asset settings reload. All 16 indexed sampling comparisons pass.
The 4 MB memory exercises pass on native levels 1, 4 and 15 and indexed levels
1 and 4; these repeat full saves/quickloads before and after gameplay, test all
24 weapons while the level remains loaded, then check menus and palette changes.
The level-4 native real-time exercise also passes at 40,000 cycles with zero
audio underruns. Additional memory checks use 200,000 cycles and fixed logic
ticks to shorten validation; their diagnostic FPS is not a performance result.
Windows Release, the feedback suite, DJGPP raster/object checks and ZIP CRC,
executable and default-INI checks pass. Real hardware remains unmeasured.

### Conversion and shading within 4 MB

RGB565 sprite shade tables are now prepared during packaging. The executable
contains the same 17,280 bytes of authored shades, in native word order;
sprites select those blocks directly instead of rebuilding a 32-color table
on each draw. The raster check compares all 8,640 words with their original
`.pal` values. Sky words also acquire native byte order during the existing
in-place load conversion. The polygon renderer uses its owned palette
directly, removing a duplicate 7,680-byte allocation.

Wall and floor shades already use the original asset tables in native format.
A general 4096-color x 64-shade x 2-byte table would cost 512 KiB and is not
needed for these paths. A tested sixteen-block sprite cache added 1,088 bytes
without a measurable improvement; direct preconversion replaces it.
Weapon and disabled-UI images retain compact source indices with small SDL
palette maps, preserving the live-level 4 MB budget.

The larger speed gain in this pass comes from exact floor sampling. Equal
Gouraud endpoints dispatch directly to the constant shade path. Axis spans
stepping at most half a texel reuse the current color until the original
16.16 coordinate carries or borrows into the next texel. Every output pixel,
UV wrap and shade transition is retained. Other steps keep the original
sampling path. The standalone regression covers both signs and axes, the
half-texel threshold and all eight unroll tails.

With the same 320x240, 40,000-cycle, 4 MB SB16/mouse settings, 64-frame
real-time runs improve level 6 from 15.45 to 16.45 FPS and level 15 from
24.47 to 25.87 FPS. Level 1 improves from 17.04 to 17.97 FPS over 128 frames.
Palette preconversion itself has a small effect in these views; most of the
gain comes from floor reuse and eight-sample wall/floor loops.

The DOS test runners set DOSBox `nosound=true`, muting host output while
retaining guest Sound Blaster emulation, game mixing and underrun checks.
SDL's small UI palette-to-RGB565 blit tables are also prepared with the HUD,
before loading a level. This avoids first-use allocation failures in a full,
fragmented heap. The warm-up restores the target pixel and clip rectangle.

The local GloomAmiga DOS renderer is a useful comparison: it defaults to
320x200 INDEX8, pre-shades complete floor/roof and opaque wall texture banks,
and removes ray hits hidden by an opaque wall. Its two 128x128 flat banks
alone use 512 KiB for sixteen shades; its build script sets an 8 MiB stack
reservation (`minstack=8M`). This port's native 320x240 RGB565 frame
contains 153,600 bytes versus Gloom's 64,000-byte indexed frame. Copying its
complete shaded-bank strategy would compete with the 4 MB live-level budget.
Before UI blit warm-up, a 4,480-byte shaded-row trial preserved samples but slowed
levels 1/6/15 to 17.11/15.51/24.29 FPS versus 17.86/16.24/25.59; it was removed.
The palette preconversions above remain independent of that trial.

### Retained DOS canvas and prepared wall streams

DOS now draws into the retained native image directly. The previous path
copied the last complete image into a second color buffer before every frame.
All world and water background reads use the active image, and presentation is
synchronous, so retaining that image preserves the same initial pixels and
draw order. The change removes one full-frame copy and a 150 KiB allocation at
320x240 (62.5 KiB at indexed 320x200). Desktop buffering remains separate.

The original-versus-cached weapon comparison now uses SDL's independent display
pixels as scratch. It compares every row, respects surface pitch and rejects
aliased scratch. All 24 weapon frames are still tested with a live level.

`tools/dos_wall.py` prepares the original `WallRoutine3.ChipMem.s` three five-bit
streams into contiguous byte planes during packaging. Each plane retains the
original word order, so every level-authored VALSHIFT continues to address the
same texel. Vertical kernels no longer multiply the sample index by three.
The 73,840 extra texel bytes from the earlier unpacking remain unchanged; the
fourteen walls add 168 metadata bytes. Original Amiga files are untouched.
DOS neither decompresses nor expands walls at startup. RGB565 wall palettes
are also prepared offline; indexed caches keep their original palette words.
The versioned ABW1 cache records its format and plane length, which the loader
validates before exposing the palette and texels. Update the executable and
DATA together. The DJGPP regression tests the generated cache against all
65,536 packed words and checks the host palette converter on all 4,096 colors.

The canvas saving permits whole-program `-O3` with the existing i486 target,
LTO and 128 KiB stack. No fast-math option is enabled. The larger code consumes
part of the recovered memory; both packages still pass the 4 MB no-swap
exercises. Renderer-only `-O3` measured 18.76/17.11/27.13 FPS on levels 1/6/15;
whole-program `-O2` with that renderer measured 18.77/17.02/27.19. The selected
whole-program `-O3` build measures 18.87/17.11/27.56, using 128/64/64 real-time
frames at the common 320x240, 40,000-cycle, SB16/mouse-on settings.

All sixteen native starts and six 48-frame turns (levels 1, 2, 4, 6, 15 and 16)
match the preserved original captures. All sixteen complete indexed 320x200
starts also match the prior indexed build. The separate indexed 320x240
sampling comparison checks the original per-level color mapping. Native 4 MB
exercises pass on levels 1, 4 and 15; indexed exercises pass on levels 1 and 4.
Tests remain muted at the host while retaining guest audio emulation.

An adjacent-column copy trial was correct but slower (18.13/16.55/26.17 versus
18.25/16.59/26.32 for contiguous streams alone) and was removed. A separate
gprof sampling experiment produced no samples, then hit a DOSBox page fault
when instrumenting startup. Its executable is confined to an ignored build;
the normal packages use the existing frame and renderer timers.

### Unrotated floor loop and repeated object projections

The main thread and workers now reserve column-floor scratch only when that
renderer is enabled and the framebuffer is column-major. DOS's row-major
path never used these allocations. At 320x200/240, omitting the row records,
column bounds and column walk tables removes 24,588 requested heap bytes.

A 6,392-byte DOS billboard cache stores each object/shot's world dimensions,
screen X and estimated width for the current frame. Connected-room passes
reuse those values while retaining each room's original clipping and draw
order. Validity resets after rotating object points, before drawing any room;
level changes, moving objects and camera turns cannot reuse an older frame.
The final billboard draw also reuses the dimensions and screen X. Its room
projection, shading and actual draw-size calculations remain unchanged.

Native floor spans with both UV steps nonzero now process eight pixels per
iteration, as the axis-aligned loops already did. Every sample still uses the
original 32-bit wrapping accumulators. This adds no texture or shade bank.
The standalone DJGPP regression checks all unrolled tails against the original
expressions within its 8,192 floor-span cases.

At the same 40,000-cycle normal 486, 4 MB, SB16/mouse-on settings, the final
package measures 18.87 FPS on level 1, 17.32 on level 6 and 27.60 on level 15
(128/64/64 real-time frames). The earlier canvas build measured
18.87/17.11/27.56. A separate 128-frame level-one turning run improves from
24.80 before these changes to 25.28 FPS. Different views have different work;
this turning result does not meet the stationary baseline's 25.80 FPS target.
Level one remains 2.19x the original full-height baseline.

All sixteen native starts and six 48-frame turns match the original native
captures. All sixteen complete indexed 320x200 starts match the preserved
indexed build. Both packages pass full 4 MB exercises on levels 1 and 4,
including all 24 weapon frames with a live level. Native level 4 also passes
the full exercise at 40,000 cycles. Windows Release, feedback and DOS raster
checks pass. Host output remains muted with guest audio active. Final package
hashes and measurements are in `build/dos/gloom-comparison-performance.json`.

### Exact wall wrapping without a per-pixel mask

The native 320-pixel wall column loop now shifts the authored wrapping bits
to the top of a 32-bit accumulator. For masks 15/31/63/127/255, each sample
needs one right shift instead of a right shift followed by a memory-operand
AND. Shifting the initial coordinate and step preserves unsigned wrapping
after every addition, including negative steps and texture phase. Eight-pixel
groups and their tails use that representation. Other masks retain an
unrolled general loop; other pitches and short columns keep the general path.
No texture or working buffers are added, and no palette samples change.

The same native package now measures 19.10/17.47/28.08 FPS on levels 1/6/15,
versus 18.87/17.32/27.60 before this pass. The separate level-one turning run
improves from 25.28 to 26.25 FPS. These are 128/64/64/128 real-time frames at
the established 40,000-cycle normal 486, 4 MB, SB16/mouse-on settings. The
stationary level-one result is 2.22x the original 8.60 FPS; its 25.80 FPS
target remains unmet. Measurements, package hashes and validation records
are in `build/dos/wall-wrap-performance.json`.

The DOS raster regression now includes 16-row walls, noncanonical masks and
additional signed fractional floor steps. Complete native captures on all
sixteen starts and the six 48-frame turns remain the fidelity checks. Native
4 MB exercises cover levels 1, 4 and 15, including all 24 live-level weapon
frames; level 4 is also exercised at the normal 40,000 cycles. The indexed
executable is byte-identical to the previously validated package.

Exact floor-run fills and solid-shade classification trials were slower and
were removed. An isolated flat profiler started successfully after level
loading, but delivered zero SIGPROF callbacks despite an expired timer; its
histogram is not used as performance evidence. Release builds contain none
of that profiler setup. All emulator runs remain muted at the host.

### Shared projections and contiguous presentation copy

DOS reuses the unused 8,192-byte legacy 96-column projection array for exact
screen X coordinates. Shared level vertices reuse their projection within the
current rotation frame. Inactive, unrotated or stale entries use the original
calculation; near-clipped endpoints also retain it. Projections beyond signed
32-bit range remain fully 64-bit and uncached. The live-level diagnostic checks
432 input/validity combinations, including cache hits, stale entries and wide
results. No additional per-vertex allocation is required.

A single-entry Y quotient cache retains the full numerator and depth, with the
screen centre applied afterwards. Water projection cancels equal coordinate
scales before dividing, and vertical refraction cancels a common factor before
its division. These preserve the original signed rounding. The DJGPP raster
check compares 35,169 water projection/refraction cases with the original
expanded expressions, and checks changes to both Y-cache inputs.

Floor setup no longer clears unused brightness rows. Every valid edge bound
writes its brightness at the same time, and invalid spans are rejected before
reading it. Left/right bounds still initialize normally. Polygon edge loops
also advance their final vertex without a variable remainder operation.

The largest gain comes from presentation. For a complete, unclipped RGB565
view with contiguous SDL pitch, DOS locks the surface and copies the image in
one `memcpy`, replacing SDL's row-by-row blit. Other sizes, pitches and clip
rectangles retain the SDL path. Presentation falls from 6.77 to 3.81 ms in the
level-one run. The retained world canvas and separate UI surface still preserve
water background reads and HUD layering.

Fresh public-package timings are 20.66/18.92/31.01 FPS on levels 1/6/15 and
28.68 FPS while turning on level 1, using the same 128/64/64/128 real-time
frames, 40,000 cycles, 4 MB and SB16/mouse settings. The stationary level-one
result is 2.40x the original 8.60 FPS; the 25.80 FPS target remains unmet.
Measurements, package hashes and validation records are in
`build/dos/projection-performance.json`.

All sixteen native starts and six 48-frame turns match the original captures.
All sixteen indexed starts and the same six turns match the preserved indexed
package. Native 4 MB exercises pass on levels 1, 4 and 15; indexed exercises
pass on levels 1 and 4. They include the projection diagnostic, all 24 live-level
weapon frames, menus, save/load and zero audio underruns. Native level 4 also
passes at 40,000 cycles. An old 200-line INI produces the same full 240-line
native capture. Windows Release, feedback and DOS raster checks pass. All DOS
test output remains muted at the host, with guest audio emulation enabled.

Extending the X cache into wall endpoints was slightly slower in the matched
views and was removed. Full-frame pixels and authored shade values remain
unchanged; this pass adds no texture or shade bank.

### Repeated automap wall updates

DOS keeps a 1 KiB table of recently used automap hash buckets. Each hit reads
the current bucket and verifies the graphics offset and both complete
endpoints, accepting either drawing direction. A different segment, empty
bucket, rebuilt table or invalid index uses the existing hash lookup. The
cache retains indices rather than pointers into reallocatable wall storage.
Door/key colors, zone hints and internal-line metadata still refresh through
the same function on every discovered-wall update. Discovery order, hash keys
and the save format remain unchanged.

On levels 1 and 4, complete map captures and all six saves' level/map payloads
match the preserved build before this cache. The binary comparison excludes
the GameState snapshot, which contains runtime pointers; the existing
save/load diagnostics check its restoration. Save headers, complete level
chunks and discovered-wall records match exactly.
The comparison records are in `build/dos/automap-save-parity.json`.

Final public-package timings are 20.97/19.14/31.25 FPS on levels 1/6/15 and
28.95 FPS while turning on level 1, at the established 40,000-cycle, 4 MB,
SB16/mouse-on settings with 128/64/64/128 real-time frames. The stationary
result is 2.44x the original; the 25.80 FPS target remains unmet. Measurements
and package hashes are in `build/dos/automap-performance.json`.

Both final packages pass all sixteen level-start comparisons and six
48-frame turning comparisons. Native 4 MB exercises pass on levels 1, 4 and
15; indexed exercises pass on levels 1 and 4. Native level 4 also passes at
40,000 cycles. Windows Release and feedback checks pass. The raster kernels
are byte-identical to those covered by the preceding DOS raster regression.
Host test output remains muted with guest audio active. A short-row copy
experiment made no meaningful FPS difference and was removed; the isolated
automap timing probe is also absent from the final sources.

### Frame lookup for sprite spill order

DOS builds a 512-byte lookup of each room's first position in the current
draw order. Sprite spill checks reuse it during that world frame. It includes
rooms rejected by the clip prepass, preserving the existing ordering rules;
the prepass's visible-room lookup cannot substitute for it. Duplicate room
IDs retain their first match. Other GameState owners, IDs outside 0..255 and
calls outside the active world frame retain the original linear search.
The lookup rebuilds each frame and is disabled after world drawing.

The live-level diagnostic compares 429 cases with the original search,
covering repeated hits, changing lists, duplicate/missing/negative/large IDs,
clamped counts and other owners. The DOS exercise requires this diagnostic
in both native and indexed builds.

Public-package timings are 21.04/19.43/31.27 FPS on levels 1/6/15 and
28.95 FPS while turning on level 1, using the same 128/64/64/128 real-time
frames, 40,000 cycles, 4 MB and SB16/mouse settings. The previous build
measured 20.97/19.14/31.25/28.95. The 25.80 FPS target remains unmet.
Package hashes, measurements and validation records are in
`build/dos/zone-order-performance.json`.

All sixteen native and indexed starts and six 48-frame turns in each build
match their preserved references. Native 4 MB exercises pass on levels 1,
4 and 15, including a normal-cycle level-4 exercise; indexed exercises pass
on levels 1 and 4. Map captures and saved level/map payloads remain identical
on levels 1 and 4. Windows Release and feedback checks pass. Tests remain
muted at the host with guest audio enabled.

### Connected-room cache collisions

The DOS connected-room cache now chooses its existing 128 slots directly
from room ID instead of mixing the ID with the LevelState pointer. Authored
visible IDs tend to cluster. In level one's saved draw order, the old hash
put rooms 6 and 17 in the same slot, repeatedly replacing their adjacency
lists and repeating the complete reverse-link scan. Direct slots separate
those IDs. The entry still checks the complete level/data/zone/floor-line
identity before using a cached result; room traversal and list order are
unchanged. Desktop hashing is unchanged, and no storage is added.

At the established 40,000-cycle, 4 MB, SB16/mouse-on settings, public-package
timings are 22.28/21.65/31.27 FPS on levels 1/6/15 and 29.31 FPS while turning
on level 1, over 128/64/64/128 real-time frames. The previous build measured
21.04/19.43/31.27/28.95. The stationary baseline is now 2.59x the original
8.60 FPS; its 25.80 FPS target remains unmet.

A separate matched 32-frame comparison covers all sixteen starting views.
Its results are in `build/dos/connected-all-level-performance.json`, with
individual raw logs preserved. This shorter sample is separate from the
128-frame target measurement. Package hashes and validation records are in
`build/dos/connected-slot-performance.json`.

The short sample's largest gain is level 13, from 26.47 to 33.01 FPS. Small
decreases on levels 5 and 7 were checked with matched 128-frame runs: level 5
improves from 18.51 to 18.66 FPS, while level 7 measures 68.97 versus 68.94
FPS (both round to 14.50 ms per frame). These longer follow-ups are also
recorded with the comparison results.

All sixteen native and indexed starts and six 48-frame turns in each build
match the preserved captures. Before reusing each reference capture, the
runner verifies the logged level/frame count, fixed-tick and turn settings,
executable, CWSDPMI and every mapped asset against the frozen reference game.
Capture hashes and those input checks are recorded in
`build/dos/connected-slot-pixel-evidence.json`. Native 4 MB exercises pass
on levels 1, 4 and 15, including normal-cycle level 4; indexed exercises pass
on levels 1 and 4. All runs remain muted with guest audio active for the
performance and exercise checks.

### Inline DOS wall clip registration

DOS now forces the wall clip-registration helper into its callers. The
guard, span merge, depth replacement and boundary-extension rules remain the
same. This removes the generic helper call from approximately 1,100 columns
per level-one frame and permits optimization using the caller's values.
The native executable adds 5,632 text bytes; data and BSS sizes are unchanged.
Both packages still pass the full 4 MB exercises.

Matched public-package timings improve from 22.28/21.65/31.27/29.31 to
22.33/21.66/31.39/29.53 FPS for the established level 1/6/15/turning cases,
using 128/64/64/128 real-time frames at 40,000 cycles with SB16 and mouse.
The stationary result is 2.60x the original; the 25.80 FPS target remains
unmet. Measurements, code sizes and package hashes are in
`build/dos/clip-inline-performance.json`.

All sixteen native/indexed starts and six 48-frame turns in each build
match the verified reference captures. Native live-level 4 MB exercises
pass on levels 1, 4 and 15, including normal-cycle level 4; indexed exercises
pass on levels 1 and 4. Map captures and all six saves' level/map payloads
remain identical on levels 1 and 4. The non-DOS declaration and function body
are unchanged. All DOS tests remain muted with guest audio active where
requested. A combined zone-stream query trial saved only about 0.07 ms per
level-one frame and did not improve every measured view; it was removed.

### Exact 32-bit wall texture interpolation

The DOS binary already uses the 32-bit DJGPP/i486 ABI: `int`, `long` and
pointers occupy four bytes. Explicit `int64_t` values still occupy eight
bytes and can require multiple instructions or software division. Some wide
intermediates remain necessary for fixed-point products and near clipping.
The existing exact 64/32 helper already uses native 386/486 divide operations.

For constant-depth wall spans of at least sixteen columns, texture selection
now retains the original interpolation fraction and division remainder in
32-bit fields. Their carries reproduce the original floor division, including
negative coordinates and steps. Setup keeps the wide phase, and steps whose
integer part exceeds signed 32 bits retain the original division loop. Texture
pointers are prepared before the palette/clip loop, without changing draw
order, shade selection or samples. The walk uses 28 bytes of local state and
adds no heap allocation.

The DJGPP raster check compares 491,736 samples with the original expanded
expression and independent division at quotient boundaries. It also verifies
the wide-step guards. Native captures match all sixteen starts and six
48-frame turns. The live-level 4 MB exercises pass on levels 1, 4 and 15,
including normal-cycle level 4; map captures and all six saves' level/map
payloads remain identical on levels 1 and 4. The indexed executable and all
231 indexed assets are byte-identical to their preceding validated versions.
Windows Release and feedback checks pass.

Public native timings are 22.45/21.66/31.44/29.53 FPS for levels 1/6/15/turning,
over the established 128/64/64/128 real-time frames at 40,000 cycles, 4 MB,
SB16 and mouse. The previous build measured 22.33/21.66/31.39/29.53. The
stationary result is 2.61x the original; the 25.80 FPS target remains unmet.
Measurements and package hashes are in `build/dos/texture-walk-performance.json`.
Tests remain muted at the host with guest audio enabled for timing/exercises.

Two generic division-narrowing trials also passed 147,690 independent cases,
but their per-call range checks did not improve the principal level-one view.
The signed-32-bit-quotient guard measured 22.43/21.70/31.51/29.59 FPS; a simpler
signed-32-bit-numerator guard measured 22.40/21.73/31.47/29.59, against the
retained build's 22.45/21.66/31.44/29.53. Both global changes were removed.
The existing general divide helper remains, alongside the targeted texture
walk whose bounds are established once per span. Trial evidence is retained
locally in `build/dos/division-experiments/results.json`.


### Validate native wall clip inputs once per segment

The native wall paths now validate the column range and six clip-array pointers
once per segment. They pass those checked inputs to the existing span merge
routine; both paths already clamp depth to at least one. General rendering and
neighbor extensions retain the guarded entry point. Span normalization, merging,
replacement, sorting and boundary rules are unchanged. No cache or data buffer
is added; native text shrinks by 3,072 bytes, with data and BSS unchanged.

Public native timings are 22.58/21.80/31.54/29.54 FPS for levels 1/6/15/turning,
against 22.45/21.66/31.44/29.53 before this change. The level-1 average frame is
44.29 ms, render 37.33 ms and p95 44.82 ms. These use the same sequential,
real-time 40,000-cycle, 4 MB, SB16/mouse setup. Raw logs and executable/archive
hashes are retained in `build/dos/clip-guards-performance.json` and its named
logs. The main view is 2.63x the original; the 3x target remains unmet.

All sixteen starts and six turning views match the original captures exactly
in each package (44 comparisons). Native 4 MB exercises on levels 1, 4 and 15
and indexed exercises on levels 1 and 4 pass, as does native level 4 at real-time
40,000 cycles. Automap captures and all six saved level/map payloads on levels
1 and 4 match the earlier projection baseline. Windows Release, feedback,
DOS arithmetic/raster tests, ZIP CRCs and package contents pass; all 231 assets
in each package match the corresponding preserved assets. Testing remains
muted at the host, with guest audio active in performance and exercise runs.


### Reuse exact wall interpolation endpoints

DOS wall interpolation now reuses a zero or full-span endpoint and constant
attributes directly, avoiding their wide multiply/divide. Identical endpoint
values also retain the initialized zero step. Other clipped positions keep
the original signed 64/32 division and fractional precision. This adds no
cache or shade bank. The DOS regression compares 4,972 cases with the original
product/division, including signed deltas, wide products and clipped endpoints;
its random state is restored so the existing raster corpus stays unchanged.

Public native timings are 22.81/21.80/31.59/29.74 FPS for levels 1/6/15/turning,
against 22.58/21.80/31.54/29.54. Level 1 averages 43.83 ms per frame with a
36.87 ms render and 44.35 ms p95. Logs and executable/archive hashes are in
`build/dos/wall-endpoints-performance.json`. The main view is now 2.65x the
original; the 25.80 FPS target remains unmet.

Both packages pass all sixteen start captures and six turning captures,
native 4 MB exercises on levels 1/4/15 and indexed exercises on levels 1/4.
The real-time 40,000-cycle native level-4 exercise has no audio underruns.
Automap captures and all six saved level/map payloads on levels 1 and 4 match
the earlier projection baseline. Windows Release, feedback, DOS raster/math,
ZIP CRC/contents/default-INI checks pass; all 231 assets per package are
unchanged. Host audio remains muted throughout testing.


### Carry-based signed projection rounding

DOS projection rounding now compares the original high word as unsigned to
put the sign-dependent bias bit in carry, adds the exact bias with two ADCs,
and returns the low quotient word with SHRD. The original 64-bit product and
rounding of signed half ties are preserved. This replaces temporary bias
registers, sign branches and an unused high-word shift with four instructions;
the portable implementation remains unchanged. The regression adds 8,235
cases using a quotient/remainder oracle, including high-word sign boundaries,
low-word carries, signed half ties and wide returned quotients. It restores
the random generator state before the existing raster cases.

Public native timings are 22.83/21.88/31.55/29.74 FPS for levels 1/6/15/turning,
against 22.81/21.80/31.59/29.74. Render times change from
36.87/36.09/22.56/26.73 ms to 36.83/35.92/22.53/26.73 ms; level 15's total
frame time rises from 31.66 to 31.70 ms, with update time rising by 0.08 ms.
The main-view gain is small. Raw evidence and executable/archive hashes are
in `build/dos/round-carry-performance.json`. The 25.80 FPS target remains unmet.

All 44 original native/indexed captures match exactly. The native 4 MB
exercises on levels 1/4/15, indexed exercises on levels 1/4, and real-time
40,000-cycle native level-4 exercise pass without audio underruns. Automap
captures and saved level/map data on levels 1 and 4 match the projection
baseline. Windows Release, feedback, DOS raster/math and package checks pass;
all 231 assets per package are unchanged. No shade/cache bank is added.


### Clip native columns before texture/depth work

Native wall columns now project and clip before calculating their texture
column, rounded depth and shade. All interpolants advance in the loop's step
clause, including rejected columns. Accepted columns keep the original values,
cap and side pixels, clip registration order and painter order. No arithmetic
precision, allocation or asset format changes.

Public native timings are 22.87/21.92/31.58/29.77 FPS for levels 1/6/15/turning,
against 22.83/21.88/31.55/29.74. Level 1 averages 43.72 ms per frame, with
36.76 ms render and 44.23 ms p95. Evidence and executable/archive hashes are
in `build/dos/wall-visible-performance.json`. The 25.80 FPS target remains unmet.

All sixteen native starts and six turning captures match the preserved original.
The full native 4 MB exercises on levels 1/4/15 pass at 200,000 cycles with
fixed logic ticks; those diagnostic timings are not benchmark results. The
four public real-time benchmark runs at 40,000 cycles pass with SB16 and mouse
active and no audio underruns. All testing is muted at the host. Windows
Release, feedback, package checks, exact automap captures and saved level/map
payload comparisons on levels 1 and 4 pass. The indexed package is unchanged:
its executable and all 231 assets match all 22 previously validated capture
inputs, recorded in `build/dos/wall-visible-indexed-unchanged.json`.


### Shade native wall core and side pixels together

A native wall's first/last column previously sampled and shaded again for
each side extension. A dedicated two/three-word raster now stores those
identical adjacent pixels while the original shaded color is still in a
register. The 320-pixel path unrolls eight rows; other pitches keep the same
sampling expression. Interior columns retain their existing kernel. Vertical
caps, clipping, interpolation, clip registration and painter order are unchanged.
No cache or asset bank is added. Native text grows by 1,024 bytes; data and BSS
are unchanged.

The DOS regression adds 384 edge rasters against the original packed-word
samples, covering two/three adjacent words, canonical and general masks,
zero-length/full 240-row draws, every unroll tail, alternate pitches, unsigned
coordinate wrapping and untouched surrounding pixels. Existing raster/math
and object-order cases pass with their original random input sequence.

Public native timings are 23.06/22.01/31.61/29.88 FPS for levels 1/6/15/turning,
against 22.87/21.92/31.58/29.77. Level 1 averages 43.36 ms per frame with a
36.40 ms render and 43.86 ms p95. Raw logs and executable/archive hashes are
in `build/dos/wall-edges-performance.json`. The main view is 2.68x the original;
the 25.80 FPS target remains unmet.

All sixteen native starts and six turning captures match the preserved original.
Full native 4 MB exercises on levels 1/4/15 pass with fixed ticks at 200,000
cycles; those timings are diagnostic only. The four real-time 40,000-cycle
benchmark runs pass with SB16/mouse enabled and zero audio underruns. Host
sound remains muted. Windows Release, feedback, package contents and exact
automap/saved level-map data comparisons on levels 1 and 4 pass. The indexed
executable and 231 assets remain identical to all 22 previously validated
indexed capture inputs.


### Classify identical darkest-shade floor rows at load

The native DOS loader now classifies each interleaved horizontal/vertical
floor row against its actual darkest RGB565 palette. A row whose 64 samples
all produce the same word can use the existing fill loop. Other rows retain
their exact texture samples. A completely uniform palette also permits the
fill for spans with both coordinates moving. The check applies only to shade
14 and spans of at least 16 pixels, keeping the ordinary floor loop unchanged.
Classification is cleared before asset replacement and rebuilt after loading
the floor, bump and smooth-bump textures with their existing palette fallbacks.

The three records occupy 804 bytes, rather than a pre-shaded texture bank;
the linked native image gains 1,024 bytes each of text and padded BSS. Assets,
save formats and rendering precision are unchanged. The DOS regression adds
9,219 oracle comparisons covering both axes and signs, all sixteen interleaved
tile channels, palette selection, two-coordinate motion, unsigned UV wrapping,
zero/short/long spans, untouched guards and reloads at the same addresses.

Public native timings are 23.19/22.22/31.62/29.88 FPS for levels 1/6/15/turning,
against 23.06/22.01/31.61/29.88. Level 1 averages 43.12 ms per frame with a
36.15 ms render and 43.62 ms p95. Raw logs and executable/archive hashes are
in `build/dos/solid-floor-performance.json`. This is 2.70x the original;
the 25.80 FPS target remains unmet.

All 44 native/indexed start and turning captures match the preserved originals.
The native 4 MB exercises on levels 1/4/15, indexed exercises on levels 1/4,
and real-time 40,000-cycle native level-4 exercise pass without audio underruns.
Automap captures and all six saved level/map payloads on levels 1 and 4 match
the projection baseline. Windows Release, feedback, DOS raster/math and package
checks pass; all 231 assets per package are unchanged. Host output is muted,
with guest SB16 and mouse active. Real 486 hardware remains unmeasured.
