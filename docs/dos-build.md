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

### Compiler feedback build

For the faster native build, supply an installed DOSBox Staging executable:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_dos.ps1 `
  -ProfileGuided -DosBox <path-to-dosbox.exe>
```

This rebuilds feedback from the current sources, using all sixteen levels,
six turning runs and a save/load exercise. Instrumented training uses 16 MB;
those timings are not performance measurements. Before packaging, the
uninstrumented image must reproduce all 22 ordinary 4 MB captures exactly
and pass the real-time 4 MB level-4 SB16/mouse exercise. Host audio is muted.
Training takes several minutes and stays under the selected build directory.
No compiler counters or additional texture/shade banks enter the package.
The option currently targets native RGB565; the indexed build is unchanged.
An ordinary build command clears the CMake feedback option and recompiles
the ordinary executable. Missing or mismatched feedback fails compilation.

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
mouse in-game; Esc releases it for menus. Enable Mouse Look in Options, or set
`mouse_look=1` in `AB3D.INI`, for vertical look. Menu options save using DOS 8.3
temporary filenames. In DOSBox, click inside the window to capture the host
mouse; Ctrl+F10 releases it. DOSBox's `[mouse] dos_mouse_driver=true` enables its
built-in INT 33h driver. Mouse accuracy still needs real-hardware testing.

The focused input check uses the actual SDL DOS driver and game's capture,
button, motion, menu-release/recapture and INI paths in 4 MB:

```powershell
python tools/check_dos_mouse.py --dosbox <dosbox.exe>
```

On Windows, append `--physical` to also send small host mouse movements into
a dedicated DOSBox window. It verifies both game input axes through INT 33h
and refuses to send input if that test process loses focus. All checks mute
host audio; the default check does not move the host pointer.

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
| Compiler feedback build (`-ProfileGuided`) | 24.32 | 34.47 ms | 41.12 ms | 41.67 ms |
| Compiler feedback without tracer duplication | 24.47 | 34.21 ms | 40.87 ms | 41.42 ms |
| Reuse flat-wall palettes and column scratch | 24.75 | 33.74 ms | 40.40 ms | 40.94 ms |

The trained build is a 2.88x increase over the original at the same rendering
size. The ordinary build without compiler feedback now measures 23.43 FPS.
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


### Reproducible compiler feedback

The optional `-ProfileGuided` build now trains from current sources, then uses
the collected counts through the normal CMake target. DOS writes counters
under unique short object names. Matching GCC auxiliary names in the final
compile preserve the IDs of static functions; see
[GCC 12.2's profile-ID implementation](https://github.com/gcc-mirror/gcc/blob/releases/gcc-12.2.0/gcc/coverage.cc#L515).
Missing feedback and checksum mismatches are compilation errors. Every object
depends on the feedback files: touching one rebuilt all 29 sources and
reproduced the exact verified executable. Ordinary builds clear the feedback
option and reproduce the preceding ordinary image byte-for-byte.

The 23 instrumented training runs use 16 MB, with fixed ticks and host audio
muted. These are not FPS measurements. The resulting uninstrumented image
must match 22 ordinary 4 MB captures and pass the real-time 4 MB level-4
SB16/mouse exercise before packaging. The ordinary images and executable
remain under `build/dos/pgo/ordinary-captures` and `ordinary.exe`; feedback,
logs and `report.json` remain under `build/dos/pgo`. No counters enter the ZIP.

The packaged native build measures 24.32/23.47/33.01/30.86 FPS for levels
1/6/15/turning, against 23.19/22.22/31.62/29.88. Level 1 averages 41.12 ms per
frame, with a 34.47 ms render and 41.67 ms p95. The four measurements remain
sequential real-time 40,000-cycle runs with 4 MB, SB16 and mouse. Raw logs and
executable/archive hashes are in `build/dos/compiler-feedback-performance.json`.
The main view reaches 2.83x the original; the 25.80 FPS target remains unmet.

All 22 native captures also match verified original inputs. The full native
4 MB exercises on levels 1/4/15 pass; automap captures and all six saved
level/map payloads on levels 1/4 remain exact. Both ZIPs pass CRC, contents,
executable and default-INI checks, with all 231 assets unchanged. The indexed
rebuild reproduces its previously validated executable. The 94 C/header files
in `src` and `tests` are unchanged from the preceding Windows Release, feedback
and DOS raster/math validation. Native text shrinks by 23,040 bytes; data is
unchanged and padded BSS grows by 1,024 bytes. Real hardware remains unmeasured.


### Compiler loop choices after feedback

The feedback build now uses `-fno-tracer`. Disabling automatic loop unrolling
and peeling reduced stationary performance in every tested level; disabling
late instruction scheduling was effectively neutral. Those changes were
rejected. All trials used the same verified feedback, strict checksum checks,
unchanged 320x240 pixels, 4 MB, 40,000 cycles, SB16 and mouse, with host output
muted. A control build reproduced the preceding executable byte-for-byte.

| Feedback variant | Level 1 | Level 6 | Level 15 | Level 1 turning |
|---|---:|---:|---:|---:|
| Previous | 24.32 | 23.47 | 33.01 | 30.86 |
| No automatic unrolling/peeling | 24.04 | 23.24 | 32.53 | 30.86 |
| No tracer duplication, final package | 24.47 | 23.50 | 32.92 | 30.86 |
| No late instruction scheduling | 24.32 | 23.47 | 32.99 | 30.87 |

The retained flag reduces level-1 render time from 34.47 to 34.21 ms. Its
average frame falls from 41.12 to 40.87 ms, with p95 falling from 41.67 to
41.42 ms. Level 15's frame rises from 30.29 to 30.37 ms; its update rises by
0.06 ms and render by 0.02 ms. This small tradeoff accompanies the main-view
gain. Turning remains unchanged. The native text section shrinks by another
1,024 bytes, with data/BSS unchanged and no additional runtime tables.

All sixteen native starts and six turning captures remain byte-identical
to the verified original inputs. The native 4 MB exercises on levels 1/4/15
and the real-time 40,000-cycle level-4 exercise pass without audio underruns.
Automap captures and all six saved level/map payloads on levels 1/4 remain
exact. Both packages pass CRC, contents, executable/default-INI and all 231
asset comparisons. The indexed image and all 94 C/header files in `src` and
`tests` retain their preceding validated bytes. Timings and package hashes
are in `build/dos/no-tracer-performance.json`; the rejected trials are under
`build/dos-compiler-trials`. The main view is 2.85x the original; the 25.80 FPS
target remains unmet. Real 486 hardware remains unmeasured.


### Reuse flat-wall palette selection and column scratch

A fixed-tick diagnostic run recorded 891 native flat-wall spans covering
28,446 columns, all with constant brightness and one shared palette. It also
found no vertical wall steps at or below half a texel, so adding another
repeated-sample kernel would not help that view. These counters were used for
workload analysis only and are absent from the final executable.

Native `ScreenWallstripdraw` spans with constant inverse Z and brightness now
select the authored palette once per span. Varying brightness retains its
original calculation for every column. Palette preparation finishes before
the existing clip-registration loop, whose column and neighbor order is
unchanged; the update flag is checked once outside that loop. Projection,
texture coordinates, sampling, painter order, caps and side extensions remain
exact.

The native renderer also owns a column array sized with its framebuffer,
releasing it on resize/shutdown. DOS rendering is synchronous, so successive
spans reuse it. This removes the variable-size stack array; the persistent
heap cost is 2,560 bytes at width 320, plus allocator metadata. No shade bank
is added, and the linked text/data/BSS section sizes are unchanged. The
palette-only ordinary build measured 23.41 FPS; array reuse raised that to
23.43, against 23.19 before this change.

Fresh feedback training and both ordinary/optimized 4 MB capture checks pass.
The final package measures 24.75/23.60/33.04/30.86 FPS for levels 1/6/15/turning,
against 24.47/23.50/32.92/30.86. Level 1 averages 40.40 ms per frame with a
33.74 ms render and 40.94 ms p95. The four sequential real-time runs retain
40,000 cycles, 4 MB, SB16 and mouse, with host output muted. Logs and hashes
are in `build/dos/flat-prepared-performance.json`; the main view reaches 2.88x
the original, with the 25.80 FPS target still unmet.

All 22 native captures match verified original inputs. Native 4 MB exercises
on levels 1/4/15 and the real-time level-4 exercise pass without audio underruns,
including all 24 weapon frames and repeated saves/quickloads. Automap images
and all six saved level/map payloads on levels 1/4 remain exact. The indexed
rebuild reproduces its preceding executable. Windows Release, feedback, DOS
raster/math/object checks and both package audits pass; all 231 assets per
package are unchanged. Real hardware remains unmeasured.


### Measured wall-clip merge alternatives

A temporary fixed-tick native diagnostic recorded 10,560 empty first slots,
20,548 existing first-span fast merges, 1,156 generic first-span merges,
3,209 second-span merges and 2,048 second-span insertions. Every second-span
insertion needed the original ordering swap. Startup drawing is included;
these counters describe the workload and are not FPS measurements.

Three alternatives passed 252,768 independent clip cases, including sequential
updates, invalid spans, depth ties and signed-16-bit writeback boundaries.
Their ordinary-build FPS for levels 1/6/15/turning were 23.33/22.27/31.69/29.84
(first-span shortcut), 23.43/22.30/31.69/29.85 (direct insertion/swap), and
23.38/22.25/31.67/29.84 (ordered merge writeback). The unchanged ordinary
control measured 23.43/22.28/31.71/29.88. These use the established sequential
real-time 40,000-cycle, 4 MB, SB16/mouse runs with muted host output.

None improved the principal view. All three changes and their temporary
production header were removed, and rebuilding reproduced the original
ordinary executable byte-for-byte. Trial sources, oracle cases, counters and
logs are retained locally in `build/dos-clip-experiments`. The public compiler
feedback build remained at 24.75 FPS while the next presentation trial began.


### Present native gameplay through a borrowed canvas

The native 320x240 bar HUD now presents the renderer's existing RGB565 canvas
directly. Its two 10x130 bar bounds and 134x36 key band come from the same
layout functions used to draw the HUD, including collected-key underlines.
Before overlay drawing, 7,424 world pixels are preserved in the existing SDL
RAM surface. They are restored after the video-memory copy. This replaces a
153,600-byte full-frame copy with 29,696 bytes of HUD backup/restoration.
The renderer's world history is exact when presentation returns, including
with frame clearing disabled. SDL retains ownership of its original pixels.

The pinned SDL driver loans its public surface and cached presentation pointer
only for that synchronous interval. Later surface reads and partial UI writes
materialize the displayed video-memory image on demand; a complete frame
replacement discards that pending readback. Readback uses the displayed LFB
page or readable banked window. Other HUD/overlay configurations and video
layouts continue through the existing independent surface path. Captures made
during presentation read the complete current image while the loan is active.

`cmake/dos-present.patch` contains the driver changes. `tools/setup_dos.ps1`
applies it alongside the existing SDL patch; existing dependency caches need
the setup step rerun. The patch applies to a clean pinned driver, reproduces
the working driver, and reverses exactly. The game adds surface metadata and
three saved rectangles, using the existing pixel allocations. Native text
grows by 5,632 bytes; data and BSS sizes are unchanged.

A native diagnostic compares both actual presentation paths for nine
combinations of key ownership and health/ammo, checking every display and
world word plus RenderReadPixels. It uses disk oracles to fit the 4 MB target.
All nine comparisons pass in the level 1/4/15 fixed-tick exercises and the
real-time level-4 exercise: 36 complete display/readback/history comparisons.
The same exercises retain all 24 weapon frames and repeated full quickloads.

Fresh compiler-feedback training covers all 23 scenarios. All sixteen native
starts and six turns match the preserved original images in both ordinary and
optimized builds. The indexed rebuild also matches all sixteen starts and six
turns, with 4 MB exercises passing on levels 1/4. Native automap images and all
six saves' level/map payloads remain exact on levels 1/4. Both packages pass
CRC, contents, executable/default-INI and all 231 asset comparisons. Windows
Release and feedback checks pass.

The final native package measures 25.11/24.02/33.75/31.36 FPS for levels
1/6/15/turning, against 24.75/23.60/33.04/30.86. These are the established
128/64/64/128 sequential real-time frames at 40,000 cycles, 4 MB, SB16 and
mouse, with host output muted. Level 1 falls from 40.40 to 39.82 ms per frame;
presentation falls from 3.81 to 3.25 ms, and p95 falls from 40.94 to 40.37 ms.
Logs and package hashes are in `build/dos/hud-canvas-performance.json`;
validation is in `build/dos/hud-canvas-validation.json` and
`build/dos/hud-canvas-exercises.json`. The main view reaches 2.92x the original;
the 25.80 FPS target remains unmet. Real 486 hardware remains unmeasured.


### Pass shared wall palettes and prepare HUD copies

Native constant-brightness flat walls pass their known authored palette directly
to the row rasterizer. This removes per-column palette writes and repeated
classification. Varying brightness retains automatic classification and its
original samples. An additional 128 packed-word oracle comparisons verify the
explicit-palette path with every per-column palette field set to NULL.

The HUD backup now copies each rectangle with one strided assembly loop. Its
leading/trailing word handling is selected once per rectangle; row pointers,
copy counts and stride stay in registers. HUD bounds are prepared with the UI
assets using the existing layout functions and saved rectangle storage. A
192-case comparison against ordinary row copies covers alignment, odd widths,
empty sizes and untouched row guards. The full image/readback/history check
also retains verification when hardware uses the independent surface fallback.

The ordinary control measured 23.69/22.54/32.24/30.30 FPS for levels
1/6/15/turning. Passing the palette measured 23.75/22.56/32.25/30.30; strided
copying raised this to 23.84/22.62/32.41/30.51, and preparing the bounds once
reached 23.85/22.62/32.42/30.51. Source sampling, shading, painter order, HUD
geometry and world history remain exact. No pixel allocation or shade bank is
added. Final native text grows by 512 bytes, with data/BSS unchanged.

Fresh feedback training and both ordinary/optimized 4 MB capture checks pass.
The final package measures 25.25/24.04/33.79/31.53 FPS, against the preceding
25.11/24.02/33.75/31.36. Level 1 averages 39.60 ms per frame, 33.63 ms rendering,
3.13 ms presentation and 40.15 ms p95. These retain the established sequential
128/64/64/128 real-time frames, 40,000 cycles, 4 MB, SB16 and mouse, with muted
host output. Logs/hashes are in `build/dos/row-copy-performance.json`.
The main view reaches 2.94x the original; the 25.80 FPS target remains unmet.

All 22 native captures match verified original inputs. Native 4 MB exercises
on levels 1/4/15 and real-time level 4 pass, including 768 guarded rectangle
cases, 36 complete canvas/readback/history comparisons, all 24 weapon frames
and repeated full quickloads. Map images and all six saved level/map payloads
remain exact on levels 1/4. The indexed rebuild reproduces its validated
executable byte-for-byte. Windows Release, feedback, DOS raster/math/object
checks and both package audits pass; all 231 assets per package are unchanged.
Evidence is in `build/dos/row-copy-validation.json` and `row-copy-exercises.json`.

A separate diagnostic found 891 shared-palette flat spans and 28,446 texture
columns, with no adjacent repeated columns; horizontal reuse was therefore
not added. GCC pointer analysis measured only 23.70/22.55/32.24/30.30 FPS and
was removed. A mapped linear-framebuffer trial measured 17.54 FPS with page
flipping and 22.11 FPS with direct visible-page writes, against the banked
23.75 FPS control. Both linear trials passed the level-1 canvas/readback check,
but were removed; the pinned driver/mode code was restored exactly. Trial
records are under `build/dos-row-runs`, `build/dos-pta-trial` and
`build/dos-lfb-trial`. Real hardware remains unmeasured.


### Cancel exact floor/projection factors and reuse sky bounds

The DOS executable is 32-bit i486 code. Explicit 64-bit arithmetic still costs
software work, so these changes remove repeated wide division only where its
original result is preserved. A temporary floor diagnostic recorded 10,804
row preparations across 33 draws; every sampled row used width 320/base 640.
For that exact ratio, cancelling the common width factor turns the original
rounded quotient into a signed shift. Signed half ties and INT32 extremes
remain exact, including in the four-instruction 386 shift specialization.
Other ratios and the existing unrounded full-width path are unchanged.

Billboard scales with no fractional 16.16 bits cancel their common 65536
factor before division. Fractional scales retain the original wide expression.
Sprite, explosion, zone-clip and vector-object projections reuse the existing
exact 64/32 divider, retaining wide numerators and genuinely wide quotients.
The sky caches its horizontal quotient with both width and view columns as
keys. Its existing offset table now records the largest offset; a subtraction
check proves all two-byte reads fit before using an unchecked sampling loop.
Short inputs retain the original checked loop. The two additions need 16 bytes
of cache fields within existing BSS padding, with no new pixel or shade bank.
Native text grows by 3,072 bytes; data and BSS remain 22,016/285,184 bytes.

Independent DOS checks pass 171,040 signed floor-rounding cases, 66,265
whole/fractional billboard projections, 3,120 sky-cache checks and 35,512
word-span bounds checks. These cover signed half ties, extreme values,
changed cache keys, wide quotients, short inputs and unsigned overflow.
The existing raster, water, sprite, wall and object-order checks also pass.
A separate diagnostic found no eligible constant-stride flat-wall spans;
no wall-stride kernel or full row-math cache was added. Diagnostic code is
absent from both release packages.

Fresh compiler-feedback training covers all 23 scenarios. The final package
measures 25.83/24.41/34.16/31.68 FPS for levels 1/6/15/turning, against the
preceding 25.25/24.04/33.79/31.53. The main view averages 38.72 ms per frame,
32.76 ms rendering, 3.13 ms presentation and 39.24 ms p95. An independent
128-frame repeat also returns 25.83 FPS. Both use the established sequential
real-time runs at normal/486_slow, 40,000 cycles, 4 MB, SB16 and mouse, with
host output muted and no fixed-tick shortcut. The main view reaches 3.00x
the original 8.60 FPS and meets the 25.80 FPS target. Real 486 hardware remains
unmeasured. Full 320x240 RGB565 rendering, sampling and shading are preserved.

All sixteen native starts and six turns match the verified original images
in both ordinary and optimized builds. The optional indexed build also
matches all sixteen starts and six turns. Native 4 MB exercises on levels
1/4/15 and real-time level 4 pass, including 768 guarded rectangle cases,
36 complete canvas/readback/history comparisons, all 24 weapon frames and
repeated full quickloads, without audio underruns. Indexed 4 MB exercises
pass on levels 1/4. Native automap images and all six saved level/map payloads
remain exact on levels 1/4. Both ZIPs pass CRC, executable/default-INI and all
231 asset comparisons. Windows Release and feedback checks pass.

Logs, executable/package hashes and confirmation are recorded in
`build/dos/floor-project-performance.json`; capture/package validation is in
`floor-project-validation.json`, live checks in `floor-project-exercises.json`
and indexed image evidence in `floor-project-indexed-pixel-evidence.json`.
The immutable source/package/evidence copy is `build/dos-floor-project-baseline`.


### Working DOS mouse capture and saved look options

The pinned DOS SDL backend had a relative-motion polling branch but did not
register `DOSVESA_SetRelativeMouseMode`. SDL therefore rejected capture, and
`input_apply_relative_mouse` correctly discarded gameplay mouse motion.
`cmake/dos-mouse.patch` registers the driver callback and clears old INT 33h
motion counters at capture transitions, so menu movement cannot jump the view.
It is applied by `tools/setup_dos.ps1` alongside the existing patches; rerun
setup for an existing dependency cache. With no mouse driver, capture still
reports failure.

Relative events now pass signed INT 33h/0Bh motion counts directly, matching
SDL's desktop raw-input path. The old backend incorrectly divided by the
INT 33h/1Bh sensitivity settings, making small moves disappear in integer
consumers. These are different driver quantities, as documented in the
[FreeDOS mouse interrupt reference](https://github.com/FDOS/mouse/blob/master/int33.lst).
The unused sensitivity fields and query are removed. Gameplay control scales,
mouse-look projection and aim logic remain unchanged.

`settings_path_with_suffix` replaces the INI extension for the DOS temporary
file, producing `ab3d.tmp` rather than the invalid `ab3d.ini.tmp`. Options can
now persist mouse look and inversion through the existing replacement path.
Desktop and web settings paths are unchanged.

`tools/check_dos_mouse.py` exercises the real DOS SDL capture callback and
actual game input/INI code in 4 MB, including motion gating, buttons, menu
release/recapture, missing-driver rejection and saved mouse-look/inversion
values. Its Windows `--physical` case also verifies small host movements in
both directions on both game axes through INT 33h, without injected SDL events.
The test refuses to send host input after another process gains focus.
All three cases pass, as do clean application of all three dependency patches
and reproduction of the working SDL source from its pinned revision.

Fresh feedback training covers all 23 scenarios. The final raw-motion backend
correction relinks that feedback without changing instrumented game sources;
the final public/private feedback executables match byte-for-byte. All sixteen
native starts and six turns, and all sixteen indexed starts and six turns,
match the verified original images. Final native and indexed 4 MB level-4
exercises pass, including repeated full quickloads and all 24 weapon frames.
Native map pixels and all six saved level/map payloads remain exact. Both
ZIPs pass CRC, executable/default-INI and all 231 asset comparisons. Windows
Release and the existing feedback regressions also pass.

The final native package retains 25.83 FPS in the sequential 128-frame level-1
benchmark at normal/486_slow, 40,000 cycles, 4 MB, SB16 and mouse enabled, with
host audio muted. Frame/render/present times remain 38.72/32.76/3.13 ms, with
39.24 ms p95 and no audio underruns. This is still 3.00x the original 8.60 FPS;
real hardware remains unmeasured. Package hashes, final input/image/exercise
checks and timing are recorded in `build/dos/mouse-validation.json`.


### Reject unreachable blast distances before visibility traversal

`compute_blast_apply_target` now applies its existing cylinder-edge distance
bucket rejection before `can_it_be_seen`. Targets outside that unchanged range
cannot take damage, so their room-portal traversal is unnecessary. Damage,
knockback, particles, corpse gibbing, random calls and update order are retained.

A diagnostic triggers the real `object_handle_barrel` handler after a lethal
weapon-1-equivalent hit on a real barrel, using original level 1/6 geometry.
Across quiet and close-up explosion cases, all 128 per-frame object/shot/point
hashes and all 20 selected complete images match the preceding code exactly.
The level-1 detonation drops from 109 to 32 visibility calls and 3.063 to
2.277 ms; level 6 drops from 67 to 30 calls and 3.587 to 2.339 ms. These are
fixed-tick diagnostic measurements at 40,000 cycles/4 MB, with host audio muted.
Windows Release and the existing feedback/corpse regressions pass.

This removes avoidable blast work but does not solve the large rendering
pause: the close-up explosion still tests over two million sprite pixels in
one frame. Diagnostic sources, images and timing are retained locally under
`build/dos-effects-trace` while that rendering work is investigated.


### Copy magnified native sprites by opaque row spans

The native 320-pixel renderer now groups identical source columns, decodes
one RGB565 row per distinct sampled source row, and copies its opaque spans
across repeated destination rows. Texel-zero cutouts remain separate from
pixel values, so opaque black stays opaque. Packed PTR modes, 16.16 stepping,
clipped sampling origins, strip offsets, truncated inputs and authored shade
palettes retain their original values. This uses about 6.3 KiB of stack scratch
without a new heap allocation or texture/shade bank. Magnified sprites
use the row path when no picking channel is active and the complete vertical
interval remains visible. Existing disjoint room-mask ranges are copied as
separate rectangles. A front wall span retains the general clipping path.

All 144 independent DOS row-raster cases pass, including every packed mode,
odd widths, clipping origins, repeated rows, black/cutout texels, short inputs
and untouched guards. In the real level-1/6 close-barrel diagnostic, all 128
per-frame object/shot/point hashes and all 20 selected complete images match
the original renderer. An additional 128 hashes match with detailed rendering
profiling disabled. All tests use 4 MB with host audio muted.

With detailed profiling disabled, the worst post-detonation frame in the
32-frame level-1 fixture falls from 1160.547 to 153.751 ms. Level 6 falls from
599.112 to 98.208 ms over the same post-detonation interval. These are diagnostic
ordinary-build timings at normal/486_slow, 40,000 cycles and fixed game ticks;
they are separate from the stationary real-time FPS benchmark. All explosion
particles, gibs and gameplay updates are preserved. Room-spill masks
retain their existing geometry and source sampling; any wall-depth occluder
retains the original per-column handling. All 256 state hashes and 20 selected
images remain exact after extending the row path to unobstructed room masks.


The final compiler-feedback package passes fresh training on all 23 scenarios
and all 22 ordinary/optimized 4 MB capture comparisons. Its sixteen starts and
six turns also match the verified original native images; the indexed build
matches all 22 of its original images. Native 4 MB exercises pass on levels
1/4/15, plus the real-time level-4 exercise; the indexed level-4 exercise also
passes. Repeated full quickloads, all weapon frames and native canvas/history
checks pass without audio underruns. Native map images and all six saved
level/map payloads remain exact on levels 1/4. Both packages retain all 231
assets and pass ZIP CRC, executable and default-INI checks.

The normal level-1 view measures 25.82 FPS over 128 sequential real-time frames
at normal/486_slow, 40,000 cycles, 4 MB, SB16 and mouse, with host audio muted.
Frame/render/present times are 38.72/32.76/3.13 ms, with 39.24 ms p95. This
retains the 3x target while reducing the barrel-explosion spike. Native text,
data and BSS measure 1,180,696/22,016/284,160 bytes; the row scratch stays on
the existing stack. Windows Release and feedback regressions pass. Real
hardware remains unmeasured. Final package hashes, diagnostic comparisons,
live exercises and timing are in `build/dos/effects-performance.json`; original
image, package, asset and save checks are in `effects-validation.json`.

### DOS water-surface conversion and interpolation

The level-1 pool slowdown reproduces from the dry shores in zones 91, 89 and
90. The old four-entry nibble-blend cache misses 115-157 times per frame in
these views. The native DOS path now prepares the original OldBrightenFile
colours as three byte-separated nibbles and uses an 8 KB difference table
for exact channel interpolation. It follows AB3DI.s texturedwater sampling,
including wateroff carrying from U into V, and retains the existing
fractional vertical refraction, rounding, animation and painter order.

The prepared data adds 31,264 bytes, including 544 bytes of native output
colours. It avoids building a blend row for every water span and converting
both tinted RGB565 samples back to Amiga12 for each pixel. The original
path still handles indexed output and diagnostic water tracing.

At normal/486_slow, 40,000 cycles, 4 MB, SB16 and mouse, with host sound muted,
64-frame fixed-tick diagnostic runs preserve all 192 game-state hashes and
15 selected full-frame images across the three shores. The 4 MB raster
check also verifies all 65,536 alpha/colour pairs plus 384 spans covering
UV wrap/carry, guards, clipping lengths and framebuffer aliasing. With
detailed rendering profiling off, mean game-loop times after eight warmup
frames fall from 110.35/98.96/111.72 ms to 81.11/76.19/84.70 ms. These
ordinary-build diagnostic timings are not the main real-time FPS benchmark
and do not establish physical 486 cache behaviour.

An identical pair of tinted samples now bypasses interpolation entirely.
That result is exact for every alpha. The same 15 frames, 192 state hashes
and 4 MB raster cases still match. The three ordinary-build mean loop
times improve further to 76.76/72.22/81.11 ms, with no profile audio
underruns in these runs.

Final compiler-feedback builds were also compared using the same diagnostic
camera/loop wrapper linked against the preserved old and new optimized
objects. Profiling and captures are off during the timed runs; all emulators
and builds run sequentially for those measurements. Each view runs 64 fixed
ticks, with the first eight frames excluded from the mean.

| Dry shore (level 1) | Camera X/Z, angle | Old loop ms | New loop ms | Reduction |
| --- | --- | ---: | ---: | ---: |
| Zone 91 | -48/2901, 0 | 120.57 | 74.57 | 38.2% |
| Zone 89 | -48/3285, 4096 | 106.57 | 70.42 | 33.9% |
| Zone 90 | 240/3100, 6144 | 120.87 | 79.12 | 34.5% |

All accepted comparisons preserve every per-frame simulation hash. One
timing attempt diverged after frame 27 and was rejected; its repeat matched
all 64 hashes. The cause of that divergent attempt was not established.
The optimized builds also preserve all 15 selected pool images. Independent
ordinary-build profiles put water work alone at 73.76/58.62/68.11 ms before
and 38.33/30.58/36.12 ms afterwards.

The normal 128-frame real-time level-1 benchmark remains 25.82 FPS: 38.72 ms
per frame, 32.76 ms render, 3.13 ms present and 39.24 ms p95, with no audio
underruns. This retains the original roughly-threefold performance gain.
All 22 ordinary and optimized starting/turning views match the preserved
original images. Level 1/4/15 full 4 MB exercises pass, including quickloads,
save payloads, canvas/readbacks, HUD, menus and weapon caches. The real-time
40,000-cycle level-4 exercise also passes with zero profile audio underruns.
Its complete disk-backed save/readback checks exceed four minutes, so the
host exercise deadline is now six minutes; the timed frame window is unchanged.

The indexed DOS executable is byte-identical to its previous validated
build. Windows Release and the feedback suite, including 3,600 authored
waterline cases, pass. Both packages retain all 231 original prepared assets
and 238 archive entries, with generated defaults and no diagnostic fixture
instrumentation. Native sections are text 1,182,232, data 22,016 and BSS
315,392 bytes. The final native EXE SHA-256 is
`fbe3540543005aaf0d0a0ae7092bedbdf33c18e0d55e34187036571667273a8b`;
local comparison and package evidence is in `build/dos/water-performance.json`
and `build/dos/water-validation.json`. Physical 486 hardware is unmeasured.

### Native clipped billboard columns

Normal-size and wall-clipped sprites now use a compact 486 packed/decoded
column sampler after the existing room, plane and wall clipping. It retains
ObjDraw3.ChipRam.s BitMapObj source rows, packed shifts, authored palettes
and texel-zero cutouts. Repeated screen columns also reuse their decoded
source column. This adds no texture bank, heap allocation or static RAM.

All 768 independent column cases pass at 4 MB, including packed/decoded
sources, clipping origins, strides, short columns, odd addresses, opaque
black, transparent texels and guards. Three real level-1 diagnostic views
retain all 192 simulation hashes and 15 selected complete images. Separate
ordinary-build profiles reduce sprite work from 1.517/15.364/4.576 ms to
1.067/13.097/3.126 ms across the starting, close-barrel and pool views.
Those profiles include diagnostic overhead; whole-frame timings and the
final compiler-feedback package are checked separately. Shooting rules,
including PlayerShoot.s's distance-based instant-hit accuracy roll, are unchanged.

### Preserve short DOS mouse presses across long frames

The SDL DOS backend previously sampled only the current INT 33h/03h button
state. A complete press/release during an enemy draw could disappear. The
new `cmake/dos-mouse-buttons.patch` queries the driver's latched press counts
with INT 33h/05h, emits a press edge and restores the actual held state.
This follows the [FreeDOS mouse API](https://github.com/FDOS/mouse/blob/master/wheelapi.txt).
The DOS SDL adapter drains old counts at capture transitions. Per-frame
event polling cadence, raw movement and mouse sensitivity are unchanged.

DOS game input also retains received left/right press edges until the next
control update consumes mouse input. Otherwise DOWN and UP events dequeued
together would still erase the press. Menu/capture changes clear pending
presses; a held physical button remains held after consuming its edge.
This does not buffer a shot past the original weapon cooldown or change
PlayerShoot.s's distance-based random hit roll, damage or target rules.

The 4 MB mouse regression passes capture/release/recapture, rapid queued taps,
holds and missing-driver checks. It compiles the installed SDL button helper
with a scripted INT 33h boundary and the real SDL event queue/game input,
checking latched counts, released taps and menu transitions. A live Windows
mouse probe was stopped when foreground ownership could not be retained;
that physical probe is not counted as passing. The optional physical test
also checks a released 30 ms host click across a 1,500 ms polling gap.
Repeated dependency setup recognizes the cached fourth patch correctly.

The final packages were built from committed DOS source `f4dce86` in
`build/dos-enemies-source`. A concurrent edit to the main worktree's game
loop invalidated an earlier training run; the detached checkout keeps the
compiler inputs fixed and leaves that separate engine/editor work intact.
Both compiler training and the 22 ordinary/optimized capture checks pass.
The indexed build matches all 22 original captures as well.

Matched optimized-object diagnostics retain all 192 per-frame simulation
hashes and 15 enemy-view images. With detailed profiling disabled, mean
loop times over frames 8-63 change as follows at normal/486_slow, 40,000
cycles, 4 MB, SB16 and mouse, with host sound muted:

| Level-1 view | Previous loop ms | Updated loop ms |
| --- | ---: | ---: |
| Starting room | 39.03 | 38.80 |
| Close barrel / approaching enemies | 44.18 | 42.81 |
| Pool north shore | 74.58 | 73.75 |

These are modest whole-frame improvements: larger sprites and scene work
still cost time. They are separate from the real-time 128-frame starting-view
benchmark, which now measures 26.00 FPS (38.46 ms frame, 32.46 ms render,
3.13 ms present, 38.98 ms p95). Button-counter polling raises input time
from 0.31 to 0.36 ms while preserving per-frame polling and short presses.

Native 4 MB full exercises on levels 1/4/15, the real-time level-4 exercise,
and the indexed level-4 exercise all pass without profile audio underruns.
Maps and all six saved level/map payloads remain exact. Both 238-entry
packages preserve all 231 prepared assets, generated defaults and ZIP CRCs;
diagnostic markers are absent. Local mouse-look preferences are retained.
Windows Release and the feedback suite pass, including 40 current-aim shot
cases. The queued-input/driver-boundary tests pass; the live host-click
probe remains unverified because it lost foreground ownership.

Native text/data/BSS are 1,180,696/22,016/315,392 bytes; total static RAM is
unchanged. Final EXE SHA-256 is
`070f7c4a0db8673f27a89280a061d42ee957c9e60a6ef0102e36133e338a1722`.
Local checks and comparisons are in `build/dos/enemies-performance.json`
and `build/dos/enemies-validation.json`. Physical 486 hardware is unmeasured.


### Native billboard row expansion at every scale

The DOS RGB565 row sampler now keeps BitMapObj packed-word extraction,
palette lookup and horizontal run expansion in one 486 register loop.
Opaque spans are recorded at alpha transitions and copied to repeated
screen rows with REP MOVSL. Ordinary-size and reduced sprites use this
path too when their existing screen, plane and geometric masks form
rectangles. Spill sprites with a front wall retain the clipped column
sampler. Source texel zero remains transparent; opaque black and equal
palette colours retain their original coverage.

The kernel uses approximately 7 KiB of bounded stack scratch at width 320,
about 640 bytes more than the previous row sampler. It adds no heap
allocation, static RAM, texture conversion cache or shade bank. The native
4 MB raster oracle covers 288 row cases, including repeated rows, all
packed modes, short WAD/PTR data, clipping origins, opaque black, shared
colours, noncanonical RGB565 word one, and surrounding guards. The 768
column cases and the existing wall, floor and water checks pass.


Matched detailed renderer diagnostics reduce sprite time from
1.069/13.095/3.124 ms to 0.853/8.022/2.233 ms in the level-1 starting,
close-barrel/enemy and pool views: about 20-39% less billboard work.
The following separate comparisons use the unchanged production renderer,
raster and other engine object files from each optimized release, with
only the diagnostic entry point relinked. Detailed profiling and captures
are disabled during timing. These are normal/486_slow, fixed 40,000-cycle,
4 MB, native 320x240, SB16/mouse runs with host sound muted; means cover
frames 8-63 of the same 64 fixed game updates.

| Level-1 view | Previous loop ms | Updated loop ms |
| --- | ---: | ---: |
| Starting room | 38.802 | 38.618 |
| Close barrel / approaching enemies | 42.809 | 38.422 |
| Pool north shore | 73.750 | 73.004 |
| Close barrel after detonation | 75.501 | 65.702 |

This is 11.4% more FPS in the close enemy view and 14.9% more during the
burst. Its worst loop falls from 188.754 to 146.697 ms, with p95 falling
from 186.978 to 145.403 ms; explosion frames still have visible stalls.
The explosion comparison invokes the original lethal barrel handler on
frame eight. All 256 optimized simulation hashes, 20 selected complete
images and four final captures match, including the blast/debris sequence.

The independently run real-time starting-view benchmark improves from
26.00 to 26.11 FPS: frame 38.30 ms, render 32.29 ms, present 3.13 ms,
input 0.36 ms, p95 38.81 ms. This lightly populated view sees much less
benefit than the close-up and burst scenes.

Both builds' 22 native captures match the original start/turn references.
The optimized 4 MB level-1/4/15 full exercises and real-time level-4
SB16/mouse exercise pass with zero profile audio underruns. Maps and all
six saved level/map payloads remain exact. The indexed executable is
byte-identical to its previously validated release. Windows Release and
the feedback suite pass. Both packages preserve all 231 prepared assets,
238 archive entries and generated defaults; diagnostic instrumentation
is absent, and local mouse-look preferences are retained.

The release is built from committed source `cd29e15` in
`build/dos-billboards-source`, keeping concurrent editor/engine work out of
compiler training. Native text/data/BSS are 1,179,672/22,016/315,392 bytes.
The final native EXE SHA-256 is
`0c9723ac5b4d3a26e8ea09f7546bfca975f599ef2a644c6fe542c857b69baac3`.
Local evidence is in `build/dos/billboards-performance.json` and
`build/dos/billboards-validation.json`. Physical 486 hardware is unmeasured.


### Dense native billboard passes: preserve pixels while avoiding overdraw

The original ObjDraw3.ChipRam.s BitMapObj composites opaque samples in
painter order, with source index zero as its only cutout. A DOS RGB565 pass
containing only billboards can therefore traverse that same sorted list in
reverse and retain the first opaque write at each pixel. The final pixels,
including equal-depth ownership and opaque black, are identical. The
coverage mask is reset at each room/section object pass so intervening walls,
floors and water keep their existing ordering. Passes containing polygon
objects and all picking/debug paths retain the original traversal.

The 320x240 mask and per-row counters use approximately 9.6 KiB of the
already reserved DOS stack, with no per-sprite texture/shade bank or new
heap allocation.
Packed and decoded clipped columns share the same coverage bits as row
spans. Fully covered rectangles and repeated source rows skip decoding.
Small lists retain the ordinary path; larger lists activate coverage only
when their existing projected-width estimates cover at least one screen.
The estimates reuse padding in the draw-list entries. This avoids paying
for coverage when late/distant debris has spread into sparse particles.

The independent DOS compositor oracle covers 32 layered row/column cases,
including source cutouts, black and RGB565 word one, mask-word boundaries,
complete coverage, repeated passes and surface guards. The existing 288 row,
768 column and floor/wall/water oracles pass at 4 MB. Both level-1 and level-6
burst comparisons capture every frame: all 128 simulation hashes and 130
complete images, including the final frames, match the unmodified current
renderer.


Matched diagnostics against the current source reduce burst billboard work
from 22.74 to 14.26 ms. Separate comparisons relink only the diagnostic
entry point against each release's unchanged optimized engine objects.
Detailed profiling and captures are disabled for those timings. At
normal/486_slow, fixed 40,000 cycles, 4 MB, 320x240, SB16 and mouse with
host sound muted, frames 8-63 of 64 identical updates measure:

| View | Previous release loop ms | Updated release loop ms |
| --- | ---: | ---: |
| Level-1 starting room | 38.648 | 38.757 |
| Close barrel / approaching enemies | 38.421 | 38.499 |
| Pool north shore | 73.005 | 73.081 |
| Close barrel after detonation | 65.702 | 58.156 |

The burst gains 13.0% more FPS. Early burst frames 8-15 fall from
142.808 to 86.386 ms on average. Burst p95 falls from 145.403 to 112.939 ms,
and the worst frame falls from 146.696 to 124.092 ms. Ordinary views change
by approximately 0.1 ms; late sparse debris retains its original traversal.
The independent real-time 128-frame starting benchmark changes from 26.11
to 26.04 FPS (38.40 ms frame, 32.39 ms render, 3.13 ms present, 0.36 ms input,
38.91 ms p95). This work targets expensive overlap rather than light scenes.

Keeping the mask on the already reserved 128 KiB DOS stack avoids reducing
save/reload headroom. Existing wall-column, sprite-row, sky-offset, floor
reciprocal and gun-column scratch now share one viewport-sized allocation:
6,084 bytes at 320x240 versus 12,804 bytes for the old allocation and static
tables. Original 96-column gun indices fit in bytes, enforced at compile
time. Cached sky offsets and floor reciprocals invalidate on reallocation.
The supported render limits and output dimensions are unchanged.

All 22 ordinary and optimized native captures and all 22 indexed captures
match the original start/turn references. The 4 MB full level-1/4/15 fixed
exercises and real-time level-4 SB16/mouse exercise pass, including repeated
quickloads and UI/weapon checks, with zero profile audio underruns. Maps and
all six saved level/map payloads remain exact. Both 238-entry packages
retain all 231 prepared assets and generated defaults, with no diagnostic
instrumentation. Windows Release and the feedback suite pass. Local mouse
settings are preserved when staging the packages.

The final code is committed as `88d0193` in `build/dos-consistency-source`.
Native text/data/BSS are 1,183,768/22,016/305,152 bytes: total static memory
is 6 KiB smaller than the previous release. The optimized shared-asset
checkpoint reports 108 KiB free physical memory versus 104 KiB previously.
The native EXE SHA-256 is
`f44c2a2a4b4a1794a32cd3f96a468664f75d5d299f496d07d39097b6d52a9ef4`.
Evidence is in `build/dos/consistency-performance.json` and
`build/dos/consistency-validation.json`. Physical 486 hardware is unmeasured.

The broader performance goal remains active. In the pool view, detailed
profiling attributes 38.33 ms to water within 40.27 ms of inclusive floor
work, versus 13.40 ms for walls and 2.22 ms for sprites. The next investigation
is water alpha switching and equal-background sample frequency, to select
an exact blend optimization that fits the same memory budget.


### Exact native water sampling with less table storage

The next profile isolates water as the dominant pool-view cost. Across
64 fixed updates it uses 220 alpha values, changing alpha in 9,872 of
10,232 spans; 527,267 samples brighten to the same colour and 608,682 need
interpolation. Rebuilding a large per-alpha bank would add work to almost
every span. The selected DOS kernel instead decodes RGB565 green/blue
indices through one 2 KiB table and shares its prepared Brighten colours
with the renderer, removing the duplicate 11 KiB native colour array.

The existing four-bit rounding is exactly symmetric when exchanging the
two samples and replacing alpha with 255-alpha. Only 128 delta rows are
needed. At half-alpha 0..8, every possible channel difference rounds to
zero, so a single-source walk is exact. The same kernel handles spans
without a second refraction row. Its loop is unrolled for the 486 while
retaining the original AB3DI.s texturedwater phase indexing and 16-bit
wateroff carry. No visual approximation or animation timing changes are
introduced.

The independent DOS oracle checks all 65,536 native RGB565 inputs,
65,536 alpha/colour pairs, all 5,632 shared Brighten colours and 768
single/blended spans with wrapping, aliases, clipping and guards. All
192 simulation hashes and 195 complete pool-view images match at three
camera angles. Matched diagnostics with detailed profiling and captures
disabled measure:

| Pool yaw | Previous loop ms | Updated loop ms |
| --- | ---: | ---: |
| 0 | 74.103 | 67.867 |
| 1024 | 58.672 | 55.417 |
| -1024 | 60.011 | 56.519 |

These are fixed 40,000-cycle normal/486_slow, 4 MB, 320x240 RGB565,
SB16/mouse runs with host sound muted, averaging frames 8-63 of the same
64 updates. The source is committed as `ac4da24` in `build/dos-water2-source`.

The completed PGO releases were also compared by relinking the same
diagnostic entry point against each release's unchanged optimized engine
objects. Profiling and captures were disabled for these measurements:

| Pool yaw | Previous release loop ms | Updated release loop ms |
| --- | ---: | ---: |
| 0 | 73.082 | 64.787 |
| 1024 | 57.889 | 53.494 |
| -1024 | 59.310 | 54.636 |

This reduces pool-view frame time by 7.6-11.3% at the same fixed-cycle
settings. Native text/data/BSS are 1,184,280/22,016/291,840 bytes: total
static memory falls by 12,800 bytes. The shared-assets checkpoint reports
120 KiB free physical memory versus the previous release's 108 KiB.

The separate 128-frame real-time starting-view benchmark remains stable:
26.04 to 26.03 FPS, 38.40 to 38.42 ms per frame and 38.91 to 38.93 ms p95.
This change targets water-heavy views rather than the starting room.

All 22 ordinary and optimized native captures and all 22 indexed captures
match the original start/turn references. Full 4 MB level-1/4/15 fixed
exercises and the real-time level-4 SB16/mouse exercise pass, including
save/reload and UI checks, with zero profile audio underruns. Level-1/4
maps and all six saved level/map payloads remain exact. Windows Release
and the feedback suite pass. Both packages retain the 231 prepared assets,
238 archive entries and generated defaults; private diagnostic markers
are absent. Local mouse settings are preserved during staging.

The native EXE SHA-256 is
`994925f074121a5f52687b3b20303e948a9dc9126a551f55adb7dd2621f6c00c`.
Local evidence is in `build/dos/water2-performance.json` and
`build/dos/water2-validation.json`. Physical 486 hardware is unmeasured.

### Keep native billboard run state in the decoder

The native DOS row sampler now retains its transparent/opaque run state
in separate assembly loops. It reads the source row into EDX once per
unique row, and closes spans on cutouts or the final opaque sample. This
removes the previous stack flag checks and repeated row-index loads from
BitMapObj sampling. Source index zero, packed modes, clipping, bounds,
opaque black, repeated rows and reverse-pass coverage keep their existing
semantics. No heap allocation, palette bank or static data is added.

The independent raster oracle now covers 308 row cases, including 20
explicit run-boundary cases: all-transparent/all-opaque rows, alternating
cutouts at the full 160-span capacity, odd/even widths, rejected rows,
repeated rows, opaque black and RGB565 word one. The 768 column and 32
reverse compositor cases and the other arithmetic/raster checks pass.
Five matched source diagnostics cover the level-1 starting, enemy and
pool views and barrel bursts on levels 1 and 6. All 320 simulation hashes
and 325 complete images match. Production kernel changes are committed as
`e119fc3` and `b79c997`; the expanded oracle is `e32b83f`.
Both optimized releases also match all 128 simulation hashes and 130 complete
images in the level-1/6 barrel-burst comparisons.

Two related experiments were rejected. A paired-pixel palette and STOSL
expansion made the enemy and burst views slower overall. Lowering the
reverse-pass object cutoff from 16 to four slowed the enemy view; eight
provided no useful burst gain. The existing cutoff and pixel-copy path
are retained.

The completed PGO releases were compared using the same diagnostic entry
point and each release's unchanged optimized engine objects. Profiling and
captures were disabled for timings; means cover frames 8-63 of 64 identical
updates at normal/486_slow, fixed 40,000 cycles, 4 MB, 320x240 RGB565,
SB16 and mouse, with host sound muted:

| View | Previous release loop ms | Updated release loop ms |
| --- | ---: | ---: |
| Level-1 starting room | 38.731 | 38.638 |
| Close barrel / approaching enemies | 38.652 | 37.668 |
| Pool north shore | 64.774 | 64.470 |
| Level-1 barrel burst | 58.307 | 57.369 |
| Level-6 barrel burst | 52.190 | 51.484 |

Enemy-view frame time falls 2.5%. Level-1 burst p95 improves from 112.942 to
109.118 ms and its worst frame from 124.082 to 123.305 ms. Level-6 burst p95
falls from 76.315 to 74.070 ms and its worst frame from 79.537 to 77.338 ms.
The early level-1 burst remains expensive; this improves the decoder without
claiming to eliminate that pause. The independent 128-frame real-time
starting benchmark improves from 26.03 to 26.10 FPS, 38.42 to 38.32 ms per
frame and 38.93 to 38.83 ms p95, passing the 25.80 FPS acceptance gate.

All 22 ordinary and optimized native captures match the verified original
start/turn references. The full 4 MB level-1/4/15 fixed exercises and real-time
level-4 SB16/mouse exercise pass with zero profile audio underruns, including
repeated quickloads, all five autosave slots, UI and weapon checks. Level-1/4
maps and all six saved level/map payloads remain exact. Native text/data/BSS
remain 1,184,280/22,016/291,840 bytes. The unchanged indexed package and both
packages' 231 assets, 238 archive entries, generated defaults and absent
diagnostic markers are verified. This change is inside the existing native
DOS guard; modern and indexed rendering source is unchanged.

The native EXE SHA-256 is
`79b43201f73076bee1b487877f3e49a4530ad35a05f3d77325580793ec8c871c`.
Local evidence is in `build/dos/sprite2-performance.json` and
`build/dos/sprite2-validation.json`. Physical 486 hardware is unmeasured.

### Reuse exact copy ranges across repeated covered billboard rows

The next barrel-burst diagnostic counts 2,937,727 requested batch-copy
pixels but only 1,365,857 fresh writes across frames 8-63. At the worst
frame, about 70% of visited mask words contain no fresh bits. Among repeated
source-row groups, 31,724 adjacent destination rows have identical coverage
versus 3,112 with different coverage: 91.1% match.

The native DOS compositor now records the first row's fresh copy ranges
and replays them while following rows have identical prior coverage.
Different masks start a new set of ranges. Complete mask words are compared,
including bits outside the clipped rectangle; closed-word counts are
adjusted per destination row. Original BitMapObj index-zero cutouts,
opaque black, draw ordering and per-room mask resets remain exact.
The bounded range list and mask snapshot use 1,320 bytes of temporary
stack storage, with no heap allocation or persistent table.

The initial C-library comparison made the burst slower. Disassembly shows
DJGPP memcmp walking byte by byte. The selected helpers compare and copy
32-bit mask words with CMPSL/MOVSL. Matched source diagnostics then reduce
burst mean/p95/worst loop time from 58.026/109.167/118.845 ms to
56.362/95.957/102.601 ms. The ordinary enemy view stays at 38.485 ms.
These diagnostics use the same fixed-cycle settings as the release
comparisons below, with profiling and captures disabled for timing.

The DOS raster oracle adds 18 repeated-band cases, covering all 160 fresh
copy ranges, changing masks, partial mask words, coverage outside clips,
different closed-word counts, opaque black and guards. The existing 308
row, 768 column, 32 reverse compositor and other raster/arithmetic checks
pass. All 320 simulation hashes and 325 complete images match in the five
level-1/6 starting, enemy, pool and barrel-burst scenes. The implementation
and oracle are committed as `621903c`.
The optimized release-object barrel comparisons also preserve all 128
simulation hashes and 130 complete images on levels 1 and 6.

The completed PGO release comparison relinks the same diagnostic entry
point against each release's unchanged optimized engine objects. Detailed
profiling and captures are disabled during timing. The mean loop times
cover frames 8-63 of 64 identical updates at normal/486_slow, fixed 40,000
cycles, 4 MB, 320x240 RGB565, SB16/mouse and muted host sound:

| View | Previous release loop ms | Updated release loop ms |
| --- | ---: | ---: |
| Level-1 starting room | 38.628 | 38.630 |
| Close barrel / approaching enemies | 37.636 | 37.640 |
| Pool north shore | 64.490 | 64.493 |
| Level-1 barrel burst | 57.338 | 55.010 |
| Level-6 barrel burst | 51.517 | 51.515 |

Level-1 burst p95 falls from 109.166 to 97.043 ms and its worst frame from
123.335 to 101.576 ms, a 17.6% reduction. Early burst frames 8-15 average
86.064 versus 80.071 ms. Sparse and ordinary views retain their existing
cost; the separate real-time 128-frame starting benchmark stays at
26.10 FPS, 38.32 ms per frame and 38.83 ms p95, passing the 25.80 FPS gate.

All 22 ordinary and optimized native captures match the verified original
references. Full 4 MB level-1/4/15 fixed exercises and the real-time level-4
SB16/mouse exercise pass with zero profile audio underruns, including
repeated quickloads, autosaves, UI and weapons. Level-1/4 maps and all six
saved level/map payloads remain exact. Both packages retain 231 prepared
assets, 238 archive entries, generated defaults and no private diagnostic
markers. The indexed package and modern/indexed rendering source are
unchanged. Local mouse settings are preserved when staging.

Native text/data/BSS remain 1,184,280/22,016/291,840 bytes, and the
shared-assets checkpoint still reports 120 KiB free physical memory.
The native EXE SHA-256 is
`18bbd6a8f5cfe3d8c8fc3ba69a40872d17dc6e579d0155af0cefc2066149107d`.
Local evidence is in `build/dos/copy-performance.json` and
`build/dos/copy-validation.json`. Physical 486 hardware is unmeasured.

The broader consistency investigation remains active. Private coarse
profiling attributes about 17.6 ms to object updates immediately after
the barrel explosion. A separate probe attributes most of that work to
substepped movement: 1,127 calls take 4,677 movement steps across the
64-frame burst, predominantly two to six steps per call. These diagnostics
add profiling overhead and identify the next investigation; gameplay and
movement have not been changed here.

### Reject movement non-hits before room and height work

The movement changes and polygon scratch fix are validated together. The
former explosion-image discrepancy was traced to failed
polygon scratch allocations in the earlier release-object diagnostic at
4 MB, as described below.

The movement profile identifies wall checks as the largest part of the
projectile/debris update cost. The port now defers destination-room lookup
and crossing-height calculation until the existing primary wall/segment
checks accept a possible hit. FindRoom likewise tests its existing crossing
gate before resolving the exit's zone ID. Room transition order, signed
word arithmetic, hit positions, bounce results, four-unit substeps and the
32-step limit are preserved from the current ObjectMove.s translation.

The near-wall flag test also avoids a software wide division. With positive
side and denominator, `side / denominator < threshold` is exactly
`side < denominator * threshold`. The existing denominator clamp and
position scales 0/16 keep the product within int64_t. Wall flags at floorline
offset 14 retain their original OR semantics. Crossing height returns the
existing Y immediately when old and new Y are equal; moving-Y division
order and clamping are retained.

Independent division oracles check 65,984 wall-touch cases, including exact
threshold boundaries, signed lengths/extents, clamped denominators, both
coordinate scales and all 16 line bytes. Another 32,768 crossing-height
cases cover signed division order, zero vertical motion/divisors and clamps.
These run in both the native 4 MB DOS raster suite and WebAssembly feedback
suite. All previous raster/compositor checks, Windows Release, and the full
movement/shooting/stairs/doors/robot/corpse feedback checks pass. The code and
tests are committed as `5b11810`.

Five matched source scenes preserve all 320 simulation hashes and 325
complete images. Separate burst fixtures also capture the complete mutable
level data, floorlines, zone table, points, objects and both shot pools,
plus normalized player coordinates/room/angle/energy, after every update.
These records compare bytes directly and include the wall-touch flags.

The two source burst fixtures preserve all 128 simulation hashes, 130
complete images and 128 mutable-state snapshots. The fresh native build
also passes the 22 ordinary/optimized reference captures, 4 MB level-1/4/15
exercises, and level-4 real-time SB16/mouse exercise. Windows Release,
the indexed build's 22 reference captures, package assets, and saved
level/map payload checks pass.

The initial optimized-object timing fixture carried extra profiling arrays.
Its level-1 barrel mean improved from 54.932 to 53.333 ms, but its memory
pressure allowed the compared builds to drop different polygon draws.
The final measurements below use a smaller timing entry point and include
the polygon scratch fix; the initial figures are not release-speed claims.

The initial optimized level-1 burst comparison differed at `EN23.BMP`: 63
pixels within x=79..139, y=101..116. All 64 simulation hashes and complete
mutable-state snapshots match, as do the captured previous/current object
points and nonpointer game state. Precision changes and unprofiled relinks
changed the image, but these experiments also changed the memory layout.
Direct probes in the unchanged release binaries show that billboard draws
match, and the first divergent pixels come from a polygon object between
two billboard passes. Renderer state, wall-depth spans, object records,
rotated points, texture/palette bytes and model data all match.

### Keep DOS polygon scratch on the reserved stack

The direct allocation-return probe establishes the cause: the earlier
release-object diagnostic at 4 MB returns NULL for both 3,840-byte
PolygonObj scratch allocations in the disputed frame. It silently skips those models and
retries allocation for each object. The movement candidate successfully
allocates the scratch once and draws the models. Running the earlier
executable at 16 MB produces exactly the candidate's disputed image.
The reference for this case is the larger-memory run, where the authored
models are present; a missing model cannot be an image-parity requirement.
The diagnostic adds profiling storage and its failure does not establish
that the shipped executable drops these models in the same scene.

DOS now uses a bounded local PolyThreadContext on its already reserved
128 KiB stack. The original rotobj/convtoscr loops initialize all fields
used by the model's validated point indices. This needs no additional
persistent table or heap allocation and avoids the TLS lookup, silent
failure and repeated allocation attempts during drawing. Other platforms
retain their existing per-thread context. The change is `df80b72`.

`tools/check_dos_models.py` relinks only a scene entry point against the
actual release engine objects. It detonates authored barrels on levels
1 and 6 through object_handle_barrel, runs 64 fixed updates per scene with
SB16/mouse and 32 KiB of extra static residency, and compares every complete
native RGB565 framebuffer and simulation hash at 4 and 16 MB. Frames are
written directly as complete 320x240 native buffers, avoiding BMP
conversion overhead during the audio exercise. This exercises late scratch
allocation under memory pressure. It is an image/memory oracle, not a
timing benchmark.
Local allocation, primitive and memory evidence remains under
`build/dos-move-trace/`; probes are absent from the playable executable.

The permanent oracle reproduces the earlier engine's level-1 frame-23
failure and passes all 128 simulation hashes and complete framebuffers
on levels 1 and 6 with the fix. It is committed as `d1b2cca` and can be run
after a native PGO build:

```powershell
python tools/check_dos_models.py --build-dir build/dos/pgo --dosbox <dosbox.exe>
```

Matched release-object timings use normal/486_slow, fixed 40,000 cycles,
4 MB, 320x240 RGB565, SB16/mouse and muted host sound. The smaller entry
point holds 512 bytes of timing samples, captures only frame 23 outside
the timed update, and has no residency stress or detailed profiling.
That frame matches byte for byte between the timed builds. Results cover
frames 8-63 of 64 identical fixed updates:

| View | Previous mean/p95/worst ms | Updated mean/p95/worst ms |
| --- | ---: | ---: |
| Level-1 barrel burst | 55.120 / 97.354 / 102.317 | 55.470 / 94.657 / 99.481 |
| Level-6 barrel burst | 51.492 / 74.053 / 77.216 | 50.879 / 71.413 / 73.673 |

Level-1 early burst frames 8-15 improve from 80.086 to 74.924 ms. Its
overall mean rises by 0.350 ms while p95/worst improve by about 2.8%.
Level-6 early burst mean improves from 73.444 to 71.217 ms. These results
support steadier burst updates, rather than a broad average-FPS increase.
The separate unmodified 128-frame starting-room benchmark passes the
25.80 FPS gate at 26.09 FPS, 38.33 ms mean and 38.84 ms p95, versus
26.10 FPS, 38.32 ms and 38.83 ms before this change.

The fresh native build completes all 23 compiler-feedback training runs,
retains all 22 ordinary/optimized and original-reference captures, and
passes the 4 MB level-1/4/15 exercises and real-time level-4 SB16/mouse
exercise with zero profile audio underruns. All 22 indexed references,
Windows Release, package assets and level/map save payload checks pass.
The movement arithmetic/feedback checks above also pass.

Native text/data/BSS are 1,182,232/22,016/292,864 bytes: total static
storage is 1 KiB smaller than the previous release. The shared-assets
checkpoint still reports 120 KiB free physical memory. The native EXE
SHA-256 is `7207a8b0eeeb01648b0733367e10abeeba11ac7bf9696a6b757c8b0098c7fbb9`;
the indexed EXE is
`c240b01e8cd950083630c3e946a517a721cd3fc10afc3369367a5e2120148b09`.
Both archives retain 231 prepared assets and 238 entries. Existing local
settings and saved games are preserved when staging.

Local evidence is in `build/dos/move-performance.json`,
`build/dos/move-validation.json` and the isolated build's
`build/dos/pgo/model-check/report.json`. Further burst-cost investigation
remains active; physical 486 hardware is unmeasured.

### Reuse an exact single-span billboard spill clip

Spill drawing first narrows the billboard to the destination room geometry.
When that produces one range, DOS now passes the resulting exact bounds
through to the rasterizer without walking the same geometry again. Multiple
ranges retain their original per-span mask, and spill wall-depth clipping
remains enabled. This adds no persistent table or heap allocation.

Matched diagnostics retain the existing release engine objects and rebuild
both baseline and candidate renderers with the same unprofiled options.
At normal/486_slow, 40,000 cycles, 4 MB, native 320x240 and SB16/mouse,
the level-1 barrel burst changes from 56.996/95.623/99.716 ms mean/p95/worst
to 56.781/94.534/98.826 ms across fixed updates 8-63. The ordinary close
view and level-6 burst are effectively unchanged. These are diagnostic
results, not measurements of a newly trained PGO release or physical hardware.

All 320 complete RGB565 framebuffers and simulation hashes match between
the diagnostic builds: 64 updates each in the level-1 starting, close-enemy
and water views, plus barrel bursts on levels 1 and 6. Every run reports
zero audio underruns. Local evidence is in
`build/dos-sprite3-trace/narrow-validation.json` and the corresponding
`renderer-base`/`narrow` scene directories. The model scene fixture also
includes the headers declaring its level and barrel APIs.

### Prove billboard row bounds once per repeated band

The native decoder now finds the minimum valid source-column limit during
descriptor setup. A repeated source row within that limit needs one bounds
check for the band, instead of one per column. Setup also records whether
all source columns are nonempty, allowing that band to omit repeated null
checks. Empty columns, truncated data and negative/out-of-range rows keep
their original checked path. These changes are `adf5532` and `2b76b00`.
They add no persistent table or heap allocation.

A fresh native PGO build includes the spill-clip reuse above. Matched
release-engine diagnostics use normal/486_slow, 40,000 fixed cycles, 4 MB,
320x240 RGB565, SB16/mouse and muted host sound. Both entry points keep
512 bytes of timing samples and capture only frame 23 outside the timed
update. Results cover fixed updates 8-63:

| View | Previous mean/p95/worst ms | Updated mean/p95/worst ms |
| --- | ---: | ---: |
| Starting room | 38.663 / 38.720 / 38.721 | 38.611 / 38.669 / 38.669 |
| Close enemies/barrel | 39.730 / 40.679 / 40.790 | 39.344 / 40.338 / 40.437 |
| Pool surface | 64.229 / 65.352 / 65.938 | 64.110 / 65.023 / 65.245 |
| Level-1 barrel burst | 55.466 / 94.647 / 99.472 | 54.615 / 92.220 / 97.414 |
| Level-6 barrel burst | 50.863 / 71.404 / 73.653 | 50.512 / 70.320 / 70.548 |

The separate, unmodified 128-frame starting-room test passes the 25.80 FPS
gate at 26.13 FPS, 38.28 ms mean and 38.79 ms p95, versus 26.09 FPS,
38.33 ms and 38.84 ms before. These are small gains; expensive burst frames
remain, and physical 486 hardware is unmeasured.

All 320 complete release-engine images and simulation hashes match across
the five views. The 32 KiB residency-stress model oracle also retains all
128 image/hash comparisons at 4 versus 16 MB on levels 1 and 6. The native
486 raster suite passes, including 308 row, 768 column, 32 reverse-batch
and 18 repeated-band cases. Fresh compiler feedback completes 23 training
runs and 22 ordinary/optimized captures; all 22 original native and 22
indexed references remain exact. The real-time level-4 sound/mouse check
and full 4 MB level-1/4/15 exercises pass with zero profile audio underruns.
Original assets, map images and all six saved level/map payloads on levels
1 and 4 retain their earlier checks.

Native text/data/BSS are 1,183,768/22,016/292,864 bytes: 1.5 KiB more code,
with unchanged static data and BSS. The shared-assets checkpoint still
reports 120 KiB free physical memory. The staged native EXE SHA-256 is
`45c8da02bd757f4b41e831582f8f436cb6b4c3ad92e285bc5640925d53d65d3d`;
the indexed EXE is
`13e7b28467d1cad540839f8978b8aa8567e05a1c39ad763e2b7aac88f7e981f4`.
Both archives retain 231 prepared assets and 238 entries. Staging preserves
all 241 native and 238 indexed non-EXE game files, including local settings
and saves.

Local evidence is in `build/dos/sprite3-performance.json`,
`build/dos/sprite3-validation.json`, `build/dos/sprite3-staging.json` and
`build/dos-sprite3-source/build/dos/pgo/model-check/report.json`. Diagnostic
and source-format audit files remain under `build/dos-sprite3-trace/`.

### Advance magnified billboard descriptors by repeated source columns

Descriptor setup previously advanced the source DDA once per destination
pixel, even when many adjacent pixels sampled the same source column.
The native decoder now computes the first repeated-column run, then uses
the quotient and remainder of 65536/du to advance subsequent runs exactly.
Clamping and unsigned 32-bit wrapping retain their original behavior;
minified sprites keep their original traversal. The change is `7854d5b`.
It adds no persistent table or heap allocation.

Matched diagnostic profiling puts descriptor setup in the worst level-1
burst frame at 7.565 ms before and 3.621 ms afterwards. Coverage reset is
only 0.132 ms in that frame and is unchanged. Those probes add overhead
and are absent from the playable build.

Fresh native release-engine comparisons use normal/486_slow, 40,000 fixed
cycles, 4 MB, 320x240 RGB565, SB16/mouse and muted host sound. Each entry
point retains 512 bytes of timing samples, captures only frame 23 outside
the timed update, and reports fixed updates 8-63:

| View | Previous mean/p95/worst ms | Updated mean/p95/worst ms |
| --- | ---: | ---: |
| Starting room | 38.611 / 38.669 / 38.669 | 38.648 / 38.705 / 38.705 |
| Close enemies/barrel | 39.344 / 40.338 / 40.437 | 39.270 / 40.265 / 40.367 |
| Pool surface | 64.110 / 65.023 / 65.245 | 64.126 / 65.036 / 65.260 |
| Level-1 barrel burst | 54.615 / 92.220 / 97.414 | 54.334 / 86.629 / 93.064 |
| Level-6 barrel burst | 50.512 / 70.320 / 70.548 | 50.320 / 69.464 / 69.819 |

The improvement is concentrated in the largest magnified sprites:
level-1 burst p95 falls by about 6.1% and its worst frame by 4.5%.
The separate real-time starting-room test changes from 26.13 to 26.09 FPS,
38.28 to 38.33 ms mean and 38.79 to 38.85 ms p95, retaining the 25.80 FPS
gate. Starting and water views therefore show small regressions, while
the expensive burst frames improve. Physical 486 hardware is unmeasured.

The independent per-pixel row oracle now includes 144 additional fractional,
clamped-column and unsigned-wrap combinations: all 452 row cases pass,
alongside the original column, batch, water, floor and movement checks.
The arithmetic/pixel regression runs at fixed 200,000 cycles to accommodate
its expanded workload; performance comparisons above remain at 40,000.
The test flushes progress lines so a slow or failed run can be localized.

All 320 complete release-engine images and simulation hashes match across
the five views. All 128 residency-stress image/hash comparisons at 4/16 MB
on levels 1 and 6 remain exact. Fresh native compiler feedback completes
23 training runs and all 22 ordinary/optimized captures, also matching the
original native references. The real-time level-4 sound/mouse exercise
and full 4 MB level-1/4/15 checks pass with zero profile audio underruns.
Both packages retain the original 231 assets, map and saved level/map
payload checks. The indexed rebuild is byte-for-byte identical to its
previous executable, so its existing 22 reference captures remain valid.

Native text/data/BSS are 1,184,280/22,016/292,864 bytes: 512 more code bytes,
with unchanged data and BSS. Shared-assets free physical memory remains
120 KiB. The staged native EXE SHA-256 is
`4ae5bea86c2bf81cc8d8180d47b6828463c57d716b6b3ec4b9f071bf222f88da`;
the indexed EXE remains
`13e7b28467d1cad540839f8978b8aa8567e05a1c39ad763e2b7aac88f7e981f4`.
Staging preserves all 241 native and 238 indexed non-EXE game files,
including local settings and saves.

Local evidence is in `build/dos/sprite4-performance.json`,
`build/dos/sprite4-validation.json`, `build/dos/sprite4-staging.json` and
`build/dos-sprite4-source/build/dos/pgo/model-check/report.json`.
Diagnostic profiling remains under `build/dos-sprite4-trace/`.

### Reject separated spill bounds before wide geometry arithmetic

DOS now checks inclusive bounds before segment orientations and point-edge
cross products. Capsule admission first checks the radius-inflated bounds,
widening the additions to avoid signed 32-bit overflow. Separated shapes
return immediately; overlapping or touching bounds retain the original
wide geometry tests. This avoids unnecessary multiplications and software
division during billboard spill checks, without a new cache or heap allocation.
The changes are `aa4f04b` and `8e88664`.

`tools/check_dos_geometry.py` compiles the actual renderer helpers with the
unused renderer discarded. At 4 MB on normal/486_slow it checks 39,329
geometry cases against the original definitions, including endpoint order,
collinear and zero-length segments, point ownership and large coordinates.
Another 3,750 capsule cases cover squared lengths around 2^31 and signed
radius extremes. The tool uses fixed 200,000 cycles for this arithmetic
oracle and verifies both the success marker and DOS exit status:

```powershell
python tools/check_dos_geometry.py --dosbox <dosbox.exe>
```

Fresh release-engine timings use normal/486_slow, 40,000 fixed cycles,
4 MB, native 320x240 RGB565, SB16/mouse and muted host sound. Both entry
points retain 512 bytes of timing samples, capture only frame 23 outside
the timed update, and report updates 8-63. Turning views hold the original
right-turn input throughout the same 64 fixed updates.

| View | Previous mean/p95/worst ms | Updated mean/p95/worst ms |
| --- | ---: | ---: |
| Starting room | 38.648 / 38.705 / 38.705 | 38.573 / 38.630 / 38.630 |
| Close enemies/barrel | 39.270 / 40.265 / 40.367 | 39.134 / 40.146 / 40.241 |
| Pool surface | 64.126 / 65.036 / 65.260 | 63.902 / 64.820 / 65.039 |
| Level-1 barrel burst | 54.334 / 86.629 / 93.064 | 53.221 / 87.474 / 88.431 |
| Level-6 barrel burst | 50.320 / 69.464 / 69.819 | 49.762 / 68.603 / 68.967 |
| Level-1 burst, turning | 55.799 / 92.886 / 103.250 | 54.706 / 92.843 / 99.320 |
| Level-6 burst, turning | 63.516 / 85.915 / 88.809 | 63.188 / 85.058 / 87.956 |
| Close enemies, turning | 39.736 / 48.490 / 48.588 | 39.634 / 48.368 / 48.466 |

Means and worst frames improve across these views. Stationary level-1
burst p95 rises by 0.845 ms while its mean falls by 1.113 ms and worst
frame falls by 4.633 ms. The unmodified 128-frame starting-room test
passes the 25.80 FPS gate at 26.13 FPS, 38.27 ms mean and 38.78 ms p95,
versus 26.09 FPS, 38.33 ms and 38.85 ms before. Physical hardware remains
unmeasured, and expensive burst frames remain.

All 512 complete release-engine images and simulation hashes match
across the eight views. The 32 KiB residency-stress model oracle retains
all 128 image/hash comparisons between 4 and 16 MB on levels 1 and 6.
Fresh compiler feedback completes 23 training runs and all 22
ordinary/optimized captures; all original native and indexed references
remain exact. The real-time 4 MB level-4 SB16/mouse exercise and full
level-1/4/15 exercises pass with zero profile audio underruns. Windows
Release also builds. Both packages retain the original 231 assets and
238 archive entries; map images and the six saved level/map payloads on
levels 1 and 4 retain their earlier comparisons.

Native text/data/BSS are 1,184,792/22,016/292,864 bytes: 512 more code bytes
and unchanged persistent data. Shared-assets free physical memory remains
120 KiB. The staged native EXE SHA-256 is
`82ecf263981519e8fb3a46e753e11b2a54ba74dbb50fce187dde3e0a79a2c7f0`;
the indexed EXE is
`c14000e45459a7b2835cfd1a3230e4855dded488ac4e740a94a4802894ccb5f3`.
Staging preserves all 241 native and 238 indexed non-EXE game files,
including local settings and saves.

Local evidence is in `build/dos/sprite5-performance.json`,
`build/dos/sprite5-validation.json`, `build/dos/sprite5-staging.json` and
`build/dos-sprite5-source/build/dos/pgo/model-check/report.json`.
Diagnostic profiling remains under `build/dos-sprite5-trace/`.

### Reject invalid billboard spills before sorting and draw preparation

DOS now performs the existing bitmap spill admission check during object
and shot selection, before sorting, vertical projection, shading and frame
setup. Accepted entries reuse that result during drawing. The check uses
the same cached dimensions, horizontal projection and signed-16-bit clamp
as the final draw path. Polygon sprites and pick/debug rendering retain
their original path. The change is `0e4ebad`.

The native batching threshold still uses the original candidate count,
projected area and polygon presence before this additional rejection.
Dropping an invisible entry therefore does not change which batching
algorithm the surviving entries use. The admission flag occupies existing
object-entry padding; a compile-time check keeps each entry at 32 bytes.
No persistent table or heap allocation is added. Axis-aligned segment and
point-edge tests also return directly after their inclusive bounds prove
intersection; other overlapping geometry retains the original wide math.

The geometry oracle now covers 101,829 cases, including 62,500 additional
large-coordinate horizontal/vertical combinations with reversed endpoints
and degenerate segments. All pass against the original definitions, as do
the separate 3,750 capsule boundary and signed-radius cases.

Matched native release-engine measurements use normal/486_slow, 40,000
fixed cycles, 4 MB, 320x240 RGB565, SB16/mouse and muted host sound. Both
entry points retain 512 bytes of timing samples and capture only frame 23
outside the timed update. Results cover fixed updates 8-63; turning holds
the original right-turn input throughout the same 64 updates.

| View | Previous mean/p95/worst ms | Updated mean/p95/worst ms |
| --- | ---: | ---: |
| Starting room | 38.573 / 38.630 / 38.630 | 38.484 / 38.540 / 38.541 |
| Close enemies/barrel | 39.134 / 40.146 / 40.241 | 39.028 / 40.033 / 40.126 |
| Pool surface | 63.902 / 64.820 / 65.039 | 63.722 / 64.642 / 64.863 |
| Level-1 barrel burst | 53.221 / 87.474 / 88.431 | 51.963 / 82.704 / 88.221 |
| Level-6 barrel burst | 49.762 / 68.603 / 68.967 | 49.096 / 67.451 / 67.811 |
| Level-1 burst, turning | 54.706 / 92.843 / 99.320 | 53.570 / 88.339 / 98.802 |
| Level-6 burst, turning | 63.188 / 85.058 / 87.956 | 62.799 / 83.862 / 86.761 |
| Close enemies, turning | 39.635 / 48.369 / 48.465 | 39.555 / 48.257 / 48.354 |

Level-1 stationary burst p95 falls by 5.5%; its first eight burst updates
average 67.203 ms versus 70.644 ms. Turning burst p95 falls by 4.9%.
The worst burst frames improve much less and still approach 99 ms.
The unmodified 128-frame starting-room test passes the 25.80 FPS gate at
26.18 FPS, 38.20 ms mean and 38.72 ms p95, versus 26.13 FPS, 38.27 ms and
38.78 ms before. These remain emulator measurements; physical 486
hardware is unmeasured and a general threefold improvement is not proven.

All 512 release-engine framebuffers and simulation hashes match across
the eight scenes. A separate 512-frame diagnostic comparison is also
exact. The 32 KiB residency-stress oracle retains all 128 image/hash
comparisons at 4 versus 16 MB on levels 1 and 6. Fresh compiler feedback
completes 23 training runs and all 22 ordinary/optimized native captures;
all 22 original native and 22 freshly rebuilt indexed references match.
The real-time 4 MB level-4 SB16/mouse exercise and full level-1/4/15
exercises pass with zero profile audio underruns. Windows Release builds.
Both packages retain 231 original assets and 238 archive entries; map
images and all six saved level/map payloads on levels 1 and 4 remain exact.

Native text/data/BSS are 1,186,328/22,016/292,864 bytes: 1,536 more code
bytes and unchanged persistent data. The level-4 sound/mouse check reports
116 KiB free physical memory after shared assets, versus 120 KiB before,
and 24 KiB after the level is loaded. The staged native EXE SHA-256 is
`4aa706172e8c399e78d04d9d5e186639f22d13808f573fd3f2ad9de8df48802b`;
the indexed EXE is
`2b03b3b597bb424dcf1b1eb4419cfead03e13450df7f4b2c920c461b18289e8e`.
Staging preserves all 241 native and 238 indexed non-EXE game files,
including local settings and saves.

Local evidence is in `build/dos/sprite6-performance.json`,
`build/dos/sprite6-validation.json`, `build/dos/sprite6-staging.json` and
`build/dos-sprite6-source/build/dos/pgo/model-check/report.json`.
Diagnostic profiling remains under `build/dos-sprite6-trace/`.

### Reduce repeated billboard room selection and projection work

`f37e163` removes repeated preparation from the ObjDraw/BitMapObj path:
room/section clip validation runs once for the whole object pass, and
shot sizing follows the existing room, section and near-plane rejects.
DOS also reuses the already cached projected width during final bitmap
drawing when `EXPLOSION_SIZE_CORRECTION == 1`.

Adjacent-room membership for IDs 0-255 uses an eight-word stack mask
instead of repeatedly searching the connected-room list. Other signed
zone IDs keep the original list search. The mask adds 32 stack bytes,
with no persistent table, heap allocation or larger object-entry layout.
The geometry oracle now verifies 1,179,648 membership queries against an
independent linear scan: every signed 16-bit ID, six list lengths including
zero, and ordered, reversed and duplicate-heavy lists. The original
101,829 geometry and 3,750 capsule cases also pass.

Diagnostic profiling of level-1 burst frame 18 puts object selection at
7.504 ms versus 8.385 ms in the previous diagnostic build. Sprite drawing
still takes 35.614 ms. These probes add overhead and are absent from the
playable executable.

Matched release-engine timings use normal/486_slow, 40,000 fixed cycles,
4 MB, 320x240 RGB565, SB16/mouse and muted host sound. Both entry points
keep 512 bytes of timing samples and capture frame 23 outside the timed
update. Results cover fixed updates 8-63; turning views hold the original
right-turn input throughout all 64 updates.

| View | Previous mean/p95/worst ms | Updated mean/p95/worst ms |
| --- | ---: | ---: |
| Starting room | 38.484 / 38.540 / 38.541 | 38.511 / 38.567 / 38.568 |
| Close enemies/barrel | 39.028 / 40.033 / 40.126 | 39.026 / 40.031 / 40.126 |
| Pool surface | 63.722 / 64.642 / 64.863 | 63.760 / 64.683 / 64.902 |
| Level-1 barrel burst | 51.963 / 82.704 / 88.221 | 51.518 / 81.999 / 87.451 |
| Level-6 barrel burst | 49.096 / 67.451 / 67.811 | 48.950 / 67.259 / 67.619 |
| Level-1 burst, turning | 53.570 / 88.339 / 98.802 | 53.192 / 87.673 / 98.018 |
| Level-6 burst, turning | 62.799 / 83.862 / 86.761 | 62.604 / 83.640 / 86.541 |
| Close enemies, turning | 39.554 / 48.257 / 48.352 | 39.553 / 48.254 / 48.352 |

The gain is modest and concentrated in bursts: level-1 first-eight-burst
updates average 66.423 ms versus 67.203 ms, and its worst turning frame
falls by 0.784 ms. Starting-room and pool means regress by 0.027 and
0.038 ms. The separate unmodified 128-frame starting-room test retains
the 25.80 FPS gate at 26.15 FPS, 38.23 ms mean and 38.74 ms p95, versus
26.18 FPS, 38.20 ms and 38.72 ms before. Its worst frame rises from
77.60 to 78.13 ms. Expensive burst frames still approach 98 ms; a general
threefold improvement and physical 486 performance remain unproven.

All 512 complete release-engine framebuffers and simulation hashes match
across the eight views, as do all 512 diagnostic comparisons. The 32 KiB
residency-stress oracle retains all 128 image/hash pairs between 4 and
16 MB on levels 1 and 6. Fresh compiler feedback completes 23 training
runs and all 22 ordinary/optimized native captures; all 22 original native
and 22 freshly rebuilt indexed references match. The real-time 4 MB
level-4 SB16/mouse exercise and full level-1/4/15 exercises pass with zero
profile audio underruns. Windows Release also builds. Both packages keep
231 original assets and 238 archive entries, with unchanged map images
and all six saved level/map payloads on levels 1 and 4.

Native text/data/BSS remain 1,186,328/22,016/292,864 bytes. The level-4
sound/mouse check still reports 116 KiB free physical memory after shared
assets and 24 KiB after loading the level. The staged native EXE SHA-256 is
`e2660946c9ed236e3a1780788c08810f122daabb6a1b3b4c295dd70668c9d886`;
the indexed EXE is
`74fae2a6061e729d13e4495b6b764e56ef7c4a8c104ee07ba4f1ee8b482c00e8`.
Staging preserves all 241 native and 238 indexed non-EXE game files,
including local settings and saves.

Local evidence is in `build/dos/sprite7-performance.json`,
`build/dos/sprite7-validation.json`, `build/dos/sprite7-staging.json` and
`build/dos-sprite7-source/build/dos/pgo/model-check/report.json`.
Diagnostic profiling remains under `build/dos-sprite7-trace/`.

### Transport packed sprite words in native byte order

`c845469` moves sprite word byte swapping into `tools/dos_sprite.py` and
the DOS packaging step. All 14 WADs retain their original packing of three
five-bit pixels per word, PTR offsets and 240,344 payload bytes. Each transported
file adds an eight-byte `ABS1` version/length header, for 112 extra disk
bytes across the package. DOS validates and consumes that header before
allocating the original payload size. Old raw WADs fail with a specific
error instead of silently using the wrong byte order. Use the matching
EXE and DATA files from the updated package together.

`sprite_format.h` centralizes native DOS word reads/writes while keeping
desktop assets in Amiga order. The readers cover general billboard paths,
keys, weapon caches, RGBA extraction and `io_sprite_clear_texel` fixups.
The native row and column assembly samplers no longer rotate each word's
bytes. RGB565 and indexed packages use the same prepared sprite format;
palette derivation still reads original source bytes before conversion.

The source audit verifies every one of the 120,172 original words and
all referenced word alignments. The other 217 asset payloads remain exact.
The raster oracle also loads all 65,536 word values through the actual
preparer and checks native unaligned reads/writes and invalid headers.
Packed row/column, reverse-batch and repeated-band cases pass with the
native transport ABI, including malformed PTR views and guards. The test
runner now captures assertion diagnostics and checks DOS exit status.
An actual DOS startup test rejects an old raw gun WAD with exit status 1.

Matched release-engine timings use normal/486_slow, 40,000 fixed cycles,
4 MB, 320x240 RGB565, SB16/mouse and muted host sound. Each entry point
keeps 512 bytes of timing samples and captures frame 23 outside the timed
update. Results cover fixed updates 8-63; turning holds the original
right-turn input throughout the 64 updates.

| View | Previous mean/p95/worst ms | Updated mean/p95/worst ms |
| --- | ---: | ---: |
| Starting room | 38.511 / 38.567 / 38.568 | 38.500 / 38.556 / 38.557 |
| Close enemies/barrel | 39.026 / 40.031 / 40.126 | 38.832 / 39.817 / 39.931 |
| Pool surface | 63.760 / 64.683 / 64.902 | 63.689 / 64.612 / 64.833 |
| Level-1 barrel burst | 51.518 / 81.999 / 87.451 | 51.304 / 81.576 / 86.782 |
| Level-6 barrel burst | 48.950 / 67.259 / 67.619 | 48.842 / 66.722 / 70.212 |
| Level-1 burst, turning | 53.192 / 87.673 / 98.018 | 53.066 / 90.313 / 96.710 |
| Level-6 burst, turning | 62.604 / 83.640 / 86.541 | 62.500 / 83.096 / 85.662 |
| Close enemies, turning | 39.553 / 48.253 / 48.351 | 39.493 / 48.071 / 48.229 |

Means improve across these views, but stationary level-6 worst time rises
by 2.593 ms and turning level-1 p95 rises by 2.640 ms. A separate matched
sound-off diagnostic improves p95 and worst time in all four burst views:
worst frames change from 81.186/63.051/92.360/78.996 ms to
80.871/62.477/91.892/78.417 ms respectively. This suggests audio-related
pacing needs further investigation; it does not establish the cause of
the SB16 regressions. The unmodified 128-frame starting-room test passes
the 25.80 FPS gate at 26.19 FPS, 38.19 ms mean and 38.70 ms p95, versus
26.15 FPS, 38.23 ms and 38.74 ms before. Physical hardware is unmeasured.

All 512 release-engine framebuffers and simulation hashes match, as do
all 512 diagnostic comparisons. All 128 residency-stress image/hash pairs
at 4 versus 16 MB match on levels 1 and 6. Fresh compiler feedback completes
23 training runs and 22 ordinary/optimized captures; all 22 original native
and 22 fresh indexed references remain exact. Full 4 MB level-1/4/15 and
real-time level-4 SB16/mouse exercises pass with zero profile underruns.
Windows Release builds. Both packages retain 231 assets and 238 archive
entries. Map images and compared saved level/map chunks match for all six
saves on levels 1 and 4, excluding only unused bytes after negative door
and lift table terminators. Active table records and terminator words
remain exact; `level.c` initializes only the sentinel word in those tails.

Native text/data/BSS are 1,186,840/22,016/292,864 bytes: 512 more code bytes,
with unchanged persistent data. Shared-assets free physical memory remains
116 KiB; the level-4 loaded checkpoint reports 16 KiB versus 24 KiB before,
despite identical resident sprite payload sizes. This lower headroom and
the mixed burst pacing remain limitations of this release.

The staged native EXE SHA-256 is
`d03ce3c74ed8b73b07fd57665f90f5090f42be13881c3b07ab3f2342fe75b5fe`;
the indexed EXE is
`b19af1db3ac1570579d6c02d74bbb273df4c7858952582288d163154c28d023d`.
Staging refreshes the 14 prepared sprites with each executable and preserves
the other 227 native and 224 indexed non-EXE files, including settings and saves.

The original detailed diagnostic incorrectly attributed decoding time to
copying: its insertion hook no longer matched the source boundary. With
that hook repaired, the same burst-peak diagnostic reports 15.286 ms in
decoding and 15.982 ms in copying/coverage, plus 3.745 ms in setup. These
instrumented values include probe overhead. The release timings above
remain valid; the small byte-order gain does not resolve the remaining
burst stutter or prove a general threefold improvement.

Local evidence is in `build/dos/sprite8-performance.json`,
`build/dos/sprite8-validation.json`, `build/dos/sprite8-staging.json` and
`build/dos-sprite8-source/build/dos/pgo/model-check/report.json`.
Source-word, invalid-package, sound-off and diagnostic evidence remains
under `build/dos-sprite8-trace/`.

### Skip repeated billboard footprints within a reverse pass

`dos_raster_sprite_rows` now remembers four recent sampling footprints
within a reverse billboard pass. Repeating one cannot claim another opaque
pixel: coverage only increases during that pass. Matching includes geometry,
WAD/PTR identity and lengths, clipping, horizontal sampling, strip selection
and a private copy of the sampled rows. Palette differences do not change
source-index transparency. The existing whole-rectangle rejection runs
first, preserving visible/drawn accounting. Every new pass resets history;
source WAD/PTR data must remain immutable during a pass.

The history uses 4,104 bytes of the existing reserved DOS stack, with one
additional pointer in persistent state and no heap allocation. Ordinary
non-batch row drawing also decodes the first row directly into the
framebuffer, then copies opaque spans to repeated rows.

A captured level-1 barrel-burst frame contains 89 batch calls. Replaying
each call 64 times with its original incoming coverage/history state takes
22.991 ms versus 28.314 ms before, an 18.8% reduction in the recorded
billboard kernel. All framebuffer pixels, coverage bits, open-word counts
and visible/drawn tallies match. This is a kernel result, not overall FPS.
The replay uses normal/486_slow, 40,000 fixed cycles, 4 MB and no audio.

Across eight diagnostic scenes, all 512 complete framebuffers and
simulation hashes match, with zero reported profile audio underruns.
The 4 MB DOS raster oracle also passes 560 row cases, 768 column cases,
32 independent painter comparisons, 18 repeated-band cases and 216 cached
draw comparisons. The latter exercise palette changes, every sampling key,
reused row storage, eviction and pass reset.

Matched release-engine timings use normal/486_slow, 40,000 fixed cycles,
4 MB, 320x240 RGB565, SB16/mouse and muted host sound. Both entry points
keep 512 bytes of timing samples and capture frame 23 outside the timed
update. Results cover fixed updates 8-63; turning holds the original
right-turn input throughout all 64 updates.

| View | Previous mean/p95/worst ms | Updated mean/p95/worst ms |
| --- | ---: | ---: |
| Starting room | 38.500 / 38.556 / 38.557 | 38.486 / 38.542 / 38.543 |
| Close enemies/barrel | 38.832 / 39.817 / 39.931 | 38.559 / 39.551 / 39.672 |
| Pool surface | 63.689 / 64.612 / 64.833 | 63.650 / 64.571 / 64.792 |
| Level-1 barrel burst | 51.304 / 81.576 / 86.782 | 50.672 / 76.640 / 79.324 |
| Level-6 barrel burst | 48.842 / 66.722 / 70.212 | 48.598 / 65.699 / 69.556 |
| Level-1 burst, turning | 53.066 / 90.313 / 96.710 | 52.357 / 82.579 / 90.544 |
| Level-6 burst, turning | 62.500 / 83.096 / 85.662 | 62.130 / 82.206 / 84.607 |
| Close enemies, turning | 39.493 / 48.069 / 48.229 | 39.404 / 47.892 / 48.059 |

All eight means, p95s and maxima improve in this comparison. The level-1
burst worst frame falls by 7.458 ms; while turning, its p95 falls by
7.734 ms and worst time by 6.166 ms. The separate unmodified 128-frame
starting-room test retains the 25.80 FPS gate at 26.20 FPS, 38.17 ms mean,
38.68 ms p95 and 77.96 ms worst, versus 26.19 FPS, 38.19/38.70/77.97 ms
before. Burst frames still approach 91 ms; general threefold improvement
across these views and physical 486 performance remain unproven.

All 512 complete release-engine framebuffers and simulation hashes match.
The 32 KiB residency-stress oracle retains all 128 image/hash pairs between
4 and 16 MB on levels 1 and 6. Fresh compiler feedback completes 23 training
runs and all 22 ordinary/optimized native captures; all 22 original native
references and 22 freshly rebuilt indexed references match. The real-time
4 MB level-4 SB16/mouse exercise and full level-1/4/15 exercises pass with
zero profile audio underruns. Windows Release builds. Both packages retain
231 assets and 238 archive entries. Maps and all six saved level/map payloads
on levels 1 and 4 retain the same active-byte comparisons described above.

Native text/data/BSS are 1,187,352/22,016/292,864 bytes: 512 more code bytes
and unchanged persistent data size. The level-4 sound/mouse check reports
116 KiB free physical memory after shared assets and 32 KiB after loading
the level, versus 116/16 KiB previously. The reserved stack accommodates
the history without another heap buffer.

The staged native EXE SHA-256 is
`904953697e13ebda52a2f0622998644bb79e15bcd14090abbbe6d364652d4b79`;
the freshly rebuilt indexed EXE remains
`b19af1db3ac1570579d6c02d74bbb273df4c7858952582288d163154c28d023d`.
Staging replaces only the EXE and distribution archive, preserving all
241 native and 238 indexed non-EXE game files, including settings and saves.

Local evidence is in `build/dos/sprite9-performance.json`,
`build/dos/sprite9-validation.json`, `build/dos/sprite9-staging.json` and
`build/dos-sprite9-source/build/dos/pgo/model-check/report.json`.
Diagnostic and captured-kernel evidence remains under
`build/dos-sprite9-trace/`; the raster oracle log is
`build/dos-sprite9-source/build/dos-raster-regression/RASTER.LOG`.

### Reuse prepared native water colours

`dos_raster_prepare_water` now stores a ready-to-write RGB565 colour in
the low word of each existing 32-bit Brighten entry. Its high word retains
the original four-bit components in five-bit lanes. Guard bits allow the
three signed channel differences to be calculated together for the existing
rounded interpolation. Single-source and equal-sample pixels write the
prepared colour directly instead of reconstructing it through two tables.
`dos_raster_water_color` also returns that prepared word directly.

The original AB3DI.s Brighten colours, sampling and rounding still define
the output. Table sizes remain unchanged, with no extra heap allocation.
The actual 486 raster oracle checks all 5,632 prepared colours, all 65,536
native inputs, 65,536 alpha/colour combinations and 768 single/blended,
wrapping, aliasing and clipping spans. Its other raster/arithmetic checks
also pass. Across eight diagnostic views, all 512 complete framebuffers
and simulation hashes remain exact, with zero profile audio underruns.

The diagnostic pool view averages 63.728 ms versus 66.646 ms, a 4.4%
reduction. These diagnostics replace selected engine objects with ordinary
O3 builds. The rebuilt compiler-feedback release uses normal/486_slow,
40,000 fixed cycles, 4 MB, 320x240 RGB565, SB16/mouse and muted host sound.
Both benchmark entry points keep 512 bytes of timing samples and capture
frame 23 outside the timed update. Results cover fixed updates 8-63;
turning holds the original right-turn input throughout all 64 updates.

| View | Previous mean/p95/worst ms | Updated mean/p95/worst ms |
| --- | ---: | ---: |
| Starting room | 38.486 / 38.542 / 38.543 | 38.479 / 38.535 / 38.536 |
| Close enemies/barrel | 38.559 / 39.551 / 39.672 | 38.553 / 39.545 / 39.661 |
| Pool surface | 63.650 / 64.571 / 64.792 | 60.504 / 61.802 / 61.919 |
| Level-1 barrel burst | 50.672 / 76.640 / 79.324 | 50.667 / 76.639 / 79.318 |
| Level-6 barrel burst | 48.598 / 65.699 / 69.556 | 48.554 / 65.689 / 66.340 |
| Level-1 burst, turning | 52.357 / 82.579 / 90.544 | 52.353 / 82.574 / 90.851 |
| Level-6 burst, turning | 62.130 / 82.206 / 84.607 | 62.119 / 82.196 / 84.599 |
| Close enemies, turning | 39.404 / 47.892 / 48.059 | 39.398 / 47.888 / 48.052 |

The pool mean improves by 3.146 ms, or 4.9%, and its p95 by 2.769 ms.
Other means change little. The level-1 turning-burst maximum rises by
0.307 ms despite a slightly lower mean and p95. The unmodified 128-frame
starting-room test retains the 25.80 FPS gate at 26.21 FPS, 38.16 ms mean,
38.67 ms p95 and 77.96 ms worst, versus 26.20 FPS, 38.17/38.68/77.96 ms
before. Heavy bursts still approach 91 ms; physical 486 performance and
a general threefold improvement across these views remain unproven.

All 512 complete release-engine images and simulation hashes match, as do
all 128 image/hash pairs in the 32 KiB residency-stress test at 4 versus
16 MB on levels 1 and 6. Fresh compiler feedback completes 23 training
runs and all 22 ordinary/optimized native captures. All 22 original native
and 22 freshly rebuilt indexed reference images match. The real-time
4 MB level-4 SB16/mouse exercise and full level-1/4/15 exercises pass with
zero profile audio underruns. Windows Release builds. Both packages keep
231 original assets and 238 archive entries; maps and all six saved
level/map payloads on levels 1 and 4 retain the active-byte comparisons
described above.

Native text/data/BSS are 1,186,328/22,016/292,864 bytes: 1,024 fewer code
bytes and unchanged persistent data size. Shared-assets free physical
memory remains 116 KiB. The level-4 loaded checkpoint reports 16 KiB
versus 32 KiB before, so memory headroom remains a limitation despite
identical table sizes and the smaller executable.

The staged native EXE SHA-256 is
`0e566ed7d16f25d8826214f61787abb63a289671634b195b0edaeb52f6b3b9a5`;
the freshly rebuilt indexed EXE remains
`b19af1db3ac1570579d6c02d74bbb273df4c7858952582288d163154c28d023d`.
Staging replaces only the EXE and distribution archive, preserving all
241 native and 238 indexed non-EXE files, including settings and saves.

Local evidence is in `build/dos/water3-performance.json`,
`build/dos/water3-validation.json`, `build/dos/water3-staging.json` and
`build/dos-water3-source/build/dos/pgo/model-check/report.json`.
Diagnostic evidence is under `build/dos-water3-trace/`, with the raster
log at `build/dos-water3-source/build/dos-raster-regression/RASTER.LOG`.

A separate, unshipped compiler-training trial added two authored barrel
scenes to the previous release's unchanged engine. Its level-1 burst mean
improved by 0.162 ms and worst time by 0.577 ms, but the starting-room
mean regressed by 0.047 ms and level-6 turning p95/worst by 0.062/0.066 ms.
Those small, mixed changes do not justify adopting the extra training yet.
The isolated trial preserves all five checked frame-23 images and hashes;
its evidence remains under `build/dos-water3-trace/burst-training/`.

### Transport the water phase views directly

DOS `WaterFile` now carries an eight-byte `ABW1` version/length header and
a permutation of its complete 65,536-byte source payload. The first 32 KiB
contains the eight contiguous 64x64 phase planes selected by the original
AB3DI.s waterpt offsets. The second 32 KiB retains every odd source byte in
word order. `water_source_byte` preserves original byte addressing for scalar
sampling; the source maximum is unchanged by the permutation.

`renderer_build_water_level_phase_lut` uses a view into that existing
conventional-memory bank on DOS. It no longer builds or stores another
32 KiB phase array. Desktop keeps the original asset and phase builder.
DOS consumes and validates the header before allocating the same 64 KiB
resident bank. Both RGB565 and indexed packages prepare the new transport
automatically. Use matching updated EXE and DATA files together; raw water
files fail with an explicit ABW1 version/length error.

The source audit proves a bijection over all 65,536 bytes and all 32,768
phase samples. The actual 486 oracle loads the preparer's output, checks
every original byte through the scalar accessor and every direct phase
sample against the AB3DI.s address expression, and rejects invalid headers.
All other raster checks pass. Across eight diagnostic scenes, all 512 full
framebuffers and simulation hashes match, with zero profile audio underruns.
A real DOS startup test also rejects the old raw WaterFile.

Diagnostic BSS falls from 292,352 to 259,584 bytes, exactly 32 KiB. Its
pool mean is essentially unchanged at 63.712 versus 63.728 ms; this change
targets memory headroom rather than removing another rendering stage.
Fresh compiler-feedback release validation and package staging are pending.
The playable packages still contain the preceding water-colour release.
Local byte/phase, startup-rejection and diagnostic evidence is under
`build/dos-phase1-trace/`; the raster log is
`build/dos-phase1-source/build/dos-raster-regression/RASTER.LOG`.
