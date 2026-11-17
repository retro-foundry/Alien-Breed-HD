# Desktop frame timing

The desktop renderer draws the world in software at the configured internal
resolution. A 3840x2160 image contains four times as many pixels as 1920x1080.
Room coverage, water, polygon objects and viewing direction affect its cost.
`display.c` enables SDL presentation VSync, so missing a refresh deadline can
amplify the visible FPS change. At 120 Hz the frame budget is about 8.33 ms.

## Frame history copy, 2026-11-17

`renderer_prefill_color_columns` preserves the previous frame for the original
stream-ordered raster, including water refraction. Previously the main thread
copied the entire image before waking the renderer workers. At 4K that is a
16.6 MB color-word copy on the serial path, followed by workers writing the
same cache lines.

Each world worker now copies its own disjoint columns immediately before
rendering them. The source history remains read-only until all workers finish.
The synchronous path retains the full-image copy. ARGB history is copied only
when an ARGB raster consumer is active. Raster order, worker count, resolution,
gameplay timing and presentation settings are unchanged.

Repeated Windows x64 MSVC Release measurements at 3840x2160, 24 requested
workers, 240 timed frames per view, with alternating before/after runs:

| Level | Rooms | Mean render time before | After | Reduction | Mean per-view p95 before / after |
| --- | --- | ---: | ---: | ---: | ---: |
| 10 | 9, 49 | 2.683 ms | 1.833 ms | 32% | 3.567 / 2.305 ms |
| 11 | 20, 40 | 3.766 ms | 2.928 ms | 22% | 4.691 / 3.479 ms |
| 12 | 8, 37 | 2.594 ms | 1.732 ms | 33% | 3.552 / 2.146 ms |

Each row contains four headings per room and two runs per executable. These
are CPU render measurements from frozen authored scenes, excluding game logic,
audio, GPU composition and VSync. They demonstrate removal of the copy
bottleneck; they do not establish that every source of FPS variation is fixed.

Validation also compared 1,040 lower-room views across levels 10-12 at 960x540,
12 workers and four headings: their color-word buffers matched the prior
renderer. The native checks passed 576 CPU weapon/water cases and 576 actual
SDL composition cases, covering threaded/synchronous rendering, ARGB/CW, both
players and HUD preservation. The WebAssembly checks passed 288 weapon/water
cases and the color/diagnostic tests. `feedback_tags.c` additionally checks
partial-column ownership, untouched history sources, clamped/empty ranges,
ARGB ownership and the full-image synchronous copy.

## Reproduce CPU scene measurements

Build the normal Release game first, then configure the existing native render
test project. From the repository root in PowerShell:

```powershell
$renderData = (Resolve-Path build/Release/data).Path
$renderSdl = (Resolve-Path build/_deps/sdl2-src).Path
cmake -S tests/render -B build/render-checks -A x64 "-DAB3D_TEST_DATA=$renderData" "-DFETCHCONTENT_SOURCE_DIR_SDL2=$renderSdl"
cmake --build build/render-checks --config Release --target desktop_scenes --parallel
build/render-checks/Release/desktop_scenes.exe 12 3840 2160 240 24 8
```

Arguments are level (1-16), width, height, timed frames per heading, requested
worker cap, and optional room slot. A worker cap of zero uses automatic
selection. Omitting the room sweeps lower rooms whose edge-average center lies
inside the room and has player headroom; this is not a playable route through
the level. The tool uses the original assets and does not write saves or INIs.
It reports average, median, p95 and maximum render duration, without display
pacing. Run performance comparisons without `AB3D_PROFILE_RENDER` enabled and
without simultaneous builds or other benchmark processes.

Build `renderer_tags`, `weapon_water_cpu` and `weapon_water_display` in the same
test directory to run the native regressions. Run their executables from the
repository root so their authored asset fixtures resolve. The Windows CI job
runs these checks as well.
