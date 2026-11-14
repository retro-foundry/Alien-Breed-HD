# Billboard spill review

Billboard spill is a port enhancement. The original
`ObjDraw3.ChipRam.s:ObjDraw/itsinthiszone` selects objects by their owning room;
`BitMapObj` supplies their camera-facing dimensions. Gameplay ownership and
collision remain unchanged by the enhancement.

The review found these rendering problems:

- Candidate selection considered only immediately connected rooms. The unused
  helper named `renderer_collect_spill_zones_recursive` also visited one hop.
- A fixed backward painter-order tolerance rejected geometrically valid pieces.
- Clipping included the secondary neighboring collision lines after the first
  negative boundary-list entry. `ObjectMove.s:checkwalls/nomorewalls` and the
  editor's `boundaryIds` distinguish these from the primary room outline.
- Integer-rounded midpoint classification could assign an outside interval to
  a boundary. Fractional intersections now retain fractional test positions.
- Owned-room draws omitted the polygon mask applied to spill draws.
- The mask used the current object point while rotation used interpolation.
  Interpolation can also put the displayed center in a previous room after
  gameplay ownership has crossed a portal.
- Height heuristics discarded valid fragments below half the billboard height.
- A real room band whose projected top and bottom coincided was treated as an
  unset clip. One-row bands now retain their clip, and inverted bands draw no
  rows. A dedicated regression reproduced 256 writes where 32 were expected.

`src/renderer.c` now follows actual portal intersections along the displayed
billboard segment, including reverse links and multiple hops, without a painter
order cutoff. Interpolated display ownership is resolved along the actual motion
segment, before walking the billboard plane. Primary outlines determine the
column masks for both owned and spill pieces; source/destination room planes and
the existing spill wall-depth clipping retain vertical occlusion.

The stable-slot candidate index moved from `src/platform/dos/objects_dos.*` to
`src/renderer_objects.*` and is prepared once before rendering workers start.
Objects, both shot pools and explosions retain separate membership bits and
their existing painter insertion order. The existing billboard enhancement
setting controls spill; disabling it uses owned-room draws. View percentages,
source assets and gameplay state layouts are unchanged by this review.

Run the focused portable regressions with:

```text
python tools/check_feedback.py --emcc <emcc.py> --only spill water door_render settings
```

`tests/feedback_spill.c` checks seven-cell crossings, closed/disconnected and
reverse-only portals, rotated planes, arbitrary painter order, fractional
ownership, secondary collision lines, interpolated portal crossings, separate
object/shot/explosion bits, frame resets and two-row, one-row and empty room
bands. It checks 152,039 geometry/position/raster checkpoints, all 13,904 starting billboard
orientations and 1,767,316 independent pixel samples across the sixteen shipped
levels. Another 104,128 real portal-corner/midpoint orientations include 8,348
nonempty pieces beyond the first adjacent room.

Desktop Release and the focused portable tests pass. The same targeted geometry
and raster fixtures also pass in a 4 MB DOSBox 486 run at 200,000 cycles; this
is a correctness check, not a performance measurement. Existing native DOS
raster, movement and stable-slot selection regressions pass after the index
move. Local reproduction scripts and logs are under `build/spill-review`.

The final native release uses fresh feedback from 31 scenarios. Its 22 starting
and turning captures match both the ordinary build and the previous release.
The real-time 40,000-cycle, normal/486_slow, 4 MB SB16/mouse exercise passes
all 16 full quickloads and save slots with zero audio underruns. All four view
sizes also pass 96 weapon-frame comparisons and 32 framebuffer switches. The
final indexed build passes its 4 MB view-size and complete UI/save/load checks
at 200,000 cycles with fixed ticks, used for correctness only.

The 128-frame real-time starting-scene profile at 320x240, 40,000 cycles, 4 MB,
SB16 and mouse measures:

| View size | FPS | Mean frame | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| 100% | 26.62 | 37.56 ms | 37.56 ms | 56.14 ms |
| 80% | 27.68 | 36.12 ms | 35.46 ms | 111.82 ms |
| 60% | 33.15 | 30.17 ms | 29.47 ms | 113.51 ms |
| 40% | 40.58 | 24.65 ms | 23.95 ms | 107.01 ms |

A matched rerun of the previous executable at 100% gives 27.80 FPS / 35.98 ms,
so the corrected visibility costs 1.58 ms per frame in this scene. These are
starting-scene measurements, not all-level performance claims. The matched
baseline log is `build/spill-review/performance-before-matched.log`; final raw
profiles and pacing are recorded in `build/spill-review/performance.json`.

Text/data/BSS are 1,225,752 / 22,016 / 278,528 bytes: 3 KiB more text and 6 KiB
more BSS than the previous release. The existing heap reservation is retained.
The shared-assets snapshot reports 88 KiB free; level 4 reports 36 KiB free.

The verified native executable is staged at `build/dos/game/ABHD.EXE` and
`build/dos/ABHD.exe`; `build/dos/ABHD-DOS.zip` is 3,436,939 bytes.
Executable SHA-256:
`53e28f16511c4b3128f3a3f8ab4c7a5f9e0d66f9dd3107e89e3fe31c80f5b45a`.
Archive SHA-256:
`d6a74bb4e1659b163d583076864dfc10837fb5e4a42017a60291c63c4f4683e1`.
Only the executable's archive entry changes. Staging preserves 241 other native
files, including settings and saves, and all 239 indexed files. The previous
native artifacts are under `build/spill-review/staging-backup`.

The unfiltered portable suite currently stops in the pre-existing
`feedback_wall_touch.c` inclusion of `dos_movement.c`, which references
DOS-only `movement_integer_wall_clear` and `movement_clear_room_path` without
the DOS build definitions. The focused rendering checks above run independently.
