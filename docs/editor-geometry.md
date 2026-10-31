# Editor geometry evidence

The port accepts either drawing direction. Geometry normalisation owns the
runtime's orientation; users never configure winding. All 2,496 collision
outlines, 2,567 floor polygons, 2,318 roof polygons and 231 water polygons in
the original 16 maps have negative signed area in world X/Z coordinates.

## Original assembly oracle

`amiga/archive/amos/leved.amos` includes an attached `AmBk` bank numbered 9,
named `Asm`. Its 1,376-byte payload is the external `BETPTS` helper called in
the AMOS listing as `Call Start(9)+54`. The payload SHA-256 is
`b73b5c1984829910cfc3fbc888c56a24e9fb26857604289ac16860d0840458f9`.
Instructions include scaled indexed addressing; the execution oracle uses a
68020 CPU. This is evidence about the editor helper, not a new game CPU claim.

`tools/editor_geometry_oracles.py` checks both original-file and bank hashes,
loads the bank in Unicorn 2.1.4, supplies the original argument memory layout,
and executes the entry point with an instruction bound and explicit return
address. The browser ships neither this emulator nor the AMOS program.

| Routine | Bank address | Meaning established | Port evidence |
| --- | --- | --- | --- |
| Dispatch | `+0x36` | Mode at `+40` selects the original operation | Emulated entry and return |
| `CALCORDER` | `+0x2a6` | Tests source vertices against each target edge; wall type 1 is skipped; three bits per edge start at bit 1 | 120 independent emulated fixtures match TypeScript exactly |
| `BETPTS` | `+0x346` | Four signed side tests classify active wall vertices relative to a portal envelope; modes 1/2 handle a collapsed end | 90 independent fixtures, including all three modes, inactive vertices and signed-word wrap, match exactly |
| Left/right clip elimination | `+0x11a` / `+0x56` | Caller parameters and pairwise side tests are identified | Not yet promoted to a complete port contract |
| Separating-line check | `+0x1f4` | Tests all left/right clip pairs against both room outlines | 120 independent emulated fixtures match TypeScript exactly |

`editor/src/model/visibilityGeometry.ts` preserves the actual SUB.W/MULS.W/
SUB.L arithmetic, including signed 16-bit coordinate differences and signed
32-bit results. The fixtures deliberately include larger differences that
would distinguish this from an unbounded JavaScript cross product.

## Current-format ordering proof

The original helper produces a cached mask, rather than the final runtime word.
`LEVELSAVE` lines 1188–1228 adds the exit gates and handles the source room's
own entry. Source map Y is negative world Z. Applying this source-backed
conversion independently rebuilds **all 49,797 ordering masks** in all sixteen
shipped maps exactly (2,496 self entries and 47,301 other entries). This test
derives masks from outlines, portal connections and existing visible membership;
it does not copy their stored mask words.

## Visibility repair

A bounded translation of the older `INITDEFORDER` wall-chain traversal was
audited against the shipped levels. Independent straight sightlines disproved
some of its rejections, including Level 1 zone 71 to zone 91. The unverified
traversal was removed from application source and retained only as ignored
local analysis evidence. The source helpers above now have 330 emulated cases.

`portalVisibility.ts` implements the requested visibility repair using the
actual exit/clip predicates in `src/visibility.c`. An oriented line is
`a*x + b*z + c = 0`, along direction `(b,-a)`. Passing a directed portal requires
positive line value at its start and negative value at its end. A sequence of
portals therefore gives linear halfplane constraints on the line. Four charts
cover its directions; rational vertices and BigInt predicates make feasibility
and positive-area checks exact. There is no sampling threshold in this solver.

Each cell propagates feasible line regions through its portals. Regions already
contained in a previously reached region are discarded. Every visible pair
gets an interior sightline witness. Legacy left/right clip points are emitted
only when their halfplanes contain **every** feasible route, preserving
alternate openings. Solid-wall previous/next links follow the original format.

`visibilityImport.ts` splits the few concave original boundaries into internal
convex cells, retaining original zone identities and game bytes. It preserves
portal endpoints that lie on straight walls. Authoring geometry continues to
accept both directions and automatically creates convex runtime pieces.

All 16 original maps now pass model compilation and exhaustive original wall
link comparisons. Clip offsets remain below 32,768 words in every map (largest
23,034). New membership totals 49,150 entries, compared with 49,797 stored:
661 stored entries are removed and 14 are added. Independent grid/ray checks
found no clear sightline in the removed pairs; this sampling is additional
evidence, separate from the exact feasibility predicates. Clear ray witnesses
support ten added entries in Level 13 and two in Level 14. The remaining two
additions in Level 15 still need inspection. These differences need generated
file and actual engine scenarios before replacing existing maps' visibility.

Tests also cover a straight corridor, disconnected rooms, two bends, the
specific Level 1 regression, and both openings around an occluder. Every
generated clip halfplane admits its room pair's concrete witness.

## Binary and engine checkpoint

`visibilityBuild.ts` now writes the derived data back into game files. The
layout audit accounts for all 2,496 original zone blocks: each boundary list
precedes its header, followed by visibility and rotation lists. It preserves
the non-zone prefix and every graphics stream, while replacing the complete
derived zone region. A second rebuild produces identical files. Geometry,
materials, objects, mechanisms, teleports, starts and heights pass retention
checks across all 16 maps.

Door/lift header `+8` pointers address **LEVELGRAPHICS**, as shown in
`amiga/Anims.s`. They must stay unchanged during this level-data rebuild.
The old assumption that they pointed to data-file height fields was rejected
by both the original assembly and the shipped bytes. ZoneAdds entries in the
graphics file are updated to the new data offsets.

Both Chromium and Firefox load all sixteen rebuilt maps in the actual game
engine. At each original player-one start, a reset renderer capture has the
same full-frame checksum as the original map. This checks a real rebuild,
rather than byte passthrough. The preview capture uses the source zone/spawn
directly: collision's extra corner-wall lists must not reject an inspection
camera at a valid original start (Level 3 exposed this distinction).

Broader camera sweeps, doors in motion, closed-door overdraw, upper/lower rooms,
water and created/edited geometry remain required. Visibility is not yet
automatically replaced by the application during normal save/export. Full
geometry/source compilation and its integration must preserve the document's
source history independently of these generated files.
