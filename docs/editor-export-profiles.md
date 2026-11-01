# Original-engine export evidence

R12 requires a distinct original Amiga engine profile. The current browser
preview uses the C port and cannot prove original-engine compatibility.
This note records source contracts for that profile; the profile is not yet
implemented or offered as a compatible export option.

## Map point indices

`amiga/AB3DI.s`, `RotateLevelPts` / `pointrotlop`, reads each point ID from
`PointsToRotatePtr` and writes a long to `Rotated + ID * 8`. The adjacent
rotated depth occupies the other long of that entry. `Rotated` is declared
`ds.l 2*800`, so map point IDs must be below 800. This is an index constraint,
not a limit on how many points happen to be visible in a particular view.

The file header at data offset 14 stores the last point index. The current
editor parser computes `i16(14) + 1` and allows 2,048 map points. That parser
limit must not be presented as proof of original Amiga compatibility. A strict
profile must validate the generated artifact and reject out-of-range native
point references before export, while keeping the editable project saveable.

## Object and ordering buffers

`amiga/AB3DI.s` declares each player's object-distance table as `ds.w 250`.
The C port's `src/game_state.h` declares `MAX_OBJECTS 250`; the editor already
uses the generated object limit, including reserved source slots. Original
runtime shot-pool expansion and collision-point indexing still need a separate
profile audit; the source record count alone does not prove their safety.

`amiga/OrderZones.s` declares `ToDrawTab`, `OrderTab`, and `FinalOrder` as
`ds.l 400`. These are not all 400-room tables: `settodraw` uses a byte indexed
by room ID, whereas `putinn` advances an eight-byte ordering record and writes
sentinels. Derive each table's usable range from its writes and traversal
before publishing a zone or visible-room threshold. Do not translate a buffer
declaration directly into an unverified user-facing limit.

## Remaining profile proof

Audit graphics command storage, clipping/scratch buffers, moving mechanisms,
fixed-point arithmetic and packaged resources against their assembly consumers.
Check all sixteen original levels and adversarial boundaries against those
contracts. Export controls must identify the selected engine, give actionable
diagnostics, and leave source and recovery intact on rejection. The existing
current-port preview/export guards remain useful but do not complete R12.
