# Port level exit configuration

The React editor saves the target game slot and exit choice in its `.ab3d`
project. Export level writes the three original game files and `twolev.cfg`
under the selected `levels/level_a/` through `levels/level_p/` directory.
Extract this ZIP into the port's data directory. Opening the ZIP in the editor
restores both its target slot and exit choice. When opening individual level
files, include `twolev.cfg` with the triplet to retain custom exit setup.

`twolev.cfg` is a port extension. It does not alter the original Amiga geometry
layout, and original Amiga engines continue to use their built-in exit zones.
Custom exits require a game port build containing this loader.

The file is exactly eight bytes:

| Offset | Size | Value |
| --- | --- | --- |
| 0 | 4 | ASCII `ABEX` |
| 4 | 2 | Big-endian unsigned version, currently 1 |
| 6 | 2 | Big-endian signed exit-zone choice |

The exit choice is -2 for the original exit of the selected game slot, -1 to
disable level completion, or an existing runtime zone ID from 0 to 255.
Unsupported versions, malformed sizes and nonexistent custom zones are errors.
A missing file preserves original-game behavior. Exports always write an
explicit choice, including -2, so extracting a replacement map cannot retain
an old custom exit configuration.

`io_load_level_exit_zone` loads the setting after parsing the map. Each load
starts from `game_data.c`'s original `end_zones` value for that slot. The game
loop uses the state's effective exit and otherwise retains the original
completion condition: player one entering that zone completes the level.

The editor project format is now version 3. Version 2 projects retain their
saved slot and use the original exit. Version 1 projects did not save either
setting and open at slot 1 with its original exit. Original game directories
and exported ZIP directory names identify their slot independently of a
project's display name.

Source-room remapping after topology edits, including exits spanning multiple
convex runtime cells, remains part of the full geometry compiler work.
