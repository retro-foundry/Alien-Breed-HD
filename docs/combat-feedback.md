# Combat feedback changes

The six requested changes are implemented in the PC and web paths:

1. `marine_hitscan_projectiles=0` in both the template and engine defaults.
   `objects.c` already separates the normal rifle's `marine_hitscan_burst`
   from the other ranks' attacks. Existing INI files with `=1` retain that preference;
   change it to `=0` to select hitscan.
2. The original `AB3DI.s` `hitcol` pulse now drives a translucent red overlay
   across the borderless view. Both players' timers fade independently in
   VBlank units, so batched ticks do not prolong the pulse. HUD draws afterward.
3. Missing keys use alpha 40 instead of 128; collected keys remain fully
   opaque and gain a bright underline, giving an additional ownership cue.
4. Vertical side bars replace the numeric health/ammo block. `EnergyBar` and
   `AmmoBar` in `amiga/AB3DI.s` provide the left/right placement, row-24 offset,
   top-down depletion, 127-health/63-ammo ranges and two rows per ammo unit.
   The original `healthstrip` and `ammostrip` four-plane silhouettes are decoded
   directly, using `ScreenSetup.s`'s `HealthPal` green/yellow/red ramp.
   Following the user's Amiga screenshot, the compact borderless bars occupy
   half the viewport height, use narrow strips at 38% opacity, and have a very
   faint empty track. They centre vertically and anchor to the sides for widescreen.
   The old framebuffer bar writes (before the world render) are removed.
   Options > HUD switches between BARS (default, `hud_bars=1`) and the previous
   NUMBERS display (`hud_bars=0`). The choice persists in the desktop INI or
   browser local storage, and loading a game keeps the current HUD preference.
   Small viewports use filtered bar textures (mipmaps on the GL path), a
   four-pixel minimum width, and upward-rounded fill heights. Non-zero health
   or ammo remains visible, and full values fill the track at every size.
5. `AB3DI.s` `oknothalf` / `nowaterfull` switch the CIA audio filter whenever
   `fillscrnwater` is nonzero. The port follows that same water state, independently
   of `post_tint`, using a 2.2 kHz two-pole Butterworth filter on the mixed output.
   A 20 ms wet/dry transition avoids abrupt switching. This is an audible
   software approximation, not an analog circuit simulation. Leaving gameplay
   clears the water target so menu/end music is not left muffled.
6. Keyboard and mouse/gamepad control share a movement acceleration helper
   that applies the selected 50-200% multiplier only while running, after the
   original `Plr1Control.s` integer speed/crouch calculation. Walking, friction,
   turning and the run/walk modifier retain their existing behavior. Options >
   Run Speed adjusts in 10-point steps; 100% is original speed and the default
   130% retains the requested PC boost. `run_speed` persists in the desktop INI
   or browser local storage, and loading a game keeps the current preference.

No replacement gameplay rules or placeholder effects were introduced.

Enemy attacks, idle vocals, pain/death cries and wall fire-spouts now share
`SoundPlayer.s` `MakeSomeNoise` distance attenuation instead of fixed mixer
gains. Original enemy handlers and `AlienControl.s` `FireAtPlayer1/2` supply
the sound strengths. Simultaneous sources of the same sample retain the
loudest contribution without restarting the burst. The active local player
determines the listener, including the slave player in multiplayer.

The initial level 8 walker check covered straight movement through the
authored `level_h` exits. The subsequent report exposed differences in
wandering, targeting, turn thresholds and attack recovery; see the level 8
robot section below. Its original collision dimensions remain in use.

A follow-up to the level 15 worm report found that targeting used the previous
tick's position/aim. `AB3DI.s` refreshes `CalcPLR1InLine` / `CalcPLR2InLine`
after player control, before `Anims.s` invokes `Player1Shot` / `Player2Shot`.
The port had refreshed only after object movement. The public shooting entry
points now refresh immediately before selecting a target. Turning/strafing onto
a worm can hit on that tick; turning/strafing away no longer damages a target
from the stale list. This also corrects controller projectile auto-aim when the
old cached distance was zero. Level 15 (`level_o/twolev.bin`) contains 13 worms
with authored health 80. Their original `HalfWorm.s` quarter-damage rule and
the previously requested run-speed boost are preserved.

Validation: Windows Release and single-threaded Emscripten Release builds;
focused movement and audio checks using the production C routines; browser
level smoke checks through both the OpenGL and SDL display paths, with no
browser errors. The visual harness exercises full/partial bars, collected and
missing keys, and the red hit overlay. Asset paths use the shipped `HealthStrip`
and `AmmoStrip` capitalization for case-sensitive web/Linux filesystems. Run:

```sh
python tools/check_feedback.py
# If needed, select Emscripten's Python entry point explicitly:
python tools/check_feedback.py --emcc /path/to/emscripten/emcc.py
```

The movement checks cover 1,920 combinations of control mode, always-run,
modifier, crouch, direction and tick batching. Audio checks cover desktop
8007 Hz and web 48000 Hz, dry bypass, low/high frequency response, silence,
and entering/leaving water.
HUD checks cover bottom alignment, monotonic fill, empty/full values and a
visible last health/ammo unit at small sizes. Browser pixel comparisons at
160x120, 320x200, 640x480 and 1280x720 pass through both GL and SDL paths.
Shooting checks cover 40 cases: both players, mouse/fixed-view pistol and
shotgun, and controller plasma auto-aim, each turning/strafing into and out
of a worm target. The turn-onto-target case failed before the cache fix
(zero damage, ammo consumed) and passes afterward.

The shotgun soldier report exposed an unreachable shotgun branch in
`object_handle_marine`: `OBJ_NBR_FLAME_MARINE` always selected a flame projectile.
Despite its legacy name, `amiga/FlameMarine.s` `FlameMarineAttackPLR1` fires
five instant-hit pellets, each dealing two damage, with sample 21 at Noisevol
200 and a `200+(rand&255)` recovery timer after the attack timer drops below 20.
That attack now replaces the flamer branch, and the enemy data describes the
shotgun. The source's conflicting `FlameMarineAttackPLR2` flamer code is
deliberately not reproduced: both targets receive the single-player shotgun
attack. The rifle projectile preference does not affect this enemy.

`tests/feedback_enemy_audio.c` adds 16 shotgun attack cases covering both
targets, both rifle-option values, near/far damage, single/batched ticks,
windup, sound, recovery, and an unused projectile pool. These use synthetic
states and the production handler; an original-game emulator comparison
has not been performed. For an oracle check, capture `FlameMarineAttackPLR1`
before/after firing at close range: `FourthTimer`, `ThirdTimer`, sample/volume,
player `damagetaken`, and `NastyShotData`; expect sample 21, five hit rolls,
at most ten queued damage, and no projectile allocation.

The next player report led to three changes:

- Distance fading retains `SoundPlayer.s` `MakeSomeNoise`'s inverse-distance
  curve, source loudness and cap, with fractional distance/gain retained until
  conversion to the mixer's 0..255 volume. This deliberately smooths the
  original Paula rounding and approximate square root. At distances 1600..3200,
  a Noisevol-200 source now has 64 volume changes instead of 16. Sample
  coalescing and the loudest simultaneous emitter still work as before.
- The underwater filter cutoff is reduced from 3 kHz to 2.2 kHz so muffling
  affects more of the original samples' audible band. The 20 ms transition
  and exact dry bypass remain. This is requested audio tuning, not a claim
  of analog circuit equivalence.
- The browser ending now returns to the title menu after the end music/fade.
  `EM_OUTER_BR_ENDGAME` previously returned zero, making `em_frame` tear down
  the game and cancel its callback. Keeping that callback alive allows another
  New Game without refreshing the page. Desktop exit behavior is unchanged.

The teleporter investigation also found a port bug. In `AB3DI.s`
`PLR1_Control` / `PLR2_Control`, an occupied destination restores the attempted
walking X/Z before ordinary collision runs. `check_teleport` left its attempted
destination X/Z in the context on failure. A second collision test could then
hit the remote blocker again and reject walking, trapping the player on the
pad. Failed teleports now restore walking X/Z. Occupied exits still refuse
teleportation; walking off and retrying after the blocker clears both work.
This reproduces a plausible cause of the report, without claiming to recover
the player's exact incident.

`tests/feedback_teleport.c` loads the original level 15 data and checks all
four authored teleporters (including the extra zone-table slot): 96 cases
cover both players, standing/crouching, stationary/moving attempts, occupied
destinations and retry. The blocked moving cases fail before the fix.
`tests/feedback_ending.c` is a browser harness for the production ending and
menu lifecycle, with only the music clock advanced for the check. Two repeated
ending -> title -> New Game cycles pass on each of the GL and SDL display paths.
Build that harness with `python tools/build_ending_check.py` after building
the single-threaded web target (the same `--emcc` option is supported).
Windows and single-threaded Emscripten Release builds and the complete
`tools/check_feedback.py` suite also pass.

Shotgun recovery remains `200+(rand&255)` VBlanks with the original windup;
the slight aggression difference has not been demonstrated as a timer bug.
Flamethrower damage/burning balance is unchanged. An original-game emulator
capture of blocked teleport attempts remains an oracle follow-up: capture
`PLR1_Control`'s teleport collision return, `newx/newz`, `hitwall`, player
position and `PLR1_Roompt`, then compare a walking attempt with an occupied
exit against the native fixture.

By request, the starting pulse rifle (gun 0, `PISTOL` in the original data)
and the optional PC flamethrower (gun 3) now share gun 0's inventory counter.
`player_weapon_ammo_ptr` owns the mapping for firing, recharge and the HUD.
This is a deliberate change from the original separate gun slots, not an
assembly-parity correction. Each shot consumes eight internal ammo units
(one displayed round), preserving one flamethrower charge per shot.
The existing three-second recharge delay, one charge per five VBlanks and
20-charge refill ceiling remain. Firing either shared weapon holds off the
refill; it also runs while another weapon is selected. The refill ceiling
does not trim ammo gained from the original `Anims.s` `ItsAnAmmoClip` pickup.
Flamethrower enable/disable and saved-game config application leave the
shared stock intact. The existing gun 0 counter in passwords and full saves
carries the pool without changing save layouts. Other weapons retain their
own ammunition.

`tests/feedback_shared_ammo.c` checks both players' shared consumption and
empty clicks, actual rifle ammo pickups, unrelated plasma ammo, delayed and
batched refill, stopping refill while firing, preserving stock above the
refill ceiling, config toggles and existing infinite-ammo behavior.
The browser runtime check passes on GL and SDL: both weapons consume the
same displayed supply, refill resumes after emptying it, and an actual
autosave/load restores 37 shared rounds without trimming them to 20.
Windows and single-threaded Emscripten Release builds and the full feedback
regression suite pass.

## Windows crash after death on level 12

Report received 2026-10-30: the downloaded Windows executable intermittently
crashes after the player is killed on level 12, while returning to the main
menu. The Windows platform was confirmed after an initial browser-build
investigation. Game version, PC specifications, crash message and a save/replay
were not supplied. Status: unresolved; native Windows reproduction is pending.

The transition follows `amiga/AB3DI.s` `quitGame`, `playGameOver`,
`weveLost` and `CleanupForMainMenu`. The shared implementation spans
`game_loop_tick`, `play_the_game_after_game_loop`,
`play_the_game_finalize_session` and `play_game_outer_should_continue`.
The browser uses the separate `EM_OUTER_POST`/`EM_FRONT_MENU` phases in
`src/emscripten_loop.c` for the outer transition.

`tests/feedback_ending.c` now supports a configured starting level and exports
`feedback_die`, `feedback_current_level` and `feedback_save_autosave`.
With `start_level=12` in the preloaded INI, load `ending-check.html`, wait for
`Module._feedback_started_new_game()` to return 1, and call
`Module._feedback_save_autosave()` followed by `Module._feedback_die()`.
Death runs through the regular browser callback, including the final game
frame, level cleanup, full game-over music/red fade and title-menu setup.
Wait for `Module._feedback_at_title()` to return 1, select New Game, dismiss
the level text and verify that level 12 starts again. Repeat to exercise
cleanup and autosave rotation.

Build the harness with `python tools/build_ending_check.py`. Its `--asan`
option supports a separate single-threaded web build compiled and linked
with `-fsanitize=address`; use `--build-dir` to select that build. Assertions
and stack-overflow checks are also enabled when linking that harness.

All six full-fade death/restart cycles passed in local Chromium: three each
through GL and SDL presentation, with autosaves created before death and
AddressSanitizer/assertion/stack checks enabled. The existing final-ending
regression also passed twice per presentation path. These are short
stationary browser scenarios with controlled death; they do not reproduce
or rule out the reported Windows crash. No production behavior was changed
without a reproduced failure or a confirmed cause. The next native check
needs the reported Windows version and its `ab3d.log`/crash details, or a
save and combat sequence that reproduces the failure.

## Crawling up the level 7 flame-spout stairs

The level 7 (`level_g`) flame-spout stairways have authored 16*256 risers.
The separate crouched 10*256 step allowance blocked these while standing
could climb them. Player movement now shares the existing 40*256 step-up
limit between both stances, as requested, while using the current body
height for roof clearance. This deliberately adjusts the archived
`AB3DI.s` `PLR1_Control`/`PLR2_Control` crouched override to provide the
requested crawling behavior. The standing
limit, step-down allowance and `ObjectMove.s` clearance rules are unchanged.

`tests/feedback_stairs.c` loads the original `level_g/twolev.bin` and graphics.
The zone 123->124 crouched ascent failed before the fix. All 96 authored
crossings on both flame-spout stairways now pass for both players, both
stances and three bob phases. Another 18 cases check the step-height limit
and roof clearance, including openings that only admit a crouched player.
The 28 hazard cases fire the seven actual stairway emitters through the
production gas-pipe and bullet handlers: standing players are hit and
crouched players pass beneath the flames. `Anims.s` `ItsAGasPipe`'s authored
height minus 80 and `ItsABullet`'s collision-height checks are retained.

The full `tools/check_feedback.py` suite, Windows Release build and
single-threaded Emscripten Release build pass.

## Door texture alignment across all levels

Updating a door used its legacy texture-byte base directly as the newer
wall record's tile/row words. This discarded the horizontal tile offset
on 25 door faces. Scrolling relative to the closed position also discarded
the authored vertical offset on doors whose closed height was not zero.
For example, level 6's zone 120/123 faces changed from tile 2, row 32 to
tile 0, row 0; level 12's zone 72 faces lost their row 80 offset.

The shared door routine now decodes the legacy base into the separate
tile/row fields read by `amiga/WallRoutine3.ChipMem.s` `itsawalldraw`, then
applies `amiga/Anims.s` `DoorRoutine`'s absolute `-(curr>>2)&255` scroll.
The original door lists use 4096 bytes per tile, including faces with a
different wall `VALSHIFT`. Existing wall scaling, door motion, authored
level data and full-save layouts remain unchanged.

`tests/feedback_doors.c` loads all 16 original maps and compares every door
face against its authored initial texture origin throughout opening and
closing. The old code failed on level 1's zone 22 before any movement.
All 123 doors / 245 faces now pass 8,085 texture samples, including 25
nonzero horizontal tile offsets and 151 faces at nonzero height. The test
also checks wall heights, untouched graphics fields and unchanged legacy
door-list bytes. The full feedback suite, Windows Release build and
single-threaded Emscripten Release build pass.

## Level 8 robot movement, attacks and death

The follow-up report is addressed in the shared robot handler, using
`amiga/Robot.s` `ItsARobot`, `.waitandsee`, `.nonewdir`,
`ROBAttack_Player`, `MUSTPLR1/2` and `amiga/ObjectMove.s` `HeadTowardsAng`:

- Wandering now retains `CurrCPt` and turns toward it gradually, including
  stopping forward movement when facing away. The shared wander helper's
  immediate random-facing replacement did not reproduce this behavior.
- The attack turn comparison now scales the sine-table threshold to the
  long cross product. Previously even a player almost directly ahead
  caused a full turn, repeatedly overshooting the aim direction.
- Pursuit reads resolved `PLR*_xoff/zoff` instead of the `p_*` inputs from
  before player collision. A stale target could send the robot toward a
  different wall or away from the player it was shooting at.
- `ObjTimer` expiry can resume pursuit during `ThirdTimer` recovery, as in
  `.waitandsee`. Wall contact retains the original `ObjTimer=-1` signal.
  The original shot bursts, random recovery values, turning increments,
  speed, 300-unit attack range, 160-unit extent and step limits are retained.
- Damage is applied after movement, and a pain hit suppresses firing on
  that tick. The death blast uses the original 120 while prowling or 400
  while attacking, replacing the port's fixed 40. The blue key still drops.
  `Robot.s` calls `ComputeBlast` on death; it does not fire a grenade volley.

`tests/feedback_robot.c` reproduces the old aim oscillation and stale-position
targeting failures. It checks both targets, gradual prowl turns, recovery
interruption, single/batched shot timing, actual projectiles, pain suppression,
death blast damage and the blue key. The original `level_h` data supplies
eight routes from the authored spawn to adjoining rooms and four centered
crossings through the outer enclosure's 192-unit exits, with the original
collision size. Diagonal contact with a door jamb may still restart wandering;
these checks do not assert that every approach angle must pass unobstructed.

The complete `tools/check_feedback.py` suite, Windows Release build and
single-threaded Emscripten Release build pass with these changes.

An original-game emulator replay remains pending. For that comparison,
capture `ItsARobot` entry/exit around a narrow exit during recovery: the
64-byte object, its `ObjectPoints` slot, `TempFrames`, `CurrCPt`, `Facing`,
`ObjTimer`, `ThirdTimer`, `FourthTimer`, `SinRet/CosRet`, `canshootgun`,
`objroom` and `oldx/oldz/newx/newz`, plus projectile-pool changes when firing.
Replay those state windows in the native fixture; also capture a lethal
hit in each prowl/attack branch to compare `ComputeBlast` input and key output.

## Level 9 teleporter arrival collision mask

A follow-up report described intermittent failed teleports on level 9,
possibly after enemies used a pad. The authored `level_i` data has four
teleporters (35->36, 42->85, 90->109 and 134->82) and four tough marines.
The reported incident has no replay, but a matching port-specific refusal
is reproducible on the actual pad records.

`amiga/AB3DI.s` `PLR1_Control` / `PLR2_Control` uses the 18-bit
`%111111111111111111` (`0x3FFFF`) arrival collision mask. The separate
`amiga/ObjectMove.s` `CheckTeleport` routine used by enemies uses 19 bits
(`0x7FFFF`). The shared C helper hard-coded the latter mask for players
as well. Bit 18 adds tough marines as arrival blockers, although the
original player routine ignores that type. Thus an enemy reaching the
arrival point could prevent a player teleport that the original permits.

`check_teleport` now takes the caller's arrival collision mask explicitly:
players pass `0x3FFFF`; enemies retain `0x7FFFF`. It restores the ordinary
walking collision mask afterward. This restores the original player rule
throughout the game without modifying any map or teleporter records.
The earlier failed-attempt rollback remains: a blocked attempt preserves
walking X/Z and height, and a later tick retries normally. Ordinary aliens
still block occupied arrivals until they move away or die; enemy activity
does not permanently disable a pad. No alternate landing spot or telefrag
behavior is added.

`tests/feedback_teleport.c` now checks all four pads on levels 9 and 15:
480 cases across both players, standing/crouching, stationary/moving
attempts, clear arrivals, live/dead aliens, tough marines and shotgun
marines. Repeated blocked attempts recover after a living alien moves
away or dies. Enemy probes retain the wider mask, restore their caller's
mask, use the same pad and leave its six-byte teleport record intact.
The old code fails on level 9 pad 35 with a tough marine at destination 36.
All 480 cases, the complete feedback suite, Windows Release build and
single-threaded Emscripten Release build pass after the correction.

An original-game replay is still pending. Capture `PLR1_Control` /
`PLR2_Control` immediately before and after their arrival `Collision`
call, including `CollideFlags`, `hitwall`, `newx/newz/newy`, the tough
marine's object/point records and the source/destination room pointers.
Compare this with the native fixture before and after that enemy uses
the source pad.

## Silent corpse gibbing from explosions

By request, every retained enemy corpse can now be destroyed by an explosion
without replaying its pain or death sound. This deliberately extends the
original behavior, while reusing `amiga/Anims.s` `ComputeBlast` reach/visibility
and the existing `ExplodeIntoBits` debris effect.

The floater report reproduced before the fix: blasting its resting frame 20
corpse produced two sounds, including pain sample 8. Soft-dead floaters keep
their live enemy type, so queued blast damage re-entered the living death
handler. Other corpses use `OBJ_NBR_DEAD` and were skipped by blast targeting.
Their raw byte 19 stores the original enemy type rather than pending damage.

The shared blast path now recognizes both corpse formats. A reached corpse
spawns debris immediately, clears its burning/pending explosion state and
is removed. It bypasses living damage/death code and never queues damage
into the corpse's original-type tag. Copying that type into the debris
source preserves the existing per-type gib effect, including floaters.
Queued damage on dead floaters also bypasses pain/death processing; ordinary
non-explosive damage is discarded. Corpse blasts cannot repeat death blasts
or loot drops, or regenerate debris from an already removed body.

Animated alien, worm, rifle/shotgun/tough marine and small-red/claws corpses
are covered, along with falling and settled floaters. The existing corpse
metadata check protects decorative billboard and vector props. Living
damage/pain, blue-key drops, explosion audio and debris impact sounds retain
their existing behavior. No state or full-save layouts change.

`tests/feedback_corpses.c` covers the reported floater sound and 429
corpse-animation/blast-strength cases across all seven retained corpse
types, including dormant/burning bodies and repeat explosions. Further
checks cover zero/out-of-range blasts, props, keys, living pain, stale
queued damage and a full debris pool. Actual rocket floor impacts, grenade
fuses and barrel detonations remove corpses with only the weapon's explosion
sound. The robot death regression also checks that its own blast leaves
the blue key in a collectible zone.

The complete feedback suite, Windows Release build and single-threaded
Emscripten Release build pass.
