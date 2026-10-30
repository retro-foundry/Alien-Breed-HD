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

The level 8 walker was checked against the authored `level_h` geometry and
`Robot.s` movement dimensions. It can cross all four starting-room exits;
its movement and size were left unchanged because no blocked gap was identified.

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
