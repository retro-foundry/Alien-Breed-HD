ALIEN BREED 3D I (PC PORT)
CONTROLS

MOVEMENT
- W or Up Arrow: Move forward
- S or Down Arrow: Move backward
- A or . : Strafe left
- D or / : Strafe right
- Left Arrow / Right Arrow: Turn left/right
- Mouse X: Turn left/right (when mouse is captured)
- Shift (Left or Right), or Caps Lock while enabled: Run
- C: Duck (toggle)
- L: Look behind

ACTIONS
- Left Mouse Button: Fire
- Right Alt / Left Alt / Left Ctrl / Right Ctrl: Fire
- Space: Operate / Use
- Right Mouse Button: Operate / Use (when mouse is captured)

GAMEPAD (MODERN FPS LAYOUT)
- Left Stick: Move
- Right Stick (X): Turn / look left-right
- Right Trigger: Fire; controller aim assist when mouse look is on
- Left Trigger or A: Operate / Use
- B (press): Duck toggle
- Left Bumper / Right Bumper: Previous / next weapon
- Left Stick Click (or full stick tilt): Run
- Right Stick Click or Y: Look behind
- Start: Open the in-game menu
- Back: Toggle automap
- D-pad Up / Down: Automap zoom in / out

MENU
- D-pad or Left Stick: Move selection
- A or Start: Select
- B or Back: Back / close menu

WEAPONS
- 1-6 (or Numpad 1-6): Select weapon
- Mouse Wheel: Cycle weapons

GAME / UI
- Quit: Alt + F4
- P: Pause
- Tab: Toggle automap
- Page Up / Page Down: Automap zoom in/out
- Esc: Open the in-game menu; also releases mouse capture when captured
- Options: Invert Mouse toggles vertical mouse-look inversion and saves the setting.
- Options: HUD switches between BARS (default) and the previous NUMBERS display.
- Options: Run Speed adjusts from 50% to 200% in 10-point steps and saves the setting.
- Autosaves: Load from the Autosaves menu
- F5/F9: Quick save/load only when quicksave_load=1 is enabled in ab3d.ini
- F6: Toggle Gouraud visualize mode

COMBAT / HUD
- The starting pulse rifle (key 1) and flamethrower (key 6) share ammunition.
  Both use one round per shot. After three seconds without firing either,
  automatic refill restores up to 20 rounds at the existing rate.
  Collected ammunition above 20 rounds is kept.
- Normal mutant rifles use original instant-hit attacks by default. Set
  marine_hitscan_projectiles=1 in ab3d.ini to enable the optional orange shots.
- Shotgun soldiers fire five instant-hit pellets with the shotgun sound.
- Taking damage flashes the view red; repeated hits refresh the pulse.
- Collected keys are opaque and underlined; missing keys are faint.
- Compact translucent vertical health (left) and ammo (right) bars replace the
  numbers, using the original green/yellow/red palette.
  Ammo uses the original bar scale (full at 63 displayed rounds).
- Water enables a software low-pass audio filter, including with post_tint=0.
- Browser: finishing the final level returns to the title menu after the fade.
- Run Speed: 100% is original Amiga speed; the default 130% retains the PC boost.
  Walking speed is unchanged.
- Enemy attacks, cries and deaths, and wall fire-spouts fade with distance.

MOUSE CAPTURE
- The mouse pointer is visible until you left-click inside the game window to capture it (windowed or fullscreen).
- Press Esc while captured to release capture, show the pointer again, and open the menu.

LINUX BUILD
- Extract the entire .tar.gz into a writable directory and run ./ab3d1 inside the extracted folder.
- Keep data/ and fonts/ beside the executable.
- Linux x64 releases are built on Ubuntu 22.04 and use system audio/display libraries.

WEB BUILD
- The web build plays all 16 levels, with the same progression and ending as the desktop build.
- Browser autosaves, menu options, and settings persist in local storage for the site.
