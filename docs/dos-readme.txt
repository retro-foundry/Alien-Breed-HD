ALIEN BREED HD - DOS / 486 BUILD

Requires a 486, 4 MB RAM, DOS 5+ and VGA.
Run ABHD.BAT for automatic video selection: RGB565, then RGB555/RGB444
high-color VESA modes, then standard 320x200 256-color VGA. High-color modes
retain all 4096 source colors. Basic VGA uses the existing palette per level;
some colors/shades are merged. A VBE 2 BIOS or UNIVBE helps high-color support
but is not required for the VGA fallback.
An FPU is optional. A DX2/DX4 is recommended for speed; 486SX systems use
the included software coprocessor emulator automatically. No extra file is needed.

ABHD.EXE runs the high-color version directly; ABVGA.EXE --dos-vga forces VGA.
Only one executable runs at a time. The launcher switches only when no usable
high-color mode exists; other errors are left visible in ab3d.log.

Unzip the entire package into a writable directory, then run ABHD.BAT.
When updating, keep the program files and DATA from the same package.
CWSDPMI.EXE is included; the launcher disables disk swapping. Use a lean DOS
boot with XMS enabled and EMS disabled, aiming for about 600 KB free conventional
RAM at the DOS prompt for texture/sample banks, the runtime and audio driver. Extra TSRs
and larger video modes can require more RAM. Set BLASTER to your card's resources beforehand;
if unset, the launcher uses A220 I7 D1 H5 T6. Sound Blaster 16 and compatible
cards are supported by the SDL DOS backend. To run without sound:
  SET AB3D_DOS_SOUND=0
  ABHD.BAT

Keyboard: arrows move/turn; A/D strafe; Ctrl fire; Space use; 1-6 weapons;
Esc menu; Enter selects; Tab map; F5/F9 save/load. See the in-game options.
Weapon keys: 1 pulse rifle, 2 shotgun, 3 plasma, 4 grenade, 5 rocket,
6 flamethrower. The flamethrower shares the rifle's ammunition.
New installs show a yellow crosshair. Esc > Options > Crosshair changes its
color or switches it off. For an existing INI, select YELLOW there or set
crosshair=3 in AB3D.INI.
DOS menus use the keyboard. Mouse input is enabled automatically when a DOS
mouse driver is installed; DOSBox's built-in driver works. Capture starts
in-game; Esc releases it for menus. Set AB3D_DOS_MOUSE=0 to disable it. Mouse
Look in Options, or mouse_look=1 in AB3D.INI, enables vertical look. Options
save with DOS 8.3 filenames. In DOSBox, click inside its window to capture the
host mouse; Ctrl+F10 releases it. Mouse behavior still needs real-hardware
validation. DOS always uses a full screen.

AB3D.INI requests a small video mode, one renderer thread and normal game
rules. By default the world fills the selected mode behind the HUD: 320x240
on S3 with RGB565, or standard 320x200 VGA in the indexed build. Existing INI
files requesting 320x200 also fill the selected mode. Native RGB565 pixels
are presented 1:1 with SDL, without conversion, enlargement or OpenGL.

For better performance, open Options and change SCREEN SIZE with Left/Right
or Enter. Choose 100%, 80%, 60% or 40%. Smaller settings draw the 3D view in
the middle of a black border; the HUD and menus keep their native size.
The choice saves as dos_screen_size in AB3D.INI; the default is 100%.

Menus, HUD and the map use native pixel sizes through SDL. Weapons
still scale to the game view; their exact sampled frames are cached for
fast SDL blits. Level briefings use VGA 80-column text mode so the complete
original text fits on one screen. Press a key after loading to continue;
the game restores its graphics mode, palette and mouse input automatically.
UI uses opaque colors and cutout masks without alpha blending. Uncollected
keys appear grey; collected keys use their colors. Damage flashes the full
view red while keeping the scene visible and the HUD readable.
HUD keys are prefiltered to 16x16 before loading, then drawn without scaling.
Menus and text stay inside the visible screen.
Key underlines and bars leave room for the complete HUD in 200-line output.
High-color output retains all 4096 Amiga colors; VGA uses 256 colors per level.
The DOS build uses the same game logic
and source assets as the desktop port. Original samples mix at 8007 Hz;
music is rendered offline and streamed. Levels are unpacked offline into
identical raw data to reduce loading memory and CPU work. Save payloads are validated and then
streamed into the reloaded level, without staging another level in RAM.
Supersampling must remain 1. Gamepads are not included in the DOS build.

Startup errors and diagnostics are recorded in AB3D.LOG. This is an initial
DOS port, tested in DOSBox 486 mode; real hardware performance still needs
measurement. Fixed-cycle emulator results and build instructions are in the
source repository's docs/dos-build.md.

Source: https://github.com/retro-foundry/Alien-Breed-HD
Original Alien Breed 3D: Team17; original source and data are in amiga/.
DOS SDL transport adapted from the local GloomAmiga port. SDL is distributed
under the zlib license; see SDL-LIC.TXT. Our DOS SDL changes are marked in the
source patches (Sound Blaster ring, event scheduling, 486 CPU probe, native
presentation and relative mouse input).

CWSDPMI Copyright (C) 1995-2010 Charles W Sandmann. See CWSDPMI.DOC for
redistribution terms. You have the right to receive its source and updates:
https://www.delorie.com/pub/djgpp/current/v2misc/ (csdpmi7s.zip sources).
