ALIEN BREED HD - DOS / 486 BUILD

Requires a 486 DX2/DX4 with x87 FPU, 4 MB RAM, DOS 5+ and VESA RGB565
video (a mode of at least 320x200). A VBE 2 BIOS or UNIVBE is recommended.
The smallest available RGB565 mode is selected; many cards use 320x240.

Unzip the entire package into a writable directory, then run ABHD.BAT.
CWSDPMI.EXE is included; the launcher disables disk swapping. Use a lean DOS
boot with XMS enabled and EMS disabled, aiming for about 600 KB free conventional
RAM at the DOS prompt for texture/sample banks, the runtime and audio driver. Extra TSRs
and larger video modes can require more RAM. Set BLASTER to your card's resources beforehand;
if unset, the launcher uses A220 I7 D1 H5 T6. Sound Blaster 16 and compatible
cards are supported by the SDL DOS backend. To run without sound:
  SET AB3D_DOS_SOUND=0
  ABHD

Keyboard: arrows move/turn; A/D strafe; Ctrl fire; Space use; 1-6 weapons;
Esc menu; Enter selects; Tab map; F5/F9 save/load. See the in-game options.
DOS menus use the keyboard. Optional mouse input requires a DOS mouse driver:
  SET AB3D_DOS_MOUSE=1
This enables automatic capture in-game; Esc releases it for menus. Mouse
behavior still needs real-hardware validation. DOS always uses a full screen.

AB3D.INI defaults to true 320x200 rendering, one renderer thread and normal
game rules. The game draws native RGB565 pixels and presents them 1:1 with
SDL: no framebuffer color conversion, enlargement or OpenGL. S3's 320x240
mode centers the 320x200 view. Other modes also retain 1:1 pixels.

Menus, HUD, text and the map use native pixel sizes through SDL. Weapons
still scale to the game view; their exact sampled frames are cached for
fast SDL blits. Long level text wraps into pages; press a key to continue.
UI uses opaque colors and cutout masks without alpha blending. Uncollected
keys appear grey; collected keys use their colors. Damage uses a red border.
Menus and text stay inside the visible screen, with paging for long text.
Key underlines and bars leave room for the complete HUD in 200-line output.
Output retains all 4096 Amiga colors. The DOS build uses the same game logic
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
source patch (Sound Blaster ring, event scheduling and 486 CPU probe).

CWSDPMI Copyright (C) 1995-2010 Charles W Sandmann. See CWSDPMI.DOC for
redistribution terms. You have the right to receive its source and updates:
https://www.delorie.com/pub/djgpp/current/v2misc/ (csdpmi7s.zip sources).
