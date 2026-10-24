ALIEN BREED 3D - GENERATIVE WEAPON REDRAWS, 4X

24 redrawn animation frames for all six populated weapon slots:
gun 0 pistol, gun 1 plasma, gun 2 rocket, gun 3 flamethrower,
gun 4 grenade launcher, gun 7 shotgun.
Slots 5 and 6 have no original artwork and are not fabricated here.

The artwork was redrawn with the built-in image-generation tool using the
original sprites as visual references. No Python drawing/upscaling script
was used. Native image operations only package generated artwork into exact
frame sizes and assemble the contact-sheet preview.

Brown side shapes represent fingers where present in the original.
The blue plasma rifle has no visible hands or fingers in any frame.

FILES
- gunN_frameF.png: individual transparent RGBA frames.
- gunN-sheet-4x.png: 2-by-2 sheets in frame order 0,1 / 2,3.
- masters/: original generated full-resolution sheets.
- preview.png: labeled overview of the six idle frames.
- manifest.json: frame sizes, mappings and alpha validation counts.
- prompts.txt: generation prompts and the plasma correction prompt.

DIMENSIONS
Original complete frames are 96x59, except the rocket which is 96x79.
Delivered 4x frames are 384x236, or 384x316 for the rocket.

RUNTIME
This is an artwork delivery. The current game overlay and legacy WAD importer
use 96x58 frames; they do not yet load these high-resolution PNG assets.
Importing these through the legacy importer would discard the extra detail.
Original runtime assets have not been overwritten.
