# AB3D I vs AB3D II: The Killing Grounds Engine Differences

This is a source-backed porting note comparing the engine in this AB3D I codebase with the sequel source for **Alien Breed 3D II: The Killing Grounds**.

Source shorthand used below:

- **AB3D I**: this repository, especially `amiga/` and `src/`.
- **TKG**: `<tkg-source>/`, the sequel checkout's `ab3d2_source/` directory.

The focus is engine architecture and porting risk. It is not a design argument that every TKG feature should move back into AB3D I.

## Executive Summary

Both games share the same family tree: fixed-point software 3D, convex zones, back-to-front rendering, sprite/object tables, 4096-step angle math, and Amiga-era CPU-conscious assembly.

TKG is not just "AB3D I with mouse look". It broadens the engine in several directions:

- More data-driven game and asset definitions through the game link file.
- A more explicit zone/PVS visibility system, including per-edge and lift/door-aware refinements in the rebuilt source.
- More rendering variants: hi-res paths, Gouraud-style wall/floor lighting, dynamic lighting toggles, RTG/chunky support, and CPU-specific draw/C2P routines.
- More player/view state: vertical look, keyboard look, center look, mouse inversion, crosshair/no-auto-aim preferences, and aim-aware projectile code.
- Wider runtime configuration: graphics, controls, gameplay preferences, stats/progress persistence, and development toggles.

AB3D I is tighter and more baked-in. That makes it simpler to preserve, but it also means TKG features should usually be ported as narrow slices, not by lifting whole subsystems.

## Main Differences

| Area | AB3D I / this port | TKG | Porting impact |
|------|--------------------|-----|----------------|
| Data ownership | Original source includes more game data directly in assembly and fixed binary structures. The PC port stages original data into modern loaders. | More data is controlled through the game link file and related tables: level names, object graphics, sounds, guns, bullets, alien definitions, wall graphics, floors, ambient SFX, music, echo data, and more. | TKG tables are useful references, but importing them wholesale would turn into a resource/data-loader refactor. |
| Level structure | Zone-based 2.5D maps with points, floor lines, object data, doors, lifts, switches, and zone graph data loaded from AB3D I formats. | Explicit `ZoneT`, `EdgeT`, and `PVST` style structures with zone, edge, point, height, water, brightness, teleport, noise, and PVS data. | The concepts overlap, but the structures are not drop-in replacements. |
| Visibility | Uses zone ordering/graph logic from the original AB3D I engine and the port's equivalent visibility code. | Editor-generated PVS lists, zone ordering reduction, edge-level PVS improvements, errata, and door/lift visibility masks in the rebuilt source. | This is one of the biggest engine differences. Porting it would affect traversal, clipping, overdraw, and object visibility. |
| Rendering target | Original AB3D I targets its Amiga chunky/copper-era renderer. The PC port renders through a host framebuffer/SDL/Web path while keeping AB3D I projection behavior. | Supports native Amiga and RTG-style paths, 320x240/320x256 modes, chunky-to-planar routines, Akiko support, and CPU-tuned 030/040/060 draw paths. | Most TKG display backend work is not directly useful to this SDL/Web port, except as algorithm reference. |
| Wall/floor renderer | AB3D I has textured wall/floor/ceiling drawing, water/sky handling, bumpy/chunky floor logic, and brightness handling from the original routines. | Adds or formalizes more draw variants, including hi-res wall paths, Gouraud wall/floor routines, draw quality preferences, dynamic lighting, and simple-lighting fallbacks. | Individual effects may port, but the full renderer is more complex than AB3D I needs. |
| Sprites and objects | Uses AB3D I object/sprite/vector-object behavior, now translated into `src/renderer.c`, `src/renderer_3dobj.c`, and object/gameplay modules. | Has its own high-resolution object drawing path and broader object definitions tied into TKG data tables. | Use TKG as reference for specific behaviors, but keep AB3D I anchoring, scale, and collision authoritative. |
| Player view | Originally fixed vertical view. This port now has optional TKG-style vertical look as a screen-center/view-offset feature. | Native engine carries aim speed, vertical look, keyboard look, center look, mouse inversion, and snap/view state in player structures. | The vertical look slice ports well because it does not require true pitch rotation. |
| Aiming and shooting | AB3D I shooting, auto-aim, and projectile behavior are the baseline. This port now conditionally uses TKG-style vertical aiming when mouse look is enabled. | Projectile and auto-aim code accounts for mouse/look state, no-auto-aim preference, and crosshair-oriented play. | This can stay feature-gated. Do not let it silently change fixed-view AB3D I weapon behavior. |
| Controls | The PC port has modern SDL input plus compatibility paths for original control behavior. | More built-in modern controls: mouse+keyboard defaults, original mouse option, always-run, look keys, invert mouse, and related preferences. | Good source for optional controls, but each should remain configurable for AB3D I parity. |
| AI/object movement | AB3D I enemy/object behavior is based on original movement and alien-control code, with port-side high-frame-rate fixes layered carefully. | TKG has newer alien/object movement modules and more data-driven definitions. | High risk to mix wholesale. AI changes can easily affect level balance and collision assumptions. |
| Doors, lifts, and visibility | Doors/lifts exist as level mechanics and are integrated into AB3D I zone/render behavior. | Rebuilt TKG adds door/lift-aware PVS masking to reduce invalid visibility and overdraw. | Attractive idea, but it belongs with a visibility rewrite, not as a small feature port. |
| Lighting | AB3D I has palette/brightness-style lighting, wall side brightness, floor effects, and special cases such as light beams. | Broader lighting controls, dynamic lighting toggle, simple lighting fallback, Gouraud steps, and development lighting options. | Can be ported effect by effect, but lighting touches many render assumptions. |
| Audio/ambience | AB3D I uses its original sound/music/event model. | TKG link data includes more ambient SFX, echo tables, level music mapping, and preference/state plumbing. | Echo/ambient ideas may port, but the data format and mixer assumptions need mapping. |
| Saves/settings/progress | This port has modern `.ini` settings and save compatibility code around AB3D I state. | TKG persists more player settings, preferences, stats, and progress. | Preference-style features port cleanly if added as independent fields with versioned save handling. |
| Modding/resource overrides | AB3D I is mostly original-data driven. | TKG rebuilt source supports more overrides and extension points, including floor/wall texture overrides and extra game properties. | Useful long-term direction, but it expands loader and asset ownership complexity. |

## Shared DNA

The two engines are close enough that small gameplay/view slices can port cleanly:

- 2.5D world model based around zones and vertical floor/roof information.
- Fixed-point coordinate and angle math.
- Back-to-front software rendering rather than hardware 3D.
- Separate world projection from weapon/HUD overlay.
- Object and projectile systems built around compact tables and per-tick updates.
- Amiga performance assumptions visible in the original layout and routine boundaries.

That shared DNA is why the vertical look work was a good fit. TKG's look model is a screen-center offset and aim-speed model, not a fully free-look polygon renderer. It gives the feel of looking up/down without requiring the AB3D I world renderer to become a true pitched camera.

## Data Model

AB3D I has a more compact and baked-in structure. The original code pulls in many data files directly and the current port mirrors those formats through loaders such as the level, object, texture, and asset paths.

TKG is more data-driven. Its game link file and related structures describe many things AB3D I treats as fixed or less externally configurable: guns, bullets, aliens, animations, object graphics, wall graphics, floor data, sound names, ambient sound, music, echo data, and level metadata.

For this port, that means TKG source is excellent as a behavior reference, but not a plug-in content system. Borrowing one field or behavior is often fine. Borrowing the whole game-link model would change how AB3D I owns its assets.

## Zone Visibility and Ordering

AB3D I already has zones, zone ordering, and clipping logic. It is not a raycaster. The important difference is that TKG formalizes and extends this side of the engine much further.

TKG's documented model uses convex polygon zones and editor-generated Potentially Visible Sets. The rebuilt source adds fixes and refinements around:

- Reducing zone ordering work based on player movement.
- PVS errata for problem cases.
- Per-edge visibility information.
- Door and lift masks so closed or height-changing geometry can affect visibility.
- Adjacent-zone clipping improvements.

Porting this area wholesale would be risky. It would affect what gets rendered, how clips are generated, when objects are visible, and how much overdraw the renderer expects. It is a renderer architecture project, not a small parity patch.

## Rendering Pipeline

AB3D I's renderer is the smaller engine:

- Textured wall columns.
- Floor and ceiling span drawing.
- Water and sky cases.
- Bumpy/chunky floor effects.
- Sprite and vector-object drawing.
- Brightness and special-case effects from the original routines.

TKG expands this into a broader renderer family:

- Hi-res wall and object paths.
- Gouraud-style wall and floor routines.
- Dynamic/simple lighting preferences.
- RTG/chunky output paths.
- Native Amiga C2P paths, including CPU-specific and Akiko variants.
- More rendering preferences exposed to the player or developer.

The practical lesson: renderer features should be ported one at a time and tested against AB3D I content. TKG draw routines often assume TKG's zone, lighting, clip, or display backend state.

## Player View, Aim, and Controls

This is the area with the cleanest proven transfer.

AB3D I originally uses a fixed vertical view. The port now has optional vertical mouse look based on the TKG approach: a clamped vertical screen-center offset, with current/old state for smooth rendering and an `.ini` switch to keep fixed view as default.

TKG has a fuller control/view model:

- Mouse look and inverted mouse.
- Keyboard look up/down.
- Center-look behavior.
- Aim-speed state in the player structure.
- Shooting code aware of mouse/look state.
- Optional no-auto-aim and crosshair-oriented play.

The good boundary for AB3D I is feature-gating. Mouse-look aiming can exist when `mouse_look=1`, while fixed-view mode should continue to behave like AB3D I.

## Gameplay and Object System

AB3D I gameplay code is more compact and level-tuned. Enemy movement, projectile behavior, pickup amounts, animation cadence, sprite scaling, object collision, and soft-lock prevention all depend on AB3D I assumptions.

TKG's gameplay code is newer and broader, but it is also tied to TKG data tables and level design. Porting enemy or object movement directly could change balance, collision, and scripted behavior in ways that are hard to spot from code review alone.

Good candidates for selective reference:

- Specific input/view/aim mechanics.
- Preference toggles that can be isolated.
- Mathematical fixes where both engines use the same representation.
- Renderer effects that do not change world traversal.

Risky candidates:

- AI movement wholesale.
- Full projectile system replacement.
- Full object definition table replacement.
- TKG zone/PVS traversal inside AB3D I levels.

## Runtime and Platform

The current AB3D I port is already a modern runtime: SDL/native builds, web builds, `.ini` settings, versioned saves, and host-side rendering glue.

TKG's rebuilt source is modernized in a different direction: it still serves Amiga targets while adding RTG support, CPU-specific routines, C modules, preference files, and build variants. Some ideas translate well, but many backend details are specific to Amiga display and CPU paths.

For this repository, TKG's platform code is usually less important than its gameplay/view algorithms. The SDL/Web port already solves display and input differently.

## Practical Porting Guidance

Low-risk slices:

- Optional control preferences.
- Mouse look, invert Y, and center-look style behavior.
- Aim-aware projectile offsets when gated behind mouse look.
- Crosshair/no-auto-aim options if kept optional and save-compatible.
- Small math or timing corrections that preserve AB3D I behavior.

Medium-risk slices:

- Per-level texture/floor overrides.
- Ambient/echo/music mapping.
- More dynamic lighting controls.
- TKG-style configuration persistence.
- Expanded weapon/projectile metadata.

High-risk engine changes:

- Replacing AB3D I zone traversal with TKG PVS traversal.
- Porting TKG's per-edge/lift/door visibility system without a renderer design pass.
- Importing the TKG game-link data model wholesale.
- Replacing AB3D I AI/object movement with TKG modules.
- Reworking the renderer around TKG hi-res/Gouraud paths as a batch.

The safest rule is: use TKG as the reference for a specific feature, then adapt that feature to AB3D I's data and renderer boundaries.

## Source Map

AB3D I source points:

- `README.md`
- `amiga/AB3DI.s`
- `amiga/WallRoutine3.ChipMem.s`
- `amiga/ObjDraw3.ChipRam.s`
- `amiga/ObjectMove.s`
- `amiga/AlienControl.s`
- `amiga/PlayerShoot.s`
- `amiga/Plr1Control.s`
- `amiga/OrderZones.s`
- `amiga/Anims.s`
- `src/renderer.c`
- `src/renderer_3dobj.c`
- `src/level.h`
- `src/visibility.c`
- `src/player.c`
- `src/movement.c`
- `src/objects.c`

TKG source points:

- `<tkg-source>/README.md`
- `<tkg-source>/docs/README.md`
- `<tkg-source>/docs/PVS.md`
- `<tkg-source>/defs.i`
- `<tkg-source>/hires.s`
- `<tkg-source>/plr1control.s`
- `<tkg-source>/modules/player.s`
- `<tkg-source>/newplayershoot.s`
- `<tkg-source>/newanims.s`
- `<tkg-source>/newaliencontrol.s`
- `<tkg-source>/objectmove.s`
- `<tkg-source>/objdrawhires.s`
- `<tkg-source>/hireswall.s`
- `<tkg-source>/hiresgourwall.s`
- `<tkg-source>/modules/draw.s`
- `<tkg-source>/modules/draw/draw_wall.s`
- `<tkg-source>/modules/draw/draw_wall_060.s`
- `<tkg-source>/modules/draw/draw_floor.s`
- `<tkg-source>/modules/draw/draw_floor_060.s`
- `<tkg-source>/modules/c2p/`
- `<tkg-source>/c/draw.c`
- `<tkg-source>/c/game_preferences.c`
- `<tkg-source>/c/game_properties.c`
- `<tkg-source>/c/zone_edge_pvs.c`
- `<tkg-source>/c/zone_errata.c`
- `<tkg-source>/c/zone_liftable_pvs.c`
