/*
 * Alien Breed 3D I - PC Port
 * game_loop.h - Main per-frame game loop
 *
 * Translated from: AB3DI.s mainLoop (~line 1296)
 *
 * Each frame:
 *   1. Handle pause
 *   2. Wait for vblank (frame sync)
 *   3. Swap screen buffers
 *   4. Handle gun selection / ammo display
 *   5. Animate water
 *   6. Save old positions
 *   7. Multiplayer sync (or single player cheat check)
 *   8. Player control (PLR1_Control / PLR2_Control)
 *   9. Zone brightness updates
 *  10. Visibility checks (multi)
 *  11. Object movement/animation (ObjMoveAnim)
 *  12. Energy/ammo bars
 *  13. Draw display (rendering pipeline)
 *  14. Copy copper buffer
 *  15. Object worry flags
 *  16. Check quit/death/level-end conditions
 *  17. Loop back to 1
 */

#ifndef GAME_LOOP_H
#define GAME_LOOP_H

#include "game_state.h"

#include <SDL.h>

/* Persistent state for one in-level session (VBlank accums, FPS meter, etc.) */
typedef struct GameLoopCtx {
    int frame_count;
    int logic_count;
    Uint32 last_ticks;
    int pending_vblanks;
    Uint32 vblank_remainder_ms;
    Uint32 fps_log_start_ms;
    int fps_frames_in_period;
} GameLoopCtx;

void game_loop_ctx_init(GameLoopCtx *ctx, GameState *state);
void game_loop_tick(GameState *state, GameLoopCtx *ctx);

/* Run the main game loop until level ends, player dies, or quit */
void game_loop(GameState *state);

#endif /* GAME_LOOP_H */
