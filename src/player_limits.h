#ifndef AB3D_PLAYER_LIMITS_H
#define AB3D_PLAYER_LIMITS_H

/* Current port allowance shared by player control, movement defaults and
 * editor stair authoring. JSON numeric literals are C integer constants. */
enum {
    AB3D_PLAYER_STEP_UP_WORLD =
#include "player_step_up.json"
};

#endif
