/* Production firing, pickups and recharge for the rifle/flamethrower pool. */
#include "../src/player.c"
#include <assert.h>
#undef printf

RendererState g_renderer;
int renderer_get_width(void) { return 320; }
int renderer_get_height(void) { return 200; }
static int played_sample;
void audio_play_sample(int id, int volume) { played_sample = id; (void)volume; }
bool input_gamepad_fire_held(void) { return false; }

static void shoot(GameState *state, int player, int gun)
{
    PlayerState *p = player == 1 ? &state->plr1 : &state->plr2;
    p->gun_selected = gun;
    p->time_to_shoot = 0;
    p->p_fire = p->p_clicked = 1;
    if (player == 1) player1_shoot(state); else player2_shoot(state);
    p->p_fire = p->p_clicked = 0;
}

int main(void)
{
    math_tables_init();
    for (int player = 1; player <= 2; player++) {
        static GameState state;
        static GameObject objects[1], shots[PLAYER_SHOT_SLOT_COUNT];
        static uint8_t points[(PLAYER_SHOT_SLOT_COUNT + 1) * 8];
        game_state_init(&state);
        game_state_setup_default(&state);
        memset(objects, 0, sizeof(objects)); memset(shots, 0, sizeof(shots));
        OBJ_SET_CID(&objects[0], -1);
        state.level.object_data = (uint8_t *)objects;
        state.level.object_points = points;
        state.level.num_object_points = PLAYER_SHOT_SLOT_COUNT + 1;
        state.level.player_shot_data = (uint8_t *)shots;
        for (int i = 0; i < PLAYER_SHOT_SLOT_COUNT; i++) {
            OBJ_SET_CID(&shots[i], i + 1); OBJ_SET_ZONE(&shots[i], -1);
        }
        state.temp_frames = 1;
        PlayerState *p = player == 1 ? &state.plr1 : &state.plr2;
        PlayerState *other = player == 1 ? &state.plr2 : &state.plr1;
        player_apply_weapon_config(&state);
        int16_t *pool = player_weapon_ammo_ptr(p, AB3D_GUN_PISTOL);
        assert(pool == player_weapon_ammo_ptr(p, AB3D_GUN_FLAMETHROWER));
        assert(*pool == 160); /* Twenty starting rounds, shared immediately. */
        p->gun_data[AB3D_GUN_FLAMETHROWER].ammo = 999; /* Legacy counter is unused. */
        shoot(&state, player, AB3D_GUN_PISTOL);
        assert(*pool == 152);
        shoot(&state, player, AB3D_GUN_FLAMETHROWER);
        assert(*pool == 144 && SHOT_SIZE(shots[0]) == AB3D_GUN_FLAMETHROWER);
        assert(other->gun_data[AB3D_GUN_PISTOL].ammo == 160);
        assert(p->gun_data[AB3D_GUN_FLAMETHROWER].ammo == 999);

        p->gun_data[AB3D_GUN_PLASMA].visible = -1;
        p->gun_data[AB3D_GUN_PLASMA].ammo = 160;
        shoot(&state, player, AB3D_GUN_PLASMA);
        assert(*pool == 144 && p->gun_data[AB3D_GUN_PLASMA].ammo == 152);

        for (int gun = 0; gun <= 3; gun += 3) {
            *pool = 7;
            shoot(&state, player, gun);
            assert(*pool == 7 && played_sample == 12);
        }
        *pool = 0;
        player_flamethrower_mark_used(&state, player);
        for (int i = 0; i < 149; i++)
            player_flamethrower_tick_recharge(&state, p, player, false);
        assert(*pool == 0);
        state.temp_frames = 5;
        player_flamethrower_tick_recharge(&state, p, player, false);
        assert(*pool == 8);
        player_flamethrower_tick_recharge(&state, p, player, false);
        assert(*pool == 16);
        state.temp_frames = 200;
        player_flamethrower_tick_recharge(&state, p, player, false);
        assert(*pool == 160);

        /* Holding either shared weapon pauses refill, even between shots. */
        for (int gun = 0; gun <= 3; gun += 3) {
            *pool = 80;
            player_flamethrower_reset_recharge(&state, player);
            p->gun_selected = gun; p->time_to_shoot = 20; p->p_fire = 1;
            state.temp_frames = 5;
            player_shoot_internal(&state, p, player,
                                 player == 1 ? default_plr1_guns : default_plr2_guns);
            assert(*pool == 80);
            assert(*player_flamethrower_recharge_delay_ptr(&state, player) == 150);
        }
        p->p_fire = 0; p->time_to_shoot = 0;

        /* Pickups go straight to the shared counter, including above 20 rounds. */
        *pool = 160;
        GameObject clip = {0};
        OBJ_SET_CID(&clip, 0); OBJ_SET_ZONE(&clip, 0);
        clip.obj.number = OBJ_NBR_AMMO; obj_sw(clip.raw + 18, AB3D_GUN_PISTOL);
        p->zone = other->zone = 0;
        p->xoff = 0; other->xoff = 1000 * 65536;
        state.mode = player == 1 ? MODE_SINGLE : MODE_SLAVE;
        object_handle_ammo(&clip, &state);
        assert(*pool == 280 && OBJ_ZONE(&clip) == -1);
        state.temp_frames = 200;
        player_flamethrower_tick_recharge(&state, p, player, false);
        player_apply_weapon_config(&state);
        assert(*pool == 280); /* Neither recharge nor config/load application trims pickups. */

        state.cfg_flamethrower_weapon = false;
        player_apply_weapon_config(&state);
        assert(*pool == 280 && p->gun_data[AB3D_GUN_FLAMETHROWER].visible == 0);
        *pool = 80;
        player_flamethrower_tick_recharge(&state, p, player, false);
        assert(*pool == 80);
        state.cfg_flamethrower_weapon = true;
        player_apply_weapon_config(&state);
        assert(*pool == 80);
        state.infinite_ammo = true;
        shoot(&state, player, AB3D_GUN_PISTOL);
        assert(*pool == 80);
        shoot(&state, player, AB3D_GUN_FLAMETHROWER);
        assert(*pool == 72); /* Preserve the existing rechargeable-flamer cheat behavior. */
    }
    puts("PASS: both players share rifle/flamethrower ammo; firing, empty clicks, pickups, refill delay/rate/ceiling, config and other guns");
    return 0;
}
