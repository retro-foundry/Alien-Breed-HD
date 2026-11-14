/* AB3DI.s GunAnims[*][0] is each weapon's authored rest frame. Exercise the
 * real selection paths and renderer metadata, including interrupted reloads. */
#include "../src/player.c"
#include <assert.h>
#undef printf

static int wheel_steps, gamepad_steps;
bool input_consume_key_press(uint8_t key) { (void)key; return false; }
bool input_gamepad_duck_toggle_requested(void) { return false; }
void input_read_mouse(MouseState *mouse)
{
    memset(mouse, 0, sizeof(*mouse));
    mouse->wheel_y = (int16_t)wheel_steps;
    wheel_steps = 0;
}
int16_t input_consume_gamepad_weapon_cycle_steps(void)
{
    int16_t steps = (int16_t)gamepad_steps;
    gamepad_steps = 0;
    return steps;
}

static GameState state;
static uint8_t keys[128], old_space;
static const KeyBindings bindings = { .operate = 64, .duck = 65 };
static unsigned checked;

static void controls(PlayerState *player)
{
    player_always_keys(player, keys, &bindings, &old_space, &state);
}

static void expect_draw_slot(PlayerState *player, int frame)
{
    int slot = -1;
    assert(renderer_get_gun_draw_info(&state, &slot, NULL, NULL, NULL, NULL));
    assert(slot == player->gun_selected * 4 + frame);
    checked++;
}

static void prepare(PlayerState *player, int gun, int frame)
{
    memset(keys, 0, sizeof(keys));
    wheel_steps = gamepad_steps = 0;
    player->gun_selected = (int16_t)gun;
    player->gun_frame = (int16_t)frame;
    player->time_to_shoot = 17;
    for (int i = 0; i < GUN_KEY_COUNT; i++) {
        player->gun_data[gun_key_map[i]].visible = -1;
        player->gun_data[gun_key_map[i]].ammo = 240;
    }
}

int main(void)
{
    /* Draw-info resolves metadata only and does not read asset bytes. These
     * nonnull markers enable that path without initializing a video device. */
    static const uint8_t asset_available;
    g_renderer.width = g_renderer.present_width = 320;
    g_renderer.height = g_renderer.present_height = 240;
    g_renderer.gun_wad = g_renderer.gun_ptr = g_renderer.gun_pal = &asset_available;
    for (int player_number = 1; player_number <= 2; player_number++) {
        game_state_init(&state); game_state_setup_default(&state);
        state.mode = player_number == 1 ? MODE_SINGLE : MODE_SLAVE;
        PlayerState *p = player_number == 1 ? &state.plr1 : &state.plr2;
        PlayerState *other = player_number == 1 ? &state.plr2 : &state.plr1;
        other->gun_frame = 42;
        for (int from = 0; from < GUN_KEY_COUNT; from++) {
            int source = gun_key_map[from];
            for (int tick = 0; tick <= gun_anims[source].num_frames; tick++) {
                for (int to = 0; to < GUN_KEY_COUNT; to++) {
                    int target = gun_key_map[to];
                    prepare(p, source, tick);
                    keys[to + 1] = 1;
                    controls(p);
                    assert(p->gun_selected == target);
                    assert(p->gun_frame == (source == target ? tick : 0));
                    expect_draw_slot(p, source == target ? gun_anims[source].frames[tick] : 0);
                    assert(p->time_to_shoot == 17 && other->gun_frame == 42);
                    for (int i = 0; i < GUN_KEY_COUNT; i++)
                        assert(p->gun_data[gun_key_map[i]].ammo == 240);
                }
            }
        }
        /* A long shotgun countdown must not reappear as another gun's firing
         * frames when it eventually falls into that gun's shorter table. */
        prepare(p, AB3D_GUN_SHOTGUN, 63);
        keys[1] = 1; controls(p); memset(keys, 0, sizeof(keys));
        for (int tick = 0; tick < 64; tick++) {
            if (p->gun_frame > 0) p->gun_frame--;
            controls(p); expect_draw_slot(p, 0);
        }
        /* Number-key priority is resolved once, so multiple held selection
         * keys do not keep resetting the final selected weapon's animation. */
        prepare(p, AB3D_GUN_SHOTGUN, 20);
        keys[1] = keys[2] = 1; p->p_fire = 1;
        controls(p);
        assert(p->gun_selected == AB3D_GUN_SHOTGUN && p->gun_frame == 20 && p->p_fire == 1);
        expect_draw_slot(p, gun_anims[AB3D_GUN_SHOTGUN].frames[20]);
        for (int device = 0; device < 2; device++) for (int direction = -1; direction <= 1; direction += 2) {
            prepare(p, AB3D_GUN_PISTOL, 2);
            if (device) gamepad_steps = direction; else wheel_steps = direction;
            controls(p);
            assert(p->gun_selected == (direction < 0 ? AB3D_GUN_SHOTGUN : AB3D_GUN_FLAMETHROWER));
            assert(p->gun_frame == 0 && p->time_to_shoot == 17);
            expect_draw_slot(p, 0);
            /* Hidden guns are skipped, and cycling back to the only owned gun
             * preserves its current animation. */
            prepare(p, AB3D_GUN_PISTOL, 2);
            for (int i = 1; i < GUN_KEY_COUNT; i++) p->gun_data[gun_key_map[i]].visible = 0;
            if (device) gamepad_steps = direction; else wheel_steps = direction;
            controls(p);
            assert(p->gun_selected == AB3D_GUN_PISTOL && p->gun_frame == 2);
            expect_draw_slot(p, 2);
        }
        prepare(p, AB3D_GUN_PISTOL, 2);
        p->gun_data[AB3D_GUN_SHOTGUN].visible = 0; keys[2] = 1; controls(p);
        assert(p->gun_selected == AB3D_GUN_PISTOL && p->gun_frame == 2);
        expect_draw_slot(p, 2);
        prepare(p, -1, 5); controls(p);
        assert(p->gun_selected == AB3D_GUN_PISTOL && p->gun_frame == 0);
        expect_draw_slot(p, 0);
        prepare(p, AB3D_GUN_FLAMETHROWER, 4);
        state.cfg_flamethrower_weapon = false; player_apply_weapon_config(&state);
        assert(p->gun_selected == AB3D_GUN_PISTOL && p->gun_frame == 0);
        expect_draw_slot(p, 0);
    }
    printf("PASS: %u weapon selection/draw checkpoints cover both players, every firing/reload tick, keyboard/wheel/controller, held/hidden guns, availability and unchanged ammo/cooldown\n", checked);
    return 0;
}
