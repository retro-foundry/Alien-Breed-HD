/* INT 33h relative counts must retain source units at every DOS view size. */
#include "../src/player.c"
#include <assert.h>
#undef printf

RendererState g_renderer;
int renderer_get_width(void) { return g_renderer.present_width; }
int renderer_get_height(void) { return g_renderer.present_height; }

int main(void)
{
    static GameState state;
    game_state_init(&state);
    assert(state.cfg_mouse_look);
    const int sizes[][2] = {{320,200},{320,240},{256,192},{192,144},{128,96}};
    for (unsigned size=0; size<sizeof(sizes)/sizeof(sizes[0]); size++) {
        g_renderer.present_width=sizes[size][0];
        g_renderer.present_height=sizes[size][1];
        for (int axis=0; axis<2; axis++)
        for (int player=0; player<2; player++)
        for (int delta=-40; delta<=40; delta++) {
            PlayerState *plr=player ? &state.plr2 : &state.plr1;
            assert(player_mouse_delta_to_reference(&state,plr,axis,delta)==delta);
        }
    }
    puts("PASS: DOS raw mouse counts retain small motion and sensitivity at all view sizes");
    return 0;
}
