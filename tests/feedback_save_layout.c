/* Historical v7 fixture: wasm32 GameState from 38859d9^:src/game_state.h,
 * before authored draw groups enlarged its embedded LevelState. Generated
 * with the scalar values asserted below; contains no pointers or user data. */
#include "../src/player.c"
#include <assert.h>
#undef printf

int main(void)
{
    static GameState restored, current;
    FullSaveHeader hdr = {0};
    FILE *f = fopen("tests/fixtures/save-v7-before-draw-groups.bin", "rb");
    assert(f);
    hdr.version = 7;
    hdr.game_state_size = 5960;
    assert(player_save_read_game_state(f, &hdr, &restored));
    fclose(f);
    assert(restored.mode == MODE_SINGLE);
    assert(restored.plr1.energy == 87 && restored.plr1.xoff == 123456);
    assert(restored.plr1.zoff == -654321);
    assert(restored.current_level == 11 && restored.max_level == 15);
    assert(restored.running && restored.do_anything && !restored.finished_level);
    assert(restored.energy == 87 && restored.temp_frames == 1);
    assert(restored.key_map[3] == 1 && restored.end_zones[11] == 132);
    assert(restored.enemy_burn_time_left[12] == 75);
    assert(restored.cfg_hud_bars && restored.cfg_run_speed == 130);

    /* v7 saves made after the layout change and new v8 saves both retain
     * their offsets. A short payload must fail rather than partially load. */
    current = restored;
    current.cfg_dos_screen_size = 80;
    for (unsigned version = 7; version <= SAVE_VERSION_FULL; version++) {
        hdr.version = version;
        hdr.game_state_size = sizeof(current);
        f = fopen("build/feedback-tests/save-layout.tmp", "w+b"); assert(f);
        assert(fwrite(&current, 1, sizeof(current), f) == sizeof(current));
        rewind(f);
        assert(player_save_read_game_state(f, &hdr, &restored));
        assert(restored.current_level == 11 && restored.running);
        assert(restored.cfg_dos_screen_size == 80);
        fclose(f);
    }
    f = fopen("build/feedback-tests/save-layout.tmp", "w+b"); assert(f);
    assert(fwrite(&current, 1, sizeof(current) - 1, f) == sizeof(current) - 1);
    rewind(f);
    assert(!player_save_read_game_state(f, &hdr, &restored));
    fclose(f);
    hdr.game_state_size = 1;
    f = fopen("build/feedback-tests/save-layout.tmp", "w+b"); assert(f);
    assert(!player_save_read_game_state(f, &hdr, &restored));
    fclose(f);
    remove("build/feedback-tests/save-layout.tmp");
    puts("Historical v7, current v7/v8 and truncated save layout checks passed");
    return 0;
}
