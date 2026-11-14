/* Exercise the desktop INI path in Node's isolated WASM filesystem. SDL is
 * included before selecting the desktop branch so its target stays unchanged. */
#include <SDL.h>
#undef __EMSCRIPTEN__
#include "../src/settings.c"
#include <assert.h>
#undef printf

int main(void)
{
    static GameState state;
    game_state_init(&state);
    assert(state.cfg_run_speed==130);
    assert(state.cfg_dos_screen_size==100);
    strcpy(g_menu_options_save_path,"/test.ini");
    FILE *f=fopen("/test.ini","wb");
    assert(f);
    fputs("# Preserve custom options\nstart_level=8\nrun_default=0\n",f);
    fclose(f);
    for(int speed=50;speed<=200;speed+=10) {
        state.cfg_run_speed=(int16_t)speed;
        settings_save_menu_options(&state);
        GameState loaded={0};
        parse_file(&loaded,"/test.ini");
        assert(loaded.cfg_run_speed==speed && loaded.cfg_start_level==7);
        f=fopen("/test.ini","rb");
        char text[4096]={0};
        assert(f); fread(text,1,sizeof(text)-1,f); fclose(f);
        assert(strstr(text,"# Preserve custom options\nstart_level=8\n"));
        char *key=strstr(text,"run_speed=");
        assert(key && !strstr(key+1,"run_speed="));
    }
    char low[]="run_speed=49", high[]="run_speed=201";
    state.cfg_run_speed=130;
    apply_line(&state,low); apply_line(&state,high);
    assert(state.cfg_run_speed==130);
    state.cfg_run_speed=0; apply_runtime_constraints(&state);
    assert(state.cfg_run_speed==50);
    state.cfg_run_speed=1000; apply_runtime_constraints(&state);
    assert(state.cfg_run_speed==200);
    puts("PASS: run-speed default, all INI values round-trip, append/update without duplicate keys, invalid values and bounds");
    const int sizes[]={40,60,80,100};
    const int widths[]={128,192,256,320},heights[]={96,144,192,240};
    for(int i=0;i<4;i++) {
        state.cfg_dos_screen_size=(int16_t)sizes[i];
        settings_save_menu_options(&state);
        GameState loaded={0};parse_file(&loaded,"/test.ini");
        assert(loaded.cfg_dos_screen_size==sizes[i] && loaded.cfg_start_level==7);
        f=fopen("/test.ini","rb");char text[4096]={0};assert(f);
        fread(text,1,sizeof(text)-1,f);fclose(f);
        char *key=strstr(text,"dos_screen_size=");
        assert(key && !strstr(key+1,"dos_screen_size="));
        int w,h;dos_screen_size_dimensions(320,240,sizes[i],&w,&h);
        assert(w==widths[i] && h==heights[i]);
        dos_screen_size_dimensions(320,200,sizes[i],&w,&h);
        assert(w==widths[i] && h==sizes[i]*2);
        assert(dos_screen_size_adjust(sizes[i],-1)==sizes[i==0?0:i-1]);
        assert(dos_screen_size_adjust(sizes[i],1)==sizes[i==3?3:i+1]);
    }
    state.cfg_dos_screen_size=60;
    char invalid_screen[]="dos_screen_size=50";apply_line(&state,invalid_screen);
    assert(state.cfg_dos_screen_size==60);
    state.cfg_dos_screen_size=0;apply_runtime_constraints(&state);
    assert(state.cfg_dos_screen_size==100);
    puts("PASS: DOS screen-size defaults, all four INI values, no duplicate keys, native VGA geometry and adjustment bounds");
}
